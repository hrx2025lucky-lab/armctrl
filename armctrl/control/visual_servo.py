"""视觉伺服：用「看到的」而不是「事先记住的」去对准目标。

要解决的问题
-----------
开环抓取是这么干的：开始之前算一次抓取位姿，然后照着走。
只要工件没动，它就work。**但工件一旦动了，机械臂还去老地方**——
抓空、把工件撞倒、甚至撞坏夹爪。

实测这个失败：把工件从标称位置挪开 60 mm，开环抓取的目标点纹丝不动，
机械臂扑空并把工件撞倒（高度 45 mm → 20.9 mm）。

闭环的做法是：**边看边调**。每来一帧新的观测，就把目标点更新一次，
误差会被持续压下去。工件动了、掉了、滚到别处，只要还看得见，就能重新对准。

两种视觉伺服
-----------
**PBVS（位置型，Position-Based）**——本模块实现的这种。
先由感知模块把图像还原成**三维位姿**，再在笛卡尔空间里做位置控制。
优点是控制律直观、可以直接复用已有的笛卡尔阻抗控制器；
缺点是完全依赖位姿估计的精度，标定误差会直接变成定位误差。

**IBVS（图像型，Image-Based）**——本模块**没有**实现。
直接在**图像平面**上定义误差（例如「让这四个角点落到画面里这四个位置」），
用图像雅可比把像素误差映射成关节速度。
优点是对相机标定误差不敏感（因为目标也是用像素描述的）；
缺点是深度信息不可直接观测，且笛卡尔空间里的轨迹可能很奇怪（会绕远路）。

工业上抓取多用 PBVS（因为要和抓取位姿规划衔接），
对准/插装类任务常用 IBVS。

这个「相机」的边界（必须说清楚）
----------------------------
`ObjectPoseSensor` **不是真的视觉**。它直接从 MuJoCo 读工件位姿，
再叠加噪声、延迟、有限更新率和视场判据。也就是说它模拟的是
**「一个已经工作正常的感知模块」的输出特性**，而不是感知本身。

真实视觉还要解决而这里完全没有的：

1. **检测**——先得在图像里找到工件。光照、遮挡、背景杂乱都会让它失败，
   而且失败方式是「报告一个错误的位置」而不是「报告看不见」。
2. **从像素到位姿**——需要相机内参标定、手眼标定。手眼标定误差是
   真机上定位误差的主要来源之一，本模型里完全没有。
3. **尺度与深度**——单目相机没有深度，要靠已知尺寸、双目或深度相机。
4. **误检**——把别的东西当成工件。本模型只会「看不见」，不会「看错」。

所以本场景能演示的是：**观测的延迟、噪声、更新率、丢失如何影响闭环性能**，
以及闭环相对开环的价值。**不能**演示感知本身的难点。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class SensorSpec:
    """模拟位姿传感器的特性参数。

    rate_hz     更新率。工业相机常见 30–60 Hz，远低于 1 kHz 的控制周期，
                所以两帧之间控制器只能靠外推或保持。
    latency     从「光子进入相机」到「位姿可用」的总延迟，秒。
                包含曝光、传输、检测、位姿解算。真机上 30–100 ms 很常见，
                **这是视觉伺服最主要的性能瓶颈**。
    pos_noise   位置噪声标准差，米。
    max_range   超过这个距离就认为看不见（视场/分辨率限制）。
    """

    rate_hz: float = 30.0
    latency: float = 0.05
    pos_noise: float = 2.0e-3
    max_range: float = 1.2


class ObjectPoseSensor:
    """模拟的工件位姿传感器（相机 + 感知模块的**输出特性**模型）。

    实现上是一个延迟队列：每到更新时刻就把「当前真值 + 噪声」推进队列，
    读取时取出 latency 秒之前入队的那一帧。这样延迟是**真的延迟**，
    而不是把噪声调大来假装延迟——两者对闭环的影响完全不同：
    噪声让输出变毛躁但不改变相位，延迟会**吃掉相位裕度**，是导致振荡的元凶。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 body: str, spec: SensorSpec | None = None, seed: int = 0):
        self.model = model
        self.data = data
        self.spec = spec or SensorSpec()
        self.rng = np.random.default_rng(seed)
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self.bid < 0:
            raise ValueError(f"工件 body 不存在: {body}")
        self.reset()

    def reset(self) -> None:
        self._queue: deque = deque()
        self._last_sample_t = -1e9
        #: 下一帧**应该**曝光的绝对时刻。累加而不是"上次实际采样时刻 + 周期"，
        #: 否则会被控制周期系统性地拖慢（见 update 里的说明）。
        self._next_sample_t = -1e9
        self._latest: np.ndarray | None = None
        self._latest_t = -1e9
        self.n_frames = 0
        self.visible = True

    def _in_view(self, p_obj: np.ndarray, p_cam: np.ndarray) -> bool:
        """能不能看见。这里只用距离做判据，是最粗的一档模型。

        真实的视场判据还要考虑角度、遮挡（夹爪自己会挡住工件！）、
        运动模糊。夹爪遮挡在真机上尤其要命：越接近抓取位置，工件越可能
        被自己的手指挡住，恰恰是最需要看清的时候看不见了。
        """
        return bool(np.linalg.norm(p_obj - p_cam) <= self.spec.max_range)

    def update(self, t: float, p_cam: np.ndarray) -> None:
        """按更新率采样一帧，压进延迟队列。t 为仿真时刻。

        ⭐ 为什么维护 `_next_sample_t` 而不是 `上次采样时刻 + 周期`
        --------------------------------------------------------
        这个函数只在控制周期（500 Hz，2 ms 一格）被调到，
        所以曝光时刻只能落在 2 ms 的整数倍上。
        旧写法是 ``if t - _last < period: return; _last = t``——
        每一帧都把基准**重新对齐到实际采样时刻**，误差于是逐帧累计：

            要 120 Hz（8.333 ms）→ 只能等到 10 ms 才满足 → 实际 **100.0 Hz**（慢 16.7%）
            要  90 Hz（11.11 ms）→ 12 ms → 实际 83.3 Hz（慢 7.4%）
            要  60 Hz（16.67 ms）→ 18 ms → 实际 55.6 Hz（慢 7.4%）
            要  30 Hz（33.33 ms）→ 34 ms → 实际 29.4 Hz（慢 2.0%）

        也就是说滑条上写 120 Hz，环路里承受的其实是 100 Hz 的延迟——
        任何"帧率对稳定边界的影响"的结论都会跟着错。

        改成累加**绝对时刻**之后，帧间隔在 8 ms 和 10 ms 之间抖动，
        但长期平均正好是 8.333 ms。这也更接近真机：
        相机的曝光时钟和控制器的时钟本来就不同步。
        """
        period = 1.0 / max(self.spec.rate_hz, 1e-6)
        if self._next_sample_t < -1e8:            # 第一帧：立刻采
            self._next_sample_t = t
        if t < self._next_sample_t:
            return
        self._last_sample_t = t
        self._next_sample_t += period
        if self._next_sample_t <= t:              # 长时间没被调用，重新对齐
            self._next_sample_t = t + period
        p_true = self.data.xpos[self.bid].copy()
        self.visible = self._in_view(p_true, np.asarray(p_cam, dtype=float))
        if not self.visible:
            return
        noisy = p_true + self.rng.normal(0.0, self.spec.pos_noise, 3)
        self._queue.append((t + self.spec.latency, noisy))
        self.n_frames += 1

    def read(self, t: float):
        """取出此刻**已经可用**的最新一帧。没有可用帧时返回上一次的结果。

        返回 (position, age)：age 是这一帧对应的真值距今多久（秒）。
        age 就是控制器实际承受的延迟——它比 spec.latency 还要大一点，
        因为两帧之间只能保持。
        """
        while self._queue and self._queue[0][0] <= t:
            ready_t, p = self._queue.popleft()
            self._latest = p
            self._latest_t = ready_t - self.spec.latency
        if self._latest is None:
            return None, 0.0
        return self._latest.copy(), float(t - self._latest_t)


class PositionVisualServo:
    """位置型视觉伺服（PBVS）。

    控制律就是一个笛卡尔空间的比例控制：

        e = p_target − p_current
        Δp = clip( λ · e ,  ±step_max )

    输出是**目标点的增量**，交给下游的笛卡尔阻抗控制器去跟。
    分成两层是有意的：伺服层只管「往哪挪」，阻抗层管「多硬地挪」，
    这样接触发生时仍然是柔顺的。把视觉误差直接接到位置伺服上，
    一旦碰到东西就会硬顶。

    为什么要限幅 step_max
    -------------------
    观测有噪声，λ·e 会跟着抖；观测还有延迟，大步长会导致**过冲后再回来**，
    表现为末端在目标附近来回振荡。限幅让单次修正有上界，
    相当于给这个闭环加了一道饱和保护。

    λ 与延迟的关系（可以自己推）
    -------------------------
    一阶闭环 ṗ = −λ(p − p_target) 的时间常数是 1/λ。
    观测延迟 τ 相当于在环路里插了一个纯滞后，相位滞后约 λτ 弧度。
    要保持稳定裕度，粗略地需要 **λ·τ 明显小于 1**。
    延迟 50 ms 时，λ 超过约 20 就开始明显振荡——这条可以在场景里直接调出来。
    """

    def __init__(self, gain: float = 4.0, step_max: float = 0.004,
                 deadband: float = 1.0e-3):
        self.gain = float(gain)
        self.step_max = float(step_max)
        #: 死区：误差小于它就不再修正，避免在噪声里反复微调
        self.deadband = float(deadband)
        self.reset()

    def reset(self) -> None:
        self.last_error = np.zeros(3)
        self.converged = False

    def step(self, p_current, p_target, dt: float) -> np.ndarray:
        """返回本周期目标点应当移动的增量。"""
        e = np.asarray(p_target, dtype=float) - np.asarray(p_current, dtype=float)
        self.last_error = e
        n = float(np.linalg.norm(e))
        self.converged = n < self.deadband
        if self.converged:
            return np.zeros(3)
        delta = self.gain * e * float(dt)
        m = float(np.linalg.norm(delta))
        if m > self.step_max:
            delta = delta * (self.step_max / m)
        return delta
