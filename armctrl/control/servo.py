"""伺服环路控制与振动抑制。

这一模块回答工业机器人最实际的一个问题：

    电机明明按指令转了，为什么末端还在抖？增益调高一点就振，怎么办？

答案的起点是：**电机和负载之间不是刚性连接**。谐波减速器、皮带、
长悬臂连杆都会带来扭转柔度。把这段柔度画成一根弹簧，
就得到了伺服领域最基础的模型 —— **双质量弹性系统**。

    τ_m ──▶ [ 电机 J_m ] ──弹簧 K_s , 阻尼 D_s── [ 负载 J_l ] ──▶ 输出
                  ▲
              编码器装在这里（关键！）

⭐ **编码器装在电机侧**是理解一切的前提：控制器看到的是 θ_m，
而我们真正想控制的是 θ_l。中间隔着一根弹簧，弹簧会晃，
控制器却看不见负载晃 —— 这就是振动抑制这门学问存在的理由。


一、两个频率是怎么来的（自行推导，不引用手册）
------------------------------------------------

运动方程（τ_s 是弹簧传过去的力矩）：

    J_m θ̈_m = τ_m − τ_s
    J_l θ̈_l = τ_s
    τ_s = K_s (θ_m − θ_l) + D_s (θ̇_m − θ̇_l)

**谐振频率**：令 δ = θ_m − θ_l（弹簧的扭转量），两式相减

    δ̈ = θ̈_m − θ̈_l = (τ_m − τ_s)/J_m − τ_s/J_l
       = τ_m/J_m − τ_s (1/J_m + 1/J_l)

代入 τ_s 并整理

    δ̈ + [(J_m+J_l)/(J_m J_l)] D_s δ̇ + [(J_m+J_l)/(J_m J_l)] K_s δ = τ_m/J_m

这是一个标准二阶系统，于是

    ω_res = sqrt( K_s (J_m + J_l) / (J_m J_l) ) = sqrt( K_s / J_eq )
    J_eq  = J_m J_l / (J_m + J_l)            ← 串联惯量（调和平均的一半形式）

**反谐振频率**：看从 τ_m 到 θ_m 的传递函数（先忽略 D_s）。由第二式

    θ_l = K_s / (J_l s² + K_s) · θ_m

代回第一式

    τ_m = θ_m · s² · [ J_m J_l s² + K_s (J_m + J_l) ] / (J_l s² + K_s)

    ⇒  θ_m/τ_m = (J_l s² + K_s) / { s² [ J_m J_l s² + K_s (J_m+J_l) ] }

**分子的零点**就是反谐振：J_l s² + K_s = 0 ⇒ ω_ares = sqrt(K_s / J_l)
**分母的极点**就是谐振，与上面用 δ 推的结果一致。✓ 两条路互相印证。

两者的比值只跟惯量比有关：

    ω_res / ω_ares = sqrt( 1 + J_l / J_m )

⭐ **惯量比 J_l/J_m 是伺服选型的头号数字**。它越大，两个频率离得越远，
谐振越「深」，速度环增益越调不上去。工业经验值是把它压在 5~10 以内。


二、这两个频率的物理含义（零基础版）
------------------------------------

- **反谐振 ω_ares = sqrt(K_s/J_l)**：负载自己吊在弹簧上晃的频率。
  在这个频率上负载晃得最欢，反过来把电机「顶住」——
  电机推不动了，**从电机看出去阻抗无穷大**。
  现象：末端在抖，电机却几乎不动。

- **谐振 ω_res = sqrt(K_s/J_eq)**：电机和负载**反着晃**，
  弹簧中间有个不动的点（节点）。这时**从电机看出去阻抗为零**，
  一丁点力矩就能激起很大的振荡。
  现象：速度环增益一调高，整条轴在这个频率上啸叫。

⇒ **速度环带宽被谐振频率钳死**，这是振动抑制要解决的核心矛盾。


三、三环串级（伺服驱动器的标准结构）
------------------------------------

    位置指令 →[位置环 P]→ 速度指令 →[速度环 PI]→ 电流指令 →[电流环]→ 力矩

从内到外带宽依次降低，通常相差 3~10 倍：

| 环 | 典型带宽 | 调节器 | 为什么 |
|---|---|---|---|
| 电流环 | 1~2 kHz | PI | 电气时间常数最小，可以做得最快 |
| 速度环 | 100~300 Hz | PI | **被机械谐振钳死** |
| 位置环 | 20~50 Hz | P（+前馈） | 加 I 会超调，用前馈代替 |

**为什么内环必须比外环快**：外环把内环当成「理想的、瞬间完成的执行器」。
内环慢了这个假设就破产，外环的相位裕度被内环的滞后吃掉 → 振荡。

**为什么位置环通常只用 P**：位置环的被控对象里已经有一个积分器
（速度积分成位置）。再加 I 就是双积分，相位一开始就 −180°，
稳定裕度很难做。要消稳态误差用**前馈**，因为前馈不进反馈回路，
不影响稳定性。


四、三种振动抑制手段
--------------------

1. **陷波滤波（notch）** —— 在速度环里把谐振频率那一段「挖掉」。
   治标，但见效最快，是伺服驱动器的标配。
   代价：陷波带来额外相位滞后；频率估偏了就没用甚至更糟。

2. **输入整形（input shaping）** —— 把指令拆成几个错开半个振动周期的
   脉冲，让后一个脉冲激起的振动**正好抵消**前一个留下的。
   治本（根本不激励振动），但只能用于**指令前馈通道**，
   对外部扰动引起的振动无效；且会引入 T_d/2 的时间延迟。

3. **限 jerk 轨迹**（本仓库 `planning/trajectory.py` 的七段 S 曲线）——
   让指令本身就不含高频成分。这其实是最朴素的一种输入整形。

⭐ 三者是**互补**关系不是替代：陷波管反馈通道，整形管前馈通道，
限 jerk 管指令源头。工业上经常三个一起用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TwoMassJoint",
    "resonance_frequency",
    "antiresonance_frequency",
    "NotchFilter",
    "InputShaper",
    "residual_vibration",
    "CascadeServo",
]


# ============================================================ 双质量弹性关节


def resonance_frequency(k_s: float, j_m: float, j_l: float) -> float:
    """谐振角频率 ω_res = sqrt(K_s (J_m+J_l) / (J_m J_l))（rad/s）。

    电机与负载**反相**振荡的频率。速度环带宽被它钳死。
    """
    if k_s <= 0.0 or j_m <= 0.0 or j_l <= 0.0:
        raise ValueError("K_s, J_m, J_l 必须为正")
    return float(np.sqrt(k_s * (j_m + j_l) / (j_m * j_l)))


def antiresonance_frequency(k_s: float, j_l: float) -> float:
    """反谐振角频率 ω_ares = sqrt(K_s / J_l)（rad/s）。

    负载吊在弹簧上自己晃的频率。此时电机看到的阻抗最大，推不动负载。
    注意它**只与负载惯量有关**，与电机惯量无关——这是区分两个频率的
    最快办法：换个更大的电机，谐振频率会变，反谐振频率不变。
    """
    if k_s <= 0.0 or j_l <= 0.0:
        raise ValueError("K_s, J_l 必须为正")
    return float(np.sqrt(k_s / j_l))


@dataclass
class TwoMassJoint:
    """一个带串联柔度的关节：电机侧在软件里积分，负载侧交给物理引擎。

    为什么这么分工
    --------------
    真实机器人里负载就是连杆，惯量随位形变化、还与别的轴耦合，
    用一个标量 J_l 是**近似**。把负载留给 MuJoCo 去算，
    只在软件里补上「电机 + 弹簧」这一段，得到的就是真实惯量下的
    弹性关节，比拿两个标量惯量做玩具模型有说服力。

    用法：每个控制周期
        tau_s = joint.step(tau_motor, theta_load, omega_load, dt)
    把 `tau_s` 施加到物理引擎的那个关节上即可。

    Attributes
    ----------
    j_m : 折算到关节侧的电机转动惯量（kg·m²）。
          「折算」是指已经**乘以减速比的平方**：J_m,关节侧 = n² · J_m,电机轴。
          方向别记反：电机转 n 圈、关节才转 1 圈，所以从电机侧看关节侧惯量要
          **除** n²，从关节（输出）侧看电机惯量要 **乘** n²。本类的所有量都在
          关节侧，故用乘。这也是「减速比越大，电机惯量在关节侧越占主导」的由来。
    k_s : 传动扭转刚度（N·m/rad）
    d_s : 传动扭转阻尼（N·m·s/rad）
    """

    j_m: float = 0.05
    k_s: float = 800.0
    d_s: float = 0.6

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, theta: float = 0.0, omega: float = 0.0) -> None:
        self.theta_m = float(theta)
        self.omega_m = float(omega)

    def spring_torque(self, theta_l: float, omega_l: float) -> float:
        """弹簧 + 阻尼传给负载的力矩（N·m）。"""
        return (self.k_s * (self.theta_m - theta_l)
                + self.d_s * (self.omega_m - omega_l))

    def step(self, tau_m: float, theta_l: float, omega_l: float,
             dt: float) -> float:
        """推进电机侧一步，返回这一步传给负载的力矩。

        用半隐式（symplectic Euler）：先更新速度再更新位置。
        显式欧拉在 K_s 大时会往系统里注入能量而发散，
        半隐式对这类弹簧-质量系统是**保能量**的，同样一行代码却稳得多。
        """
        tau_s = self.spring_torque(theta_l, omega_l)
        self.omega_m += (tau_m - tau_s) / self.j_m * dt
        self.theta_m += self.omega_m * dt
        return tau_s

    def resonance(self, j_l: float) -> float:
        return resonance_frequency(self.k_s, self.j_m, j_l)

    def antiresonance(self, j_l: float) -> float:
        return antiresonance_frequency(self.k_s, j_l)

    def inertia_ratio(self, j_l: float) -> float:
        """惯量比 J_l / J_m —— 伺服选型的头号数字。"""
        return float(j_l / self.j_m)


# ================================================================ 陷波滤波器


class NotchFilter:
    r"""二阶陷波：在指定频率上挖一个凹口，其余频段尽量不动。

    连续域传递函数

        N(s) = (s² + 2 ζ_n ω_n s + ω_n²) / (s² + 2 ζ_d ω_n s + ω_n²)

    分子分母只差阻尼比。在 s = j ω_n 处

        N(jω_n) = (2 ζ_n ω_n · jω_n) / (2 ζ_d ω_n · jω_n) = ζ_n / ζ_d

    所以**凹口深度就等于 ζ_n/ζ_d**，与 ω_n 无关。这条关系是设计陷波器的
    全部：想挖 −20 dB 就令 ζ_n/ζ_d = 0.1。
    而 ζ_d 单独决定凹口的**宽度**（ζ_d 越大越宽、越钝）。

    ⚠️ 陷波不是免费的：它在凹口两侧引入相位起伏，
    把陷波器放进速度环会吃掉相位裕度。频率估偏了更糟——
    该压的没压住，不该动的地方倒是变慢了。

    离散化用**预畸变双线性变换**（Tustin with prewarping）：
    普通双线性会把频率轴压缩，导致离散后的凹口偏离设计频率；
    预畸变把 ω_n 这一点钉死，正是我们最在意的那一点。
    """

    def __init__(self, freq_hz: float, depth: float = 0.1,
                 width: float = 0.7, dt: float = 1e-3):
        self.configure(freq_hz, depth, width, dt)

    def configure(self, freq_hz: float, depth: float, width: float,
                  dt: float) -> None:
        if freq_hz <= 0.0 or dt <= 0.0:
            raise ValueError("频率与步长必须为正")
        if not (0.0 < depth <= 1.0):
            raise ValueError("depth 必须在 (0,1]，1 表示不衰减")
        if width <= 0.0:
            raise ValueError("width（ζ_d）必须为正")
        self.freq_hz = float(freq_hz)
        self.depth = float(depth)
        self.width = float(width)
        self.dt = float(dt)

        zeta_d = float(width)
        zeta_n = float(depth) * zeta_d
        w0 = 2.0 * np.pi * self.freq_hz

        # 预畸变：让离散化后 w0 这一点的响应与连续域完全一致
        nyq = np.pi / self.dt
        if w0 >= nyq:
            raise ValueError(
                f"陷波频率 {self.freq_hz:.1f} Hz 超过奈奎斯特 "
                f"{nyq / (2 * np.pi):.1f} Hz")
        k = w0 / np.tan(w0 * self.dt / 2.0)

        # s → k(1−z⁻¹)/(1+z⁻¹) 代入并整理
        b0 = k * k + 2.0 * zeta_n * w0 * k + w0 * w0
        b1 = 2.0 * (w0 * w0 - k * k)
        b2 = k * k - 2.0 * zeta_n * w0 * k + w0 * w0
        a0 = k * k + 2.0 * zeta_d * w0 * k + w0 * w0
        a1 = b1
        a2 = k * k - 2.0 * zeta_d * w0 * k + w0 * w0
        self._b = np.array([b0, b1, b2]) / a0
        self._a = np.array([1.0, a1 / a0, a2 / a0])
        self.reset()

    def reset(self) -> None:
        self._x = [0.0, 0.0]
        self._y = [0.0, 0.0]

    def gain_at(self, freq_hz: float) -> float:
        """连续域幅频响应 |N(j2πf)|，用于独立核对凹口深度。"""
        w = 2.0 * np.pi * float(freq_hz)
        w0 = 2.0 * np.pi * self.freq_hz
        zeta_d = self.width
        zeta_n = self.depth * zeta_d
        num = complex(w0 * w0 - w * w, 2.0 * zeta_n * w0 * w)
        den = complex(w0 * w0 - w * w, 2.0 * zeta_d * w0 * w)
        return float(abs(num / den))

    def __call__(self, x: float) -> float:
        y = (self._b[0] * x + self._b[1] * self._x[0] + self._b[2] * self._x[1]
             - self._a[1] * self._y[0] - self._a[2] * self._y[1])
        self._x = [float(x), self._x[0]]
        self._y = [float(y), self._y[0]]
        return float(y)


# ================================================================ 输入整形


class InputShaper:
    r"""输入整形：把一条指令拆成几个脉冲，让振动自己抵消自己。

    原理（零基础版）
    ----------------
    推秋千：你推一下，秋千开始摆。等它摆到**正好半个周期**时再推一下
    大小合适的反向……不对，是同向但时机相反——第二次推产生的摆动
    与第一次剩下的摆动**相位差 180°**，两者叠加抵消。
    于是秋千停住，而人已经被送到了新位置。

    数学
    ----
    一个二阶欠阻尼系统被单位冲激激励后，残余振动的幅值与相位为
    e^{−ζω t} 与 ω_d t。若在 t_i 施加幅值 A_i 的一串冲激，
    在最后一个冲激时刻 t_n 之后的残余振动为

        V = e^{−ζω t_n} · sqrt( C² + S² )
        C = Σ A_i e^{ζω t_i} cos(ω_d t_i)
        S = Σ A_i e^{ζω t_i} sin(ω_d t_i)

    令 V = 0，加上「幅值和为 1」（否则终点位置会变），
    两个方程解两个未知数，得到 **ZV**（Zero Vibration）整形器

        K = exp( −ζπ / sqrt(1−ζ²) )
        A = [ 1, K ] / (1+K)          t = [ 0, T_d/2 ]，T_d = 2π/ω_d

    再补一个条件 dV/dω = 0（对频率误差不敏感），得到 **ZVD**

        A = [ 1, 2K, K² ] / (1+K)²    t = [ 0, T_d/2, T_d ]

    **代价**：ZV 延迟半个周期，ZVD 延迟一个周期。
    鲁棒性是拿时间换的——这是输入整形唯一的取舍。

    ⚠️ 边界：输入整形只作用在**指令**上。外力撞击引起的振动它管不了，
    那是反馈通道（陷波、阻尼注入）的事。
    """

    KINDS = ("none", "zv", "zvd")

    def __init__(self, kind: str = "zv", freq_hz: float = 20.0,
                 zeta: float = 0.02, dt: float = 1e-3):
        self.configure(kind, freq_hz, zeta, dt)

    @staticmethod
    def impulses(kind: str, freq_hz: float, zeta: float
                 ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (幅值数组, 时刻数组[s])。"""
        kind = str(kind).lower()
        if kind not in InputShaper.KINDS:
            raise ValueError(f"未知整形器 {kind}")
        if kind == "none":
            return np.array([1.0]), np.array([0.0])
        if not (0.0 <= zeta < 1.0):
            raise ValueError("阻尼比必须在 [0,1)")
        if freq_hz <= 0.0:
            raise ValueError("频率必须为正")
        wn = 2.0 * np.pi * freq_hz
        wd = wn * np.sqrt(1.0 - zeta * zeta)
        td = 2.0 * np.pi / wd
        k = float(np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta * zeta)))
        if kind == "zv":
            a = np.array([1.0, k]) / (1.0 + k)
            t = np.array([0.0, td / 2.0])
        else:
            a = np.array([1.0, 2.0 * k, k * k]) / (1.0 + k) ** 2
            t = np.array([0.0, td / 2.0, td])
        return a, t

    def configure(self, kind: str, freq_hz: float, zeta: float,
                  dt: float) -> None:
        self.kind = str(kind).lower()
        self.freq_hz = float(freq_hz)
        self.zeta = float(zeta)
        self.dt = float(dt)
        self.amps, self.times = self.impulses(self.kind, freq_hz, zeta)
        self._delay = np.rint(self.times / self.dt).astype(int)
        self._n = int(self._delay.max()) + 1
        self.reset()

    def reset(self, value: float = 0.0) -> None:
        self._buf = [float(value)] * self._n

    @property
    def lag(self) -> float:
        """整形引入的时间延迟（秒），等于最后一个脉冲的时刻。"""
        return float(self.times[-1])

    def __call__(self, x: float) -> float:
        self._buf.insert(0, float(x))
        if len(self._buf) > self._n:
            self._buf = self._buf[:self._n]
        return float(sum(a * self._buf[d]
                         for a, d in zip(self.amps, self._delay)))


def residual_vibration(amps, times, freq_hz: float, zeta: float) -> float:
    r"""一串冲激在系统 (freq_hz, zeta) 上留下的**相对残余振动**。

    这是输入整形的**独立判据**：它只用冲激序列与系统参数，
    不碰 `InputShaper` 的任何内部量。设计频率上 ZV / ZVD 应当给出 0。

        V = e^{−ζω t_n} sqrt(C² + S²),  C = Σ A_i e^{ζω t_i} cos(ω_d t_i)
                                        S = Σ A_i e^{ζω t_i} sin(ω_d t_i)

    参数 freq_hz 是**系统真实频率**，可以与整形器的设计频率不同 ——
    这正是画鲁棒性曲线（sensitivity curve）的方法。
    """
    a = np.asarray(amps, dtype=float)
    t = np.asarray(times, dtype=float)
    wn = 2.0 * np.pi * float(freq_hz)
    wd = wn * np.sqrt(1.0 - zeta * zeta)
    c = float(np.sum(a * np.exp(zeta * wn * t) * np.cos(wd * t)))
    s = float(np.sum(a * np.exp(zeta * wn * t) * np.sin(wd * t)))
    return float(np.exp(-zeta * wn * t[-1]) * np.hypot(c, s))


# ================================================================ 三环串级


class CascadeServo:
    """位置环 P + 速度环 PI + 电流环一阶等效。

    电流环为什么用一阶惯性等效
    --------------------------
    电流环带宽通常是速度环的 10 倍以上，在速度环的时间尺度上看，
    它的动态可以压缩成一句话：「指令给下去，过 1/(2πf_i) 秒才到位」。
    保留这个滞后很重要 —— 它正是**内环速度不够快就会拖垮外环**
    这一现象的来源，去掉它就演示不出串级设计的核心约束。

    ⭐ 反馈量取**电机侧**（`theta_m`, `omega_m`），不是负载侧。
    真实伺服的编码器就装在电机上。如果拿负载侧反馈，
    谐振问题会轻得多 —— 那是全闭环（光栅尺）才有的待遇。
    """

    def __init__(self, kpp: float = 20.0, kvp: float = 0.8,
                 kvi: float = 12.0, f_current: float = 800.0,
                 tau_max: float = 87.0):
        self.kpp = float(kpp)
        self.kvp = float(kvp)
        self.kvi = float(kvi)
        self.f_current = float(f_current)
        self.tau_max = float(tau_max)
        self.notch: NotchFilter | None = None
        self.reset()

    def reset(self) -> None:
        self._vi = 0.0
        self.tau = 0.0
        self.v_cmd = 0.0
        self.v_err = 0.0
        if self.notch is not None:
            self.notch.reset()

    def update(self, theta_ref: float, omega_ref: float,
               theta_m: float, omega_m: float, dt: float) -> float:
        """走一个控制周期，返回力矩指令（N·m）。"""
        # 位置环：纯 P + 速度前馈
        self.v_cmd = self.kpp * (theta_ref - theta_m) + omega_ref

        # 速度环：PI，作用在电机侧速度误差上
        err = self.v_cmd - omega_m
        if self.notch is not None:
            err = self.notch(err)
        self.v_err = float(err)

        tau_cmd = self.kvp * err + self._vi
        # 抗积分饱和：饱和时不再累加同号误差
        if abs(tau_cmd) < self.tau_max or tau_cmd * err < 0.0:
            self._vi += self.kvi * self.kvp * err * dt
            tau_cmd = self.kvp * err + self._vi
        tau_cmd = float(np.clip(tau_cmd, -self.tau_max, self.tau_max))

        # 电流环：一阶惯性
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * self.f_current))
        self.tau += alpha * (tau_cmd - self.tau)
        return float(self.tau)
