"""轨迹精度补偿：迭代学习控制（ILC）与关节柔度前馈。

要解决的是什么问题
----------------
工业机械臂**重复走同一条轨迹**时，实际轨迹与指令轨迹之间存在一个
**几乎每次都一样**的偏差。来源有三类：

1. **几何误差** —— 连杆长度、关节零位与名义值有出入（装配公差、磨损）。
2. **关节柔度** —— 减速器与连杆不是刚体。负载力矩越大，弯得越多。
   工业臂的关节刚度典型量级 10⁴–10⁶ N·m/rad，看着很大，
   但几十牛·米的力矩就能带来零点几毫米的末端偏移。
3. **摩擦、齿隙、热漂移** —— 难以建模。

前两类**可以建模**，第三类不好建模。但它们有一个共同点：

    ⭐ **只要轨迹一样、负载一样，这个偏差每一遍都几乎一样。**

这就是「**重复性误差**」。ILC 专治这一类。

ILC 的核心思想（一句话）
----------------------
    **这一遍哪里偏了，下一遍就在那个时刻提前补上。**

它不试图理解误差的成因，只利用「误差可重复」这一条：把上一遍的误差
按时间对齐，加进下一遍的前馈指令里。

    u_{k+1}(t) = Q · [ u_k(t) + L · e_k(t + Δ) ]

* ``u_k`` 第 k 遍的前馈修正量（加在参考轨迹上）
* ``e_k`` 第 k 遍的跟踪误差
* ``L``   学习增益
* ``Δ``   **时间超前量**。系统有滞后，误差出现在「因」之后，
          补偿必须**提前**送出去，否则学出来的东西是错位的。
* ``Q``   低通滤波（Q-filter）。⭐ **它决定收敛与否，见下。**

⚠️ ILC 与自适应控制的区别（很容易混）
-----------------------------------
======================  ==========================  =========================
                         自适应控制（03 篇）           ILC（本模块）
======================  ==========================  =========================
 学的是什么              **物理参数**（质量、惯量）    **某条轨迹上的误差信号**
 换一条轨迹              仍然有效                     ⭐ **完全失效，要重学**
 在时间上                连续更新                     ⭐ **一遍一遍地更新**
 需要重复吗              不需要                       ⭐ **必须重复同一条轨迹**
 能解释误差成因吗        能（给出参数）               不能（只是记住了长什么样）
======================  ==========================  =========================

所以 ILC 适合**产线上固定节拍、反复走同一条轨迹**的场合——
这正是工业机械臂最常见的工况。

⭐ 收敛条件：为什么必须有 Q-filter
--------------------------------
把误差写成 ``e_k = r − P·(u_k + ...)``（P 是闭环系统），代入更新律可得

    e_{k+1} = Q·(I − L·P)·e_k

于是收敛的充要条件是谱半径

    ‖ Q·(I − L·P) ‖ < 1

* ``L`` 太大 ⇒ ``|1 − L·P| > 1`` ⇒ **发散**；
* 高频段 ``P`` 的相位严重滞后（模型不准），``|1 − L·P|`` 必然大于 1；
* ``Q`` 是低通，把高频段整体压下去，**用「不学高频」换取稳定**。

⭐ **代价很实在**：Q 的截止频率以上的误差 ILC **学不掉**，稳态残差就卡在那儿。
这不是调参没调好，是这个方法的固有上限。

本实现的边界
----------
* 只做**单调轨迹重复**的时域 ILC，没有做频域设计、没有做鲁棒 ILC。
* Q-filter 用**零相位**滤波（前向 + 反向各滤一次）。
  因果滤波会引入额外相位滞后，反过来破坏收敛条件——
  ILC 是离线逐遍更新的，允许非因果操作，这是它相对在线控制的一个便利。
* 柔度补偿用**线性关节刚度**模型，不含齿隙与迟滞。
"""

from __future__ import annotations

import numpy as np


class JointComplianceModel:
    """关节柔度：把「受多大力矩」换算成「弯了多少角度」。

    最简单的线性模型::

        Δq = τ_load / k_joint

    ``k_joint`` 是关节刚度（N·m/rad）。真实减速器还有齿隙与迟滞，
    这里**不建模**——它们不可重复，正是留给 ILC 去学的部分。

    参数
    ----
    stiffness   各关节刚度，N·m/rad。给标量则所有关节相同
    n           关节数
    """

    def __init__(self, stiffness, n: int = 7):
        k = np.asarray(stiffness, dtype=float)
        self.k = np.full(n, float(k)) if k.ndim == 0 else k.reshape(n).copy()
        if np.any(self.k <= 0):
            raise ValueError("关节刚度必须为正")
        self.n = int(n)

    def deflection(self, tau_load) -> np.ndarray:
        """给定关节力矩，返回各关节的弯曲角（rad）。"""
        return np.asarray(tau_load, dtype=float).reshape(self.n) / self.k

    def compensate(self, q_cmd, tau_load) -> np.ndarray:
        """前馈补偿：**多转一点**，把弯掉的量抵消回来。

        ⭐ 符号很容易搞反：连杆在负载下**顺着力矩方向**弯，
        所以指令要**朝同方向多给**，而不是反向。
        写反了误差会**翻倍**——这正是 :func:`~armctrl.tests.test_ilc` 里
        用「补偿后误差必须小于不补偿」这一条独立判据锁住的地方。
        """
        return np.asarray(q_cmd, dtype=float).reshape(self.n) + self.deflection(tau_load)


def zero_phase_lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """零相位一阶低通（前向滤一次，反向再滤一次）。

    ``alpha`` 越小截止频率越低、滤得越狠；``alpha = 1`` 表示不滤。

    ⭐ **为什么要零相位**：普通（因果）滤波会让信号整体延后，
    而 ILC 的收敛条件 ``‖Q(I−LP)‖<1`` 对相位极其敏感——
    额外的滞后会直接把收敛条件破坏掉。
    ILC 是**逐遍离线更新**的，整段数据都在手上，所以可以反向再滤一次
    把相位抵消掉。这是 ILC 相对在线控制的一个便利。
    """
    x = np.asarray(x, dtype=float)
    a = float(np.clip(alpha, 1e-6, 1.0))
    if a >= 1.0:
        return x.copy()

    def _pass(sig):
        out = np.empty_like(sig)
        acc = sig[0].copy()
        for i in range(len(sig)):
            acc = acc + a * (sig[i] - acc)
            out[i] = acc
        return out

    return _pass(_pass(x[::-1])[::-1])


class IterativeLearningController:
    """P 型迭代学习控制：一遍一遍地把重复性误差学掉。

    参数
    ----
    n_steps    一遍轨迹的采样点数
    n_joints   关节数
    gain       学习增益 L
    q_alpha    Q-filter 的一阶低通系数（越小滤得越狠，1 = 不滤）
    lead       ⭐ **时间超前量（步）**。补偿要比误差**提前**这么多送出去
    limit      单关节修正量的上限（rad），防止学飞

    用法::

        ilc = IterativeLearningController(n_steps, 7)
        for k in range(n_trials):
            err = run_one_trial(ref + ilc.correction())   # 跑一遍，收集误差
            ilc.update(err)                               # 学一次
    """

    def __init__(self, n_steps: int, n_joints: int = 7, gain: float = 0.6,
                 q_alpha: float = 0.15, lead: int = 6, limit: float = 0.35):
        self.n_steps = int(n_steps)
        self.n_joints = int(n_joints)
        self.gain = float(gain)
        self.q_alpha = float(q_alpha)
        self.lead = int(lead)
        self.limit = float(limit)
        self.reset()

    def reset(self) -> None:
        self.u = np.zeros((self.n_steps, self.n_joints))
        self.trial = 0
        self.rms_history: list[float] = []

    def correction(self) -> np.ndarray:
        """当前这一遍要叠加到参考上的前馈修正量，形状 (n_steps, n_joints)。"""
        return self.u.copy()

    def update(self, error: np.ndarray) -> float:
        """用刚跑完这一遍的误差更新前馈，返回该遍的误差 RMS。

        ``error[i]`` 是第 i 个采样点上的 **参考 − 实际**（关节空间，rad）。
        """
        e = np.asarray(error, dtype=float).reshape(self.n_steps, self.n_joints)
        rms = float(np.sqrt(np.mean(e ** 2)))
        self.rms_history.append(rms)

        # ⭐ 时间超前：把误差整体往前挪 lead 步再学。
        # 系统响应有滞后，t 时刻看到的误差其实是 t−lead 时刻的指令造成的，
        # 所以补偿要提前送。不做超前的话，学到的修正与真正的成因错位，
        # 轻则收敛变慢，重则直接发散。
        e_lead = np.roll(e, -self.lead, axis=0)
        if self.lead > 0:
            e_lead[-self.lead:] = e[-1]          # 末段用最后一个值填，避免绕回

        u_new = self.u + self.gain * e_lead
        u_new = zero_phase_lowpass(u_new, self.q_alpha)
        self.u = np.clip(u_new, -self.limit, self.limit)
        self.trial += 1
        return rms

    # ------------------------------------------------------------ 分析

    def convergence_ratio(self) -> float:
        """最近一遍与第一遍的误差之比。**越小说明学得越好。**

        返回 nan 表示还没跑够两遍。
        """
        if len(self.rms_history) < 2:
            return float("nan")
        first = self.rms_history[0]
        return float(self.rms_history[-1] / first) if first > 0 else float("nan")

    @staticmethod
    def contraction_factor(gain: float, plant_gain: float) -> float:
        """理论收缩因子 ``|1 − L·P|``。

        这是**独立于本实现**的解析判据：ILC 每遍把**「与最终残差的距离」**
        乘上这个因子，小于 1 才收敛。用来对照实测的逐遍误差比。

        * ``L·P = 1``  → 因子 0，理论上**一遍到位**
        * ``L·P = 2``  → 因子 1，**临界**，距离不增不减
        * ``L·P > 2``  → 因子 > 1，**发散**

        ⭐ 所以「学习增益越大越好」是错的：``L`` 超过 ``2/P`` 就会发散。

        ⚠️ **本函数只描述「多快」，不描述「停在哪」。**
        完整递推是

            e_{k+1} = (I−Q)(r−d) + Q(I−LP) e_k

        终点由**仿射项** ``(I−Q)(r−d)`` 单独决定：

            e_∞ = [I − Q(I−LP)]⁻¹ (I−Q)(r−d)

        即使 ``L·P = 1`` 让收缩因子归零，误差也只是**一遍就跳到 e_∞ 并冻在那里**，
        不是归零。实测（标量对象、q_alpha=0.15）：直流分量残差 0.000000，
        而 150 周期的高频分量 e0=0.7071 → e1=e10=e39=0.702098，只学掉 0.7%。

        之所以直流能学干净，是因为 :func:`zero_phase_lowpass` 的**直流增益恰为 1**，
        ``(I−Q)`` 把直流完全抹掉；残差地板正好等于 ``Q`` 挡掉的那段频带。
        """
        return abs(1.0 - float(gain) * float(plant_gain))
