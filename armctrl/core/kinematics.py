"""数值逆运动学：阻尼最小二乘（DLS）+ 零空间优化，以及 SO(3) 姿态误差的连续 lift。

三种求解器
----------
    伪逆 (pinv)   Δq = J⁺ Δx
                  σ_min → 0 时 Δq 发散；np.linalg.pinv 的 rcond 是**硬截断**，
                  σ 略高于阈值时增益 ≈ 1/σ 极大，略低时直接归零——本身就是
                  不连续的，不能靠 step_clip 把它"看起来变连续"。
    DLS           Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ Δx
                  即 min ‖JΔq − Δx‖² + λ²‖Δq‖²，用精度换奇异点附近的有界性。
    自适应 DLS    λ 随最小奇异值自动增大（Nakamura & Hanafusa）。

任务度量 W —— 本模块的统一数学契约
--------------------------------
位置行的单位是 m，姿态行的单位是 rad。把两者放进同一个最小二乘问题、共用一个
奇异值阈值，或把 pos_err(m) 与 rot_err(rad) 直接相加，都是量纲错误。

本模块用一个**显式**的任务度量 W 消除量纲混合：

    位置任务（3 行）   W = I₃
    位姿任务（6 行）   W = diag(1, 1, 1, L, L, L)      L = 任务特征长度（米）

加权后实际求解的是

    min_Δq ‖W (J Δq − e)‖² + λ²‖Δq‖²     ⟺     J_task = W J,  e_task = W e

**伪逆、DLS、奇异值谱、数值秩、阻尼阈值、零空间投影、残差 ‖J_task Δq − e_task‖
全部来自同一个 (J_task, e_task) 和同一个 rcond。** 上一版的 bug 正是：σ_min 和
rank 由归一化后的 Jn 计算，而 np.linalg.pinv 却作用在未归一化的 J 上，阈值附近
会出现"实际 rank=6、报告 rank=5"。

零空间投影（更正上一版的无条件错误说明）
--------------------------------------
    Δq = Δq_task + N Δq_null

只有 N = I − J⁺J（**精确** Moore–Penrose 伪逆，且截断阈值与数值秩一致）时才有
J·(N Δq_null) = 0，"零空间分量不改变末端位姿"才成立。

若用阻尼伪逆 J⁺_λ = Jᵀ(JJᵀ+λ²I)⁻¹ 构造 N_λ = I − J⁺_λ J，做 SVD J = UΣVᵀ 得
J⁺_λ J = V·diag(σᵢ²/(σᵢ²+λ²))·Vᵀ，因此沿右奇异方向 vᵢ：

    N_λ vᵢ = λ²/(σᵢ²+λ²) · vᵢ            （零空间分量**没有被投影掉**）
    J N_λ vᵢ = σᵢ λ²/(σᵢ²+λ²) · uᵢ       （末端**确实**被扰动）

σ ≫ λ 时泄漏 ≈ λ²/σ²；σ = λ 时 50%；σ ≪ λ 时 → 100%，是 O(1) 而不是 O(λ²)。
本实现默认 null_projector="svd"（精确投影），并**每步记录 ‖J_task Δq_null‖**，
选 "damped" 时该值就是上式给出的泄漏量。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from armctrl.core.robot import ArmModel


TWO_PI = 2.0 * np.pi

#: 标定与测试用的 Panda 常用工作构型。放在这里是为了让"阈值依赖采样分布"
#: 这件事在源码里可见——它不是一个普适常数。
Q_HOME_PANDA = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])

POSITION_TASK = "position"
POSE_TASK = "pose"


# ----------------------------------------------------------------- SO(3)


def skew(v: np.ndarray) -> np.ndarray:
    """反对称矩阵 [v]ₓ，满足 [v]ₓ·u = v × u。"""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """SO(3) 指数映射（罗德里格斯公式），是 :func:`so3_log` 的逆。

        R = I + (sinθ/θ)·K + ((1−cosθ)/θ²)·K²,   K = [ω]ₓ, θ = ‖ω‖

    θ→0 时两个系数都是 0/0，必须换成泰勒展开：

        sinθ/θ      → 1 − θ²/6  + θ⁴/120
        (1−cosθ)/θ² → ½ − θ²/24 + θ⁴/720

    直接用原式在 θ ~ 1e-8 时会丢掉全部有效位。阈值取 1e-6：
    此处泰勒截断误差约 θ⁴/120 ~ 1e-26，远低于双精度分辨率。
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    if theta < 1e-6:
        c1 = 1.0 - theta * theta / 6.0
        c2 = 0.5 - theta * theta / 24.0
    else:
        c1 = np.sin(theta) / theta
        c2 = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + c1 * K + c2 * (K @ K)


def so3_right_jacobian(omega: np.ndarray) -> np.ndarray:
    """SO(3) 右雅可比 J_r(ω)，满足

        Exp(ω + δ) ≈ Exp(ω)·Exp(J_r(ω)·δ)        （δ 为小量）

    误差状态卡尔曼滤波里，「陀螺零偏的小扰动如何影响姿态」正是靠它换算的：
    陀螺零偏偏差 δb_g 在一步内让转角变化 −δb_g·dt，落到姿态误差上要乘 J_r。

        J_r = I − ((1−cosθ)/θ²)·K + ((θ−sinθ)/θ³)·K²

    θ→0 的泰勒展开：

        (1−cosθ)/θ² → ½ − θ²/24
        (θ−sinθ)/θ³ → ⅙ − θ²/120

    dt 很小时 J_r ≈ I，很多实现直接省略它。这里保留是因为它可以被
    **数值雅可比独立验证**——省掉之后误差藏在协方差里，看状态估计发现不了。
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    if theta < 1e-6:
        c1 = 0.5 - theta * theta / 24.0
        c2 = 1.0 / 6.0 - theta * theta / 120.0
    else:
        c1 = (1.0 - np.cos(theta)) / (theta * theta)
        c2 = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) - c1 * K + c2 * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) 主值对数：返回 ω ∈ R³，exp(skew(ω)) = R，‖ω‖ = 旋转角 ∈ [0, π]。

    分三个分支，避免 axis = vee(R − Rᵀ)/(2 sinθ) 在 θ→0 和 θ→π 处除零：

    θ ≈ 0     一阶展开 ω ≈ ½ vee(R − Rᵀ)·(1 + θ²/6)。
    一般情形   ω = θ/(2 sinθ)·vee(R − Rᵀ)。
    θ ≈ π     sinθ → 0，改用对称部分。由 R = I + sinθ·K + (1−cosθ)·K² 得
              (R + Rᵀ)/2 = I + (1−cosθ)·K²，故 K² = [(R + Rᵀ)/2 − I]/(1−cosθ)，
              而 aaᵀ = K² + I。取 aaᵀ 对角元最大的一列除以其平方根得到轴，
              再用反对称部分 vee(R − Rᵀ) = 2 sinθ·a 定符号。此分支 1−cosθ ≈ 2。

    用 atan2(sinθ, cosθ) 而不是 arccos(cosθ)：θ→π 时 arccos 的导数趋于无穷，
    双精度下角度误差约 sqrt(eps) ≈ 1e-8；atan2 在该处误差保持 1e-16 量级。

    ⚠️ θ = π 时 R(a, π) = 2aaᵀ − I = R(−a, π)，主值两值，符号本质不唯一；
    此时 argmax(diag(aaᵀ)) 在三个「异号 tie 面」(|aᵢ| = |aⱼ|, aᵢaⱼ < 0) 上会切换，
    输出整体变号。这是主值对数在 SO(3) 上的**拓扑切割**，不是本函数的缺陷。
    控制回路请用 OrientationErrorTracker 取连续 lift。
    """
    cos_t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    skew_vec = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    sin_t = float(np.linalg.norm(skew_vec) / 2.0)
    theta = float(np.arctan2(sin_t, cos_t))

    if theta < 1e-8:
        return 0.5 * skew_vec * (1.0 + theta * theta / 6.0)

    if theta < np.pi - 1e-4:
        return (theta / (2.0 * np.sin(theta))) * skew_vec

    K2 = ((R + R.T) / 2.0 - np.eye(3)) / (1.0 - cos_t)
    aat = K2 + np.eye(3)                      # = a aᵀ
    i = int(np.argmax(np.diag(aat)))
    denom = np.sqrt(max(aat[i, i], 1e-300))
    axis = aat[:, i] / denom
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.zeros(3)
    axis = axis / n
    if float(np.dot(skew_vec, axis)) < 0.0:   # 用反对称部分定符号
        axis = -axis
    return theta * axis


def rot_to_axis_angle_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """姿态误差的**主值**轴角表示 e_o，满足 R_des = exp(skew(e_o))·R_cur。

    世界系左误差约定，与世界系几何雅可比的角速度行一致。

    ⚠️ 无状态调用在 θ = π 附近不连续（见 so3_log）。控制回路请用
    OrientationErrorTracker；IK 迭代请用 so3_select_branch。
    """
    return so3_log(R_des @ R_cur.T)


def so3_equivalent_branch(omega: np.ndarray) -> np.ndarray:
    """返回同一旋转的等价轴角向量 ω − 2π·ω̂（即格点指标 k = −1 的那一支）。"""
    n = float(np.linalg.norm(omega))
    if n < 1e-12:
        return np.asarray(omega, float).copy()
    return omega - TWO_PI * (omega / n)


def so3_lift_candidates(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """列出与 omega 表示同一旋转、范数 < 2π 的**全部** lift 候选。

    独立推导：设 ω = θ·a（‖a‖ = 1, θ = ‖ω‖ ∈ [0, π]）。R(u, φ) = R(a, θ) 当且
    仅当 (u = a 且 φ ≡ θ mod 2π) 或 (u = −a 且 φ ≡ −θ mod 2π)。第二种情形
    φ·u = (−θ + 2πk)(−a) = (θ − 2πk)·a，与第一种合并后原像集是**一维格点**

        L(R) = { (θ + 2πk)·a : k ∈ ℤ }

    注意 −θ·a **不在** L 中（除非 θ ∈ {0, π}）：exp(skew(−θa)) = R(a, θ)ᵀ。
    范数小于 2π 的只有 k = 0 与 k = −1，即返回的 (θa, (θ−2π)a)，两者相距恰好 2π。

    θ = 0（R = I）时格点退化，任何 2π·u 都是合法 lift；本函数只返回 (0, 0)，
    即"绕圈状态在 R_err = I 处归零"。这是本模块对误差信号做的明确取舍：
    误差 lift 被限制在半径 2π 的球内，不累积无界绕圈。
    """
    omega = np.asarray(omega, float)
    n = float(np.linalg.norm(omega))
    if n < 1e-12:
        z = np.zeros(3)
        return z, z.copy()
    return omega.copy(), omega - TWO_PI * (omega / n)


def so3_select_branch(
    omega: np.ndarray,
    prev_lift: np.ndarray | None = None,
    prev_branch: int = 0,
    switch_margin: float = 0.0,
) -> tuple[np.ndarray, int]:
    """在等价格点候选中选出与 prev_lift 最近的连续 lift。

    返回 (lift, k)，k ∈ {0, −1} 是所选格点指标。保证 exp(skew(lift)) = exp(skew(omega))。

    - prev_lift is None（首次调用 / reset 之后）→ 返回主值，k = 0。
    - 两个候选相距恰好 2π，所以只要 prev_lift 不是正好落在两者中点，择优就是
      良态的；距离差 ≤ switch_margin 时**保持上一次的分支**，保证确定性。
    """
    c0, c1 = so3_lift_candidates(omega)
    if prev_lift is None:
        return c0, 0
    prev = np.asarray(prev_lift, float)
    d0 = float(np.linalg.norm(c0 - prev))
    d1 = float(np.linalg.norm(c1 - prev))
    if d1 < d0 - switch_margin:
        k = -1
    elif d0 < d1 - switch_margin:
        k = 0
    else:
        k = int(prev_branch)
    return (c0 if k == 0 else c1), k


class OrientationErrorTracker:
    """姿态误差的 SO(3) 连续 lift 跟踪器。**两级状态严格分离。**

    第 1 级 — 连续 lift（数学层，`lift()`）
        输出 e_lift ∈ L(R_err)，即 **exp(skew(e_lift)) 恒等于 R_err**（浮点精度内）。
        分支只由「上一拍的 e_lift」决定，**永远不被限速/限幅的结果污染**。
        ‖e_lift‖ ∈ [0, 2π)，允许超过 π —— 这正是跨 π 切割面连续所必需的。
        上一版把限幅后的值写回 _prev，导致 ‖ω_alt‖ > π 被 max_norm=π 缩放后
        e_lift 已不再代表同一个旋转；本版不再有这个问题。

    第 2 级 — 控制命令（工程层，`update()`）
        command = norm_saturate(rate_limit(e_lift))。**两者默认全部关闭。**
        一旦开启，command **不再是 R_err 的精确对数**，它是控制命令的限速/饱和
        结果；它只写回第 2 级状态，不写回第 1 级分支历史。

    契约
    ----
    首次调用 / `reset()` 之后：无历史，e_lift 取主值（k = 0），command = e_lift。
        因此 `reset()` 后重放同一帧，必然复现无状态的 `so3_log` 主值。
    episode 切换：调用方**必须**调用 `reset()`。两个控制器都暴露了 `reset()`。
    确定性：同一 (R_cur, R_des) 序列 → 逐位相同的输出。单帧输入相同但历史不同时，
        e_lift 可能落在两个等价格点之一，但**两者都精确表示同一个 R_err**；
        绝不会把被限速/限幅的值当作 lift 返回。
    限速：`max_rate` 默认为 None（关闭）。开启时必须提供**真实 dt**（构造参数或
        `update(dt=...)` 逐拍传入），否则抛 ValueError——不存在隐藏的 0.002 s。
    """

    def __init__(
        self,
        max_rate: float | None = None,
        max_norm: float | None = None,
        dt: float | None = None,
        switch_margin: float = 0.0,
    ):
        self.max_rate = None if max_rate is None else float(max_rate)
        self.max_norm = None if max_norm is None else float(max_norm)
        self.dt = None if dt is None else float(dt)
        self.switch_margin = float(switch_margin)
        self.reset()

    # -------------------------------------------------------- 状态

    def reset(self) -> None:
        """清空**两级**状态与全部计数器。下一次调用等同于首次调用。"""
        self._lift: np.ndarray | None = None      # 第 1 级
        self._branch: int = 0
        self._cmd: np.ndarray | None = None       # 第 2 级
        self.branch_switches = 0
        self.rate_limited = 0
        self.norm_saturated = 0
        self.last_theta = 0.0

    @property
    def branch(self) -> int:
        """当前所选的格点指标 k ∈ {0, −1}。"""
        return self._branch

    @property
    def branch_flips(self) -> int:
        """兼容旧名。语义已修正为「实际发生的分支切换次数」，
        不再对每一拍"选了 alt"累加。"""
        return self.branch_switches

    @property
    def last_lift(self) -> np.ndarray | None:
        return None if self._lift is None else self._lift.copy()

    @property
    def last_command(self) -> np.ndarray | None:
        return None if self._cmd is None else self._cmd.copy()

    # -------------------------------------------------------- 第 1 级

    def lift(self, R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        """连续 lift。保证 exp(skew(返回值)) = R_des·R_curᵀ。"""
        omega = so3_log(R_des @ R_cur.T)
        self.last_theta = float(np.linalg.norm(omega))
        lift, k = so3_select_branch(
            omega, self._lift, self._branch, self.switch_margin
        )
        if self._lift is not None and k != self._branch:
            self.branch_switches += 1
        self._lift, self._branch = lift.copy(), k
        return lift

    # -------------------------------------------------------- 第 2 级

    def update(
        self, R_cur: np.ndarray, R_des: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        """连续 lift → （可选）限速 → （可选）限幅 → 控制命令。"""
        cmd = self.lift(R_cur, R_des)

        if self.max_rate is not None and self._cmd is not None:
            step_dt = self.dt if dt is None else float(dt)
            if step_dt is None or not np.isfinite(step_dt) or step_dt <= 0.0:
                raise ValueError(
                    "启用 max_rate 时必须提供真实的 dt（构造参数或 update(dt=...)）"
                )
            d = cmd - self._cmd
            dn = float(np.linalg.norm(d))
            lim = self.max_rate * step_dt
            if dn > lim:
                cmd = self._cmd + d * (lim / dn)
                self.rate_limited += 1

        if self.max_norm is not None:
            n = float(np.linalg.norm(cmd))
            if n > self.max_norm:
                # 这是**控制命令饱和**，不再是 R_err 的精确对数。
                cmd = cmd * (self.max_norm / n)
                self.norm_saturated += 1

        self._cmd = np.asarray(cmd, float).copy()
        return self._cmd.copy()


# ----------------------------------------------------------------- IK


@dataclass
class Sigma0Calibration:
    """一次阻尼阈值标定的完整记录（阈值不是普适常数，必须连同条件一起看）。"""

    sigma0_pos: float
    sigma0_pose: float
    activate_frac: float
    samples: int
    seed: int
    distribution: str
    char_length: float
    quantiles_pos: dict
    quantiles_pose: dict


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool                 # 求解器**自报**：迭代内达到容差
    iters: int
    pos_err: float                  # 终点 FK 独立复算
    rot_err: float

    # ---- 每步诊断（长度 = 实际执行的迭代步数）----
    manip_trace: list = field(default_factory=list)          # prod(σ(J_task))
    dq_norm_trace: list = field(default_factory=list)        # step_clip **后**
    dq_raw_norm_trace: list = field(default_factory=list)    # step_clip **前**
    dq_raw_trace: list = field(default_factory=list)         # 限幅前的完整向量
    sigma_trace: list = field(default_factory=list)          # J_task 的**全谱**
    sigma_min_trace: list = field(default_factory=list)
    rank_trace: list = field(default_factory=list)           # 与 pinv 同源的数值秩
    lam_trace: list = field(default_factory=list)
    raw_residual_trace: list = field(default_factory=list)   # ‖J_task dq_raw − e_task‖
    clipped_residual_trace: list = field(default_factory=list)
    pos_err_trace: list = field(default_factory=list)        # 每步独立任务残差
    rot_err_trace: list = field(default_factory=list)
    null_leak_trace: list = field(default_factory=list)      # ‖J_task dq_null‖
    branch_trace: list = field(default_factory=list)         # 每步的格点指标 k
    joint_margin_trace: list = field(default_factory=list)

    # ---- 汇总计数 ----
    dq_raw_max: float = 0.0
    step_clip_count: int = 0            # 步长范数限幅（数值稳定手段）
    joint_limit_clip_count: int = 0     # 关节限位裁剪（物理可行性，两者必须分开）
    branch_switch_count: int = 0
    min_joint_margin: float = float("inf")     # 迭代过程中的最小限位裕量
    final_joint_margin: float = float("inf")   # 返回解的限位裕量
    branch: str = "primary"

    # ---- 独立验收 ----
    fk_pos_err: float = float("nan")
    fk_rot_err: float = float("nan")
    fk_verified: bool = False           # 由独立 FK 复算判定，不信求解器自报
    task_err_ndim: float = float("inf")  # max(pos_err/pos_tol, rot_err/rot_tol)
    solve_time: float = 0.0
    branch_report: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        """唯一应当被外部信任的成功判据：**独立 FK 复算**通过。

        故意不与求解器自报的 `converged` 相与：`converged` 只是"迭代循环内看到
        误差进入容差"，跑满 max_iters 的最后一步可能已经把 q 送进容差却来不及
        再判一次（实测 300 个近 π 目标中出现过 1 例）。谁对谁错由 FK 说了算。
        """
        return bool(self.fk_verified)

    @property
    def self_report_mismatch(self) -> bool:
        """求解器自报与独立 FK 复核不一致——出现即说明自报不可信。"""
        return bool(self.converged) != bool(self.fk_verified)


class DLSInverseKinematics:
    """阻尼最小二乘逆运动学求解器。

    参数
    ----
    lam0            基础阻尼系数。method="fixed" 时直接使用。
    method          "pinv" | "fixed" | "adaptive"
    sigma0_pos      位置任务（W = I₃）的 σ_min 阈值。
    sigma0_pose     位姿任务（W = diag(1,1,1,L,L,L)）的 σ_min 阈值。

                    为什么必须分开标定：即使做了 W 归一化，两类任务的 σ_min 分布
                    仍不同（σ_min(6×n) ≤ σ_min(3×n) 结构性成立，见下），近奇异
                    左尾的差距远大于中位数的差距。缺省值由 calibrate_sigma0()
                    在**混合分布**（均匀 / 常用工作区 / 边界奇异区各 1/3，
                    q_center = Q_HOME_PANDA，seed=7，6000 样本）上按
                    "最接近奇异的 10% 构型激活阻尼"标定得到：

                        sigma0_pos  = 0.030707      sigma0_pose = 0.008869

                    ⚠️ 这两个数不是"Panda 专属常数"那么简单。它们同时依赖：
                      机器人型号、**TCP 选取**（本项目取两指之间的抓取中心，
                      即法兰再往前 0.1029 m；换成法兰口径时 sigma0_pos 会从
                      0.030707 变成 0.025468）、
                      **任务类型**（3 行 / 6 行）、**采样分布**和**任务权重 W**
                      （L 变了 σ 就变）。换任何一项都要重标。实测同一组阈值在
                      不同分布上的激活比例（留出集）：

                        mixed    ≈ 10%        （设计点）
                        uniform  ≈ 1.6% / 6.1%
                        home     ≈ 0.3% / 1.3%
                        singular ≈ 28% / 24%

                      也就是说"10% 激活"这句话离开采样分布就没有意义。

    rcond           数值秩 / pinv 的相对截断阈值，作用在 **J_task = W J** 上。
                    这是**硬截断**：σ 略高于 rcond·σ_max 时增益 ≈ 1/σ，略低时
                    直接归零——它本来就是不连续的，step_clip 只是把跳变的幅度
                    压小，并没有让它连续。
    char_length     任务特征长度 L（米），构成 W 的姿态行缩放。None 时用
                    _estimate_reach() 估计（TCP 到**第一关节轴锚点**的最大距离）。
    null_gain       零空间次要任务（关节限位回避）增益，0 表示关闭。实测在 300 个
                    近 π 目标上（dual_branch=True）：

                        null_gain   成功率   贴限位解   限位裁剪总次数
                        0.0         93.3%      52         5550
                        0.2         93.7%      28         5003
                        1.0         93.7%      19         4605

                    即它主要改善**限位安全裕量**，对成功率影响很小。
    null_projector  "svd"（默认，精确 Moore–Penrose 零空间，无泄漏）
                    | "damped"（I − J⁺_λ J，**有泄漏** λ²/(σ²+λ²)，仅供对照）。
    step_clip       步长范数上限（None 关闭）。
    """

    def __init__(
        self,
        robot: ArmModel,
        lam0: float = 0.05,
        method: str = "adaptive",
        sigma0_pose: float = 0.008869,
        sigma0_pos: float = 0.030707,
        char_length: float | None = None,
        rcond: float = 1e-6,
        null_gain: float = 0.0,
        null_projector: str = "svd",
        step_clip: float | None = 0.2,
        margin_ok: float = 5e-3,
    ):
        self.robot = robot
        self.lam0 = float(lam0)
        self.method = method
        self.sigma0_pose = float(sigma0_pose)
        self.sigma0_pos = float(sigma0_pos)
        self.rcond = float(rcond)
        self.null_gain = float(null_gain)
        if null_projector not in ("svd", "damped"):
            raise ValueError(f"null_projector 只支持 'svd' 或 'damped'，收到 {null_projector!r}")
        self.null_projector = null_projector
        self.step_clip = None if step_clip is None else float(step_clip)
        self.margin_ok = float(margin_ok)
        self._stretch_cache: tuple[np.ndarray, np.ndarray] | None = None
        self.char_length = (
            float(char_length) if char_length is not None else self._estimate_reach()
        )

    # ------------------------------------------------ 任务度量 W

    @staticmethod
    def task_from_rows(rows: int) -> str:
        """行数 → 任务类型。**只接受 3 或 6**，其它行数抛错而不是默默当成位姿。"""
        if rows == 3:
            return POSITION_TASK
        if rows == 6:
            return POSE_TASK
        raise ValueError(
            f"任务雅可比只支持 3 行（位置）或 6 行（位姿），收到 {rows} 行；"
            "如需其它子任务请显式传入 task 并扩展 task_weight()。"
        )

    def task_weight(self, task: str) -> np.ndarray:
        """任务度量 W。位置任务 I₃；位姿任务 diag(1,1,1,L,L,L)。"""
        if task == POSITION_TASK:
            return np.eye(3)
        if task == POSE_TASK:
            L = self.char_length
            return np.diag([1.0, 1.0, 1.0, L, L, L])
        raise ValueError(f"未知任务类型 {task!r}，只支持 {POSITION_TASK!r} / {POSE_TASK!r}")

    def _normalize_jacobian(self, J: np.ndarray, task: str | None = None) -> np.ndarray:
        """J_task = W J。task 省略时由行数推断（只接受 3/6 行）。"""
        task = self.task_from_rows(J.shape[0]) if task is None else task
        return self.task_weight(task) @ J

    def _normalize_error(self, e: np.ndarray, task: str | None = None) -> np.ndarray:
        """e_task = W e。**必须**与 _normalize_jacobian 配对使用。"""
        task = self.task_from_rows(len(e)) if task is None else task
        return self.task_weight(task) @ e

    # ------------------------------------------------ 特征长度

    def _arm_base_origin(self) -> np.ndarray:
        """第一个手臂关节的轴锚点（世界系）。Panda 为 (0, 0, 0.333)。

        交给 ArmModel.joint_anchor 去保证状态已 forward。以前这里直接读
        `robot.data.xanchor`，隐含要求调用方**先**调过一次 fk 把 data 填好——
        这个隐含顺序一旦被打破（例如查询改走独立缓冲），锚点会静默变成零向量，
        可达半径被算成从世界原点起算的 1.19 m，而不是从基座起算的 0.855 m，
        姿态行缩放随之系统性偏大 39%。
        """
        try:
            return self.robot.joint_anchor(0)
        except Exception:                                    # pragma: no cover
            return np.zeros(3)

    def _stretched_config(self, sweeps: int = 4, grid: int = 41) -> tuple[np.ndarray, np.ndarray]:
        """坐标上升求"最大可达半径"构型 q_stretch，以及每个关节对可达半径的敏感度。

        边界奇异（boundary singularity）的普适定义就是"手臂伸到可达边界"：此时
        沿径向已无法再前进，雅可比掉秩。这个构造与具体机型无关，只需要 FK。
        敏感度可忽略的关节（例如绕手臂轴自转的第 1/第 7 关节）在采样时完全随机化。

        **无副作用**：由调用方（_estimate_reach / sample_joint_configs）负责保存恢复。
        """
        if getattr(self, "_stretch_cache", None) is not None:
            return self._stretch_cache
        r = self.robot
        base = self._arm_base_origin()

        def reach(q):
            p, _ = r.fk(q)
            return float(np.linalg.norm(p - base))

        q = 0.5 * (r.q_lower + r.q_upper)
        for _ in range(sweeps):
            for i in range(r.n):
                vals = np.linspace(r.q_lower[i], r.q_upper[i], grid)
                best, bv = -1.0, q[i]
                for g in vals:
                    qq = q.copy()
                    qq[i] = g
                    d = reach(qq)
                    if d > best:
                        best, bv = d, g
                q[i] = bv
        r0 = reach(q)
        sens = np.zeros(r.n)
        for i in range(r.n):
            vals = np.linspace(r.q_lower[i], r.q_upper[i], grid)
            worst = 0.0
            for g in vals:
                qq = q.copy()
                qq[i] = g
                worst = max(worst, abs(reach(qq) - r0))
            sens[i] = worst
        self._stretch_cache = (q.copy(), sens)
        return self._stretch_cache

    def _estimate_reach(self, samples: int = 4096, seed: int = 0) -> float:
        """标称可达半径 L = max ‖p_TCP(q) − p_base‖。

        **几何意义**：从**第一个手臂关节的轴锚点**到 TCP 的最大距离，也就是这条
        手臂能张开的半径。它不是"世界原点到 TCP 的最大三维范数"——那会把基座
        台高度算进来（Panda 是 0.333 m），得到约 1.189 m，比 Panda 标称 0.855 m
        的可达半径大 39%，用它做姿态行缩放会系统性高估角速度行的权重。

        **它是真实最大值的下界**：随机采样不保证取到全局最优构型。样本里显式
        包含文档声称的零位 q = 0 与关节行程中点，其余为确定性随机构型。

        **无副作用**：进入时保存 robot 的 qpos/qvel，finally 中恢复。
        """
        r = self.robot
        q_save, qd_save = r.get_q(), r.get_qd()
        try:
            base = None
            rng = np.random.default_rng(seed)
            fixed = [
                np.zeros(r.n),                                   # 文档声称的零位
                r.clamp_to_limits(np.zeros(r.n)),                # 零位裁剪到限位内
                0.5 * (r.q_lower + r.q_upper),                   # 行程中点
            ]
            best = 0.0
            for q in fixed:
                p, _ = r.fk(np.asarray(q, float))
                if base is None:
                    base = self._arm_base_origin()
                best = max(best, float(np.linalg.norm(p - base)))
            q_stretch, _ = self._stretched_config()              # 坐标上升的伸展构型
            p, _ = r.fk(q_stretch)
            best = max(best, float(np.linalg.norm(p - base)))
            for _ in range(int(samples)):
                q = rng.uniform(r.q_lower, r.q_upper)
                p, _ = r.fk(q)
                best = max(best, float(np.linalg.norm(p - base)))
            return max(best, 1e-3)
        finally:
            r.set_state(q_save, qd_save)

    # ------------------------------------------------ 阻尼调度

    def _damping(self, sigma_min: float, task: str) -> float:
        """基于 J_task 最小奇异值的阻尼调度（Nakamura–Hanafusa 形式）。

        σ_min ≥ σ0 不加阻尼（保精度）；低于阈值平滑增大（保稳定）。
        用 σ_min 而不是 √det(JJᵀ)：后者是全部奇异值之积，单方向接近奇异时会被
        其它大奇异值掩盖。**task 必须显式传入**（"3 行就是位置，否则一律位姿"
        的隐式规则在出现 2/5 行子任务时会静默走错分支）。
        """
        if self.method == "pinv":
            return 0.0
        if self.method == "fixed":
            return self.lam0
        if task == POSITION_TASK:
            s0 = self.sigma0_pos
        elif task == POSE_TASK:
            s0 = self.sigma0_pose
        else:
            raise ValueError(f"未知任务类型 {task!r}")
        if sigma_min >= s0:
            return 0.0
        return self.lam0 * (1.0 - sigma_min / s0) ** 2

    def _null_space_gradient(self, q: np.ndarray) -> np.ndarray:
        """关节限位回避的梯度：把关节推向行程中点。"""
        mid = 0.5 * (self.robot.q_lower + self.robot.q_upper)
        span = np.maximum(self.robot.q_upper - self.robot.q_lower, 1e-6)
        return -(q - mid) / span**2

    # ------------------------------------------------ 阈值标定

    def sample_joint_configs(
        self,
        samples: int,
        seed: int,
        distribution: str = "mixed",
        q_center: np.ndarray | None = None,
        spread: float = 0.6,
    ) -> np.ndarray:
        """生成标定/验证用的关节构型样本。

        distribution
            "uniform"   全关节行程均匀。覆盖广，但**不代表实际任务分布**：
                        真实机器人不会均匀访问整个关节空间。
            "home"      在 q_center（缺省 = 行程中点）附近截断高斯，代表常用工作区。
            "singular"  边界奇异区：围绕坐标上升得到的"最大可达半径"构型抖动，
                        对可达半径不敏感的关节（绕臂轴自转的那几个）完全随机化。
                        代表最危险的左尾，只看中位数会完全漏掉。
            "mixed"     上述三者各 1/3（缺省）。

        **无副作用**：保存并恢复 robot 状态。
        """
        r = self.robot
        rng = np.random.default_rng(seed)
        lo, hi = r.q_lower, r.q_upper
        center = 0.5 * (lo + hi) if q_center is None else np.asarray(q_center, float)

        def uni(n):
            return rng.uniform(lo, hi, size=(n, r.n))

        def home(n):
            sd = spread * 0.5 * (hi - lo)
            return np.clip(center + rng.normal(0.0, 1.0, size=(n, r.n)) * sd, lo, hi)

        def sing(n):
            q_save, qd_save = r.get_q(), r.get_qd()
            try:
                q_str, sens = self._stretched_config()
            finally:
                r.set_state(q_save, qd_save)
            free = sens < 1e-6                       # 不影响可达半径 → 完全随机
            Q = np.clip(q_str + rng.normal(0.0, 0.25, size=(n, r.n)), lo, hi)
            if np.any(free):
                Q[:, free] = rng.uniform(lo[free], hi[free], size=(n, int(np.sum(free))))
            return Q

        if distribution == "uniform":
            return uni(samples)
        if distribution == "home":
            return home(samples)
        if distribution == "singular":
            return sing(samples)
        if distribution == "mixed":
            a = samples // 3
            b = samples // 3
            c = samples - a - b
            return np.vstack([uni(a), home(b), sing(c)])
        raise ValueError(f"未知采样分布 {distribution!r}")

    def sample_sigma_min(
        self,
        samples: int = 3000,
        seed: int = 7,
        distribution: str = "mixed",
        q_center: np.ndarray | None = None,
        spread: float = 0.6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (σ_min 位置任务, σ_min 位姿任务) 两个样本数组。**无副作用。**"""
        r = self.robot
        q_save, qd_save = r.get_q(), r.get_qd()
        try:
            Q = self.sample_joint_configs(samples, seed, distribution, q_center, spread)
            s3 = np.empty(len(Q))
            s6 = np.empty(len(Q))
            for i, q in enumerate(Q):
                J6 = r.jacobian(q)
                s6[i] = np.linalg.svd(
                    self._normalize_jacobian(J6, POSE_TASK), compute_uv=False
                )[-1]
                s3[i] = np.linalg.svd(
                    self._normalize_jacobian(J6[:3, :], POSITION_TASK), compute_uv=False
                )[-1]
            return s3, s6
        finally:
            r.set_state(q_save, qd_save)

    def calibrate_sigma0(
        self,
        activate_frac: float = 0.10,
        samples: int = 3000,
        seed: int = 7,
        distribution: str = "mixed",
        q_center: np.ndarray | None = None,
        apply: bool = True,
    ) -> Sigma0Calibration:
        """按目标激活比例标定两个阈值，返回完整标定记录。

        标定集与验证集必须分开：用不同 seed 调用 validate_sigma0() 检验。
        """
        s3, s6 = self.sample_sigma_min(samples, seed, distribution, q_center)
        p = 100.0 * float(activate_frac)
        qs = (1.0, 5.0, 10.0, 25.0, 50.0, 90.0)
        cal = Sigma0Calibration(
            sigma0_pos=float(np.percentile(s3, p)),
            sigma0_pose=float(np.percentile(s6, p)),
            activate_frac=float(activate_frac),
            samples=int(len(s3)),
            seed=int(seed),
            distribution=distribution,
            char_length=float(self.char_length),
            quantiles_pos={x: float(np.percentile(s3, x)) for x in qs},
            quantiles_pose={x: float(np.percentile(s6, x)) for x in qs},
        )
        if apply:
            self.sigma0_pos = cal.sigma0_pos
            self.sigma0_pose = cal.sigma0_pose
        return cal

    def validate_sigma0(
        self,
        samples: int = 3000,
        seed: int = 101,
        distribution: str = "mixed",
        q_center: np.ndarray | None = None,
    ) -> dict:
        """在**留出集**上报告实际激活比例与分位数（含尾部），不修改实例。"""
        s3, s6 = self.sample_sigma_min(samples, seed, distribution, q_center)
        qs = (1.0, 5.0, 10.0, 25.0, 50.0)
        return {
            "seed": seed,
            "distribution": distribution,
            "samples": int(len(s3)),
            "frac_pos": float(np.mean(s3 < self.sigma0_pos)),
            "frac_pose": float(np.mean(s6 < self.sigma0_pose)),
            "min_pos": float(np.min(s3)),
            "min_pose": float(np.min(s6)),
            "quantiles_pos": {x: float(np.percentile(s3, x)) for x in qs},
            "quantiles_pose": {x: float(np.percentile(s6, x)) for x in qs},
        }

    # ------------------------------------------------ 求解

    def _joint_margin(self, q: np.ndarray) -> float:
        return float(
            np.min(np.minimum(q - self.robot.q_lower, self.robot.q_upper - q))
        )

    def _solve_branch(
        self,
        p_des: np.ndarray,
        R_des: np.ndarray | None,
        q_init: np.ndarray,
        max_iters: int,
        pos_tol: float,
        rot_tol: float,
        flip_branch: bool,
    ) -> IKResult:
        """单分支求解。flip_branch=True 时**首步**取等价格点 k = −1。"""
        t0 = time.perf_counter()
        q = np.asarray(q_init, float).copy()
        task = POSITION_TASK if R_des is None else POSE_TASK
        W = self.task_weight(task)
        m = W.shape[0]
        Im = np.eye(m)
        In = np.eye(self.robot.n)

        res = IKResult(
            q.copy(), False, 0, float("nan"), float("nan"),
            branch="flipped" if flip_branch else "primary",
        )
        prev_lift: np.ndarray | None = None
        prev_branch = 0

        for it in range(max_iters):
            p_cur, R_cur = self.robot.fk(q)
            e_p = p_des - p_cur
            pos_err = float(np.linalg.norm(e_p))

            if R_des is None:
                e_o = np.zeros(3)
                rot_err = 0.0
            else:
                omega = rot_to_axis_angle_error(R_cur, R_des)   # 主值
                # 收敛判据与任务残差始终用**主值**，翻支不影响验收
                rot_err = float(np.linalg.norm(omega))
                if prev_lift is None:
                    lift, k = (
                        (so3_lift_candidates(omega)[1], -1) if flip_branch
                        else so3_select_branch(omega, None, 0)
                    )
                else:
                    lift, k = so3_select_branch(omega, prev_lift, prev_branch)
                    if k != prev_branch:
                        res.branch_switch_count += 1
                prev_lift, prev_branch = lift.copy(), k
                res.branch_trace.append(k)
                e_o = lift

            res.pos_err_trace.append(pos_err)
            res.rot_err_trace.append(rot_err)
            margin = self._joint_margin(q)
            res.joint_margin_trace.append(margin)
            res.min_joint_margin = min(res.min_joint_margin, margin)

            if pos_err < pos_tol and (R_des is None or rot_err < rot_tol):
                res.converged = True
                res.iters = it
                break

            J = self.robot.jacobian(q)
            if R_des is None:
                J = J[:3, :]
                e = e_p
            else:
                e = np.concatenate([e_p, e_o])

            # === 同一个数学系统：J_task 与 e_task 必须成对 ===
            J_task = W @ J
            e_task = W @ e

            sv = np.linalg.svd(J_task, compute_uv=False)
            sigma_min = float(sv[-1])
            rank = int(np.sum(sv > self.rcond * sv[0]))
            lam = self._damping(sigma_min, task)
            res.sigma_trace.append(sv.copy())
            res.sigma_min_trace.append(sigma_min)
            res.rank_trace.append(rank)
            res.lam_trace.append(lam)
            res.manip_trace.append(float(np.prod(sv)))

            if self.method == "pinv" or lam <= 0.0:
                J_pinv = np.linalg.pinv(J_task, rcond=self.rcond)
                dq = J_pinv @ e_task
            else:
                reg = J_task @ J_task.T + lam**2 * Im
                dq = J_task.T @ np.linalg.solve(reg, e_task)
                J_pinv = J_task.T @ np.linalg.solve(reg, Im)

            leak = 0.0
            if self.null_gain > 0.0:
                g = self.null_gain * self._null_space_gradient(q)
                if self.null_projector == "svd":
                    # 精确 Moore–Penrose 零空间：截断阈值与上面的 rank 同源
                    _, _, Vt = np.linalg.svd(J_task, full_matrices=True)
                    Vr = Vt[:rank].T
                    dq_null = g - Vr @ (Vr.T @ g)
                else:
                    dq_null = g - J_pinv @ (J_task @ g)
                leak = float(np.linalg.norm(J_task @ dq_null))
                dq = dq + dq_null
            res.null_leak_trace.append(leak)

            dq_norm = float(np.linalg.norm(dq))
            res.dq_raw_trace.append(dq.copy())
            res.dq_raw_norm_trace.append(dq_norm)
            res.dq_raw_max = max(res.dq_raw_max, dq_norm)
            res.raw_residual_trace.append(
                float(np.linalg.norm(J_task @ dq - e_task))
            )

            dq_cmd = dq
            if self.step_clip is not None and dq_norm > self.step_clip:
                dq_cmd = dq * (self.step_clip / dq_norm)
                res.step_clip_count += 1
            res.dq_norm_trace.append(float(np.linalg.norm(dq_cmd)))
            res.clipped_residual_trace.append(
                float(np.linalg.norm(J_task @ dq_cmd - e_task))
            )

            q_new = q + dq_cmd
            q_clamped = self.robot.clamp_to_limits(q_new)
            if bool(np.any(q_clamped != q_new)):
                res.joint_limit_clip_count += 1
            q = q_clamped
            res.iters = it + 1

        # === 终点由独立 FK 复算验收，不信求解器自报 ===
        res.q = q.copy()
        p_f, R_f = self.robot.fk(q)
        res.fk_pos_err = float(np.linalg.norm(p_des - p_f))
        res.fk_rot_err = (
            0.0 if R_des is None else float(np.linalg.norm(so3_log(R_des @ R_f.T)))
        )
        res.pos_err, res.rot_err = res.fk_pos_err, res.fk_rot_err
        res.fk_verified = bool(
            res.fk_pos_err < pos_tol and (R_des is None or res.fk_rot_err < rot_tol)
        )
        res.task_err_ndim = float(
            max(
                res.fk_pos_err / pos_tol,
                0.0 if R_des is None else res.fk_rot_err / rot_tol,
            )
        )
        res.final_joint_margin = self._joint_margin(q)
        res.min_joint_margin = min(res.min_joint_margin, res.final_joint_margin)
        res.solve_time = time.perf_counter() - t0
        return res

    # -------------------------------------------------- 分支择优

    def _margin_class(self, margin: float) -> int:
        """限位安全等级：0 = 宽裕，1 = 偏紧，2 = 贴住限位。"""
        if margin >= 10.0 * self.margin_ok:
            return 0
        if margin >= self.margin_ok:
            return 1
        return 2

    def _score(self, res: IKResult) -> tuple:
        """无量纲择优键，按重要性字典序：

        1. 是否**真实**成功（求解器自报 ∧ 独立 FK 复核）
        2. 迭代中撞关节限位的次数（越少越好）
        3. 返回解的限位安全等级（越安全越好）
        4. 无量纲任务误差 max(pos_err/pos_tol, rot_err/rot_tol)

        绝不把米和弧度直接相加。
        """
        return (
            0 if res.success else 1,
            res.joint_limit_clip_count,
            self._margin_class(res.final_joint_margin),
            res.task_err_ndim,
        )

    @staticmethod
    def _summary(res: IKResult) -> dict:
        return {
            "branch": res.branch,
            "converged": res.converged,
            "fk_verified": res.fk_verified,
            "iters": res.iters,
            "time_s": res.solve_time,
            "pos_err": res.pos_err,
            "rot_err": res.rot_err,
            "task_err_ndim": res.task_err_ndim,
            "joint_limit_clip_count": res.joint_limit_clip_count,
            "step_clip_count": res.step_clip_count,
            "branch_switch_count": res.branch_switch_count,
            "final_joint_margin": res.final_joint_margin,
            "min_joint_margin": res.min_joint_margin,
        }

    def solve(
        self,
        p_des: np.ndarray,
        R_des: np.ndarray | None,
        q_init: np.ndarray,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        dual_branch: bool = True,
    ) -> IKResult:
        """求解逆运动学。

        第二分支**什么时候才跑**（避免无谓的 2× 开销）：仅当主分支
        「未通过独立 FK 验收」或「迭代中撞过关节限位」或「返回解的限位裕量
        小于 margin_ok」时才启动。两个分支的迭代数与耗时都记录在
        result.branch_report 里。

        两支的姿态误差是同一目标旋转的两个等价 lift（格点 k = 0 与 k = −1），
        在关节空间对应"绕近路"与"绕远路"，因此可能撞到不同的关节限位。
        """
        primary = self._solve_branch(
            p_des, R_des, q_init, max_iters, pos_tol, rot_tol, False
        )
        report = [self._summary(primary)]
        if R_des is None or not dual_branch:
            primary.branch_report = report
            return primary

        clean = (
            primary.success
            and primary.joint_limit_clip_count == 0
            and primary.final_joint_margin >= self.margin_ok
        )
        if clean:
            primary.branch_report = report
            return primary

        alt = self._solve_branch(
            p_des, R_des, q_init, max_iters, pos_tol, rot_tol, True
        )
        report.append(self._summary(alt))
        best = min((primary, alt), key=self._score)
        best.branch_report = report
        return best
