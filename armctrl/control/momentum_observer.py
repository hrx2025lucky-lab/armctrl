"""广义动量观测器：无力矩传感器的外力估计与碰撞检测。

原理
----
广义动量 p = M(q) q̇ 的导数满足

    ṗ = τ + Cᵀ(q,q̇) q̇ − g(q) + τ_ext

其中用到了 Ṁ = C + Cᵀ 这一斜对称性质。直接对上式积分需要 q̈，
噪声很大；De Luca 的做法是构造一阶低通形式的残差

    r(t) = K_I [ p(t) − p(0) − ∫₀ᵗ ( τ + Cᵀq̇ − g + r ) ds ]

则 r 满足 ṙ = K_I (τ_ext − r)，即 r 是外力矩 τ_ext 经一阶低通后的估计，
带宽由 K_I 决定。r 超过阈值即判定碰撞。

与 LESO 的关系
--------------
两者思想同源：都把"未建模量 + 外部作用"打包成一个待估计的总扰动，
用一个动态观测器把它在线估出来。
    LESO   扩张一个状态 z2 去估计总扰动 f，带宽 wo 决定收敛速度与噪声放大。
    动量观测器  用残差 r 估计外力矩 τ_ext，增益 K_I 决定带宽与噪声放大。
区别在于动量观测器利用了 Ṁ − 2C 斜对称这一结构性质，因此不需要 q̈，
并且能把估计量直接解释为关节外力矩，可进一步经 Jᵀ 伪逆映射成笛卡尔外力。
"""

from __future__ import annotations

import numpy as np

from armctrl.core.robot import ArmModel


class MomentumObserver:
    """基于广义动量的外力矩观测器。

    参数
    ----
    k_i    观测器增益（1/s），等于估计带宽。大 → 响应快但噪声放大。
    dt     离散步长。
    """

    def __init__(self, robot: ArmModel, k_i: float = 25.0, dt: float = 0.002):
        self.robot = robot
        self.k_i = float(k_i)
        self.dt = float(dt)
        self.n = robot.n
        self.reset()

    def reset(self, q: np.ndarray | None = None, qd: np.ndarray | None = None) -> None:
        self.integral = np.zeros(self.n)
        self.r = np.zeros(self.n)
        if q is not None and qd is not None:
            self.p0 = self.robot.mass_matrix(q) @ qd
        else:
            self.p0 = np.zeros(self.n)
        self._init = q is not None

    def _coriolis_transpose_qd(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """计算 Cᵀ(q,q̇) q̇。

        利用 Ṁ = C + Cᵀ  ⇒  Cᵀq̇ = Ṁ q̇ − C q̇。
        Ṁ 用中心差分沿当前速度方向估计：Ṁ ≈ [M(q+hq̇) − M(q−hq̇)] / (2h)。
        """
        h = 1e-6
        M_p = self.robot.mass_matrix(q + h * qd)
        M_m = self.robot.mass_matrix(q - h * qd)
        M_dot = (M_p - M_m) / (2.0 * h)
        return M_dot @ qd - self.robot.coriolis_times_qd(q, qd)

    def update(self, q: np.ndarray, qd: np.ndarray, tau_cmd: np.ndarray) -> np.ndarray:
        """推进一步，返回外力矩估计 r ≈ τ_ext。"""
        if not self._init:
            self.p0 = self.robot.mass_matrix(q) @ qd
            self._init = True

        p = self.robot.mass_matrix(q) @ qd
        beta = self.robot.gravity(q) - self._coriolis_transpose_qd(q, qd)
        self.integral += (tau_cmd - beta + self.r) * self.dt
        self.r = self.k_i * (p - self.p0 - self.integral)
        return self.r.copy()

    def cartesian_force(self, q: np.ndarray) -> np.ndarray:
        """把关节外力矩残差映射成笛卡尔外力/力矩：F ≈ (Jᵀ)⁺ r。"""
        J = self.robot.jacobian(q)
        return np.linalg.pinv(J.T) @ self.r

    def detect(self, threshold: np.ndarray | float) -> bool:
        """超阈值判定碰撞。threshold 可为标量或逐关节向量。"""
        thr = np.full(self.n, threshold) if np.isscalar(threshold) else np.asarray(threshold)
        return bool(np.any(np.abs(self.r) > thr))


class GravityCompensationTeaching:
    """拖动示教（零力控制）。

    控制律：τ = g(q) + C(q,q̇)q̇ + τ_friction_comp − D q̇

    三个要点：
      1. 重力补偿必须准确，否则手一松就掉。
      2. 摩擦补偿是能否"拖得动"的关键：静摩擦不补偿时需要很大人力才能起步，
         补偿过度又会自己爬行。这里用符号函数 + 速度死区做保守补偿。
      3. 小阻尼 D 用于抑制拖动时的振荡，代价是拖动阻力略增。
    """

    def __init__(
        self,
        robot: ArmModel,
        friction_coulomb: np.ndarray | None = None,
        friction_viscous: np.ndarray | None = None,
        comp_ratio: float = 0.8,
        damping: float = 0.5,
        vel_deadband: float = 1e-3,
    ):
        self.robot = robot
        n = robot.n
        self.fc = np.zeros(n) if friction_coulomb is None else np.asarray(friction_coulomb, float)
        self.fv = np.zeros(n) if friction_viscous is None else np.asarray(friction_viscous, float)
        # 摩擦补偿故意不补满：补过头会自激爬行（limit cycle）
        self.comp_ratio = float(comp_ratio)
        self.damping = float(damping)
        self.vel_deadband = float(vel_deadband)
        self.recorded: list[np.ndarray] = []

    def compute(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        tau = self.robot.bias(q, qd)
        # 速度死区内不做库仑补偿，避免零速附近来回切换导致抖动
        active = np.abs(qd) > self.vel_deadband
        tau_fric = self.comp_ratio * (self.fc * np.sign(qd) * active + self.fv * qd)
        tau = tau + tau_fric - self.damping * qd
        return self.robot.saturate_torque(tau)

    def record(self, q: np.ndarray) -> None:
        self.recorded.append(q.copy())

    def smoothed_trajectory(self, window: int = 21) -> np.ndarray:
        """对记录的示教轨迹做滑动平均，去掉人手抖动。"""
        if not self.recorded:
            return np.zeros((0, self.robot.n))
        traj = np.array(self.recorded)
        if window < 3 or traj.shape[0] < window:
            return traj
        kernel = np.ones(window) / window
        pad = window // 2
        out = np.empty_like(traj)
        for j in range(traj.shape[1]):
            padded = np.pad(traj[:, j], pad, mode="edge")
            out[:, j] = np.convolve(padded, kernel, mode="valid")
        return out


# ======================================================================
# 摩擦补偿的被动性判据（第四轮审核 P1-05）
#
# 旧说法：「补偿比 η < 1 就是安全的，因为净效果仍是阻碍」。
# ⚠️ 这个说法**只在控制器和被控对象用同一个摩擦模型时成立**。
#
# 本项目里两边并不一样：
#   * 被控对象（真实摩擦）： −f_c·tanh(v/a) − f_v·v      平滑，a = 2e-3
#   * 控制器（补偿）：      +η·f_c·sign(v)·1{|v|>db} + η·f_v·v   不平滑
#
# 在 |v| 刚出死区的地方，tanh(v/a) 还远没有饱和到 1
#   —— 例如 v = 1e-3 时 tanh(0.5) = 0.462 ——
# 而 sign(v) 已经跳到 1。于是补偿量 η·f_c 反而**大于**真实摩擦 0.462·f_c，
# 净效果变成顺着运动方向推：**注入能量**。
# ======================================================================


def _passivity_g(u, fc, fv, damping, tanh_scale):
    """被动性判据的**右端函数** g(u)。

    净功率 ``P(u) <= 0``（u = |v| > 死区）等价于::

        η·(f_c + f_v·u)  <=  f_c·tanh(u/a) + (f_v + D)·u
        ⟺  η  <=  g(u) := [f_c·tanh(u/a) + (f_v+D)·u] / (f_c + f_v·u)

    ⭐ 注意 **D（控制器阻尼）出现在分子里**——它是白送的耗散，
    会把允许的补偿比抬高。旧代码把 D 漏了，于是得到一个偏保守的界。
    """
    u = np.asarray(u, dtype=float)
    return (fc * np.tanh(u / tanh_scale) + (fv + damping) * u) / (fc + fv * u)


def _passivity_ceiling(fv, damping):
    """``sup_u g(u)``：补偿比 η 再大也不可能超过它。

    ``u -> ∞`` 时 ``g -> (f_v + D)/f_v = 1 + D/f_v``。
    ``f_v = 0`` 时若 ``D > 0`` 则 g 无上界（返回 ``inf``），
    若 ``D = 0`` 则 ``g -> 1``。
    """
    if fv <= 0.0:
        return float("inf") if damping > 0.0 else 1.0
    return 1.0 + damping / fv


def _search_top(level, fc, fv, damping):
    """给一个**可证明足够大**的扫描上限：超过它，``g(u) > level`` 必成立。

    因为 ``g(u) >= (f_v+D)·u / (f_c + f_v·u)``（丢掉非负的 tanh 项），
    令右边 ``>= level`` 解出 ``u >= f_c·level / ((f_v+D) − level·f_v)``。
    ⭐ 有了这个上限，"扫到 0.05 就收工"这种拍脑袋的截断就没有了。
    """
    den = (fv + damping) - level * fv
    if den <= 0.0:
        return float("inf")
    return fc * level / den


def friction_power_margin(fc, fv, eta, damping, deadband, tanh_scale,
                          v_max: float = 0.05, n: int = 200001):
    """单关节最坏情况净功率（W）。**> 0 表示系统在注入能量，即非被动。**

    净功率（速度 v，取 u = |v|）：

    .. code-block:: text

        P(u) = f_c·u·( η·1{u>db} − tanh(u/a) )      ← 库仑项
             + (η − 1)·f_v·u²                        ← 粘滞项（η<1 时耗散）
             − D·u²                                  ← 控制器阻尼（永远耗散）

    返回 ``(worst_power, worst_velocity)``。

    ⚠️ **扫描上限不是拍脑袋定的。** 记 ``c = D − (η−1)·f_v``：

    * ``c <= 0`` → 二次项非负，``P`` 随速度**无界增长**，
      返回 ``(inf, inf)``。旧版本会闷头扫到 ``v_max`` 然后报一个
      有限的小数字，等于**把发散当成了合格**。
    * ``c > 0`` → 用 ``tanh <= 1`` 放缩得 ``P(u) <= f_c·η·u − c·u²``，
      于是 ``u >= f_c·η/c`` 之后 ``P`` 必然 ≤ 0。把扫描上限撑到这里，
      峰值就**一定**落在窗口内。

    ⚠️ 上限里还留着一项 ``5·deadband``。它**现在是冗余的**：
    注能只可能发生在 ``u <= f_c·η/c`` 之内（``η<1`` 时更只在
    ``u <= a·atanh(η)`` 那一小段），第二条已经包住它了。
    保留是防御性的——万一以后有人改坏了 ``f_c·η/c`` 这一项。

    ⭐ 说清楚"这行代码现在没在起作用"很重要：变异测试里删掉它，
    测试**杀不掉**，那不是测试有洞，是这行本来就不承重。
    如实写下来，比留一句"没有它就会误报被动"的过期注释强。
    """
    fc = float(fc); fv = float(fv); eta = float(eta)
    damping = float(damping); deadband = float(deadband)
    c = damping - (eta - 1.0) * fv
    if c <= 0.0 and eta > 0.0:
        return float("inf"), float("inf")
    hi = max(float(v_max), 5.0 * deadband, 1.05 * fc * eta / c if c > 0 else 0.0)
    # ⚠️ 只用一张均匀网格撑到 hi 会**丢掉近原点的分辨率**：
    # 尖峰宽度是 tanh 尺度 a（2e-3）和死区量级，而 hi 可能是几米每秒，
    # 均匀网格的步长会比峰还宽，于是峰被跨过去、功率被低报。
    # 修法：近区细网格 + 远区粗网格拼起来，两边都不丢。
    near = min(hi, max(60.0 * float(tanh_scale), 20.0 * deadband,
                       float(v_max)))
    n = int(n)
    u = np.unique(np.concatenate([np.linspace(0.0, near, n),
                                  np.linspace(near, hi, n)]))
    coulomb = fc * u * (eta * (u > deadband) - np.tanh(u / tanh_scale))
    quad = (eta - 1.0) * fv * u ** 2 - damping * u ** 2
    p = coulomb + quad
    i = int(np.argmax(p))
    return float(p[i]), float(u[i])


def passivity_deadband(eta: float, tanh_scale: float,
                       fc: float = 1.0, fv: float = 0.0,
                       damping: float = 0.0) -> float:
    """保证被动所需的**最小**速度死区。

    要求死区外处处 ``η <= g(u)``（见 :func:`_passivity_g`），
    所以最小死区就是**违规区间的上确界**::

        db_min = sup{ u > 0 : g(u) < η }

    **两参数调用**（``f_v = D = 0``）退化成教科书里的闭式解::

        η <= tanh(u/a)  ⟺  db >= a·atanh(η)      η >= 1 时无解 → inf

    ⚠️ **但"η ≥ 1 就没救"只在 D = 0 时成立。**
    有控制器阻尼时上限是 ``1 + D/f_v``（见 :func:`_passivity_ceiling`）：
    例如 ``f_v=0.5, D=0.5`` 时 η 可以放到 **2.0**，
    ``η=1.2`` 只要死区 ≥ 1.0 rad/s 依然被动。
    传入 ``fc/fv/damping`` 才能拿到这个更准的答案。
    """
    e = float(eta); a = float(tanh_scale)
    fc = float(fc); fv = float(fv); damping = float(damping)
    if e <= 0.0:
        return 0.0
    if fv == 0.0 and damping == 0.0:
        return float("inf") if e >= 1.0 else a * float(np.arctanh(e))
    if e >= _passivity_ceiling(fv, damping):
        return float("inf")
    top = _search_top(e, fc, fv, damping)
    if not np.isfinite(top):
        return float("inf")
    u = np.geomspace(a * 1e-6, top * 1.05, 400001)
    bad = u[_passivity_g(u, fc, fv, damping, a) < e]
    if bad.size == 0:
        return 0.0
    lo, hi = float(bad.max()), top * 1.05
    for _ in range(200):                      # 二分收紧上确界
        mid = 0.5 * (lo + hi)
        if _passivity_g(mid, fc, fv, damping, a) < e:
            lo = mid
        else:
            hi = mid
    return hi


def max_passive_ratio(deadband: float, tanh_scale: float,
                      fc: float = 1.0, fv: float = 0.0,
                      damping: float = 0.0) -> float:
    """给定死区，能保证被动的**最大**补偿比 ``η_max = inf_{u>db} g(u)``。

    **两参数调用**（``f_v = D = 0``）退化成 ``tanh(db/a)``——
    默认 ``db = 1e-3, a = 2e-3`` 时只有 **0.4621**，
    ⭐ 远小于文档里长期写的「只要 η < 1 就安全」。

    ⚠️ 两参数形式是一个**保守（偏小）**的界，不是真界：它把
    ``f_v`` 和 ``D`` 这两份白送的耗散丢掉了。同样的默认参数
    （``f_c=2, f_v=0.5, D=0.5``）真界是 **0.4625015**，比 0.4621172 略大。
    偏小意味着"宁可少补一点"，**方向是安全的**，但当成精确值去引用就错了。
    """
    db = float(deadband); a = float(tanh_scale)
    fc = float(fc); fv = float(fv); damping = float(damping)
    if fv == 0.0 and damping == 0.0:
        return float(np.tanh(db / a))
    edge = float(_passivity_g(db, fc, fv, damping, a))
    top = _search_top(edge, fc, fv, damping)
    if not np.isfinite(top):
        top = max(db * 10.0, 1.0)
    u = np.geomspace(db, max(top * 1.05, db * 1.0001), 400001)
    return float(min(edge, float(_passivity_g(u, fc, fv, damping, a).min())))