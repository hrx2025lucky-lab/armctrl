"""TOPP-RA 时间最优轨迹参数化，以及与逐段 S 曲线的对照。

三种时间律的定位
----------------
给定一条**几何路径**（走哪儿），时间参数化决定**什么时候走到哪**。

    逐段 S 曲线   armctrl.planning.path_timing.JointPathTrajectory
                  每段静止到静止，jerk 有界。代价：每个路径点都要停一次。
                  适合折线路径，因为折线拐角处切线跳变，不停就是加速度冲激。

    圆弧倒角 + TOPP-RA（本模块）
                  先把折线拐角倒成圆弧（armctrl.planning.blend），路径变 C¹，
                  再用 TOPP-RA 在速度/加速度约束下求**时间最优**速度剖面。
                  拐角不用停，总时长显著更短。
                  代价：加速度是 bang-bang 的，**jerk 无界**。

    这不是"谁更好"，而是"约束是什么"：
      要保护结构、抑制高频模态  →  要 jerk 有界，选 S 曲线
      要节拍、要产能            →  要时间最短，选 TOPP-RA

TOPP-RA 在做什么（零基础版）
----------------------------
把路径写成 q(s)，s 是沿路径走过的弧长。机器人的运动就剩一个自由度：
"s 随时间怎么变"。令 ṡ 为沿路径的速度，则

    q̇ = q′(s)·ṡ          q̈ = q′(s)·s̈ + q″(s)·ṡ²

关键观察：如果把 **x = ṡ²** 当未知量，那么"关节速度不超限""关节加速度不超限"
这两类约束对 (x, s̈) 都是**线性**的。于是"在每个 s 处 ṡ 最大能是多少"
变成一串线性规划，可以从终点往起点做一遍可达性分析（Reachability Analysis，
TOPP-RA 名字的来历），再从起点往终点贪心地取最大速度。

结果就是"在不违反约束的前提下，每一点都尽可能快"，即时间最优。

诚实边界
--------
* TOPP-RA 只约束速度与加速度，**不约束 jerk**。最优解必然在若干点上
  加速度从上界跳到下界（bang-bang），jerk 是冲激。
* 它是在**离散网格**上求解的，实测峰值会略微超过限值（本实现实测约 0.02%），
  这是离散化误差，不是实现错误。
* 倒角改变了几何路径，必须重新做碰撞检测（见 blend.blend_and_verify）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armctrl.planning.blend import BlendedPath, blend_corners


class ToppraTrajectory:
    """对一条几何路径做 TOPP-RA 时间最优参数化。

    接口与 path_timing.JointPathTrajectory 一致（duration / __call__ / sample），
    可以直接互换，便于同条路径做对照。

    参数
    ----
    path        BlendedPath，或一串路径点（会按折线处理，不倒角）
    v_max/a_max 逐关节上限，标量会广播
    n_samples   把路径离散成多少个样本喂给样条插值。太少会让样条偏离原路径，
                太多则求解变慢。默认按弧长自适应。
    """

    def __init__(self, path, v_max, a_max, n_samples: int | None = None,
                 n_grid: int = 2000, enforce_limits: bool = True,
                 max_attempts: int = 10, verify_dt: float = 0.002,
                 tol: float = 1e-3):
        import toppra
        import toppra.algorithm as algo
        import toppra.constraint as constraint

        if not isinstance(path, BlendedPath):
            path = blend_corners(path, max_deviation=0.0)
        self.path = path
        if path.length <= 1e-12:
            raise ValueError("路径退化：长度为零")

        probe = path(0.0)
        n = len(probe)
        self.n = n
        self.v_max = np.broadcast_to(np.asarray(v_max, dtype=float), (n,)).copy()
        self.a_max = np.broadcast_to(np.asarray(a_max, dtype=float), (n,)).copy()
        if np.any(self.v_max <= 0) or np.any(self.a_max <= 0):
            raise ValueError("速度/加速度上限必须为正")

        if n_samples is None:
            n_samples = int(np.clip(round(path.length * 120), 50, 1200))
        ss, Q = path.sample(n_samples)
        # toppra 要求路径参数单调递增；用归一化弧长，这样 s∈[0,1]
        self.grid = ss / path.length
        self._interp = toppra.SplineInterpolator(self.grid, Q)
        self._gridpoints = np.linspace(0.0, 1.0, int(n_grid))
        self._algo, self._cons = algo, constraint

        # TOPP-RA 只在离散网格点上施加约束，且 ParametrizeSpline 用样条去拟合
        # 本身带折点的 bang-bang 速度剖面，会在折点处过冲，实测峰值可能超限
        # 若干个百分点，且**随网格加密并不单调下降**。
        # 因此这里不假设求解结果满足限值，而是"求解 → 密采样实测 → 按超限比例
        # 降额 → 重解"，直到实测峰值真的落在限值以内，并如实记录降额系数。
        scale = 1.0
        self.attempts = 0
        for _ in range(int(max_attempts)):
            traj = self._solve(scale)
            ratio_v, ratio_a = self._peak_ratios(traj, verify_dt)
            self.attempts += 1
            self._traj = traj
            self.scale = scale
            self.peak_ratio = max(ratio_v, ratio_a)
            if not enforce_limits or self.peak_ratio <= 1.0 + tol:
                break
            scale *= 0.99 / max(ratio_v, ratio_a)
        self.duration = float(self._traj.duration)
        self.limits_satisfied = self.peak_ratio <= 1.0 + tol

    def _solve(self, scale: float):
        vlim = np.column_stack([-self.v_max, self.v_max]) * scale
        alim = np.column_stack([-self.a_max, self.a_max]) * scale
        pc_vel = self._cons.JointVelocityConstraint(vlim)
        pc_acc = self._cons.JointAccelerationConstraint(alim)
        # Interpolation 比默认的 Collocation 保守：它在网格区间上而不是只在
        # 网格点上施加约束，实测超限幅度小一个量级
        interp = self._cons.DiscretizationType.Interpolation
        pc_vel.set_discretization_type(interp)
        pc_acc.set_discretization_type(interp)
        inst = self._algo.TOPPRA([pc_vel, pc_acc], self._interp,
                                 gridpoints=self._gridpoints,
                                 parametrizer="ParametrizeSpline")
        traj = inst.compute_trajectory()
        if traj is None:
            raise RuntimeError("TOPP-RA 求解失败：检查速度/加速度上限是否为正")
        return traj

    def _peak_ratios(self, traj, dt: float) -> tuple[float, float]:
        """密采样实测峰值与限值之比。只看轨迹的输出，不看求解器内部。"""
        ts = np.arange(0.0, float(traj.duration), dt)
        V = np.abs(np.asarray(traj(ts, 1), dtype=float).reshape(len(ts), self.n))
        A = np.abs(np.asarray(traj(ts, 2), dtype=float).reshape(len(ts), self.n))
        return (float(np.max(V.max(axis=0) / self.v_max)),
                float(np.max(A.max(axis=0) / self.a_max)))

    def __call__(self, t: float):
        """返回 (q, qd, qdd)。t 会被钳制到 [0, duration]。"""
        t = float(np.clip(t, 0.0, self.duration))
        return (np.asarray(self._traj(t), dtype=float).reshape(-1),
                np.asarray(self._traj(t, 1), dtype=float).reshape(-1),
                np.asarray(self._traj(t, 2), dtype=float).reshape(-1))

    def sample(self, dt: float):
        n = int(np.floor(self.duration / dt)) + 1
        ts = np.arange(n) * dt
        Q = np.asarray(self._traj(ts), dtype=float).reshape(n, self.n)
        V = np.asarray(self._traj(ts, 1), dtype=float).reshape(n, self.n)
        A = np.asarray(self._traj(ts, 2), dtype=float).reshape(n, self.n)
        return ts, Q, V, A


@dataclass
class TimingReport:
    """一次时间参数化的可核对指标。"""

    name: str
    duration: float
    max_v: float
    max_a: float
    max_jerk: float
    stops: int
    path_length: float
    deviation: float = 0.0
    clearance: float = float("nan")

    def row(self) -> str:
        return (f"{self.name:<22}{self.duration:>9.3f}{self.max_v:>10.3f}"
                f"{self.max_a:>10.3f}{self.max_jerk:>12.1f}{self.stops:>8}"
                f"{self.path_length:>10.3f}{self.deviation*1e3:>10.2f}"
                f"{self.clearance*1e3:>11.2f}")

    @staticmethod
    def header() -> str:
        return (f"{'时间律':<22}{'时长/s':>9}{'峰值v':>10}{'峰值a':>10}"
                f"{'峰值jerk':>12}{'停顿':>8}{'路径长':>10}{'偏离/mm':>10}"
                f"{'最小间隙/mm':>11}")


def analyze(traj, name: str, path_length: float, dt: float = 0.002,
            v_eps: float = 1e-3, deviation: float = 0.0,
            clearance: float = float("nan")) -> TimingReport:
    """用采样出来的 q(t) 独立统计各项峰值。

    只用轨迹对外暴露的采样值，不看内部的 ṡ、Tj、Ta 等中间量：
      峰值速度/加速度  直接取 |q̇|、|q̈| 的最大值
      峰值 jerk        对 q̈ 做一阶差分（S 曲线的 jerk 分段常值，
                       TOPP-RA 的加速度在切换点跳变，差分会给出很大的值——
                       这正是要暴露的区别）
      停顿次数         速度范数掉到 v_eps 以下再重新上去，记一次
    """
    ts, Q, V, A = traj.sample(dt)
    speed = np.linalg.norm(V, axis=1)
    low = speed < v_eps
    # 内部（不含首尾）出现的"速度归零再启动"次数
    stops = 0
    i = 1
    while i < len(low) - 1:
        if low[i] and not low[i - 1]:
            j = i
            while j < len(low) - 1 and low[j]:
                j += 1
            if j < len(low) - 1:
                stops += 1
            i = j
        i += 1
    jerk = np.abs(np.diff(A, axis=0)) / dt if len(A) > 1 else np.zeros((1, A.shape[1]))
    return TimingReport(
        name=name, duration=float(traj.duration),
        max_v=float(np.abs(V).max()), max_a=float(np.abs(A).max()),
        max_jerk=float(jerk.max()), stops=stops, path_length=path_length,
        deviation=deviation, clearance=clearance,
    )
