"""把几何路径（折线）接上 S 曲线时间律，变成可执行轨迹。

思路
----
规划器给出的是**几何**路径：一串关节构型 q₀ → q₁ → … → qₙ，只说"走哪儿"，
不说"什么时候到哪儿"。控制器需要的是 q(t)、q̇(t)、q̈(t)。

对其中一段 qₐ → q_b：

    Δq = q_b − qₐ,   L = ‖Δq‖₂,   u = Δq / L        （u 是常向量，‖u‖₂ = 1）
    q(t) = qₐ + u · s(t)

因为 u 沿整段不变，逐关节求导直接得到

    q̇ᵢ = uᵢ · ṡ,     q̈ᵢ = uᵢ · s̈,     q⃛ᵢ = uᵢ · s⃛

于是"每个关节都不超限"等价于对标量 s 的限制：

    |q̇ᵢ| ≤ vᵢ  ∀i   ⟺   ṡ ≤ min_i ( vᵢ / |uᵢ| )     （只对 uᵢ ≠ 0 取最小）

加速度、jerk 同理。把这三个标量上限交给已有的 SCurveProfile，
它给出的 s(t) 就自动让**所有**关节同时满足各自的限制。
这是充分且不保守的：取最小值的那个关节恰好跑到限。

已知的取舍：路径点处速度归零
----------------------------
本实现把每一段都做成"静止到静止"的 S 曲线，因此机械臂会在每个路径点停一下。

原因：折线在拐角处方向 u 是**跳变**的。若拐角处速度不为零，则 q̇ = u·ṡ 跳变，
意味着加速度是冲激——任何有界力矩的机器人都执行不了。要在拐角保持速度，
必须先把几何路径在拐角处倒圆（样条或抛物线过渡），让 u 连续变化。
本实现**没有做**拐角倒圆，所以只能停。

代价是时间：短路平滑后路径点通常只剩 3–6 个，多花的时间可以接受；
收益是每一段都严格满足关节速度/加速度/jerk 限制，且边界条件干净。
"""

from __future__ import annotations

import numpy as np

from armctrl.planning.trajectory import SCurveProfile


class JointPathTrajectory:
    """折线关节路径 + 逐段七段 S 曲线时间律。

    参数
    ----
    waypoints   关节构型序列，至少两个点
    v_max/a_max/j_max
                逐关节上限，标量会广播到所有关节

    属性
    ----
    duration    总时长
    segments    每段的 (q_start, u, profile, t_offset)
    """

    def __init__(self, waypoints, v_max, a_max, j_max):
        pts = [np.asarray(p, dtype=float) for p in waypoints]
        if len(pts) < 2:
            raise ValueError("至少需要两个路径点")
        n = len(pts[0])
        self.n = n
        self.v_max = np.broadcast_to(np.asarray(v_max, dtype=float), (n,)).copy()
        self.a_max = np.broadcast_to(np.asarray(a_max, dtype=float), (n,)).copy()
        self.j_max = np.broadcast_to(np.asarray(j_max, dtype=float), (n,)).copy()
        if np.any(self.v_max <= 0) or np.any(self.a_max <= 0) or np.any(self.j_max <= 0):
            raise ValueError("速度/加速度/jerk 上限必须为正")

        self.waypoints = pts
        self.segments = []
        t = 0.0
        for qa, qb in zip(pts[:-1], pts[1:]):
            d = qb - qa
            L = float(np.linalg.norm(d))
            if L < 1e-12:                       # 重复点，跳过
                continue
            u = d / L
            v_s, a_s, j_s = self._scalar_limits(u)
            prof = SCurveProfile(L, v_s, a_s, j_s)
            self.segments.append((qa.copy(), u, prof, t))
            t += prof.T
        if not self.segments:
            raise ValueError("路径退化：所有点重合")
        self.duration = t

    def _scalar_limits(self, u: np.ndarray) -> tuple[float, float, float]:
        """把逐关节上限折算成路径参数 s 的标量上限。

        只有 uᵢ 明显非零的关节才形成约束：uᵢ = 0 的关节在这一段根本不动，
        它的限制永远满足，不该参与取最小（否则会除以 0）。
        """
        au = np.abs(u)
        act = au > 1e-12
        return (
            float(np.min(self.v_max[act] / au[act])),
            float(np.min(self.a_max[act] / au[act])),
            float(np.min(self.j_max[act] / au[act])),
        )

    def __call__(self, t: float):
        """返回 (q, qd, qdd)。t 超出 [0, duration] 时钳制到端点（速度为零）。"""
        t = float(np.clip(t, 0.0, self.duration))
        qa, u, prof, t0 = self.segments[-1]
        for seg in self.segments:
            if t <= seg[3] + seg[2].T:
                qa, u, prof, t0 = seg
                break
        s, sd, sdd = prof(t - t0)
        return qa + u * s, u * sd, u * sdd

    def sample(self, dt: float):
        """按固定步长采样整条轨迹，返回 (ts, Q, V, A)。"""
        n = int(np.floor(self.duration / dt)) + 1
        ts = np.arange(n) * dt
        Q = np.empty((n, self.n))
        V = np.empty((n, self.n))
        A = np.empty((n, self.n))
        for i, t in enumerate(ts):
            Q[i], V[i], A[i] = self(t)
        return ts, Q, V, A
