"""模型预测控制（MPC）：把约束显式写进在线优化。

与本项目其它控制器的本质区别
----------------------------
PD / CTC / 阻抗 / 自适应都是**瞬时反馈律**：只看当前状态算出一个力矩，
遇到力矩上限只能事后限幅（clip）。限幅会破坏原本设计好的闭环特性，
在饱和期间控制器实际上处于开环，容易超调甚至失稳。

MPC 是**在线优化**：用模型预测未来 N 步，把力矩上限、关节限位、速度限位
作为约束显式写进优化问题，求解后只执行第一步，下一周期用新状态重新求解
（滚动时域 receding horizon）。因此它能"提前几步就知道要饱和"，主动降速避开。

设计：反馈线性化 + 线性 MPC
---------------------------
直接对完整非线性动力学做 MPC 需要在线解非凸问题，实时性差。本实现采用
经典的两层结构：

第一层  反馈线性化（计算力矩）
        τ = M(q)·a + h(q, q̇)
        其中 a 是"期望关节加速度"。代入 M q̈ + h = τ 得 q̈ = a，
        即系统被精确线性化成 n 个解耦的双积分器。

第二层  在双积分器上做线性 MPC，决策变量是 a
        状态 x = [q; q̇] ∈ R^{2n}
        离散化（零阶保持，精确）：
            q_{k+1}  = q_k + q̇_k·Δt + ½·a_k·Δt²
            q̇_{k+1} = q̇_k + a_k·Δt

约束为什么仍然是线性的（关键）
------------------------------
力矩上限写成 τ_min ≤ M(q)·a + h(q,q̇) ≤ τ_max。在一个控制周期内把 M 和 h
冻结在当前状态上，这就是关于决策变量 a 的**线性不等式**，因此整个问题
仍是凸二次规划（QP），可以用 OSQP 在毫秒级求解。

这是标准做法，代价是 M、h 在预测时域内被视为常数（冻结模型近似）。
预测时域较短（本实现 0.1–0.2 s）时该近似是合理的；时域很长或构型变化剧烈时
需要沿预测轨迹重新线性化（LTV-MPC），本实现未做。

代价函数
--------
    J = Σ_{k=1..N} [ ‖q_k − q_d,k‖²_Q + ‖q̇_k − q̇_d,k‖²_R ]
      + Σ_{k=0..N-1} [ ‖a_k − a_d,k‖²_S + ‖a_k − a_{k-1}‖²_S_rate ]

其中 a_d 是期望轨迹的前馈加速度。加入 a 的变化率惩罚可抑制求解结果抖动。

实现方式：稠密（condensed）QP
-----------------------------
把所有状态用初始状态和输入序列表示，消去状态变量，只保留 N·n 个输入作为
决策变量。矩阵规模小（7 关节 × 10 步 = 70 个变量），OSQP 求解很快。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

try:
    import osqp
    _HAS_OSQP = True
except ImportError:  # pragma: no cover
    _HAS_OSQP = False


class JointSpaceMPC:
    """反馈线性化 + 线性 MPC 的关节空间轨迹跟踪控制器。

    参数
    ----
    n           关节数
    horizon     预测步数 N
    dt          MPC 步长（可以大于仿真步长，减小问题规模）
    w_pos       位置跟踪权重
    w_vel       速度跟踪权重
    w_acc       加速度幅值权重（等价于控制代价）
    w_rate      加速度变化率权重（抑制抖动）
    tau_margin  力矩约束保留裕度比例，1.0 表示用满额定力矩
    use_limits  是否启用关节位置/速度约束
    """

    def __init__(
        self,
        n: int,
        tau_limit: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        v_limit: np.ndarray | None = None,
        horizon: int = 10,
        dt: float = 0.02,
        w_pos: float = 4000.0,
        w_vel: float = 40.0,
        w_acc: float = 0.05,
        w_rate: float = 0.5,
        tau_margin: float = 1.0,
        use_limits: bool = True,
        soft_state: bool = True,
        rho_lin: float = 1e6,
        rho_quad: float = 1e4,
    ):
        if not _HAS_OSQP:
            raise ImportError("需要 osqp：pip install osqp")

        self.n = int(n)
        self.N = int(horizon)
        self.dt = float(dt)
        self.tau_limit = np.asarray(tau_limit, float) * float(tau_margin)
        self.q_lower = np.asarray(q_lower, float)
        self.q_upper = np.asarray(q_upper, float)
        self.v_limit = (
            np.full(self.n, 3.0) if v_limit is None else np.asarray(v_limit, float)
        )
        self.use_limits = bool(use_limits)
        # 状态约束必须软化：一旦当前状态已越限，硬约束 q_k <= q_max 在整个
        # 预测时域上无解，QP 直接不可行。实测硬约束下 3 s 仿真出现 143~255 次
        # 求解失败，回退动作反而把关节推得更远。
        # 软化用精确罚函数：q_k <= q_max + s，s >= 0，代价加 rho_lin·Σs + rho_quad·‖s‖²。
        # rho_lin 足够大时（超过最优对偶变量的上界），可行时松弛量精确为零，
        # 即软约束与硬约束给出相同解；不可行时给出"越限最小"的解而不是失败。
        self.soft_state = bool(soft_state) and self.use_limits
        self.rho_lin = float(rho_lin)
        self.rho_quad = float(rho_quad)
        self.n_slack = self.n if self.soft_state else 0

        self.w_pos, self.w_vel = float(w_pos), float(w_vel)
        self.w_acc, self.w_rate = float(w_acc), float(w_rate)

        self._build_prediction_matrices()
        self._build_cost()
        self._solver = None
        self._a_prev = np.zeros(self.n)
        self.last_status = ""
        self.solve_failures = 0
        self.max_slack = 0.0
        #: 最终力矩被硬裁剪的次数。求解成功时应恒为 0（力矩已是 QP 的硬约束）；
        #: 只有在 QP 不可行、走制动兜底那条路径上才可能非零。
        self.torque_clips = 0
        #: 历史最大越限量：裁剪前 max(|tau| − tau_limit) 的最大值，单位 N·m。
        self.max_torque_excess = 0.0

    # ------------------------------------------------------------ 预测矩阵

    def _build_prediction_matrices(self) -> None:
        """构造稠密形式  Q = Sq·A + Tq·U,  V = Sv·A + Tv·U。

        双积分器逐步展开：
            q_k  = q_0 + k·dt·q̇_0 + Σ_{i<k} [ (k−i−½)·dt² ]·a_i
            q̇_k = q̇_0 + Σ_{i<k} dt·a_i
        """
        N, n, dt = self.N, self.n, self.dt
        I = np.eye(n)

        Sq = np.zeros((N * n, 2 * n))       # 初始状态 → 预测位置
        Sv = np.zeros((N * n, 2 * n))       # 初始状态 → 预测速度
        Tq = np.zeros((N * n, N * n))       # 输入序列 → 预测位置
        Tv = np.zeros((N * n, N * n))       # 输入序列 → 预测速度

        for k in range(1, N + 1):
            r = (k - 1) * n
            Sq[r:r + n, :n] = I
            Sq[r:r + n, n:] = (k * dt) * I
            Sv[r:r + n, n:] = I
            for i in range(k):
                c = i * n
                Tq[r:r + n, c:c + n] = ((k - i - 0.5) * dt * dt) * I
                Tv[r:r + n, c:c + n] = dt * I

        self.Sq, self.Sv, self.Tq, self.Tv = Sq, Sv, Tq, Tv

    def _build_cost(self) -> None:
        """P = 2(Tqᵀ Wq Tq + Tvᵀ Wv Tv + Wa + Dᵀ Wr D)，q 向量每步更新。"""
        N, n = self.N, self.n
        Wq = self.w_pos * np.eye(N * n)
        Wv = self.w_vel * np.eye(N * n)
        Wa = self.w_acc * np.eye(N * n)

        # 变化率算子 D：第 0 步与上一周期的 a_prev 比较（进 q 向量），其余为相邻差分
        D = np.zeros((N * n, N * n))
        for k in range(N):
            r = k * n
            D[r:r + n, r:r + n] = np.eye(n)
            if k > 0:
                D[r:r + n, (k - 1) * n:(k - 1) * n + n] = -np.eye(n)
        self.D = D
        Wr = self.w_rate * np.eye(N * n)

        self.H = (
            self.Tq.T @ Wq @ self.Tq
            + self.Tv.T @ Wv @ self.Tv
            + Wa
            + D.T @ Wr @ D
        )
        self.H = 0.5 * (self.H + self.H.T)      # 强制对称
        self.Wq, self.Wv, self.Wa, self.Wr = Wq, Wv, Wa, Wr

    # ------------------------------------------------------------ 求解

    def _slack_expand(self) -> np.ndarray:
        """把 n 个松弛量展开到 N·n 行：每个关节在整个时域共享一个松弛量。"""
        return np.tile(np.eye(self.n), (self.N, 1))

    def _constraint_matrix(self, M: np.ndarray) -> sparse.csc_matrix:
        """约束矩阵。决策变量顺序 [u (N·n); s (n_slack)]。

        行块：
            力矩   [blkdiag(M), 0]        硬约束（执行器物理上无法超出）
            位置上 [Tq, −E]               软化：q_free + Tq·u − s <= q_max
            位置下 [Tq, +E]               软化：q_free + Tq·u + s >= q_min
            速度上 [Tv, −E]
            速度下 [Tv, +E]
            松弛   [0, I]                 s >= 0

        位置/速度是状态约束，必须软化：当前状态已越限时硬约束无解，QP 直接
        不可行。上下界需要各一个**单边**行块（−E 与 +E），不能合成一个双边行。
        """
        N, n, ns = self.N, self.n, self.n_slack
        zero_u = sparse.csc_matrix((ns, N * n))
        pad = sparse.csc_matrix((N * n, ns))

        rows = [
            sparse.hstack(
                [sparse.block_diag([sparse.csc_matrix(M)] * N, format="csc"), pad],
                format="csc",
            )
        ]
        if self.use_limits:
            E = sparse.csc_matrix(self._slack_expand()) if ns else pad
            Tq = sparse.csc_matrix(self.Tq)
            Tv = sparse.csc_matrix(self.Tv)
            rows.append(sparse.hstack([Tq, -E], format="csc"))   # 上界
            rows.append(sparse.hstack([Tq, E], format="csc"))    # 下界
            rows.append(sparse.hstack([Tv, -E], format="csc"))
            rows.append(sparse.hstack([Tv, E], format="csc"))
        if ns:
            rows.append(
                sparse.hstack([zero_u, sparse.eye(ns, format="csc")], format="csc")
            )
        return sparse.vstack(rows, format="csc")

    def _constraint_bounds(self, x0, M, h):
        N, n, ns = self.N, self.n, self.n_slack
        INF = np.inf
        lo = [np.tile(-self.tau_limit - h, N)]
        hi = [np.tile(self.tau_limit - h, N)]

        if self.use_limits:
            q_free = self.Sq @ x0
            v_free = self.Sv @ x0
            lo.append(np.full(N * n, -INF))
            hi.append(np.tile(self.q_upper, N) - q_free)
            lo.append(np.tile(self.q_lower, N) - q_free)
            hi.append(np.full(N * n, INF))
            lo.append(np.full(N * n, -INF))
            hi.append(np.tile(self.v_limit, N) - v_free)
            lo.append(np.tile(-self.v_limit, N) - v_free)
            hi.append(np.full(N * n, INF))
        if ns:
            lo.append(np.zeros(ns))
            hi.append(np.full(ns, INF))
        return np.concatenate(lo), np.concatenate(hi)

    def compute(self, q, v, q_ref_seq, v_ref_seq, a_ref_seq, M, h):
        """求解一次 MPC，返回 (tau, a0, info)。

        q_ref_seq / v_ref_seq / a_ref_seq 形状 (N, n)，是未来 N 步的期望轨迹。
        M, h 为当前构型的质量矩阵与偏置力（由调用方提供，便于注入模型误差）。

        安全契约（无条件成立，与求解是否成功无关）
        ------------------------------------------
        返回的 ``tau`` **永远**满足 ``|tau_i| <= tau_limit_i``，并且返回的 ``a0``
        与它一致：``M @ a0 + h == tau``（在数值精度内）。

        - 求解成功时，力矩本来就是 QP 的硬约束，裁剪是恒等操作。
        - 求解**失败**时走制动兜底 ``a0 = −8v``，这条路径不受 QP 约束保护，
          必须在这里裁剪。旧实现漏了这一步：实测真实 Panda 上
          ``primal infeasible`` 时关节 2 返回 152.7 N·m（上限 87），**1.76 倍越限**。

        info 里的诊断字段
        -----------------
        ``tau_unclipped`` 裁剪前的力矩；``tau_excess`` 裁剪前的最大越限量（N·m，
        负值代表还有余量）；``tau_clipped`` 本周期是否真的裁了；
        ``saturated`` 越限是否超过 0.1%（用来区分数值毛刺和真越限）。
        """
        N, n = self.N, self.n
        x0 = np.concatenate([q, v])

        q_free = self.Sq @ x0
        v_free = self.Sv @ x0
        q_ref = np.asarray(q_ref_seq, float).reshape(-1)
        v_ref = np.asarray(v_ref_seq, float).reshape(-1)
        a_ref = np.asarray(a_ref_seq, float).reshape(-1)

        # 线性项：来自跟踪误差、加速度前馈、以及与上一周期 a_prev 的连续性
        g = (
            self.Tq.T @ self.Wq @ (q_free - q_ref)
            + self.Tv.T @ self.Wv @ (v_free - v_ref)
            - self.Wa @ a_ref
        )
        d_ref = np.zeros(N * n)
        d_ref[:n] = self._a_prev
        g = g - self.D.T @ self.Wr @ d_ref

        A_c = self._constraint_matrix(M)
        lo, hi = self._constraint_bounds(x0, M, h)

        ns = self.n_slack
        if ns:
            # 扩展到 [u; s]：松弛量用精确罚函数 rho_lin·Σs + rho_quad·‖s‖²
            nu = N * n
            H_ext = np.zeros((nu + ns, nu + ns))
            H_ext[:nu, :nu] = self.H
            H_ext[nu:, nu:] = self.rho_quad * np.eye(ns)
            g_ext = np.concatenate([g, 0.5 * self.rho_lin * np.ones(ns)])
            P = sparse.csc_matrix(2.0 * H_ext)
            qv = 2.0 * g_ext
        else:
            P = sparse.csc_matrix(2.0 * self.H)
            qv = 2.0 * g

        if self._solver is None:
            self._solver = osqp.OSQP()
            self._solver.setup(
                P, qv, A_c, lo, hi,
                verbose=False, warm_starting=True,
                # polish 会在求得近似解后做一步高精度精化，把约束违反压到 1e-10 量级。
                # 不开时 OSQP 只保证 eps_rel 级别的可行性（87 N·m 上限对应约 1e-3 N·m
                # 的越限），对硬力矩约束不够。
                polishing=True,
                eps_abs=1e-8, eps_rel=1e-8, max_iter=8000,
            )
        else:
            self._solver.update(q=qv, l=lo, u=hi)
            self._solver.update(Ax=A_c.data)

        res = self._solver.solve()
        self.last_status = str(res.info.status)
        ok = res.x is not None and np.all(np.isfinite(res.x))
        slack = np.zeros(max(ns, 1))
        if not ok or "solved" not in self.last_status:
            # 求解失败时**不能**退回 a_ref：若参考正把关节推向限位，
            # 前馈会继续加剧越限。改为纯制动（阻尼），这是安全的兜底动作。
            self.solve_failures += 1
            a0 = -8.0 * np.asarray(v, float)[:n]
        else:
            a0 = res.x[:n]
            if ns:
                slack = res.x[N * n:]
                self.max_slack = max(self.max_slack, float(np.max(slack)))

        self._a_prev = a0.copy()
        tau_raw = M @ a0 + h

        # ---- 硬力矩边界：以**最终执行的力矩**为安全边界 ----------------------
        # 为什么必须在这里裁：求解成功时力矩是 QP 的硬约束（见 _build_constraints
        # 里的 lo/hi = ∓tau_limit − h），本裁剪是恒等操作；但 QP **不可行**时走的是
        # 制动兜底 a0 = −8v，这条路径从未受任何力矩约束保护。
        # 实测（真实 Panda，M/h 取自 Pinocchio，速度取官方上限）：
        #   status = primal infeasible → 关节 2 返回 152.7 N·m，上限 87 N·m，**1.76 倍越限**。
        # 旧实现只把越限写进 info["saturated"] 这个标志位，没有任何代码据此裁剪，
        # 所以"求解失败后仍安全"并不成立。
        tau_raw = M @ a0 + h
        excess = float(np.max(np.abs(tau_raw) - self.tau_limit))
        self.max_torque_excess = max(self.max_torque_excess, excess)
        # 数值容差：OSQP 开了 polishing，求解成功时约束违反实测在 1e-11 N·m 量级
        # （目标 0.3 rad 时 +6.68e-11），那是浮点噪声不是"真越限"。
        # 容差取 1e-6×上限（87 N·m ⇒ 8.7e-5），比噪声高 6 个数量级，
        # 又比真实兜底越限（实测 65.7 N·m）低 6 个数量级，两边都留足余量。
        tol = 1e-6 * np.maximum(self.tau_limit, 1.0)
        tau = np.clip(tau_raw, -self.tau_limit, self.tau_limit)
        clipped = bool(np.any(np.abs(tau_raw) > self.tau_limit + tol))
        if clipped:
            self.torque_clips += 1
            # 反算加速度：返回给调用方的 a0、以及内部用于速率连续性的 _a_prev，
            # 都必须和**真正被执行的** tau 一致。否则下一周期的 a_prev 连续性项
            # 会基于一个从未发生过的加速度，状态预测也会系统性偏离。
            try:
                a0 = np.linalg.solve(M, tau - h)
            except np.linalg.LinAlgError:
                a0 = np.linalg.lstsq(M, tau - h, rcond=None)[0]
            self._a_prev = a0.copy()

        info = {
            "status": self.last_status,
            "a0": a0,
            "tau_unclipped": tau_raw.copy(),
            # 裁剪前是否越限（沿用 0.1% 容差，用于区分"数值毛刺"和"真越限"）
            "saturated": bool(np.any(np.abs(tau_raw) > self.tau_limit * 1.001)),
            # 本周期是否触发了**有意义的**裁剪（超过数值容差）
            "tau_clipped": clipped,
            # 裁剪前的最大越限量，负值表示留有余量
            "tau_excess": excess,
            "slack": slack.copy(),
        }
        return tau, a0, info

    def reset(self) -> None:
        self._a_prev[:] = 0.0
        self._solver = None
        self.solve_failures = 0
        self.max_slack = 0.0
        self.torque_clips = 0
        self.max_torque_excess = 0.0
