"""MPC 数学不变量测试。**每一项不变量一个独立用例**，失败时能直接定位。

上一版把所有检查塞进一个 `main()`，再用 `test_all_invariants` 断言返回 0：
失败时只知道"有断言失败"，不知道是哪一项，也无法单独重跑。

本文件不修改 `armctrl/control/mpc.py`；需要看完整时域解时，直接对**同一个已经
setup 好的求解器**再 solve 一次（同一 QP、同样的 warm start，返回同一个解），
取出全部 N·n 个决策变量，而不是自己另建一个 QP 来"验证自己"。

覆盖：
    1. 预测矩阵 vs 独立逐步积分
    2. 代价矩阵装配 vs 独立装配；无约束解 vs 正规方程解析解
    3. 力矩约束（硬约束）
    4. 完整时域的关节位置约束
    5. 完整时域的关节速度约束
    6. 求解器状态
    7. 不可行 / 兜底动作
    8. mutation：关掉位置/速度约束后，第 4、5 项必须失败
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.control.mpc import JointSpaceMPC


N_JOINT = 7
HORIZON = 10
DT = 0.02
TAU_LIM = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
Q_LO = np.full(N_JOINT, -2.8)
Q_HI = np.full(N_JOINT, 2.8)
V_LIM = np.full(N_JOINT, 0.5)


def reference_rollout(x0, A_seq, dt):
    """独立参考实现：逐步积分双积分器（零阶保持精确离散）。"""
    n = len(x0) // 2
    q, v = x0[:n].copy(), x0[n:].copy()
    qs, vs = [], []
    for a in A_seq:
        q = q + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        qs.append(q.copy())
        vs.append(v.copy())
    return np.array(qs), np.array(vs)


def full_horizon_solution(mpc):
    """取出上一次 compute() 对应 QP 的**完整**输入序列 u*（N·n 个变量）。

    对同一个已 setup 的 OSQP 实例再 solve 一次：数据未变，warm start 下返回同一解。
    这是读取被测对象自己的解，不是另建一个 QP。
    """
    res = mpc._solver.solve()
    assert res.x is not None, "无法取回 QP 解"
    return np.asarray(res.x[: mpc.N * mpc.n], float), str(res.info.status)


def predict_full_horizon(mpc, q0, v0, u):
    x0 = np.concatenate([q0, v0])
    q_pred = (mpc.Sq @ x0 + mpc.Tq @ u).reshape(mpc.N, mpc.n)
    v_pred = (mpc.Sv @ x0 + mpc.Tv @ u).reshape(mpc.N, mpc.n)
    return q_pred, v_pred


def _push_into_limit_case(mpc):
    """一个"参考要求冲出关节限位"的算例，返回 (q0, v0, refs, M, h)。"""
    n, N = mpc.n, mpc.N
    q0 = np.full(n, 2.7)                     # 贴近上限
    v0 = np.zeros(n)
    q_ref = np.tile(np.full(n, 10.0), (N, 1))
    zeros = np.zeros((N, n))
    return q0, v0, (q_ref, zeros, zeros), np.eye(n), np.zeros(n)


class TestPredictionMatrices(unittest.TestCase):
    """1. 稠密预测矩阵必须等价于逐步积分。"""

    def test_matrices_match_independent_integration(self):
        rng = np.random.default_rng(0)
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        worst = 0.0
        for _ in range(200):
            x0 = rng.standard_normal(2 * N_JOINT) * 0.5
            A_seq = rng.standard_normal((HORIZON, N_JOINT)) * 2.0
            u = A_seq.reshape(-1)
            q_pred = (mpc.Sq @ x0 + mpc.Tq @ u).reshape(HORIZON, N_JOINT)
            v_pred = (mpc.Sv @ x0 + mpc.Tv @ u).reshape(HORIZON, N_JOINT)
            q_ref, v_ref = reference_rollout(x0, A_seq, DT)
            worst = max(worst, float(np.abs(q_pred - q_ref).max()),
                        float(np.abs(v_pred - v_ref).max()))
        self.assertLess(worst, 1e-12, f"预测矩阵与逐步积分偏差 {worst:.3e}")

    def test_causality_structure(self):
        """u_k 不得影响 k 之前的状态（下三角结构）。"""
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        n, N = N_JOINT, HORIZON
        for k in range(N):
            for i in range(k + 1, N):
                blk = mpc.Tq[k * n:(k + 1) * n, i * n:(i + 1) * n]
                self.assertLess(float(np.abs(blk).max()), 1e-15,
                                f"Tq 违反因果性：第 {k} 步依赖第 {i} 步输入")


class TestCostAssembly(unittest.TestCase):
    """2. 代价矩阵与解析最优解。"""

    def test_hessian_matches_independent_assembly(self):
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        m = HORIZON * N_JOINT
        H = (mpc.Tq.T @ (mpc.w_pos * np.eye(m)) @ mpc.Tq
             + mpc.Tv.T @ (mpc.w_vel * np.eye(m)) @ mpc.Tv
             + mpc.w_acc * np.eye(m)
             + mpc.D.T @ (mpc.w_rate * np.eye(m)) @ mpc.D)
        H = 0.5 * (H + H.T)
        self.assertLess(float(np.abs(mpc.H - H).max()), 1e-9)
        w = np.linalg.eigvalsh(mpc.H)
        self.assertGreater(float(w.min()), 0.0, "H 不是正定的，QP 不是凸的")

    def test_unconstrained_solution_matches_normal_equations(self):
        rng = np.random.default_rng(0)
        mpc = JointSpaceMPC(N_JOINT, np.full(N_JOINT, 1e9), Q_LO, Q_HI,
                            horizon=HORIZON, dt=DT, use_limits=False)
        M = np.eye(N_JOINT) * 2.0
        h = np.zeros(N_JOINT)
        x0 = rng.standard_normal(2 * N_JOINT) * 0.2
        q_ref = rng.standard_normal((HORIZON, N_JOINT)) * 0.1
        zeros = np.zeros((HORIZON, N_JOINT))
        tau, a0, info = mpc.compute(x0[:N_JOINT], x0[N_JOINT:], q_ref, zeros, zeros, M, h)
        g = (mpc.Tq.T @ mpc.Wq @ (mpc.Sq @ x0 - q_ref.reshape(-1))
             + mpc.Tv.T @ mpc.Wv @ (mpc.Sv @ x0)
             - mpc.Wa @ zeros.reshape(-1))
        a_analytic = np.linalg.solve(mpc.H, -g)
        err = float(np.abs(a_analytic[:N_JOINT] - a0).max())
        self.assertIn("solved", info["status"])
        self.assertLess(err, 1e-3, f"无约束解与正规方程解析解偏差 {err:.3e}")

    def test_feedback_linearisation_torque_identity(self):
        """返回的 τ 必须严格等于 M·a0 + h。"""
        rng = np.random.default_rng(4)
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        M = np.diag(rng.uniform(0.3, 3.0, N_JOINT))
        h = rng.standard_normal(N_JOINT) * 4.0
        q0 = rng.uniform(-1.0, 1.0, N_JOINT)
        v0 = rng.standard_normal(N_JOINT) * 0.3
        zeros = np.zeros((HORIZON, N_JOINT))
        tau, a0, _ = mpc.compute(q0, v0, np.tile(q0 + 0.5, (HORIZON, 1)),
                                 zeros, zeros, M, h)
        self.assertLess(float(np.abs(tau - (M @ a0 + h)).max()), 1e-12)


class TestTorqueConstraint(unittest.TestCase):
    """3. 力矩硬约束。"""

    def test_torque_limits_under_aggressive_references(self):
        rng = np.random.default_rng(0)
        worst, violations = 0.0, []
        for trial in range(60):
            mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
            M = np.diag(rng.uniform(0.3, 3.0, N_JOINT))
            h = rng.standard_normal(N_JOINT) * 4.0
            q0 = rng.uniform(-1.0, 1.0, N_JOINT)
            v0 = rng.standard_normal(N_JOINT) * 0.3
            q_ref = np.tile(q0 + rng.choice([-1.0, 1.0], N_JOINT) * 1.2, (HORIZON, 1))
            zeros = np.zeros((HORIZON, N_JOINT))
            tau, _, info = mpc.compute(q0, v0, q_ref, zeros, zeros, M, h)
            viol = float(np.max(np.abs(tau) - TAU_LIM))
            worst = max(worst, viol)
            if viol > 1e-4:
                violations.append(f"trial {trial}: 越限 {viol:.3e} status={info['status']}")
        self.assertEqual(violations, [], f"最大越限 {worst:.3e} N·m；{violations}")

    def test_torque_constraint_holds_over_the_full_horizon(self):
        """不是只有首步：整个时域的 M·a_k + h 都必须在限内。"""
        rng = np.random.default_rng(1)
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        M = np.diag(rng.uniform(0.5, 2.0, N_JOINT))
        h = rng.standard_normal(N_JOINT) * 3.0
        q0 = np.zeros(N_JOINT)
        zeros = np.zeros((HORIZON, N_JOINT))
        mpc.compute(q0, np.zeros(N_JOINT), np.tile(np.full(N_JOINT, 2.0), (HORIZON, 1)),
                    zeros, zeros, M, h)
        u, status = full_horizon_solution(mpc)
        self.assertIn("solved", status)
        A = u.reshape(HORIZON, N_JOINT)
        worst = float(np.max(np.abs(A @ M.T + h) - TAU_LIM))
        self.assertLess(worst, 1e-4, f"时域内力矩越限 {worst:.3e} N·m")

    def test_tau_margin_monotonically_reduces_first_step_acceleration(self):
        M = np.diag(np.full(N_JOINT, 1.5))
        h = np.zeros(N_JOINT)
        q0 = np.zeros(N_JOINT)
        v0 = np.zeros(N_JOINT)
        q_ref = np.tile(np.full(N_JOINT, 0.8), (HORIZON, 1))
        zeros = np.zeros((HORIZON, N_JOINT))
        amps = []
        for margin in (1.0, 0.3, 0.1):
            mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON,
                                dt=DT, tau_margin=margin)
            tau, a0, _ = mpc.compute(q0, v0, q_ref, zeros, zeros, M, h)
            amps.append(float(np.abs(a0).max()))
            viol = float(np.max(np.abs(tau) - TAU_LIM * margin))
            self.assertLess(viol, 1e-4, f"margin={margin} 越限 {viol:.3e}")
        self.assertTrue(amps[0] >= amps[1] >= amps[2] - 1e-9,
                        f"裕度收紧时首步加速度未单调减小：{amps}")


class TestStateConstraints(unittest.TestCase):
    """4/5. 完整时域的关节位置与速度约束（不是只看首步）。"""

    def _solve_pushed_case(self, v_limit=V_LIM, **kw):
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM * 100, Q_LO, Q_HI, v_limit=v_limit,
                            horizon=HORIZON, dt=DT, **kw)
        q0, v0, refs, M, h = _push_into_limit_case(mpc)
        mpc.compute(q0, v0, *refs, M, h)
        u, status = full_horizon_solution(mpc)
        q_pred, v_pred = predict_full_horizon(mpc, q0, v0, u)
        return mpc, q_pred, v_pred, status

    # 位置用例必须让**位置约束**成为起作用的那一个：把速度上限放宽，否则
    # v_limit=0.5 rad/s × 0.2 s 时域本身就只允许移动 0.1 rad，位置约束根本不 binding，
    # 用例会在位置约束被破坏时仍然通过（这是 mutation 测出来的真实缺口）。
    LOOSE_V = np.full(N_JOINT, 5.0)

    def test_position_constraint_over_full_horizon(self):
        """检查 QP **原始解**在整个时域上满足位置约束。

        注意这里不断言求解器状态：位置约束紧贴激活时 OSQP 达不到 eps=1e-8
        （见 test_tightly_active_position_constraint_hits_iteration_limit），
        但它返回的原始解本身仍然满足约束。两件事必须分开检查，不能混为一谈。
        """
        mpc, q_pred, _, status = self._solve_pushed_case(v_limit=self.LOOSE_V,
                                                         use_limits=True)
        viol = float(q_pred.max() - Q_HI[0])
        self.assertLess(viol, 1e-3,
                        f"时域内位置越限 {viol:.4e}（max q = {q_pred.max():.4f}，"
                        f"status={status}）")
        self.assertGreater(q_pred.max(), Q_HI[0] - 0.05,
                           "参考没有把关节推到限位附近，本用例是空的")

    def test_tightly_active_position_constraint_hits_iteration_limit(self):
        """**已知局限**（在 mpc.py 里，本轮不得修改生产代码，只做如实记录）。

        位置约束紧贴激活时，OSQP 在 max_iter=8000 内达不到 eps_abs=eps_rel=1e-8，
        状态是 "maximum iterations reached"。compute() 把任何非 "solved" 状态都
        当成求解失败，于是**丢弃这个其实可用的解**，退回纯制动 a = −8v。

        也就是说：这个场景下 MPC 实际上没有在做 MPC。这不是测试的问题，
        是生产代码的求解器容差 / 失败判据需要另开条目修的问题。
        """
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM * 100, Q_LO, Q_HI,
                            v_limit=self.LOOSE_V, horizon=HORIZON, dt=DT,
                            use_limits=True)
        q0, v0, refs, M, h = _push_into_limit_case(mpc)
        _, a0, info = mpc.compute(q0, v0, *refs, M, h)
        self.assertNotIn("solved", info["status"])
        self.assertEqual(mpc.solve_failures, 1)
        self.assertTrue(np.allclose(a0, -8.0 * v0),
                        "非 solved 时应当退回纯制动")
        # 但被丢弃的那个原始解其实是满足约束的
        u, _ = full_horizon_solution(mpc)
        q_pred, _ = predict_full_horizon(mpc, q0, v0, u)
        self.assertLess(float(q_pred.max() - Q_HI[0]), 1e-3)

    def test_velocity_constraint_over_full_horizon(self):
        mpc, _, v_pred, status = self._solve_pushed_case(v_limit=V_LIM,
                                                         use_limits=True)
        self.assertIn("solved", status)
        viol = float(np.abs(v_pred).max() - V_LIM[0])
        self.assertLess(viol, 1e-3,
                        f"时域内速度越限 {viol:.4e}（max |v| = {np.abs(v_pred).max():.4f}）")
        self.assertGreater(float(np.abs(v_pred).max()), 0.4 * V_LIM[0],
                           "参考没有逼近速度上限，本用例是空的")

    def test_mutation_disabling_limits_violates_position(self):
        """**mutation**：关掉状态约束后，同一算例必须**越限**。"""
        _, q_pred, _, _ = self._solve_pushed_case(v_limit=self.LOOSE_V,
                                                  use_limits=False)
        self.assertGreater(float(q_pred.max() - Q_HI[0]), 1e-2,
                           f"关掉限位后仍未越限（max q = {q_pred.max():.4f}），"
                           "说明位置约束用例根本没有约束力")

    def test_mutation_disabling_limits_violates_velocity(self):
        _, _, v_pred, _ = self._solve_pushed_case(v_limit=V_LIM, use_limits=False)
        self.assertGreater(float(np.abs(v_pred).max() - V_LIM[0]), 1e-2,
                           f"关掉限位后速度仍未越限（max |v| = {np.abs(v_pred).max():.4f}）")

    def test_slack_is_zero_when_the_problem_is_feasible(self):
        """精确罚函数：可行时松弛量必须精确为零（软约束 = 硬约束）。"""
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM * 100, Q_LO, Q_HI, v_limit=V_LIM,
                            horizon=HORIZON, dt=DT, use_limits=True, soft_state=True)
        q0, v0, refs, M, h = _push_into_limit_case(mpc)
        _, _, info = mpc.compute(q0, v0, *refs, M, h)
        self.assertLess(float(np.max(info["slack"])), 1e-6,
                        f"可行算例出现非零松弛 {np.max(info['slack']):.3e}")


class TestSolverStatusAndFallback(unittest.TestCase):
    """6/7. 求解器状态与不可行兜底。"""

    def test_status_is_solved_in_a_normal_case(self):
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        zeros = np.zeros((HORIZON, N_JOINT))
        _, _, info = mpc.compute(np.zeros(N_JOINT), np.zeros(N_JOINT),
                                 np.tile(np.full(N_JOINT, 0.3), (HORIZON, 1)),
                                 zeros, zeros, np.eye(N_JOINT), np.zeros(N_JOINT))
        self.assertIn("solved", info["status"])
        self.assertEqual(mpc.solve_failures, 0)

    def test_hard_state_constraints_go_infeasible_and_fall_back_to_braking(self):
        """当前状态已越限 + 硬状态约束 ⇒ QP 无解；兜底动作必须是纯制动 a = −8v。"""
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, v_limit=V_LIM,
                            horizon=HORIZON, dt=DT, use_limits=True, soft_state=False)
        self.assertEqual(mpc.n_slack, 0, "soft_state=False 时不应有松弛变量")
        q0 = Q_HI + 0.5                       # 已经越限
        v0 = np.full(N_JOINT, 1.0)            # 也超速
        zeros = np.zeros((HORIZON, N_JOINT))
        _, a0, info = mpc.compute(q0, v0, np.tile(q0, (HORIZON, 1)), zeros, zeros,
                                  np.eye(N_JOINT), np.zeros(N_JOINT))
        self.assertNotIn("solved", info["status"], f"竟然求解成功：{info['status']}")
        self.assertEqual(mpc.solve_failures, 1)
        self.assertTrue(np.allclose(a0, -8.0 * v0),
                        f"兜底动作不是纯制动：{a0}")

    # ---------------------------------------------------------------- P0-04
    # 下面三个用例守的是同一条安全契约：
    #   compute() 返回的 tau 无条件满足 |tau_i| <= tau_limit_i，
    #   且返回的 a0 与它一致（M a0 + h == tau）。
    # 旧实现只在求解成功时靠 QP 硬约束保证力矩，求解失败走 a = −8v 兜底时
    # 完全没有力矩边界。实测真实 Panda：primal infeasible 时关节 2 输出
    # 152.7 N·m，上限 87 N·m，1.76 倍越限。

    def test_infeasible_fallback_torque_never_exceeds_the_hard_limit(self):
        """★ QP 不可行时，最终执行的力矩仍必须逐关节满足硬上限。

        构造方式：单位惯量 + 零偏置，速度取得足够大，使制动力矩 −8v 天然超过
        力矩上限；同时把状态推到位限之外并关掉软化，逼 QP 报 infeasible。
        """
        tau_lim = np.full(N_JOINT, 5.0)
        mpc = JointSpaceMPC(N_JOINT, tau_lim, Q_LO, Q_HI, v_limit=V_LIM,
                            horizon=HORIZON, dt=DT, use_limits=True,
                            soft_state=False)
        q0 = Q_HI + 0.5
        v0 = np.full(N_JOINT, 3.0)            # −8·3 = −24，远超 5 N·m
        zeros = np.zeros((HORIZON, N_JOINT))
        tau, a0, info = mpc.compute(q0, v0, np.tile(q0, (HORIZON, 1)), zeros,
                                    zeros, np.eye(N_JOINT), np.zeros(N_JOINT))

        self.assertNotIn("solved", info["status"], "本用例必须走不可行分支")
        # 裁剪前确实越限——否则这个用例根本没测到东西
        self.assertTrue(np.all(np.abs(info["tau_unclipped"]) > tau_lim),
                        f"用例失效：裁剪前并未越限 {info['tau_unclipped']}")
        self.assertGreater(info["tau_excess"], 0.0)
        self.assertTrue(info["tau_clipped"])
        self.assertEqual(mpc.torque_clips, 1)

        # 真正的安全断言：逐关节
        for i in range(N_JOINT):
            self.assertLessEqual(
                abs(tau[i]), tau_lim[i] + 1e-9,
                f"关节 {i} 最终力矩 {tau[i]:.4f} 越过上限 {tau_lim[i]}")
        # 制动方向必须保留（裁剪不能把符号裁反）
        self.assertTrue(np.all(np.sign(tau) == np.sign(-v0)),
                        f"裁剪后制动方向被破坏：{tau}")
        # 返回的 a0 必须与被执行的 tau 自洽
        self.assertTrue(np.allclose(np.eye(N_JOINT) @ a0, tau, atol=1e-9),
                        f"a0 与 tau 不自洽：a0={a0} tau={tau}")

    def test_fallback_acceleration_stays_consistent_with_executed_torque(self):
        """★ 非单位惯量 + 非零偏置下，a0 与 tau 仍必须严格自洽。

        这条比上一条更严：M 非对角、h 非零时，"裁力矩"和"裁加速度"不是同一件事，
        必须真的反算 a0 = M⁻¹(tau − h)，不能只裁一个不被执行的中间量。
        """
        rng = np.random.default_rng(7)
        A = rng.normal(size=(N_JOINT, N_JOINT))
        M = A @ A.T + 3.0 * np.eye(N_JOINT)          # 对称正定
        h = rng.normal(scale=4.0, size=N_JOINT)
        tau_lim = np.full(N_JOINT, 6.0)
        mpc = JointSpaceMPC(N_JOINT, tau_lim, Q_LO, Q_HI, v_limit=V_LIM,
                            horizon=HORIZON, dt=DT, use_limits=True,
                            soft_state=False)
        q0 = Q_HI + 0.5
        v0 = np.full(N_JOINT, 2.0)
        zeros = np.zeros((HORIZON, N_JOINT))
        tau, a0, info = mpc.compute(q0, v0, np.tile(q0, (HORIZON, 1)), zeros,
                                    zeros, M, h)

        self.assertNotIn("solved", info["status"])
        self.assertTrue(info["tau_clipped"], "本用例应当触发裁剪")
        self.assertTrue(np.all(np.abs(tau) <= tau_lim + 1e-9),
                        f"最终力矩越限：{tau} vs {tau_lim}")
        self.assertTrue(np.allclose(M @ a0 + h, tau, atol=1e-8),
                        f"M·a0 + h = {M @ a0 + h} 与执行的 tau = {tau} 不符")
        # 内部记录的 a_prev 也必须是被执行的那个，而不是裁剪前的 −8v
        self.assertTrue(np.allclose(mpc._a_prev, a0, atol=1e-12))
        self.assertFalse(np.allclose(a0, -8.0 * v0),
                         "a0 仍等于未裁剪的 −8v，说明没有反算")

    def test_solved_path_is_untouched_by_the_torque_clip(self):
        """★ 对照组：求解成功时力矩本就是 QP 硬约束，裁剪必须是恒等操作。

        没有这条，上面两条可能被"无脑一律裁一刀"糊弄过去，
        那会悄悄改变正常路径的行为。
        """
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI,
                            horizon=HORIZON, dt=DT)
        zeros = np.zeros((HORIZON, N_JOINT))
        tau, a0, info = mpc.compute(
            np.zeros(N_JOINT), np.zeros(N_JOINT),
            np.tile(np.full(N_JOINT, 0.3), (HORIZON, 1)), zeros, zeros,
            np.eye(N_JOINT), np.zeros(N_JOINT))

        self.assertIn("solved", info["status"])
        self.assertEqual(mpc.torque_clips, 0, "正常路径不该发生有意义的裁剪")
        self.assertFalse(info["tau_clipped"])
        # 求解成功时 OSQP 的约束违反只到浮点噪声量级（实测 ~7e-11 N·m）
        self.assertLess(info["tau_excess"], 1e-6 * float(np.max(TAU_LIM)),
                        "正常路径的越限量应当只是数值噪声")
        # 裁剪前后逐元素一致到浮点噪声 ⇒ 恒等操作
        self.assertTrue(np.allclose(tau, info["tau_unclipped"], atol=1e-9))
        self.assertTrue(np.allclose(a0, tau, atol=1e-9))

    def test_reset_clears_the_torque_clip_counters(self):
        mpc = JointSpaceMPC(N_JOINT, np.full(N_JOINT, 5.0), Q_LO, Q_HI,
                            v_limit=V_LIM, horizon=HORIZON, dt=DT,
                            use_limits=True, soft_state=False)
        zeros = np.zeros((HORIZON, N_JOINT))
        mpc.compute(Q_HI + 0.5, np.full(N_JOINT, 3.0),
                    np.tile(Q_HI + 0.5, (HORIZON, 1)), zeros, zeros,
                    np.eye(N_JOINT), np.zeros(N_JOINT))
        self.assertEqual(mpc.torque_clips, 1)
        self.assertGreater(mpc.max_torque_excess, 0.0)
        mpc.reset()
        self.assertEqual(mpc.torque_clips, 0)
        self.assertEqual(mpc.max_torque_excess, 0.0)

    def test_soft_state_stays_feasible_in_the_same_situation(self):
        """同一越限初值下，软化后必须仍然可解——这正是软化的目的。"""
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, v_limit=V_LIM,
                            horizon=HORIZON, dt=DT, use_limits=True, soft_state=True)
        q0 = Q_HI + 0.5
        v0 = np.full(N_JOINT, 1.0)
        zeros = np.zeros((HORIZON, N_JOINT))
        _, _, info = mpc.compute(q0, v0, np.tile(q0, (HORIZON, 1)), zeros, zeros,
                                 np.eye(N_JOINT), np.zeros(N_JOINT))
        self.assertIn("solved", info["status"])
        self.assertEqual(mpc.solve_failures, 0)
        self.assertGreater(float(np.max(info["slack"])), 0.0,
                           "不可行算例的松弛量应当为正")

    def test_reset_clears_solver_state(self):
        mpc = JointSpaceMPC(N_JOINT, TAU_LIM, Q_LO, Q_HI, horizon=HORIZON, dt=DT)
        zeros = np.zeros((HORIZON, N_JOINT))
        mpc.compute(np.zeros(N_JOINT), np.zeros(N_JOINT),
                    np.tile(np.full(N_JOINT, 0.3), (HORIZON, 1)), zeros, zeros,
                    np.eye(N_JOINT), np.zeros(N_JOINT))
        self.assertTrue(np.any(mpc._a_prev != 0.0))
        mpc.reset()
        self.assertTrue(np.all(mpc._a_prev == 0.0))
        self.assertIsNone(mpc._solver)
        self.assertEqual(mpc.solve_failures, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
