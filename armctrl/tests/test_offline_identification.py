"""离线辨识 / 激励轨迹设计 / 能耗模型 的直接测试。

判据独立性说明（第五轮 §0 要求）
--------------------------------
- ``OfflineBaseLS`` 的正确性用**合成数据**验证：自己造 A 和 β，算出 τ，
  再看解回来的 β 对不对。不碰机器人，也不拿被测实现的中间量作证。
- 增量 QR 的结果拿 ``np.linalg.lstsq`` 对整批数据直接求解来对照——
  两条完全不同的数值路径。
- 傅里叶轨迹的解析导数拿**数值差分**对照。
- 条件数放大律拿教科书不等式 ‖δβ‖/‖β‖ ≤ κ·‖δτ‖/‖τ‖ 当上界检查。
- 场景层只做「两个估计器在同一份数据上谁准」这一件事，
  两边的偏差都由 ``_identifiability_stats`` 用同一把尺子量。
"""

import inspect
import os
import unittest

import mujoco
import numpy as np

from armctrl.identification.offline_ls import (
    OfflineBaseLS, copper_loss, mechanical_power, potential_power)
from armctrl.tuner.scenes import AdaptiveScene, N_ARM
from armctrl.tests._docs import requires_docs

DOC = os.path.join(os.path.dirname(__file__), "..", "..",
                   "armctrl_讲解", "03_自适应控制.md")


def _synthetic(rows, ncol, cond, seed=0):
    """造一个条件数**指定**的最小二乘问题。

    用 SVD 反着构造：U diag(s) Vᵀ，s 从 1 按对数均匀降到 1/cond。
    这样 κ 是我们说了算的，不是测出来的——判据才独立。
    """
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.normal(size=(rows, ncol)))[0]
    V = np.linalg.qr(rng.normal(size=(ncol, ncol)))[0]
    s = np.logspace(0.0, -np.log10(cond), ncol)
    A = U @ np.diag(s) @ V.T
    beta = rng.normal(size=ncol)
    return A, beta, A @ beta


class TestOfflineLeastSquaresItself(unittest.TestCase):
    """不碰机器人，只验证求解器本身。"""

    def test_it_recovers_the_exact_parameters_from_clean_data(self):
        A, beta, tau = _synthetic(400, 20, cond=1e3, seed=1)
        ls = OfflineBaseLS(np.eye(20))
        for i in range(0, 400, 7):
            ls.add(A[i:i + 7], tau[i:i + 7])
        got, info = ls.solve()
        self.assertIsNotNone(got)
        self.assertLess(np.linalg.norm(got - beta) / np.linalg.norm(beta), 1e-10)
        self.assertEqual(info["rank"], 20)

    def test_reset_really_forgets_the_previous_run(self):
        """⭐ 杀掉「reset 只清计数不清 R」这种最难看出来的污染。

        先灌一批**完全无关**的数据，reset，再灌真数据。
        如果 R 因子没被清掉，两批互相矛盾的方程会混在一起解，
        结果必然偏离真值——而 rows 计数看起来却是对的。
        """
        A1, _, tau1 = _synthetic(400, 12, cond=1e2, seed=3)
        A2, beta2, tau2 = _synthetic(400, 12, cond=1e2, seed=4)
        ls = OfflineBaseLS(np.eye(12))
        ls.add(A1, tau1)
        ls.reset()
        self.assertEqual(ls.rows, 0)
        self.assertAlmostEqual(float(np.linalg.norm(ls._R)), 0.0,
                               msg="reset 之后 R 因子必须归零")
        ls.add(A2, tau2)
        got, info = ls.solve()
        self.assertEqual(info["rows"], 400)
        self.assertLess(np.linalg.norm(got - beta2) / np.linalg.norm(beta2),
                        1e-9, "reset 之后旧数据不该再影响结果")

    def test_incremental_qr_agrees_with_a_plain_batch_lstsq(self):
        """两条完全不同的数值路径，结果必须一致。"""
        A, beta, tau = _synthetic(300, 15, cond=1e5, seed=2)
        ref = np.linalg.lstsq(A, tau, rcond=None)[0]
        ls = OfflineBaseLS(np.eye(15))
        for i in range(0, 300, 11):
            ls.add(A[i:i + 11], tau[i:i + 11])
        got, _ = ls.solve()
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)

    def test_it_projects_onto_P_rather_than_solving_the_full_problem(self):
        """P 不是单位阵时，解出来的是**基参数坐标**，不是原始参数。"""
        rng = np.random.default_rng(3)
        P = np.linalg.qr(rng.normal(size=(30, 8)))[0]
        beta = rng.normal(size=8)
        Y = rng.normal(size=(200, 30))
        tau = Y @ (P @ beta)
        ls = OfflineBaseLS(P)
        ls.add(Y, tau)
        got, info = ls.solve()
        self.assertEqual(got.shape, (8,))
        np.testing.assert_allclose(got, beta, rtol=1e-8, atol=1e-9)
        self.assertLess(info["residual"], 1e-10)

    def test_residual_is_relative_and_matches_a_direct_recomputation(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(200, 10))
        tau = rng.normal(size=200)          # 故意不在 A 的列空间里
        ls = OfflineBaseLS(np.eye(10))
        ls.add(A, tau)
        got, info = ls.solve()
        direct = (np.linalg.norm(A @ got - tau) / np.linalg.norm(tau))
        self.assertAlmostEqual(info["residual"], direct, places=10)
        self.assertGreater(info["residual"], 0.5)   # 随机右端项拟合不了

    def test_it_refuses_to_guess_before_it_has_enough_rows(self):
        ls = OfflineBaseLS(np.eye(12))
        ls.add(np.ones((5, 12)), np.ones(5))
        got, info = ls.solve()
        self.assertIsNone(got)
        self.assertEqual(info["rows"], 5)
        self.assertEqual(info["rank"], 0)

    def test_rank_deficient_data_is_reported_not_hidden(self):
        rng = np.random.default_rng(5)
        B = rng.normal(size=(100, 6))
        A = np.hstack([B, B])               # 后 6 列是前 6 列的复制
        ls = OfflineBaseLS(np.eye(12))
        ls.add(A, A @ np.ones(12))
        _, info = ls.solve()
        self.assertEqual(info["rank"], 6)

    def test_reset_really_clears_the_accumulation(self):
        A, beta, tau = _synthetic(200, 10, cond=1e2, seed=6)
        ls = OfflineBaseLS(np.eye(10))
        ls.add(A, tau)
        ls.reset()
        self.assertEqual(ls.rows, 0)
        got, info = ls.solve()
        self.assertIsNone(got)
        self.assertTrue(np.isnan(info["residual"]))

    def test_ridge_shrinks_the_solution_and_is_off_by_default(self):
        A, beta, tau = _synthetic(200, 10, cond=1e6, seed=7)
        plain, _ = self._solve(A, tau, 10, ridge=0.0)
        reg, _ = self._solve(A, tau, 10, ridge=1e-2)
        self.assertLess(np.linalg.norm(reg), np.linalg.norm(plain))
        self.assertEqual(OfflineBaseLS(np.eye(3)).ridge, 0.0)

    @staticmethod
    def _solve(A, tau, n, ridge):
        ls = OfflineBaseLS(np.eye(n), ridge=ridge)
        ls.add(A, tau)
        return ls.solve()

    def test_bad_inputs_are_rejected_loudly(self):
        with self.assertRaises(ValueError):
            OfflineBaseLS(np.eye(4), ridge=-1.0)
        with self.assertRaises(ValueError):
            OfflineBaseLS(np.ones(4))                      # 一维
        ls = OfflineBaseLS(np.eye(5))
        with self.assertRaises(ValueError):
            ls.add(np.ones((3, 4)), np.ones(3))            # 列数不对
        with self.assertRaises(ValueError):
            ls.add(np.ones((3, 5)), np.ones(2))            # 行数对不上


class TestSmallResidualDoesNotMeanCorrectParameters(unittest.TestCase):
    """本项目最贵的一条教训，用合成数据把它钉死。"""

    @staticmethod
    def _one(cond):
        """同样大小的相对力矩误差，只改条件数，看参数错多少。"""
        A, beta, tau = _synthetic(600, 20, cond=cond, seed=8)
        rng = np.random.default_rng(9)
        noise = rng.normal(size=tau.size)
        noise *= 1e-4 * np.linalg.norm(tau) / np.linalg.norm(noise)
        ls = OfflineBaseLS(np.eye(20))
        ls.add(A, tau + noise)
        got, info = ls.solve()
        return (np.linalg.norm(got - beta) / np.linalg.norm(beta),
                info["residual"])

    def test_the_residual_stays_the_same_while_the_parameters_blow_up(self):
        """⭐ 一张表说清「残差小 ≠ 参数对」。

        四个条件数、同一个相对力矩误差 1e-4：
        残差**一模一样**，参数误差跨 5 个数量级。
        """
        rows = [(c,) + self._one(c) for c in (1e2, 1e4, 1e6, 1e8)]
        resid = [r[2] for r in rows]
        par = [r[1] for r in rows]

        # 残差对条件数完全不敏感
        self.assertLess(max(resid) / min(resid), 1.05,
                        f"残差不该随条件数变化，实测 {resid}")
        self.assertLess(max(resid), 2e-4)
        # 参数误差却跨越 5 个数量级，并且单调递增
        self.assertEqual(par, sorted(par))
        self.assertLess(par[0], 1e-3)
        self.assertGreater(par[-1], 10.0)
        self.assertGreater(par[-1] / par[0], 1e4)

    def test_a_well_conditioned_problem_stays_accurate_under_the_same_noise(self):
        rel, _ = self._one(1e2)
        self.assertLess(rel, 0.05)


class TestTheEnergyModelFunctions(unittest.TestCase):

    def test_mechanical_power_is_plain_dot_product_when_positive(self):
        self.assertAlmostEqual(mechanical_power([1.0, 2.0], [3.0, 4.0]), 11.0)

    def test_braking_energy_is_thrown_away_unless_k_regen_says_otherwise(self):
        self.assertAlmostEqual(mechanical_power([1.0], [-2.0]), 0.0)
        self.assertAlmostEqual(mechanical_power([1.0], [-2.0], k_regen=0.3), -0.6)

    def test_k_regen_outside_zero_to_one_is_rejected(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                mechanical_power([1.0], [1.0], k_regen=bad)

    def test_potential_power_is_positive_when_lifting(self):
        self.assertAlmostEqual(potential_power([2.0], [1.0], g=10.0), 20.0)
        self.assertAlmostEqual(potential_power([2.0], [-1.0], g=10.0), -20.0)
        with self.assertRaises(ValueError):
            potential_power([1.0, 2.0], [1.0])

    def test_copper_loss_matches_a_hand_computation(self):
        # τ=2, R=0.5, k_i=0.1, r=10  ->  4*0.5/(100*0.01) = 2.0
        self.assertAlmostEqual(copper_loss([2.0], [0.5], [0.1], [10.0]), 2.0)

    def test_copper_loss_is_quadratic_in_torque(self):
        one = copper_loss([1.0], [1.0], [1.0], [1.0])
        three = copper_loss([3.0], [1.0], [1.0], [1.0])
        self.assertAlmostEqual(three, 9.0 * one)

    def test_copper_loss_has_no_default_motor_constants(self):
        """⭐ 守规矩用的测试：不给默认值，就不会有人误用编出来的数。"""
        sig = inspect.signature(copper_loss)
        for name in ("resistance", "torque_const", "gear_ratio"):
            self.assertIs(sig.parameters[name].default,
                          inspect.Parameter.empty,
                          f"{name} 不能有默认值——Panda 的电机手册常数拿不到，"
                          f"随手编一个会让输出看起来像瓦特")

    def test_copper_loss_rejects_zero_gear_or_torque_constant(self):
        with self.assertRaises(ValueError):
            copper_loss([1.0], [1.0], [0.0], [1.0])
        with self.assertRaises(ValueError):
            copper_loss([1.0], [1.0], [1.0], [0.0])
        with self.assertRaises(ValueError):
            copper_loss([1.0, 1.0], [1.0], [1.0], [1.0])


class _SceneCase(unittest.TestCase):
    """共用一个已 build 的场景：build 里有一次 SVD，很贵。"""

    @classmethod
    def setUpClass(cls):
        cls.sc = AdaptiveScene()
        cls.model = cls.sc.build()
        cls.sub = int(round(cls.sc.dt / cls.model.opt.timestep))

    def _run(self, dur, **params):
        sc = self.sc
        for k, v in params.items():
            sc.set(k, v)
        sc.reset()
        t = 0.0
        tel = {}
        while t < dur:
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
            for _ in range(self.sub):
                mujoco.mj_step(self.model, sc.data)
            t += sc.dt
            tel = sc.telemetry(t, q, v)
        return tel


class TestOfflineBeatsOnlineOnTheSameData(_SceneCase):

    def test_the_two_estimators_are_orders_of_magnitude_apart(self):
        tel = self._run(8.0, controller="mbac", exc="sine",
                        freq=0.6, amp=0.30, gamma=0.5)
        self.assertGreater(tel["base_err"], 50.0,
                           "在线自适应在 8 秒内不该收敛，否则这条对照就没意义")
        self.assertLess(tel["ls_base_err"], 1.0)
        self.assertGreater(tel["base_err"] / max(tel["ls_base_err"], 1e-12), 100.0)

    def test_the_offline_estimate_does_not_care_about_the_adaptive_gain(self):
        """⭐ 最强的区分力：改控制器增益，在线那栏天翻地覆，离线那栏几乎不动。

        如果 ls_base_err 只是把 base_err 抄了一份，这个测试立刻挂。
        """
        off = self._run(8.0, controller="mbac", exc="sine",
                        freq=0.6, amp=0.30, gamma=0.0)
        on = self._run(8.0, controller="mbac", exc="sine",
                       freq=0.6, amp=0.30, gamma=0.5)
        # Γ=0 表示完全不学，在线那栏必须停在 100%
        self.assertGreater(off["base_err"], 99.0)
        self.assertGreater(off["base_err"] - on["base_err"], 5.0,
                           f"Γ=0 与 Γ=0.5 的在线偏差应该差很多："
                           f"{off['base_err']:.2f} vs {on['base_err']:.2f}")
        # 离线那栏两边都准，而且几乎相同——它根本不看控制器
        self.assertLess(max(off["ls_base_err"], on["ls_base_err"]), 1.0)

    def test_the_fit_residual_needs_no_ground_truth_and_stays_tiny(self):
        tel = self._run(6.0, controller="mbac", exc="sine", freq=0.6, amp=0.30)
        self.assertLess(tel["ls_resid"], 0.01)
        self.assertFalse(np.isnan(tel["ls_resid"]))

    def test_the_readouts_start_as_nan_instead_of_inventing_a_number(self):
        self.sc.set("controller", "mbac")
        self.sc.reset()
        tel = self.sc.telemetry(0.0, self.sc.data.qpos[self.sc.robot.qpos_idx],
                                self.sc.data.qvel[self.sc.robot.qvel_idx])
        self.assertTrue(np.isnan(tel["ls_base_err"]))
        self.assertTrue(np.isnan(tel["ls_resid"]))


class TestBetterExcitationActuallyBuysAccuracy(_SceneCase):
    """⭐ 闭上最后一环：条件数只是手段，参数准不准才是目的。

    没有这条测试，前面所有「条件数从 10^18 降到 10^2.5」的功夫
    都只是**过程指标**——没人证明过它买到了什么。
    """

    def test_the_designed_trajectory_gives_a_better_estimate(self):
        sine = self._run(25.0, controller="mbac", exc="sine",
                         freq=0.6, amp=0.30, gamma=0.5)
        four = self._run(25.0, controller="mbac", exc="fourier",
                         freq=0.6, amp=0.30, gamma=0.5)
        # 条件数和参数偏差必须**同向**变好——这才叫"条件数是放大倍率"
        self.assertLess(four["pe_cond"], sine["pe_cond"] - 1.0,
                        f"傅里叶的条件数应明显更小："
                        f"{four['pe_cond']:.2f} vs {sine['pe_cond']:.2f}")
        self.assertLess(four["ls_base_err"], sine["ls_base_err"] / 2.0,
                        f"条件数好了参数却没更准，说明这套设计没买到东西："
                        f"{four['ls_base_err']:.5f}% vs "
                        f"{sine['ls_base_err']:.5f}%")
        # 两边都必须已经是"很准"的量级，否则上面的比值没有意义
        self.assertLess(sine["ls_base_err"], 0.05)

    def test_the_online_law_is_not_rescued_by_better_excitation(self):
        """⚠️ 反面：激励变好帮不了在线自适应——限制它的不是数据质量。"""
        four = self._run(25.0, controller="mbac", exc="fourier",
                         freq=0.6, amp=0.30, gamma=0.5)
        self.assertGreater(
            four["base_err"], 50.0,
            f"就算用最好的激励，在线自适应仍应停在两位数百分比，"
            f"实测 {four['base_err']:.2f}%")
        self.assertGreater(four["base_err"] / four["ls_base_err"], 1000.0)


class TestExcitationTrajectoryDesign(_SceneCase):

    def _reference(self, exc, n=400, freq=0.6, amp=0.30):
        self.sc.set("exc", exc)
        self.sc.set("freq", freq)
        self.sc.set("amp", amp)
        ts = np.linspace(0.0, 1.0 / freq, n, endpoint=False)
        out = [self.sc._ref(t) for t in ts]
        return (ts, np.array([o[0] for o in out]), np.array([o[1] for o in out]),
                np.array([o[2] for o in out]))

    def test_the_sine_reference_leaves_three_joints_completely_still(self):
        _, q, _, _ = self._reference("sine")
        span = q.max(axis=0) - q.min(axis=0)
        self.assertEqual(int(np.sum(span < 1e-12)), 3)

    def test_the_fourier_reference_moves_every_joint(self):
        _, q, _, _ = self._reference("fourier")
        span = q.max(axis=0) - q.min(axis=0)
        self.assertEqual(span.size, N_ARM)
        self.assertTrue(np.all(span > 0.1),
                        f"每个关节都得动起来，实测行程 {np.round(span, 4)}")

    def test_the_fourier_amplitude_still_obeys_the_slider(self):
        for amp in (0.10, 0.30):
            _, q, _, _ = self._reference("fourier", amp=amp)
            peak = np.max(np.abs(q - q.mean(axis=0)), axis=0)
            self.assertTrue(np.all(peak < amp * 1.05 + 1e-9),
                            f"峰值 {np.round(peak, 4)} 超过滑块 {amp}")
            self.assertGreater(peak.min(), amp * 0.4)

    def test_the_analytic_derivatives_match_finite_differences(self):
        """独立判据：解析的 v/a 拿数值差分来验，不是拿自己验自己。"""
        h = 1e-6
        for exc in ("sine", "fourier"):
            self.sc.set("exc", exc)
            self.sc.set("freq", 0.6)
            self.sc.set("amp", 0.30)
            for t in (0.13, 0.77, 1.41):
                q0, v0, a0 = self.sc._ref(t)
                qp = self.sc._ref(t + h)[0]
                qm = self.sc._ref(t - h)[0]
                vp = self.sc._ref(t + h)[1]
                vm = self.sc._ref(t - h)[1]
                with self.subTest(exc=exc, t=t):
                    np.testing.assert_allclose((qp - qm) / (2 * h), v0,
                                               rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose((vp - vm) / (2 * h), a0,
                                               rtol=1e-5, atol=1e-5)

    def test_the_fourier_reference_repeats_every_period(self):
        self.sc.set("exc", "fourier")
        self.sc.set("freq", 0.6)
        T = 1.0 / 0.6
        for t in (0.2, 0.9):
            for i in (0, 1, 2):
                np.testing.assert_allclose(self.sc._ref(t + i * T)[0],
                                           self.sc._ref(t)[0], atol=1e-9)

    def _design_cond(self, exc):
        """设计期条件数：只用参考轨迹，不跑仿真。"""
        _, q, v, a = self._reference(exc, n=120)
        P = self.sc.mbac.P
        R = np.zeros((P.shape[1], P.shape[1]))
        for i in range(q.shape[0]):
            R = np.linalg.qr(
                np.vstack([R, self.sc.reg.regressor(q[i], v[i], a[i]) @ P]),
                mode="r")
        s = np.linalg.svd(R, compute_uv=False)
        return float(np.log10(s[0] / s[-1]))

    def test_the_designed_trajectory_is_orders_of_magnitude_better(self):
        sine = self._design_cond("sine")
        four = self._design_cond("fourier")
        self.assertGreater(sine, 12.0, "单频+只动4关节应该是近乎奇异的")
        self.assertLess(four, 4.0)
        self.assertGreater(sine - four, 10.0,
                           f"差距应有 10 个数量级以上，实测 {sine:.2f} vs {four:.2f}")

    def test_running_the_designed_trajectory_actually_lowers_the_readout(self):
        sine = self._run(8.0, controller="mbac", exc="sine", freq=0.6, amp=0.30)
        four = self._run(8.0, controller="mbac", exc="fourier",
                         freq=0.6, amp=0.30)
        self.assertGreater(sine["pe_cond"] - four["pe_cond"], 1.0,
                           f"实跑条件数应明显下降：{sine['pe_cond']:.2f} -> "
                           f"{four['pe_cond']:.2f}")
        self.assertLess(four["ls_base_err"], sine["ls_base_err"])


class TestTheChosenFourierSeedIsActuallyTheBestOne(_SceneCase):
    """⭐ 文档说「300 组里挑条件数最小的那一组」——那就把这句话真的验一遍。

    没有这条测试的话，把 ``_FOURIER_SEED`` 随手改成别的数，
    整套测试照样全绿（第一遍变异扫描里 M22 就是这么活下来的）。
    判据独立：这里**自己**在参考轨迹上累一遍 QR 算条件数，
    不读场景的 ``pe_cond``。
    """

    N_SEEDS = 300
    NPTS = 200

    def _design_cond(self, seed):
        sc = self.sc
        P = sc.mbac.P
        sc._FOURIER_SEED = seed
        sc._fc = None
        sc._fpk = None
        R = np.zeros((P.shape[1], P.shape[1]))
        for t in np.linspace(0.0, 1.0 / sc.get("freq"), self.NPTS,
                             endpoint=False):
            q, v, a = sc._ref(t)
            R = np.linalg.qr(np.vstack([R, sc.reg.regressor(q, v, a) @ P]),
                             mode="r")
        sv = np.linalg.svd(R, compute_uv=False)
        return float(np.log10(sv[0] / sv[-1]))

    def setUp(self):
        self.sc.set("exc", "fourier")
        self.sc.set("amp", 0.30)
        self.sc.set("freq", 0.6)
        self._orig = self.sc._FOURIER_SEED

    def tearDown(self):
        self.sc._FOURIER_SEED = self._orig
        self.sc._fc = None
        self.sc._fpk = None

    def test_seed_124_wins_the_search_it_claims_to_have_won(self):
        vals = {sd: self._design_cond(sd) for sd in range(self.N_SEEDS)}
        best = min(vals, key=vals.get)
        self.assertEqual(
            best, 124,
            f"讲解说 seed 124 是 {self.N_SEEDS} 组里最好的，"
            f"实测最好的是 seed {best}（{vals[best]:.3f}）")
        self.assertEqual(self._orig, 124, "场景用的种子应该就是搜出来的那个")
        # 讲解里的三个数：最好 2.538 / 中位 2.889 / 最差 3.242
        med = float(np.median(list(vals.values())))
        self.assertAlmostEqual(vals[124], 2.538, delta=0.01)
        self.assertAlmostEqual(med, 2.889, delta=0.01)
        self.assertAlmostEqual(max(vals.values()), 3.242, delta=0.01)

    def test_the_win_over_the_median_is_small_which_is_the_real_lesson(self):
        """⭐ 挑系数只赚 0.35 个数量级，远小于「让 7 关节都动」的 11 个。"""
        vals = [self._design_cond(sd) for sd in range(60)]
        gain = float(np.median(vals)) - self._design_cond(124)
        self.assertLess(gain, 1.0,
                        f"系数优化的收益应该很小，实测 {gain:.2f} 个数量级——"
                        f"如果它变大了，讲解里那句「大头在关节数不在系数」就要改")
        self.assertGreater(gain, 0.1)


class TestTheConditionNumberReadoutCanReportBadNews(_SceneCase):

    def test_pe_cond_is_not_silently_capped_at_the_rank_threshold(self):
        """⭐ 修掉的缺陷：老写法用 s0/s_{r-1}，按定义永远 ≤ 8。

        直接喂一个条件数 10^12 的 R 因子，看读数报不报得出来。
        """
        sc = self.sc
        r = sc.mbac.P.shape[1]
        rng = np.random.default_rng(11)
        U = np.linalg.qr(rng.normal(size=(r, r)))[0]
        s = np.logspace(0.0, -12.0, r)
        # 造一个在 P 的列空间里、奇异值谱已知的 R
        R_small = U @ np.diag(s) @ np.linalg.qr(rng.normal(size=(r, r)))[0].T
        sc._pe_R = sc.mbac.P @ R_small @ sc.mbac.P.T
        sc._pe_rows = sc.reg.n_par + 1
        sc._pe_t_last = -1e9
        _, cond = sc._pe_stats(1.0)
        self.assertGreater(cond, 9.0,
                           f"读数被截断在 8 以内了：{cond:.2f}")


@requires_docs
class TestTheDocumentQuotesTheMeasuredNumbers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(DOC, encoding="utf-8") as f:
            cls.text = f.read()

    def test_it_reports_both_estimators_with_the_measured_values(self):
        for token in ("84.40", "0.0071", "12000"):
            self.assertIn(token, self.text, f"讲解里缺少实测数 {token}")

    def test_it_records_the_excitation_design_result(self):
        for token in ("18.25", "2.538", "seed 124"):
            self.assertIn(token, self.text)

    def test_the_excitation_table_keeps_the_numbers_on_the_right_rows(self):
        """⭐ 光检查「18.25 出现过」太弱：它在正文里还出现好几次，
        把**表格那一格**改成编的数照样能通过。这里改成检查
        「哪一行写哪个数」，行和数必须绑在一起。
        """
        rows = {"只动 4 个关节": "18.25",
                "7 个关节全动": "7.26",
                "seed 124": "2.538"}
        for key, number in rows.items():
            hits = [ln for ln in self.text.splitlines()
                    if key in ln and ln.lstrip().startswith("|")]
            self.assertTrue(hits, f"激励表里找不到「{key}」这一行")
            self.assertTrue(
                any(number in ln for ln in hits),
                f"「{key}」那一行应该写 {number}，实际是：{hits}")

    def test_it_no_longer_claims_the_project_has_no_friction_parameters(self):
        self.assertNotIn("⛔ **完全没有**（$\\pi$ 里那 3 个摩擦参数从未被激励）",
                         self.text)
        self.assertIn("armature", self.text)

    def test_it_warns_that_a_small_residual_proves_nothing(self):
        # 加粗的那条警告本身必须在，而且正文里至少被引用两次
        self.assertIn("**残差小 ≠ 参数对。**", self.text)
        self.assertGreaterEqual(self.text.count("残差小 ≠ 参数对"), 2)
        self.assertIn("123.6", self.text)

    def test_the_condition_number_table_is_recomputed_not_typed_in(self):
        """⭐ 讲解里那张「残差不变、参数爆炸」的表必须和代码算出来的一致。

        改文档里任何一格，或者改求解器让放大规律变了，这条都会红。
        """
        one = TestSmallResidualDoesNotMeanCorrectParameters._one
        self.assertIn("9.905e-05", self.text)
        for cond, shown in ((1e2, 0.00017), (1e4, 0.0082),
                            (1e6, 0.481), (1e8, 31.6)):
            par, resid = one(cond)
            self.assertIn(f"{shown:g}", self.text,
                          f"讲解里缺 κ={cond:.0e} 那一格（应为 {shown:g}）")
            self.assertAlmostEqual(
                par, shown, delta=0.02 * shown,
                msg=f"κ={cond:.0e} 实测参数误差 {par:.5g}，"
                    f"讲解写的是 {shown:g}")
            self.assertAlmostEqual(resid, 9.905e-05, delta=1e-7)

    def test_it_shows_what_the_better_excitation_actually_bought(self):
        """条件数是手段，参数偏差才是目的——文档必须两个都报。"""
        line = [ln for ln in self.text.splitlines()
                if ln.lstrip().startswith("|") and "fourier" in ln
                and "2.53" in ln]
        self.assertTrue(line, "讲解里缺少 sine/fourier 的实跑对照行")
        self.assertIn("0.0013", line[0],
                      f"那一行应该报离线 LS 偏差 0.0013 %，实际是：{line[0]}")
        self.assertIn("71.64", line[0])

    def test_it_admits_copper_loss_cannot_be_computed_here(self):
        self.assertIn("copper_loss", self.text)
        self.assertIn("不公开", self.text)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
