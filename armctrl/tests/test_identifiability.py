"""可辨识性 oracle：把「参数收敛」拆开验证（第四轮审核 P0-08）。

被审核出来的问题
----------------
自适应场景把 ``‖π̂−π‖/‖π‖``（完整参数偏差）当成「持续激励让物理参数收敛」
的证据。但 Slotine-Li 只在基参数坐标 β 上更新，估计值恒为 ``π̂ = P·β``，
被**锁死在 P 的列空间里**。真参数 π 里那部分落在 ``(I − PPᵀ)`` 上的分量：

* 对关节力矩**没有任何影响**（这正是它不可辨识的定义）；
* 因此无论激励多充分、跑多久，都学不到；
* 于是完整参数偏差有一个**硬下限**，它降不下去说明不了任何事。

本文件的 oracle 全部独立于自适应律本身：
用真参数 π、投影矩阵 P、以及在**独立随机位形**上重新算的回归矩阵 Y，
不读控制器的任何中间量。
"""

import unittest

import numpy as np

from armctrl.tuner.scenes import AdaptiveScene


def _fresh_scene():
    sc = AdaptiveScene()
    sc.build()
    sc.reset()
    return sc


class _SceneFixture(unittest.TestCase):
    """建场景要做一次 300 样本的 SVD，很慢，整个类共用一个。"""

    @classmethod
    def setUpClass(cls):
        cls.sc = _fresh_scene()
        cls.P = cls.sc.mbac.P
        cls.pi = cls.sc.pi_true
        cls.reg = cls.sc.reg

    def setUp(self):
        # 每个用例开始前把估计清零，避免相互污染
        self.sc.mbac.reset()

    def _random_regressor(self, seed):
        rng = np.random.default_rng(seed)
        return self.reg.regressor(rng.uniform(-1.5, 1.5, self.reg.nv),
                                  rng.uniform(-1.0, 1.0, self.reg.nv),
                                  rng.uniform(-2.0, 2.0, self.reg.nv))


class TestNullSpaceIsReallyInvisible(_SceneFixture):
    """先证明「不可辨识」这件事本身是真的，否则后面的话都无意义。"""

    def test_a_pure_null_space_perturbation_changes_no_torque_at_all(self):
        # 造一个纯零空间向量：任取随机向量，减掉它在 P 上的投影
        rng = np.random.default_rng(11)
        raw = rng.normal(size=self.pi.size)
        null = raw - self.P @ (self.P.T @ raw)
        self.assertGreater(np.linalg.norm(null), 1.0,
                           "零空间维数应远大于 0，这个向量不该退化")

        pi_bad = self.pi + 5.0 * null / np.linalg.norm(null) * np.linalg.norm(self.pi)

        worst = 0.0
        for seed in range(40):
            Y = self._random_regressor(seed)
            ref = np.linalg.norm(Y @ self.pi)
            diff = np.linalg.norm(Y @ (pi_bad - self.pi))
            worst = max(worst, diff / max(ref, 1e-12))
        self.assertLess(worst, 1e-9,
                        f"把参数沿零空间挪了 5 倍‖π‖，力矩相对变化居然有 "
                        f"{worst:.3e}——那它就不是零空间")

    def test_but_that_same_perturbation_wrecks_the_full_parameter_error(self):
        """同一个扰动：力矩毫无变化，「完整参数偏差」却爆到 500%。"""
        rng = np.random.default_rng(11)
        raw = rng.normal(size=self.pi.size)
        null = raw - self.P @ (self.P.T @ raw)
        pi_bad = self.pi + 5.0 * null / np.linalg.norm(null) * np.linalg.norm(self.pi)

        full, base, pred = self.sc._identifiability_stats(pi_bad)
        self.assertGreater(full, 4.0,
                           f"完整参数偏差应 ≈500%，实测 {full * 100:.1f}%")
        self.assertLess(base, 1e-9,
                        f"可辨识子空间偏差应精确为 0，实测 {base:.3e}")
        self.assertLess(pred, 1e-9,
                        f"力矩预测误差应精确为 0，实测 {pred:.3e}")


class TestTheFloorIsRealAndLarge(_SceneFixture):

    def test_perfect_convergence_still_leaves_the_full_error_at_the_floor(self):
        """把 β 直接设成真值投影（比任何自适应律都强），完整偏差仍卡在下限。"""
        self.sc.mbac.beta[:] = self.P.T @ self.pi
        full, base, pred = self.sc._identifiability_stats(self.sc.mbac.pi_hat)

        self.assertAlmostEqual(full, self.sc._floor, places=12,
                               msg="β 完美收敛时，完整偏差应精确等于不可辨识下限")
        self.assertLess(base, 1e-9, f"可辨识子空间偏差应为 0，实测 {base:.3e}")
        self.assertLess(pred, 1e-9, f"力矩预测误差应为 0，实测 {pred:.3e}")

    def test_the_floor_is_not_a_rounding_error_it_is_most_of_the_parameters(self):
        # 实测 89.52%：能学的部分其实是少数
        self.assertGreater(self.sc._floor, 0.5,
                           f"下限只有 {self.sc._floor * 100:.2f}%，"
                           f"那这条纪律就没那么要紧了，需要重新检查")
        self.assertLess(self.sc._floor, 1.0)

    def test_full_error_can_never_dip_below_the_floor(self):
        """数学上 ‖π̂−π‖ ≥ ‖(I−PPᵀ)π‖，随便试都不该破。"""
        rng = np.random.default_rng(5)
        for k in range(30):
            self.sc.mbac.beta[:] = rng.normal(scale=2.0, size=self.sc.mbac.beta.size)
            full, _, _ = self.sc._identifiability_stats(self.sc.mbac.pi_hat)
            self.assertGreaterEqual(
                full, self.sc._floor - 1e-12,
                f"第 {k} 次随机 β 让完整偏差 {full:.6f} 掉到下限 "
                f"{self.sc._floor:.6f} 以下，说明下限算错了")


class TestProjectionAndDimensions(_SceneFixture):

    def test_projection_columns_are_orthonormal(self):
        G = self.P.T @ self.P
        self.assertLess(np.abs(G - np.eye(G.shape[0])).max(), 1e-10)

    def test_rank_is_strictly_smaller_than_the_parameter_count(self):
        self.assertEqual(self.P.shape, (self.reg.n_par, self.sc.mbac.rank))
        self.assertLess(self.sc.mbac.rank, self.reg.n_par,
                        "rank 若等于 n_par 就没有不可辨识方向，"
                        "整个 P0-08 也就不成立了")

    def test_parameter_count_comes_from_the_model_not_a_hardcoded_91(self):
        """文档里手写的 91 是错的；现场读出来是 10·nb + 3·nb = 117。"""
        self.assertEqual(self.reg.n_par, 117)
        self.assertNotEqual(self.reg.n_par, 91)
        # 而且要真的是「读出来的」：readout 值必须跟 regressor 对得上
        tel = self._telemetry_now()
        self.assertEqual(int(tel["n_par_now"]), self.reg.n_par)
        self.assertEqual(int(tel["rank_now"]), self.sc.mbac.rank)

    def _telemetry_now(self):
        q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
        v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
        return self.sc.telemetry(0.0, q, v)


class TestSceneReadouts(_SceneFixture):

    def test_all_four_identifiability_readouts_are_present(self):
        q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
        v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
        tel = self.sc.telemetry(0.0, q, v)
        for key in ("param_err", "param_floor", "base_err", "pred_err",
                    "n_par_now", "rank_now"):
            self.assertIn(key, tel, f"读数 {key} 没接上")
        declared = {r.key for r in AdaptiveScene.readouts}
        self.assertTrue(declared.issubset(set(tel)),
                        f"声明了但没输出：{declared - set(tel)}")

    def test_exact_ctc_reports_zero_on_every_identifiability_metric(self):
        """对照组：ctc_exact 用的就是真参数，四个量都该是 0。"""
        self.sc.set("controller", "ctc_exact")
        q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
        v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
        tel = self.sc.telemetry(0.0, q, v)
        for key in ("param_err", "param_floor", "base_err", "pred_err"):
            self.assertAlmostEqual(tel[key], 0.0, places=9,
                                   msg=f"{key} 在 ctc_exact 下应为 0")
        self.sc.set("controller", "mbac")

    def test_rbf_reports_nan_instead_of_inventing_a_number(self):
        """RBF 逼近的是函数，没有可比的「真参数」——空着比编一个数诚实。"""
        self.sc.set("controller", "rbf")
        q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
        v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
        tel = self.sc.telemetry(0.0, q, v)
        for key in ("param_err", "base_err", "pred_err"):
            self.assertTrue(np.isnan(tel[key]), f"{key} 应为 NaN")
        self.sc.set("controller", "mbac")


# ======================================================================
# 以下是第五轮审核 P0-08 的修复验收。
#
# 第四轮的测试全部复用 sc.mbac.P 和 sc._floor，把投影换成任何别的秩都照样
# 全绿——等于没在测 rank 这件事。这里的 oracle 一律**自己重新算**。
# ======================================================================

def _standalone_regressor():
    """不建场景，独立造一个和 AdaptiveScene 同构型的回归器。

    故意不走 AdaptiveScene：如果场景把夹爪工况配错了，用场景的对象来测
    等于用被测实现给自己作证。
    """
    from armctrl.identification.regressor import DynamicsRegressor
    from armctrl.tuner.scenes import PANDA_XML, _make_arm_model
    reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
    reg.set_finger_positions(0.04)          # 场景默认 gripper_width=0.08
    return reg


class TestGripperModeDeterminesTheRank(unittest.TestCase):
    """rank 不是一个常数，它取决于「假设夹爪能怎么动」。

    第五轮审核发现旧实现让两指**各自独立**随机，秩虚高到 70。
    平行夹爪只有一个自由度，物理上做不到。
    """

    #: 独立复算得到的三组秩。改动实现时这三个数必须原地不动。
    EXPECTED = {"fixed": 62, "coupled": 67, "independent": 70}

    @classmethod
    def setUpClass(cls):
        cls.reg = _standalone_regressor()

    def test_the_three_gripper_modes_give_exactly_62_67_70(self):
        for mode, want in self.EXPECTED.items():
            _, r, _ = self.reg.base_parameter_svd(samples=150, finger_mode=mode)
            self.assertEqual(r, want,
                             f"{mode} 工况下秩应为 {want}，实测 {r}。"
                             f"这三个数是独立 SVD 复算出来的，不许漂。")

    def test_the_default_is_fixed_because_that_is_what_scenes_actually_do(self):
        """所有场景 build() 里 set_finger_positions() 之后手指就不动了。"""
        _, r_default = self.reg.base_parameter_projection(samples=150)
        self.assertEqual(r_default, self.EXPECTED["fixed"],
                         "默认工况必须是 fixed（=62）。默认成 independent"
                         "就等于偷偷假设了一个不存在的夹爪。")

    def test_the_lower_level_svd_entry_point_defaults_to_fixed_too(self):
        """两个入口各有一份默认值，只测其中一个会漏。"""
        _, r, _ = self.reg.base_parameter_svd(samples=150)
        self.assertEqual(r, self.EXPECTED["fixed"],
                         "base_parameter_svd 自己的默认值也必须是 fixed")

    def test_the_impossible_mode_inflates_the_rank_by_eight(self):
        """把「做不到的工况」当成默认，会凭空多出 8 维可辨识性。"""
        gap = self.EXPECTED["independent"] - self.EXPECTED["fixed"]
        self.assertEqual(gap, 8)
        # 联动（真做得到的最强激励）也只能到 67，仍比 70 少 3
        self.assertLess(self.EXPECTED["coupled"], self.EXPECTED["independent"],
                        "同宽联动是平行夹爪的上限，不可能追平双指独立")

    def test_rank_does_not_depend_on_sample_count_or_seed(self):
        """秩是结构性质。它要是随抽样漂，这三个数就没有资格被引用。"""
        for mode, want in self.EXPECTED.items():
            for samples in (60, 150):
                for seed in (0, 7, 2024):
                    _, r, _ = self.reg.base_parameter_svd(
                        samples=samples, seed=seed, finger_mode=mode)
                    self.assertEqual(
                        r, want,
                        f"{mode} samples={samples} seed={seed} 给出 {r}，"
                        f"不是 {want}——秩不该依赖抽样")

    def test_every_mode_has_a_singular_value_cliff_of_at_least_1e9(self):
        """秩要能被引用，断崖必须干净——否则「阈值 1e-8」就是随手挑的。"""
        for mode, want in self.EXPECTED.items():
            _, r, s = self.reg.base_parameter_svd(samples=150,
                                                  finger_mode=mode)
            cliff = s[r - 1] / max(s[r], 1e-300)
            self.assertGreater(
                cliff, 1e9,
                f"{mode} 在 s[{r-1}]/s[{r}] 处只有 {cliff:.2e} 倍断崖，"
                f"说明 rank={want} 是阈值挑出来的，不是结构决定的")

    def test_the_rank_is_insensitive_to_the_threshold(self):
        """断崖够宽的话，阈值在 1e-6~1e-12 之间随便取都是同一个秩。"""
        for mode, want in self.EXPECTED.items():
            for tol in (1e-6, 1e-8, 1e-10, 1e-12):
                _, r, _ = self.reg.base_parameter_svd(
                    samples=150, finger_mode=mode, tol=tol)
                self.assertEqual(r, want,
                                 f"{mode} 在 tol={tol:g} 下变成 {r}")

    def test_an_unknown_finger_mode_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            self.reg.base_parameter_svd(samples=20, finger_mode="whatever")


class TestTheSceneReallyUsesTheFixedGripperRank(_SceneFixture):
    """把上面的独立结论接到真场景上：调参台报的必须是 62。"""

    def test_controller_rank_matches_the_independently_computed_62(self):
        reg = _standalone_regressor()
        _, r_indep, _ = reg.base_parameter_svd(samples=150,
                                               finger_mode="fixed")
        self.assertEqual(self.sc.mbac.rank, r_indep,
                         f"场景用的秩是 {self.sc.mbac.rank}，独立复算是 "
                         f"{r_indep}——场景配错了夹爪工况")
        self.assertEqual(self.sc.mbac.rank, 62)

    def test_the_readout_shows_62_not_70(self):
        q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
        v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
        tel = self.sc.telemetry(0.0, q, v)
        self.assertEqual(int(tel["rank_now"]), 62)
        self.assertNotEqual(int(tel["rank_now"]), 70,
                            "70 是双指独立随机的产物，物理上做不到")

    def test_the_floor_matches_the_audits_independent_recomputation(self):
        """第五轮审核独立算出固定夹爪下的下界约 90.207%。"""
        self.assertAlmostEqual(self.sc._floor * 100.0, 90.21, places=1,
                               msg=f"下界 {self.sc._floor*100:.2f}% 对不上"
                                   f"独立复算的 90.21%")
        self.assertNotAlmostEqual(self.sc._floor * 100.0, 89.52, places=1,
                                  msg="89.52% 是错误 P70 下的旧数字")


class TestWhatIsUnitInvariantAndWhatIsNot(unittest.TestCase):
    """审核要求：明确证明哪些量**不**具物理不变性。

    做法：把参数重标定 π' = Dπ，同时 Y' = YD⁻¹，于是 Y'π' ≡ Yπ——
    力矩一个字节都没变，只是换了单位。
    真正的物理性质必须不动；靠欧氏范数比出来的百分比会乱跳。
    """

    @classmethod
    def setUpClass(cls):
        cls.reg = _standalone_regressor()
        cls.pi = cls.reg.true_parameters()
        rng = np.random.default_rng(7)
        cls.W = np.vstack([
            cls.reg.regressor(rng.uniform(-1.5, 1.5, cls.reg.n_arm),
                              rng.uniform(-1.0, 1.0, cls.reg.n_arm),
                              rng.uniform(-2.0, 2.0, cls.reg.n_arm))
            for _ in range(120)])

    def _rank_and_floor(self, d):
        W = self.W @ np.diag(1.0 / d)
        _, s, Vt = np.linalg.svd(W, full_matrices=False)
        r = int(np.sum(s > s[0] * 1e-8))
        P = Vt[:r].T
        pi = d * self.pi
        null = pi - P @ (P.T @ pi)
        return r, float(np.linalg.norm(null) / np.linalg.norm(pi))

    def _scalings(self):
        n = self.pi.size
        rng = np.random.default_rng(3)
        return [("原单位", np.ones(n)),
                ("质量项 kg→g", np.where(np.arange(n) % 10 == 0, 1e3, 1.0)),
                ("随机对角", 10.0 ** rng.uniform(-2, 2, n))]

    def test_torque_is_genuinely_unchanged_by_the_rescaling(self):
        """先证明这个变换确实没改物理，否则后面两个测试都不成立。"""
        tau0 = self.W @ self.pi
        for label, d in self._scalings():
            tau = (self.W @ np.diag(1.0 / d)) @ (d * self.pi)
            rel = np.linalg.norm(tau - tau0) / np.linalg.norm(tau0)
            self.assertLess(rel, 1e-12,
                            f"{label} 把力矩改了 {rel:.2e}，这就不是纯换单位")

    def test_rank_is_invariant_so_it_may_be_quoted_as_physical(self):
        ranks = {label: self._rank_and_floor(d)[0]
                 for label, d in self._scalings()}
        self.assertEqual(set(ranks.values()), {62},
                         f"秩在换单位后变了：{ranks}——那它也不能引用了")

    def test_the_floor_percentage_is_NOT_invariant_so_it_may_not(self):
        floors = {label: self._rank_and_floor(d)[1]
                  for label, d in self._scalings()}
        spread = max(floors.values()) - min(floors.values())
        self.assertGreater(
            spread, 0.05,
            f"不可辨识下限在换单位后只动了 {spread*100:.2f}%：{floors}。"
            f"审核的论点是它**会**大幅变动，若真不变则该结论要重写")
        self.assertGreater(max(floors.values()), 0.98,
                           f"质量项换成克之后应逼近 99%，实测 {floors}")


class TestTrajectoryExcitationIsADifferentQuestionFromRank(_SceneFixture):
    """结构秩 ≠ 持续激励。审核要求两者分开报。"""

    def _run(self, amp, freq, dur=6.0):
        import mujoco
        self.sc.set("controller", "mbac")
        self.sc.set("amp", amp)
        self.sc.set("freq", freq)
        self.sc.reset()
        m = self.sc.model
        sub = int(round(self.sc.dt / m.opt.timestep))
        t = 0.0
        tel = {}
        while t < dur:
            q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
            v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
            self.sc.data.ctrl[self.sc.robot.arm_actuator_ids] = \
                self.sc.control(t, q, v)
            for _ in range(sub):
                mujoco.mj_step(m, self.sc.data)
            t += self.sc.dt
            tel = self.sc.telemetry(t, q, v)
        return tel

    def test_both_quantities_are_reported_separately(self):
        keys = {r.key for r in AdaptiveScene.readouts}
        self.assertIn("rank_now", keys, "结构秩")
        self.assertIn("rank_traj", keys, "轨迹激励维数")
        self.assertIn("pe_cond", keys, "激励条件数")

    def test_it_starts_at_zero_instead_of_inventing_a_number(self):
        self.sc.reset()
        q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
        v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
        tel = self.sc.telemetry(0.0, q, v)
        self.assertEqual(tel["rank_traj"], 0.0,
                         "还没跑就报一个维数出来是编的")
        self.assertTrue(np.isnan(tel["pe_cond"]))

    def test_strong_and_weak_excitation_reach_the_same_dimension(self):
        """⭐ 这是本条的要害：弱激励的秩**不比**强激励小。"""
        strong = self._run(0.35, 0.8)
        weak = self._run(0.05, 0.2)
        self.assertEqual(strong["rank_traj"], 62.0)
        self.assertEqual(weak["rank_traj"], 62.0,
                         "若弱激励维数真的更小，那『看维数』就够了，"
                         "也就不需要条件数这一栏——结论要重写")

    def test_but_their_conditioning_differs_by_orders_of_magnitude(self):
        """维数一样、条件数差三个数量级——持续激励说的是后者。"""
        strong = self._run(0.35, 0.8)
        weak = self._run(0.05, 0.2)
        self.assertGreater(
            weak["pe_cond"] - strong["pe_cond"], 2.0,
            f"弱激励 log10(cond)={weak['pe_cond']:.2f} 相对强激励 "
            f"{strong['pe_cond']:.2f} 没拉开，条件数就没有区分力")

    def test_trajectory_rank_never_exceeds_the_structural_rank(self):
        """轨迹是结构的子集，秩不可能更大。用 Gramian 算会算出 66>62，
        那正是 G=WᵀW 把条件数平方的数值假象——所以实现里用增量 QR。"""
        tel = self._run(0.35, 0.8)
        self.assertLessEqual(tel["rank_traj"], float(self.sc.mbac.rank),
                             f"轨迹秩 {tel['rank_traj']} 超过结构秩 "
                             f"{self.sc.mbac.rank}，数值方法有问题")

    def test_a_still_robot_excites_fewer_dimensions_than_the_structure_allows(self):
        """⭐ 这一条给 rank_traj 提供区分力。

        变异测试发现：把 rank_traj 直接抄成结构秩（假装测过了），在
        强/弱激励下都察觉不到——因为两者恰好都是 62。必须找一个
        「轨迹确实张不满」的工况，这一栏才算真的在测轨迹。

        把幅值设成 0：机械臂几乎不动，实测只张开 57 维。
        """
        tel = self._run(0.0, 0.8)
        self.assertLess(
            tel["rank_traj"], float(self.sc.mbac.rank),
            f"机械臂不动时轨迹激励维数仍报 {tel['rank_traj']}，"
            f"等于结构秩 {self.sc.mbac.rank}——这一栏根本没在看轨迹")
        self.assertGreater(tel["rank_traj"], 0.0,
                           "跑了 6 秒还报 0，说明累积没接上")

    def test_projecting_onto_P_does_not_change_the_answer_here(self):
        """记录一个**已证明等价**的实现细节，免得后人以为它可以随便删。

        实现里算的是 svd(R @ P) 而不是 svd(R)。在当前工况下两者的非零
        奇异值集合相同——因为 R 的行空间本来就落在 P 张成的子空间里。
        保留投影是为了让「rank_traj ≤ rank_now」在结构上恒成立，
        而不是靠运气。
        """
        self._run(0.35, 0.8)
        s_proj = np.linalg.svd(self.sc._pe_R @ self.sc.mbac.P,
                               compute_uv=False)
        s_raw = np.linalg.svd(self.sc._pe_R, compute_uv=False)
        n = len(s_proj)
        rel = np.abs(s_proj - s_raw[:n]) / max(s_raw[0], 1e-30)
        self.assertLess(float(rel.max()), 1e-9,
                        "两者奇异值不再相同，那这段注释就要重写")


class TestBaseErrIsNotClaimedToConverge(_SceneFixture):
    """审核要求：不再声称 base_err→0，除非其子空间确实可达且真的降下去。

    实测（amp=0.35, Γ=0.5）：25 s → 82.2%，600 s → 72.7%，并未收敛。
    """

    def test_it_is_still_above_half_after_twenty_five_seconds(self):
        import mujoco
        self.sc.set("controller", "mbac")
        self.sc.set("amp", 0.35)
        self.sc.set("freq", 0.8)
        self.sc.set("gamma", 0.5)
        self.sc.reset()
        m = self.sc.model
        sub = int(round(self.sc.dt / m.opt.timestep))
        t = 0.0
        while t < 25.0:
            q = self.sc.data.qpos[self.sc.robot.qpos_idx].copy()
            v = self.sc.data.qvel[self.sc.robot.qvel_idx].copy()
            self.sc.data.ctrl[self.sc.robot.arm_actuator_ids] = \
                self.sc.control(t, q, v)
            for _ in range(sub):
                mujoco.mj_step(m, self.sc.data)
            t += self.sc.dt
        _, base, _ = self.sc._identifiability_stats(self.sc.mbac.pi_hat)
        self.assertGreater(base, 0.5,
                           f"实测 25 s 后 base_err={base*100:.1f}%。若它真的"
                           f"降到了 50% 以下，讲解里的『不要指望降到 0』"
                           f"就要重写，不能放着不管")

    def test_no_document_claims_it_goes_to_zero(self):
        """把措辞也锁住：这句话被改回去，测试要炸。"""
        blob = next(r.help for r in AdaptiveScene.readouts
                    if r.key == "base_err")
        self.assertIn("不要指望它降到 0", blob,
                      "base_err 的说明必须明确否认『趋于 0』")
        self.assertNotIn("这一栏才应该趋于 0", blob,
                         "旧的过度声称不许回来")

    def test_the_floor_readout_warns_that_it_is_unit_dependent(self):
        """90.21% 是坐标相关的，说明里必须写清楚，否则会被当成物理常数外引。"""
        blob = next(r.help for r in AdaptiveScene.readouts
                    if r.key == "param_floor")
        self.assertIn("不是物理不变量", blob,
                      "param_floor 必须声明自己不是物理不变量")
        self.assertIn("99.10", blob,
                      "要给出换单位后的反例数字，空口说『会变』没有说服力")
        self.assertNotIn("这个百分比是物理不变量", blob)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
