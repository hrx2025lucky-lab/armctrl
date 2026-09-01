"""摩擦补偿的被动性 oracle（第四轮审核 P1-05）。

被审核出来的问题
----------------
文档长期用这条推导证明"补偿是安全的"::

    −f_c·sign(q̇)  +  η·f_c·sign(q̇)  =  (η−1)·f_c·sign(q̇)     η<1 ⇒ 阻碍

⚠️ **它把被控对象的摩擦也写成了 sign。** 实际上：

* 被控对象： ``−f_c·tanh(v/a) − f_v·v``     （平滑，a = 2e-3）
* 控制器：   ``+η·f_c·sign(v)·1{|v|>db}``   （不平滑，db = 1e-3）

在 ``|v|`` 刚出死区处 ``tanh(0.5) = 0.462``，而 ``sign`` 已经是 1。
于是 ``η = 0.8`` 的补偿量 **大于** 真实摩擦 ``0.462 f_c``——净效果是助推。

本文件的 oracle 全部从**能量**出发（功率 = 力矩 × 速度），
不读控制器内部量，也不复用被测函数的中间结果。
"""

import pathlib
import re
import unittest

import numpy as np

from armctrl.tests._docs import requires_docs
from armctrl.control.momentum_observer import (
    friction_power_margin, passivity_deadband, max_passive_ratio)

A = 2e-3        # 被控对象 tanh 尺度
FC, FV, D = 2.0, 0.5, 0.5


def brute_force_worst_power(fc, fv, eta, damping, deadband, a,
                            v_max=0.05, n=400001):
    """独立实现：直接按定义"功率 = (真实摩擦 + 补偿 + 阻尼) × 速度"扫描。

    ⭐ 故意**不复用** :func:`friction_power_margin`，也不化简公式——
    照抄场景里那两行代码的物理含义重写一遍，这样两边算错的方式不会相同。
    """
    # ⚠️ 上限必须超过死区，否则整段扫描都在死区内、补偿恒为 0，会误报"被动"
    v = np.linspace(1e-9, max(v_max, 5.0 * deadband), n)   # 只看 v>0，负半轴对称
    tau_plant = -(fc * np.tanh(v / a) + fv * v)
    tau_comp = eta * (fc * np.sign(v) * (np.abs(v) > deadband) + fv * v)
    tau_damp = -damping * v
    power = (tau_plant + tau_comp + tau_damp) * v
    return float(power.max())


def _clipped_margin(fc, fv, eta, damping, deadband, a, v_max):
    """**故意复刻修复前的行为**：扫描窗口硬截断在 ``v_max``，不做任何撑大。

    只用来当回归见证——证明"低报 5.26 倍"这件事确实发生过，
    也保证以后谁把撑窗口的逻辑删掉，测试会立刻红。
    """
    u = np.linspace(0.0, float(v_max), 400001)
    p = (fc * u * (eta * (u > deadband) - np.tanh(u / a))
         + ((eta - 1.0) * fv - damping) * u ** 2)
    return float(p.max())


class TestTheOldClaimIsFalse(unittest.TestCase):
    """先把"η<1 就安全"这句话证伪，否则后面的修复没有理由。"""

    def test_eta_below_one_still_injects_energy_at_the_deadband_edge(self):
        worst = brute_force_worst_power(FC, FV, 0.8, D, 1e-3, A)
        self.assertGreater(
            worst, 1e-6,
            f"η=0.8（<1）时最坏净功率应为 +6.75e-4 W，实测 {worst:.3e}。"
            f"若它 ≤0，说明被控对象摩擦模型被改过，本用例前提不成立")

    def test_the_injection_band_is_exactly_where_tanh_has_not_saturated(self):
        """理论预测注入区间是 (db, a·atanh(η))，逐点验证。"""
        eta, db = 0.8, 1e-3
        hi = A * np.arctanh(eta)             # = 2.197e-3
        v = np.linspace(1e-9, 0.05, 200001)
        tau = (-(FC * np.tanh(v / A) + FV * v)
               + eta * (FC * (v > db) + FV * v) - D * v)
        pos = v[(tau * v) > 0]
        self.assertGreater(pos.size, 0, "应当存在注入能量的速度区间")
        self.assertGreater(pos.min(), db * 0.999,
                           f"注入区下界 {pos.min():.4e} 应 ≈ 死区 {db:.4e}")
        self.assertLess(pos.max(), hi * 1.02,
                        f"注入区上界 {pos.max():.4e} 应 ≈ a·atanh(η) = {hi:.4e}")

    def test_a_sign_only_plant_would_indeed_be_safe(self):
        """反向确认：如果被控对象真是 sign 型，旧推导就是对的。

        这一条说明旧文档**不是算错了**，而是**模型写错了**——
        分清这两件事，结论才站得住。
        """
        v = np.linspace(1e-9, 0.05, 200001)
        tau = -(FC * np.sign(v) + FV * v) + 0.8 * (FC * np.sign(v) + FV * v) - D * v
        self.assertLessEqual((tau * v).max(), 0.0,
                             "sign 型被控对象下 η<1 应当处处耗散")


class TestPassivityCondition(unittest.TestCase):

    def test_max_ratio_matches_an_independent_bisection(self):
        """η_max = tanh(db/a) 这个闭式解，用二分法独立求一遍对照。"""
        for db in (5e-4, 1e-3, 2e-3, 4e-3):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if brute_force_worst_power(FC, FV, mid, D, db, A) <= 0.0:
                    lo = mid
                else:
                    hi = mid
            closed = max_passive_ratio(db, A)
            self.assertAlmostEqual(
                lo, closed, delta=2e-3,
                msg=f"db={db}: 二分得 η_max={lo:.5f}，闭式给 {closed:.5f}")

    def test_required_deadband_matches_an_independent_bisection(self):
        for eta in (0.3, 0.5, 0.8, 0.95):
            lo, hi = 0.0, 0.05
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if brute_force_worst_power(FC, FV, eta, D, mid, A) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            closed = passivity_deadband(eta, A)
            self.assertAlmostEqual(
                hi, closed, delta=2e-5,
                msg=f"η={eta}: 二分得 db={hi:.6f}，闭式给 {closed:.6f}")

    def test_full_compensation_is_hopeless_only_when_damping_is_zero(self):
        """η ≥ 1 无解，**只在 D = 0（两参数形式）时成立**。

        ⚠️ 这里原来的标题是「任何死区都救不了」——那是错的，
        反例见 :class:`TestDampingRescuesEtaAboveOne`。
        """
        self.assertEqual(passivity_deadband(1.0, A), float("inf"))
        self.assertEqual(passivity_deadband(1.3, A), float("inf"))
        for db in (1e-3, 1e-2, 1e-1):
            self.assertGreater(
                brute_force_worst_power(FC, FV, 1.2, D, db, A,
                                        v_max=5.0 * max(db, 0.05)), 0.0,
                f"η=1.2 配这么小的死区（{db}）不可能被动——判据写反了")

    def test_the_default_limit_is_much_stricter_than_one(self):
        """⭐ 结论本身就是教学点：0.4621，不是 1。"""
        self.assertAlmostEqual(max_passive_ratio(1e-3, A), 0.46212, places=5)

    def test_library_and_brute_force_agree_everywhere(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            fc = rng.uniform(0.5, 6.0)
            fv = rng.uniform(0.0, 3.0)
            eta = rng.uniform(0.05, 1.4)
            d = rng.uniform(0.0, 8.0)
            db = rng.uniform(2e-4, 5e-3)
            c = d - (eta - 1.0) * fv
            if c <= 0.0:
                continue                      # 发散配置，见 TestPowerMarginScansFarEnough
            a = friction_power_margin(fc, fv, eta, d, db, A)[0]
            # ⚠️ 暴力实现也必须扫到同一个可证明足够大的上限，
            # 否则它自己会截断，"两边不一致"就成了假警报。
            b = brute_force_worst_power(fc, fv, eta, d, db, A,
                                        v_max=1.2 * fc * eta / c, n=1500001)
            self.assertAlmostEqual(a, b, delta=1e-5 + 1e-3 * abs(b),
                                   msg=f"fc={fc:.3f} η={eta:.3f} db={db:.5f}")


class TestTeachingSceneEnforcesIt(unittest.TestCase):
    """场景级：打开保护后余量必须 ≤ 0，关掉后必须暴露出正数。"""

    @classmethod
    def setUpClass(cls):
        from armctrl.tuner.scenes import TeachingScene
        cls.sc = TeachingScene()
        cls.sc.build()
        cls.sc.reset()

    def _stats(self, passive, eta=None, damping=D):
        # ⚠️ 显式写回 fc/fv/D：紧界依赖它们四个，
        # 上一条用例改过阻尼的话，这里必须复位，否则用例之间互相串味。
        self.sc.set("fc", FC)
        self.sc.set("fv", FV)
        self.sc.set("damping", damping)
        if eta is not None:
            self.sc.set("comp_ratio", eta)
        self.sc.set("passive", passive)
        return self.sc._passivity_stats()

    def test_protection_on_makes_the_margin_non_positive(self):
        for eta in (0.2, 0.5, 0.8, 0.95):
            pw, emax, db = self._stats("on", eta)
            self.assertLessEqual(pw, 1e-9,
                                 f"η={eta} 打开保护后余量仍为 {pw:+.5f} mW")
            self.assertGreaterEqual(emax + 1e-9, eta,
                                    f"η={eta} 但允许上限只有 {emax:.4f}")

    def test_protection_off_reproduces_the_original_defect(self):
        pw, emax, db = self._stats("off", 0.8)
        self.assertAlmostEqual(pw, 0.67514, places=4,
                               msg=f"关闭保护、η=0.8 时余量应为 +0.67514 mW，"
                                   f"实测 {pw:+.5f}")
        # ⭐ 场景报的是**含 f_v、D 的紧界** 0.4625015，
        # 不是只看 tanh 的 0.4621172。差 4e-4，方向上后者偏保守。
        self.assertAlmostEqual(emax, 0.4625015, places=6)
        self.assertNotAlmostEqual(emax, 0.4621172, places=6,
                                  msg="场景又退回只看 tanh 的保守界了")
        self.assertAlmostEqual(db, 1.0, places=6)

    def test_protection_widens_the_deadband_to_the_derived_value(self):
        _, _, db = self._stats("on", 0.8)
        want = passivity_deadband(0.8, A, fc=FC, fv=FV, damping=D) * 1e3
        self.assertAlmostEqual(db, want, places=6,
                               msg=f"死区应被撑到紧界 {want:.4f} mrad/s")
        loose = passivity_deadband(0.8, A) * 1e3            # = 2.197
        self.assertLess(db, loose,
                        f"紧界 {db:.4f} 必须**小于**只看 tanh 的 {loose:.4f}——"
                        f"f_v 和 D 是白送的耗散，把它们算进来只会更省死区")

    def test_full_compensation_is_reported_as_unsafe_even_with_protection_on(self):
        """⭐ 保护不是万能的：η 拉到 1.0 以上时它必须**如实报警**，
        而不是悄悄把 η 改小假装安全。"""
        self.sc.set("comp_ratio", 1.2)
        self.sc.set("passive", "on")
        pw, emax, db = self.sc._passivity_stats()
        self.assertGreater(pw, 0.0,
                           f"η=1.2 时余量应 > 0，实测 {pw:+.5f} mW")
        self.assertLess(emax, 1.0)
        self.sc.set("comp_ratio", 0.8)

    def test_high_damping_lets_the_scene_accept_eta_above_one(self):
        """⭐⭐ 这条就是旧结论被推翻的现场：同样 η=1.2，
        阻尼从 0.5 拉到 8.0，场景从**报警**变成**被动**。"""
        self.sc.set("passive", "on")
        self.sc.set("comp_ratio", 1.2)

        self.sc.set("damping", 0.5)
        weak, _, db_weak = self.sc._passivity_stats()
        self.assertGreater(weak, 0.0,
                           f"η=1.2、D=0.5 需要 1.0 rad/s 死区（超过上限 "
                           f"{type(self.sc).DEADBAND_CAP}），必须如实报警，"
                           f"实测 {weak:+.5f} mW")

        self.sc.set("damping", 8.0)
        strong, emax, db_strong = self.sc._passivity_stats()
        self.assertLessEqual(strong, 1e-9,
                             f"η=1.2、D=8.0 时理论上 50.6 mrad/s 死区就够，"
                             f"场景应判为被动，实测 {strong:+.5f} mW")
        self.assertGreaterEqual(emax + 1e-9, 1.2,
                                f"允许上限应 ≥ 1.2，实测 {emax:.4f}")
        self.assertGreater(db_strong, db_weak,
                           "被救回来的代价就是死区变大，它必须真的变大")
        self.assertLess(db_strong, type(self.sc).DEADBAND_CAP * 1e3,
                        "死区不能超过可用上限，否则等于没补偿")

        self.sc.set("damping", D)
        self.sc.set("comp_ratio", 0.8)

    def test_the_scene_never_silently_widens_the_deadband_past_the_cap(self):
        """代价太大时**不能偷偷放大死区假装安全**，必须保持 base 并报警。"""
        self.sc.set("passive", "on")
        for eta, dmp in ((1.2, 0.5), (1.9, 8.0), (2.5, 8.0)):
            self.sc.set("comp_ratio", eta)
            self.sc.set("damping", dmp)
            pw, _, db = self.sc._passivity_stats()
            self.assertLessEqual(db, type(self.sc).DEADBAND_CAP * 1e3 + 1e-9,
                                 f"η={eta} D={dmp}: 死区 {db:.1f} 超过上限")
            self.assertGreater(pw, 0.0,
                               f"η={eta} D={dmp}: 没保护到就必须报警，"
                               f"实测 {pw:+.5f} mW")
        self.sc.set("damping", D)
        self.sc.set("comp_ratio", 0.8)

    def test_changing_any_of_the_four_knobs_recomputes_the_deadband(self):
        """⭐ 紧界依赖 η、f_c、f_v、D 四个量，改**任何一个**都必须重算。

        ⚠️ 旧代码只挂在 η 上，于是"调大阻尼救回 η>1"在界面上调了没反应。
        """
        base = dict(comp_ratio=0.8, fc=FC, fv=FV, damping=D)
        for k, v in (("comp_ratio", 0.95), ("fc", 5.0),
                     ("fv", 2.5), ("damping", 6.0)):
            for bk, bv in base.items():
                self.sc.set(bk, bv)
            self.sc.set("passive", "on")
            before = self.sc.teach.vel_deadband
            self.sc.set(k, v)
            after = self.sc.teach.vel_deadband
            self.assertNotAlmostEqual(
                before, after, places=9,
                msg=f"改 {k}: {bv} -> {v} 后死区仍是 {after:.6f}，没重算")
        for bk, bv in base.items():
            self.sc.set(bk, bv)

    def test_the_four_knobs_actually_reach_the_controller(self):
        """反向护栏：参数要真的写进控制器，不能只改了界面上的数。

        ⚠️ 构造参数名 (``friction_coulomb``) 和属性名 (``fc``) 不一样，
        写错了 Python **不会报错**，只会新建一个没人读的属性。
        """
        self.sc.set("fc", 4.0)
        self.sc.set("fv", 1.5)
        self.sc.set("comp_ratio", 0.6)
        self.sc.set("damping", 3.0)
        self.assertAlmostEqual(float(np.max(self.sc.teach.fc)), 4.0, places=9,
                               msg="f_c 没写进控制器（属性名写错了？）")
        self.assertAlmostEqual(float(np.max(self.sc.teach.fv)), 1.5, places=9,
                               msg="f_v 没写进控制器（属性名写错了？）")
        self.assertAlmostEqual(self.sc.teach.comp_ratio, 0.6, places=9)
        self.assertAlmostEqual(self.sc.teach.damping, 3.0, places=9)
        for k, v in (("fc", FC), ("fv", FV),
                     ("comp_ratio", 0.8), ("damping", D)):
            self.sc.set(k, v)

    def test_scene_and_plant_share_the_same_tanh_scale(self):
        """判据必须引用**被控对象实际用的** a，不能各写各的。"""
        from armctrl.tuner.scenes import TeachingScene
        v = np.array([1.0e-3])
        self.sc.set("fc", 2.0)
        self.sc.set("fv", 0.0)
        got = -self.sc._true_friction(v)[0]
        want = 2.0 * np.tanh(1.0e-3 / TeachingScene.TANH_SCALE)
        self.assertAlmostEqual(got, want, places=12,
                               msg="_true_friction 用的 a 和 TANH_SCALE 不一致")
        self.sc.set("fv", 0.5)



class TestTheBoundIsTighterThanTanh(unittest.TestCase):
    """P1-05 ①：``tanh(db/a)`` 不是真界，只是一个**偏保守**的下界。"""

    @staticmethod
    def _bisect_bound(fc, fv, damping, db, a):
        """独立二分：只用 :func:`brute_force_worst_power` 的**符号**找 η 上界。

        ⭐ 全程不碰 ``max_passive_ratio`` 的公式，两边不可能一起错。
        """
        lo, hi = 0.0, 5.0
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            w = brute_force_worst_power(fc, fv, mid, damping, db, a,
                                        v_max=max(1.0, 20.0 * db), n=800001)
            if w <= 0.0:
                lo = mid
            else:
                hi = mid
        return lo

    def test_the_default_tight_bound_is_0_4625015_not_0_4621172(self):
        """⭐ 审核报告的核心数字，用闭式 + 二分双重确认。"""
        tight = max_passive_ratio(1e-3, A, fc=FC, fv=FV, damping=D)
        loose = max_passive_ratio(1e-3, A)
        # 闭式：下确界落在死区边缘 u = db+
        closed = (FC * np.tanh(1e-3 / A) + (FV + D) * 1e-3) / (FC + FV * 1e-3)
        self.assertAlmostEqual(tight, closed, places=9)
        self.assertAlmostEqual(tight, 0.4625015, places=6,
                               msg=f"真界应为 0.4625015，实测 {tight:.7f}")
        self.assertAlmostEqual(loose, 0.4621172, places=6,
                               msg="两参数形式必须保持 tanh(db/a) 的老行为")
        self.assertGreater(tight, loose, "真界必须比保守界大")
        self.assertAlmostEqual(self._bisect_bound(FC, FV, D, 1e-3, A), tight,
                               delta=2e-4, msg="独立二分不同意这个界")

    def test_the_two_argument_form_is_conservative_never_optimistic(self):
        """⭐ 关键安全性质：老公式可以偏小，**绝不能偏大**。

        偏小 = 少补一点摩擦，人推起来费劲，但不会自己跑；
        偏大 = 以为安全其实在注能，是会伤人的方向。
        """
        rng = np.random.default_rng(11)
        for _ in range(40):
            fc = rng.uniform(0.5, 6.0)
            fv = rng.uniform(0.0, 3.0)
            dmp = rng.uniform(0.0, 8.0)
            db = rng.uniform(2e-4, 5e-2)
            tight = max_passive_ratio(db, A, fc=fc, fv=fv, damping=dmp)
            loose = max_passive_ratio(db, A)
            self.assertLessEqual(
                loose, tight + 1e-9,
                f"fc={fc:.3f} fv={fv:.3f} D={dmp:.3f} db={db:.5f}: "
                f"保守界 {loose:.6f} 竟然大于真界 {tight:.6f}——方向错了")

    def test_the_tight_bound_really_is_the_boundary(self):
        """紧界必须真的"紧"：略低于它被动，略高于它注能。"""
        for fc, fv, dmp, db in ((FC, FV, D, 1e-3), (3.0, 1.0, 2.0, 5e-3),
                                (1.0, 0.2, 0.0, 2e-3)):
            b = max_passive_ratio(db, A, fc=fc, fv=fv, damping=dmp)
            vm = max(1.0, 20.0 * db)
            below = brute_force_worst_power(fc, fv, b * 0.99, dmp, db, A,
                                            v_max=vm, n=800001)
            above = brute_force_worst_power(fc, fv, b * 1.01, dmp, db, A,
                                            v_max=vm, n=800001)
            self.assertLessEqual(below, 1e-9,
                                 f"η=0.99·界 竟然注能 {below:+.3e}")
            self.assertGreater(above, 0.0,
                               f"η=1.01·界 竟然还被动——界给松了")


class TestDampingRescuesEtaAboveOne(unittest.TestCase):
    """P1-05 ②：推翻「η ≥ 1 时任何死区都救不了」。"""

    #: (η, D, 独立 brentq 求出的最小死区 rad/s)。三条都满足 η >= 1。
    CASES = ((1.0, 0.5, 0.0070353671),
             (1.2, 8.0, 0.0506329114),
             (1.2, 0.5, 1.0000000000))

    def test_the_three_counterexamples_really_are_passive(self):
        """⭐⭐ 三个反例，用**暴力扫描**（不是判据本身）确认它们真的被动。"""
        for eta, dmp, db in self.CASES:
            w = brute_force_worst_power(FC, FV, eta, dmp, db, A,
                                        v_max=max(5.0, 20.0 * db), n=2000001)
            self.assertLessEqual(
                w, 1e-9,
                f"η={eta} D={dmp} db={db}: 旧结论说这不可能被动，"
                f"但暴力扫描给出最坏净功率 {w:+.3e} W")

    def test_the_library_reproduces_those_deadbands(self):
        for eta, dmp, want in self.CASES:
            got = passivity_deadband(eta, A, fc=FC, fv=FV, damping=dmp)
            self.assertTrue(np.isfinite(got),
                            f"η={eta} D={dmp} 应有有限解，却返回 {got}")
            self.assertAlmostEqual(got, want, delta=1e-6 * max(want, 1e-2),
                                   msg=f"η={eta} D={dmp}: 得 {got:.8f}，"
                                       f"独立 brentq 给 {want:.8f}")

    def test_slightly_smaller_deadbands_do_inject(self):
        """反向护栏：给的死区必须是**最小**的，再小一点就该注能。"""
        for eta, dmp, db in self.CASES:
            w = brute_force_worst_power(FC, FV, eta, dmp, db * 0.9, A,
                                        v_max=max(5.0, 20.0 * db), n=2000001)
            self.assertGreater(w, 0.0,
                               f"η={eta} D={dmp} db={db * 0.9:.4f}（缩小 10%）"
                               f"竟然还被动，说明返回的死区偏大了")

    def test_the_ceiling_is_one_plus_damping_over_viscous(self):
        """天花板 ``sup_u g(u) = 1 + D/f_v``：越过它才真的无解。"""
        for fv, dmp in ((0.5, 0.5), (0.5, 8.0), (1.0, 0.2), (2.0, 6.0)):
            cap = 1.0 + dmp / fv
            self.assertEqual(
                passivity_deadband(cap * 1.02, A, fc=FC, fv=fv, damping=dmp),
                float("inf"),
                f"fv={fv} D={dmp}: η 越过天花板 {cap:.3f} 必须返回 inf")
            got = passivity_deadband(cap * 0.9, A, fc=FC, fv=fv, damping=dmp)
            self.assertTrue(np.isfinite(got),
                            f"fv={fv} D={dmp}: η 在天花板 {cap:.3f} 之下"
                            f"应当有解，却返回 {got}")

    def test_zero_damping_restores_the_old_hopeless_verdict(self):
        """D = 0 时旧结论是**对的**——修复不能把这个正确的部分也改掉。"""
        for eta in (1.0, 1.3, 2.0):
            self.assertEqual(
                passivity_deadband(eta, A, fc=FC, fv=FV, damping=0.0),
                float("inf"),
                f"D=0、η={eta} 时确实无解，必须返回 inf")


class TestPowerMarginScansFarEnough(unittest.TestCase):
    """P1-05 ③：``friction_power_margin`` 的扫描窗口曾经太短。"""

    def test_the_old_default_window_under_reported_by_five_times(self):
        """⭐ 回归见证：审核报告说低报 5.26 倍，这里把它钉死。"""
        clipped = _clipped_margin(FC, FV, 1.2, D, 1e-3, A, v_max=0.05)
        true_peak, true_v = friction_power_margin(FC, FV, 1.2, D, 1e-3, A)
        self.assertAlmostEqual(clipped, 0.019, places=3,
                               msg=f"旧窗口应报 0.019 W，实测 {clipped:.4f}")
        self.assertAlmostEqual(true_peak, 0.100, places=3,
                               msg=f"真峰应为 0.100 W，实测 {true_peak:.4f}")
        self.assertAlmostEqual(true_v, 0.5, places=2)
        self.assertGreater(true_peak / clipped, 5.0,
                           "低报倍数应 > 5，否则这条回归见证失效了")

    def test_the_true_peak_matches_a_closed_form(self):
        """tanh 饱和后 ``P(u) = f_c(η−1)u − c·u²``，峰在 ``u = f_c(η−1)/(2c)``。

        默认参数：``c = D − (η−1)f_v = 0.4``，``u* = 2×0.2/0.8 = 0.5``，
        ``P* = 2×0.2×0.5 − 0.4×0.25 = 0.1 W``。⭐ 和实测一模一样。
        """
        eta = 1.2
        c = D - (eta - 1.0) * FV
        u_star = FC * (eta - 1.0) / (2.0 * c)
        p_star = FC * (eta - 1.0) * u_star - c * u_star ** 2
        got_p, got_u = friction_power_margin(FC, FV, eta, D, 1e-3, A)
        self.assertAlmostEqual(got_u, u_star, places=3)
        self.assertAlmostEqual(got_p, p_star, places=4)

    def test_unbounded_configs_are_reported_as_infinite(self):
        """``c = D − (η−1)f_v <= 0`` ⇒ 功率随速度发散，必须返回 inf。

        ⚠️ 旧版本会闷头扫到 0.05 然后报一个**有限的小数字**，
        等于把发散当成了合格——这是最危险的一类错。
        """
        for eta, dmp in ((1.2, 0.0), (2.0, 0.4), (1.5, 0.25)):
            self.assertLessEqual(dmp - (eta - 1.0) * FV, 0.0, "用例前提写错了")
            w, v = friction_power_margin(FC, FV, eta, dmp, 1e-3, A)
            self.assertEqual(w, float("inf"),
                             f"η={eta} D={dmp} 功率发散，却报了 {w}")
            self.assertEqual(v, float("inf"))
            far = _clipped_margin(FC, FV, eta, dmp, 1e-3, A, v_max=50.0)
            near = _clipped_margin(FC, FV, eta, dmp, 1e-3, A, v_max=5.0)
            self.assertGreater(far, near * 5.0,
                               f"η={eta} D={dmp}: 扫得越远功率应越大（发散），"
                               f"5→50 只从 {near:.2f} 涨到 {far:.2f}")

    def test_the_window_contains_the_peak_for_random_configs(self):
        """随机配置：把窗口再扩大 20 倍，峰值不应该变。"""
        rng = np.random.default_rng(5)
        checked = 0
        for _ in range(60):
            fc = rng.uniform(0.5, 6.0)
            fv = rng.uniform(0.05, 3.0)
            eta = rng.uniform(0.05, 1.4)
            dmp = rng.uniform(0.0, 8.0)
            db = rng.uniform(2e-4, 5e-3)
            if dmp - (eta - 1.0) * fv <= 0.0:
                continue
            checked += 1
            got = friction_power_margin(fc, fv, eta, dmp, db, A)[0]
            wide = brute_force_worst_power(fc, fv, eta, dmp, db, A,
                                           v_max=20.0 * fc * eta
                                           / (dmp - (eta - 1.0) * fv),
                                           n=1500001)
            self.assertAlmostEqual(
                got, wide, delta=1e-5 + 2e-3 * abs(wide),
                msg=f"fc={fc:.3f} fv={fv:.3f} η={eta:.3f} D={dmp:.3f}: "
                    f"窗口内 {got:.6f}，扩大 20 倍后 {wide:.6f}")
        self.assertGreater(checked, 25, "有效随机用例太少，这条测试没覆盖到")

    def test_deadband_larger_than_v_max_still_sees_the_compensation(self):
        """老坑回归：死区比 v_max 大时不能整段落在死区内误报"被动"。

        取 η=1.2、D=0.5、db=0.06 > v_max=0.05：真实峰在 u*=0.5，
        完全落在旧窗口之外。旧实现整段扫描都在死区**内**（补偿恒为 0），
        会报 0.0 也就是"被动"——⚠️ **最危险的一种错**。
        """
        w, v = friction_power_margin(FC, FV, 1.2, D, 0.06, A)
        self.assertGreater(w, 0.0,
                           f"db=0.06 > v_max=0.05，补偿区间被漏掉了，"
                           f"报了 {w:+.4e}")
        self.assertAlmostEqual(w, 0.1, places=3)
        self.assertAlmostEqual(v, 0.5, places=2)
        blind = _clipped_margin(FC, FV, 1.2, D, 0.06, A, v_max=0.05)
        self.assertAlmostEqual(blind, 0.0, places=12,
                               msg="旧窗口本该完全瞎掉（恒 0），这条见证失效了")


DOC = pathlib.Path(__file__).resolve().parents[2] / "armctrl_讲解" / "10_拖动示教.md"


def _num(cell):
    """把表格单元里的数字抠出来，兼容全角减号和 **加粗**。"""
    t = cell.replace("−", "-").replace("*", "").replace("`", "")
    t = t.replace("\\times10^{", "e").replace("}", "").replace("$", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", t)
    return float(m.group()) if m else None


def _rows(text, start, stop):
    """取 [start, stop) 之间的表格行，返回去掉首尾空列的单元列表。"""
    seg = text[text.index(start):text.index(stop)]
    out = []
    for line in seg.splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        out.append([c.strip() for c in line.strip("|").split("|")])
    return out[1:]                      # 丢掉表头


@requires_docs
class TestTheTeachingDocMatchesTheCode(unittest.TestCase):
    """⭐ 文档里的每个数字都必须能被重新算出来。

    ⚠️ 这是 P1-05 最根本的一条修复：旧文档的 0.4621 / 2.197 / 1.90e-2
    没有任何测试解析过，所以公式改了、数字过期了，**没人会知道**。
    """

    @classmethod
    def setUpClass(cls):
        cls.d = DOC.read_text(encoding="utf-8")

    def test_the_eta_table_reproduces(self):
        rows = _rows(self.d, "### 4.2.3", "**注入能量的速度区间**")
        checked = 0
        for r in rows:
            eta, want = _num(r[0]), _num(r[1])
            if eta is None or want is None:
                continue
            got = friction_power_margin(FC, FV, eta, D, 1e-3, A)[0]
            self.assertAlmostEqual(
                got, want, delta=6e-3 * max(abs(want), 1e-5) + 1e-9,
                msg=f"文档 4.2.3 说 η={eta} 时最坏净功率 {want:.4e}，"
                    f"重算得 {got:.4e}")
            checked += 1
        self.assertGreaterEqual(checked, 6, "4.2.3 的 η 表没解析到，路径变了？")

    def test_the_deadband_table_reproduces(self):
        rows = _rows(self.d, "### 4.2.4", "代码里加了一个开关")
        checked = 0
        for r in rows:
            db, want = _num(r[0]), _num(r[1])
            if db is None or want is None:
                continue
            got = friction_power_margin(FC, FV, 0.8, D, db, A)[0]
            self.assertAlmostEqual(
                got, want, delta=6e-3 * max(abs(want), 1e-5) + 1e-9,
                msg=f"文档 4.2.4 说 db={db} 时余量 {want:.4e}，重算得 {got:.4e}")
            checked += 1
        self.assertGreaterEqual(checked, 5, "4.2.4 的死区表没解析到")

    def test_the_doc_quotes_the_tight_bound_not_the_loose_one(self):
        self.assertIn("0.4625015", self.d,
                      "文档必须写出真界 0.4625015")
        self.assertIn("2.193574", self.d,
                      "文档必须写出真的最小死区 2.193574e-3")
        head = self.d[:self.d.index("### 4.2.5")]
        self.assertNotIn("上限是 0.4621", head,
                         "标题里还在把保守界当真界")
        self.assertRegex(
            self.d, r"0\.4621172[^\n]*|[^\n]*0\.4621172",
            "保守界 0.4621172 也要留在文档里，用来对照")

    def test_the_counterexample_table_reproduces(self):
        """⭐⭐ 推翻「η≥1 没救」的那张表，逐行用暴力扫描复核。"""
        rows = _rows(self.d, "三个反例", "**直觉是什么？**")
        checked = 0
        for r in rows:
            eta, dmp, cap = _num(r[0]), _num(r[1]), _num(r[2])
            if eta is None or dmp is None or cap is None:
                continue
            self.assertAlmostEqual(cap, 1.0 + dmp / FV, places=6,
                                   msg=f"η={eta} D={dmp}: 天花板写成 {cap}")
            db = _num(r[3])
            if db is None:                       # 「—」= 真的没救
                self.assertEqual(
                    passivity_deadband(eta, A, fc=FC, fv=FV, damping=dmp),
                    float("inf"),
                    f"η={eta} D={dmp} 文档说没救，代码却给了有限死区")
                checked += 1
                continue
            got = passivity_deadband(eta, A, fc=FC, fv=FV, damping=dmp) * 1e3
            self.assertAlmostEqual(
                got, db, delta=2e-3 * max(db, 1.0),
                msg=f"η={eta} D={dmp}: 文档说要 {db} mrad/s，重算得 {got:.4f}")
            # ⚠️ 用**精确值**做被动性复核，不能用文档里那个四舍五入的数：
            # 50.6 比真值 50.6329 小一丝，就会漏出 +1.3e-5 W。
            # 文档显示几位是排版问题，判据成不成立是物理问题，别混。
            worst = brute_force_worst_power(FC, FV, eta, dmp, got * 1e-3, A,
                                            v_max=max(5.0, 0.1 * db),
                                            n=2000001)
            self.assertLessEqual(
                worst, 1e-8,
                f"η={eta} D={dmp} db={db} mrad/s 文档说被动，"
                f"暴力扫描却给 {worst:+.3e} W")
            checked += 1
        self.assertGreaterEqual(checked, 4, "4.2.5 的反例表没解析到")

    def test_the_doc_critical_eta_really_is_critical(self):
        """⭐ 光验证"这一行功率是 0"是不够的——**保守值也是 0**。

        ⚠️ 变异测试抓到的洞：把文档里的临界 η 从 0.4625015 改回
        0.4621172，两边算出来都是 0 W，测试全绿。
        真正能区分的问题是：**再大一丁点会不会立刻注能？**
        """
        rows = _rows(self.d, "### 4.2.3", "**注入能量的速度区间**")
        crit = [_num(r[0]) for r in rows if "临界" in r[1]]
        self.assertEqual(len(crit), 1, "4.2.3 应当恰好标注一个临界 η")
        eta = crit[0]
        self.assertLessEqual(friction_power_margin(FC, FV, eta, D, 1e-3, A)[0],
                             1e-12, f"文档标的临界 η={eta} 本身就在注能")
        above = friction_power_margin(FC, FV, eta * 1.0002, D, 1e-3, A)[0]
        self.assertGreater(
            above, 0.0,
            f"把临界 η={eta} 放大 0.02 % 后仍然被动（{above:+.3e} W），"
            f"说明文档标的不是**真**临界，而是一个偏保守的值")

    def test_the_doc_critical_deadband_really_is_minimal(self):
        """同上，反向：临界死区**再小一丁点就该注能**。"""
        rows = _rows(self.d, "### 4.2.4", "代码里加了一个开关")
        crit = [_num(r[0]) for r in rows
                if r[1].replace("*", "").strip().startswith("0")
                and "✅" in r[2] and "**" in r[0]]
        self.assertEqual(len(crit), 1, "4.2.4 应当恰好加粗一个临界死区")
        db = crit[0]
        self.assertLessEqual(friction_power_margin(FC, FV, 0.8, D, db, A)[0],
                             1e-12, f"文档标的临界死区 {db} 本身就在注能")
        below = friction_power_margin(FC, FV, 0.8, D, db * 0.999, A)[0]
        self.assertGreater(
            below, 0.0,
            f"把临界死区 {db} 缩小 0.1 % 后仍然被动（{below:+.3e} W），"
            f"说明文档标的不是**最小**死区，只是某个够大的值")

    def test_the_doc_keeps_the_old_formula_as_a_contrast(self):
        """⭐ 旧公式那一行不能删——"新旧差多少"本身就是教学内容。"""
        seg = self.d[self.d.index("### 4.2.4"):self.d.index("代码里加了一个开关")]
        self.assertIn("旧公式给的", seg,
                      "死区表必须保留「旧公式给的」对照行，否则读者看不出差别")
        self.assertIn("2.197", seg, "对照行应写出旧公式的 2.197e-3")
        rows = _rows(self.d, "### 4.2.4", "代码里加了一个开关")
        self.assertGreaterEqual(len([r for r in rows if _num(r[0])]), 6,
                                "死区表少了行，对照被删掉了？")

    def test_the_doc_retracted_the_hopeless_claim(self):
        seg = self.d[self.d.index("### 4.2.5"):self.d.index("### 4.2.6")]
        for marker in ("这句话也是错的", "只在 $D=0$ 时成立",
                       "1+\\frac{D}{f_v}", "DEADBAND_CAP"):
            self.assertIn(marker, seg, f"4.2.5 缺少撤回标记「{marker}」")
        self.assertNotIn("**任何死区都救不回来**。因为此时补偿在", seg,
                         "旧的错误说法还留在文档里")

    def test_the_criterion_section_shows_the_full_g_function(self):
        seg = self.d[self.d.index("### 4.2.2"):self.d.index("### 4.2.3")]
        self.assertIn("f_c\\tanh(u/a)+(f_v+D)", seg.replace(" ", ""),
                      "4.2.2 必须写出完整的 g(u)，含 f_v 和 D")
        self.assertIn("第五轮审核（P1-05）", seg, "缺少撤回出处")


if __name__ == "__main__":     # pragma: no cover
    unittest.main()
