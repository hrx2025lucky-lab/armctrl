"""SO(3) 连续 lift、π 切割面、任务度量 W、pinv 硬截断、阻尼标定与零空间泄漏的测试。

测试红线（第二/三轮审查）
------------------------
1. 可被 unittest discover 自动发现。
2. oracle 独立：不用 exp∘log ≈ R 作为 log 正确性的唯一判据，而是用独立构造的
   已知 (轴, 角) 反查；分支/连续性的判据不读取被测对象的内部中间量。
3. 真回归：必须能在**真实旧代码**（/tmp/armctrl_baseline，通过 importlib 以
   独立命名空间加载）上失败，而不是"用当前代码的开关模拟旧代码"。
4. mutation：关掉被测机制后，对应用例必须失败；且**不能**被限速或力矩饱和救活。
5. 不通过放宽容差、降低打印精度、提高增益、扩大正则、只看中位数或忽略失败样本过关。
6. 力矩类用例必须同时检查 **raw wrench / raw torque / saturated torque**。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest

import numpy as np

from armctrl.core.robot import make_panda
from armctrl.core.kinematics import (
    POSE_TASK,
    POSITION_TASK,
    Q_HOME_PANDA,
    DLSInverseKinematics,
    OrientationErrorTracker,
    so3_equivalent_branch,
    so3_lift_candidates,
    so3_log,
    so3_select_branch,
)
from armctrl.control.impedance import (
    CartesianImpedanceController,
    HybridForcePositionController,
    ImpedanceGains,
)


Q_HOME = Q_HOME_PANDA
TWO_PI = 2.0 * np.pi

# 整个测试文件共用一个 Panda（构造 MuJoCo 模型较贵），但**每个用例自己造控制器
# 和 IK 实例**，避免跨用例状态污染。
_ROBOT = None


def panda():
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = make_panda()
    return _ROBOT


# ------------------------------------------------------------------ oracle


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rot_from_axis_angle(axis, theta):
    """独立 oracle：由已知轴角构造 R（Rodrigues）。"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = skew(a)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def expm_so3(w):
    """独立指数映射，用于校验"lift 是否精确表示同一个旋转"。"""
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3)
    return rot_from_axis_angle(w / th, th)


def rot_pi_symmetric(axis):
    """θ = π 的**真正对称**旋转矩阵 R = 2aaᵀ − I（反对称部分严格为零）。

    与 Rodrigues(a, np.pi) 不同：后者因 sin(np.pi) ≈ 1.22e-16 残留反对称项。
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return 2.0 * np.outer(a, a) - np.eye(3)


def cut_pair(axis, comp, eps=1e-13):
    """构造切割面两侧的一对 θ=π 目标姿态。

    axis 落在某个 tie 面上（两个分量模相等、符号相反，或体对角线三个分量模相等），
    沿分量 comp 做 ±eps 微扰使 argmax(diag(aaᵀ)) 切换，主值 log 整体变号。
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    d = np.zeros(3)
    d[comp] = eps
    return rot_pi_symmetric(a + d), rot_pi_symmetric(a - d)


_C = np.sqrt(0.5)
_T = 1.0 / np.sqrt(3.0)

#: 三个异号 tie 面 + 三个混合符号体对角线。**不是只围绕一个 [c,−c,0] 设计。**
CUT_CASES = [
    ("tie-xy", [_C, -_C, 0.0], 0),
    ("tie-yz", [0.0, _C, -_C], 1),
    ("tie-xz", [_C, 0.0, -_C], 0),
    ("diag+-+", [_T, -_T, _T], 0),
    ("diag-++", [-_T, _T, _T], 1),
    ("diag++-", [_T, _T, -_T], 2),
]

Q_INITS = [
    Q_HOME.copy(),
    np.array([0.4, -0.6, 0.3, -1.8, 0.2, 1.6, 0.5]),
    np.array([-0.5, 0.4, -0.7, -2.4, 0.6, 2.4, -1.0]),
]


# ------------------------------------------------------------------ so3_log


class TestSO3Log(unittest.TestCase):
    """so3_log 的正确性：用独立构造的 (轴, 角) 反查。"""

    def test_identity(self):
        self.assertLess(float(np.linalg.norm(so3_log(np.eye(3)))), 1e-12)

    def test_known_axis_angle(self):
        rng = np.random.default_rng(0)
        worst_ang, worst_axis = 0.0, 0.0
        for _ in range(3000):
            a = rng.standard_normal(3)
            a /= np.linalg.norm(a)
            th = rng.uniform(1e-9, np.pi - 1e-9)
            w = so3_log(rot_from_axis_angle(a, th))
            nw = float(np.linalg.norm(w))
            worst_ang = max(worst_ang, abs(nw - th))
            worst_axis = max(worst_axis, float(np.linalg.norm(w / max(nw, 1e-300) - a)))
        self.assertLess(worst_ang, 1e-9, f"角度误差 {worst_ang:.3e}")
        self.assertLess(worst_axis, 1e-6, f"轴误差 {worst_axis:.3e}")

    def test_near_pi_magnitude(self):
        """θ→π 时角度必须仍然精确。真实旧实现在此处发散到 1e7 量级（见下面的
        TestLegacyBaselineRegression，那里用 /tmp/armctrl_baseline 独立证明）。"""
        rng = np.random.default_rng(1)
        for delta in (1e-3, 1e-6, 1e-9, 1e-12):
            th = np.pi - delta
            worst = 0.0
            for _ in range(200):
                a = rng.standard_normal(3)
                a /= np.linalg.norm(a)
                w = so3_log(rot_from_axis_angle(a, th))
                self.assertTrue(np.all(np.isfinite(w)))
                worst = max(worst, abs(float(np.linalg.norm(w)) - th))
            self.assertLess(worst, 1e-9, f"delta={delta:.0e} 角度误差 {worst:.3e}")

    def test_exact_pi_symmetric_input(self):
        rng = np.random.default_rng(2)
        for _ in range(500):
            a = rng.standard_normal(3)
            a /= np.linalg.norm(a)
            R = rot_pi_symmetric(a)
            self.assertLess(float(np.abs(R - R.T).max()), 1e-15)
            w = so3_log(R)
            self.assertTrue(np.all(np.isfinite(w)))
            self.assertAlmostEqual(float(np.linalg.norm(w)), np.pi, places=7)
            u = w / np.linalg.norm(w)
            self.assertGreater(abs(float(np.dot(u, a))), 1.0 - 1e-7)

    def test_equivalent_branch_same_rotation(self):
        """omega_alt = omega − 2π·â 必须表示同一旋转。"""
        rng = np.random.default_rng(3)
        for _ in range(500):
            a = rng.standard_normal(3)
            a /= np.linalg.norm(a)
            th = rng.uniform(0.1, np.pi - 0.1)
            w = th * a
            w_alt = so3_equivalent_branch(w)
            R1 = rot_from_axis_angle(a, th)
            n2 = float(np.linalg.norm(w_alt))
            R2 = rot_from_axis_angle(w_alt / n2, n2)
            self.assertLess(float(np.abs(R1 - R2).max()), 1e-12)
            self.assertAlmostEqual(n2, TWO_PI - th, places=9)


class TestLiftLattice(unittest.TestCase):
    """lift 候选集必须是格点 {(θ+2πk)a}，且 −θa 一般**不是** lift。"""

    def test_both_candidates_are_exact_lifts_and_2pi_apart(self):
        rng = np.random.default_rng(11)
        for _ in range(400):
            a = rng.standard_normal(3)
            a /= np.linalg.norm(a)
            th = rng.uniform(1e-3, np.pi)
            R = rot_from_axis_angle(a, th)
            c0, c1 = so3_lift_candidates(th * a)
            for c in (c0, c1):
                self.assertLess(float(np.abs(expm_so3(c) - R).max()), 1e-12)
            self.assertAlmostEqual(float(np.linalg.norm(c0 - c1)), TWO_PI, places=9)
            self.assertLess(float(np.linalg.norm(c1)), TWO_PI + 1e-12)

    def test_negated_omega_is_not_a_lift_except_at_0_and_pi(self):
        rng = np.random.default_rng(12)
        worst_mid, best_pi = 0.0, 0.0
        for _ in range(300):
            a = rng.standard_normal(3)
            a /= np.linalg.norm(a)
            th = rng.uniform(0.3, np.pi - 0.3)
            R = rot_from_axis_angle(a, th)
            worst_mid = max(worst_mid, float(np.abs(expm_so3(-th * a) - R).max()))
            Rp = rot_pi_symmetric(a)
            best_pi = max(best_pi, float(np.abs(expm_so3(-np.pi * a) - Rp).max()))
        self.assertGreater(worst_mid, 0.1, "−θa 竟然是一般 θ 的 lift？")
        self.assertLess(best_pi, 1e-12, "θ=π 时 ±πa 应当都是 lift")

    def test_select_branch_first_call_is_principal(self):
        w = np.array([0.1, -0.2, 0.3])
        lift, k = so3_select_branch(w, None, 0)
        self.assertEqual(k, 0)
        self.assertTrue(np.allclose(lift, w))

    def test_select_branch_tie_keeps_previous(self):
        """prev 落在两候选正中间时必须保持上一分支（确定性，不抖动）。"""
        a = np.array([0.0, 0.0, 1.0])
        w = np.pi * a
        mid = (w + so3_equivalent_branch(w)) / 2.0
        for prev_branch in (0, -1):
            _, k = so3_select_branch(w, mid, prev_branch)
            self.assertEqual(k, prev_branch)


class TestCutLocusFixtures(unittest.TestCase):
    """前提检查：六个 fixture 必须真的落在切割面上（否则后面的用例是空的）。"""

    def test_all_six_fixtures_trigger_2pi_jump(self):
        for name, axis, comp in CUT_CASES:
            with self.subTest(case=name):
                RA, RB = cut_pair(axis, comp)
                self.assertLess(float(np.abs(RA - RB).max()), 1e-11,
                                f"{name}: 两个矩阵差得太远，不算切割面")
                jump = float(np.linalg.norm(so3_log(RA) - so3_log(RB)))
                self.assertAlmostEqual(jump, TWO_PI, places=6,
                                       msg=f"{name}: 无状态跳变 {jump:.6f}，未命中切割面")


# ------------------------------------------------------------------ tracker


class TestOrientationTracker(unittest.TestCase):
    """两级状态：第 1 级 lift 必须**精确**且连续；第 2 级命令才允许限速/限幅。"""

    def test_lift_is_exact_on_every_cut_fixture(self):
        worst = 0.0
        for name, axis, comp in CUT_CASES:
            RA, RB = cut_pair(axis, comp)
            tr = OrientationErrorTracker()
            for R in (RA, RB):
                w = tr.lift(np.eye(3), R)
                worst = max(worst, float(np.abs(expm_so3(w) - R).max()))
        self.assertLess(worst, 1e-12, f"lift 不是精确对数，最差 {worst:.3e}")

    def test_lift_continuous_across_every_tie_plane(self):
        """扫过切割面时**连续 lift** 的单拍跳变必须 ≪ 2π。

        这里检查的是第 1 级（无限速、无限幅），所以不可能被 0.016 rad 的
        单拍限速"救活"。
        """
        for name, axis, comp in CUT_CASES:
            with self.subTest(case=name):
                a = np.asarray(axis, float)
                a = a / np.linalg.norm(a)
                d = np.zeros(3)
                d[comp] = 1.0
                tr = OrientationErrorTracker()
                prev, worst = None, 0.0
                for s in np.linspace(-1e-13, 1e-13, 41):
                    R = rot_pi_symmetric(a + s * d)
                    w = tr.lift(np.eye(3), R)
                    self.assertLess(float(np.abs(expm_so3(w) - R).max()), 1e-12)
                    if prev is not None:
                        worst = max(worst, float(np.linalg.norm(w - prev)))
                    prev = w
                self.assertLess(worst, 1e-6, f"{name}: 连续 lift 仍跳变 {worst:.3e}")

    def test_continuous_crossing_pi_on_general_axis(self):
        """一般轴上误差角连续从 π−δ 增到 π+δ：lift 必须逐点连续且范数超过 π。"""
        a = np.array([0.3, -0.5, 0.81])
        a /= np.linalg.norm(a)
        ths = np.linspace(np.pi - 0.05, np.pi + 0.05, 2001)
        step = float(ths[1] - ths[0])
        tr = OrientationErrorTracker()
        prev, worst = None, 0.0
        for th in ths:
            R = rot_from_axis_angle(a, th)
            w = tr.lift(np.eye(3), R)
            self.assertLess(float(np.abs(expm_so3(w) - R).max()), 1e-10)
            if prev is not None:
                worst = max(worst, float(np.linalg.norm(w - prev)))
            prev = w
        self.assertLess(worst, 3.0 * step, f"跨 π 时单拍跳变 {worst:.3e} ≫ 步长 {step:.3e}")
        self.assertAlmostEqual(float(np.linalg.norm(prev)), float(ths[-1]), places=6)
        self.assertEqual(tr.branch_switches, 1, "跨 π 应当恰好切换一次分支")

    def test_step_from_zero_to_near_pi_takes_principal(self):
        a = np.array([0.2, 0.9, -0.3])
        tr = OrientationErrorTracker()
        tr.lift(np.eye(3), np.eye(3))
        w = tr.lift(np.eye(3), rot_from_axis_angle(a, np.pi - 0.01))
        self.assertAlmostEqual(float(np.linalg.norm(w)), np.pi - 0.01, places=9)
        self.assertEqual(tr.branch, 0)
        self.assertEqual(tr.branch_switches, 0)

    def test_high_rate_reference_has_no_lag(self):
        """5 / 8 / 10 / 20 rad/s 参考下，缺省 tracker 输出必须与无状态主值**逐位相同**。

        上一版缺省 max_rate=8 rad/s、dt=0.002 → 单拍 0.016 rad，这些参考会被限速，
        产生被隐藏的跟踪滞后。现在限速缺省关闭。
        """
        dt = 0.002
        for rate in (5.0, 8.0, 10.0, 20.0):
            with self.subTest(rate=rate):
                tr = OrientationErrorTracker()
                worst = 0.0
                for k in range(1000):
                    Rd = rot_from_axis_angle([0.0, 0.0, 1.0], 0.6 * np.sin(rate * k * dt))
                    w = tr.update(np.eye(3), Rd, dt=dt)
                    worst = max(worst, float(np.linalg.norm(w - so3_log(Rd))))
                self.assertEqual(tr.rate_limited, 0, f"{rate} rad/s 触发了限速")
                self.assertLess(worst, 1e-15, f"{rate} rad/s 输出偏离主值 {worst:.3e}")

    def test_first_call_and_reset_reproduce_stateless_principal(self):
        RA, _ = cut_pair(*CUT_CASES[0][1:])
        tr = OrientationErrorTracker()
        first = tr.lift(np.eye(3), RA)
        self.assertTrue(np.allclose(first, so3_log(RA)))
        # 把它推进另一个分支
        tr.lift(np.eye(3), rot_from_axis_angle(-np.asarray(CUT_CASES[0][1]), np.pi - 0.2))
        tr.lift(np.eye(3), RA)
        tr.reset()
        self.assertTrue(np.allclose(tr.lift(np.eye(3), RA), so3_log(RA)),
                        "reset 后没有回到无状态主值")
        self.assertEqual(tr.branch_switches, 0)

    def test_same_frame_different_history_still_exact(self):
        """历史不同 → lift 可能落在两个等价格点之一，但**都必须精确表示同一旋转**，
        且绝不会是被限速/限幅的值。"""
        RA, _ = cut_pair(*CUT_CASES[0][1:])
        outs = []
        for pre in (None, np.pi - 0.2, -(np.pi - 0.2)):
            tr = OrientationErrorTracker()
            if pre is not None:
                sign = 1.0 if pre > 0 else -1.0
                tr.lift(np.eye(3),
                        rot_from_axis_angle(sign * np.asarray(CUT_CASES[0][1]), abs(pre)))
            outs.append(tr.lift(np.eye(3), RA))
        for w in outs:
            self.assertLess(float(np.abs(expm_so3(w) - RA).max()), 1e-12)
        self.assertGreater(float(np.linalg.norm(outs[1] - outs[2])), 1.0,
                           "历史相反时应当落在不同格点（否则这个用例没测到东西）")

    def test_branch_switch_counter_counts_switches_not_alt_selections(self):
        """持续处于 k=−1 的绕圈状态时，计数只能加一次。"""
        a = np.array([0.0, 0.0, 1.0])
        tr = OrientationErrorTracker()
        for th in np.linspace(np.pi - 0.02, np.pi + 0.3, 400):
            tr.lift(np.eye(3), rot_from_axis_angle(a, th))
        self.assertEqual(tr.branch, -1, "应当停在绕圈分支")
        self.assertEqual(tr.branch_switches, 1,
                         f"分支切换计数语义错误：{tr.branch_switches}")
        self.assertEqual(tr.branch_flips, tr.branch_switches)

    def test_rate_limit_requires_explicit_dt(self):
        a = np.array([0.0, 1.0, 0.0])
        tr = OrientationErrorTracker(max_rate=8.0)          # 没给 dt
        tr.update(np.eye(3), rot_from_axis_angle(a, 0.1))
        with self.assertRaises(ValueError):
            tr.update(np.eye(3), rot_from_axis_angle(a, 1.0))

    def test_rate_limit_never_pollutes_the_lift(self):
        """限速只作用在第 2 级；第 1 级 lift 必须仍是精确对数。"""
        a = np.array([0.0, 1.0, 0.0])
        tr = OrientationErrorTracker(max_rate=8.0, dt=0.002)
        tr.update(np.eye(3), rot_from_axis_angle(a, 0.1))
        R = rot_from_axis_angle(a, 1.0)
        cmd = tr.update(np.eye(3), R)
        self.assertEqual(tr.rate_limited, 1)
        self.assertLess(float(np.linalg.norm(cmd)), 0.9, "限速没生效")
        self.assertTrue(np.allclose(tr.last_lift, so3_log(R)),
                        "限速污染了 lift")
        self.assertLess(float(np.abs(expm_so3(tr.last_lift) - R).max()), 1e-12)

    def test_norm_saturation_is_command_only_and_not_a_log(self):
        """限幅后的命令**不再**是 R_err 的对数——必须能被检测到，而不是被宣称成 log。"""
        a = np.array([0.0, 0.0, 1.0])
        tr = OrientationErrorTracker(max_norm=1.0)
        R = rot_from_axis_angle(a, 3.0)
        cmd = tr.update(np.eye(3), R)
        self.assertEqual(tr.norm_saturated, 1)
        self.assertAlmostEqual(float(np.linalg.norm(cmd)), 1.0, places=12)
        self.assertGreater(float(np.abs(expm_so3(cmd) - R).max()), 0.1,
                           "限幅后竟然还等于 R_err —— 那说明限幅没生效")
        self.assertLess(float(np.abs(expm_so3(tr.last_lift) - R).max()), 1e-12,
                        "lift 被限幅污染了")

    def test_mutation_disabling_branch_selection_breaks_continuity(self):
        """**mutation test**：把分支择优换成"永远取主值"后，连续性用例必须失败。

        这里**没有**限速、**没有**限幅，因此不可能被 0.016 rad 单拍限速救活。
        """
        for name, axis, comp in CUT_CASES:
            with self.subTest(case=name):
                RA, RB = cut_pair(axis, comp)
                # 变异体：只用主值（等价于关掉 so3_select_branch 的分支择优）
                jump = float(np.linalg.norm(so3_log(RA) - so3_log(RB)))
                self.assertGreater(jump, 6.0, "变异体没有产生跳变，mutation 无效")
                tr = OrientationErrorTracker()
                good = float(np.linalg.norm(
                    tr.lift(np.eye(3), RA) - tr.lift(np.eye(3), RB)))
                self.assertLess(good, 1e-6)
                self.assertGreater(jump / max(good, 1e-300), 1e6,
                                   "保护前后差距不足，说明用例没有真正测到分支择优")


# ------------------------------------------------------------------ 控制器


class TestControllerContinuity(unittest.TestCase):
    """力矩连续性：**同时**检查 raw wrench、raw torque 与饱和后的 torque。

    只看饱和后的力矩会掩盖控制律本身的跳变（饱和会把两个方向相反的巨大力矩
    都压到同一个上限附近）。
    """

    def _impedance_pair(self, RA, RB, **kw):
        robot = panda()
        p0, Rc = robot.fk(Q_HOME)
        ctrl = CartesianImpedanceController(
            robot, ImpedanceGains(k_trans=600.0, k_rot=30.0, zeta=1.0),
            q_null=Q_HOME, **kw,
        )
        out = []
        for Rd in (RA @ Rc, RB @ Rc):
            tau = ctrl.compute(Q_HOME, np.zeros(7), p0, Rd)
            out.append((ctrl.last_wrench.copy(), ctrl.last_tau_raw.copy(), tau.copy()))
        return out

    def _hybrid_pair(self, RA, RB, **kw):
        robot = panda()
        p0, Rc = robot.fk(Q_HOME)
        ctrl = HybridForcePositionController(robot, [1, 1, 0, 1, 1, 1], **kw)
        out = []
        for Rd in (RA @ Rc, RB @ Rc):
            tau = ctrl.compute(Q_HOME, np.zeros(7), p0, Rd, np.zeros(6), np.zeros(6))
            out.append((ctrl.last_wrench.copy(), ctrl.last_tau_raw.copy(), tau.copy()))
        return out

    def test_impedance_continuous_on_every_cut_fixture(self):
        for name, axis, comp in CUT_CASES:
            with self.subTest(case=name):
                RA, RB = cut_pair(axis, comp)
                (wA, rA, sA), (wB, rB, sB) = self._impedance_pair(RA, RB)
                self.assertLess(float(np.linalg.norm(wA - wB)), 1e-8,
                                f"{name}: raw wrench 跳变")
                self.assertLess(float(np.linalg.norm(rA - rB)), 1e-8,
                                f"{name}: raw torque 跳变")
                self.assertLess(float(np.linalg.norm(sA - sB)), 1e-8,
                                f"{name}: 饱和后 torque 跳变")

    def test_hybrid_continuous_on_every_cut_fixture(self):
        for name, axis, comp in CUT_CASES:
            with self.subTest(case=name):
                RA, RB = cut_pair(axis, comp)
                (wA, rA, sA), (wB, rB, sB) = self._hybrid_pair(RA, RB)
                self.assertLess(float(np.linalg.norm(wA - wB)), 1e-8,
                                f"{name}: raw wrench 跳变")
                self.assertLess(float(np.linalg.norm(rA - rB)), 1e-8,
                                f"{name}: raw torque 跳变")
                self.assertLess(float(np.linalg.norm(sA - sB)), 1e-8,
                                f"{name}: 饱和后 torque 跳变")

    def test_mutation_track_orientation_off_breaks_both_controllers(self):
        """**mutation test**：关掉 lift 跟踪后，raw 与饱和量都必须出现巨大跳变。"""
        RA, RB = cut_pair(*CUT_CASES[0][1:])
        (wA, rA, sA), (wB, rB, sB) = self._impedance_pair(
            RA, RB, track_orientation=False
        )
        self.assertGreater(float(np.linalg.norm(wA - wB)), 100.0)
        self.assertGreater(float(np.linalg.norm(rA - rB)), 100.0)
        self.assertGreater(float(np.linalg.norm(sA - sB)), 100.0)
        (wA, rA, sA), (wB, rB, sB) = self._hybrid_pair(
            RA, RB, track_orientation=False
        )
        self.assertGreater(float(np.linalg.norm(wA - wB)), 100.0)
        self.assertGreater(float(np.linalg.norm(rA - rB)), 100.0)

    def test_reset_clears_branch_history_in_controllers(self):
        robot = panda()
        p0, Rc = robot.fk(Q_HOME)
        RA, _ = cut_pair(*CUT_CASES[0][1:])
        ctrl = CartesianImpedanceController(robot, q_null=Q_HOME)
        first = ctrl.compute(Q_HOME, np.zeros(7), p0, RA @ Rc).copy()
        axis = np.asarray(CUT_CASES[0][1], float)
        ctrl.compute(Q_HOME, np.zeros(7), p0,
                     rot_from_axis_angle(-axis, np.pi - 0.2) @ Rc)
        ctrl.compute(Q_HOME, np.zeros(7), p0, RA @ Rc)
        ctrl.reset()
        again = ctrl.compute(Q_HOME, np.zeros(7), p0, RA @ Rc)
        self.assertTrue(np.allclose(first, again), "reset 后没有回到首次调用行为")

        hy = HybridForcePositionController(robot, [1, 1, 0, 1, 1, 1])
        t1 = hy.compute(Q_HOME, np.zeros(7), p0, RA @ Rc, np.zeros(6), np.zeros(6)).copy()
        hy.compute(Q_HOME, np.zeros(7), p0, RA @ Rc, np.ones(6), np.zeros(6))
        hy.reset()
        t2 = hy.compute(Q_HOME, np.zeros(7), p0, RA @ Rc, np.zeros(6), np.zeros(6))
        self.assertTrue(np.allclose(t1, t2), "Hybrid reset 未清空力积分/分支历史")


# ------------------------------------------------------------------ 真实 baseline


_BASE_DIR = "/tmp/armctrl_baseline"


def _load_legacy():
    """用 importlib 在**独立命名空间**加载真实旧实现，不依赖它有新 API。"""
    path = os.path.join(_BASE_DIR, "core", "kinematics.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("armctrl_legacy_probe_kin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_LEGACY = _load_legacy()


@unittest.skipUnless(_LEGACY is not None, f"缺少真实 baseline：{_BASE_DIR}")
class TestLegacyBaselineRegression(unittest.TestCase):
    """在**真实旧代码**上复现原 bug。

    旧模块没有 so3_log / OrientationErrorTracker，但它有公开的
    rot_to_axis_angle_error()，用它做 adapter 就足以证明根因。
    """

    def test_legacy_lacks_new_api(self):
        self.assertFalse(hasattr(_LEGACY, "so3_log"))
        self.assertFalse(hasattr(_LEGACY, "OrientationErrorTracker"))
        self.assertTrue(hasattr(_LEGACY, "rot_to_axis_angle_error"))

    def test_legacy_diverges_near_pi_current_does_not(self):
        a = np.array([0.3, -0.5, 0.81])
        a /= np.linalg.norm(a)
        for delta, min_legacy_err in ((1e-9, 1e3), (1e-12, 1e3)):
            with self.subTest(delta=delta):
                th = np.pi - delta
                R = rot_from_axis_angle(a, th)
                old = _LEGACY.rot_to_axis_angle_error(np.eye(3), R)
                new = so3_log(R)
                self.assertGreater(abs(float(np.linalg.norm(old)) - th), min_legacy_err,
                                   "旧实现没有复现出发散")
                self.assertLess(abs(float(np.linalg.norm(new)) - th), 1e-9)

    def test_legacy_reports_zero_error_for_180_degree_misalignment(self):
        """真正的旧 bug：θ=π 精确对称输入下，旧实现返回**零误差**。

        控制器因此完全不会去纠正一个 180° 的姿态偏差。
        """
        for name, axis, comp in CUT_CASES:
            with self.subTest(case=name):
                RA, _ = cut_pair(axis, comp)
                old = _LEGACY.rot_to_axis_angle_error(np.eye(3), RA)
                self.assertLess(float(np.linalg.norm(old)), 1e-6,
                                "旧实现没有复现出零误差")
                self.assertAlmostEqual(float(np.linalg.norm(so3_log(RA))), np.pi,
                                       places=6)

    def test_legacy_impedance_torque_is_wrong_by_hundreds_of_newton_metres(self):
        """把旧的姿态误差喂进当前控制律，力矩与正确值差多少（真实 baseline 对比）。"""
        robot = panda()
        p0, Rc = robot.fk(Q_HOME)
        RA, _ = cut_pair(*CUT_CASES[0][1:])
        Rd = RA @ Rc
        gains = ImpedanceGains(k_trans=600.0, k_rot=30.0, zeta=1.0)
        ctrl = CartesianImpedanceController(robot, gains, q_null=Q_HOME)
        tau_new = ctrl.compute(Q_HOME, np.zeros(7), p0, Rd)
        J = robot.jacobian(Q_HOME)
        e_old = np.concatenate([p0 - robot.fk(Q_HOME)[0],
                                _LEGACY.rot_to_axis_angle_error(robot.fk(Q_HOME)[1], Rd)])
        K = gains.stiffness()
        tau_old = robot.saturate_torque(J.T @ (K @ e_old) + robot.bias(Q_HOME, np.zeros(7)))
        self.assertGreater(float(np.linalg.norm(tau_new - tau_old)), 20.0,
                           "旧姿态误差与新实现没有产生显著力矩差异")


# ------------------------------------------------------------------ IK


class TestIKCutLocus(unittest.TestCase):
    """IK 在切割面附近必须稳定，并且**两侧结果一致**。"""

    def test_ik_succeeds_on_all_fixtures_and_qinits(self):
        robot = panda()
        failures = []
        for qi_idx, qi in enumerate(Q_INITS):
            p0, Rc = robot.fk(qi)
            for name, axis, comp in CUT_CASES:
                RA, RB = cut_pair(axis, comp)
                for side, Rd in (("A", RA @ Rc), ("B", RB @ Rc)):
                    ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
                    res = ik.solve(p0, Rd, qi.copy(), max_iters=200)
                    if not res.success:
                        failures.append(
                            f"q{qi_idx}/{name}/{side}: iters={res.iters} "
                            f"pos={res.fk_pos_err:.2e} rot={res.fk_rot_err:.2e} "
                            f"clips={res.joint_limit_clip_count}"
                        )
        # ⚠️ 这条原来写的是 assertLessEqual(len(failures), 1)，
        # 名字和注释都说"全部样本"，实际却**允许放过一个确定性失败**
        # （第四轮审核 T-03）。实测 36 个样本 **0 失败**，
        # 那条松弛纯属多余，现在收紧成"一个都不许失败"。
        self.assertEqual(
            failures, [],
            f"{len(failures)}/{len(Q_INITS)*len(CUT_CASES)*2} 个切割面用例失败：{failures}",
        )

    def test_both_sides_of_cut_locus_reach_the_same_target_pose(self):
        """切割面两侧都必须**独立 FK 验收通过**，末端位姿一致。

        注意：关节解**不要求**逐位相同。θ = π 时 +πa 与 −πa 是同一个目标姿态的
        两个合法 lift，它们在关节空间是两条不同的路径；7 自由度冗余臂本来就有
        多个 IK 解。要求关节角相同是错误的判据。这里检验的是「两侧都真的到达
        了同一个目标位姿」，并统计有多少 fixture 落到同一个关节解上。
        """
        robot = panda()
        same_q, total, failures = 0, 0, []
        for name, axis, comp in CUT_CASES:
            RA, RB = cut_pair(axis, comp)
            p0, Rc = robot.fk(Q_HOME)
            got = []
            for side, Rd in (("A", RA @ Rc), ("B", RB @ Rc)):
                ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
                r = ik.solve(p0, Rd, Q_HOME.copy(), max_iters=200)
                got.append((side, Rd, r))
                total += 1
                if not r.success:
                    failures.append(f"{name}/{side}")
            if len(failures) == 0:
                for side, Rd, r in got:
                    p_f, R_f = robot.fk(r.q)
                    self.assertLess(float(np.linalg.norm(p_f - p0)), 1e-4)
                    self.assertLess(float(np.linalg.norm(so3_log(Rd @ R_f.T))), 1e-3)
                if float(np.linalg.norm(got[0][2].q - got[1][2].q)) < 1e-6:
                    same_q += 1
        self.assertEqual(failures, [], f"{len(failures)}/{total} 侧未通过 FK 验收")
        self.assertGreaterEqual(same_q, 1,
                                "没有任何 fixture 两侧落到同一关节解，说明分支择优完全失效")

    def test_no_branch_chatter_during_iteration(self):
        """迭代中不得反复切支。上一版实测最多切换 6 次。"""
        robot = panda()
        rng = np.random.default_rng(3)
        worst, worst_case = 0, ""
        for k in range(60):
            qi = Q_INITS[k % len(Q_INITS)]
            _, Rc = robot.fk(qi)
            qt = rng.uniform(robot.q_lower, robot.q_upper)
            pt, Rt = robot.fk(qt)
            if float(np.linalg.norm(so3_log(Rt @ Rc.T))) < np.pi - 0.35:
                continue
            ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
            res = ik.solve(pt, Rt, qi.copy(), max_iters=200)
            if res.branch_switch_count > worst:
                worst, worst_case = res.branch_switch_count, f"sample {k}"
        self.assertLessEqual(worst, 2, f"分支抖动 {worst} 次（{worst_case}）")

    def test_dual_branch_measurably_improves_near_pi_targets(self):
        """双分支必须在**多目标**上有可测收益，而不是只在一个 53 步 fixture 上有效。"""
        robot = panda()
        rng = np.random.default_rng(3)
        cases = []
        while len(cases) < 60:
            qi = Q_INITS[len(cases) % len(Q_INITS)]
            _, Rc = robot.fk(qi)
            qt = rng.uniform(robot.q_lower, robot.q_upper)
            pt, Rt = robot.fk(qt)
            if float(np.linalg.norm(so3_log(Rt @ Rc.T))) > np.pi - 0.35:
                cases.append((qi, pt, Rt))
        stats = {}
        for dual in (False, True):
            ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
            ok = clips = 0
            t0 = time.perf_counter()
            for qi, pt, Rt in cases:
                res = ik.solve(pt, Rt, qi.copy(), max_iters=200, dual_branch=dual)
                ok += int(res.success)
                clips += res.joint_limit_clip_count
            stats[dual] = (ok, clips, time.perf_counter() - t0)
        s_off, s_on = stats[False], stats[True]
        self.assertGreater(
            s_on[0], s_off[0] + 5,
            f"双分支未带来可测收益：off={s_off} on={s_on}（{len(cases)} 个近 π 目标）",
        )
        self.assertLess(s_on[1], s_off[1], f"双分支未减少撞限位次数 off={s_off} on={s_on}")
        self.assertLess(s_on[2], 4.0 * s_off[2], f"双分支开销超过 4×：{s_on[2]/s_off[2]:.2f}×")

    def test_second_branch_only_runs_when_needed(self):
        """主分支干净时不得白跑第二分支（记录在 branch_report 里）。"""
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
        res = ik.solve(p0 + np.array([0.03, 0.0, 0.01]), R0, Q_HOME.copy())
        self.assertTrue(res.success)
        self.assertEqual(len(res.branch_report), 1, "主分支干净却仍跑了第二分支")
        for key in ("iters", "time_s", "branch", "joint_limit_clip_count"):
            self.assertIn(key, res.branch_report[0])

    def test_success_ignores_solver_self_report_when_they_disagree(self):
        """构造一个 converged=False 但 FK 复核**通过**的算例。

        做法：先用充足的 max_iters 得到收敛步数 K，再用 max_iters=K 重跑——
        循环在第 K−1 步施加了那一步更新后就结束，来不及再判一次容差，于是
        求解器自报 converged=False，而返回的 q 其实已经在容差内。

        此时 success 必须听 FK 的（True），而不是听求解器自报的（False）。
        这正是"success 必须由独立 FK 验证"的可检验含义。
        """
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        p_des = p0 + np.array([0.08, -0.04, 0.02])
        ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
        full = ik.solve(p_des, R0, Q_HOME.copy(), max_iters=200, dual_branch=False)
        self.assertTrue(full.success)
        K = full.iters
        self.assertGreater(K, 0)
        cut = ik.solve(p_des, R0, Q_HOME.copy(), max_iters=K, dual_branch=False)
        self.assertFalse(cut.converged, "构造失败：求解器仍自报收敛")
        self.assertTrue(cut.fk_verified, "构造失败：FK 复核未通过")
        self.assertTrue(cut.self_report_mismatch)
        self.assertTrue(cut.success, "success 跟着求解器自报走了，而不是跟着 FK")

    def test_success_rejects_a_false_positive_self_report(self):
        """反向：即使求解器自报收敛，只要 FK 复核不过，success 也必须是 False。"""
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
        res = ik.solve(p0, R0, Q_HOME.copy(), max_iters=200, dual_branch=False)
        self.assertTrue(res.success)
        res.converged = True                    # 伪造一个"自报成功"
        res.fk_verified = False                 # 但独立复核不过
        self.assertFalse(res.success)
        self.assertTrue(res.self_report_mismatch)

    def test_success_is_verified_by_independent_fk(self):
        """求解器自报不作数：success 必须由独立 FK 复算决定。"""
        robot = panda()
        rng = np.random.default_rng(9)
        checked = 0
        for k in range(40):
            qi = Q_INITS[k % len(Q_INITS)]
            qt = np.clip(qi + rng.normal(0, 0.6, robot.n), robot.q_lower, robot.q_upper)
            pt, Rt = robot.fk(qt)
            ik = DLSInverseKinematics(robot, method="adaptive", lam0=0.05)
            res = ik.solve(pt, Rt, qi.copy(), max_iters=200)
            p_f, R_f = robot.fk(res.q)
            self.assertAlmostEqual(res.fk_pos_err,
                                   float(np.linalg.norm(pt - p_f)), places=12)
            self.assertAlmostEqual(res.fk_rot_err,
                                   float(np.linalg.norm(so3_log(Rt @ R_f.T))), places=12)
            self.assertEqual(res.success,
                             res.fk_pos_err < 1e-4 and res.fk_rot_err < 1e-3)
            checked += 1
        self.assertEqual(checked, 40)


class TestIKDiagnostics(unittest.TestCase):
    """诊断量必须齐全、同源、可定位到具体某一步。"""

    def _run(self, **kw):
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        ik = DLSInverseKinematics(robot, **kw)
        res = ik.solve(p0 + np.array([0.12, -0.05, 0.03]), R0, Q_HOME.copy(),
                       max_iters=60, dual_branch=False)
        return ik, res

    def test_all_per_step_traces_have_consistent_length(self):
        ik, res = self._run(method="adaptive", lam0=0.05)
        n = len(res.sigma_min_trace)
        self.assertGreater(n, 0)
        for name in ("sigma_trace", "rank_trace", "lam_trace", "manip_trace",
                     "dq_raw_norm_trace", "dq_raw_trace", "dq_norm_trace",
                     "raw_residual_trace", "clipped_residual_trace",
                     "null_leak_trace"):
            self.assertEqual(len(getattr(res, name)), n, f"{name} 长度不一致")
        # 每步残差可定位：不是一个标量最大值
        self.assertEqual(len(res.dq_raw_trace[0]), ik.robot.n)
        self.assertEqual(len(res.sigma_trace[0]), 6)

    def test_rank_and_pinv_use_the_same_matrix(self):
        """C 类核心 bug：σ/rank 必须与实际被求逆的矩阵同源（都用 J_task）。"""
        robot = panda()
        ik, res = self._run(method="pinv", rcond=1e-6)
        # 用 result 里记录的 q 轨迹无法反推，改为逐步复核第一步
        J = robot.jacobian(Q_HOME)
        J_task = ik._normalize_jacobian(J, POSE_TASK)
        sv = np.linalg.svd(J_task, compute_uv=False)
        self.assertTrue(np.allclose(res.sigma_trace[0], sv, atol=1e-12),
                        "记录的奇异值谱不是 J_task 的谱")
        self.assertEqual(res.rank_trace[0], int(np.sum(sv > ik.rcond * sv[0])))
        # 同一 rcond 下，raw J 与 J_task 的谱不同——所以必须同源
        sv_raw = np.linalg.svd(J, compute_uv=False)
        self.assertFalse(np.allclose(sv, sv_raw, atol=1e-6))

    def test_task_residual_is_recorded_and_matches_definition(self):
        robot = panda()
        ik, res = self._run(method="adaptive", lam0=0.05)
        self.assertGreater(len(res.raw_residual_trace), 0, "未记录任务残差")
        self.assertGreater(len(res.clipped_residual_trace), 0)
        J = robot.jacobian(Q_HOME)
        J_task = ik._normalize_jacobian(J, POSE_TASK)
        e = np.concatenate([res.dq_raw_trace[0] * 0.0, np.zeros(3)])  # 占位，见下
        # 直接用第一步的 dq_raw 复核残差定义 ‖J_task dq − e_task‖
        p0, R0 = robot.fk(Q_HOME)
        p_des = p0 + np.array([0.12, -0.05, 0.03])
        e = np.concatenate([p_des - p0, so3_log(R0 @ R0.T)])
        e_task = ik._normalize_error(e, POSE_TASK)
        expect = float(np.linalg.norm(J_task @ res.dq_raw_trace[0] - e_task))
        self.assertAlmostEqual(res.raw_residual_trace[0], expect, places=10)

    def test_step_clip_and_joint_limit_clip_are_counted_separately(self):
        """两类裁剪必须能被**分别**触发：一个只触发步长限幅，一个只触发关节限位裁剪。

        只断言"两个数不相等"是无效判据（它们可以偶然相等）。
        """
        robot = panda()

        # (a) 只触发关节限位裁剪：起点贴住上限、目标在限位外，步长足够小
        q0 = robot.q_upper - 1e-6
        p_far, _ = robot.fk(robot.q_upper)
        ik_a = DLSInverseKinematics(robot, method="fixed", lam0=0.5, step_clip=10.0)
        res_a = ik_a.solve(p_far + np.array([0.4, 0.4, 0.4]), None, q0.copy(),
                           max_iters=25, dual_branch=False)
        self.assertGreater(res_a.joint_limit_clip_count, 0, "(a) 未触发关节限位裁剪")
        self.assertEqual(res_a.step_clip_count, 0,
                         f"(a) 不该触发步长限幅，却触发了 {res_a.step_clip_count} 次")

        # (b) 只触发步长限幅：从行程中点出发、目标很远，但远离关节限位
        q_mid = 0.5 * (robot.q_lower + robot.q_upper)
        p_mid, R_mid = robot.fk(q_mid)
        ik_b = DLSInverseKinematics(robot, method="fixed", lam0=0.05, step_clip=1e-3)
        res_b = ik_b.solve(p_mid + np.array([0.05, 0.02, -0.03]), R_mid, q_mid.copy(),
                           max_iters=20, dual_branch=False)
        self.assertGreater(res_b.step_clip_count, 0, "(b) 未触发步长限幅")
        self.assertEqual(res_b.joint_limit_clip_count, 0,
                         f"(b) 不该撞限位，却裁剪了 {res_b.joint_limit_clip_count} 次")
        self.assertGreater(res_b.dq_raw_max, ik_b.step_clip)
        self.assertTrue(np.isfinite(res_b.min_joint_margin))
        robot.set_state(np.zeros(robot.n), np.zeros(robot.n))

    def test_step_clip_masks_large_raw_dq_under_normal_parameters(self):
        """**正常参数**（rcond=1e-6, step_clip=0.2）下 step_clip 是否也会掩盖大 dq。

        上一版只用 rcond=1e-12 + step_clip=1e-4 这种人为极端值触发限幅，
        属于制造条件让测试通过。这里用缺省参数 + 真实近奇异构型。
        """
        robot = panda()
        ik = DLSInverseKinematics(robot, method="pinv", rcond=1e-6)
        q_sing, _ = ik._stretched_config()
        q_sing = robot.clamp_to_limits(q_sing)
        p_s, R_s = robot.fk(q_sing)
        res = ik.solve(p_s + np.array([0.05, 0.05, 0.0]), R_s, q_sing.copy(),
                       max_iters=40, dual_branch=False)
        self.assertGreater(res.step_clip_count, 0,
                           "缺省参数下没触发 step_clip，本用例需要重新选构型")
        self.assertGreater(res.dq_raw_max, 5.0 * ik.step_clip,
                           f"原始 dq 峰值仅 {res.dq_raw_max:.4f}")
        # 被限幅的那一步，残差必然变大——限幅不是免费的
        idx = int(np.argmax(res.dq_raw_norm_trace))
        self.assertGreater(res.clipped_residual_trace[idx],
                           res.raw_residual_trace[idx])

    def test_null_leak_is_recorded(self):
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        for proj, bound in (("svd", 1e-9), ("damped", None)):
            ik = DLSInverseKinematics(robot, method="fixed", lam0=0.05,
                                      null_gain=1.0, null_projector=proj)
            res = ik.solve(p0 + np.array([0.05, 0.0, 0.0]), R0, Q_HOME.copy(),
                           max_iters=40, dual_branch=False)
            self.assertGreater(len(res.null_leak_trace), 0)
            worst = max(res.null_leak_trace)
            if bound is None:
                self.assertGreater(worst, 1e-4, "damped 投影竟然没有泄漏？")
            else:
                self.assertLess(worst, bound, f"svd 投影泄漏 {worst:.3e}")


# ------------------------------------------------------------------ 任务度量


class TestTaskWeighting(unittest.TestCase):
    """W 必须显式，且雅可比与误差必须用同一个 W。"""

    def test_task_weight_is_explicit_and_rejects_other_row_counts(self):
        ik = DLSInverseKinematics(panda())
        self.assertTrue(np.allclose(ik.task_weight(POSITION_TASK), np.eye(3)))
        W = ik.task_weight(POSE_TASK)
        self.assertTrue(np.allclose(np.diag(W), [1, 1, 1] + [ik.char_length] * 3))
        for rows in (1, 2, 4, 5, 7):
            with self.assertRaises(ValueError):
                DLSInverseKinematics.task_from_rows(rows)
        with self.assertRaises(ValueError):
            ik.task_weight("twist")

    def test_jacobian_and_error_share_the_same_weight(self):
        robot = panda()
        ik = DLSInverseKinematics(robot)
        J = robot.jacobian(Q_HOME)
        e = np.array([0.01, -0.02, 0.03, 0.1, -0.2, 0.3])
        W = ik.task_weight(POSE_TASK)
        self.assertTrue(np.allclose(ik._normalize_jacobian(J, POSE_TASK), W @ J))
        self.assertTrue(np.allclose(ik._normalize_error(e, POSE_TASK), W @ e))
        # 加权最小二乘的解必须等于 pinv(WJ)(We)
        dq = np.linalg.pinv(W @ J, rcond=ik.rcond) @ (W @ e)
        self.assertLess(float(np.linalg.norm((W @ J) @ dq - (W @ e))), 1e-9)

    def test_damping_requires_explicit_task_type(self):
        ik = DLSInverseKinematics(panda(), method="adaptive")
        self.assertGreater(ik._damping(0.0, POSITION_TASK), 0.0)
        self.assertGreater(ik._damping(0.0, POSE_TASK), 0.0)
        self.assertEqual(ik._damping(10.0, POSE_TASK), 0.0)
        with self.assertRaises(ValueError):
            ik._damping(0.0, 3)                       # 行数不再能冒充任务类型
        with self.assertRaises(ValueError):
            ik._damping(0.0, "orientation")

    def test_score_is_dimensionless(self):
        """择优键里不得出现 米 + 弧度 的裸加法。"""
        robot = panda()
        ik = DLSInverseKinematics(robot)
        p0, R0 = robot.fk(Q_HOME)
        res = ik.solve(p0 + np.array([0.02, 0.0, 0.0]), R0, Q_HOME.copy())
        key = ik._score(res)
        self.assertEqual(len(key), 4)
        self.assertAlmostEqual(
            key[3], max(res.fk_pos_err / 1e-4, res.fk_rot_err / 1e-3), places=9
        )


# ------------------------------------------------------------------ pinv


class TestPinvHardCutoff(unittest.TestCase):
    """rcond 的硬截断必须在**真实 Panda** 构型上验证，不是拿 np.diag 测 NumPy。"""

    def test_rcond_configurable_and_diagnostics_recorded(self):
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        for rc in (1e-3, 1e-6, 1e-12):
            ik = DLSInverseKinematics(robot, method="pinv", rcond=rc)
            res = ik.solve(p0 + np.array([0.05, 0.0, 0.0]), R0, Q_HOME.copy())
            self.assertEqual(ik.rcond, rc)
            self.assertGreater(len(res.sigma_trace), 0, "未记录奇异值谱")
            self.assertGreater(len(res.rank_trace), 0, "未记录数值秩")
            self.assertGreater(len(res.lam_trace), 0, "未记录阻尼")
            self.assertGreater(len(res.raw_residual_trace), 0, "未记录任务残差")

    def test_rcond_1e6_never_truncates_on_reachable_panda_configs(self):
        """独立扫描：Panda 可达构型下 J_task 的相对 σ_min 远高于 1e-6。

        结论必须写清楚：`rcond=1e-6` 对 Panda 而言**不会过早截断**，但也意味着
        硬截断分支在正常使用中几乎不会被触发。
        """
        robot = panda()
        ik = DLSInverseKinematics(robot, method="pinv", rcond=1e-6)
        rng = np.random.default_rng(0)
        q_sing, _ = ik._stretched_config()
        worst_rel = np.inf
        configs = [Q_HOME, 0.5 * (robot.q_lower + robot.q_upper), q_sing]
        configs += [np.clip(q_sing + rng.normal(0, 0.1, robot.n),
                            robot.q_lower, robot.q_upper) for _ in range(60)]
        configs += [rng.uniform(robot.q_lower, robot.q_upper) for _ in range(300)]
        for q in configs:
            sv = np.linalg.svd(ik._normalize_jacobian(robot.jacobian(q), POSE_TASK),
                               compute_uv=False)
            worst_rel = min(worst_rel, float(sv[-1] / sv[0]))
            self.assertEqual(int(np.sum(sv > 1e-6 * sv[0])), 6)
        self.assertGreater(worst_rel, 1e-5,
                           f"最小相对奇异值 {worst_rel:.3e} 已逼近 rcond=1e-6")

    def test_hard_cutoff_is_discontinuous_on_a_real_panda_config(self):
        """在真实近奇异 Panda 构型上跨过 rcond 阈值两侧，增益必须发生数量级跳变。

        这不是在测 NumPy 的 np.diag，而是在测「本项目会不会因为 rcond 选取
        产生不连续行为」。step_clip 只能压小跳变幅度，压不掉不连续本身。
        """
        robot = panda()
        ik = DLSInverseKinematics(robot, method="pinv", rcond=1e-6)
        q_sing, _ = ik._stretched_config()
        J_task = ik._normalize_jacobian(robot.jacobian(q_sing), POSE_TASK)
        sv = np.linalg.svd(J_task, compute_uv=False)
        rel = float(sv[-1] / sv[0])
        above = np.linalg.pinv(J_task, rcond=rel * 0.99)     # 保留最小奇异值
        below = np.linalg.pinv(J_task, rcond=rel * 1.01)     # 截断掉
        self.assertEqual(int(np.sum(sv > rel * 0.99 * sv[0])), 6)
        self.assertEqual(int(np.sum(sv > rel * 1.01 * sv[0])), 5)
        ratio = float(np.linalg.norm(above) / np.linalg.norm(below))
        self.assertGreater(ratio, 20.0,
                           f"rcond 变动 2% 只带来 {ratio:.1f}× 增益变化，硬截断没被测到")

    def test_rank_reported_matches_the_matrix_actually_inverted(self):
        """上一版 bug 的直接复现：同一 rcond 下 rank(J_task) 与 rank(J_raw) 会不同。

        使用一个真实 Panda 构型；本实现必须报告 **J_task** 的秩。
        """
        robot = panda()
        ik = DLSInverseKinematics(robot, method="pinv")
        q = np.array([-2.278094, -1.363590, 2.470867, -0.884264,
                      2.817645, 2.203854, -2.897300])
        J = robot.jacobian(q)
        J_task = ik._normalize_jacobian(J, POSE_TASK)
        a = np.linalg.svd(J_task, compute_uv=False)
        b = np.linalg.svd(J, compute_uv=False)
        rc = float(np.sqrt((a[-1] / a[0]) * (b[-1] / b[0])))
        rank_task = int(np.sum(a > rc * a[0]))
        rank_raw = int(np.sum(b > rc * b[0]))
        self.assertNotEqual(rank_task, rank_raw,
                            "这个构型不再能区分两个矩阵，需要重新挑选")
        ik.rcond = rc
        p_t, R_t = robot.fk(q)
        res = ik.solve(p_t + np.array([0.01, 0.0, 0.0]), R_t, q.copy(),
                       max_iters=1, dual_branch=False)
        self.assertEqual(res.rank_trace[0], rank_task,
                         "报告的秩不是实际被求逆矩阵的秩")


# ------------------------------------------------------------------ 特征长度


class TestCharacteristicLength(unittest.TestCase):

    def test_estimate_reach_has_no_side_effects(self):
        robot = panda()
        q_mark = np.array([0.11, -0.22, 0.33, -1.44, 0.55, 1.66, 0.77])
        v_mark = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        robot.set_state(q_mark, v_mark)
        ik = DLSInverseKinematics(robot)                     # 构造时会估计 reach
        self.assertTrue(np.allclose(robot.get_q(), q_mark), "reach 估计污染了 qpos")
        self.assertTrue(np.allclose(robot.get_qd(), v_mark), "reach 估计污染了 qvel")
        ik.sample_sigma_min(50, seed=1)
        self.assertTrue(np.allclose(robot.get_q(), q_mark), "采样污染了 qpos")
        self.assertTrue(np.allclose(robot.get_qd(), v_mark), "采样污染了 qvel")
        robot.set_state(np.zeros(robot.n), np.zeros(robot.n))

    def test_reach_is_measured_from_arm_base_not_world_origin(self):
        """Panda 法兰标称可达半径约 0.855 m；世界原点口径会得到约 1.19 m。

        装上夹爪后控制点前移 0.1029 m，可达半径必然变大，但**至多**只能大
        一个工具长度（工具方向一般不与径向重合）——这是纯几何上界，可推导。
        """
        from armctrl.core.robot import PANDA_TCP_OFFSET, make_panda as _mk
        tool = float(np.linalg.norm(PANDA_TCP_OFFSET))
        robot = panda()
        ik = DLSInverseKinematics(robot)                     # 缺省 = 指尖 TCP
        flange = _mk(tcp="flange")
        ik_f = DLSInverseKinematics(flange)
        self.assertGreater(ik_f.char_length, 0.82, "法兰可达半径偏离 Panda 标称")
        self.assertLess(ik_f.char_length, 0.90)
        self.assertGreater(ik.char_length, ik_f.char_length,
                           "装了工具反而更短，说明 TCP 没有生效")
        self.assertLessEqual(ik.char_length, ik_f.char_length + tool + 1e-9,
                             "可达半径增量超过工具长度，几何上不可能")

        base = ik._arm_base_origin()
        self.assertGreater(float(np.linalg.norm(base)), 0.2,
                           "基座锚点竟然在世界原点，本用例无法区分两种口径")
        rng = np.random.default_rng(0)
        world_max = 0.0
        for _ in range(2000):
            p, _ = robot.fk(rng.uniform(robot.q_lower, robot.q_upper))
            world_max = max(world_max, float(np.linalg.norm(p)))
        self.assertGreater(world_max, 1.15, "世界原点口径没有复现出偏大的值")
        self.assertGreater(world_max - ik.char_length, 0.25,
                           "两种口径差别太小，说明没有真的换口径")
        robot.set_state(np.zeros(robot.n), np.zeros(robot.n))

    def test_reach_sample_includes_documented_zero_configuration(self):
        """文档声称"包含零位"，就必须真的算过零位。"""
        robot = panda()
        ik = DLSInverseKinematics(robot)
        p0, _ = robot.fk(np.zeros(robot.n))
        base = ik._arm_base_origin()
        self.assertGreaterEqual(ik.char_length,
                                float(np.linalg.norm(p0 - base)) - 1e-12)
        robot.set_state(np.zeros(robot.n), np.zeros(robot.n))

    def test_reach_is_stable_across_seeds(self):
        robot = panda()
        base = DLSInverseKinematics(robot)
        vals = [base._estimate_reach(samples=2000, seed=s) for s in (1, 2, 3)]
        self.assertLess(max(vals) - min(vals), 5e-3,
                        f"reach 对 seed 敏感：{vals}")
        robot.set_state(np.zeros(robot.n), np.zeros(robot.n))


# ------------------------------------------------------------------ 阻尼标定


class TestDampingCalibration(unittest.TestCase):
    """标定集与验证集分开；报告分位数与尾部；说明阈值的分布依赖性。"""

    @classmethod
    def setUpClass(cls):
        cls.robot = panda()
        cls.ik = DLSInverseKinematics(cls.robot, method="adaptive")

    def test_defaults_are_exactly_reproducible_from_documented_calibration(self):
        """缺省阈值必须能被文档写明的标定条件**精确**复现（相对误差 < 1%）。

        上一版用 delta=0.02 / 0.01 容忍 27% / 74% 的相对误差，等于没有约束。
        """
        ik = DLSInverseKinematics(self.robot, method="adaptive")
        cal = ik.calibrate_sigma0(activate_frac=0.10, samples=6000, seed=7,
                                  distribution="mixed", q_center=Q_HOME,
                                  apply=False)
        for name, got, want in (("pos", cal.sigma0_pos, 0.030707),
                                ("pose", cal.sigma0_pose, 0.008869)):
            rel = abs(got - want) / want
            self.assertLess(rel, 0.01,
                            f"sigma0_{name} 复现失败：{got:.6f} vs {want:.6f} "
                            f"(相对 {rel:.1%})")

    def test_heldout_activation_fraction_across_seeds(self):
        """**留出集**（与标定不同的 seed）上的激活比例必须贴近设计的 10%。"""
        fr_pos, fr_pose, tails = [], [], []
        for seed in (101, 202, 303, 404, 505):
            v = self.ik.validate_sigma0(1500, seed=seed, distribution="mixed",
                                        q_center=Q_HOME)
            fr_pos.append(v["frac_pos"])
            fr_pose.append(v["frac_pose"])
            tails.append((v["min_pos"], v["min_pose"]))
        for name, fr in (("位置", fr_pos), ("位姿", fr_pose)):
            self.assertGreater(min(fr), 0.06,
                               f"{name}留出激活比例过低 {fr}")
            self.assertLess(max(fr), 0.16,
                            f"{name}留出激活比例过高 {fr}")
            self.assertLess(max(fr) - min(fr), 0.06,
                            f"{name}多 seed 波动过大 {fr}")
        # 尾部必须真的存在（否则标定集里根本没有近奇异样本）
        self.assertLess(min(t[1] for t in tails), 0.005,
                        f"留出集里没有近奇异尾部：{tails}")

    def test_activation_fraction_is_distribution_dependent(self):
        """同一组阈值换采样分布后激活比例会差一个数量级——阈值不是普适常数。"""
        fr = {}
        for dist in ("uniform", "home", "singular"):
            v = self.ik.validate_sigma0(1200, seed=909, distribution=dist,
                                        q_center=Q_HOME)
            fr[dist] = (v["frac_pos"], v["frac_pose"])
        self.assertLess(fr["home"][0], 0.05, f"home 分布激活比例 {fr}")
        self.assertGreater(fr["singular"][0], 0.15, f"singular 分布激活比例 {fr}")
        self.assertGreater(fr["singular"][0] / max(fr["home"][0], 1e-6), 5.0,
                           f"分布依赖性没有体现出来：{fr}")

    def test_singular_sampler_really_reaches_near_singular_region(self):
        s3, s6 = self.ik.sample_sigma_min(600, seed=17, distribution="singular")
        self.assertLess(float(np.median(s6)), 0.05,
                        f"singular 分布的位姿 σ_min 中位数 {np.median(s6):.4f} 不够小")
        self.assertLess(float(np.min(s6)), 2e-3)

    def test_pose_sigma_min_never_exceeds_position(self):
        """结构性关系：6×n 含 3×n 的全部行，最小奇异值不可能更大。

        独立证明：σ_min(A) = min_{‖x‖=1} ‖Ax‖。把 A 的行集合扩大只会让 ‖Ax‖ 变大，
        所以 σ_min 是**行集合的单调增函数**……在列满秩方向上取 min 得到
        σ_min(6×n) ≥ σ_min(3×n)？不对——这里比较的是 min over x of ‖A x‖，
        6 行的 ‖A₆x‖² = ‖A₃x‖² + ‖A_rotx‖² ≥ ‖A₃x‖²，故 σ_min(6×n) ≥ σ_min(3×n)。
        但对 n > rows 的**宽矩阵**，σ_min 定义为最小的非零奇异值（第 rows 个），
        位置任务只取 3 个而位姿取 6 个，序号更靠后，因此 σ_min(6×n) ≤ σ_min(3×n)。
        本用例用数值验证后一结论。
        """
        s3, s6 = self.ik.sample_sigma_min(800, seed=7, distribution="mixed",
                                          q_center=Q_HOME)
        self.assertTrue(bool(np.all(s6 <= s3 + 1e-12)))

    def test_thresholds_are_separately_calibrated(self):
        self.assertGreater(self.ik.sigma0_pos, self.ik.sigma0_pose * 2.0,
                           "两个阈值太接近，分别标定没有意义")

    def test_char_length_monotonically_moves_the_thresholds(self):
        """阈值依赖任务权重 W，不是机器人常数。

        独立推导：σ_k(WJ)² = λ_k(Jᵀ W² J) = λ_k(J_posᵀJ_pos + L²·J_rotᵀJ_rot)。
        L 增大只是往 PSD 序里加一个半正定项，由 Weyl 单调性定理，**每一个**特征值
        都单调不减，因此 σ_min(WJ) 与它的任何分位数都随 L 单调不减。
        """
        robot = self.robot
        rng = np.random.default_rng(21)
        Ls = [0.05, 0.2, 0.8577, 3.0]
        # 1) 逐构型验证 Weyl 单调性（纯数学，不依赖标定）
        for _ in range(50):
            q = rng.uniform(robot.q_lower, robot.q_upper)
            J = robot.jacobian(q)
            prev = -np.inf
            for L in Ls:
                W = np.diag([1.0, 1.0, 1.0, L, L, L])
                s = float(np.linalg.svd(W @ J, compute_uv=False)[-1])
                self.assertGreaterEqual(s, prev - 1e-12, "σ_min 随 L 不单调")
                prev = s
        # 2) 标定阈值也必须随之单调移动，且量级变化明显
        vals = []
        for L in Ls:
            ik = DLSInverseKinematics(robot, method="adaptive", char_length=L)
            cal = ik.calibrate_sigma0(samples=1200, seed=7, distribution="mixed",
                                      q_center=Q_HOME, apply=False)
            vals.append(cal.sigma0_pose)
        for a, b in zip(vals, vals[1:]):
            self.assertGreaterEqual(b, a - 1e-12, f"sigma0_pose 随 L 不单调：{vals}")
        self.assertGreater(vals[-1] / vals[0], 1.3,
                           f"L 从 {Ls[0]} 到 {Ls[-1]} 阈值几乎没变：{vals}")
        robot.set_state(np.zeros(robot.n), np.zeros(robot.n))


# ------------------------------------------------------------------ 零空间


class TestNullSpaceLeakage(unittest.TestCase):
    """在**真实 Panda 3×7 / 6×7 雅可比**上验证泄漏公式与两种投影器的行为。"""

    def _cases(self):
        robot = panda()
        ik = DLSInverseKinematics(robot)
        q_sing, _ = ik._stretched_config()
        for task, rows in ((POSE_TASK, 6), (POSITION_TASK, 3)):
            for name, q in (("home", Q_HOME), ("near-singular", q_sing)):
                J = robot.jacobian(q)
                J = J if rows == 6 else J[:3, :]
                yield f"{task}/{name}", ik, ik._normalize_jacobian(J, task), rows

    def test_leakage_formula_on_real_panda_jacobians(self):
        """独立推导：J = UΣVᵀ ⇒ J⁺_λJ = V diag(σᵢ²/(σᵢ²+λ²)) Vᵀ，故

            (I − J⁺_λJ)vᵢ = λ²/(σᵢ²+λ²) vᵢ,   J(I − J⁺_λJ)vᵢ = σᵢλ²/(σᵢ²+λ²) uᵢ
        """
        for name, ik, J, rows in self._cases():
            with self.subTest(case=name):
                U, s, Vt = np.linalg.svd(J, full_matrices=True)
                for lam in (0.01, 0.05, 0.2):
                    Jp = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(rows),
                                               np.eye(rows))
                    N = np.eye(J.shape[1]) - Jp @ J
                    for i in range(len(s)):
                        v = Vt[i]
                        pred = lam**2 / (s[i] ** 2 + lam**2)
                        self.assertAlmostEqual(float(np.linalg.norm(N @ v)), pred,
                                               places=10)
                        self.assertAlmostEqual(float(np.linalg.norm(J @ (N @ v))),
                                               s[i] * pred, places=10)

    def test_damped_projector_leaks_order_one_near_singular(self):
        robot = panda()
        ik = DLSInverseKinematics(robot)
        q_sing, _ = ik._stretched_config()
        J = ik._normalize_jacobian(robot.jacobian(q_sing), POSE_TASK)
        s = np.linalg.svd(J, compute_uv=False)
        lam = 0.05
        Jp = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), np.eye(6))
        N = np.eye(robot.n) - Jp @ J
        v_min = np.linalg.svd(J, full_matrices=True)[2][len(s) - 1]
        leak = lam**2 / (s[-1] ** 2 + lam**2)
        self.assertGreater(leak, 0.9,
                           f"σ_min={s[-1]:.5f} λ={lam} 时相对泄漏仅 {leak:.3f}")
        self.assertGreater(float(np.linalg.norm(N @ v_min)), 0.9)

    def test_svd_projector_is_exact_on_real_jacobians(self):
        for name, ik, J, rows in self._cases():
            with self.subTest(case=name):
                sv = np.linalg.svd(J, compute_uv=False)
                r = int(np.sum(sv > ik.rcond * sv[0]))
                Vr = np.linalg.svd(J, full_matrices=True)[2][:r].T
                N = np.eye(J.shape[1]) - Vr @ Vr.T
                g = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0])
                self.assertLess(float(np.linalg.norm(J @ (N @ g))), 1e-10)

    def test_null_gain_actually_disturbs_the_task_with_damped_projector(self):
        """null_gain=0 不能当作"功能正确"的证据。这里用 1.0 与 4.0 实测任务扰动。"""
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        prev = -1.0
        for gain in (1.0, 4.0):
            ik = DLSInverseKinematics(robot, method="fixed", lam0=0.05,
                                      null_gain=gain, null_projector="damped")
            res = ik.solve(p0 + np.array([0.05, 0.0, 0.0]), R0, Q_HOME.copy(),
                           max_iters=30, dual_branch=False)
            worst = max(res.null_leak_trace)
            self.assertGreater(worst, 1e-4,
                               f"null_gain={gain} 的阻尼投影没有产生任务空间扰动")
            self.assertGreater(worst, prev)
            prev = worst

    def test_default_projector_is_the_exact_one(self):
        """缺省必须是精确投影。（mutation 发现的缺口：之前所有用例都显式指定
        null_projector，把缺省改成有泄漏的 "damped" 竟然没有任何用例失败。）"""
        robot = panda()
        ik = DLSInverseKinematics(robot, method="fixed", lam0=0.05, null_gain=1.0)
        self.assertEqual(ik.null_projector, "svd")
        p0, R0 = robot.fk(Q_HOME)
        res = ik.solve(p0 + np.array([0.05, 0.0, 0.0]), R0, Q_HOME.copy(),
                       max_iters=30, dual_branch=False)
        self.assertGreater(len(res.null_leak_trace), 0)
        self.assertLess(max(res.null_leak_trace), 1e-9,
                        "缺省零空间投影存在泄漏")

    def test_svd_projector_keeps_task_undisturbed_at_the_same_gain(self):
        robot = panda()
        p0, R0 = robot.fk(Q_HOME)
        for gain in (1.0, 4.0):
            ik = DLSInverseKinematics(robot, method="fixed", lam0=0.05,
                                      null_gain=gain, null_projector="svd")
            res = ik.solve(p0 + np.array([0.05, 0.0, 0.0]), R0, Q_HOME.copy(),
                           max_iters=30, dual_branch=False)
            self.assertLess(max(res.null_leak_trace), 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
