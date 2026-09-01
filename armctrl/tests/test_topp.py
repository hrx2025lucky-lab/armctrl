"""圆弧倒角与 TOPP-RA 时间参数化的数学不变量测试。

oracle 独立性
-------------
* 倒角的偏离量用**密采样实测**（圆弧上离拐点最近的点到拐点的距离）核对，
  不读实现里算好的 deviation 字段之外的任何中间量；
* 切线连续性用**自己推的曲率上界**判定：圆弧半径 r、采样步长 ds 时，
  相邻切线夹角恰为 ds/r。折线在拐角处会给出几十度，倒角后只有 ds/r，
  两者相差两个数量级，测试有明确区分力（见
  test_tangent_turn_rate_is_bounded_by_curvature 里对折线的反向断言）；
* 速度/加速度上限用 **Lipschitz 判据**检验：函数导数有界 M ⟺ 它是 M-Lipschitz。
  只用轨迹采样出的 q、q̇，不看 toppra 内部的 ṡ 网格或降额系数。

运行：
    export PYTHONPATH=.
    python -m unittest armctrl.tests.test_topp
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.planning.blend import BlendedPath, blend_corners, blend_and_verify
from armctrl.planning.topp import ToppraTrajectory, analyze
from armctrl.planning.path_timing import JointPathTrajectory
from armctrl.planning.rrt import path_length


def _corner_case(theta: float, l1: float = 1.0, l2: float = 1.0):
    """构造一个转角恰为 theta 的三点折线（放在 xy 平面里便于手算）。"""
    q0 = np.zeros(3)
    q1 = q0 + np.array([l1, 0.0, 0.0])
    q2 = q1 + l2 * np.array([np.cos(theta), np.sin(theta), 0.0])
    return [q0, q1, q2]


def _min_dist_to_samples(p: np.ndarray, Q: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(Q - p, axis=1)))


def _adjacent_tangent_angles(path: BlendedPath, n: int):
    """密采样后相邻弦方向之间的夹角（弧度）与采样步长。

    用**位置差分**得到方向，不调用 path.tangent()，
    这样即使 tangent() 实现错了也测得出来。
    """
    ss, Q = path.sample(n)
    T = np.diff(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)[:, :]
    cosang = np.clip(np.sum(T[:-1] * T[1:], axis=1), -1.0, 1.0)
    return np.arccos(cosang), path.length / (n - 1)


# ============================================================ 圆弧倒角


class TestCircularBlend(unittest.TestCase):

    def test_deviation_matches_independent_measurement(self):
        """实现声称的偏离量必须等于圆弧到拐点的实测最近距离。

        推导：δ = L·(1 − cos(θ/2)) / sin(θ/2)。
        这里不验证公式本身，而是验证"实现算出来的 δ"和"几何上真的偏了多少"一致。
        """
        worst = 0.0
        for deg in (10, 30, 45, 60, 90, 120, 150):
            theta = np.radians(deg)
            path = blend_corners(_corner_case(theta), max_deviation=0.05)
            _, Q = path.sample(40001)
            measured = _min_dist_to_samples(np.array([1.0, 0.0, 0.0]), Q)
            claimed = path.corners[0]["deviation"]
            worst = max(worst, abs(measured - claimed))
        self.assertLess(worst, 1e-6, f"最大偏差 {worst:.3e}")

    def test_deviation_never_exceeds_budget(self):
        budget = 0.04
        for deg in (5, 20, 45, 90, 135, 165):
            path = blend_corners(_corner_case(np.radians(deg)), max_deviation=budget)
            _, Q = path.sample(20001)
            measured = _min_dist_to_samples(np.array([1.0, 0.0, 0.0]), Q)
            self.assertLessEqual(measured, budget + 1e-9,
                                 f"θ={deg}° 实测偏离 {measured:.6f} 超过预算")

    def test_uses_the_full_deviation_budget_when_not_capped(self):
        """没被线段长度截断时，实际偏离必须**恰好等于**预算，而不只是不超。

        倒角的意义就是拿偏离换速度：只用了预算的一小部分，圆弧半径就偏小，
        拐角还是得减速。只断言"≤ 预算"会放过这类"过于保守"的实现缺陷。

        取 θ=60°/90°/120°、线段长 1.0，此时
        L = δ·sin(θ/2)/(1−cos(θ/2)) 分别约为 3.73δ / 2.41δ / 1.73δ，
        在 δ=0.05 下都远小于线段一半 0.5，因此不会被截断。
        """
        budget = 0.05
        for deg in (60, 90, 120):
            path = blend_corners(_corner_case(np.radians(deg), 1.0, 1.0),
                                 max_deviation=budget)
            c = path.corners[0]
            self.assertTrue(c["blended"])
            self.assertLess(c["L"], 0.5, f"θ={deg}° 被线段长度截断了，前提不成立")
            _, Q = path.sample(40001)
            measured = _min_dist_to_samples(np.array([1.0, 0.0, 0.0]), Q)
            self.assertAlmostEqual(measured, budget, delta=1e-6,
                                   msg=f"θ={deg}° 只用了预算的 "
                                       f"{measured/budget*100:.1f}%")

    def test_cut_back_never_exceeds_half_segment(self):
        """退回距离必须受两侧线段各自一半的限制，否则相邻拐角的圆弧会重叠。"""
        pts = [np.zeros(3), np.array([0.2, 0.0, 0.0]),
               np.array([0.2, 0.2, 0.0]), np.array([0.2, 0.2, 0.2])]
        path = blend_corners(pts, max_deviation=1.0)      # 预算大到必然被截断
        for i, c in enumerate(path.corners):
            self.assertLessEqual(c["L"], 0.1 + 1e-12,
                                 f"第 {i} 个拐角退回 {c['L']:.4f} 超过线段一半")

    def test_tangent_turn_rate_is_bounded_by_curvature(self):
        """倒角后路径是 C¹ 的：相邻弦方向夹角 ≤ ds/r_min。

        独立推导：半径为 r 的圆弧，走过弧长 ds 时切线转过 ds/r。
        直线段转 0。所以整条路径上相邻切线夹角的上界就是 ds / r_min。

        同时反向断言原折线**通不过**这条判据（拐角处一次转过 90°），
        证明这个测试确实在检验倒角带来的连续性，而不是恒成立。
        """
        pts = [np.zeros(3), np.array([1.0, 0.0, 0.0]),
               np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0])]
        blended = blend_corners(pts, max_deviation=0.08)
        angles, ds = _adjacent_tangent_angles(blended, 4001)
        r_min = min(c["radius"] for c in blended.corners if c["blended"])
        bound = ds / r_min
        self.assertLessEqual(float(angles.max()), bound * 1.01,
                             f"实测 {angles.max():.5f} rad 超过曲率上界 {bound:.5f}")

        polyline = blend_corners(pts, max_deviation=0.0)
        angles_p, _ = _adjacent_tangent_angles(polyline, 4001)
        self.assertGreater(float(angles_p.max()), 1.0,
                           "折线在拐角处本应有大角度跳变，测试失去区分力")

    def test_endpoints_are_preserved_exactly(self):
        pts = [np.zeros(4), np.array([1.0, 0.5, 0.0, 0.0]),
               np.array([1.0, 1.5, 0.7, 0.0]), np.array([0.2, 1.5, 0.7, 0.9])]
        path = blend_corners(pts, max_deviation=0.07)
        np.testing.assert_allclose(path(0.0), pts[0], atol=1e-12)
        np.testing.assert_allclose(path(path.length), pts[-1], atol=1e-12)

    def test_blending_never_lengthens_the_path(self):
        """圆弧是抄近道，总长只会变短或不变。"""
        rng = np.random.default_rng(3)
        for _ in range(10):
            pts = [rng.uniform(-1, 1, size=5) for _ in range(5)]
            raw = path_length(pts)
            path = blend_corners(pts, max_deviation=0.05)
            self.assertLessEqual(path.length, raw + 1e-9)

    def test_arc_lies_in_the_plane_of_its_two_segments(self):
        """圆弧必须落在两段方向张成的平面内，不能拐到第三个方向去。

        独立检验：取与 u₁、u₂ 都正交的方向 w，圆弧上所有点在 w 上的投影
        必须与拐点相同。
        """
        theta = np.radians(70)
        pts = _corner_case(theta)
        u1 = np.array([1.0, 0.0, 0.0])
        u2 = np.array([np.cos(theta), np.sin(theta), 0.0])
        w = np.cross(u1, u2)
        w /= np.linalg.norm(w)
        path = blend_corners(pts, max_deviation=0.06)
        _, Q = path.sample(2001)
        proj = Q @ w - float(pts[1] @ w)
        self.assertLess(float(np.abs(proj).max()), 1e-12)

    def test_zero_deviation_reproduces_the_polyline(self):
        pts = [np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
        path = blend_corners(pts, max_deviation=0.0)
        self.assertEqual(path.n_blended, 0)
        self.assertAlmostEqual(path.length, path_length(pts), places=12)
        np.testing.assert_allclose(path(path.length / 2), pts[1], atol=1e-12)

    def test_near_reversal_is_left_as_a_sharp_corner(self):
        """接近 180° 的原路返回无法倒角：r = L/tan(θ/2) → 0，圆弧退化成点。

        实现必须如实保留尖角并标记未倒角，而不是硬算出一个半径极小的圆弧。
        """
        path = blend_corners(_corner_case(np.radians(179.5)), max_deviation=0.05)
        self.assertEqual(path.n_blended, 0)
        self.assertFalse(path.corners[0]["blended"])

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            blend_corners([np.zeros(3)], max_deviation=0.05)
        with self.assertRaises(ValueError):
            blend_corners([np.zeros(3), np.zeros(3)], max_deviation=0.05)


class _FakeChecker:
    """只允许离某个球心足够远的构型，用来测 blend_and_verify 的退让逻辑。"""

    def __init__(self, center, radius, margin):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.safety_margin = float(margin)

    def clearances(self, q):
        d = float(np.linalg.norm(np.asarray(q, dtype=float) - self.center)) - self.radius
        return d, 10.0


class TestBlendAndVerify(unittest.TestCase):

    def test_keeps_full_budget_when_there_is_room(self):
        pts = [np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
        ck = _FakeChecker([5.0, 5.0, 5.0], 0.1, 0.03)      # 障碍物在很远的地方
        path, dev, clear = blend_and_verify(pts, ck, max_deviation=0.08)
        self.assertAlmostEqual(dev, 0.08, places=12)
        self.assertGreater(path.n_blended, 0)
        self.assertGreater(clear, ck.safety_margin)

    def test_backs_off_when_the_full_budget_would_collide(self):
        """障碍物贴在转弯内侧时，倒角必须自动缩小偏离预算。

        圆弧总是往内侧切，这正是"原路径无碰撞不蕴含倒角后无碰撞"的原因。
        夹具几何（都由解析球-点距离独立算出）：
            折线最小间隙          +0.050  合法
            δ=0.04 倒角后         +0.039  合法
            δ=0.08 倒角后         −0.001  非法 ← 满预算会撞
        所以正确行为是退到 0.04 一档，仍然倒角、仍然合法。
        """
        pts = [np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
        ck = _FakeChecker([0.93, 0.07, 0.0], 0.02, 0.03)
        raw = blend_corners(pts, max_deviation=0.0)
        _, Q = raw.sample(8001)
        self.assertGreater(min(ck.clearances(q)[0] for q in Q), ck.safety_margin,
                           "前置条件不成立：折线本身就该是合法的")
        full = blend_corners(pts, max_deviation=0.08)
        _, Qf = full.sample(8001)
        self.assertLess(min(ck.clearances(q)[0] for q in Qf), ck.safety_margin,
                        "前置条件不成立：满预算倒角本该撞上")

        path, dev, clear = blend_and_verify(pts, ck, max_deviation=0.08, attempts=4)
        self.assertLess(dev, 0.08)
        self.assertGreater(dev, 0.0)
        self.assertEqual(path.n_blended, 1)
        self.assertGreater(clear, ck.safety_margin)

    def test_falls_back_to_the_polyline_when_attempts_are_exhausted(self):
        """退让次数用尽仍不合法时，必须退回原折线而不是交出一条会撞的路径。"""
        pts = [np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
        ck = _FakeChecker([0.93, 0.07, 0.0], 0.02, 0.03)
        path, dev, clear = blend_and_verify(pts, ck, max_deviation=0.08, attempts=1)
        self.assertEqual(dev, 0.0)
        self.assertEqual(path.n_blended, 0)
        self.assertGreater(clear, ck.safety_margin)


# ============================================================ TOPP-RA


class TestToppraTrajectory(unittest.TestCase):

    V = np.array([1.0, 0.8, 1.2])
    A = np.array([2.0, 1.5, 3.0])
    PTS = [np.zeros(3), np.array([1.0, 0.0, 0.0]),
           np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0])]

    @classmethod
    def setUpClass(cls):
        cls.blended = blend_corners(cls.PTS, max_deviation=0.08)
        cls.traj = ToppraTrajectory(cls.blended, cls.V, cls.A, n_grid=1200)

    def test_reports_that_limits_are_satisfied(self):
        self.assertTrue(self.traj.limits_satisfied)
        self.assertLessEqual(self.traj.scale, 1.0 + 1e-12)
        self.assertGreater(self.traj.scale, 0.5,
                           "降额过深说明求解或验证环节有问题")

    def test_velocity_bound_via_lipschitz(self):
        """|q̇| ≤ v_max 等价于 q 是 v_max-Lipschitz 的。只用采样出的 q。"""
        _, Q, _, _ = self.traj.sample(1e-3)
        ratio = np.max(np.abs(np.diff(Q, axis=0)) / 1e-3 / self.V)
        self.assertLessEqual(ratio, 1.0 + 1e-3, f"实测比值 {ratio:.5f}")

    def test_acceleration_bound_via_lipschitz(self):
        _, _, Vv, _ = self.traj.sample(1e-3)
        ratio = np.max(np.abs(np.diff(Vv, axis=0)) / 1e-3 / self.A)
        self.assertLessEqual(ratio, 1.0 + 1e-3, f"实测比值 {ratio:.5f}")

    def test_endpoints_and_boundary_velocity(self):
        q0, v0, _ = self.traj(0.0)
        qT, vT, _ = self.traj(self.traj.duration)
        np.testing.assert_allclose(q0, self.PTS[0], atol=1e-5)
        np.testing.assert_allclose(qT, self.PTS[-1], atol=1e-5)
        self.assertLess(float(np.linalg.norm(v0)), 5e-3)
        self.assertLess(float(np.linalg.norm(vT)), 5e-3)

    def test_samples_stay_on_the_geometric_path(self):
        """时间参数化只能改变"什么时候到哪"，不能改变"走哪条线"。

        独立 oracle：把几何路径密采样，逐点求轨迹采样到它的最近距离。
        容差 1e-3 rad 来自样条插值对倒角路径的拟合误差（实测约 7e-5）。
        """
        _, Q_path = self.blended.sample(20001)
        _, Q_traj, _, _ = self.traj.sample(5e-3)
        worst = max(_min_dist_to_samples(q, Q_path) for q in Q_traj)
        self.assertLess(worst, 1e-3, f"最大偏离 {worst:.3e} rad")

    def test_progress_along_the_path_is_monotone(self):
        """时间最优解不应该走回头路：沿路径的进度必须单调不减。"""
        _, Q_path = self.blended.sample(4001)
        _, Q_traj, _, _ = self.traj.sample(1e-2)
        idx = [int(np.argmin(np.linalg.norm(Q_path - q, axis=1))) for q in Q_traj]
        self.assertTrue(all(b >= a - 2 for a, b in zip(idx[:-1], idx[1:])),
                        "轨迹在几何路径上出现了倒退")

    def test_blending_is_what_makes_toppra_fast(self):
        """核心论点：光换时间律不够，必须先把拐角倒圆。

        折线上做 TOPP-RA，拐角处曲率极大（离散化后近似成很小的半径），
        速度被压得很低；倒角之后才真正快起来。
        """
        polyline = blend_corners(self.PTS, max_deviation=0.0)
        t_poly = ToppraTrajectory(polyline, self.V, self.A, n_grid=1200)
        self.assertLess(self.traj.duration, t_poly.duration,
                        f"倒角 {self.traj.duration:.3f}s 并不快于折线 "
                        f"{t_poly.duration:.3f}s")

    def test_faster_than_per_segment_scurve(self):
        """逐段 S 曲线要在每个路径点停一次，时间最优参数化应当更快。"""
        s = JointPathTrajectory(self.PTS, self.V, self.A, np.full(3, 20.0))
        self.assertLess(self.traj.duration, s.duration)
        rep_s = analyze(s, "s", path_length(self.PTS))
        rep_t = analyze(self.traj, "t", self.blended.length)
        self.assertGreater(rep_s.stops, 0, "S 曲线本应在路径点处停顿")
        self.assertEqual(rep_t.stops, 0, "TOPP-RA 不应有中途停顿")

    def test_jerk_is_unbounded_unlike_the_scurve(self):
        """诚实记录代价：TOPP-RA 的加速度是 bang-bang 的，jerk 远大于 S 曲线。

        这条测试存在的意义是防止把"时间更短"写成"全面更好"。
        """
        s = JointPathTrajectory(self.PTS, self.V, self.A, np.full(3, 20.0))
        rep_s = analyze(s, "s", path_length(self.PTS))
        rep_t = analyze(self.traj, "t", self.blended.length)
        self.assertLessEqual(rep_s.max_jerk, 20.0 * 1.05)
        self.assertGreater(rep_t.max_jerk, 20.0 * 5)

    def test_rejects_bad_limits(self):
        with self.assertRaises(ValueError):
            ToppraTrajectory(self.blended, np.zeros(3), self.A)
        with self.assertRaises(ValueError):
            ToppraTrajectory(self.blended, self.V, -self.A)


if __name__ == "__main__":
    unittest.main()
