"""路径规划模块的数学不变量测试。

oracle 独立性
-------------
本文件里所有"正确答案"都不来自被测实现，也不来自 MuJoCo 的碰撞例程：

* 平面两连杆的正运动学由连杆链条**手工推导**（见 _planar_fk 的注释），
  不调用 mj_kinematics；
* 点-线段距离、点-长方体有符号距离都是**自己写的闭式解**；
* 轨迹的速度/加速度/jerk 上限用**Lipschitz 常数**检验：
  一个函数导数有界 M ⟺ 它是 M-Lipschitz 的。
  只用采样出来的 q(t)、q̇(t)、q̈(t) 做差分，不看实现内部的 ṡ、Tj、Ta。

这样即使实现和 MuJoCo 同时错，测试也不会跟着错。

运行：
    export PYTHONPATH=.
    python -m unittest discover -s armctrl/tests -t .
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from armctrl.planning.collision import CollisionChecker
from armctrl.planning.rrt import (
    RRTConnect, RRTConnectConfig, path_length, shortcut_smooth, prune_waypoints,
)
from armctrl.planning.path_timing import JointPathTrajectory


# ============================================================ 独立解析 oracle


def _seg_point_distance(c: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """点 c 到线段 pq 的欧氏距离（自己推的闭式解）。

    线段上的点写成 p + t(q−p)，t ∈ [0,1]。
    ‖c − p − t(q−p)‖² 对 t 求导置零得 t* = (c−p)·(q−p) / ‖q−p‖²，
    这是无约束最小值；因为目标是 t 的开口向上抛物线，
    在 [0,1] 上的最小值就是 t* 截断到 [0,1]。
    """
    d = q - p
    L2 = float(d @ d)
    if L2 < 1e-30:
        return float(np.linalg.norm(c - p))
    t = float(np.clip((c - p) @ d / L2, 0.0, 1.0))
    return float(np.linalg.norm(c - (p + t * d)))


def _box_signed_distance(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    """点到轴对齐长方体的有符号距离（自己推的闭式解）。

    令 δ = |p − center| − half（逐轴）。
      · 若某轴 δ > 0，该轴方向已经在盒外，超出量就是 δ；
        外部距离 = ‖max(δ, 0)‖₂。
      · 若三轴 δ 都 ≤ 0，点在盒内，到最近面的距离是 −max(δ)，
        取负号表示内部：inside = min(max(δ), 0)。
    两项相加即有符号距离（内部时 outside = 0，外部时 inside = 0）。
    """
    d = np.abs(p - center) - half
    outside = float(np.linalg.norm(np.maximum(d, 0.0)))
    inside = float(min(np.max(d), 0.0))
    return outside + inside


def _planar_fk(q1: float, q2: float, L1: float, L2: float):
    """平面两连杆的连杆轴线端点（手工推导，不调用 MuJoCo）。

    连杆 1 绕世界系 z 轴转 q1，轴线从原点 O 指向体坐标 (L1,0,0)，
    世界系下即 A = L1·(cos q1, sin q1, 0)。
    连杆 2 的基座固定在连杆 1 末端，再绕 z 轴相对转 q2，
    因此它的世界姿态是 R_z(q1+q2)，轴线从 A 指向
    B = A + L2·(cos(q1+q2), sin(q1+q2), 0)。
    """
    O = np.zeros(3)
    A = np.array([L1 * np.cos(q1), L1 * np.sin(q1), 0.0])
    ang = q1 + q2
    B = A + np.array([L2 * np.cos(ang), L2 * np.sin(ang), 0.0])
    return O, A, B


# ============================================================ 测试用模型


L1 = L2 = 0.6
LINK_RADIUS = 0.05
_ARM_OBSTACLES = [                       # (名字, 球心, 半径)
    ("ball_a", np.array([0.75, 0.45, 0.0]), 0.18),
    ("ball_b", np.array([0.20, -0.80, 0.0]), 0.22),
]


def _build_planar_arm():
    """平面两连杆（胶囊连杆）+ 两个球障碍物。"""
    obs = "".join(
        f'<geom name="{n}" type="sphere" size="{r}" pos="{c[0]} {c[1]} {c[2]}"/>'
        for n, c, r in _ARM_OBSTACLES
    )
    xml = f"""<mujoco model="planar2">
      <worldbody>
        <body name="l1" pos="0 0 0">
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
          <geom name="g1" type="capsule" fromto="0 0 0 {L1} 0 0" size="{LINK_RADIUS}"/>
          <body name="l2" pos="{L1} 0 0">
            <joint name="j2" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
            <geom name="g2" type="capsule" fromto="0 0 0 {L2} 0 0" size="{LINK_RADIUS}"/>
          </body>
        </body>
        {obs}
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)

    def gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    return model, [gid("g1"), gid("g2")], [gid(n) for n, _, _ in _ARM_OBSTACLES]


def _arm_clearance_analytic(q: np.ndarray) -> float:
    """两连杆构型下的真实最小间隙（全部由解析式算出）。"""
    O, A, B = _planar_fk(q[0], q[1], L1, L2)
    best = np.inf
    for _, c, r in _ARM_OBSTACLES:
        for p, s in ((O, A), (A, B)):
            best = min(best, _seg_point_distance(c, p, s) - LINK_RADIUS - r)
    return float(best)


# 点机器人（三个移动关节）：构型空间与工作空间重合，几何完全可解析
PT_RADIUS = 0.02
_WALL_WITH_GAP = [                       # (名字, 中心, 半边长)
    ("w_top", np.array([0.0, 0.0, 0.575]), np.array([0.02, 1.0, 0.425])),
    ("w_bot", np.array([0.0, 0.0, -0.575]), np.array([0.02, 1.0, 0.425])),
    ("w_left", np.array([0.0, -0.275, 0.0]), np.array([0.02, 0.725, 0.15])),
    ("w_right", np.array([0.0, 0.975, 0.0]), np.array([0.02, 0.025, 0.15])),
]
_WALL_SEALED = [("w_all", np.zeros(3), np.array([0.02, 1.0, 1.0]))]


def _build_point_robot(walls):
    obs = "".join(
        f'<geom name="{n}" type="box" size="{h[0]} {h[1]} {h[2]}" '
        f'pos="{c[0]} {c[1]} {c[2]}"/>'
        for n, c, h in walls
    )
    xml = f"""<mujoco model="point3">
      <worldbody>
        <body name="pt" pos="0 0 0">
          <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
          <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
          <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
          <geom name="pt" type="sphere" size="{PT_RADIUS}"/>
        </body>
        {obs}
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)

    def gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    return model, [gid("pt")], [gid(n) for n, _, _ in walls]


def _point_clearance_analytic(p: np.ndarray, walls) -> float:
    """点机器人构型 p 的真实最小间隙（解析）。"""
    return float(min(_box_signed_distance(p, c, h) - PT_RADIUS for _, c, h in walls))


def _point_to_polyline(p: np.ndarray, path) -> float:
    """点到折线的距离，用于验证轨迹确实落在几何路径上。"""
    return float(min(_seg_point_distance(p, np.asarray(a, float), np.asarray(b, float))
                     for a, b in zip(path[:-1], path[1:])))


# ============================================================ 碰撞检测


class TestCollisionChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.robot, cls.obs = _build_planar_arm()

    def _checker(self, margin=0.0, distmax=1.0, resolution=0.02):
        return CollisionChecker(self.model, [0, 1], self.robot, self.obs,
                                safety_margin=margin, distmax=distmax,
                                resolution=resolution)

    def test_clearance_matches_analytic_capsule_sphere(self):
        """有符号间隙必须与手推的线段-球距离一致。

        这是整套规划的地基：间隙算错，后面所有"无碰撞"的结论都是空的。
        """
        ck = self._checker(distmax=10.0)
        rng = np.random.default_rng(20260825)
        worst = 0.0
        for _ in range(300):
            q = rng.uniform(-np.pi, np.pi, size=2)
            worst = max(worst, abs(ck.clearance(q) - _arm_clearance_analytic(q)))
        self.assertLess(worst, 1e-9, f"最大偏差 {worst:.3e}")

    def test_is_valid_is_exactly_clearance_above_margin(self):
        """合法性判据必须就是"解析间隙 > 安全裕度"，不多不少。"""
        margin = 0.03
        ck = self._checker(margin=margin, distmax=10.0)
        rng = np.random.default_rng(7)
        n_true = 0
        for _ in range(300):
            q = rng.uniform(-np.pi, np.pi, size=2)
            truth = _arm_clearance_analytic(q)
            if abs(truth - margin) < 1e-9:      # 恰在边界，浮点不可判，跳过
                continue
            expect = truth > margin
            n_true += int(expect)
            self.assertEqual(ck.is_valid(q), expect,
                             f"q={q}, 解析间隙={truth:.6f}, 裕度={margin}")
        self.assertGreater(n_true, 30, "样本几乎全是合法/全是碰撞，测试没有区分力")

    def test_clearance_is_truncated_at_distmax(self):
        """distmax 是截断值：远离障碍时返回的是 distmax 而不是真实距离。

        这是必须写进测试的行为约定，否则调用方会把截断值当成真实间隙报告。
        """
        dmax = 0.05
        ck = self._checker(distmax=dmax)
        q = np.array([np.pi / 2, 0.0])           # 手臂指向 +y，远离两个球
        self.assertGreater(_arm_clearance_analytic(q), dmax)
        self.assertAlmostEqual(ck.clearance(q), dmax, places=12)

    def test_segment_check_rejects_line_through_obstacle(self):
        """两端合法但中间穿过障碍的直线段必须被判为非法。

        端点检查通不过这一关——这正是逐段采样存在的理由。
        """
        ck = self._checker(margin=0.0, distmax=10.0, resolution=0.01)
        qa = np.array([1.2, 0.0])
        qb = np.array([-0.2, 0.0])
        self.assertGreater(_arm_clearance_analytic(qa), 0.0)
        self.assertGreater(_arm_clearance_analytic(qb), 0.0)
        mids = [qa + t * (qb - qa) for t in np.linspace(0, 1, 401)]
        self.assertTrue(any(_arm_clearance_analytic(q) <= 0.0 for q in mids),
                        "构造的样例本身没有中途碰撞，测试无效")
        self.assertFalse(ck.segment_valid(qa, qb))

    def test_segment_check_detects_obstacle_away_from_midpoint(self):
        """碰撞不在中点时也必须查出来——段检查要覆盖**全部**采样点。

        点机器人沿 x 从 −1 走到 +1，在 x = −0.7（即 t = 0.15）放一片薄板。
        若段检查只查中点 t = 0.5（x = 0，空的），就会漏掉这次碰撞。
        薄板半厚 0.01 m，加点半径 0.02 与裕度 0.01 共约 0.04 m 宽的非法区间，
        而采样步长 0.02 m，必然有采样点落进去。
        """
        xml = """<mujoco><worldbody>
          <body name="pt" pos="0 0 0">
            <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
            <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
            <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
            <geom name="pt" type="sphere" size="0.02"/>
          </body>
          <geom name="slab" type="box" size="0.01 1 1" pos="-0.7 0 0"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        ck = CollisionChecker(model, [0, 1, 2], [gid("pt")], [gid("slab")],
                              safety_margin=0.01, distmax=0.3, resolution=0.02)
        qa = np.array([-1.0, 0.0, 0.0])
        qb = np.array([1.0, 0.0, 0.0])
        self.assertTrue(ck.is_valid(qa))
        self.assertTrue(ck.is_valid(qb))
        self.assertTrue(ck.is_valid(np.zeros(3)), "中点必须是空的，否则测不出目标缺陷")
        self.assertFalse(ck.segment_valid(qa, qb))

    def test_segment_check_accepts_clear_line(self):
        ck = self._checker(margin=0.0, distmax=10.0, resolution=0.01)
        qa = np.array([1.5, 0.2])
        qb = np.array([2.2, -0.4])
        for t in np.linspace(0, 1, 201):
            self.assertGreater(_arm_clearance_analytic(qa + t * (qb - qa)), 0.0)
        self.assertTrue(ck.segment_valid(qa, qb))

    def test_distmax_must_exceed_margin(self):
        with self.assertRaises(ValueError):
            CollisionChecker(self.model, [0, 1], self.robot, self.obs,
                             safety_margin=0.1, distmax=0.05)


# ============================================================ RRT-Connect


class _PointRobotFixture(unittest.TestCase):
    walls = _WALL_WITH_GAP
    margin = 0.01
    resolution = 0.02

    def setUp(self):
        self.model, robot, obs = _build_point_robot(self.walls)
        self.checker = CollisionChecker(self.model, [0, 1, 2], robot, obs,
                                        safety_margin=self.margin,
                                        distmax=0.3,
                                        resolution=self.resolution)
        self.lo = np.full(3, -1.0)
        self.hi = np.full(3, 1.0)
        self.planner = RRTConnect(
            self.checker, self.lo, self.hi,
            RRTConnectConfig(step_size=0.2, max_iters=4000),
        )
        # 起终点连线正对墙面实心处，必须绕到 y≈0.7、z≈0 的缺口才过得去
        self.start = np.array([-0.6, -0.6, -0.6])
        self.goal = np.array([0.6, -0.6, -0.6])

    def dense_min_clearance(self, path, samples_per_unit=400):
        worst = np.inf
        for a, b in zip(path[:-1], path[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) * samples_per_unit)), 1)
            for k in range(n + 1):
                worst = min(worst, _point_clearance_analytic(
                    a + (k / n) * (b - a), self.walls))
        return float(worst)


class TestRRTConnect(_PointRobotFixture):

    def test_direct_connection_is_blocked_in_this_fixture(self):
        """先证明这个测试夹具确实是"直线走不通"的，否则后面的测试没有意义。"""
        self.assertFalse(self.checker.segment_valid(self.start, self.goal))

    def test_succeeds_on_every_seed(self):
        for seed in range(6):
            res = self.planner.plan(self.start, self.goal, seed=seed)
            self.assertTrue(res.success, f"seed={seed} 失败：{res.reason}")

    def test_path_endpoints_are_exact(self):
        """必须逐 seed 检查：两棵树每轮交换一次，
        连通发生在奇数轮还是偶数轮决定了 tree_a 是起点树还是终点树。
        只测一个 seed 的话，"交换后忘记把路径反转"这个 bug 有一半概率漏网。
        """
        for seed in range(8):
            res = self.planner.plan(self.start, self.goal, seed=seed)
            self.assertTrue(res.success, f"seed={seed} 失败")
            np.testing.assert_allclose(res.path[0], self.start, atol=0.0, rtol=0.0,
                                       err_msg=f"seed={seed} 起点不对")
            np.testing.assert_allclose(res.path[-1], self.goal, atol=0.0, rtol=0.0,
                                       err_msg=f"seed={seed} 终点不对")

    def test_path_stays_inside_joint_limits(self):
        res = self.planner.plan(self.start, self.goal, seed=1)
        for q in res.path:
            self.assertTrue(np.all(q >= self.lo - 1e-12), f"{q} 低于下限")
            self.assertTrue(np.all(q <= self.hi + 1e-12), f"{q} 高于上限")

    def test_path_is_collision_free_under_analytic_oracle(self):
        """用比规划分辨率密 20 倍的采样、且用解析距离复核整条路径。

        这同时检验两件事：碰撞检测本身对不对，以及
        分辨率 0.02 rad + 裕度 0.01 m 的组合有没有留下钻缝的空子。
        """
        for seed in range(4):
            res = self.planner.plan(self.start, self.goal, seed=seed)
            self.assertTrue(res.success)
            self.assertGreater(self.dense_min_clearance(res.path), 0.0,
                               f"seed={seed} 的路径在细采样下发生碰撞")

    def test_path_is_actually_connected(self):
        """相邻路径点之间的间距不得超过一个扩展步长（树的边就是这么长的）。"""
        res = self.planner.plan(self.start, self.goal, seed=2)
        for a, b in zip(res.path[:-1], res.path[1:]):
            self.assertLessEqual(np.linalg.norm(b - a), 0.2 + 1e-9)

    def test_reports_failure_when_free_space_is_disconnected(self):
        """整面实心墙把构型空间切成两半时，必须报失败而不是给一条假路径。"""
        model, robot, obs = _build_point_robot(_WALL_SEALED)
        ck = CollisionChecker(model, [0, 1, 2], robot, obs,
                              safety_margin=0.01, distmax=0.3, resolution=0.02)
        planner = RRTConnect(ck, self.lo, self.hi,
                             RRTConnectConfig(step_size=0.2, max_iters=400))
        res = planner.plan(self.start, self.goal, seed=0)
        self.assertFalse(res.success)
        self.assertEqual(res.path, [])

    def test_rejects_colliding_start_and_goal(self):
        inside_wall = np.array([0.0, 0.0, 0.8])       # w_top 内部
        self.assertLess(_point_clearance_analytic(inside_wall, self.walls), 0.0)
        r1 = self.planner.plan(inside_wall, self.goal, seed=0)
        self.assertFalse(r1.success)
        self.assertIn("起点", r1.reason)
        r2 = self.planner.plan(self.start, inside_wall, seed=0)
        self.assertFalse(r2.success)
        self.assertIn("终点", r2.reason)

    def test_rejects_configs_outside_joint_limits(self):
        res = self.planner.plan(np.array([-5.0, 0.0, 0.0]), self.goal, seed=0)
        self.assertFalse(res.success)
        self.assertIn("限位", res.reason)

    def test_uses_straight_line_when_free_space_allows(self):
        """自由空间里不该绕路：直连可行时应当直接返回两点路径。"""
        res = self.planner.plan(np.array([-0.6, -0.6, -0.6]),
                                np.array([-0.6, 0.6, 0.6]), seed=0)
        self.assertTrue(res.success)
        self.assertEqual(len(res.path), 2)


# ============================================================ 平滑


class TestShortcutSmoothing(_PointRobotFixture):

    def test_length_never_increases(self):
        for seed in range(4):
            res = self.planner.plan(self.start, self.goal, seed=seed)
            raw = path_length(res.path)
            smoothed = shortcut_smooth(res.path, self.checker, iters=200, seed=seed)
            self.assertLessEqual(path_length(smoothed), raw + 1e-12)

    def test_length_actually_shrinks_on_rrt_output(self):
        """RRT 的原始路径必然有明显冗余；平滑必须真的缩短它，不能只是不变。"""
        gains = []
        for seed in range(4):
            res = self.planner.plan(self.start, self.goal, seed=seed)
            raw = path_length(res.path)
            smoothed = shortcut_smooth(res.path, self.checker, iters=200, seed=seed)
            gains.append(1.0 - path_length(smoothed) / raw)
        self.assertGreater(min(gains), 0.15,
                           f"最小缩短比例只有 {min(gains):.3f}")

    def test_endpoints_preserved(self):
        res = self.planner.plan(self.start, self.goal, seed=1)
        smoothed = shortcut_smooth(res.path, self.checker, iters=200, seed=1)
        np.testing.assert_allclose(smoothed[0], self.start, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(smoothed[-1], self.goal, atol=0.0, rtol=0.0)

    def test_smoothed_path_still_collision_free_under_analytic_oracle(self):
        for seed in range(4):
            res = self.planner.plan(self.start, self.goal, seed=seed)
            smoothed = shortcut_smooth(res.path, self.checker, iters=200, seed=seed)
            self.assertGreater(self.dense_min_clearance(smoothed), 0.0,
                               f"seed={seed} 平滑后发生碰撞")

    def test_free_space_zigzag_collapses_to_straight_line(self):
        """无障碍时，任何锯齿路径都应被压回起终点直线。

        独立 oracle：直线长度就是 ‖q_goal − q_start‖，不依赖任何实现细节。
        """
        model, robot, obs = _build_point_robot(_WALL_SEALED)
        ck = CollisionChecker(model, [0, 1, 2], robot, obs,
                              safety_margin=0.01, distmax=0.3, resolution=0.02)
        a = np.array([0.30, -0.8, -0.8])          # 全程留在墙的同一侧
        b = np.array([0.30, 0.8, 0.8])
        rng = np.random.default_rng(0)
        zigzag = [a]
        for t in np.linspace(0.1, 0.9, 9):
            base = a + t * (b - a)
            jitter = np.array([0.0, rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)])
            zigzag.append(base + jitter)
        zigzag.append(b)
        self.assertGreater(path_length(zigzag), np.linalg.norm(b - a) * 1.1)
        smoothed = shortcut_smooth(zigzag, ck, iters=300, seed=0)
        self.assertAlmostEqual(path_length(smoothed), float(np.linalg.norm(b - a)),
                               delta=1e-9)

    def test_prune_keeps_path_valid_and_shorter(self):
        res = self.planner.plan(self.start, self.goal, seed=0)
        pruned = prune_waypoints(res.path, self.checker)
        self.assertLessEqual(path_length(pruned), path_length(res.path) + 1e-12)
        self.assertLess(len(pruned), len(res.path))
        self.assertGreater(self.dense_min_clearance(pruned), 0.0)

    def test_shortcut_finds_gains_that_pruning_cannot(self):
        """随机短路必须比"只在已有顶点之间连线"更强。

        构造一个只有三个顶点的绕行路径 a → m → b：
          · a→b 被方块挡住，所以顶点剪枝无路可走，只能原样返回；
          · 但在 a→m 与 m→b 的**中间**各取一点，连线可以贴着方块过去，
            总长严格更短。
        独立 oracle：手算一条具体的可行捷径给出长度上界。
          a=(−0.8,0,0), m=(0,0.6,0), b=(0.8,0,0)，方块半边 0.2。
          取 p₁=(−0.4,0.3,0)、p₂=(0.4,0.3,0)（y=0.3 > 0.2+0.02+0.01，安全），
          长度 = 0.5 + 0.8 + 0.5 = 1.8，而原路径长 2·√(0.64+0.36) = 2.0。
        所以平滑结果必须 ≤ 1.8，若退化成纯剪枝就会停在 2.0。
        """
        xml = """<mujoco><worldbody>
          <body name="pt" pos="0 0 0">
            <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
            <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
            <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
            <geom name="pt" type="sphere" size="0.02"/>
          </body>
          <geom name="cube" type="box" size="0.2 0.2 0.2" pos="0 0 0"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        ck = CollisionChecker(model, [0, 1, 2], [gid("pt")], [gid("cube")],
                              safety_margin=0.01, distmax=0.3, resolution=0.01)
        a = np.array([-0.8, 0.0, 0.0])
        mid = np.array([0.0, 0.6, 0.0])
        b = np.array([0.8, 0.0, 0.0])
        detour = [a, mid, b]

        self.assertFalse(ck.segment_valid(a, b), "构造前提不成立：a→b 应当被挡住")
        pruned = prune_waypoints(detour, ck)
        self.assertAlmostEqual(path_length(pruned), 2.0, delta=1e-9)

        smoothed = shortcut_smooth(detour, ck, iters=400, seed=0)
        self.assertLessEqual(path_length(smoothed), 1.8,
                             f"平滑后长度 {path_length(smoothed):.4f}，"
                             f"未超过纯顶点剪枝能达到的水平")
        np.testing.assert_allclose(smoothed[0], a, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(smoothed[-1], b, atol=0.0, rtol=0.0)
        for p, q in zip(smoothed[:-1], smoothed[1:]):
            n = max(int(np.ceil(np.linalg.norm(q - p) * 400)), 1)
            for k in range(n + 1):
                pt = p + (k / n) * (q - p)
                self.assertGreater(
                    _box_signed_distance(pt, np.zeros(3), np.full(3, 0.2)) - PT_RADIUS,
                    0.0, "平滑后的捷径穿过了方块")


class TestSelfCollisionPairs(unittest.TestCase):
    """自碰撞几何体对的选择规则（Panda 真模型）。"""

    @classmethod
    def setUpClass(cls):
        from armctrl.planning.scene import build_panda_scene, self_collision_pairs
        cls.scene = build_panda_scene([], with_visuals=False)
        cls.arm_joint_ids = [
            mujoco.mj_name2id(cls.scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ]
        # 只看邻接规则时不传 joint_ids
        cls.pairs_raw = self_collision_pairs(cls.scene.model, cls.scene.robot_geoms)
        # 规划器实际该用的：排除掉与被规划关节无关的对（夹爪内部那些）
        cls.pairs = self_collision_pairs(cls.scene.model, cls.scene.robot_geoms,
                                         cls.arm_joint_ids)

    def test_pair_set_is_exactly_the_non_adjacent_pairs(self):
        """独立复核：不传 joint_ids 时，被排除的对必须且只能是"同一 body"或"父子 body"。

        这是对邻接规则的完整刻画，不复制实现的循环结构。
        """
        m = self.scene.model
        got = {tuple(sorted(p)) for p in self.pairs_raw}
        geoms = self.scene.robot_geoms
        for i, ga in enumerate(geoms):
            for gb in geoms[i + 1:]:
                ba, bb = int(m.geom_bodyid[ga]), int(m.geom_bodyid[gb])
                adjacent = (ba == bb
                            or int(m.body_parentid[ba]) == bb
                            or int(m.body_parentid[bb]) == ba)
                self.assertEqual(tuple(sorted((ga, gb))) in got, not adjacent,
                                 f"几何体对 ({ga},{gb}) 的取舍不对")
        self.assertGreater(len(got), 0)

    def test_pairs_unaffected_by_planned_joints_are_dropped(self):
        """两根手指的相对位姿只由 finger_joint 决定，规划器动不了它，必须排除。

        这条针对的缺陷很具体：夹爪一闭合两指就贴在一起，若把这一对留在
        自碰撞检查里，**任何**手臂构型都会被判非法，规划器当场认为起点不可行。
        不传 joint_ids 的旧口径下这一对是存在的（下面第一句断言），
        所以这条测试在原缺陷上确实会失败。

        判据独立于实现：body A 的位姿由 A→世界这条链上的全部关节决定，
        A 相对 B 只取决于两条链的对称差；对称差与被规划关节不相交，
        则该对的相对位姿在规划过程中恒定不变。
        """
        m = self.scene.model
        planned = set(self.arm_joint_ids)

        def chain_joints(body):
            out, b = set(), int(body)
            while b > 0:
                a, n = int(m.body_jntadr[b]), int(m.body_jntnum[b])
                out |= set(range(a, a + n))
                b = int(m.body_parentid[b])
            return out

        raw = {tuple(sorted(p)) for p in self.pairs_raw}
        got = {tuple(sorted(p)) for p in self.pairs}
        dropped = raw - got
        self.assertGreater(len(dropped), 0,
                           "一对都没排除，说明这条测试测不到目标缺陷")
        for ga, gb in raw:
            rel = chain_joints(int(m.geom_bodyid[ga])) ^ chain_joints(int(m.geom_bodyid[gb]))
            should_drop = not (rel & planned)
            self.assertEqual((ga, gb) in dropped, should_drop,
                             f"几何体对 ({ga},{gb}) 的取舍与相对位姿判据不一致")

    def test_home_configuration_is_self_collision_free(self):
        """常用的 Home 构型不该被判成自碰撞。

        若把父子相邻连杆也放进检查对，它们在关节处天生接触，
        任何构型都会被判非法——这条测试就是那种错误的探针。
        """
        m = self.scene.model
        qidx = np.array([m.jnt_qposadr[j] for j in range(7)])
        ck = CollisionChecker(m, qidx, self.scene.robot_geoms, [], self.pairs,
                              safety_margin=0.0, distmax=0.3)
        q_home = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        self.assertGreater(ck.clearance(q_home), 0.0)

    def test_detects_a_real_self_collision(self):
        """一个实测会自己撞自己的构型必须被判非法（否则自碰撞检查形同虚设）。"""
        m = self.scene.model
        qidx = np.array([m.jnt_qposadr[j] for j in range(7)])
        ck = CollisionChecker(m, qidx, self.scene.robot_geoms, [], self.pairs,
                              safety_margin=0.0, distmax=0.3)
        q_bad = np.array([-1.966386, 1.656769, 0.093111, -2.723971,
                          0.715574, 2.910595, 0.654809])
        self.assertLess(ck.clearance(q_bad), 0.0)
        self.assertFalse(ck.is_valid(q_bad))

    def test_self_margin_is_independent_of_environment_margin(self):
        """自碰撞必须用自己的裕度。

        Panda 的连杆碰撞几何是凸包，常用构型下自身最小间隙只有约 0.022 m。
        若拿"离障碍物 0.03 m"这个环境裕度去要求自碰撞，
        连 Home 构型都会被判非法，规划器会当场认为起点不可行。
        """
        m = self.scene.model
        qidx = np.array([m.jnt_qposadr[j] for j in range(7)])
        q_home = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        ck = CollisionChecker(m, qidx, self.scene.robot_geoms, [], self.pairs,
                              safety_margin=0.03, self_margin=0.005, distmax=0.3)
        own = ck.self_clearance(q_home)
        self.assertTrue(0.005 < own < 0.03,
                        f"自碰撞间隙 {own:.4f} 不在两个裕度之间，测试失去区分力")
        self.assertTrue(ck.is_valid(q_home))


class TestMovableGeoms(unittest.TestCase):
    """只有会动的连杆才该参与"机械臂 vs 环境"的碰撞检查。"""

    @classmethod
    def setUpClass(cls):
        from armctrl.planning.scene import build_panda_scene, movable_geoms
        cls.scene = build_panda_scene([], with_visuals=True)
        cls.model = cls.scene.model
        cls.joint_ids = [mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_JOINT,
                                           f"joint{i}") for i in range(1, 8)]
        cls.movable = movable_geoms(cls.model, cls.scene.robot_geoms, cls.joint_ids)

    def test_fixed_base_geoms_are_excluded(self):
        """独立复核：被排除的几何体，其所在 body 到世界的链路上没有任何被规划关节。"""
        m = self.model
        joint_bodies = {int(m.jnt_bodyid[j]) for j in self.joint_ids}
        kept = set(self.movable)
        for g in self.scene.robot_geoms:
            b = int(m.geom_bodyid[g])
            driven = False
            while b > 0:
                if b in joint_bodies:
                    driven = True
                    break
                b = int(m.body_parentid[b])
            self.assertEqual(g in kept, driven,
                             f"几何体 {g} 的取舍不对（是否受关节驱动={driven}）")
        self.assertLess(len(kept), len(self.scene.robot_geoms),
                        "一个都没排除，说明该场景里没有固定底座几何，测试失去意义")

    def test_home_stays_valid_with_floor_as_obstacle(self):
        """场景里有地面时，Home 构型必须仍然合法。

        Panda 的底座 link0 本来就压在地面上。若把它也算进环境碰撞，
        任何构型的间隙都 ≤ 0，规划器会直接报"起点本身发生碰撞"。
        """
        from armctrl.planning.scene import self_collision_pairs
        m = self.model
        qidx = np.array([m.jnt_qposadr[j] for j in range(7)])
        q_home = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])

        naive = CollisionChecker(m, qidx, self.scene.robot_geoms,
                                 self.scene.obstacle_geoms, [],
                                 safety_margin=0.01, distmax=0.3)
        self.assertLessEqual(naive.env_clearance(q_home), 0.0,
                             "底座本应与地面接触，否则这条测试测不到目标缺陷")

        fixed = CollisionChecker(m, qidx, self.movable,
                                 self.scene.obstacle_geoms,
                                 self_collision_pairs(m, self.scene.robot_geoms,
                                                      self.joint_ids),
                                 safety_margin=0.01, self_margin=0.005, distmax=0.3)
        self.assertGreater(fixed.env_clearance(q_home), 0.01)
        self.assertTrue(fixed.is_valid(q_home))


# ============================================================ 时间参数化


class TestJointPathTrajectory(unittest.TestCase):

    V = np.array([1.0, 0.8, 1.5, 0.6])
    A = np.array([4.0, 3.0, 6.0, 2.5])
    J = np.array([40.0, 30.0, 60.0, 25.0])

    def _traj(self):
        pts = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, -0.3, 0.9, 0.2]),
            np.array([-0.2, 0.4, 0.9, -0.5]),
            np.array([0.7, 0.7, -0.4, 0.1]),
        ]
        return JointPathTrajectory(pts, self.V, self.A, self.J), pts

    def test_boundary_conditions(self):
        traj, pts = self._traj()
        q0, v0, a0 = traj(0.0)
        qT, vT, aT = traj(traj.duration)
        np.testing.assert_allclose(q0, pts[0], atol=1e-12)
        np.testing.assert_allclose(qT, pts[-1], atol=1e-12)
        np.testing.assert_allclose(v0, 0.0, atol=1e-12)
        np.testing.assert_allclose(vT, 0.0, atol=1e-12)
        np.testing.assert_allclose(a0, 0.0, atol=1e-12)
        np.testing.assert_allclose(aT, 0.0, atol=1e-12)

    def test_passes_through_every_waypoint(self):
        traj, pts = self._traj()
        _, Q, _, _ = traj.sample(1e-3)
        for p in pts:
            self.assertLess(np.min(np.linalg.norm(Q - p, axis=1)), 2e-3,
                            f"轨迹没有经过路径点 {p}")

    def test_samples_lie_on_the_geometric_path(self):
        """时间律只能改变"什么时候走到哪"，不能改变"走哪条线"。"""
        traj, pts = self._traj()
        _, Q, _, _ = traj.sample(1e-3)
        worst = max(_point_to_polyline(q, pts) for q in Q)
        self.assertLess(worst, 1e-12, f"最大偏离 {worst:.3e}")

    def test_velocity_bound_via_lipschitz(self):
        """|q̇| ≤ v_max 等价于 q 是 v_max-Lipschitz 的。

        只用采样出的 q(t) 做差分，完全不看实现内部的 ṡ。
        """
        traj, _ = self._traj()
        h = 1e-4
        ts = np.arange(0.0, traj.duration - h, h)
        worst = np.zeros(len(self.V))
        for t in ts:
            worst = np.maximum(worst, np.abs(traj(t + h)[0] - traj(t)[0]) / h)
        self.assertTrue(np.all(worst <= self.V * (1 + 1e-9)),
                        f"实测 {worst} 超过上限 {self.V}")

    def test_acceleration_bound_via_lipschitz(self):
        traj, _ = self._traj()
        h = 1e-4
        ts = np.arange(0.0, traj.duration - h, h)
        worst = np.zeros(len(self.A))
        for t in ts:
            worst = np.maximum(worst, np.abs(traj(t + h)[1] - traj(t)[1]) / h)
        self.assertTrue(np.all(worst <= self.A * (1 + 1e-9)),
                        f"实测 {worst} 超过上限 {self.A}")

    def test_jerk_bound_via_lipschitz(self):
        """S 曲线的加速度是分段线性的，其 Lipschitz 常数就是 jerk 上限。

        直接对 q̈ 做三阶差分会在 jerk 阶跃处炸掉；
        用 Lipschitz 判据则在阶跃处依然严格成立。
        """
        traj, _ = self._traj()
        h = 1e-4
        ts = np.arange(0.0, traj.duration - h, h)
        worst = np.zeros(len(self.J))
        for t in ts:
            worst = np.maximum(worst, np.abs(traj(t + h)[2] - traj(t)[2]) / h)
        self.assertTrue(np.all(worst <= self.J * (1 + 1e-9)),
                        f"实测 {worst} 超过上限 {self.J}")

    def test_velocity_is_the_derivative_of_position(self):
        """q̇(t) 必须真的是 q(t) 的导数——查符号和整体尺度。

        独立 oracle：中心差分。q(t) 在每一段内是 t 的分段三次多项式，
        中心差分对三次多项式的截断误差是 (h²/6)·q⃛，
        因此误差上界 ≈ j_max·h²/6，与 h=1e-5 相比是 1e-9 量级；
        跨越 jerk 跳变点时误差至多 a_max·h。两者都远小于符号错误造成的 2|q̇|。
        """
        traj, _ = self._traj()
        h = 1e-5
        worst = 0.0
        for t in np.linspace(h, traj.duration - h, 4000):
            fd = (traj(t + h)[0] - traj(t - h)[0]) / (2 * h)
            worst = max(worst, float(np.max(np.abs(fd - traj(t)[1]))))
        self.assertLess(worst, self.A.max() * h * 2 + 1e-6, f"最大偏差 {worst:.3e}")

    def test_acceleration_is_the_derivative_of_velocity(self):
        """q̈(t) 必须真的是 q̇(t) 的导数。

        q̇ 在每一段内分段二次，中心差分对二次多项式**精确**，
        误差只来自跨越 jerk 跳变点的那些采样，上界 j_max·h。
        """
        traj, _ = self._traj()
        h = 1e-5
        worst = 0.0
        for t in np.linspace(h, traj.duration - h, 4000):
            fd = (traj(t + h)[1] - traj(t - h)[1]) / (2 * h)
            worst = max(worst, float(np.max(np.abs(fd - traj(t)[2]))))
        self.assertLess(worst, self.J.max() * h + 1e-6, f"最大偏差 {worst:.3e}")

    def test_limits_are_tight_not_merely_safe(self):
        """瓶颈关节必须真的跑到它的速度上限，否则说明折算过于保守。

        单关节长距离运动时，S 曲线一定进入匀速段，此时该关节速度 = v_max。
        """
        pts = [np.zeros(4), np.array([0.0, 0.0, 3.0, 0.0])]
        traj = JointPathTrajectory(pts, self.V, self.A, self.J)
        _, _, V, _ = traj.sample(1e-3)
        self.assertAlmostEqual(np.max(np.abs(V[:, 2])), self.V[2], delta=1e-6)
        np.testing.assert_allclose(V[:, [0, 1, 3]], 0.0, atol=1e-12)

    def test_multi_joint_segment_binds_the_right_joint(self):
        """多关节同时运动时，先到限的应当是 v_i/|u_i| 最小的那个关节。

        独立推导：q̇_i = u_i·ṡ，所以 max_t |q̇_i| = |u_i|·ṡ_peak。
        若某关节达到自身上限，则 ṡ_peak = v_i/|u_i|，
        该值必须等于所有关节里的最小者，否则别的关节会越限。
        """
        d = np.array([1.0, 1.0, 1.0, 1.0]) * 2.0
        traj = JointPathTrajectory([np.zeros(4), d], self.V, self.A, self.J)
        _, _, V, _ = traj.sample(1e-3)
        peak = np.max(np.abs(V), axis=0)
        u = d / np.linalg.norm(d)
        binding = int(np.argmin(self.V / np.abs(u)))
        self.assertAlmostEqual(peak[binding], self.V[binding], delta=1e-6)
        self.assertTrue(np.all(peak <= self.V * (1 + 1e-9)))

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            JointPathTrajectory([np.zeros(3)], 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            JointPathTrajectory([np.zeros(3), np.zeros(3)], 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            JointPathTrajectory([np.zeros(3), np.ones(3)], 0.0, 1.0, 1.0)

    def test_duplicate_waypoints_are_skipped(self):
        pts = [np.zeros(3), np.zeros(3), np.array([0.3, 0.0, 0.0])]
        traj = JointPathTrajectory(pts, 1.0, 4.0, 40.0)
        self.assertEqual(len(traj.segments), 1)


# ==== P0-05 场景级：三层间隙必须各算各的 =============================
# 审核报告 P0-05：页面上标着「执行中最小间隙」的读数，其实算的是
# **平滑后几何路径**的间隙。而 blend_toppra 模式下真正下发的是倒角后的
# blended 路径——倒角把拐角削圆，路径往转弯内侧挪，间隙**只会变小**。
# 所以旧读数是**系统性偏乐观**的，而且乐观的方向正是安全方向。


def _run_planning(**params):
    """跑一次 PlanningScene 的规划，返回 _stats。"""
    from armctrl.tuner.scenes import PlanningScene
    sc = PlanningScene()
    sc.build()
    for k, v in params.items():
        sc.set(k, v)
    sc.reset()
    return sc._stats


class TestPlanningClearanceLayers(unittest.TestCase):
    """⭐ P0-05：原始 / 平滑 / 下发 三层间隙。

    判据是**外部可观测的读数之间的物理关系**，不看实现的中间量。
    """

    # 实测挑出来的确定性反例：倒角把间隙从 29.76 mm 削到 21.33 mm
    BAD = dict(time_law="blend_toppra", max_deviation=0.06, seed=6, margin=0.03)

    def test_blending_can_eat_into_the_planned_safety_margin(self):
        """⭐⭐ 本条就是 P0-05 的验收条件：

        构造一个「平滑路径安全、倒角后侵入」的确定性例子，
        并要求**下发轨迹层**能抓到，而不是照样通过。

        实测（seed=6, 倒角预算 0.06, 安全裕度 30 mm）：

        | 层次 | 最小间隙 |
        |---|---|
        | 原始 RRT 路径 | 34.22 mm |
        | 平滑后路径 | 29.76 mm ← **旧读数报的是这个** |
        | **实际下发轨迹** | **21.33 mm** ← 真相 |

        旧读数虚报 **8.43 mm（28%）**，而且已经**跌破 30 mm 的规划裕度**。
        """
        st = _run_planning(**self.BAD)
        self.assertGreater(st["duration"], 0.0, "这组参数应当规划成功")
        margin_mm = self.BAD["margin"] * 1e3
        # 平滑层看起来是安全的（贴着裕度）
        self.assertGreater(st["clear_smooth"], 0.95 * margin_mm,
                           f"平滑后路径本该贴着裕度：{st['clear_smooth']:.2f} mm")
        # 下发层已经跌破裕度 —— 这就是旧读数掩盖掉的风险
        self.assertLess(st["clearance"], 0.90 * margin_mm,
                        f"下发轨迹间隙 {st['clearance']:.2f} mm 没有跌破裕度，"
                        "该重新挑一个反例（这条测试就失去意义了）")
        self.assertGreater(st["clear_loss"], 5.0,
                           f"倒角只吃掉了 {st['clear_loss']:.2f} mm，反例不够典型")

    def test_the_three_layers_are_not_the_same_number(self):
        """⭐ 变异守卫：如果有人把三个读数接到同一个来源上，这条必须失败。"""
        st = _run_planning(**self.BAD)
        vals = [st["clear_raw"], st["clear_smooth"], st["clearance"]]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertGreater(abs(vals[i] - vals[j]), 1.0,
                                   f"第 {i} 层和第 {j} 层读数几乎相同（{vals}），"
                                   "说明它们算的是同一个对象")

    def test_clearance_decreases_monotonically_down_the_pipeline(self):
        """⭐ 物理关系：平滑只会拉直（间隙可能降），倒角只会往内侧挪（间隙必降）。

        所以必须有 ``下发 ≤ 平滑``。这条在多组种子上都要成立。
        """
        for seed in (0, 3, 4, 5, 6):
            st = _run_planning(time_law="blend_toppra", max_deviation=0.06,
                               seed=seed, margin=0.03)
            if st["duration"] == 0.0:
                continue
            self.assertLessEqual(st["clearance"], st["clear_smooth"] + 1e-6,
                                 f"seed={seed}: 下发轨迹间隙 {st['clearance']:.2f} "
                                 f"> 平滑路径 {st['clear_smooth']:.2f}，"
                                 "倒角不可能让间隙变大")

    def test_scurve_mode_has_no_blending_loss(self):
        """⭐ 对照组：scurve 模式不倒角，几何路径就是下发路径，损失必须 ≈ 0。

        这条把「三层不同」这件事钉死成**倒角造成的**，
        而不是我的采样方法本身有偏差。
        """
        st = _run_planning(time_law="scurve", seed=6, margin=0.03)
        self.assertLess(abs(st["clear_loss"]), 0.5,
                        f"不倒角却损失了 {st['clear_loss']:.2f} mm，"
                        "说明轨迹采样和路径采样本身就对不齐")
        self.assertLess(abs(st["blend_dev"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
