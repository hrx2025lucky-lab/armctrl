"""抓取控制的不变量测试。

oracle 独立性
------------
最关键的一条 —— 防滑最小夹持力 N ≥ m·g/(2μ) —— **不用公式自证**，
而是搭一个独立的 MuJoCo 装置：两块板夹住一个方块，把法向力从小到大扫过去，
观察方块**实际**从哪个力开始不再下滑。实测的临界力与公式预测直接对照。

这条判据完全不经过 `armctrl.control.grasp` 的任何代码路径，
用的是 MuJoCo 的接触求解器本身。
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from armctrl.control.grasp import (
    AdaptiveGripConfig, AdaptiveGripForce, SlipReading, box_corner_points,
    constrained_grasp_frame, effective_friction, grasp_frame_from_points,
    min_normal_force, principal_axes,
)


# ==================================================================== 防滑判据


def _squeeze_rig(mass: float, mu: float, force: float, half=0.03,
                 mu_l: float | None = None, mu_r: float | None = None) -> str:
    """两块板从两侧夹住一个方块。板由位置伺服压住，夹持力由 forcerange 封顶。

    刻意做得极简：只有平移自由度、没有重力以外的外力、板是无质量滑块。
    这样"能不能夹住"只取决于摩擦，不掺别的效应。

    `mu_l` / `mu_r` 可以单独指定两块板的摩擦，默认跟 `mu` 一样。
    ⭐ 这个入口是补出来的：原来全场只有一个 `<default>` 摩擦，
    于是**每个接触的 μ 必然相同**，`min` 和 `max` 恒等——
    下面那条"取最小值"的测试因此变成了一句空话（见测试内注释）。
    """
    mu_l = mu if mu_l is None else mu_l
    mu_r = mu if mu_r is None else mu_r
    return f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.0005"/>
  <default><geom friction="{mu} 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/></default>
  <worldbody>
    <body name="obj" pos="0 0 0">
      <freejoint/>
      <geom name="obj_g" type="box" size="{half} {half} {half}" mass="{mass}"/>
    </body>
    <body name="jaw_l" pos="-{half + 0.01} 0 0">
      <joint name="jl" type="slide" axis="1 0 0"/>
      <geom name="jl_g" type="box" size="0.005 0.05 0.05" mass="0.05"
            friction="{mu_l} 0.005 0.0001"/>
    </body>
    <body name="jaw_r" pos="{half + 0.01} 0 0">
      <joint name="jr" type="slide" axis="-1 0 0"/>
      <geom name="jr_g" type="box" size="0.005 0.05 0.05" mass="0.05"
            friction="{mu_r} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="al" joint="jl" kp="8000" forcerange="-{force} {force}" ctrlrange="0 0.1"/>
    <position name="ar" joint="jr" kp="8000" forcerange="-{force} {force}" ctrlrange="0 0.1"/>
  </actuator>
</mujoco>"""


def _sink_after(mass, mu, force, seconds=1.2):
    """夹住 seconds 秒后方块下沉了多少（米），以及实际法向力与生效摩擦。"""
    m = mujoco.MjModel.from_xml_string(_squeeze_rig(mass, mu, force))
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.05                       # 命令两块板深压，实际力被 forcerange 封顶
    z0 = None
    fn_list, mu_list = [], []
    for k in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k == int(0.3 / m.opt.timestep):
            z0 = float(d.qpos[2])          # 等接触稳定后再记起点
        if k > int(0.3 / m.opt.timestep) and d.ncon:
            fn = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))}
                if "obj_g" in names and ("jl_g" in names or "jr_g" in names):
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, f6)
                    fn += abs(float(f6[0]))
                    mu_list.append(float(c.friction[0]))
            if fn > 0:
                fn_list.append(fn / 2.0)   # 单面法向力
    sink = abs(float(d.qpos[2]) - z0) if z0 is not None else np.inf
    return sink, (np.mean(fn_list) if fn_list else 0.0), \
        (np.mean(mu_list) if mu_list else 0.0)


class TestAntiSlipForceAgainstPhysics(unittest.TestCase):
    """N ≥ m·g/(2μ) 用**独立的 MuJoCo 装置实测**，不用公式自证。"""

    def test_formula_predicts_the_measured_slip_threshold(self):
        """扫描夹持力，实测临界值应当落在公式预测附近。

        实测扫描结果（m=0.5 kg, μ=0.6，公式预测 4.0875 N）：

            0.3×  完全掉出      0.6×  完全掉出      0.8×  完全掉出
            1.0×  下沉 12.3 mm（临界）
            1.2×  下沉 0.30 mm  1.6×  0.23 mm      2.5×  0.17 mm

        即**实测临界在公式值的 1.0–1.2 倍之间**，公式略偏乐观约 10–20%，
        与接触求解器的软化（solref/solimp）和有限沉降时间一致。

        判据取 0.8× 必滑、1.2× 必稳，窗口只有 1.5 倍。
        漏掉"两个接触面"的因子 2 会让预测值翻倍，一定落在窗口外被抓到。
        """
        mass, mu = 0.5, 0.6
        n_theory = min_normal_force(mass, mu)          # = 0.5*9.81/(2*0.6) ≈ 4.09 N
        # 装置里 forcerange 限的是单个执行器出力，即单面法向力
        sink_low, fn_low, mu_low = _sink_after(mass, mu, 0.8 * n_theory)
        sink_high, fn_high, mu_high = _sink_after(mass, mu, 1.2 * n_theory)

        self.assertAlmostEqual(mu_high, mu, places=6, msg="生效摩擦系数与设定不符")
        self.assertGreater(sink_low, 5e-3,
                           f"夹持力只有理论值的 0.8 倍({fn_low:.2f} N)却没下滑，"
                           "说明装置里有别的东西在托着，测试失去意义")
        self.assertLess(sink_high, 1e-3,
                        f"夹持力已达理论值的 1.2 倍({fn_high:.2f} N)仍在下滑 "
                        f"{sink_high * 1e3:.2f} mm，公式预测偏低")

    def test_missing_factor_two_would_be_caught(self):
        """守护：漏掉"两个接触面"的因子 2 会让预测值翻倍，必须能被上面那条抓到。

        直接断言两个预测值差一倍，且实测临界确实在 2 个面的那一侧。
        """
        mass, mu = 0.5, 0.6
        two_faces = min_normal_force(mass, mu, n_contacts=2)
        one_face = min_normal_force(mass, mu, n_contacts=1)
        self.assertAlmostEqual(one_face / two_faces, 2.0, places=12)
        # 用"单面"公式的一半（= 双面公式值）就该夹得住 → 说明双面才是对的
        sink, _, _ = _sink_after(mass, mu, 1.2 * two_faces)
        self.assertLess(sink, 1e-3)

    def test_scales_correctly_with_mass_friction_and_acceleration(self):
        """量纲与单调性：与 m 成正比、与 μ 成反比、加速度按 (g+a) 进入。"""
        base = min_normal_force(1.0, 0.5)
        self.assertAlmostEqual(min_normal_force(2.0, 0.5) / base, 2.0, places=12)
        self.assertAlmostEqual(min_normal_force(1.0, 1.0) / base, 0.5, places=12)
        self.assertAlmostEqual(
            min_normal_force(1.0, 0.5, accel=9.81) / base, 2.0, places=12)
        with self.assertRaises(ValueError):
            min_normal_force(1.0, 0.0)

    def test_effective_friction_takes_the_minimum_over_contacts(self):
        """多个接触时取**最小**的 μ：最容易滑的那个面决定整体。

        取平均或最大都会高估抓取裕度，是危险的错误。

        ⚠️ 这条测试第一版是**空的**：夹具全场只有一个 `<default>` 摩擦，
        所有接触的 μ 一模一样，于是 `min(mus)` 和 `max(mus)` 恒等——
        把实现改成 `max` 照样全绿（批次 E-3 变异体 M09 就是这么活下来的）。
        现在两块板给成 0.3 / 0.9，并且**先断言 μ 真的不止一个值**，
        这条测试才有鉴别力。⭐ 判别口诀：
        **「如果把实现改成另一个明显错的写法，这条测试会红吗？」**
        """
        m = mujoco.MjModel.from_xml_string(
            _squeeze_rig(0.5, 0.3, 10.0, mu_l=0.3, mu_r=0.9))
        d = mujoco.MjData(m)
        d.ctrl[:] = 0.05
        for _ in range(1200):
            mujoco.mj_step(m, d)
        ids = list(range(d.ncon))
        self.assertGreater(len(ids), 1, "接触数不足，这条测试测不到「取最小」")
        mus = [float(d.contact[i].friction[0]) for i in ids]
        self.assertGreater(
            len(set(round(x, 9) for x in mus)), 1,
            f"各接触的 μ 全都一样（{sorted(set(mus))}），min/max 无法区分——"
            f"这条测试会退化成一句空话")
        self.assertAlmostEqual(effective_friction(m, d, ids), min(mus), places=12)
        self.assertNotAlmostEqual(
            effective_friction(m, d, ids), max(mus), places=9,
            msg="取到了最大值，会高估抓取裕度")
        self.assertEqual(effective_friction(m, d, []), 0.0)


# ==================================================================== 力自适应


class TestAdaptiveGripForce(unittest.TestCase):
    """自适应律的行为必须可预测：滑就快加，不滑就慢减，且始终有界。"""

    def _cfg(self):
        return AdaptiveGripConfig(rise=60.0, decay=2.0, f_min=1.0, f_max=80.0)

    def test_rises_fast_when_slipping_and_decays_slowly_when_not(self):
        dt = 1e-3
        a = AdaptiveGripForce(10.0, self._cfg())
        slip = SlipReading(sliding=True)
        for _ in range(100):
            a.update(slip, dt)
        self.assertAlmostEqual(a.force, 10.0 + 60.0 * 0.1, places=9)

        b = AdaptiveGripForce(10.0, self._cfg())
        calm = SlipReading()
        for _ in range(100):
            b.update(calm, dt)
        self.assertAlmostEqual(b.force, 10.0 - 2.0 * 0.1, places=9)
        # 加力必须显著快于减力，否则滑起来就补不回来
        self.assertGreater(60.0, 10.0 * 2.0)

    def test_force_is_clamped_to_both_bounds(self):
        dt = 1e-3
        a = AdaptiveGripForce(10.0, self._cfg())
        for _ in range(100000):
            a.update(SlipReading(sliding=True), dt)
        self.assertAlmostEqual(a.force, 80.0, places=9)
        for _ in range(100000):
            a.update(SlipReading(), dt)
        self.assertAlmostEqual(a.force, 1.0, places=9)

    def test_counts_slip_events_on_rising_edge_only(self):
        """连续滑移只算一次事件，否则"滑移次数"会变成"滑移步数"，没有意义。"""
        a = AdaptiveGripForce(10.0, self._cfg())
        for _ in range(50):
            a.update(SlipReading(sliding=True), 1e-3)
        self.assertEqual(a.n_slip_events, 1)
        for _ in range(50):
            a.update(SlipReading(), 1e-3)
        self.assertEqual(a.n_slip_events, 1)
        a.update(SlipReading(sliding=True), 1e-3)
        self.assertEqual(a.n_slip_events, 2)

    def test_incipient_and_sliding_are_both_honoured(self):
        """预测性判据单独也能触发加力，不必等真滑。"""
        a = AdaptiveGripForce(10.0, self._cfg())
        a.update(SlipReading(incipient=True, sliding=False), 1e-2)
        self.assertGreater(a.force, 10.0)


# ==================================================================== 抓取位姿


class TestGraspPoseGeneration(unittest.TestCase):
    """开合轴必须落在**最短**主轴上，且与最长主轴垂直。"""

    def test_close_axis_is_the_shortest_principal_axis(self):
        """最短轴取 y（水平），从上方接近 —— 这是"从上往下夹一个立着的薄盒"。"""
        half = np.array([0.10, 0.015, 0.04])          # 长 x / 最短 y / 中 z
        pts = box_corner_points(half)
        _, R, width = grasp_frame_from_points(pts, approach=np.array([0, 0, -1.0]))
        close_axis = R[:, 0]
        self.assertAlmostEqual(abs(float(close_axis[1])), 1.0, places=9)
        self.assertAlmostEqual(width, 2 * 0.015, places=9)
        # 接近轴应当就是竖直向下
        self.assertAlmostEqual(float(R[2, 2]), -1.0, places=9)

    def test_flat_object_cannot_be_grasped_from_directly_above(self):
        """扁平工件（最短轴竖直）从正上方接近是**物理上不成立**的抓取。

        开合轴会与接近轴重合，等于要求把一根手指插到工件底下去。
        实现必须**报错**，而不是返回一个由数值噪声决定的方向——
        后者会让上层拿到一个看似合法、实则随机的抓取位姿。
        这也是真实抓取里著名的"桌面上的薄片抓不起来"问题。
        """
        pts = box_corner_points([0.10, 0.04, 0.015])   # 最短轴是 z
        with self.assertRaises(ValueError):
            grasp_frame_from_points(pts, approach=np.array([0.0, 0.0, -1.0]))
        # 换成水平接近就成立了
        _, R, width = grasp_frame_from_points(pts, approach=np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(abs(float(R[2, 0])), 1.0, places=9)
        self.assertAlmostEqual(width, 2 * 0.015, places=9)

    def test_close_axis_is_perpendicular_to_the_long_axis(self):
        """这是需求里点名的性质：物体主轴与夹爪开合轴垂直。"""
        rng = np.random.default_rng(4)
        for _ in range(20):
            half = np.sort(rng.uniform(0.01, 0.12, 3))[::-1]     # 严格递减
            # 随机姿态，确保结论与坐标轴对齐无关
            A = rng.standard_normal((3, 3))
            Rq, _ = np.linalg.qr(A)
            if np.linalg.det(Rq) < 0:
                Rq[:, 0] = -Rq[:, 0]
            pts = box_corner_points(half, rot=Rq)
            axes, var = principal_axes(pts)
            long_axis = axes[:, 0]
            approach = np.cross(long_axis, Rq[:, 2])
            if np.linalg.norm(approach) < 1e-6:
                approach = np.array([0.0, 0.0, -1.0])
            _, R, _ = grasp_frame_from_points(pts, approach=approach)
            self.assertLess(abs(float(np.dot(R[:, 0], long_axis))), 1e-6,
                            "开合轴与最长主轴不垂直")

    def test_frame_is_a_proper_rotation(self):
        pts = box_corner_points([0.09, 0.02, 0.05])    # 最短轴 y，可从上方接近
        _, R, _ = grasp_frame_from_points(pts, approach=np.array([0, 0, -1.0]))
        self.assertLess(float(np.abs(R.T @ R - np.eye(3)).max()), 1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_approach_parallel_to_close_axis_is_rejected(self):
        """接近方向与开合轴平行时无法定出坐标系，必须报错而不是返回噪声。"""
        half = np.array([0.10, 0.04, 0.015])
        pts = box_corner_points(half)
        with self.assertRaises(ValueError):
            grasp_frame_from_points(pts, approach=np.array([0.0, 0.0, 1.0]))

    def test_principal_axes_match_an_independent_eigen_decomposition(self):
        """SVD 求主轴的结果要与协方差矩阵的特征分解一致（另一条算路）。"""
        rng = np.random.default_rng(9)
        P = rng.standard_normal((200, 3)) @ np.diag([3.0, 1.0, 0.2])
        axes, var = principal_axes(P)
        C = np.cov(P.T, ddof=1)
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        self.assertLess(float(np.abs(np.sort(var)[::-1] - w[order]).max()), 1e-9)
        for k in range(3):
            # 特征向量方向可能整体反号，比较绝对内积
            self.assertAlmostEqual(abs(float(np.dot(axes[:, k], V[:, order[k]]))),
                                   1.0, places=6)

    def test_needs_at_least_three_points(self):
        with self.assertRaises(ValueError):
            principal_axes(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            principal_axes(np.zeros((5, 2)))


class TestSlipDetectorDoesNotWindUpBeforeContact(unittest.TestCase):
    """接触刚建立、法向力还极小时，不得报「濒临滑移」。

    这条针对一个真实发生过的缺陷：余量定义为 |f_t|/(μ·f_n)，
    在 f_n→0 时趋于 1，被当成濒临滑移，自适应律于是在**还没夹上**的时候
    就把力一路加到上限（典型的绕卷）。症状是"轻工件正常、重工件夹持力爆表"，
    很容易被误当成"自适应参数没调好"。

    这里直接测判据函数的边界行为，不依赖任何具体场景。
    """

    class _FakeGripper:
        """最小替身：只提供 SlipDetector 需要的三个接口。"""

        def __init__(self, model, data, ids, face):
            self.model, self.data = model, data
            self._ids, self._face = ids, face

        def contacts_with(self, body):
            return self._ids

        def contact_force(self, body):
            return self._face

    def _detector_with(self, fn_scale: float):
        """搭一个真实接触，再按比例缩放法向力来扫过阈值附近。"""
        from armctrl.control.grasp import SlipDetector
        m = mujoco.MjModel.from_xml_string(_squeeze_rig(0.5, 0.6, 8.0))
        d = mujoco.MjData(m)
        d.ctrl[:] = 0.05
        for _ in range(1500):
            mujoco.mj_step(m, d)
        ids = [i for i in range(d.ncon)]
        gr = self._FakeGripper(m, d, ids, 1.0)
        det = SlipDetector.__new__(SlipDetector)
        det.gripper = gr
        det.object_body = "obj"
        det.margin_threshold = 0.85
        det.speed_threshold = 2e-3
        det.min_normal_for_margin = 0.05
        det.obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        return det, m, d, ids

    def test_no_contact_reports_no_slip(self):
        from armctrl.control.grasp import SlipDetector

        class Empty(self._FakeGripper.__mro__[0]):
            pass

        det, m, d, ids = self._detector_with(1.0)
        det.gripper._ids = []
        r = det.read()
        self.assertEqual(r.n_contacts, 0)
        self.assertFalse(r.incipient)
        self.assertFalse(r.sliding)
        self.assertEqual(r.margin, 0.0)

    def test_tiny_normal_force_does_not_trigger_incipient_slip(self):
        """把阈值抬到比实际法向力还高，等效于"接触极轻"的情形。"""
        det, m, d, ids = self._detector_with(1.0)
        r_normal = det.read()
        self.assertGreater(r_normal.normal_force, 0.0)
        # 真实夹持时余量应当是有意义的小数，且不触发濒临滑移
        self.assertLess(r_normal.margin, 0.85)
        self.assertFalse(r_normal.incipient)

        det.min_normal_for_margin = r_normal.normal_force * 10.0
        r_tiny = det.read()
        self.assertFalse(r_tiny.incipient,
                         "法向力低于下限时仍报濒临滑移，自适应律会绕卷到上限")
        self.assertEqual(r_tiny.margin, 0.0)
        # 反应性判据不受影响：它不依赖法向力
        self.assertEqual(r_tiny.n_contacts, r_normal.n_contacts)


# ==================================================================== 视觉伺服


class TestVisualServoSensorModel(unittest.TestCase):
    """延迟必须是**真的延迟**，不能用噪声去假装。

    两者对闭环的影响完全不同：噪声让输出毛躁但不改变相位，
    延迟会吃掉相位裕度，是导致振荡的元凶。所以要分别测。
    """

    def _rig(self, **kw):
        from armctrl.control.visual_servo import ObjectPoseSensor, SensorSpec
        XML = """<mujoco><worldbody>
          <body name="obj" pos="0 0 0"><freejoint/>
            <geom type="box" size=".02 .02 .02" mass="1"/></body>
          </worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(XML)
        d = mujoco.MjData(m)
        spec = SensorSpec(**{**dict(rate_hz=100.0, latency=0.05,
                                    pos_noise=0.0, max_range=10.0), **kw})
        return ObjectPoseSensor(m, d, "obj", spec, seed=0), m, d

    def test_reading_lags_the_truth_by_the_configured_latency(self):
        """让工件匀速运动，读数应当落后真值约 latency×速度。"""
        sensor, m, d = self._rig(latency=0.05, rate_hz=200.0)
        v = 0.5                                   # m/s，沿 x
        dt = 0.002
        errs = []
        for k in range(int(2.0 / dt)):
            t = k * dt
            d.qpos[0] = v * t
            mujoco.mj_forward(m, d)
            sensor.update(t, np.zeros(3))
            p, age = sensor.read(t)
            if p is not None and t > 0.5:
                errs.append(v * t - p[0])         # 真值减读数 = 落后的距离
                self.assertGreater(age, 0.0)
        lag = float(np.mean(errs))
        self.assertAlmostEqual(lag / (v * 0.05), 1.0, delta=0.2,
                               msg=f"读数滞后 {lag * 1e3:.1f} mm，"
                                   f"按 50 ms 延迟应为 {v * 0.05 * 1e3:.1f} mm")

    def test_zero_latency_error_is_only_one_sample_of_quantisation(self):
        """延迟设为 0 时，残留误差应当**只有一个采样周期**的量化滞后。

        容差不能拍脑袋定：工件按 0.3·sin(2πt) 运动，峰值速度 0.3·2π ≈ 1.88 m/s，
        一个 2 ms 采样周期就走 3.77 mm。所以 3.7 mm 量级的误差是**采样固有的**，
        不是实现有问题。判据取该界的 1.5 倍——足以放过量化，
        但多滞后一帧（7.5 mm）就会被抓到。
        """
        sensor, m, d = self._rig(latency=0.0, rate_hz=500.0)
        dt = 0.002
        amp, freq = 0.3, 1.0
        v_max = amp * 2 * np.pi * freq
        bound = 1.5 * v_max * dt
        worst, worst_age = 0.0, 0.0
        for k in range(500):
            t = k * dt
            d.qpos[0] = amp * np.sin(2 * np.pi * freq * t)
            mujoco.mj_forward(m, d)
            sensor.update(t, np.zeros(3))
            p, age = sensor.read(t)
            if p is not None:
                worst = max(worst, abs(p[0] - d.qpos[0]))
                worst_age = max(worst_age, age)
        self.assertLess(worst, bound,
                        f"误差 {worst * 1e3:.2f} mm 超过一拍量化界 {bound * 1e3:.2f} mm")
        self.assertLessEqual(worst_age, dt + 1e-9,
                             "零延迟时读数的年龄不该超过一个采样周期")

    def test_update_rate_limits_how_often_frames_arrive(self):
        sensor, m, d = self._rig(rate_hz=25.0, latency=0.0)
        dt = 0.002
        for k in range(int(1.0 / dt)):
            mujoco.mj_forward(m, d)
            sensor.update(k * dt, np.zeros(3))
        self.assertLessEqual(sensor.n_frames, 26)
        self.assertGreaterEqual(sensor.n_frames, 24)

    def test_out_of_range_object_is_not_seen(self):
        sensor, m, d = self._rig(max_range=0.5, latency=0.0)
        d.qpos[0] = 2.0
        mujoco.mj_forward(m, d)
        sensor.update(0.0, np.zeros(3))
        self.assertFalse(sensor.visible)
        self.assertEqual(sensor.n_frames, 0)


class TestVisualServoLaw(unittest.TestCase):
    """比例伺服的行为：单调收敛、限幅、死区。"""

    def test_converges_monotonically_without_delay(self):
        from armctrl.control.visual_servo import PositionVisualServo
        vs = PositionVisualServo(gain=5.0, step_max=1.0, deadband=1e-4)
        p = np.zeros(3)
        target = np.array([0.1, 0.0, 0.0])
        prev = np.inf
        for _ in range(2000):
            p = p + vs.step(p, target, 1e-3)
            e = float(np.linalg.norm(target - p))
            self.assertLessEqual(e, prev + 1e-12, "误差不该变大")
            prev = e
        self.assertLess(prev, 1e-3)

    def test_step_is_clipped(self):
        from armctrl.control.visual_servo import PositionVisualServo
        vs = PositionVisualServo(gain=1000.0, step_max=0.004, deadband=0.0)
        d = vs.step(np.zeros(3), np.array([1.0, 0.0, 0.0]), 1e-3)
        self.assertAlmostEqual(float(np.linalg.norm(d)), 0.004, places=12)

    def test_deadband_stops_micro_corrections(self):
        from armctrl.control.visual_servo import PositionVisualServo
        vs = PositionVisualServo(gain=5.0, step_max=0.01, deadband=1e-3)
        d = vs.step(np.zeros(3), np.array([5e-4, 0.0, 0.0]), 1e-3)
        self.assertTrue(np.allclose(d, 0.0))
        self.assertTrue(vs.converged)

    def test_larger_gain_times_latency_causes_overshoot(self):
        """λ·τ 接近 1 时闭环开始过冲。这条把那个经验判据变成可复核的事实。

        用一个纯滞后 + 比例反馈的最小模型，不依赖任何机械臂细节。
        """
        from armctrl.control.visual_servo import PositionVisualServo

        def peak_overshoot(gain, latency, dt=1e-3, T=6.0):
            vs = PositionVisualServo(gain=gain, step_max=1.0, deadband=0.0)
            target = np.array([0.1, 0.0, 0.0])
            n = max(int(round(latency / dt)), 1)
            buf = [np.zeros(3)] * n
            p = np.zeros(3)
            worst = 0.0
            for _ in range(int(T / dt)):
                p_seen = buf.pop(0)                # 控制器看到的是延迟后的自身位置
                buf.append(p.copy())
                p = p + vs.step(p_seen, target, dt)
                worst = max(worst, float(p[0] - target[0]))
            return worst / float(target[0])

        low = peak_overshoot(gain=3.0, latency=0.05)     # λτ = 0.15
        high = peak_overshoot(gain=25.0, latency=0.05)   # λτ = 1.25
        self.assertLess(low, 0.05, f"λτ=0.15 不该有明显过冲，实测 {low:.3f}")
        self.assertGreater(high, 0.30, f"λτ=1.25 应当明显过冲，实测 {high:.3f}")


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class TestConstrainedGraspFrame(unittest.TestCase):
    """固定接近方向的抓取位姿生成——「工件倒了还能不能抓」。

    oracle 独立性：判据全部是**几何真值**，直接由长方体的半尺寸解析给出，
    不读 constrained_grasp_frame 的任何中间量：
      • 需要的开口宽度必须等于长方体在开合轴上的真实投影范围；
      • 开合轴必须与接近方向严格正交；
      • 返回的 R 必须是正交且右手的。
    """

    HALF = (0.021, 0.021, 0.045)      # 42 × 42 × 90 mm，与 GraspScene 一致
    STROKE = 0.08                     # Panda 平行夹爪行程 80 mm

    #: 覆盖立着、两个方向倒下、倒下再绕竖直轴转、以及斜靠
    POSES = {
        "立着": np.eye(3),
        "倒向y": _rot("x", np.pi / 2),
        "倒向x": _rot("y", np.pi / 2),
        "倒后转45°": _rot("z", np.pi / 4) @ _rot("x", np.pi / 2),
        "斜靠30°": _rot("x", np.pi / 6),
        "任意姿态": _rot("z", 0.7) @ _rot("y", 1.1) @ _rot("x", 0.3),
    }

    def _points(self, R):
        return box_corner_points(self.HALF, (0.52, 0.0, 0.045), rot=R)

    def test_width_always_within_stroke(self):
        """任何倾倒姿态下，需要的开口都必须在夹爪行程内。

        ⭐ 这条是原缺陷的直接判据。原实现 `grasp_frame_from_points` 无条件取
        **最短主轴**当开合轴；工件躺倒后最短主轴变成竖直、与自上而下的接近
        方向平行，它会直接抛 ValueError，抓取位姿根本构造不出来。
        """
        for name, R in self.POSES.items():
            with self.subTest(name):
                _, _, w = constrained_grasp_frame(
                    self._points(R), approach=np.array([0.0, 0.0, -1.0]))
                self.assertLess(w, self.STROKE,
                                f"{name}: 需要开口 {w * 1000:.1f} mm 超过行程")

    def test_width_equals_true_projection(self):
        """开口宽度必须等于角点在开合轴上的**真实**投影范围（几何真值）。

        算错轴但侥幸落在行程内的情况会被这条抓住。
        """
        for name, R in self.POSES.items():
            with self.subTest(name):
                P = self._points(R)
                _, Rg, w = constrained_grasp_frame(P)
                proj = P @ Rg[:, 0]
                self.assertAlmostEqual(w, float(proj.max() - proj.min()),
                                       places=12)

    def test_width_is_the_smaller_of_the_two(self):
        """必须挑水平面内**更短**的那个方向，而不是随便一个正交方向。

        独立复算：把角点投到与接近方向垂直的平面，量出副轴方向上的范围，
        它必须 >= 开合轴方向上的范围。
        """
        for name, R in self.POSES.items():
            with self.subTest(name):
                P = self._points(R)
                _, Rg, w = constrained_grasp_frame(P)
                side = P @ Rg[:, 1]
                self.assertLessEqual(w, float(side.max() - side.min()) + 1e-12)

    def test_close_axis_orthogonal_to_approach(self):
        """开合轴必须与接近方向严格正交，否则俯冲下去会被工件顶住。"""
        a = np.array([0.0, 0.0, -1.0])
        for name, R in self.POSES.items():
            with self.subTest(name):
                _, Rg, _ = constrained_grasp_frame(self._points(R), approach=a)
                self.assertAlmostEqual(float(np.dot(Rg[:, 0], a)), 0.0, places=12)
                np.testing.assert_allclose(Rg[:, 2], a, atol=1e-12)

    def test_returns_proper_rotation(self):
        """返回的 R 必须是正交且右手（det=+1）的真旋转矩阵。"""
        for name, R in self.POSES.items():
            with self.subTest(name):
                _, Rg, _ = constrained_grasp_frame(self._points(R))
                np.testing.assert_allclose(Rg.T @ Rg, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(float(np.linalg.det(Rg)), 1.0, places=12)

    def test_original_version_fails_on_tipped_block(self):
        """记录原缺陷：通用版在工件躺倒时**无法给出**抓取位姿。

        这条不是在验证新实现，而是把"为什么需要新函数"钉死成可执行的事实。
        哪天有人想把新函数删掉换回旧的，这条会立刻提醒他代价是什么。
        """
        with self.assertRaises(ValueError):
            grasp_frame_from_points(self._points(self.POSES["倒向y"]),
                                    approach=np.array([0.0, 0.0, -1.0]))

    def test_side_approach_also_supported(self):
        """接近方向换成水平时同样成立——本函数不假定一定是俯冲。"""
        a = np.array([1.0, 0.0, 0.0])
        _, Rg, w = constrained_grasp_frame(self._points(np.eye(3)), approach=a)
        np.testing.assert_allclose(Rg[:, 2], a, atol=1e-12)
        self.assertAlmostEqual(float(np.dot(Rg[:, 0], a)), 0.0, places=12)
        self.assertLess(w, self.STROKE)

if __name__ == "__main__":
    unittest.main()
