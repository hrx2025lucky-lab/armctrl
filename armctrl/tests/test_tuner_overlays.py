"""调参台三维叠加标记的几何契约测试。

**为什么需要这组测试**

调参台在原生窗口里用一对小球表示「目标在哪 / 实际在哪」，讲解里多处写着
「看这两个球差多少」。但标记的尺寸此前从未被任何测试约束过，于是出现了
两个只有肉眼才发现得了的缺陷：

1. 目标球半径 34 mm（直径 68 mm），而夹爪张开时两指垫内表面只相距约 83 mm。
   球被卡在两指之间、几乎填满开口，观众会把它当成穿模的物体而不是标记。
2. 目标球（34 mm）比实际球（20 mm）大 14 mm，所以只有末端误差超过 14 mm
   时实际球才会露出目标球表面。而计算力矩在默认轨迹下的末端误差是零点几
   毫米——「看两个球差多少」这条教学指令在本场景的主控制器下从未成立过，
   而且画面上完全看不出存在第二个球。

**oracle 独立性**

两条判据都是纯几何，且不读被测实现的任何常数：

- 夹爪开口从 MuJoCo 模型的指垫几何体现场量出来，不引用 scenes.py 里的数字。
- 「球 A 完全埋在球 B 里」用的是 ``|pA − pB| + rA ≤ rB``，初中几何，
  与 overlays() 怎么实现无关。
- 「存在一条 1:1 连线」按几何端点判定（线段两端分别落在两个球心上），
  不按几何体的 Python 类型判定，因此换一种画法也照样成立。

放大后的误差箭头**不算**满足判据：观众从放大箭头上读不出真实间距。
"""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from armctrl.tuner.base import Arrow, Frame, Ghost, Sphere, Trail
from armctrl.tuner.scenes import SCENES
from armctrl.planning.scene import WRIST_CAMERA

#: 需要成对显示「目标 / 实际」的场景，以及两个位置属性的名字。
PAIRED_SCENES = {
    "tracking": ("_p_des", "_p"),
    "adaptive": ("_p_des", "_p"),
    "estimation": ("_p_true", "_p_hat"),
}


def _scene(name):
    return {s.name: s for s in SCENES}[name]()


def _segments(items) -> list[tuple[np.ndarray, np.ndarray]]:
    """把叠加几何摊平成线段列表，按几何而不是按类型判定。"""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for it in items:
        if isinstance(it, Arrow):
            segs.append((np.asarray(it.start, float).reshape(3),
                         np.asarray(it.end, float).reshape(3)))
        elif isinstance(it, Trail):
            pts = np.asarray(it.points, float)
            if pts.ndim == 2 and len(pts) >= 2:
                segs.extend((a, b) for a, b in zip(pts[:-1], pts[1:]))
        elif isinstance(it, Frame):
            p = np.asarray(it.pos, float).reshape(3)
            R = np.asarray(it.rot, float).reshape(3, 3)
            segs.extend((p, p + R[:, k] * it.size) for k in range(3))
    return segs


def _spheres(items) -> list[tuple[np.ndarray, float]]:
    return [(np.asarray(it.pos, float).reshape(3), float(it.radius))
            for it in items if isinstance(it, (Sphere, Ghost))]


def _gripper_opening(model, data) -> float:
    """现场量出两指垫之间最窄的间距（m）。

    独立 oracle：用 MuJoCo 自己的 ``mj_geomDistance`` 去量左右指垫参与碰撞的
    几何体之间的最短距离。不解析 geom_size（指垫是 mesh，它的 geom_size 不是
    包围盒语义，早先按 size 推内表面得到过 37 mm 这种明显错误的结果），
    也不引用 scenes.py / gripper.py 里的任何常数。
    """
    mujoco.mj_forward(model, data)
    sides = []
    for name in ("left_finger", "right_finger"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        gs = [g for g in range(model.ngeom)
              if int(model.geom_bodyid[g]) == bid
              and int(model.geom_contype[g]) == 1]
        assert gs, f"{name} 上没有参与碰撞的几何体"
        sides.append(gs)
    return min(mujoco.mj_geomDistance(model, data, a, b, 1.0, None)
               for a in sides[0] for b in sides[1])


class TestMarkerGeometry(unittest.TestCase):
    """标记尺寸相对夹爪的比例。"""

    def test_target_marker_does_not_occupy_the_gripper(self):
        """目标标记的直径必须明显小于夹爪开口。

        判据：直径 ≤ 开口的一半。理由不是审美——TCP 定义在两指抓取中心，
        标记画在那里是对的；但如果标记的尺寸和夹爪机构同量级，观众就无法
        区分「这是一个标记」和「这是一个卡在夹爪里的物体」。

        原实现目标球直径 68 mm、开口约 83 mm，占比 82%，在此判据下失败。
        """
        for name in PAIRED_SCENES:
            with self.subTest(scene=name):
                sc = _scene(name)
                model = sc.build()
                sc.reset()
                if sc.gripper is not None and sc.gripper_width is not None:
                    sc.gripper.command_width(sc.gripper_width)
                    for _ in range(600):
                        mujoco.mj_step(model, sc.data)
                opening = _gripper_opening(model, sc.data)
                self.assertGreater(opening, 0.05, "量出来的开口不合理")
                radii = [r for _, r in _spheres(sc.overlays())]
                self.assertTrue(radii, f"{name} 场景没有球形标记")
                self.assertLessEqual(
                    2.0 * max(radii), 0.5 * opening,
                    f"{name}：最大标记直径 {2 * max(radii) * 1e3:.1f} mm "
                    f"超过夹爪开口 {opening * 1e3:.1f} mm 的一半，"
                    f"画面上会像是卡在夹爪里的物体")


class TestPairVisibility(unittest.TestCase):
    """两个标记重叠时，画面上是否仍看得出存在两个点。"""

    #: 取一个「已经不算重合、但还没大到两球分离」的间距。
    #: 原实现的盲区是 34−20 = 14 mm，这个值落在盲区里；
    #: 现实现的盲区是 13−7 = 6 mm，这个值在盲区之外。
    GAP = 0.009

    def _prepare(self, name):
        sc = _scene(name)
        sc.build()
        sc.reset()
        key_t, key_a = PAIRED_SCENES[name]
        p_t = np.asarray(getattr(sc, key_t), float).reshape(3).copy()
        p_a = p_t + np.array([0.0, 0.0, -self.GAP])
        setattr(sc, key_t, p_t)
        setattr(sc, key_a, p_a)
        return sc, p_t, p_a

    def test_two_points_readable_while_spheres_overlap(self):
        """间距 9 mm 时，必须能看出画面上有两个点。

        满足其一即可：
          a) 两个球没有形成「完全包含」关系（内球露头了）；
          b) 存在一条 **1:1** 的线段，两端正好落在两个球心上。

        放大过的误差箭头不满足 (b)：它的终点不是实际点，
        观众从它的长度读不出真实间距。

        原实现在这三个场景下 (a)(b) 都不成立——两球完全重合成一个，
        画面上没有任何东西表明存在第二个点。
        """
        for name in PAIRED_SCENES:
            with self.subTest(scene=name):
                sc, p_t, p_a = self._prepare(name)
                items = sc.overlays()

                marks = [(p, r) for p, r in _spheres(items)
                         if min(np.linalg.norm(p - p_t),
                                np.linalg.norm(p - p_a)) < 1e-9]
                self.assertEqual(
                    len(marks), 2,
                    f"{name}：期望目标与实际各有一个球形标记，实际 {len(marks)} 个")

                (pa, ra), (pb, rb) = marks
                d = float(np.linalg.norm(pa - pb))
                engulfed = (d + ra <= rb) or (d + rb <= ra)

                linked = any(
                    (np.linalg.norm(s - p_t) < 1e-9 and np.linalg.norm(e - p_a) < 1e-9)
                    or (np.linalg.norm(s - p_a) < 1e-9
                        and np.linalg.norm(e - p_t) < 1e-9)
                    for s, e in _segments(items))

                self.assertTrue(
                    (not engulfed) or linked,
                    f"{name}：两点相距 {self.GAP * 1e3:.0f} mm 时，"
                    f"小球被大球完全吞没（r={ra * 1e3:.0f}/{rb * 1e3:.0f} mm）"
                    f"且没有 1:1 连线，画面上看不出存在两个点")

    def test_blind_zone_is_smaller_than_the_gap_used_by_lessons(self):
        """两球从「完全重合」到「露头」所需的间距（盲区）必须小于 9 mm。

        这是把上一条测试的容差写死之后的必然要求，单独列出来是为了在失败时
        直接给出盲区的数值，而不是只说「看不见」。

        原实现盲区 14 mm，失败。
        """
        for name in PAIRED_SCENES:
            with self.subTest(scene=name):
                sc, p_t, p_a = self._prepare(name)
                marks = [(p, r) for p, r in _spheres(sc.overlays())
                         if min(np.linalg.norm(p - p_t),
                                np.linalg.norm(p - p_a)) < 1e-9]
                radii = sorted(r for _, r in marks)
                self.assertEqual(len(radii), 2)
                blind = radii[1] - radii[0]
                self.assertLess(
                    blind, self.GAP,
                    f"{name}：盲区 {blind * 1e3:.1f} mm —— "
                    f"误差小于它时实际标记完全埋在目标标记里")


class TestSceneLighting(unittest.TestCase):
    """布光契约。

    这里锁两件在画面上很显眼、但没人会去写断言的事：

    1. **场景里到底有几盏灯。** menagerie 的 panda.xml 自带一盏
       ``<light name="top" mode="trackcom"/>``：聚光、投影、diffuse 0.7，
       而且**跟着机器人质心移动**。不删掉它，本模块精心设计的三点布光
       实际是四点，最强的一盏还在动，连杆上会出现随姿态抖动的锯齿状阴影。
    2. **主光必须和默认相机同侧。** 逆光时自阴影的分界线正好落在朝向相机的
       那一面，而阴影贴图在掠射角下必然走样。实测：换到同侧后锯齿消失，
       只提高阴影分辨率压不掉。

    oracle 独立：全部从**编译后的 mjModel** 读，不引用 scene.py 里的常数。
    """

    def test_exactly_one_shadow_caster(self):
        """三盏灯、且只有一盏投影。

        原实现有 4 盏、2 盏投影，在此判据下失败。
        """
        for cls in SCENES:
            with self.subTest(scene=cls.name):
                sc = cls()
                model = sc.build()
                casters = [i for i in range(model.nlight)
                           if bool(model.light_castshadow[i])]
                self.assertEqual(
                    model.nlight, 3,
                    f"{cls.name}：灯的数量是 {model.nlight}，"
                    f"多出来的多半是模型文件自带、没被删掉的那盏")
                self.assertEqual(
                    len(casters), 1,
                    f"{cls.name}：有 {len(casters)} 盏灯在投影。"
                    f"多盏投影会叠出互相矛盾的阴影")

    def test_key_light_is_on_the_same_side_as_the_default_camera(self):
        """主光来向与默认相机方位角的夹角必须小于 90°（同侧）。

        原实现主光来向方位角约 320°、默认相机 138°，相差约 182°——
        正对着逆光，在此判据下失败。
        """
        for cls in SCENES:
            with self.subTest(scene=cls.name):
                sc = cls()
                model = sc.build()
                idx = [i for i in range(model.nlight)
                       if bool(model.light_castshadow[i])][0]
                d = np.asarray(model.light_dir[idx], float)
                # 先挡住退化情况：几乎垂直向下的光没有「哪一侧」可言，
                # 而且正是本模块 docstring 里说要避免的单顶光
                # （各连杆的圆柱面亮度一致，看不出关节朝向）。
                # 这一条是被变异测试逼出来的：np.arctan2(-0.0, -0.0) 返回 −180°，
                # 于是一盏纯顶光会**碰巧**满足下面的同侧判据。
                horiz = float(np.linalg.norm(d[:2]))
                self.assertGreater(
                    horiz, 0.25,
                    f"{cls.name}：主光水平分量只有 {horiz:.3f}，接近单顶光，"
                    f"照不出关节朝向，也谈不上在相机哪一侧")
                # 光「来自」的方向是传播方向的反向
                light_az = np.degrees(np.arctan2(-d[1], -d[0]))
                cam_az = float(cls.camera["azimuth"])
                gap = abs((light_az - cam_az + 180.0) % 360.0 - 180.0)
                self.assertLess(
                    gap, 90.0,
                    f"{cls.name}：主光来向 {light_az:.0f}° 与默认相机 "
                    f"{cam_az:.0f}° 相差 {gap:.0f}°，属于逆光，"
                    f"自阴影分界线会落在朝向相机的一面")

    #: 判据参照的渲染高度（像素）。原生窗口是可缩放的，这里取 768 作为基准，
    #: 与改版前离屏渲染用的高度一致。窗口拉得更高时每屏幕像素更小，
    #: 阴影阶梯有可能重新变得可见——这是阴影贴图法的固有限制，不是缺陷。
    REF_RENDER_HEIGHT = 768

    def test_shadow_texel_is_sub_pixel_at_the_default_camera(self):
        """阴影贴图的一个像素，在最近的默认相机距离上不得超过一个屏幕像素。

        判据是推出来的，不是拍的：阴影边缘的阶梯宽度就等于一个贴图像素在世界里
        的尺寸；只要它比一个屏幕像素还小，阶梯就落在亚像素里，看不出来。

            贴图像素 = 2 · stat.extent · shadowclip / shadowsize
            屏幕像素 = 2 · d · tan(fovy/2) / 渲染高度

        实测对照（fovy 45°、最近默认相机 0.72 m、768 px）：
          改版前 shadowclip=2.0 → 贴图像素 1.56 mm，是屏幕像素 0.78 mm 的 2 倍，
          阶梯跨两个屏幕像素，肉眼可见；改版后 0.63 mm，落到亚像素。
        """
        sc = SCENES[0]()
        model = sc.build()
        texel = (2.0 * float(model.stat.extent) * float(model.vis.map.shadowclip)
                 / float(model.vis.quality.shadowsize))
        d_min = min(float(c.camera["distance"]) for c in SCENES)
        screen = (2.0 * d_min * np.tan(np.radians(float(model.vis.global_.fovy)) / 2)
                  / self.REF_RENDER_HEIGHT)
        self.assertLess(
            texel, screen,
            f"阴影贴图每像素 {texel * 1e3:.2f} mm，而最近默认相机（{d_min:.2f} m）"
            f"上一个屏幕像素只有 {screen * 1e3:.2f} mm —— 阴影边缘会出现"
            f"跨 {texel / screen:.1f} 个像素的阶梯")


class TestWristCamera(unittest.TestCase):
    """眼在手上相机的几何契约。

    oracle 独立性：只读取编译后 MjModel/MjData 的相机外参并做解析几何，
    不引用 ``_add_wrist_camera`` 里的安装位置、目标点或四元数。
    """

    def test_every_tuner_scene_has_camera_attached_to_hand(self):
        """11 个场景都必须有同名相机，且父刚体确实是 hand。

        如果相机误加到 worldbody，Home 构型下静态截图仍可能看起来正确，
        但机械臂一运动画面就不会跟着夹爪走。这条直接检查刚体父子关系。
        """
        for cls in SCENES:
            with self.subTest(scene=cls.name):
                sc = cls()
                model = sc.build()
                cid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
                hand = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "hand")
                self.assertGreaterEqual(cid, 0, f"{cls.name} 缺少夹爪相机")
                self.assertEqual(
                    int(model.cam_bodyid[cid]), hand,
                    f"{cls.name} 的夹爪相机没有挂在 hand 上，会留在世界原地")

    def test_tcp_is_inside_camera_field_of_view(self):
        """TCP 必须落在相机纵向视场内，否则贴近工件时只会看到画面外。

        独立计算相机光轴与「相机→TCP」射线的夹角，判据为
        ``angle < fovy/2``。MuJoCo 相机沿自身 ``-z`` 看，正负号也由这条锁住：
        若错误地沿 ``+z`` 解释相机方向，夹角会接近 180°。
        """
        sc = next(cls for cls in SCENES if cls.name == "grasp")()
        model = sc.build()
        sc.reset()
        cid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        cam_pos = sc.data.cam_xpos[cid]
        cam_R = sc.data.cam_xmat[cid].reshape(3, 3)
        optical_axis = -cam_R[:, 2]
        ray = sc.robot.tcp_position() - cam_pos
        ray /= np.linalg.norm(ray)
        angle = np.degrees(np.arccos(np.clip(
            float(np.dot(optical_axis, ray)), -1.0, 1.0)))
        self.assertLess(
            angle, 0.5 * float(model.cam_fovy[cid]),
            f"TCP 与光轴夹角 {angle:.1f}°，超过半视场 "
            f"{0.5 * float(model.cam_fovy[cid]):.1f}°")

    def test_camera_pose_is_finite_and_proper(self):
        """相机外参必须有限，旋转矩阵必须正交且 det=+1。"""
        sc = next(cls for cls in SCENES if cls.name == "grasp")()
        model = sc.build()
        sc.reset()
        cid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        R = sc.data.cam_xmat[cid].reshape(3, 3)
        self.assertTrue(np.all(np.isfinite(sc.data.cam_xpos[cid])))
        self.assertTrue(np.all(np.isfinite(R)))
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)


class TestOverlayContract(unittest.TestCase):
    """全部场景的叠加几何都得是 server 认识的类型且数值有限。

    这是契约测试，不针对上面那两个缺陷：server._draw_overlays 对未知类型
    直接抛 TypeError，而 overlays() 跑在渲染线程里，抛异常只会让画面停住，
    不容易定位。
    """

    KNOWN = (Arrow, Sphere, Ghost, Frame, Trail)

    def test_every_scene_emits_only_known_finite_geometry(self):
        for cls in SCENES:
            with self.subTest(scene=cls.name):
                sc = cls()
                model = sc.build()
                sc.reset()
                for i in range(50):
                    q = sc.data.qpos[sc.robot.qpos_idx].copy()
                    v = sc.data.qvel[sc.robot.qvel_idx].copy()
                    sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(
                        i * sc.dt, q, v)
                    if sc.gripper is not None and sc.gripper_width is not None:
                        sc.gripper.command_width(sc.gripper_width)
                    mujoco.mj_step(model, sc.data)
                for it in sc.overlays():
                    self.assertIsInstance(it, self.KNOWN)
                    for field in ("pos", "start", "end", "points"):
                        val = getattr(it, field, None)
                        if val is None:
                            continue
                        arr = np.asarray(val, float)
                        self.assertTrue(np.all(np.isfinite(arr)),
                                        f"{cls.name} 的 {type(it).__name__}."
                                        f"{field} 出现非有限值")


class TestContactMarkersDefaultOff(unittest.TestCase):
    """接触点 / 接触力标记必须**默认关闭**，且开关必须真的接到两个旗标上。

    **为什么要锁这条**

    这两个旗标曾经默认打开。抓取场景夹爪合拢后 ``data.ncon`` 稳定在 32，
    32 个橙色圆盘（``vis.rgba.contactpoint`` = 0.9/0.6/0.2）叠在两个指尖上
    会连成一片，把指垫、工件棱线、插入间隙全糊掉。把单个圆盘从 60 mm 缩到
    12 mm 并不解决问题——缩的是单个，重叠总面积没变。

    **oracle 独立性**

    - ``ncon = 32`` 是在真实场景里跑出来的，不引用 scenes.py 里的任何常数。
    - 旗标状态用一个**假 viewer** 读，只看 ``_apply_contacts`` 往
      ``opt.flags`` 写了什么，不看 server.py 的中间变量。
    - 文案一致性直接读 index.html / 讲义原文，与实现无关。
    """

    class _FakeOpt:
        def __init__(self):
            self.flags = {}

    class _FakeViewer:
        """只提供 ``lock()`` 和 ``opt``，不碰 GL——测试无需显示器。"""

        def __init__(self, opt):
            self.opt = opt

        def lock(self):
            import contextlib
            return contextlib.nullcontext()

    def _runner(self):
        """不跑 __init__（那会真的开窗口），只装配 _apply_contacts 需要的字段。"""
        from armctrl.tuner.server import Runner
        r = Runner.__new__(Runner)
        opt = self._FakeOpt()
        r.viewer = self._FakeViewer(opt)
        r.show_contacts = False
        return r, opt

    def test_default_is_off(self):
        import inspect
        from armctrl.tuner.server import Runner
        src = inspect.getsource(Runner.__init__)
        self.assertIn("self.show_contacts = False", src,
                      "Runner.__init__ 必须把 show_contacts 默认设成 False；"
                      "默认打开会让抓取场景的 32 个接触点糊住指尖")

    def test_flag_follows_show_contacts(self):
        r, opt = self._runner()
        cp = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        cf = mujoco.mjtVisFlag.mjVIS_CONTACTFORCE

        r._apply_contacts()
        self.assertIs(opt.flags[cp], False, "show_contacts=False 时接触点必须关")
        self.assertIs(opt.flags[cf], False, "show_contacts=False 时接触力必须关")

        r._set_contacts("toggle")
        self.assertTrue(r.show_contacts)
        self.assertIs(opt.flags[cp], True, "toggle 后接触点必须开")
        self.assertIs(opt.flags[cf], True, "toggle 后接触力必须开")

        r._set_contacts("toggle")
        self.assertFalse(r.show_contacts)
        self.assertIs(opt.flags[cp], False, "再 toggle 必须回到关")

        r._set_contacts(True)
        self.assertIs(opt.flags[cp], True)
        r._set_contacts(False)
        self.assertIs(opt.flags[cp], False)

    def test_apply_is_safe_without_viewer(self):
        """换场景时 viewer 会短暂为 None，此时不能抛异常。"""
        from armctrl.tuner.server import Runner
        r = Runner.__new__(Runner)
        r.viewer = None
        r.show_contacts = True
        r._apply_contacts()                      # 不抛异常即通过

    def test_key_V_and_http_route_are_wired(self):
        import inspect
        from armctrl.tuner.server import Runner, Handler
        self.assertIn('ord("V")', inspect.getsource(Runner._on_key),
                      "MuJoCo 窗口必须有 V 键切换接触点标记")
        self.assertIn('"contacts"', inspect.getsource(Runner._handle),
                      "命令队列必须能处理 contacts 命令")
        self.assertIn("/contacts", inspect.getsource(Handler.do_POST),
                      "网页必须有 /contacts 路由，否则按钮点了没反应")

    def test_grasp_really_has_many_contacts(self):
        """默认关闭的理由必须是真的：夹爪合拢后接触点数远多于个位数。

        这里不引用任何写死的 32，只要求「多到会糊住画面」。
        """
        from armctrl.tuner.scenes import SCENE_BY_NAME
        sc = SCENE_BY_NAME["grasp"]()
        model = sc.build()
        sc.reset()
        n_sub = int(round(sc.dt / model.opt.timestep))
        t, peak = 0.0, 0
        for _ in range(2600):                    # 跑到夹爪已经夹住工件
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
            for _ in range(n_sub):
                mujoco.mj_step(model, sc.data)
            t += sc.dt
            peak = max(peak, int(sc.data.ncon))
        self.assertGreaterEqual(
            peak, 20,
            f"抓取时接触点峰值只有 {peak}，少于 20。"
            "如果接触点真的这么少，就该重新考虑要不要默认打开")

    def test_scene_switch_keeps_the_setting(self):
        """换场景会重开窗口。若 _tune_viewer 不再同步，用户打开的标记会悄悄丢掉。

        新窗口的 MuJoCo 出厂默认本来就是「关」，所以这个回归在
        show_contacts=False 时**看不出来**——只有用户手动打开过才暴露。
        因此这里在源码层直接锁住调用关系。
        """
        import inspect
        from armctrl.tuner.server import Runner
        self.assertIn("_apply_contacts", inspect.getsource(Runner._tune_viewer),
                      "_tune_viewer 必须调用 _apply_contacts，"
                      "否则切场景后用户打开过的接触点标记会丢失")
        self.assertNotIn(
            "self.show_contacts = False", inspect.getsource(Runner._load),
            "_load 不能重置 show_contacts：视角是跨场景保留的，标记也应一致")

    def test_docs_say_default_off(self):
        """讲义和网页说明不能还写着「默认已开」。"""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        targets = {
            "armctrl/tuner/index.html": "默认关闭",
            "armctrl_讲解/08_MuJoCo窗口操作.md": "默认是「关」的",
        }
        for rel, must in targets.items():
            path = root / rel
            if not path.exists():
                continue  # 讲义不随公开仓库分发，见 tests/_docs.py
            text = path.read_text(encoding="utf-8")
            self.assertIn(must, text, f"{rel} 没有说明接触点标记默认关闭")
            self.assertNotIn(
                "接触点与接触力已默认打开", text,
                f"{rel} 还留着「默认打开」的旧说法，与代码不一致")


if __name__ == "__main__":
    unittest.main()
