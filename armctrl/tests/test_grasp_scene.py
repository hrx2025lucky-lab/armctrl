"""GraspScene 的**在线适应**行为测试。

这一组测的不是 `control/grasp.py` 里的几何函数（那些在 test_grasp.py），
而是抓取场景把它们**接进闭环之后**的行为：标记有没有跟着感知走、
工件被碰倒之后手腕会不会转过去、以及正方形截面下会不会乱转。

被它们钉死的两个真实缺陷
----------------------
1. `p_grasp`（画面上的绿球与抓取坐标架）在 reset 里赋值一次之后再没被写过。
   闭环模式下机械臂其实精准跟上了被挪开的工件（末端误差 2 mm），
   但标记还钉在标称位置、离工件 60 mm。**机器人是对的，画面在骗人**——
   而这个标记是使用者唯一能用来判断「机器人以为工件在哪」的参照物。
2. 抓取姿态只在 `_enter(PH_ALIGN)` 时算一次。工件在对准或下降途中被碰倒，
   当前这一轮会用着过期的开合方向一路走完、抓空，然后才重抓。
   看到的现象就是「先照旧动作原样演一遍」。

oracle 独立性
------------
- 工件真实位姿一律用 MuJoCo 的 `data.xpos` / `data.xquat` 现场读，
  不经过场景自己的 `_obj_pose()`。
- 「哪个水平方向更窄」不调用 `constrained_grasp_frame`，
  而用长方体的**支撑函数**独立算：沿单位方向 u 的跨度
  等于 2·Σ|h_i·(R·e_i · u)|。这是凸几何的标准结论，
  与被测实现没有任何共享代码。
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from armctrl.tuner.scenes import SCENES

_GRASP = [s for s in SCENES if s.name == "grasp"][0]

#: 绕世界 x 轴转 90°：工件由立变躺，长边 90 mm 转到水平面内
_TIP_QUAT = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])


# ------------------------------------------------------------------ 蠕变实验台
#: 一次悬停实验要跑 17~21 秒墙钟时间，好几个测试要用同一批数据，缓存起来。
_CREEP_CACHE = {}


def _creep_run(tmax, fext=0.0, impratio=None, grip=8.0, mu=None):
    """跑一次"抓起来 → 悬停不动"，返回工件相对夹爪的下滑情况。

    ⭐ 关键设计：`fext` 是通过 `data.xfrc_applied` 直接加在工件刚体上的
    竖直向下外力，**完全不经过场景的控制通路**。也就是说它是一个
    真正独立的扰动源，而不是"把某个控制参数调坏"。这一点很重要——
    用控制参数造出来的"故障"，本质上还是在测同一份实现。

    参数
    ----
    tmax : float
        整段仿真时长（秒）。悬停段大约是 tmax − 7.7 s。
    fext : float
        悬停期间施加的竖直向下外力（N）。0 表示只有重力。
    impratio : float or None
        MuJoCo 的切向/法向约束刚度比。None 表示用模型默认值 1.0。
    grip : float
        夹持力指令（N）。

    返回
    ----
    dict: total（总下滑 mm，负号表示往下）、rate（mm/s）、dur（悬停时长 s）、
          mu（实测有效摩擦系数）、N（实测法向力 N）
    """
    key = (round(tmax, 3), round(fext, 6),
           None if impratio is None else round(impratio, 6), round(grip, 3),
           None if mu is None else round(mu, 6))
    if key in _CREEP_CACHE:
        return _CREEP_CACHE[key]

    sc = _GRASP()
    model = sc.build()
    if impratio is not None:
        model.opt.impratio = impratio
    sc.set("mode", "fixed")
    sc.set("grip_cmd", grip)
    sc.set("shake", 0.0)
    if mu is not None:
        sc.set("obj_mu", mu)
    sc.reset()

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    dt = model.opt.timestep
    rel0 = None
    t_hold0 = None
    rel_last = 0.0
    t_last = 0.0
    mu_seen, n_seen, mu_raw = [], [], []

    for k in range(int(tmax / dt)):
        t = k * dt
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)

        holding = getattr(sc, "_phase", None) == sc.PH_HOLD
        sc.data.xfrc_applied[obj_bid, 2] = -fext if holding else 0.0
        mujoco.mj_step(model, sc.data)

        if holding:
            # ⭐ 滑移必须在**夹爪坐标系**里量：夹爪在负载下会低头，
            # 只看世界系高度差的话，夹爪转 7.6° 就会被误读成 7 mm 的滑落。
            # 这里用 FK 现算夹爪位姿，不读场景自己算好的中间量。
            p_ee, R_ee = sc.robot.fk(q)
            rel = float((R_ee.T @ (sc.data.xpos[obj_bid] - p_ee))[2]) * 1e3
            if rel0 is None:
                rel0, t_hold0 = rel, t
            rel_last, t_last = rel - rel0, t
            if k % 200 == 0:
                tm = sc.telemetry(t, q, v)
                mu_seen.append(float(tm.get("mu_eff", np.nan)))
                n_seen.append(float(tm.get("grip_force", np.nan)))
                # MuJoCo 自己合成好的接触摩擦，绕开 grasp.py 的任何计算
                for i in range(sc.data.ncon):
                    c = sc.data.contact[i]
                    if sc.obj_gid in (int(c.geom1), int(c.geom2)):
                        mu_raw.append(float(c.friction[0]))

    dur = max(t_last - (t_hold0 or 0.0), 1e-9)
    out = {
        "total": rel_last,
        "rate": abs(rel_last) / dur,
        "dur": dur,
        "mu": float(np.nanmedian(mu_seen)) if mu_seen else float("nan"),
        "N": float(np.nanmedian(n_seen)) if n_seen else float("nan"),
        "mu_raw": sorted({round(x, 6) for x in mu_raw}),
    }
    _CREEP_CACHE[key] = out
    return out


def box_extent_along(half, R, u):
    """长方体沿单位方向 u 的跨度（独立 oracle）。

    凸体的支撑函数：长方体在方向 u 上的半跨度是 Σ|h_i·(R·e_i · u)|，
    因为最远的角点就是每个轴分量都取同号的那一个。跨度是它的两倍。
    """
    h = np.asarray(half, dtype=float)
    R = np.asarray(R, dtype=float)
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    return 2.0 * float(np.sum(np.abs(h * (R.T @ u))))


def _true_obj(sc):
    """工件真实位置与姿态，直接读 MuJoCo，不经过被测场景的接口。"""
    bid = mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_BODY,
                            sc.obj.body_name())
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, sc.data.xquat[bid])
    return sc.data.xpos[bid].copy(), R.reshape(3, 3)


def run_scene(tmax, tip=False, **params):
    """跑一段抓取，返回场景与逐阶段记录。

    tip=True 时把工件摆成躺倒且**恰好贴着桌面**（不靠自由沉降），
    这样初始条件是确定的，测试不会因为沉降过程的随机性而抖。
    """
    sc = _GRASP()
    model = sc.build()
    for k, val in params.items():
        sc.set(k, val)
    sc.reset()
    if tip:
        sc.data.qpos[sc.obj_qadr + 3:sc.obj_qadr + 7] = _TIP_QUAT
        # 躺倒后半高变成原来的 y 半宽
        sc.data.qpos[sc.obj_qadr + 2] = sc.OBJ_HALF[1]
        mujoco.mj_forward(model, sc.data)
    dt = model.opt.timestep
    rec = {"need_at_start": None, "axis_at_descend": None,
           "axis_hist": [], "max_axis_swing": 0.0,
           "p_obj_at_reset": _true_obj(sc)[0].copy()}
    axis0 = None
    for k in range(int(tmax / dt)):
        t = k * dt
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
        mujoco.mj_step(model, sc.data)
        if rec["need_at_start"] is None and sc._need_now is not None and k > 5:
            rec["need_at_start"] = sc._need_now
        if sc._phase == sc.PH_DESCEND and rec["axis_at_descend"] is None:
            rec["axis_at_descend"] = sc._R_des[:, 1].copy()
        if sc._phase <= sc.PH_DESCEND and k % 50 == 0:
            a = sc._R_des[:, 1].copy()
            if axis0 is None:
                axis0 = a
            # 开合轴是直线不是矢量，±同一方向等价，所以比 |cos|
            swing = np.degrees(np.arccos(np.clip(abs(float(a @ axis0)), 0, 1)))
            rec["max_axis_swing"] = max(rec["max_axis_swing"], float(swing))
    return sc, rec


# ================================================ 缺陷 1：标记必须跟着感知走


class TestGraspMarkerTracksPerception(unittest.TestCase):
    """绿球画的是「机器人以为该去抓哪里」，它必须与控制器用的是同一个量。"""

    JITTER = 0.06        # 把工件沿 y 挪开 60 mm

    @classmethod
    def setUpClass(cls):
        cls.sc_v, cls.rec_v = run_scene(3.0, obj_jitter=cls.JITTER,
                                        closed_loop="visual")
        cls.sc_o, cls.rec_o = run_scene(3.0, obj_jitter=cls.JITTER,
                                        closed_loop="openloop")

    def test_scenario_is_non_degenerate(self):
        """前置检查：obj_jitter 确实把工件挪离了标称位置。

        没有这一条，后面两条都可能因为「工件本来就在标称位置」而空过。
        用 reset 之后**还没开始运动**时的真值比，不受机械臂推挤的影响。
        """
        d = float(np.linalg.norm(
            self.rec_v["p_obj_at_reset"][:2] - self.sc_v.p_nominal[:2]))
        self.assertAlmostEqual(d, self.JITTER, delta=1e-6)

    def test_visual_marker_follows_displaced_object(self):
        """闭环：工件被挪开 60 mm，标记必须跟过去。

        原缺陷下这里是 60 mm——标记纹丝不动地留在标称位置。
        容差取 15 mm：默认观测噪声 2 mm，留足余量后仍能把 60 mm 判死。
        """
        p_true, _ = _true_obj(self.sc_v)
        gap = float(np.linalg.norm(self.sc_v.p_grasp[:2] - p_true[:2]))
        self.assertLess(gap, 0.015,
                        f"闭环下标记偏离工件 {gap * 1e3:.1f} mm，说明它没跟着感知走")

    def test_openloop_marker_stays_at_nominal(self):
        """开环：标记**必须**恒等于标称位置。

        这条是反向约束。开环看不见工件，标记要是也跟着工件跑，
        画面就会显示成「它知道工件在哪却偏不去」——那是假的。
        标记不动，正是开环失效的可视化证据，不能被"修好"。

        判据写成「恒等于 p_nominal」而不是「离工件足够远」：
        后者会被物理污染——开环下机械臂会把工件推开几毫米，
        那个距离取决于碰撞过程，不是本条要约束的性质。
        """
        np.testing.assert_allclose(self.sc_o.p_grasp[:2],
                                   self.sc_o.p_nominal[:2], atol=1e-12)


# ============================================ 缺陷 2：碰倒之后必须当场转腕


class TestWidthAlongAxis(unittest.TestCase):
    """`_width_along` 是转腕判据的唯一输入，先把它本身钉死。"""

    def test_matches_support_function(self):
        """与长方体支撑函数逐个方向比对。"""
        half = _GRASP.OBJ_HALF
        rng = np.random.default_rng(7)
        for i in range(12):
            with self.subTest(i=i):
                # 随机旋转：QR 分解取正交阵，再修成右手系
                Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1.0
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                from armctrl.control.grasp import box_corner_points
                pts = box_corner_points(half, (0.3, -0.1, 0.2), rot=Q)
                got = _GRASP._width_along(pts, u)
                want = box_extent_along(half, Q, u)
                self.assertAlmostEqual(got, want, places=12)

    def test_tipped_block_is_too_wide_along_old_axis(self):
        """这条是转腕的**动机**：躺倒后沿旧开合轴需要 90 mm，超过 80 mm 行程。

        数字自己独立算：工件 42×42×90，绕 x 转 90° 后长边转到世界 y。
        """
        half = _GRASP.OBJ_HALF
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        along_y = box_extent_along(half, R, [0, 1, 0])
        along_x = box_extent_along(half, R, [1, 0, 0])
        self.assertAlmostEqual(along_y, 2 * half[2], places=12)   # 90 mm
        self.assertAlmostEqual(along_x, 2 * half[0], places=12)   # 42 mm
        self.assertGreater(along_y, 0.080)                        # 超过行程
        self.assertLess(along_x, 0.080)


class TestReorientsOnTippedObject(unittest.TestCase):
    """工件一开始就躺倒时，第一次尝试就该成功——不该先失败一轮。"""

    @classmethod
    def setUpClass(cls):
        cls.sc, cls.rec = run_scene(9.0, tip=True, closed_loop="visual")

    def test_detects_that_old_axis_no_longer_fits(self):
        """判据量必须先报出 90 mm，否则后面转腕就是碰巧转对的。"""
        self.assertGreater(self.rec["need_at_start"], 0.080,
                           "没能识别出沿旧开合轴夹不下")

    def test_closing_axis_turns_to_the_short_side(self):
        """下降之前，开合轴必须已经转到水平面内更窄的那个方向。

        oracle 用支撑函数独立判断哪个方向更窄，不问被测实现。
        """
        axis = self.rec["axis_at_descend"]
        self.assertIsNotNone(axis, "始终没有进入下降阶段")
        _, R_true = _true_obj(self.sc)
        w_cmd = box_extent_along(self.sc.OBJ_HALF, R_true, axis)
        self.assertLess(w_cmd, self.sc.gripper.width_max,
                        f"下降时的开合方向需要 {w_cmd * 1e3:.0f} mm，夹不下")
        self.assertAlmostEqual(w_cmd, 2 * self.sc.OBJ_HALF[0], delta=3e-3)

    def test_grasped_on_first_attempt(self):
        """抓起来了，而且**没有重抓**。

        原缺陷下第一轮必然抓空（用着立姿算出的开合方向），
        要等 `_object_lost` 超时才退回重抓，`_n_retry` ≥ 1。
        """
        p_true, _ = _true_obj(self.sc)
        self.assertGreater(float(p_true[2]), 0.08,
                           "工件没有被提起来")
        self.assertEqual(self.sc._n_retry, 0,
                         "第一次尝试就该成功，不该先失败一轮再重抓")


class TestUprightSquareDoesNotFlip(unittest.TestCase):
    """正方形截面下**不能**乱转腕——这是判据用物理量而不是用轴向的理由。"""

    def test_no_spurious_reorientation(self):
        """工件立着时横截面 42×42，两个方向一样宽。

        主轴分解挑哪一个在数值上是任意的，工件稍微偏航一点就会翻 90°。
        判据要是写成「开合轴变了就重规划」，手腕会毫无理由地来回甩。
        这里要求接触前开合轴的累计摆动不超过 15°。
        """
        _, rec = run_scene(4.0, closed_loop="visual")
        self.assertLess(rec["max_axis_swing"], 15.0,
                        f"开合轴摆了 {rec['max_axis_swing']:.0f}°，"
                        "正方形截面下不该发生")


# ================================ 缺陷 3：抓稳之后工件不该在指间匀速往下爬


class TestCreepAgainstTheTaskSpec(unittest.TestCase):
    """⭐⭐⭐ 悬停蠕变：阈值必须来自**任务规格**，不能来自"实现量出来多少"。

    ⚠️ 第五轮审核 P1-x「蠕变」条：旧测试的阈值 0.10 mm/s 是这么来的——
    先跑一遍看到 0.026，再跑一个坏工况看到 0.331，然后在中间取个数。
    **这是用实现给自己划及格线**：实现变差一点，及格线跟着变松，
    永远不会红。而且 `< 0.10 mm/s` 这个形式**根本不证明收敛**——
    一个匀速下滑 0.09 mm/s 的工件也能通过，可它一直在滑。

    ⭐ 正确的做法是**从任务倒推**：

        搬运悬停时长        T      = 20 s      （任务给的）
        放置位置精度        δ      = ±1.0 mm   （装配孔间隙给的）
        滑移可占的份额      share  = 50 %      （另一半留给标定与伺服）
        ⇒ 蠕变位移预算              = 0.5 mm
        ⇒ 蠕变速率预算              = 0.5 / 20 = 0.025 mm/s

    这四个数**没有一个是从仿真里量出来的**，全部来自任务定义。
    实现达不达标是实现的事，跟及格线怎么划无关。
    """

    #: —— 任务规格（唯一的事实来源，改这里等于改需求）——
    T_HOLD_S = 20.0            # 搬运悬停时长
    PLACE_TOL_MM = 1.0         # 放置位置精度 ±1.0 mm
    SLIP_SHARE = 0.5           # 滑移最多吃掉一半误差预算
    #: 由上面三个数**算出来**，不是填进来的
    CREEP_BUDGET_MM = PLACE_TOL_MM * SLIP_SHARE
    CREEP_RATE_BUDGET = CREEP_BUDGET_MM / T_HOLD_S      # = 0.025 mm/s

    def test_the_budget_is_derived_from_the_spec_not_from_a_measurement(self):
        """把"预算是算出来的"这件事本身钉住。

        有人日后想放宽阈值，就必须去改 T / δ / share 之一——
        也就是**必须去改需求**，而不能偷偷把 0.025 改成 0.05。
        """
        self.assertAlmostEqual(self.CREEP_BUDGET_MM,
                               self.PLACE_TOL_MM * self.SLIP_SHARE, places=12)
        self.assertAlmostEqual(self.CREEP_RATE_BUDGET,
                               self.CREEP_BUDGET_MM / self.T_HOLD_S, places=12)
        self.assertAlmostEqual(self.CREEP_RATE_BUDGET, 0.025, places=12)

    def test_the_default_solver_setting_does_not_meet_the_spec(self):
        """⚠️ **诚实记录：默认配置不达标**（实测 0.040，预算 0.025）。

        与其把及格线放宽到能过，不如把不达标**钉死**，
        这样它既不会悄悄变得更差，也不会哪天被修好了却没人发现。
        """
        r = _creep_run(24.0)
        self.assertGreater(
            r["rate"], self.CREEP_RATE_BUDGET,
            f"默认配置蠕变速率 {r['rate']:.4f} mm/s 居然达标了"
            f"（预算 {self.CREEP_RATE_BUDGET}）——如果这是真的修好了，"
            "请更新本测试和 07 号讲解 §蠕变 一节")
        self.assertLess(
            r["rate"], 4.0 * self.CREEP_RATE_BUDGET,
            f"默认配置蠕变速率恶化到 {r['rate']:.4f} mm/s，"
            f"已超过预算的 4 倍（实测基线约 0.040）")

    def test_the_default_creep_is_still_growing_so_it_has_not_converged(self):
        """⭐ `< 阈值` 证明不了收敛，**"跑更久总位移还在涨"才证明没收敛**。

        实测总位移：悬停 16.3 s → 0.517 mm，22.3 s → 0.853 mm。
        多跑 6 秒又多滑 0.34 mm（+65%），**这是漂移不是沉降**。
        """
        short, long = _creep_run(24.0), _creep_run(30.0)
        self.assertGreater(long["dur"], short["dur"] + 4.0)
        self.assertGreater(
            abs(long["total"]), 1.3 * abs(short["total"]),
            f"悬停从 {short['dur']:.1f}s 拉到 {long['dur']:.1f}s，"
            f"总位移 {abs(short['total']):.3f} → {abs(long['total']):.3f} mm 没怎么涨；"
            "若真收敛了，请更新本测试")

    def test_raising_impratio_meets_the_spec_with_margin(self):
        """⭐⭐ 根因定位到具体旋钮：MuJoCo 的 `impratio`。

        `impratio` 是**切向约束刚度 / 法向约束刚度**之比，默认 1.0。
        等于说"抵抗侧向滑动"和"抵抗压入"一样软——抓取场景下这太软了，
        工件会在指垫上慢慢蹭下去。MuJoCo 官方文档明确建议抓取场景调大它。

        实测 impratio=10：0.0008 mm/s，**达标且有 31 倍余量**。
        （再调到 100 反而变差——数值条件数崩了，不是越大越好。）
        """
        r = _creep_run(24.0, impratio=10.0)
        self.assertLess(
            r["rate"], self.CREEP_RATE_BUDGET,
            f"impratio=10 时蠕变速率 {r['rate']:.4f} mm/s 仍超预算 "
            f"{self.CREEP_RATE_BUDGET} mm/s")
        base = _creep_run(24.0)
        self.assertGreater(
            base["rate"] / r["rate"], 8.0,
            f"impratio=10 只把蠕变从 {base['rate']:.4f} 改善到 "
            f"{r['rate']:.4f} mm/s，不足 8 倍——根因定位可能站不住"
            "（实测约 40 倍）")

    def test_with_impratio_the_creep_actually_converges(self):
        """⭐⭐⭐ 这才是"收敛"的证据：**跑更久，总位移几乎不再增加**。

        impratio=10 实测总位移：16.3 s → 0.013 mm，22.3 s → 0.018 mm。
        多跑 6 秒只多了 0.005 mm ⇒ 这是**一次性弹性沉降**，不是漂移。

        对照上面 `test_the_default_creep_is_still_growing`：
        同样多跑 6 秒，默认配置多滑 0.336 mm，是这里的 **65 倍**。
        **同一个实验、同一个判据，只改一个求解器参数，结论完全相反**——
        这就是"证明收敛"和"数值小于某个阈值"的区别。
        """
        short = _creep_run(24.0, impratio=10.0)
        long = _creep_run(30.0, impratio=10.0)
        grew = abs(long["total"]) - abs(short["total"])
        self.assertLess(
            abs(grew), 0.03,
            f"impratio=10 下多跑 {long['dur'] - short['dur']:.1f}s 又滑了 "
            f"{grew:.4f} mm，看不出收敛")
        base_grew = (abs(_creep_run(30.0)["total"])
                     - abs(_creep_run(24.0)["total"]))
        self.assertGreater(
            abs(base_grew) / max(abs(grew), 1e-6), 15.0,
            f"默认配置多滑 {base_grew:.4f} mm，impratio=10 多滑 {grew:.4f} mm，"
            "两者差距不足 20 倍，这个对照就不够有说服力")


# ============================ 缺陷 3b：变异必须**真的越过摩擦锥**，不能只是调小夹持力


class TestTheCriterionCatchesRealConeViolation(unittest.TestCase):
    """⭐⭐⭐ 用**外载**把切向力推出摩擦锥——这才是物理意义上的"打滑"。

    ⚠️ 第五轮审核指出：旧的"变异证明"是把夹持力从 8 N 调到 4 N。
    但算一下就知道，**4 N 根本没越过摩擦锥**：

        锥内切向极限 = 2·μ·N = 2 × 0.6 × 4 = 4.8 N
        实际切向载荷 = m·g   = 0.25 × 9.81 = 2.45 N
        ⇒ 锥余量 = 2.45 / 4.8 = 0.51，**离打滑还差一半**

    所以那个"打滑"根本不是库仑打滑，还是软接触漂移的放大版。
    **用一个假的坏例子去证明判据管用，等于没证明。**

    ⭐ 真正的做法：直接给工件加一个竖直外载 `F_ext`，
    用闭式算出越锥门槛，在门槛两侧各取一点。

        锥边界： 2·μ·N  =  m·g + F_ext
        ⇒ F_ext_crit = 2μN − mg = 2×0.600×7.916 − 2.453 = 7.047 N

    外载通过 `data.xfrc_applied` 直接加在工件刚体上，
    **完全绕开场景的控制通路**——它是一个真正独立的扰动源。
    """

    MASS = 0.25
    GRAVITY = 9.81

    @classmethod
    def setUpClass(cls):
        base = _creep_run(24.0)
        cls.mu, cls.grip = base["mu"], base["N"]
        cls.cone_limit = 2.0 * cls.mu * cls.grip          # 切向力上限 (N)
        cls.f_crit = cls.cone_limit - cls.MASS * cls.GRAVITY

    def _margin(self, fext):
        """摩擦锥余量 = 实际切向力 / 锥内极限。>1 即越锥。"""
        return (self.MASS * self.GRAVITY + fext) / self.cone_limit

    def test_the_closed_form_cone_threshold_is_what_we_think_it_is(self):
        """地基：先确认闭式算出来的门槛是个合理的数，别把单位搞错。"""
        self.assertAlmostEqual(self.mu, 0.6, places=6)
        self.assertGreater(self.grip, 7.0)
        self.assertLess(self.grip, 8.5)
        self.assertAlmostEqual(self.f_crit, 7.047, delta=0.15)
        self.assertAlmostEqual(self._margin(0.0), 0.258, delta=0.01)
        self.assertAlmostEqual(self._margin(self.f_crit), 1.0, places=6)

    def test_a_load_that_crosses_the_cone_causes_gross_slip(self):
        """⭐ 越锥 20% ⇒ 工件必须**大幅滑脱**，不是"稍微多滑一点"。

        实测：余量 1.20 时总位移 123.0 mm——工件已经**整个掉出夹爪**
        （工件半高才 45 mm）。锥内基线只有 0.52 mm，相差 **238 倍**。

        📌 这里用**总位移**当主判据而不是速率：工件掉光之后场景会提前
        结束，悬停时长从 16.3 s 缩到 13.5 s，速率是被这个变短的分母
        放大/缩小过的，不如"掉了多远"干净。
        """
        r = _creep_run(24.0, fext=1.2 * self.f_crit)
        base = _creep_run(24.0)
        self.assertGreater(
            abs(r["total"]), 50.0,
            f"外载 {1.2 * self.f_crit:.2f} N 已让锥余量到 "
            f"{self._margin(1.2 * self.f_crit):.2f} > 1，"
            f"工件却只滑了 {abs(r['total']):.3f} mm（不到自身半高 45 mm）"
            "——库仑摩擦下这不该发生")
        self.assertGreater(
            abs(r["total"]) / abs(base["total"]), 50.0,
            f"越锥位移 {abs(r['total']):.2f} 只有锥内 {abs(base['total']):.4f} 的 "
            f"{abs(r['total']) / abs(base['total']):.0f} 倍，判别力不够")
        self.assertGreater(r["rate"], 1.0,
                           f"越锥速率 {r['rate']:.4f} mm/s 太小")

    def test_a_load_well_inside_the_cone_still_holds(self):
        """⭐ 对照组：外载加到锥余量 0.45（还在锥内），必须仍然夹得住。

        没有这一条，上一条就可能是"随便加个力都会掉"，什么也没证明。
        实测余量 0.45 时 0.060 mm/s，和无外载的 0.032 同一量级。
        """
        fext = 0.45 * self.cone_limit - self.MASS * self.GRAVITY
        self.assertGreater(fext, 0.0)
        r = _creep_run(24.0, fext=fext)
        self.assertLess(
            r["rate"], 0.5,
            f"锥余量只有 {self._margin(fext):.2f}（离打滑还远），"
            f"工件却在以 {r['rate']:.4f} mm/s 滑")

    def test_the_cone_threshold_comes_from_the_shared_closed_form(self):
        """⭐ 门槛必须来自 `control/grasp.py` 的公式，不能是测试里另写一份。

        如果测试自己抄一份公式，那生产代码里的公式写错了测试也发现不了——
        两份错误各自自洽。这里反过来用 `min_normal_force`：
        给定 μ 和实测的法向力 N，问"这个 N 最多能扛多重的东西"，
        答案必须和 `2μN/g` 一致。
        """
        from armctrl.control.grasp import min_normal_force
        # min_normal_force(m, mu) = m·g/(2μ)。反解：m_max = 2μN/g
        m_max = 2.0 * self.mu * self.grip / self.GRAVITY
        self.assertAlmostEqual(
            min_normal_force(m_max, self.mu, gravity=self.GRAVITY,
                             n_contacts=2),
            self.grip, places=9,
            msg="测试用的锥边界和 control/grasp.py 的闭式对不上")
        # 换算成"还能再挂多少外载"，必须等于 f_crit
        self.assertAlmostEqual((m_max - self.MASS) * self.GRAVITY,
                               self.f_crit, places=9)

    def test_the_simulator_leaks_before_the_cone_is_actually_reached(self):
        """⚠️⚠️ **诚实记录一个坏消息**：仿真在锥余量 ~0.55 就开始漏。

        实测（默认 impratio=1.0）：

            余量 0.45 → 0.060 mm/s（总 0.98 mm）    还夹得住
            余量 0.55 → 1.815 mm/s（总 29.5 mm）    已经在滑
            余量 1.20 → 9.08  mm/s（总 123 mm）     整个掉出去

        库仑摩擦的定义是**余量 < 1 就不允许相对滑动**，所以 0.55 这一档
        是仿真的软接触在漏，不是物理。impratio=10 能把它压小约一个量级，
        但压不到余量 1.0 那么远。

        ⭐ 为什么要把这条写成测试而不是藏起来？
        因为**"摩擦锥余量 < 1 所以夹得住"这个结论在本仿真里是不成立的**。
        任何拿本项目余量读数去推真机的说法都必须带上这条限制。
        钉住它，就没人能不小心地把这句话说出口。
        """
        loose = _creep_run(24.0, fext=0.45 * self.cone_limit
                           - self.MASS * self.GRAVITY)
        leak = _creep_run(24.0, fext=0.55 * self.cone_limit
                          - self.MASS * self.GRAVITY)
        self.assertLess(loose["rate"], 0.5)
        self.assertGreater(
            leak["rate"], 10.0 * loose["rate"],
            f"锥余量 0.45 → 0.55 时速率只从 {loose['rate']:.4f} 变到 "
            f"{leak['rate']:.4f} mm/s。如果这个'提前漏'现象消失了（好事），"
            "请更新本测试和 07 号讲解")


# ==================================== 缺陷 4：夹持力箭头必须画在工件表面上


class TestTheFrictionSliderActuallyReachesBothSides(unittest.TestCase):
    """「工件摩擦系数」滑块必须真的改到 MuJoCo 实际用的那个摩擦系数。

    为什么需要这一组
    ----------------
    `_apply_friction` 的注释早就写明：MuJoCo 把两侧几何体的滑动摩擦
    **逐元素取最大值**合成，所以只改一侧等于没改。但这件事一直**没有测试守着**。
    批次 E-3 的变异扫描里，M05（「只改指垫、不改工件」）因此**活了下来**：

        工件出厂 mu 恰好 0.6，滑块默认值也恰好 0.6
        => 默认工况下「漏改工件」这个 bug 一点现象都没有。

    ⭐ 教训：**当默认值和出厂值撞在一起时，整组测试会静默失效。**
    所以下面第一条先把「两侧出厂值本来就不一样」这个前提本身钉住——
    哪天有人把工件出厂值改成 1.0，这组测试会先红，而不是悄悄变成废纸。

    oracle 独立性
    ------------
    - 合成规则**不假设**，用一个五行的独立 MJCF（两个几何体、两个不同摩擦）
      现场量出来；这个模型和 armctrl 没有任何关系。
    - 生效值**不读**场景的 `mu_eff` 读数（那要过 `control/grasp.py`），
      而是直接读 MuJoCo 写在 `data.contact[i].friction[0]` 里的合成结果。
    - 取 mu = 0.35，**和两边出厂值（0.6 / 1.0）都不同**，
      于是「哪一侧没改到」都会露馅。
    """

    MU = 0.35
    OBJ_FACTORY = 0.6
    PAD_FACTORY = 1.0

    def test_the_two_sides_ship_with_different_defaults(self):
        """前提：出厂摩擦两侧不同。相同的话本组测试就没有鉴别力了。"""
        sc = _GRASP()
        m = sc.build()                       # 注意：不 reset，读的才是出厂值
        obj = float(m.geom_friction[sc.obj_gid, 0])
        pads = sorted({round(float(m.geom_friction[g, 0]), 6)
                       for g in sc._pad_geoms})
        self.assertAlmostEqual(obj, self.OBJ_FACTORY, places=6,
                               msg="工件出厂摩擦变了，下面的数字要跟着改")
        self.assertEqual(pads, [self.PAD_FACTORY],
                         "指垫出厂摩擦变了，下面的数字要跟着改")
        self.assertNotAlmostEqual(
            obj, self.PAD_FACTORY, places=6,
            msg="两侧出厂摩擦一旦相同，M05 这类「漏改一侧」的 bug 将无法被发现")

    def test_mujoco_really_combines_friction_by_taking_the_max(self):
        """用一个独立小模型量出合成规则：取最大值，不是取平均、不是取最小。"""
        xml = """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="1 1 .1"
                  friction="0.9 0.005 0.0001"/>
            <body pos="0 0 0.0499">
              <freejoint/>
              <geom name="cube" type="box" size=".05 .05 .05"
                    friction="0.2 0.005 0.0001"/>
            </body>
          </worldbody>
        </mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertGreater(d.ncon, 0, "小模型没接触上，测不了合成规则")
        got = float(d.contact[0].friction[0])
        self.assertAlmostEqual(got, 0.9, places=6,
                               msg=f"合成规则不是取最大值？实测 {got}")
        self.assertNotAlmostEqual(got, 0.2, places=6, msg="不是取最小值")
        self.assertNotAlmostEqual(got, 0.55, places=6, msg="不是取平均值")

    def test_the_slider_lands_on_both_geoms(self):
        """把滑块拨到 0.35，工件和每一片指垫都必须变成 0.35。"""
        sc = _GRASP()
        m = sc.build()
        sc.set("obj_mu", self.MU)
        sc.reset()
        self.assertAlmostEqual(
            float(m.geom_friction[sc.obj_gid, 0]), self.MU, places=6,
            msg="工件那一侧没改到——滑块对工件无效")
        for g in sc._pad_geoms:
            self.assertAlmostEqual(
                float(m.geom_friction[g, 0]), self.MU, places=6,
                msg=f"指垫 {g} 没改到——生效的会是出厂的 {self.PAD_FACTORY}")

    def test_the_slider_reaches_the_real_grasp_contact(self):
        """端到端：真抓起来以后，MuJoCo 用的接触摩擦就是滑块上那个数。"""
        run = _creep_run(10.0, mu=self.MU)
        self.assertTrue(run["mu_raw"], "悬停期间没采到工件的接触，测不了")
        self.assertEqual(
            run["mu_raw"], [self.MU],
            f"实际接触摩擦是 {run['mu_raw']}，不等于滑块设的 {self.MU}；"
            f"通常意味着有一侧还留着出厂值")


class TestForceArrowLandsOnSurface(unittest.TestCase):
    """力箭头的起点必须贴着工件被指垫压的那两个面。

    原实现把偏移写死成 0.045，那是工件的**长半轴**（90 mm 的一半），
    而开合方向的半宽只有 21 mm —— 箭头悬在工件外 24 mm 的空中，
    看上去像力作用在空气上。

    oracle 独立性：用本文件里的 `box_extent_along` 支撑函数算真实半跨度，
    它与 `_obj_half_along` 没有共享代码。
    """

    @classmethod
    def setUpClass(cls):
        cls.sc, _ = run_scene(9.0, shake=0.0)

    def test_scenario_has_contact(self):
        self.assertGreater(self.sc._reading.n_contacts, 0,
                           "没接触就没有力箭头，测试无意义")

    def test_arrow_tip_sits_on_the_gripped_face(self):
        axis = self.sc._R_now[:, 1]
        _, R_true = _true_obj(self.sc)
        expect = 0.5 * box_extent_along(self.sc.OBJ_HALF, R_true, axis)
        got = self.sc._obj_half_along(axis)
        self.assertAlmostEqual(got, expect, places=9)
        # 并且确实是短边那一侧，不是被写死成长半轴
        self.assertLess(got, 0.030,
                        f"箭头起点距中心 {got * 1e3:.1f} mm，"
                        f"远超开合方向半宽 21 mm，箭头会悬空")

    def test_overlay_arrows_start_at_that_distance(self):
        """真的取到 overlays() 里，防止函数写对了但没被用上。"""
        from armctrl.tuner.base import Arrow
        c = self.sc.data.xpos[self.sc.obj_bid]
        axis = self.sc._R_now[:, 1]
        h = self.sc._obj_half_along(axis)
        arrows = [it for it in self.sc.overlays() if isinstance(it, Arrow)]
        self.assertEqual(len(arrows), 2, "应当有两支相对的夹持力箭头")
        for a in arrows:
            d = float(abs((np.asarray(a.end) - c) @ axis))
            self.assertAlmostEqual(d, h, places=9)


# ============= 缺陷 5：滑移检测阈值必须小于它要抓的那种慢滑，否则等于没装


def _hold_creep(tmax=26.0, **params):
    """悬停段的相对位移轨迹（mm）。oracle 同上：世界系工件 z 减末端 z。"""
    sc = _GRASP()
    model = sc.build()
    for k, val in params.items():
        sc.set(k, val)
    sc.reset()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            sc.obj.body_name())
    dt = model.opt.timestep
    ref = None
    trace = []
    for k in range(int(tmax / dt)):
        t = k * dt
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
        mujoco.mj_step(model, sc.data)
        if sc._phase == sc.PH_HOLD:
            gap = float(sc.data.xpos[bid][2] - sc._p[2])
            if ref is None:
                ref, t0 = gap, t
            trace.append((t - t0, (gap - ref) * 1e3))
    return sc, trace


class TestSlipThresholdCatchesSlowCreep(unittest.TestCase):
    """默认（自适应）配置下，悬停时工件不能持续往指尖外爬。

    机理
    ----
    `AdaptiveGripForce` 是 bang-bang 试探律：没滑就慢慢减力，滑了就快速加力。
    它靠两条判据的**并集**触发——「濒临滑移」（摩擦锥余量超阈值）与
    「已经在滑」（切向速度超阈值）。

    原来的速度阈值是 2 mm/s，而这套配置实测的缓慢蠕变约 1.4 mm/s，
    **比阈值还小**。于是反应支路在构造上永远不会触发，只剩余量支路在
    0.85 上做极限环，工件就一路匀速往下爬（悬停 29 s 爬 40 mm，最终掉落）。

    一个检测器的阈值高于它要检测的量，就等于没装。所以阈值被改到 0.5 mm/s
    ——比要捕捉的 1.4 mm/s 留 2.8 倍余量。代价是稳态夹持力从 3.60 N 升到
    3.95 N（安全系数 1.76 → 1.94），仍然贴着理论下限，
    「自适应能找到接近最小的力」这个结论不受影响。

    这不是放宽容差，是**收紧**一个定错了的检测阈值。
    """

    @classmethod
    def setUpClass(cls):
        cls.sc, cls.trace = _hold_creep()

    def test_scenario_reaches_hold(self):
        self.assertGreater(len(self.trace), 4000, "没进入悬停，测试无意义")

    def test_object_stays_in_the_fingers(self):
        self.assertGreater(self.sc._reading.n_contacts, 0, "工件已经掉了")
        self.assertLess(abs(self.trace[-1][1]), 5.0,
                        f"默认配置悬停 {self.trace[-1][0]:.1f} s "
                        f"工件爬了 {abs(self.trace[-1][1]):.2f} mm")

    def test_blind_threshold_is_much_worse(self):
        """把阈值调回 2 mm/s（高于蠕变速率），蠕变必须显著恶化。

        这条不是重复上一条：它证明「阈值高于被测量」正是根因，
        而不是别的什么改动碰巧让数字变好了。
        """
        _, blind = _hold_creep(slip_speed_thr=2e-3)
        good = abs(self.trace[-1][1])
        bad = abs(blind[-1][1])
        self.assertGreater(bad, 5.0 * max(good, 0.5),
                           f"阈值 2 mm/s 时爬 {bad:.2f} mm，"
                           f"0.5 mm/s 时爬 {good:.2f} mm，差距不够显著，"
                           f"说明根因判断有误")

if __name__ == "__main__":
    unittest.main()
