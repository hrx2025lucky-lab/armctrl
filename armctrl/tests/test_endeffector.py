"""末端执行器（TCP）与平行夹爪的测试。

判据独立性
----------
- TCP 偏置**从 mjModel 的几何重新推导**（finger body 原点 + 指尖夹持垫包络中心），
  不引用 robot.py 里的常数作为正确性依据，只用它做被测对象。
- TCP 点雅可比用**中心差分**独立验证，不依赖 mj_jac 自己。
- 杠杆项恒等式 J_tcp_lin = J_flange_lin − [R·o]× J_rot 独立推导后验证。
- 夹爪的 ctrl↔宽度映射由伺服不动点 kp(len_target − len) = 0 推出，
  不硬编码 0.04/255 这两个数字。
- 抓取用**真实接触力**判定，不看"手指关节到位了没有"。
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from armctrl.assets import panda_xml
from armctrl.core.robot import ArmModel, PANDA_TCP_OFFSET, make_panda
from armctrl.core.gripper import ParallelJawGripper, make_panda_gripper
from armctrl.core.kinematics import DLSInverseKinematics, Q_HOME_PANDA, so3_log
from armctrl.control.impedance import CartesianImpedanceController, ImpedanceGains


Q_HOME = Q_HOME_PANDA

_TIP = None
_FLANGE = None


def tip_robot():
    global _TIP
    if _TIP is None:
        _TIP = make_panda()
    return _TIP


def flange_robot():
    global _FLANGE
    if _FLANGE is None:
        _FLANGE = make_panda(tcp="flange")
    return _FLANGE


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def derive_tcp_from_model(model) -> float:
    """**独立**从 mjModel 推出抓取中心相对 hand 原点的 z 偏置。

    TCP_z = (left_finger body 在 hand 系的 z) + (指尖夹持垫 z 向包络的中点)
    """
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    assert int(model.body_parentid[lf]) == hand
    finger_z = float(model.body_pos[lf][2])
    zlo, zhi = np.inf, -np.inf
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != lf:
            continue
        if int(model.geom_group[g]) != 3:             # 只看碰撞组
            continue
        if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_BOX:
            continue                                   # 夹持垫是 box，指身是 mesh
        z, hz = float(model.geom_pos[g][2]), float(model.geom_size[g][2])
        zlo, zhi = min(zlo, z - hz), max(zhi, z + hz)
    return finger_z + 0.5 * (zlo + zhi)


class TestTcpDefinition(unittest.TestCase):

    def test_tcp_offset_matches_independent_model_derivation(self):
        r = tip_robot()
        z = derive_tcp_from_model(r.model)
        self.assertAlmostEqual(z, float(PANDA_TCP_OFFSET[2]), places=6,
                               msg=f"模型推出 {z:.6f}，代码里是 {PANDA_TCP_OFFSET[2]}")
        self.assertAlmostEqual(float(PANDA_TCP_OFFSET[0]), 0.0, places=12)
        self.assertAlmostEqual(float(PANDA_TCP_OFFSET[1]), 0.0, places=12)
        # 与 Franka 官方 panda_hand_tcp = 0.1034 m 的一致性（允许 1 mm 网格差异）
        self.assertLess(abs(z - 0.1034), 1e-3, f"与官方标称 0.1034 差 {abs(z-0.1034):.4f}")

    def test_tcp_sits_between_the_two_fingertips(self):
        """抓取中心必须在两指连线的中点上——这才叫"抓取中心"。

        ⚠️ 这条测试原来写成 ``‖(mid-p_tcp)[:2] - (mid-p_tcp)[:2]‖ < 1e-12``，
        左右两边是同一个表达式，**无论实现对错都恒为 0**（第四轮审核 T-01）。
        现在改成两条真正有方向性的判据：

        1. 把 ``p_tcp − mid`` 变换到**手爪自身坐标系**，两个**横向**分量必须为 0
           （TCP 只允许沿夹持方向 z 偏出去）；
        2. TCP 到左右指尖**等距**——朝任何一指偏一点都会破坏它。

        并且在同一个测试里做了**变异对照**：人为给 TCP 加 5 mm 横向偏移，
        两条判据都必须失败。判据抓不到变异，就等于没测。
        """
        r = tip_robot()
        m, d = r.model, r.data
        g = make_panda_gripper(r)
        for width in (0.0, 0.04, 0.08):
            r.set_state(Q_HOME)
            g.set_state(width)
            p_tcp, R = r.fk()
            lf = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")]
            rf = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")]
            mid = 0.5 * (lf + rf)
            # ① 手爪系下的偏置：只允许沿 z（夹持/接近方向），横向必须为零
            local = R.T @ (p_tcp - mid)
            self.assertLess(
                float(np.linalg.norm(local[:2])), 1e-9,
                f"width={width}: TCP 偏离两指中线 "
                f"{np.linalg.norm(local[:2]) * 1e3:.3f} mm（手爪系横向分量）")
            self.assertGreater(float(local[2]), 0.0,
                               "TCP 应当沿接近方向伸在两指之间，符号反了")
            # ② 到左右指尖等距
            d_l = float(np.linalg.norm(p_tcp - lf))
            d_r = float(np.linalg.norm(p_tcp - rf))
            self.assertAlmostEqual(d_l, d_r, places=9,
                                   msg=f"width={width}: TCP 离两指不等距，"
                                       f"{d_l * 1e3:.3f} vs {d_r * 1e3:.3f} mm")
            # ③ 沿接近方向的距离（原有判据，保留）
            self.assertAlmostEqual(float(np.linalg.norm(mid - p_tcp)), 0.0445,
                                   places=6,
                                   msg=f"width={width}: TCP 不在两指中线上")
            # ④ ⭐ 变异对照：横向挪 5 mm，上面两条判据必须都抓到
            for axis, tag in ((0, "手爪 x"), (1, "手爪 y")):
                bad = p_tcp + R[:, axis] * 5e-3
                bad_local = R.T @ (bad - mid)
                self.assertGreater(
                    float(np.linalg.norm(bad_local[:2])), 1e-9,
                    f"沿{tag}偏 5 mm 居然没被横向判据抓到")
            if float(np.linalg.norm(rf - lf)) > 1e-6:
                # ⚠️ width=0 时两指重合，"离两指等距"对任何点都平凡成立，
                # 这个变异对照在那一档没有意义，必须跳过。
                u = (rf - lf) / np.linalg.norm(rf - lf)
                bad_y = p_tcp + u * 5e-3
                self.assertNotAlmostEqual(
                    float(np.linalg.norm(bad_y - lf)),
                    float(np.linalg.norm(bad_y - rf)), places=9,
                    msg="朝一侧手指偏 5 mm 居然没被等距判据抓到")
        r.set_state(np.zeros(r.n), np.zeros(r.n))

    def test_tcp_is_the_default_and_flange_is_still_reachable(self):
        rt, rf = tip_robot(), flange_robot()
        pt, Rt = rt.fk(Q_HOME)
        pf, Rf = rf.fk(Q_HOME)
        self.assertTrue(np.allclose(Rt, Rf), "TCP 只有平移偏置，姿态不应改变")
        self.assertAlmostEqual(float(np.linalg.norm(pt - pf)),
                               float(np.linalg.norm(PANDA_TCP_OFFSET)), places=9)
        # flange_pose() 必须仍能拿到法兰
        pfl, _ = rt.flange_pose(Q_HOME)
        self.assertTrue(np.allclose(pfl, pf))
        # 偏置方向必须是 hand 系的 +z
        self.assertTrue(np.allclose(Rt.T @ (pt - pf), PANDA_TCP_OFFSET, atol=1e-12))

    def test_zero_offset_reproduces_the_old_flange_behaviour(self):
        """tcp_offset=0 时必须**逐位**退化为加 TCP 之前的实现。

        oracle 用一个**独立的 MjData** 现算，而不是读 `rf.data` ——
        查询接口按设计不再改写仿真状态，拿仿真状态当参照就变成了
        "用被测实现的副作用证明被测实现"，而且那个副作用本身正是要杜绝的东西。
        """
        rf = flange_robot()
        self.assertFalse(rf.tcp_offset.any())
        m = rf.model
        oracle = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.uniform(rf.q_lower, rf.q_upper)
            oracle.qpos[:] = 0.0
            oracle.qvel[:] = 0.0
            oracle.qpos[rf.qpos_idx] = q
            mujoco.mj_forward(m, oracle)

            p, R = rf.fk(q)
            self.assertTrue(np.allclose(p, oracle.xpos[rf.ee_body_id]))
            self.assertTrue(np.allclose(
                R, oracle.xmat[rf.ee_body_id].reshape(3, 3)))

            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, oracle, jacp, jacr, rf.ee_body_id)
            J_old = np.vstack([jacp[:, rf.qvel_idx], jacr[:, rf.qvel_idx]])
            self.assertTrue(np.allclose(rf.jacobian(q), J_old, atol=0, rtol=0))
        rf.set_state(np.zeros(rf.n), np.zeros(rf.n))

    def test_queries_never_touch_the_simulation_state(self):
        """给定构型的查询接口一律不得改写 `robot.data`。

        这条是针对一个真实缺陷的探针：`gravity(q)` 按定义要令 q̇ = 0，
        旧实现把它写回仿真 data，于是每调用一次就把**被控对象的速度抹成零**。
        动量观测器每个控制周期都要调 `gravity`，等于每步给系统加了一次
        无穷大阻尼，动力学被彻底改掉，而调用方看到的只是"读了一个重力项"。
        """
        r = tip_robot()
        q_mark = np.array([0.11, -0.22, 0.33, -1.44, 0.55, 1.66, 0.77])
        v_mark = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        r.set_state(q_mark, v_mark)
        q_other = q_mark + 0.25

        r.fk(q_other)
        r.flange_pose(q_other)
        r.jacobian(q_other)
        r.mass_matrix(q_other)
        r.bias(q_other, v_mark * 3.0)
        r.gravity(q_other)
        r.coriolis_times_qd(q_other, v_mark * 3.0)
        r.task_space_inertia(q_other)
        r.manipulability(q_other)
        r.joint_anchor(0, q_other)

        self.assertTrue(np.allclose(r.get_q(), q_mark), "查询污染了 qpos")
        self.assertTrue(np.allclose(r.get_qd(), v_mark), "查询污染了 qvel")
        r.set_state(np.zeros(r.n), np.zeros(r.n))

    def test_queries_at_a_given_configuration_are_still_correct(self):
        """隔离查询状态之后，返回值仍须与"就地设状态再读"完全一致。

        隔离若做错（例如忘了把夹爪手指的位置同步过去），质量矩阵与重力项
        会算在错误的手部构型上。用一个独立 MjData 作为 oracle 交叉验证。
        """
        r = tip_robot()
        m = r.model
        oracle = mujoco.MjData(m)
        rng = np.random.default_rng(5)
        for _ in range(10):
            q = rng.uniform(r.q_lower, r.q_upper)
            qd = rng.uniform(-1.0, 1.0, r.n)
            # 仿真状态故意设成别的构型，验证查询不受它影响
            r.set_state(rng.uniform(r.q_lower, r.q_upper), rng.uniform(-1, 1, r.n))

            oracle.qpos[:] = r.data.qpos          # 非手臂自由度（手指）应当同步
            oracle.qvel[:] = 0.0
            oracle.qpos[r.qpos_idx] = q
            oracle.qvel[r.qvel_idx] = qd
            mujoco.mj_forward(m, oracle)

            self.assertTrue(np.allclose(
                r.bias(q, qd), oracle.qfrc_bias[r.qvel_idx], atol=1e-12))
            full = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, oracle, full)
            self.assertTrue(np.allclose(
                r.mass_matrix(q), full[np.ix_(r.qvel_idx, r.qvel_idx)], atol=1e-12))
        r.set_state(np.zeros(r.n), np.zeros(r.n))


class TestTcpJacobian(unittest.TestCase):

    def _fd_jacobian(self, robot, q, h=1e-6):
        J = np.zeros((6, robot.n))
        for i in range(robot.n):
            qp, qm = q.copy(), q.copy()
            qp[i] += h
            qm[i] -= h
            pp, Rp = robot.fk(qp)
            pm, Rm = robot.fk(qm)
            J[:3, i] = (pp - pm) / (2 * h)
            # ω 由 Ṙ Rᵀ 的反对称部分给出
            W = ((Rp - Rm) / (2 * h)) @ ((Rp + Rm) / 2).T
            J[3:, i] = np.array([W[2, 1] - W[1, 2],
                                 W[0, 2] - W[2, 0],
                                 W[1, 0] - W[0, 1]]) / 2.0
        return J

    def test_point_jacobian_matches_central_differences(self):
        r = tip_robot()
        rng = np.random.default_rng(3)
        worst_lin = worst_ang = 0.0
        for _ in range(12):
            q = np.clip(Q_HOME + rng.normal(0, 0.5, r.n), r.q_lower, r.q_upper)
            J = r.jacobian(q)
            Jn = self._fd_jacobian(r, q)
            worst_lin = max(worst_lin, float(np.abs(J[:3] - Jn[:3]).max()))
            worst_ang = max(worst_ang, float(np.abs(J[3:] - Jn[3:]).max()))
        self.assertLess(worst_lin, 1e-7, f"线速度行差分误差 {worst_lin:.3e}")
        self.assertLess(worst_ang, 1e-7, f"角速度行差分误差 {worst_ang:.3e}")
        r.set_state(np.zeros(r.n), np.zeros(r.n))

    def test_lever_arm_identity(self):
        """独立推导：p_tcp = p_f + R o ⇒ ṗ_tcp = ṗ_f + ω×(R o)
        ⇒ J_tcp_lin = J_f_lin − [R o]× J_rot，角速度行不变。"""
        rt, rf = tip_robot(), flange_robot()
        rng = np.random.default_rng(4)
        worst = 0.0
        for _ in range(20):
            q = rng.uniform(rt.q_lower, rt.q_upper)
            _, R = rt.fk(q)
            Jt = rt.jacobian(q)
            Jf = rf.jacobian(q)
            S = skew(R @ PANDA_TCP_OFFSET)
            worst = max(worst, float(np.abs(Jt[:3] - (Jf[:3] - S @ Jf[3:])).max()))
            self.assertTrue(np.array_equal(Jt[3:], Jf[3:]), "角速度行不该变")
        self.assertLess(worst, 1e-12, f"杠杆项恒等式误差 {worst:.3e}")
        rt.set_state(np.zeros(rt.n), np.zeros(rt.n))

    def test_mutation_ignoring_the_lever_term_is_detectable(self):
        """**mutation**：若 TCP 雅可比错用连杆原点的 jacBody，线速度行会明显不同。

        证明前一个用例不是恒等式的空断言。
        """
        rt, rf = tip_robot(), flange_robot()
        q = Q_HOME
        Jt, Jf = rt.jacobian(q), rf.jacobian(q)
        self.assertGreater(float(np.abs(Jt[:3] - Jf[:3]).max()), 1e-3,
                           "TCP 与法兰的线速度雅可比几乎一样，工具长度没起作用")

    def test_manipulability_is_invariant_but_conditioning_is_not(self):
        """独立推导：J_tcp = T·J_flange，T = [[I, −S],[0, I]]，det(T) = 1。

        于是 det(J_tcp J_tcpᵀ) = det(T)²·det(J Jᵀ) = det(J Jᵀ)，
        **Yoshikawa 可操作度对 TCP 偏置不变**——这不是 bug，是可推导的事实。
        但 T 不是正交阵，所以条件数和任务空间惯量都会变。
        """
        rt, rf = tip_robot(), flange_robot()
        rng = np.random.default_rng(7)
        for _ in range(10):
            q = rng.uniform(rt.q_lower, rt.q_upper)
            _, R = rt.fk(q)
            S = skew(R @ PANDA_TCP_OFFSET)
            T = np.block([[np.eye(3), -S], [np.zeros((3, 3)), np.eye(3)]])
            self.assertAlmostEqual(float(np.linalg.det(T)), 1.0, places=12)
            self.assertAlmostEqual(rt.manipulability(q), rf.manipulability(q),
                                   delta=1e-9 * max(rt.manipulability(q), 1e-9))
        self.assertNotAlmostEqual(rt.condition_number(Q_HOME),
                                  rf.condition_number(Q_HOME), places=3)
        Lt = rt.task_space_inertia(Q_HOME)
        Lf = rf.task_space_inertia(Q_HOME)
        self.assertGreater(float(np.abs(Lt - Lf).max()), 1e-6,
                           "任务空间惯量没有跟着 TCP 走")
        for L in (Lt, Lf):
            self.assertGreater(float(np.linalg.eigvalsh(0.5 * (L + L.T)).min()), 0.0)
        rt.set_state(np.zeros(rt.n), np.zeros(rt.n))


class TestGripperModel(unittest.TestCase):

    def setUp(self):
        self.r = tip_robot()
        self.g = make_panda_gripper(self.r)
        self.r.set_state(np.zeros(self.r.n), np.zeros(self.r.n))

    def test_arm_actuators_exclude_the_gripper(self):
        """力矩化只能作用在手臂执行器上；夹爪必须保住它的位置伺服。"""
        m = self.r.model
        self.assertNotIn(self.g.actuator_id, self.r.arm_actuator_ids)
        self.assertEqual(len(self.r.arm_actuator_ids), self.r.n)
        for a in self.r.arm_actuator_ids:
            self.assertEqual(m.actuator_biastype[a], mujoco.mjtBias.mjBIAS_NONE)
        self.assertEqual(m.actuator_biastype[self.g.actuator_id],
                         mujoco.mjtBias.mjBIAS_AFFINE)
        self.assertGreater(self.g.kp, 0.0)

    def test_arm_actuator_resolution_matches_transmission_targets(self):
        m = self.r.model
        for jid, a in zip(self.r.joint_ids, self.r.arm_actuator_ids):
            self.assertEqual(m.actuator_trntype[a], mujoco.mjtTrn.mjTRN_JOINT)
            self.assertEqual(int(m.actuator_trnid[a, 0]), int(jid))

    def test_width_ctrl_mapping_is_derived_not_hardcoded(self):
        """由伺服不动点 kp(len_target − len) = 0 推出的映射必须与模型注释一致。

        panda.xml 注释：ctrlrange (0, 255) 映射到单指 (0, 0.04) m。
        故 width(255) 必须是 0.08 m（两指之和），width(0) 必须是 0。
        """
        g = self.g
        self.assertAlmostEqual(g.ctrl_to_width(0.0), 0.0, places=12)
        self.assertAlmostEqual(g.ctrl_to_width(255.0), 0.08, places=9)
        self.assertAlmostEqual(g.width_to_ctrl(0.08), 255.0, places=6)
        self.assertAlmostEqual(g.width_to_ctrl(0.04), 127.5, places=6)
        self.assertAlmostEqual(g.width_min, 0.0, places=12)
        self.assertAlmostEqual(g.width_max, 0.08, places=9)
        # 往返一致
        for w in np.linspace(0.0, 0.08, 17):
            self.assertAlmostEqual(g.ctrl_to_width(g.width_to_ctrl(w)), w, places=9)

    def test_width_command_is_clipped_to_the_physical_range(self):
        g = self.g
        self.assertAlmostEqual(g.command_width(0.5), 0.08, places=9)
        self.assertAlmostEqual(g.command_width(-0.5), 0.0, places=12)
        self.assertAlmostEqual(g.open(), 0.08, places=9)
        self.assertAlmostEqual(g.close(), 0.0, places=12)

    def test_width_reads_back_the_joint_state(self):
        g = self.g
        for w in (0.0, 0.02, 0.05, 0.08):
            g.set_state(w)
            self.assertAlmostEqual(g.width, w, places=9)
            self.assertAlmostEqual(
                float(self.r.data.qpos[g.qpos_idx[0]]), 0.5 * w, places=9)

    def test_gripper_actually_moves_to_the_commanded_width(self):
        """闭环验证：给宽度指令并推进仿真，实际宽度必须收敛到指令值。"""
        r, g = self.r, self.g
        r.set_state(Q_HOME)
        g.set_state(0.0)
        tau_hold = r.bias(Q_HOME, np.zeros(r.n))
        g.settle(0.06, steps=1200, arm_torque_fn=lambda q, v: tau_hold)
        self.assertAlmostEqual(g.width, 0.06, delta=2e-3,
                               msg=f"张开后宽度 {g.width:.4f}，指令 0.06")
        g.settle(0.01, steps=1200, arm_torque_fn=lambda q, v: tau_hold)
        self.assertAlmostEqual(g.width, 0.01, delta=2e-3,
                               msg=f"闭合后宽度 {g.width:.4f}，指令 0.01")
        mujoco.mj_resetData(r.model, r.data)

    def test_force_limit_is_applied_to_the_actuator(self):
        g = self.g
        g.set_force_limit(25.0)
        self.assertAlmostEqual(g.force_limit, 25.0, places=9)
        self.assertTrue(np.allclose(
            self.r.model.actuator_forcerange[g.actuator_id], [-25.0, 25.0]))
        g.set_force_limit(100.0)

    def test_no_contact_means_no_grasp(self):
        r, g = self.r, self.g
        mujoco.mj_resetData(r.model, r.data)
        r.set_state(Q_HOME)
        g.set_state(0.04)
        mujoco.mj_forward(r.model, r.data)
        self.assertEqual(g.contacts_with(), [])
        self.assertEqual(g.contact_force(), 0.0)
        self.assertFalse(g.is_grasping())


CUBE_HALF = (0.012, 0.012, 0.03)
TABLE_TOP = 0.383


def build_grasp_scene():
    """派生一个"桌面 + 自由方块"的场景（MjSpec，不改外部 XML）。

    方块必须有支撑：自由体在重力下会在夹爪闭合之前就掉走，那样测的是自由落体，
    不是夹持。桌面高度取成方块中心正好落在 Q_HOME 的 TCP 高度上。
    """
    xml = panda_xml()
    spec = mujoco.MjSpec.from_file(xml)
    table = spec.worldbody.add_body(name="table", pos=[0.5, 0.0, TABLE_TOP - 0.05])
    table.add_geom(name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.15, 0.15, 0.05], rgba=[0.6, 0.6, 0.65, 1.0])
    body = spec.worldbody.add_body(name="cube",
                                   pos=[0.484, 0.0, TABLE_TOP + CUBE_HALF[2]])
    body.add_freejoint()
    g = body.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(CUBE_HALF), rgba=[0.85, 0.3, 0.2, 1.0])
    g.mass = 0.15
    return spec.compile()


class TestGraspWithObject(unittest.TestCase):
    """真正抓一个方块并提起来。判据是**接触力**和**工件是否跟着走**，
    不是"手指关节到位了没有"。

    每个用例都**重新编译一份模型**：set_grip_force 会就地改写 mjModel 的
    gainprm/biasprm/forcerange，共用模型会造成跨用例污染（第一版就踩到了：
    "出厂伺服太弱"的用例读到了上一个用例留下的 kp=15000）。
    """

    CUBE_WIDTH = 2.0 * CUBE_HALF[1]

    def _setup(self):
        model = build_grasp_scene()
        robot = ArmModel(model=model, ee_body="hand",
                         arm_joints=[f"joint{i}" for i in range(1, 8)],
                         tcp_offset=PANDA_TCP_OFFSET)
        grip = ParallelJawGripper(robot)
        m, d = robot.model, robot.data
        mujoco.mj_resetData(m, d)
        robot.set_state(Q_HOME)
        grip.set_state(grip.width_max)
        cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        adr = int(m.jnt_qposadr[int(m.body_jntadr[cube])])
        p_tcp, R_tcp = robot.fk(Q_HOME)
        d.qpos[adr:adr + 3] = [p_tcp[0], p_tcp[1], TABLE_TOP + CUBE_HALF[2]]
        d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        return robot, grip, adr, p_tcp, R_tcp

    @staticmethod
    def _hold(robot):
        def fn(q, v):
            return robot.mass_matrix(q) @ (400.0 * (Q_HOME - q) - 40.0 * v) \
                + robot.bias(q, v)
        return fn

    def test_each_test_starts_from_the_factory_servo(self):
        """隔离性自检：新建的夹爪必须是出厂 kp，不带上一个用例的残留。"""
        _, grip, _, _, _ = self._setup()
        self.assertAlmostEqual(grip.kp, 100.0, places=6)
        self.assertAlmostEqual(grip.force_limit, 100.0, places=6)

    def test_factory_servo_is_too_weak_to_grip(self):
        """出厂 kp = 100 N/m 的位置伺服**夹不住东西**：这是模型的真实局限，
        必须被暴露出来，而不是靠调测试阈值糊过去。

        腱力上界 = kp × 单指过闭合量 = 100 × 0.012 = 1.2 N，与 forcerange 无关。
        """
        robot, grip, adr, _, _ = self._setup()
        self.assertAlmostEqual(grip.kp, 100.0, places=6)
        grip.set_force_limit(60.0)                    # 只抬上限，不动 kp
        grip.settle(0.0, steps=1500, arm_torque_fn=self._hold(robot))
        self.assertGreater(len(grip.contacts_with("cube")), 1, "没有形成两侧接触")
        self.assertLess(abs(grip.tendon_force), 2.0,
                        f"出厂伺服竟然给出了 {grip.tendon_force:.2f} N 腱力")
        self.assertLess(grip.contact_force("cube"), 2.0)

    def test_one_phase_stiff_close_fails(self):
        """**反例**：不做低刚度贴近、直接用 15000 N/m 砸下去，会穿过工件。

        这条用例的存在是为了证明 close_and_grip 的两段式不是装饰。
        """
        robot, grip, adr, _, _ = self._setup()
        w = grip.grasp(self.CUBE_WIDTH, 60.0)
        grip.settle(w, steps=1200, arm_torque_fn=self._hold(robot))
        self.assertLess(grip.width, 0.5 * self.CUBE_WIDTH,
                        f"一步到位竟然停在了 {grip.width:.4f}，反例不成立")
        self.assertFalse(grip.is_grasping("cube"))

    def test_grip_force_command_is_achieved_on_the_object(self):
        for target in (20.0, 60.0):
            with self.subTest(force=target):
                robot, grip, adr, _, _ = self._setup()
                grip.close_and_grip(self.CUBE_WIDTH, target,
                                    approach_steps=900, grip_steps=900,
                                    arm_torque_fn=self._hold(robot))
                self.assertGreater(len(grip.contacts_with("cube")), 1,
                                   "只有单侧接触，不构成夹持")
                # 腱力必须被 forcerange 精确封在指令值上
                self.assertAlmostEqual(abs(grip.tendon_force), target,
                                       delta=0.15 * target,
                                       msg=f"腱力 {grip.tendon_force:.2f} vs 指令 {target}")
                # 单指法向力 ≈ 腱力/2（split 腱两个 wrap 系数都是 0.5）
                f_side = grip.contact_force("cube")
                self.assertGreater(f_side, 0.25 * target,
                                   f"单指法向力仅 {f_side:.2f} N")
                self.assertLess(f_side, 0.75 * target)
                self.assertTrue(grip.is_grasping("cube", min_force=1.0))
                # 手指不能穿过方块
                self.assertGreater(grip.width, 0.9 * self.CUBE_WIDTH)
                self.assertLess(grip.width, 1.05 * self.CUBE_WIDTH)

    def test_grasp_holds_the_object_while_lifting(self):
        """最强的判据：把手臂抬高，工件必须跟着上来。"""
        robot, grip, adr, p_tcp, R_tcp = self._setup()
        grip.close_and_grip(self.CUBE_WIDTH, 60.0,
                            approach_steps=900, grip_steps=900,
                            arm_torque_fn=self._hold(robot))
        z0 = float(robot.data.qpos[adr + 2])
        ctrl = CartesianImpedanceController(
            robot, ImpedanceGains(k_trans=1500.0, k_rot=80.0, zeta=1.0), q_null=Q_HOME)
        p_up = p_tcp + np.array([0.0, 0.0, 0.06])
        for _ in range(2500):
            q, v = robot.get_q(), robot.get_qd()
            robot.data.ctrl[robot.arm_actuator_ids] = ctrl.compute(q, v, p_up, R_tcp)
            mujoco.mj_step(robot.model, robot.data)
        z1 = float(robot.data.qpos[adr + 2])
        self.assertGreater(z1 - z0, 0.04,
                           f"工件只上升了 {z1 - z0:.4f} m，抓取没有握住")
        self.assertTrue(grip.is_grasping("cube", min_force=1.0), "抬起后已经脱手")
        self.assertGreater(z1, TABLE_TOP + CUBE_HALF[2] + 0.03, "工件仍在桌面附近")

    def test_open_gripper_is_not_reported_as_grasping(self):
        robot, grip, adr, _, _ = self._setup()
        grip.settle(grip.width_max, steps=400, arm_torque_fn=self._hold(robot))
        self.assertFalse(grip.is_grasping("cube"), "张开状态被判成抓住了")
        self.assertEqual(grip.contacts_with("cube"), [])

    def test_contact_force_takes_the_weaker_side(self):
        """`contact_force` 必须取两指中**较小**的法向力，不是求和也不是取最大。

        不再用"把工件横向挪开制造单侧接触"那一套：S 曲线平滑闭合会把工件**归中**，
        实测偏 34 mm 也会被扫回中心（残余 0.3 mm），那个 fixture 已经不成立。
        改成两条确定性判据：
          (a) 正常夹持时，与测试内独立算出的 per-finger 最小值逐位相等，且明显
              小于两者之和——排除"求和"实现；
          (b) 白盒变异：把一侧指尖 geom 从集合里去掉，最小值必须塌到 0——
              排除"取最大"实现。
        """
        robot, grip, adr, _, _ = self._setup()
        grip.close_and_grip(self.CUBE_WIDTH, 60.0,
                            approach_steps=600, grip_steps=600,
                            arm_torque_fn=self._hold(robot))
        m, d = robot.model, robot.data
        self.assertGreater(len(grip.contacts_with("cube")), 1)

        # (a) 独立复算 per-finger 法向力
        geom_to_joint = {}
        for jid in grip.joint_ids:
            bid = int(m.jnt_bodyid[jid])
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) == bid:
                    geom_to_joint[g] = jid
        per = {jid: 0.0 for jid in grip.joint_ids}
        for i in grip.contacts_with("cube"):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            c = d.contact[i]
            jid = geom_to_joint.get(int(c.geom1), geom_to_joint.get(int(c.geom2)))
            if jid is not None:
                per[jid] += abs(float(f6[0]))
        vals = sorted(per.values())
        self.assertAlmostEqual(grip.contact_force("cube"), vals[0], places=9)
        self.assertLess(grip.contact_force("cube"), sum(vals) - 1.0,
                        "看起来是把两指的力加起来了")

        # (b) 白盒变异：只保留一侧指尖 geom
        left_body = int(m.jnt_bodyid[grip.joint_ids[0]])
        saved = set(grip._fingertip_geoms)
        grip._fingertip_geoms = {g for g in saved
                                 if int(m.geom_bodyid[g]) == left_body}
        try:
            self.assertGreater(len(grip.contacts_with("cube")), 0,
                               "变异后单侧仍应有接触，否则用例是空的")
            self.assertEqual(grip.contact_force("cube"), 0.0,
                             "单侧接触竟然报出了非零夹持力（像是取了最大值）")
            self.assertFalse(grip.is_grasping("cube"))
        finally:
            grip._fingertip_geoms = saved

    def test_contact_force_filters_by_body(self):
        """指尖没碰桌子，按 table 过滤必须得 0。"""
        robot, grip, adr, _, _ = self._setup()
        grip.close_and_grip(self.CUBE_WIDTH, 60.0,
                            approach_steps=600, grip_steps=600,
                            arm_torque_fn=self._hold(robot))
        self.assertGreater(grip.contact_force("cube"), 1.0)
        self.assertEqual(grip.contact_force("table"), 0.0)

    def test_smooth_close_recentres_an_offset_object(self):
        """S 曲线平滑闭合会把偏置的工件扫回中心——这是平滑化带来的真实收益。"""
        robot, grip, adr, p_tcp, _ = self._setup()
        robot.data.qpos[adr + 1] = p_tcp[1] + 0.030
        mujoco.mj_forward(robot.model, robot.data)
        grip.close_and_grip(self.CUBE_WIDTH, 60.0,
                            approach_steps=600, grip_steps=600,
                            arm_torque_fn=self._hold(robot))
        residual = abs(float(robot.data.qpos[adr + 1]) - p_tcp[1])
        self.assertLess(residual, 2e-3,
                        f"偏置 30 mm 的工件没有被归中，残余 {residual*1e3:.1f} mm")
        self.assertTrue(grip.is_grasping("cube", min_force=1.0))


class TestTcpIntegration(unittest.TestCase):
    """IK 与阻抗控制必须真的作用在 TCP 上。"""

    def test_ik_solves_for_the_tcp_not_the_flange(self):
        rt = tip_robot()
        ik = DLSInverseKinematics(rt, method="adaptive", lam0=0.05)
        p0, R0 = rt.fk(Q_HOME)
        p_des = p0 + np.array([0.04, -0.03, 0.02])
        res = ik.solve(p_des, R0, Q_HOME.copy(), max_iters=200)
        self.assertTrue(res.success, f"IK 未收敛：{res.fk_pos_err:.2e}")
        p_tcp, _ = rt.fk(res.q)
        self.assertLess(float(np.linalg.norm(p_tcp - p_des)), 1e-4)
        # 法兰**不该**落在目标上——否则说明 IK 其实在解法兰
        p_fl, _ = rt.flange_pose(res.q)
        self.assertGreater(float(np.linalg.norm(p_fl - p_des)), 0.05)
        rt.set_state(np.zeros(rt.n), np.zeros(rt.n))

    def test_impedance_stiffness_acts_at_the_tcp(self):
        """给 TCP 一个位置偏差，力矩必须等于 J_tcpᵀ·K·e（用 TCP 雅可比，不是法兰）。"""
        rt, rf = tip_robot(), flange_robot()
        gains = ImpedanceGains(k_trans=600.0, k_rot=30.0, zeta=1.0)
        ctrl = CartesianImpedanceController(rt, gains, damping_design="fixed")
        p0, R0 = rt.fk(Q_HOME)
        p_des = p0 + np.array([0.01, 0.0, 0.0])
        ctrl.compute(Q_HOME, np.zeros(rt.n), p_des, R0)
        e = np.concatenate([p_des - p0, np.zeros(3)])
        expect = rt.jacobian(Q_HOME).T @ (gains.stiffness() @ e)
        got = ctrl.last_tau_raw - rt.bias(Q_HOME, np.zeros(rt.n))
        self.assertLess(float(np.abs(got - expect).max()), 1e-9)
        wrong = rf.jacobian(Q_HOME).T @ (gains.stiffness() @ e)
        self.assertGreater(float(np.abs(expect - wrong).max()), 1e-3,
                           "TCP 与法兰给出的力矩几乎一样，本用例区分不了")
        rt.set_state(np.zeros(rt.n), np.zeros(rt.n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
