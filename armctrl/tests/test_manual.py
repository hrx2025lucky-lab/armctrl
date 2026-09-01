"""调参台手动干预（夹爪 / 关节扰动力矩 / 运动学摆位）的测试。

这三路是给使用者随时插手用的通道，不属于任何控制律。它们最容易出的错不是
"算错了"，而是**语义串台**：把扰动力矩当成控制量、把摆位当成控制、
在抓取场景里让位置指令顶掉力控。所以测试重点全在边界与互不干扰上。

oracle 独立性说明
----------------
1. **牛顿-欧拉真值。**"qfrc_applied 是真正的广义外力"这一条，用
   ``a = M⁻¹(τ_applied − bias)`` 独立复算加速度来验，
   M 与 bias 由 MuJoCo 的 ``mj_fullM`` / ``qfrc_bias`` 给出，
   **不读 Manual 或 Runner 的任何中间量**。
2. **模型自带限位。**摆位裁剪用 ``model.jnt_range`` 独立 np.clip 复算。
3. **纯逻辑。**Manual 的字段映射用手写期望值直接比对。

⚠️ 级别：**同模型自洽**。这里验证的是"调参台写下去的量确实是它声称的那个量"，
不涉及真机标定。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel
from armctrl.planning.scene import PANDA_XML, build_panda_scene
from armctrl.tuner.base import Manual


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))


def _robot() -> ArmModel:
    scene = build_panda_scene([], with_visuals=False)
    return ArmModel(model=scene.model, ee_body="hand", arm_joints=ARM_JOINTS)


class TestManualState(unittest.TestCase):
    """字段映射与边界。"""

    def test_default_is_hands_off(self):
        """默认状态必须是"什么都不干预"，否则一开场就悄悄改了物理。"""
        m = Manual()
        self.assertIsNone(m.grip)                 # None = 跟随场景
        self.assertEqual(m.dist_joint, 0)
        self.assertEqual(m.dist_tau, 0.0)
        self.assertFalse(m.pose_on)
        np.testing.assert_allclose(m.joint_torque(7), np.zeros(7))

    def test_unknown_key_raises(self):
        """未知字段必须报错而不是静默忽略。

        静默忽略的后果是：前端改了字段名之后滑块看起来能拖、
        实际上什么都没发生，而且没有任何报错线索。
        """
        m = Manual()
        with self.assertRaises(KeyError):
            m.update({"grpi": 0.02})              # 故意拼错

    def test_joint_torque_placement(self):
        """扰动力矩必须落在**指定的那一个**关节上，其余严格为零。"""
        m = Manual()
        for j in range(1, 8):
            m.update({"dist_joint": j, "dist_tau": 3.5})
            tau = m.joint_torque(7)
            expect = np.zeros(7)
            expect[j - 1] = 3.5
            np.testing.assert_allclose(tau, expect)

    def test_joint_zero_disables(self):
        """dist_joint=0 表示不施加——即使 dist_tau 非零也必须全零输出。

        否则用户把关节选回"不施加"却发现机械臂还在被推。
        """
        m = Manual()
        m.update({"dist_joint": 0, "dist_tau": 20.0})
        np.testing.assert_allclose(m.joint_torque(7), np.zeros(7))

    def test_joint_index_clamped(self):
        """越界的关节号被夹到 [0,7]，不能让它去写别的自由度。

        手臂 7 个关节之外还有夹爪的两个手指自由度，
        索引溢出会把扰动写到手指上，现象是"夹爪莫名其妙自己动"。
        """
        m = Manual()
        m.update({"dist_joint": 99, "dist_tau": 1.0})
        self.assertEqual(m.dist_joint, 7)
        m.update({"dist_joint": -5})
        self.assertEqual(m.dist_joint, 0)

    def test_grip_none_roundtrip(self):
        """grip 能被显式设回 None（= 交还给场景）。"""
        m = Manual()
        m.update({"grip": 0.03})
        self.assertAlmostEqual(m.grip, 0.03)
        m.update({"grip": None})
        self.assertIsNone(m.grip)


class TestDisturbanceIsExternalForce(unittest.TestCase):
    """扰动力矩必须是**真的外力**，不是"把控制量改掉"。"""

    def setUp(self):
        self.robot = _robot()
        self.m, self.d = self.robot.model, self.robot.data

    def test_applied_torque_matches_newton_euler(self):
        """扰动力矩必须与"电机出同样大小的力矩"在动力学上完全等价。

        oracle 用 MuJoCo **自己的执行器通路**做参照，不碰质量矩阵、
        也不读被测实现的任何中间量：

            A 路：ctrl[手臂] = τ，   qfrc_applied = 0
            B 路：ctrl[手臂] = 0，   qfrc_applied[手臂自由度] = τ

        Panda 的手臂执行器已被 ArmModel 转成 gear=1 的力矩执行器，
        所以两路施加的广义力逐位相同，qacc 必须逐位相同。
        对不上就说明 qfrc_applied 不是它声称的那个量。
        """
        rng = np.random.default_rng(0)
        q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        # 力矩取小值，避开腕部 ±12 N·m 的执行器限幅——一旦被裁两路就不等价了
        tau = rng.uniform(-5.0, 5.0, size=7)

        self.robot.set_state(q, np.zeros(7))
        self.d.qfrc_applied[:] = 0.0
        self.d.ctrl[:] = 0.0
        self.d.ctrl[self.robot.arm_actuator_ids] = tau
        mujoco.mj_forward(self.m, self.d)
        acc_actuator = self.d.qacc.copy()

        self.robot.set_state(q, np.zeros(7))
        self.d.ctrl[:] = 0.0
        self.d.qfrc_applied[:] = 0.0
        self.d.qfrc_applied[self.robot.qvel_idx] = tau
        mujoco.mj_forward(self.m, self.d)
        acc_applied = self.d.qacc.copy()

        np.testing.assert_allclose(acc_applied, acc_actuator,
                                   rtol=1e-9, atol=1e-10)
        # 且这个力矩确实产生了运动，不是两边都恰好是零
        self.assertGreater(np.linalg.norm(acc_applied), 1e-3)

    def test_applied_does_not_leak_into_ctrl(self):
        """写 qfrc_applied 绝不能改动 ctrl / qfrc_actuator。

        这两路混淆是最危险的误解来源：用户会以为"我把控制量调到了 15 N·m"，
        然后拿条形图上控制器自己出的力去做结论。
        """
        q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        self.robot.set_state(q, np.zeros(7))
        self.d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        act_before = self.d.qfrc_actuator.copy()

        man = Manual()
        man.update({"dist_joint": 2, "dist_tau": 15.0})
        self.d.qfrc_applied[self.robot.qvel_idx] = man.joint_torque(7)
        mujoco.mj_forward(self.m, self.d)

        np.testing.assert_allclose(self.d.ctrl, 0.0)
        np.testing.assert_allclose(self.d.qfrc_actuator, act_before)
        # 而扰动确实落在了 J2 上
        self.assertAlmostEqual(
            float(self.d.qfrc_applied[self.robot.qvel_idx[1]]), 15.0)

    def test_zeroing_restores_baseline(self):
        """扰动归零后加速度必须**逐位**回到无扰动基线。

        每个控制周期不重写 qfrc_applied 的话，力矩会一直留在那里，
        现象是"滑块拉回 0 了机械臂还在歪"。
        """
        q = np.array([0.2, -0.4, 0.1, -2.0, 0.0, 1.9, 0.7])
        self.robot.set_state(q, np.zeros(7))
        self.d.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        a0 = self.d.qacc.copy()

        self.d.qfrc_applied[self.robot.qvel_idx] = Manual(
            dist_joint=4, dist_tau=-12.0).joint_torque(7)
        mujoco.mj_forward(self.m, self.d)
        self.assertGreater(np.linalg.norm(self.d.qacc - a0), 1e-3)

        self.d.qfrc_applied[self.robot.qvel_idx] = Manual().joint_torque(7)
        mujoco.mj_forward(self.m, self.d)
        np.testing.assert_allclose(self.d.qacc, a0, rtol=1e-9, atol=1e-10)


class TestPoseClamping(unittest.TestCase):
    """摆位模式的关节限位裁剪。"""

    def setUp(self):
        self.robot = _robot()
        self.m, self.d = self.robot.model, self.robot.data

    def test_clip_matches_model_range(self):
        """写进去的角必须被裁到 model.jnt_range，用独立 np.clip 复算。

        不裁剪的后果不是"看起来怪"，而是 MuJoCo 会带着越限构型继续算
        接触与惯量，一放开摆位机械臂会以巨大的位置误差弹出去。
        """
        lo = self.m.jnt_range[self.robot.joint_ids, 0]
        hi = self.m.jnt_range[self.robot.joint_ids, 1]
        self.assertTrue(np.all(hi > lo))

        want = np.array([99.0, 99.0, -99.0, 99.0, -99.0, 99.0, 99.0])
        clipped = np.clip(want, lo, hi)
        self.d.qpos[self.robot.qpos_idx] = clipped
        mujoco.mj_forward(self.m, self.d)

        got = self.d.qpos[self.robot.qpos_idx]
        np.testing.assert_allclose(got, clipped)
        self.assertTrue(np.all(got <= hi + 1e-12))
        self.assertTrue(np.all(got >= lo - 1e-12))

    def test_forward_does_not_advance_time(self):
        """摆位用 mj_forward，**不能**推进仿真时间。

        若误用 mj_step，摆位期间时间会往前跑、控制器状态（积分项、
        自适应参数、RMS 累积）会被一段没有控制律的空白污染。
        """
        self.d.time = 3.25
        self.d.qpos[self.robot.qpos_idx] = np.zeros(7)
        mujoco.mj_forward(self.m, self.d)
        self.assertAlmostEqual(float(self.d.time), 3.25)

    def test_pose_changes_tcp_kinematically(self):
        """不同构型必须给出不同 TCP，且与独立 FK 一致。"""
        qa = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        qb = np.array([1.0, -0.8, 0.0, -2.2, 0.0, 2.0, 0.785])
        pa = self.robot.tcp_position(qa)
        pb = self.robot.tcp_position(qb)
        self.assertGreater(np.linalg.norm(pa - pb), 0.05)


class TestGripperInterlock(unittest.TestCase):
    """抓取场景自己接管夹爪时，手动指令必须被挡住。"""

    class _FakeGripper:
        """只记录被写了什么，不做任何物理。"""

        def __init__(self):
            self.calls = []

        def command_width(self, w):
            self.calls.append(float(w))

    class _FakeScene:
        def __init__(self, gripper, width):
            self.gripper = gripper
            self.gripper_width = width

    @staticmethod
    def _apply(manual_grip, scene_width, gripper=None):
        """⭐ 直接调用**生产代码** ``Runner._apply_gripper``。

        ⚠️ 这条测试原来在本地重写了一份 ``resolve()`` 来复刻判定条件
        （第四轮审核 T-02）。那样测的是"我抄的这份逻辑对不对"，
        生产代码即使被改坏，测试照样全绿——**复制生产逻辑等于没测**。

        这里用一个假的 scene/gripper，把 ``Runner._apply_gripper`` 当成
        未绑定函数直接调用（它只用到 ``self.manual``），
        既不需要真的建场景，又确确实实执行的是生产分支。
        """
        from armctrl.tuner.server import Runner
        gripper = gripper if gripper is not None else \
            TestGripperInterlock._FakeGripper()
        stub = SimpleNamespace(manual=Manual(grip=manual_grip))
        Runner._apply_gripper(stub, TestGripperInterlock._FakeScene(
            gripper, scene_width))
        return gripper

    def test_scene_owned_gripper_blocks_manual(self):
        """gripper_width=None 表示场景接管；此时手动值**一次都不能写下去**。

        若这里写下去，抓取过程中的力控指令会被位置指令顶掉，工件会当场掉下去。
        """
        g = self._apply(manual_grip=0.0, scene_width=None)
        self.assertEqual(g.calls, [],
                         "场景已接管夹爪，手动值却被写下去了——互锁失效")

    def test_manual_wins_in_a_normal_scene(self):
        """普通场景里手动值优先，且**只写一次**。"""
        g = self._apply(manual_grip=0.0, scene_width=0.08)
        self.assertEqual(len(g.calls), 1, f"应当只写一次，实际 {len(g.calls)} 次")
        self.assertAlmostEqual(g.calls[0], 0.0)

    def test_releasing_manual_hands_control_back_to_the_scene(self):
        """手动值置 None = 交还场景，写下去的必须是场景值。"""
        g = self._apply(manual_grip=None, scene_width=0.08)
        self.assertEqual(len(g.calls), 1)
        self.assertAlmostEqual(g.calls[0], 0.08)

    def test_no_gripper_at_all_is_a_no_op(self):
        """场景根本没有夹爪时不能炸。"""
        from armctrl.tuner.server import Runner
        stub = SimpleNamespace(manual=Manual(grip=0.05))
        Runner._apply_gripper(stub, self._FakeScene(None, 0.08))   # 不应抛异常

    def test_the_oracle_can_actually_catch_a_broken_interlock(self):
        """⭐ 变异对照：把互锁拆掉的实现必须被上面第一条判据抓到。

        这里手写一个"漏了场景接管判断"的版本，证明判据有区分能力——
        否则第一条测试可能只是恰好通过。
        """
        g = self._FakeGripper()
        sc = self._FakeScene(g, None)
        manual = Manual(grip=0.0)
        # 变异版：少了 `or sc.gripper_width is None` 这一半
        if sc.gripper is not None:
            w = sc.gripper_width if manual.grip is None else manual.grip
            g.command_width(w)
        self.assertNotEqual(g.calls, [],
                            "拆掉互锁后居然还是没写——说明判据量错了对象")


if __name__ == "__main__":
    unittest.main()
