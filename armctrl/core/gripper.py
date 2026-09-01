"""平行二指夹爪（end-effector）模型与控制接口。

为什么单独一个类
----------------
夹爪和手臂在建模与控制上是两套东西：

    手臂    7 个关节型执行器，被 ArmModel 改成直通力矩电机，由外层控制律给 τ。
    夹爪    1 个**腱传动**执行器驱动两根对称的滑动指，自带位置伺服（kp/kd 写在
            模型里）。给它力矩指令没有意义，正确的接口是"宽度 + 夹持力上限"。

把夹爪塞进 arm_joints 会同时破坏两件事：雅可比会多出两列不该有的自由度，
而力矩化会把夹爪的位置伺服拆掉。所以 ArmModel 只管手臂，夹爪由本类接管。

Panda 夹爪的传动（从 panda.xml 独立读出，不是抄参数）
--------------------------------------------------
    finger_joint1 / finger_joint2   滑动关节，各自行程 [0, 0.04] m
    <equality><joint j1 j2>          约束两指同步
    <tendon fixed name="split">      len = 0.5·q1 + 0.5·q2 = 单指位移
    actuator "actuator8"             tendon 位置伺服，ctrlrange [0, 255]
                                     gain = 0.04·100/255,  bias = [0, -kp, -kd]

因此：

    单指位移 s = len(split) = 0.5(q1 + q2)
    **张口宽度 width = q1 + q2 = 2s**，范围 [0, 0.08] m
    ctrl ↔ width 的映射由伺服的不动点给出：kp·(len_target − len) = 0 ⇒
        len_target = gain·ctrl / kp,     width_target = 2·gain·ctrl / kp

本类不硬编码 0/255/0.04，而是从 mjModel 的 gainprm/biasprm/ctrlrange 里
**解出**这个映射，换模型或换夹爪时自动跟随。
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel
from armctrl.planning.trajectory import SCurveProfile


class ParallelJawGripper:
    """腱驱动平行二指夹爪。

    参数
    ----
    robot        已构造的 ArmModel（共用它的 model / data）
    joint_names  两根手指的滑动关节名
    actuator     驱动夹爪的执行器名；None 时自动找唯一的腱传动执行器
    """

    def __init__(
        self,
        robot: ArmModel,
        joint_names: tuple[str, str] = ("finger_joint1", "finger_joint2"),
        actuator: str | None = None,
    ):
        self.robot = robot
        m = robot.model
        self.model, self.data = m, robot.data

        self.joint_ids = []
        for nm in joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid < 0:
                raise ValueError(f"夹爪关节不存在: {nm}")
            self.joint_ids.append(jid)
        self.qpos_idx = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_idx = np.array([m.jnt_dofadr[j] for j in self.joint_ids])

        self.actuator_id = self._resolve_actuator(actuator)
        if m.actuator_trntype[self.actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
            raise ValueError("本类只支持腱传动的平行夹爪执行器")
        self.tendon_id = int(m.actuator_trnid[self.actuator_id, 0])

        # 从模型解出 ctrl → 目标腱长 的仿射映射，而不是硬编码 0.04/255
        gain = float(m.actuator_gainprm[self.actuator_id, 0])
        kp = -float(m.actuator_biasprm[self.actuator_id, 1])
        if kp <= 0.0:
            raise ValueError("夹爪执行器不是位置伺服（biasprm[1] 应为 -kp）")
        self.kp = kp
        self.kd = -float(m.actuator_biasprm[self.actuator_id, 2])
        self._len_per_ctrl = gain / kp
        self.ctrl_range = m.actuator_ctrlrange[self.actuator_id].copy()

        # 腱长 = Σ coef·q。对 split 腱两个 coef 都是 0.5，所以 width = q1+q2 = 2·len
        adr = int(m.tendon_adr[self.tendon_id])
        num = int(m.tendon_num[self.tendon_id])
        coefs = np.array([float(m.wrap_prm[adr + k]) for k in range(num)])
        if not np.allclose(coefs, coefs[0]) or abs(coefs[0]) < 1e-12:
            raise ValueError(f"split 腱的系数不对称: {coefs}")
        self._width_per_len = float(len(coefs))          # = 2

        lo = float(m.jnt_range[self.joint_ids[0], 0])
        hi = float(m.jnt_range[self.joint_ids[0], 1])
        self.width_min = self._width_per_len * lo
        self.width_max = self._width_per_len * hi

        self.force_limit = float(m.actuator_forcerange[self.actuator_id, 1])
        self._fingertip_geoms = self._collect_finger_geoms()
        # 出厂伺服参数：set_grip_force 会就地改写 mjModel，release() 用它还原
        self._factory = (self.kp, self.kd, self.force_limit,
                         int(m.actuator_forcelimited[self.actuator_id]))

    # ------------------------------------------------------------ 解析

    def _resolve_actuator(self, name: str | None) -> int:
        m = self.model
        if name is not None:
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a < 0:
                raise ValueError(f"执行器不存在: {name}")
            return a
        cands = [a for a in range(m.nu)
                 if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON]
        if len(cands) != 1:
            raise ValueError(f"自动识别夹爪执行器失败，腱传动执行器有 {len(cands)} 个")
        return cands[0]

    def _collect_finger_geoms(self) -> set[int]:
        m = self.model
        bodies = {int(m.jnt_bodyid[j]) for j in self.joint_ids}
        return {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies}

    # ------------------------------------------------------------ 状态

    @property
    def width(self) -> float:
        """当前张口宽度 (m) = q1 + q2。"""
        return float(np.sum(self.data.qpos[self.qpos_idx]))

    @property
    def width_rate(self) -> float:
        return float(np.sum(self.data.qvel[self.qvel_idx]))

    @property
    def commanded_width(self) -> float:
        return self.ctrl_to_width(float(self.data.ctrl[self.actuator_id]))

    @property
    def tendon_force(self) -> float:
        """腱上的作动力 (N)。夹持时它就是压在工件上的内力来源。"""
        return float(self.data.actuator_force[self.actuator_id])

    # ------------------------------------------------------------ 映射

    def width_to_ctrl(self, width: float) -> float:
        """张口宽度 (m) → 执行器指令，并裁剪到 ctrlrange。"""
        c = float(width) / (self._width_per_len * self._len_per_ctrl)
        return float(np.clip(c, self.ctrl_range[0], self.ctrl_range[1]))

    def ctrl_to_width(self, ctrl: float) -> float:
        return float(ctrl) * self._len_per_ctrl * self._width_per_len

    # ------------------------------------------------------------ 指令

    def command_width(self, width: float) -> float:
        """写入宽度指令，返回实际写下去的（已裁剪的）目标宽度。"""
        c = self.width_to_ctrl(width)
        self.data.ctrl[self.actuator_id] = c
        return self.ctrl_to_width(c)

    def open(self) -> float:
        return self.command_width(self.width_max)

    def close(self) -> float:
        return self.command_width(self.width_min)

    def set_force_limit(self, force: float) -> None:
        """设置夹持力上限 (N)，即执行器 forcerange。

        ⚠️ 这**只是上限**，不代表能达到。腱力由位置伺服给出：
            f = kp·(len_target − len)
        Menagerie 出厂的 Panda 夹爪 kp = 100 N/m，单指最大行程 0.04 m，
        所以它**最多只能给出 4 N** 腱力，无论 forcerange 写多大。
        要真的夹住东西请用 set_grip_force()。

        ⚠️ 这也是**腱作动力**的上限，不等于指尖法向接触力：腱力按各 wrap 系数
        分到两指（split 腱两个系数都是 0.5），且接触力还取决于工件刚度。
        要测真实夹持力请用 contact_force()。
        """
        f = abs(float(force))
        self.model.actuator_forcerange[self.actuator_id] = [-f, f]
        self.model.actuator_forcelimited[self.actuator_id] = 1
        self.force_limit = f

    def set_servo_gains(self, kp: float, kd: float | None = None) -> None:
        """就地改写夹爪位置伺服的 kp / kd（N/m, N·s/m）。

        kd 缺省按手指等效质量取临界阻尼 kd = 2√(kp·m_finger)，避免高刚度下抖动。
        """
        m = self.model
        kp = float(kp)
        if kp <= 0.0:
            raise ValueError("kp 必须为正")
        if kd is None:
            m_finger = float(m.body_mass[int(m.jnt_bodyid[self.joint_ids[0]])])
            kd = 2.0 * np.sqrt(kp * max(m_finger, 1e-6))
        gain = self._len_per_ctrl * kp          # 保持 ctrl→目标腱长 的映射不变
        m.actuator_gainprm[self.actuator_id, 0] = gain
        m.actuator_biasprm[self.actuator_id, 1] = -kp
        m.actuator_biasprm[self.actuator_id, 2] = -float(kd)
        self.kp, self.kd = kp, float(kd)

    def set_grip_force(self, force: float, squeeze: float = 0.004) -> None:
        """把夹爪配置成能真正输出 `force` 牛顿的夹持力。

        平行夹爪靠"过闭合"产生夹持力：指令宽度比工件窄 `squeeze`，位置伺服的
        位置误差就变成力。取

            kp = force / squeeze          ⇒  |f| = kp·squeeze = force

        再用 forcerange 把腱力**硬性封顶**在 force，于是只要过闭合量 ≥ squeeze，
        实际腱力就精确等于 force，与工件尺寸无关。

        腱力按 split 腱的 wrap 系数分到两指（各 0.5），所以单指法向力 ≈ force/2。
        """
        f = abs(float(force))
        sq = abs(float(squeeze))
        if sq <= 0.0:
            raise ValueError("squeeze 必须为正")
        self.set_servo_gains(f / sq)
        self.set_force_limit(f)
        self.squeeze = sq

    def grasp(self, object_width: float, force: float,
              squeeze: float = 0.004) -> float:
        """按工件宽度夹持：配置夹持力，并命令一个过闭合的目标宽度。

        ⚠️ 这个方法**只设指令**，假设两指已经贴近工件。直接从全张开状态调用它
        是错的：kp 被抬到 force/squeeze（60 N 时是 15000 N/m）之后，伺服会让
        两指以极高刚度砸下去，实测会直接穿过工件并冲破关节限位（宽度跑到
        −0.0476 m），工件被打飞。正确顺序见 close_and_grip()。
        """
        self.set_grip_force(force, squeeze)
        return self.command_width(max(float(object_width) - 2.0 * squeeze,
                                      self.width_min))

    def restore_factory_servo(self) -> None:
        """把伺服 kp/kd 与 forcerange 还原成模型里原本的值。"""
        kp, kd, flim, limited = self._factory
        self.set_servo_gains(kp, kd)
        self.model.actuator_forcerange[self.actuator_id] = [-flim, flim]
        self.model.actuator_forcelimited[self.actuator_id] = limited
        self.force_limit = flim

    def release(self, steps: int = 0, arm_torque_fn=None,
                width: float | None = None, on_step=None) -> None:
        """松开工件。

        **必须先把伺服还原成出厂的软刚度再张开**：夹持时 kp 被抬到 15000 N/m，
        直接张开会让两指以极高刚度弹开，实测把工件从 y=0.100 打到 y=0.031
        并撞下桌面。还原后再张开就只是缓慢释放。
        """
        self.restore_factory_servo()
        w = self.width_max if width is None else float(width)
        if steps > 0:
            self.settle(w, steps, arm_torque_fn, on_step)
        else:
            self.command_width(w)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def move_width(self, target_width: float, v_max: float = 0.05,
                   a_max: float = 0.3, j_max: float = 2.0,
                   arm_torque_fn=None, on_step=None,
                   settle_steps: int = 0) -> None:
        """用 **S 曲线**把宽度指令从当前指令值平滑送到 target_width。

        `command_width()` + `settle()` 是**阶跃指令**：实测从全张开闭到 28 mm 时，
        张口在 0.246 s 内就走完了 90% 行程（峰值闭合速度 0.34 m/s），剩下 1.55 s
        干等着——又快又顿。改成 S 曲线后峰值闭合速度降到 0.029 m/s（12 倍），
        行程均匀铺满整段时间。

        用的就是本项目 planning/trajectory.py 的七段式 S 曲线，jerk 有界。
        """
        dt = self.timestep
        w0 = self.commanded_width
        prof = SCurveProfile(float(target_width) - w0, v_max, a_max, j_max)
        n = max(int(round(prof.T / dt)), 1)
        r = self.robot
        for k in range(n + int(settle_steps)):
            s, _, _ = prof(min(k * dt, prof.T))
            self.command_width(w0 + s)
            if arm_torque_fn is not None:
                tau = arm_torque_fn(r.get_q(), r.get_qd())
                self.data.ctrl[r.arm_actuator_ids] = r.saturate_torque(tau)
            mujoco.mj_step(self.model, self.data)
            if on_step is not None:
                on_step(k)

    def apply_grip_force(self, object_width: float, force: float, steps: int,
                         squeeze: float = 0.004, arm_torque_fn=None,
                         on_step=None) -> None:
        """把夹持力从 0 **斜坡**加到 force，而不是让 kp 从 100 直接跳到 15000。

        一次性跳变会在施力那一拍产生顿挫（指尖瞬时速度实测 1.73 m/s）。
        斜坡加载还原了真实夹爪"合上-加压"的过程。
        """
        w_cmd = max(float(object_width) - 2.0 * float(squeeze), self.width_min)
        r = self.robot
        steps = max(int(steps), 1)
        for k in range(steps):
            self.set_grip_force(max(force * (k + 1) / steps, 1e-6), squeeze)
            self.command_width(w_cmd)
            if arm_torque_fn is not None:
                tau = arm_torque_fn(r.get_q(), r.get_qd())
                self.data.ctrl[r.arm_actuator_ids] = r.saturate_torque(tau)
            mujoco.mj_step(self.model, self.data)
            if on_step is not None:
                on_step(k)

    def close_and_grip(self, object_width: float, force: float,
                       approach_steps: int, grip_steps: int,
                       arm_torque_fn=None, squeeze: float = 0.004,
                       clearance: float = 0.004, on_step=None) -> None:
        """两段式夹持：**先低刚度平滑贴近，再斜坡加夹持力**。

        这就是真实平行夹爪的做法，不是为了让测试好过：

            1. 保持出厂的软伺服（kp = 100 N/m），用 S 曲线把宽度收到
               工件宽度 + clearance。低刚度 + 有界 jerk，接触冲量小，不会顶飞工件。
            2. 再把夹持力从 0 斜坡加到 force（kp 随之升到 force/squeeze），
               把位置误差转成夹持力，同时用 forcerange 把腱力精确封顶在 force。

        approach_steps 是贴近到位后额外的稳定步数；grip_steps 是加载力的斜坡步数。
        一步到位的做法（直接 grasp() 再 settle）实测会失败，见 grasp() 的说明。
        """
        self.restore_factory_servo()                       # 用出厂软伺服贴近
        self.move_width(float(object_width) + float(clearance),
                        arm_torque_fn=arm_torque_fn, on_step=on_step,
                        settle_steps=approach_steps)
        self.apply_grip_force(object_width, force, grip_steps, squeeze,
                              arm_torque_fn, on_step)

    def set_state(self, width: float, rate: float = 0.0) -> None:
        """直接设定两指的运动学状态（用于初始化，不走动力学）。"""
        s = 0.5 * float(width) / 1.0
        self.data.qpos[self.qpos_idx] = 0.5 * float(width)
        self.data.qvel[self.qvel_idx] = 0.5 * float(rate)
        mujoco.mj_forward(self.model, self.data)
        del s

    # ------------------------------------------------------------ 抓取检测

    def contacts_with(self, body_name: str | None = None) -> list[int]:
        """返回指尖 geom 与目标 body（None = 任意其它 body）之间的接触编号。"""
        m, d = self.model, self.data
        target = None
        if body_name is not None:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                raise ValueError(f"body 不存在: {body_name}")
            target = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid}
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            a = g1 in self._fingertip_geoms
            b = g2 in self._fingertip_geoms
            if a == b:                       # 两个都是手指或都不是 → 不算夹持接触
                continue
            other = g2 if a else g1
            if target is None or other in target:
                out.append(i)
        return out

    def contact_force(self, body_name: str | None = None) -> float:
        """指尖与工件之间的**法向**接触力之和 (N)，两指取较小者。

        取较小者是因为稳定夹持要求两侧都压住；只报最大值会把"单指顶住"
        误判成抓稳了。
        """
        m, d = self.model, self.data
        per_body = {jid: 0.0 for jid in self.joint_ids}
        geom_to_joint = {}
        for jid in self.joint_ids:
            bid = int(m.jnt_bodyid[jid])
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) == bid:
                    geom_to_joint[g] = jid
        for i in self.contacts_with(body_name):
            c = d.contact[i]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            jid = geom_to_joint.get(int(c.geom1), geom_to_joint.get(int(c.geom2)))
            if jid is not None:
                per_body[jid] += abs(float(f6[0]))       # 接触系第 0 分量是法向
        return float(min(per_body.values())) if per_body else 0.0

    def is_grasping(self, body_name: str | None = None,
                    min_force: float = 1.0) -> bool:
        """两指都压住同一个工件、且法向力超过阈值，才算抓住了。"""
        return self.contact_force(body_name) >= float(min_force)

    # ------------------------------------------------------------ 便捷

    def settle(self, width: float, steps: int, arm_torque_fn=None,
               on_step=None) -> None:
        """给定宽度指令并推进仿真 steps 步。

        arm_torque_fn(q, qd) -> tau 用来在夹爪动作期间**保持手臂**；不给时
        手臂执行器指令保持不变（力矩化之后就是保持上一拍的力矩）。
        on_step(k) 每步回调，录像时用来抓帧——这样视频里放的就是**同一段**
        被测序列，而不是另写一遍。
        """
        self.command_width(width)
        r = self.robot
        for k in range(int(steps)):
            if arm_torque_fn is not None:
                tau = arm_torque_fn(r.get_q(), r.get_qd())
                self.data.ctrl[r.arm_actuator_ids] = r.saturate_torque(tau)
            mujoco.mj_step(self.model, self.data)
            if on_step is not None:
                on_step(k)


def make_panda_gripper(robot: ArmModel) -> ParallelJawGripper:
    """Franka Panda 标准平行夹爪。"""
    return ParallelJawGripper(robot)
