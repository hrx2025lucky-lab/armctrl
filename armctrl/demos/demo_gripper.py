"""末端执行器（TCP）与平行夹爪演示。

三组实验：

    1. TCP vs 法兰      控制点前移 0.1029 m 对运动学与阻抗控制意味着什么
    2. 夹持力标定       指令夹持力 → 实际腱力 / 单指法向接触力
    3. 抓取-提起-放置   完整 pick & place，用接触力和工件位移验收

录像：

    MUJOCO_GL=egl python armctrl/demos/demo_gripper.py --video

录像走的是**同一个** pick_and_place()，只是多传一个抓帧回调，所以视频里放的
就是被测的那一段序列，不是另写一遍的动画。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel, PANDA_TCP_OFFSET, make_panda
from armctrl.core.gripper import ParallelJawGripper
from armctrl.core.kinematics import DLSInverseKinematics, Q_HOME_PANDA
from armctrl.planning.trajectory import CartesianLine
from armctrl.control.impedance import CartesianImpedanceController, ImpedanceGains
from armctrl.demos.video import write_video
from armctrl.assets import panda_xml


PANDA_XML = panda_xml()
Q_HOME = Q_HOME_PANDA
DT = 0.002
CUBE_HALF = (0.012, 0.012, 0.03)
CUBE_MASS = 0.15
TABLE_TOP = 0.383
FPS = 50

# 笛卡尔运动的速度/加速度/jerk 限（S 曲线参考轨迹用）。
# 这些是**参考轨迹**的限制，不是控制器增益：控制器始终是笛卡尔阻抗。
V_MAX, A_MAX, J_MAX = 0.10, 0.4, 2.0


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """相机四元数：MuJoCo 相机看向自身 −z，+y 朝上。

    MjSpec 的相机只接受四元数（没有 XML 里的 xyaxes 简写），所以自己算一个。
    这里不复用 planning/scene.py 的私有实现，避免耦合到别的模块的内部函数。
    """
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, float)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("视线与 up 共线，请换一个 up")
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    R = np.column_stack([right, cam_up, -fwd])       # 相机系 → 世界系
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    return q


def _add_environment(spec) -> None:
    """地面、天空盒、灯光——只影响渲染，不影响动力学结论。

    Menagerie 的 panda.xml 只有机械臂本体，没有 scene.xml 里的地面与天空盒，
    直接渲染出来是悬在黑色虚空里。这里自己补一份，不去 import
    planning/scene.py 的私有函数，避免耦合到别的模块的内部实现。
    """
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.3, 0.5, 0.7]
    sky.rgb2 = [0.0, 0.0, 0.0]
    sky.width, sky.height = 512, 3072

    gp = spec.add_texture()
    gp.name = "groundplane"
    gp.type = mujoco.mjtTexture.mjTEXTURE_2D
    gp.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    gp.mark = mujoco.mjtMark.mjMARK_EDGE
    gp.rgb1 = [0.2, 0.3, 0.4]
    gp.rgb2 = [0.1, 0.2, 0.3]
    gp.markrgb = [0.8, 0.8, 0.8]
    gp.width, gp.height = 300, 300

    mat = spec.add_material()
    mat.name = "groundplane"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    mat.texuniform = True
    mat.texrepeat = [5, 5]
    mat.reflectance = 0.1

    light = spec.worldbody.add_light()
    light.pos = [0.0, 0.0, 2.5]
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = "groundplane"


def build_scene():
    """在 Panda 场景里派生桌面、方块、地面与录像相机（MjSpec，不修改外部 XML）。"""
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    _add_environment(spec)
    table = spec.worldbody.add_body(name="table", pos=[0.5, 0.0, TABLE_TOP - 0.05])
    table.add_geom(name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.15, 0.15, 0.05], rgba=[0.80, 0.55, 0.35, 1.0])
    leg = spec.worldbody.add_body(name="table_leg",
                                  pos=[0.5, 0.0, 0.5 * (TABLE_TOP - 0.10)])
    leg.add_geom(name="table_leg_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.035, 0.035, 0.5 * (TABLE_TOP - 0.10)],
                 rgba=[0.55, 0.35, 0.22, 1.0])
    cube = spec.worldbody.add_body(name="cube",
                                   pos=[0.484, 0.0, TABLE_TOP + CUBE_HALF[2]])
    cube.add_freejoint()
    g = cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(CUBE_HALF), rgba=[0.85, 0.30, 0.20, 1.0])
    g.mass = CUBE_MASS

    # ---- 录像相机 ----
    # 取景由候选扫描定下来（/tmp 上按视距扫描后目视挑选）：从机械臂**前右方**
    # 看过去，视距 1.25 m。不能从背后拍——实测方位角 150° 时手臂自己会把
    # 夹爪和工件挡住。这个机位保证整段搬运（+80 mm 提起、+100 mm 平移）
    # 全程都在画面内。
    focus = [0.36, 0.05, 0.44]
    cam = spec.worldbody.add_camera()
    cam.name = "grasp"
    cam.pos = [1.0963, -0.86, 0.8785]
    cam.quat = _look_at_quat(cam.pos, focus).tolist()
    close = spec.worldbody.add_camera()          # 特写：同方向拉近看夹持细节
    close.name = "grasp_close"
    close.pos = [0.9141, -0.4742, 0.6926]
    close.quat = _look_at_quat(close.pos, [0.49, 0.05, 0.44]).tolist()

    spec.stat.center = focus
    spec.stat.extent = 1.1
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    return spec.compile()


def make_scene_robot():
    model = build_scene()
    robot = ArmModel(model=model, ee_body="hand",
                     arm_joints=[f"joint{i}" for i in range(1, 8)],
                     tcp_offset=PANDA_TCP_OFFSET)
    return robot, ParallelJawGripper(robot)


def reset_scene(robot, grip):
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
    return adr, p_tcp, R_tcp


def joint_hold(robot, q_target=None):
    """关节空间保持。q_target 缺省取**调用时刻**的构型，而不是写死 Q_HOME：
    手臂已经搬到别处时还拿 Q_HOME 当目标，会把工件一起拽回来。"""
    q_ref = robot.get_q().copy() if q_target is None else np.asarray(q_target, float)

    def fn(q, v):
        return robot.mass_matrix(q) @ (400.0 * (q_ref - q) - 40.0 * v) \
            + robot.bias(q, v)
    return fn


def servo_tcp(robot, p_des, R_des, steps, k_trans=1500.0, k_rot=80.0, on_step=None):
    """**阶跃**目标 + 笛卡尔阻抗。只保留给实验 4 做对照，正常动作请用 move_line。

    实测 80 mm 阶跃：峰值速度 0.479 m/s、峰值加速度 24.8 m/s²、
    峰值 jerk 3098 m/s³，90% 行程在 0.296 s 内冲完，剩下 4.7 s 干等。
    """
    ctrl = CartesianImpedanceController(
        robot, ImpedanceGains(k_trans=k_trans, k_rot=k_rot, zeta=1.0), q_null=Q_HOME)
    for k in range(steps):
        q, v = robot.get_q(), robot.get_qd()
        robot.data.ctrl[robot.arm_actuator_ids] = ctrl.compute(q, v, p_des, R_des)
        mujoco.mj_step(robot.model, robot.data)
        if on_step is not None:
            on_step(k)


def move_line(robot, p_to, R_to, v_max=V_MAX, a_max=A_MAX, j_max=J_MAX,
              settle=0.4, k_trans=1500.0, k_rot=80.0, on_step=None):
    """沿直线平滑移动 TCP：**S 曲线时间律 + slerp 姿态插补 + 速度前馈**。

    控制器没变，还是笛卡尔阻抗（K = 1500 N/m, ζ = 1）。变的是**给它喂什么参考**：

        阶跃      p_des 一次性跳到终点，阻抗环就是一个二阶系统的阶跃响应，
                  开头猛冲、之后干等。jerk 只受闭环带宽限制。
        S 曲线    p_des(t)、R_des(t)、v_des(t) 逐拍给出，jerk 分段常值且有界，
                  再把 v_des 作为前馈送进阻抗律，跟踪误差进一步下降。

    用的就是本项目 planning/trajectory.py 里的 CartesianLine（七段式 S 曲线 + slerp）。
    """
    p_from, R_from = robot.fk()
    line = CartesianLine(p_from, R_from, p_to, R_to,
                         v_max=v_max, a_max=a_max, j_max=j_max)
    ctrl = CartesianImpedanceController(
        robot, ImpedanceGains(k_trans=k_trans, k_rot=k_rot, zeta=1.0), q_null=Q_HOME)
    n = int(round((line.T + settle) / DT))
    for k in range(n):
        t = min(k * DT, line.T)
        p_d, R_d, v_d = line(t)
        q, v = robot.get_q(), robot.get_qd()
        robot.data.ctrl[robot.arm_actuator_ids] = ctrl.compute(
            q, v, p_d, R_d, v_des=np.concatenate([v_d, np.zeros(3)]))
        mujoco.mj_step(robot.model, robot.data)
        if on_step is not None:
            on_step(k)
    return line.T


# ------------------------------------------------------------------ 实验 1


def exp_tcp_vs_flange():
    print("\n" + "=" * 78)
    print("实验 1：TCP（指尖抓取中心） vs 法兰（hand 原点）")
    print("=" * 78)
    tip = make_panda()
    flange = make_panda(tcp="flange")
    p_t, R = tip.fk(Q_HOME)
    p_f, _ = flange.fk(Q_HOME)

    print(f"  TCP 偏置（hand 系）        {PANDA_TCP_OFFSET}  |o| = "
          f"{np.linalg.norm(PANDA_TCP_OFFSET):.4f} m")
    print(f"    = finger body 原点 0.0584 + 指尖夹持垫包络中心 0.0445")
    print(f"  Q_HOME 处法兰位置          {np.array2string(p_f, precision=5)}")
    print(f"  Q_HOME 处 TCP  位置        {np.array2string(p_t, precision=5)}")
    print(f"  两者距离                   {np.linalg.norm(p_t - p_f):.6f} m")

    Jt, Jf = tip.jacobian(Q_HOME), flange.jacobian(Q_HOME)
    print(f"\n  雅可比线速度行最大差       {np.abs(Jt[:3] - Jf[:3]).max():.6f}")
    print(f"  雅可比角速度行最大差       {np.abs(Jt[3:] - Jf[3:]).max():.3e}  (应为 0)")
    print(f"  可操作度  TCP / 法兰       {tip.manipulability(Q_HOME):.9f} / "
          f"{flange.manipulability(Q_HOME):.9f}   (det(T)=1 ⇒ 不变)")
    print(f"  条件数    TCP / 法兰       {tip.condition_number(Q_HOME):.4f} / "
          f"{flange.condition_number(Q_HOME):.4f}   (T 非正交 ⇒ 会变)")

    ik_t = DLSInverseKinematics(tip)
    ik_f = DLSInverseKinematics(flange)
    print(f"\n  标称可达半径  TCP / 法兰   {ik_t.char_length:.6f} / "
          f"{ik_f.char_length:.6f} m")
    print(f"  阻尼阈值 σ0_pos  TCP/法兰  {ik_t.sigma0_pos:.6f} / 0.025468")
    print(f"  阻尼阈值 σ0_pose TCP/法兰  {ik_t.sigma0_pose:.6f} / 0.008934")

    # 同一个笛卡尔目标，两种口径解出的关节角相差多少
    p_goal = p_t + np.array([0.03, -0.02, 0.01])
    r1 = ik_t.solve(p_goal, R, Q_HOME.copy())
    r2 = ik_f.solve(p_goal, R, Q_HOME.copy())
    d_tcp, _ = tip.fk(r2.q)
    print(f"\n  对同一笛卡尔目标求 IK：")
    print(f"    按 TCP 解     成功={r1.success}  TCP 落点误差 {r1.fk_pos_err*1e3:.4f} mm")
    print(f"    按法兰解      成功={r2.success}  实际 TCP 偏离目标 "
          f"{np.linalg.norm(d_tcp - p_goal)*1e3:.2f} mm  ← 控制点选错的代价")


# ------------------------------------------------------------------ 实验 2


def exp_grip_force():
    print("\n" + "=" * 78)
    print("实验 2：夹持力标定（指令 → 腱力 → 单指法向接触力）")
    print("=" * 78)
    robot, grip = make_scene_robot()
    print(f"  出厂伺服 kp = {grip.kp:.0f} N/m, kd = {grip.kd:.0f} N·s/m, "
          f"forcerange = ±{grip.force_limit:.0f} N")
    print(f"  单指行程 0~{0.5*grip.width_max:.3f} m ⇒ 出厂伺服腱力上界 "
          f"{grip.kp * 0.5 * grip.width_max:.1f} N（与 forcerange 无关）")
    print(f"  张口宽度范围 {grip.width_min:.3f} ~ {grip.width_max:.3f} m，"
          f"ctrl 范围 {grip.ctrl_range[0]:.0f}~{grip.ctrl_range[1]:.0f}")

    print(f"\n  {'指令力':>8s} {'伺服 kp':>10s} {'腱力':>10s} {'单指法向力':>12s} "
          f"{'稳态宽度':>10s} {'接触数':>7s}")
    print("  " + "-" * 62)
    w_obj = 2.0 * CUBE_HALF[1]
    for f_cmd in (0.0, 10.0, 20.0, 40.0, 60.0):
        robot, grip = make_scene_robot()
        reset_scene(robot, grip)
        if f_cmd <= 0.0:
            grip.settle(0.0, 1500, joint_hold(robot, Q_HOME))
        else:
            grip.close_and_grip(w_obj, f_cmd, 900, 900, joint_hold(robot, Q_HOME))
        print(f"  {f_cmd:8.1f} {grip.kp:10.0f} {abs(grip.tendon_force):10.2f} "
              f"{grip.contact_force('cube'):12.2f} {grip.width:10.4f} "
              f"{len(grip.contacts_with('cube')):7d}")
    print(f"\n  工件宽度 {w_obj:.3f} m。0 N 一行用的是出厂伺服（只抬 forcerange），"
          f"说明光调上限没用。")


# ------------------------------------------------------------------ 实验 3


def pick_and_place(robot, grip, on_step=None):
    """完整 pick & place 序列。**打印版与录像版共用这一份代码**。

    on_step(k) 每个仿真步回调一次，录像时用来抓帧；不传就是纯仿真。
    返回每个阶段结束时的 (阶段名, 方块位置, 张口, 单指法向力)。
    """
    adr, p_tcp, R_tcp = reset_scene(robot, grip)
    d = robot.data
    w_obj = 2.0 * CUBE_HALF[1]

    def snap(name):
        return (name, d.qpos[adr:adr + 3].copy(), grip.width,
                grip.contact_force("cube"))

    stages = [snap("初始（张开，未接触）")]

    grip.close_and_grip(w_obj, 60.0, 400, 500, joint_hold(robot, Q_HOME),
                        on_step=on_step)
    stages.append(snap("夹持（60 N）"))

    p_up = p_tcp + np.array([0.0, 0.0, 0.08])
    move_line(robot, p_up, R_tcp, on_step=on_step)
    stages.append(snap("提起 +80 mm"))

    p_side = p_up + np.array([0.0, 0.10, 0.0])
    move_line(robot, p_side, R_tcp, on_step=on_step)
    stages.append(snap("平移 +100 mm"))

    p_down = p_side - np.array([0.0, 0.0, 0.075])
    move_line(robot, p_down, R_tcp, on_step=on_step)
    stages.append(snap("下放"))

    # 松开：先还原出厂软伺服，再用 S 曲线张开（就地保持，别拽回 Q_HOME）
    hold_here = joint_hold(robot)
    grip.restore_factory_servo()
    grip.move_width(grip.width_max, arm_torque_fn=hold_here, on_step=on_step,
                    settle_steps=300)
    move_line(robot, p_down + np.array([0.0, 0.0, 0.06]), R_tcp, on_step=on_step)
    stages.append(snap("松开并撤离"))
    return stages


def report_pick_and_place(stages) -> int:
    print(f"  {'阶段':<22s} {'方块 x':>9s} {'方块 y':>9s} {'方块 z':>9s} "
          f"{'张口':>8s} {'夹持力':>8s}")
    print("  " + "-" * 70)
    for name, p, w, f in stages:
        print(f"  {name:<22s} {p[0]:9.4f} {p[1]:9.4f} {p[2]:9.4f} "
              f"{w:8.4f} {f:8.2f}")
    p0, pf = stages[0][1], stages[-1][1]
    print(f"\n  方块净位移  Δy = {(pf-p0)[1]*1e3:+.1f} mm, "
          f"Δz = {(pf-p0)[2]*1e3:+.1f} mm")
    lifted = stages[2][1][2] - stages[1][1][2]
    print(f"  提起阶段工件上升 {lifted*1e3:+.1f} mm（TCP 指令 +80.0 mm）"
          f"  滑移 {80.0 - lifted*1e3:.1f} mm")
    ok = (pf[1] - p0[1]) > 0.05 and abs(pf[2] - p0[2]) < 0.02
    print(f"  结论：{'搬运成功' if ok else '搬运失败'}")
    return 0 if ok else 1


def exp_pick_and_place():
    print("\n" + "=" * 78)
    print("实验 3：抓取 → 提起 → 平移 → 放下")
    print("=" * 78)
    robot, grip = make_scene_robot()
    return report_pick_and_place(pick_and_place(robot, grip))


# ------------------------------------------------------------------ 录像


def record(out: str | None = None, camera: str = "grasp"):
    """录制 pick & place。走的是同一个 pick_and_place()，只多一个抓帧回调。"""
    robot, grip = make_scene_robot()
    m, d = robot.model, robot.data
    frame_every = max(int(round(1.0 / (FPS * DT))), 1)
    renderer = mujoco.Renderer(m, height=720, width=1280)
    frames = []
    counter = {"k": 0}

    def grab(_):
        if counter["k"] % frame_every == 0:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render())
        counter["k"] += 1

    # 开头停一拍便于看清初始状态
    reset_scene(robot, grip)
    for _ in range(int(0.6 / DT)):
        d.ctrl[robot.arm_actuator_ids] = joint_hold(robot, Q_HOME)(
            robot.get_q(), robot.get_qd())
        mujoco.mj_step(m, d)
        grab(0)

    stages = pick_and_place(robot, grip, on_step=grab)

    for _ in range(int(0.8 / DT)):                    # 结尾再停一拍
        mujoco.mj_step(m, d)
        grab(0)
    renderer.close()

    if out is None:
        vdir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "videos")
        os.makedirs(vdir, exist_ok=True)
        out = os.path.join(vdir, "gripper.mp4")
    written = write_video(frames, out, fps=FPS)
    print(f"[video] {len(frames)} 帧 @ {FPS} fps，相机 {camera}")
    for f in written:
        print(f"        -> {f}")
    return report_pick_and_place(stages)


# ------------------------------------------------------------------ 实验 4


def _profile_stats(P, dt):
    V = np.gradient(P, dt, axis=0)
    A = np.gradient(V, dt, axis=0)
    J = np.gradient(A, dt, axis=0)
    return (float(np.linalg.norm(V, axis=1).max()),
            float(np.linalg.norm(A, axis=1).max()),
            float(np.linalg.norm(J, axis=1).max()))


def exp_motion_smoothness():
    """同一个控制器、同一个目标，只换**参考轨迹**：阶跃 vs S 曲线。"""
    print("\n" + "=" * 78)
    print("实验 4：运动平滑性 —— 控制器不变，只换参考轨迹")
    print("=" * 78)
    print("  控制器自始至终是**笛卡尔阻抗** τ = Jᵀ[K(x_d−x) + D(ẋ_d−ẋ)] + h(q,q̇)，")
    print("  K = 1500 N/m, ζ = 1（D = 2ζ√(ΛK)，随构型自适应）。")
    print("  区别只在于给它喂什么 x_d(t)：")
    print("    阶跃    x_d 一次性跳到终点 → 阻抗环退化成二阶系统的阶跃响应")
    print(f"    S 曲线  x_d(t)/ẋ_d(t) 逐拍给出，v/a/j 限 = "
          f"{V_MAX}/{A_MAX}/{J_MAX}，并把 ẋ_d 作为前馈\n")

    rows = []
    for label, use_scurve in (("阶跃目标", False), ("S 曲线 + 速度前馈", True)):
        robot, grip = make_scene_robot()
        _, p0, R0 = reset_scene(robot, grip)
        grip.close_and_grip(2.0 * CUBE_HALF[1], 60.0, 400, 500,
                            joint_hold(robot, Q_HOME))
        p_goal = p0 + np.array([0.0, 0.0, 0.08])
        P = []
        cb = lambda k: P.append(robot.fk()[0].copy())
        if use_scurve:
            T = move_line(robot, p_goal, R0, on_step=cb)
        else:
            T = 0.0
            servo_tcp(robot, p_goal, R0, 2500, on_step=cb)
        P = np.array(P)
        v, a, j = _profile_stats(P, DT)
        z = P[:, 2] - P[0, 2]
        t = np.arange(len(P)) * DT
        i90 = int(np.argmax(z > 0.9 * 0.08)) if (z > 0.9 * 0.08).any() else -1
        rows.append((label, v, a, j, t[i90], t[-1],
                     float(np.linalg.norm(P[-1] - p_goal)) * 1e3))

    print(f"  {'参考':<20s} {'峰值速度':>10s} {'峰值加速':>10s} {'峰值jerk':>11s} "
          f"{'90%行程':>9s} {'总时长':>8s} {'终点误差':>9s}")
    print("  " + "-" * 84)
    for lab, v, a, j, t90, tt, e in rows:
        print(f"  {lab:<20s} {v:10.4f} {a:10.2f} {j:11.1f} {t90:8.3f}s "
              f"{tt:7.2f}s {e:8.3f}mm")
    print(f"\n  单位：m/s, m/s², m/s³。80 mm 提升动作。")
    print(f"  jerk 降低 {rows[0][3]/max(rows[1][3],1e-9):.0f} 倍，"
          f"加速度降低 {rows[0][2]/max(rows[1][2],1e-9):.0f} 倍。")
    print("  阶跃那一行不是「控制器不好」，是**根本没有轨迹规划**：把阶跃喂给")
    print("  弹簧-阻尼系统，它只能以闭环带宽允许的最快速度冲过去。")


def main():
    print("=" * 78)
    print("末端执行器（TCP）与平行夹爪  |  Franka Panda + Hand  |  MuJoCo")
    print("=" * 78)
    exp_tcp_vs_flange()
    exp_grip_force()
    exp_motion_smoothness()
    rc = exp_pick_and_place()
    print("\n" + "=" * 78)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true",
                    help="渲染 mp4/webm 到 armctrl/videos/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera", default="grasp", choices=["grasp", "grasp_close"])
    a = ap.parse_args()
    raise SystemExit(record(a.out, a.camera) if a.video else main())
