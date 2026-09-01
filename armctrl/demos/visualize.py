"""控制效果可视化：交互式观看 + 录像。

用法
----
交互式（弹出窗口，可用鼠标转视角）：
    MUJOCO_GL=glfw python armctrl/demos/visualize.py --scene impedance --live

录像（生成 mp4）：
    MUJOCO_GL=glfw python armctrl/demos/visualize.py --scene impedance

可选场景：
    impedance   笛卡尔阻抗控制跟踪直线，中途施加外力把末端推开
    stiffness   同一轨迹下三种刚度的对比（软 / 中 / 硬）
    collision   动量观测器检测碰撞：撞到虚拟障碍后柔顺退让
    teaching    拖动示教：外力拖动机械臂，记录轨迹后回放
    adaptive    自适应控制：从零模型知识开始跟踪，观察误差逐步收敛

说明：所有场景都在末端画一个标记球表示期望位置，便于肉眼判断跟踪效果。
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel
from armctrl.control.impedance import CartesianImpedanceController, ImpedanceGains
from armctrl.control.momentum_observer import MomentumObserver, GravityCompensationTeaching
from armctrl.demos.video import write_video
from armctrl.planning.trajectory import CartesianLine


from armctrl.assets import panda_dir

MENAGERIE = str(panda_dir())
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
DT = 0.002
FPS = 50


# ------------------------------------------------------------------ 场景 XML

def build_scene_xml(with_target_marker: bool = True) -> str:
    """在 Panda 场景基础上加一个半透明标记球表示期望末端位置。

    通过 include 引用原模型，不修改 menagerie 中的任何文件。
    """
    marker = ""
    if with_target_marker:
        marker = """
        <body name="target_marker" mocap="true" pos="0.5 0 0.5">
          <geom type="sphere" size="0.030" rgba="1.0 0.15 0.15 0.60"
                contype="0" conaffinity="0" group="2"/>
        </body>
        <body name="force_marker" mocap="true" pos="0 0 -1">
          <geom type="sphere" size="0.045" rgba="0.15 0.55 1.0 0.55"
                contype="0" conaffinity="0" group="2"/>
        </body>
        """
    xml = f"""<mujoco model="panda_vis">
      <include file="{MENAGERIE}/panda_nohand.xml"/>
      <compiler angle="radian" meshdir="assets" autolimits="true"/>
      <statistic center="0.3 0 0.4" extent="1.1"/>
      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="130" elevation="-22" offwidth="1280" offheight="720"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
                 rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                 rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
                 width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true"
                  texrepeat="5 5" reflectance="0.1"/>
      </asset>
      <worldbody>
        <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
        <camera name="closeup" pos="0.95 -0.80 0.85" xyaxes="0.64 0.77 0 -0.33 0.27 0.90"/>
        {marker}
      </worldbody>
    </mujoco>
    """
    # 被 include 的 panda_nohand.xml 用的是相对 meshdir="assets"，
    # 该路径相对主 XML 所在目录解析。为不污染 menagerie 目录，
    # 在临时目录里做一个指向真实 assets 的符号链接。
    workdir = os.path.join(tempfile.gettempdir(), "armctrl_vis")
    os.makedirs(workdir, exist_ok=True)
    link = os.path.join(workdir, "assets")
    if not os.path.exists(link):
        os.symlink(os.path.join(MENAGERIE, "assets"), link)
    path = os.path.join(workdir, "scene_vis.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def make_robot(xml_path: str) -> ArmModel:
    return ArmModel(
        xml_path=xml_path,
        ee_body="attachment",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
    )


def mocap_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return int(model.body_mocapid[bid])


# ------------------------------------------------------------------ 场景定义

class Scene:
    """场景基类：每步返回力矩，并可更新标记球位置与屏幕文字。"""

    duration = 10.0
    title = ""

    def __init__(self, robot: ArmModel):
        self.robot = robot
        self.setup()

    def setup(self) -> None:
        pass

    def step(self, t, q, v):
        raise NotImplementedError

    def markers(self):
        """返回 (target_pos, force_pos)，None 表示藏到地下。"""
        return None, None

    def overlay(self, t) -> str:
        return ""


class ImpedanceScene(Scene):
    """阻抗控制跟踪直线，第 4–6 秒施加外力把末端推开，观察柔顺退让与恢复。"""

    duration = 12.0
    title = "Cartesian Impedance Control"

    def setup(self):
        self.robot.set_state(Q_HOME, np.zeros(7))
        p0, R0 = self.robot.fk(Q_HOME)
        self.p0, self.R0 = p0.copy(), R0.copy()
        p1 = p0 + np.array([0.15, 0.18, -0.08])
        self.line = CartesianLine(p0, R0, p1, R0, v_max=0.12, a_max=0.5, j_max=4.0)
        self.ctrl = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=500.0, k_rot=35.0, zeta=1.0),
            q_null=Q_HOME,
        )
        self.p_ref = p0.copy()
        self.f_ext = np.zeros(3)

    def step(self, t, q, v):
        # 往返运动
        cyc = self.line.T * 2.0
        tt = t % cyc
        s = tt if tt <= self.line.T else cyc - tt
        p_ref, R_ref, v_ref = self.line(min(s, self.line.T))
        self.p_ref = p_ref
        self.f_ext = np.array([0.0, 0.0, -35.0]) if 4.0 <= t < 6.0 else np.zeros(3)
        return self.ctrl.compute(
            q, v, p_ref, R_ref, v_des=np.concatenate([v_ref, np.zeros(3)])
        )

    def markers(self):
        p_now, _ = self.robot.fk(self.robot.get_q())
        fpos = p_now if np.any(self.f_ext) else None
        return self.p_ref, fpos

    def overlay(self, t):
        s = "施加外力 Fz = -35 N" if np.any(self.f_ext) else "自由空间跟踪"
        return f"K = 500 N/m   {s}"


class StiffnessScene(Scene):
    """同一外力下三种刚度的稳态偏移对比（分三段依次切换）。"""

    duration = 12.0
    title = "Stiffness Comparison"

    def setup(self):
        self.robot.set_state(Q_HOME, np.zeros(7))
        self.p0, self.R0 = self.robot.fk(Q_HOME)
        self.ks = [200.0, 600.0, 1800.0]
        self.ctrls = [
            CartesianImpedanceController(
                self.robot, ImpedanceGains(k_trans=k, k_rot=35.0, zeta=1.0),
                q_null=Q_HOME,
            )
            for k in self.ks
        ]
        self.f_ext = np.zeros(3)
        self.idx = 0

    def step(self, t, q, v):
        self.idx = min(int(t // 4.0), 2)
        self.f_ext = np.array([0.0, 0.0, -25.0]) if (t % 4.0) > 1.0 else np.zeros(3)
        return self.ctrls[self.idx].compute(q, v, self.p0, self.R0)

    def markers(self):
        p_now, _ = self.robot.fk(self.robot.get_q())
        return self.p0, (p_now if np.any(self.f_ext) else None)

    def overlay(self, t):
        k = self.ks[self.idx]
        p_now, _ = self.robot.fk(self.robot.get_q())
        dz = (p_now[2] - self.p0[2]) * 1000.0
        pred = -25.0 / k * 1000.0
        return f"K = {k:.0f} N/m   实测 dz = {dz:+.1f} mm   理论 F/K = {pred:+.1f} mm"


class CollisionScene(Scene):
    """动量观测器检测碰撞：残差超阈值后自动切换为柔顺模式退让。"""

    duration = 12.0
    title = "Momentum Observer Collision Detection"

    def setup(self):
        self.robot.set_state(Q_HOME, np.zeros(7))
        p0, R0 = self.robot.fk(Q_HOME)
        self.p0, self.R0 = p0.copy(), R0.copy()
        p1 = p0 + np.array([0.20, 0.0, 0.0])
        self.line = CartesianLine(p0, R0, p1, R0, v_max=0.10, a_max=0.4, j_max=3.0)
        self.stiff = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=900.0, k_rot=40.0, zeta=1.0), q_null=Q_HOME
        )
        self.soft = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=120.0, k_rot=15.0, zeta=1.0), q_null=Q_HOME
        )
        self.obs = MomentumObserver(self.robot, k_i=30.0, dt=DT)
        self.obs.reset(Q_HOME, np.zeros(7))
        self.detected = False
        self.f_hat = np.zeros(3)
        self.p_ref = p0.copy()
        self.f_ext = np.zeros(3)

    def step(self, t, q, v):
        cyc = self.line.T * 2.0
        tt = t % cyc
        s = tt if tt <= self.line.T else cyc - tt
        p_ref, R_ref, v_ref = self.line(min(s, self.line.T))
        self.p_ref = p_ref

        # 5.0-7.5 秒模拟撞到障碍
        self.f_ext = np.array([-40.0, 0.0, 0.0]) if 5.0 <= t < 7.5 else np.zeros(3)

        ctrl = self.soft if self.detected else self.stiff
        tau = ctrl.compute(q, v, p_ref, R_ref)
        self.obs.update(q, v, tau)
        self.f_hat = self.obs.cartesian_force(q)[:3]
        if np.linalg.norm(self.f_hat) > 12.0:
            self.detected = True
        elif np.linalg.norm(self.f_hat) < 4.0:
            self.detected = False
        return tau

    def markers(self):
        p_now, _ = self.robot.fk(self.robot.get_q())
        return self.p_ref, (p_now if np.any(self.f_ext) else None)

    def overlay(self, t):
        mode = "柔顺退让" if self.detected else "刚性跟踪"
        return (f"外力估计 |F| = {np.linalg.norm(self.f_hat):5.1f} N   "
                f"真实 |F| = {np.linalg.norm(self.f_ext):5.1f} N   模式: {mode}")


class TeachingScene(Scene):
    """拖动示教：前半段用外力拖动（零力模式），后半段回放记录的轨迹。"""

    duration = 16.0
    title = "Drag Teaching and Replay"

    def setup(self):
        self.robot.set_state(Q_HOME, np.zeros(7))
        self.teach = GravityCompensationTeaching(
            self.robot,
            friction_coulomb=np.ones(7) * 0.3,
            friction_viscous=np.ones(7) * 1.0,
            comp_ratio=0.8, damping=1.2,
        )
        self.replay_ctrl = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=700.0, k_rot=40.0, zeta=1.0), q_null=Q_HOME
        )
        self.traj: list[np.ndarray] = []
        self.smoothed = None
        self.f_ext = np.zeros(3)
        self.phase = "teach"
        self.p_ref = None

    def step(self, t, q, v):
        if t < 8.0:
            self.phase = "teach"
            # 用一个缓变外力模拟人手拖动
            w = 2.0 * np.pi / 4.0
            self.f_ext = np.array([
                18.0 * np.sin(w * t),
                18.0 * np.cos(w * t * 0.7),
                10.0 * np.sin(w * t * 0.5),
            ])
            if len(self.traj) == 0 or t - self.traj[-1][0] > 0.02:
                p_now, _ = self.robot.fk(q)
                self.traj.append(np.concatenate([[t], p_now]))
            self.p_ref = None
            return self.teach.compute(q, v)

        self.phase = "replay"
        self.f_ext = np.zeros(3)
        if self.smoothed is None and self.traj:
            arr = np.array(self.traj)
            k = 25
            pad = k // 2
            sm = np.empty_like(arr[:, 1:])
            for j in range(3):
                p = np.pad(arr[:, 1 + j], pad, mode="edge")
                sm[:, j] = np.convolve(p, np.ones(k) / k, mode="valid")
            self.smoothed = np.column_stack([arr[:, 0], sm])
        if self.smoothed is None:
            return self.robot.bias(q, v)
        rt = (t - 8.0) / 8.0 * (self.smoothed[-1, 0] - self.smoothed[0, 0])
        i = int(np.clip(np.searchsorted(self.smoothed[:, 0], rt), 0, len(self.smoothed) - 1))
        self.p_ref = self.smoothed[i, 1:]
        _, R0 = self.robot.fk(Q_HOME)
        return self.replay_ctrl.compute(q, v, self.p_ref, R0)

    def markers(self):
        p_now, _ = self.robot.fk(self.robot.get_q())
        return self.p_ref, (p_now if np.any(self.f_ext) else None)

    def overlay(self, t):
        if self.phase == "teach":
            return f"示教中（零力模式）  已记录 {len(self.traj)} 个点"
        return f"回放中  共 {len(self.traj)} 个点（已平滑）"


SCENES = {
    "impedance": ImpedanceScene,
    "stiffness": StiffnessScene,
    "collision": CollisionScene,
    "teaching": TeachingScene,
}


# ------------------------------------------------------------------ 运行

def run(scene_name: str, live: bool, out: str | None):
    xml = build_scene_xml()
    robot = make_robot(xml)
    m, d = robot.model, robot.data
    scene = SCENES[scene_name](robot)

    tgt_id = mocap_id(m, "target_marker")
    frc_id = mocap_id(m, "force_marker")

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_HOME
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    n_sub = max(int(round(DT / m.opt.timestep)), 1)
    n_steps = int(scene.duration / DT)
    frame_every = max(int(round((1.0 / FPS) / DT)), 1)

    def advance(k):
        t = k * DT
        q = d.qpos[robot.qpos_idx].copy()
        v = d.qvel[robot.qvel_idx].copy()
        tau_full = scene.step(t, q, v)
        tau = robot.saturate_torque(np.asarray(tau_full).ravel()[:7])

        d.qpos[robot.qpos_idx] = q
        d.qvel[robot.qvel_idx] = v
        mujoco.mj_forward(m, d)

        tp, fp = scene.markers()
        d.mocap_pos[tgt_id] = tp if tp is not None else np.array([0.0, 0.0, -1.0])
        d.mocap_pos[frc_id] = fp if fp is not None else np.array([0.0, 0.0, -1.0])

        if np.any(scene.f_ext):
            d.xfrc_applied[robot.ee_body_id, :3] = scene.f_ext
        else:
            d.xfrc_applied[robot.ee_body_id, :] = 0.0

        d.ctrl[:7] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        return t

    if live:
        import mujoco.viewer as viewer
        print(f"[live] {scene.title}  —— 关闭窗口结束")
        with viewer.launch_passive(m, d) as vw:
            k = 0
            import time as _time
            t_wall = _time.time()
            while vw.is_running():
                t = advance(k % n_steps)
                k += 1
                if k % frame_every == 0:
                    vw.sync()
                    target = t_wall + k * DT
                    dtw = target - _time.time()
                    if dtw > 0:
                        _time.sleep(dtw)
        return

    if out is None:
        vdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos")
        os.makedirs(vdir, exist_ok=True)
        out = os.path.join(vdir, f"{scene_name}.mp4")
    renderer = mujoco.Renderer(m, height=720, width=1280)
    frames = []
    for k in range(n_steps):
        t = advance(k)
        if k % frame_every == 0:
            renderer.update_scene(d, camera="closeup")
            frames.append(renderer.render())
    renderer.close()

    written = write_video(frames, out, fps=FPS)
    print(f"[video] {scene.title}")
    print(f"        {len(frames)} 帧 @ {FPS} fps")
    for f in written:
        print(f"        -> {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="impedance", choices=list(SCENES))
    ap.add_argument("--live", action="store_true", help="弹窗交互式观看")
    ap.add_argument("--out", default=None, help="输出 mp4 路径")
    a = ap.parse_args()
    run(a.scene, a.live, a.out)


if __name__ == "__main__":
    main()
