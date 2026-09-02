"""有障碍环境下的路径规划 → 平滑 → 时间参数化 → 执行 全流程演示。

场景
----
Panda 前方立一根柱子，柱子上方横一道梁。末端要从柱子左侧摆到右侧。
起终点各自都是合法构型，但**两者的关节空间直线插值会把手臂扫过柱子**——
这正是"只有插值、没有规划"会出事的典型情形。

流水线
------
    1. 反例基线   起点到终点直接做关节插值，量出撞得多深
    2. 路径规划   RRT-Connect 在关节空间找一条无碰撞折线
    3. 平滑       短路平滑把折线压短、把路径点压少
    4. 时间参数化 逐段七段 S 曲线，得到 q(t)、q̇(t)、q̈(t)
    5. 执行       MuJoCo 中用计算力矩控制跟踪，独立复核实际间隙与真实接触

诚实说明
--------
* 控制器用的动力学参数与被控对象（MuJoCo 模型）来自同一套参数，
  属于**同模型自洽**，跟踪误差不代表真机精度。
* 跟踪误差按**同一时刻、施加控制之前**记录：t_k 时刻的实际 q 与 q_d(t_k) 相减。
  不要用 q_d(t_k) 去减下一拍的 q，那会把一整拍的运动算进误差里。
* 碰撞检测是按分辨率采样的，不是连续保证；报告的"最小间隙"来自
  比规划分辨率密得多的复核采样，但仍是采样。
* 这里的 MuJoCo 是仿真被控对象，不是真实机器人。

用法
----
    export PYTHONPATH=.
    python armctrl/demos/demo_planning.py
    MUJOCO_GL=glfw python armctrl/demos/demo_planning.py --video
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel
from armctrl.demos.video import video_path, write_video
from armctrl.identification.regressor import DynamicsRegressor
from armctrl.planning.scene import (
    Obstacle, PANDA_NOHAND_XML, build_panda_scene, movable_geoms, self_collision_pairs,
)
from armctrl.planning.collision import CollisionChecker
from armctrl.planning.rrt import (
    RRTConnect, RRTConnectConfig, path_length, shortcut_smooth,
)
from armctrl.planning.path_timing import JointPathTrajectory


N_ARM = 7
DT = 0.002
FPS = 50

# 起终点：仅第 1 关节差别很大，直线插值必然扫过正前方的柱子
Q_START = np.array([-0.85, 0.35, 0.0, -1.70, 0.0, 2.05, 0.79])
Q_GOAL = np.array([0.85, 0.35, 0.0, -1.70, 0.0, 2.05, 0.79])

OBSTACLES = [
    Obstacle("pillar", "box", (0.07, 0.07, 0.45), (0.52, 0.0, 0.45),
             rgba=(0.85, 0.35, 0.25, 1.0)),
    Obstacle("beam", "box", (0.30, 0.42, 0.035), (0.52, 0.0, 0.97),
             rgba=(0.60, 0.45, 0.30, 1.0)),
]

# 关节运动学限制（保守值，用于时间参数化）
V_MAX = np.full(N_ARM, 1.2)
A_MAX = np.full(N_ARM, 3.0)
J_MAX = np.full(N_ARM, 30.0)

SAFETY_MARGIN = 0.03        # 机械臂与障碍物之间要求的最小间隙 (m)
SELF_MARGIN = 0.005         # 机械臂自身连杆之间的最小间隙 (m)
RESOLUTION = 0.02           # 段检查的关节空间采样步长 (rad)


def build():
    """场景、机械臂封装、碰撞检测器。

    环境碰撞只查**会动的**连杆：固定底座压在地面上，把它算进去会让所有构型非法。
    自碰撞则用全部连杆几何（底座确实可能被远端连杆撞到）。
    """
    # nohand=True：纯路径规划不关心夹爪，用 7 自由度模型更贴题；
    # 带夹爪时 nq=9、末端 body 叫 hand，两者不能混。
    scene = build_panda_scene(OBSTACLES, with_visuals=True, nohand=True)
    robot = ArmModel(model=scene.model, ee_body="attachment",
                     arm_joints=[f"joint{i}" for i in range(1, N_ARM + 1)])
    env_geoms = movable_geoms(scene.model, scene.robot_geoms, robot.joint_ids)
    checker = CollisionChecker(
        scene.model, robot.qpos_idx, env_geoms, scene.obstacle_geoms,
        # ⚠️ 必须传 joint_ids：不传就会把**两根手指**也当成自碰撞对，
        # 而夹爪闭合时两指本来就贴在一起 ⇒ 任何构型都被判非法，规划必然失败。
        # 手指的相对位姿只由 finger_joint 决定，不在被规划的 7 个关节里。
        self_collision_pairs(scene.model, scene.robot_geoms, robot.joint_ids),
        safety_margin=SAFETY_MARGIN, self_margin=SELF_MARGIN,
        distmax=0.30, resolution=RESOLUTION,
    )
    return scene, robot, checker


def trajectory_clearance(checker: CollisionChecker, Q: np.ndarray):
    """沿一条已采样的轨迹逐点复核间隙，返回 (环境最小间隙, 自碰撞最小间隙)。"""
    env = own = np.inf
    for q in Q:
        e, s = checker.clearances(q)
        env, own = min(env, e), min(own, s)
    return float(env), float(own)


# ------------------------------------------------------------------ 执行


def execute(scene, robot, traj, kp=400.0, kd=40.0, settle=0.5):
    """在 MuJoCo 里用计算力矩控制跟踪轨迹。

    返回 (ts, Q_actual, err_rms_mrad, 接触次数)。
    误差按同一时刻、施加控制之前记录。
    """
    reg = DynamicsRegressor(PANDA_NOHAND_XML)
    m, d = scene.model, robot.data
    n_sub = max(int(round(DT / m.opt.timestep)), 1)
    tau_lim = robot.tau_limit
    obstacle_set = set(scene.obstacle_geoms)
    robot_set = set(scene.robot_geoms)

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = traj(0.0)[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    n_steps = int((traj.duration + settle) / DT)
    ts = np.empty(n_steps)
    Q = np.empty((n_steps, N_ARM))
    err = np.empty((n_steps, N_ARM))
    contacts = 0

    for k in range(n_steps):
        t = k * DT
        q = d.qpos[robot.qpos_idx].copy()
        v = d.qvel[robot.qvel_idx].copy()
        qd, vd, ad = traj(min(t, traj.duration))

        ts[k] = t
        Q[k] = q
        err[k] = qd - q                      # 同刻、控制前

        aq = ad + kd * (vd - v) + kp * (qd - q)
        tau = reg.mass_matrix(q) @ aq + reg.bias(q, v)
        d.ctrl[:N_ARM] = np.clip(tau, -tau_lim, tau_lim)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if ((g1 in robot_set and g2 in obstacle_set)
                    or (g2 in robot_set and g1 in obstacle_set)):
                contacts += 1

    rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1) / N_ARM)) * 1e3)
    return ts, Q, rms, contacts


# ------------------------------------------------------------------ 主流程


def main(seed: int = 0, seeds: int = 6) -> dict:
    scene, robot, checker = build()

    print("\n" + "=" * 86)
    print("有障碍环境下的机械臂路径规划：RRT-Connect + 短路平滑 + S 曲线时间律")
    print("=" * 86)
    p0, _ = robot.fk(Q_START)
    p1, _ = robot.fk(Q_GOAL)
    print(f"障碍物   立柱 半边长 {OBSTACLES[0].size} 位于 {OBSTACLES[0].pos}")
    print(f"         横梁 半边长 {OBSTACLES[1].size} 位于 {OBSTACLES[1].pos}")
    print(f"起点构型 末端位置 {np.round(p0, 3)}  "
          f"环境间隙 {checker.env_clearance(Q_START):.4f} m")
    print(f"终点构型 末端位置 {np.round(p1, 3)}  "
          f"环境间隙 {checker.env_clearance(Q_GOAL):.4f} m")
    print(f"判据     环境裕度 {SAFETY_MARGIN} m，自碰撞裕度 {SELF_MARGIN} m，"
          f"段检查分辨率 {RESOLUTION} rad")

    # ---------------------------------------------------------- 1 反例基线
    print(f"\n{'-'*86}")
    print("第 1 步  反例：不做规划，直接关节空间直线插值")
    naive = JointPathTrajectory([Q_START, Q_GOAL], V_MAX, A_MAX, J_MAX)
    _, Qn, _, _ = naive.sample(0.005)
    env_n, self_n = trajectory_clearance(checker, Qn)
    print(f"  直线插值轨迹时长 {naive.duration:.2f} s，与障碍物的最小间隙 "
          f"{env_n:+.4f} m")
    if env_n < 0:
        print(f"  负号表示已经插进障碍物里 {-env_n*1000:.1f} mm")
    print("  结论：起终点各自合法，不代表它们之间的直线合法——这正是需要规划的原因。")

    # ---------------------------------------------------------- 2 规划
    print(f"\n{'-'*86}")
    print(f"第 2 步  RRT-Connect 规划（{seeds} 个随机种子，检验稳定性）")
    planner = RRTConnect(checker, robot.q_lower, robot.q_upper,
                         RRTConnectConfig(step_size=0.30, max_iters=4000))
    print(f"  {'seed':>5}{'成功':>6}{'迭代':>8}{'节点':>8}{'碰撞检查':>10}"
          f"{'原始路径长':>12}{'平滑后':>10}{'路径点':>10}{'最小间隙':>12}")
    rows = []
    for s in range(seeds):
        res = planner.plan(Q_START, Q_GOAL, seed=s)
        if not res.success:
            print(f"  {s:>5}{'否':>6}   {res.reason}")
            continue
        raw_len = path_length(res.path)
        smooth = shortcut_smooth(res.path, checker, iters=300, seed=s)
        env_c, _ = checker.path_clearances(smooth, samples_per_rad=400.0)
        rows.append((s, res, smooth, raw_len, env_c))
        print(f"  {s:>5}{'是':>6}{res.iterations:>8}{res.nodes:>8}"
              f"{res.collision_checks:>10}{raw_len:>12.3f}"
              f"{path_length(smooth):>10.3f}{len(smooth):>10}{env_c:>12.4f}")
    if not rows:
        raise RuntimeError("所有种子都失败了")

    ratios = [path_length(r[2]) / r[3] for r in rows]
    worst_gap = min(r[4] for r in rows)
    print(f"  平滑把路径长度压到原来的 {min(ratios):.2f}–{max(ratios):.2f} 倍")
    print(f"  最小间隙一列用 400 点/rad 的细采样独立复核，比规划时的 "
          f"{RESOLUTION} rad 段检查密约 {int(400 * RESOLUTION)} 倍。")
    deficit = (SAFETY_MARGIN - worst_gap) * 1e3
    print(f"  六条路径里最差 {worst_gap:.6f} m，与规划裕度 {SAFETY_MARGIN} m 相差 "
          f"{deficit:+.3f} mm。")
    print("  这个差额就是离散采样的代价：规划只保证采样点满足裕度，"
          "采样点之间可以略微下探。间隙仍为正，未发生碰撞，但这是本场景的实测，"
          "不是连续碰撞检测意义上的保证。")
    print(f"  直线插值路径长度 {path_length([Q_START, Q_GOAL]):.3f}"
          f"（撞了，仅作长度下界参考）")

    # 取最短的一条继续
    best = min(rows, key=lambda r: path_length(r[2]))
    seed_used, res, smooth, raw_len, env_c = best
    print(f"  选用 seed={seed_used} 的结果继续")

    # ---------------------------------------------------------- 3 时间参数化
    print(f"\n{'-'*86}")
    print("第 3 步  逐段七段 S 曲线时间参数化")
    traj = JointPathTrajectory(smooth, V_MAX, A_MAX, J_MAX)
    _, Qt, Vt, At = traj.sample(0.002)
    print(f"  路径点 {len(smooth)} 个，分 {len(traj.segments)} 段，"
          f"总时长 {traj.duration:.2f} s")
    print(f"  关节速度峰值 {np.abs(Vt).max():.3f} rad/s  (上限 {V_MAX[0]:.2f})")
    print(f"  关节加速度峰值 {np.abs(At).max():.3f} rad/s² (上限 {A_MAX[0]:.2f})")
    print("  注：每段都是静止到静止，机械臂在路径点处会停一下——"
          "折线拐角处方向跳变，不停就意味着加速度冲激。")
    env_t, self_t = trajectory_clearance(checker, Qt[::5])
    print(f"  时间参数化后的轨迹：环境最小间隙 {env_t:+.4f} m，"
          f"自碰撞最小间隙 {self_t:+.4f} m")

    # ---------------------------------------------------------- 4 执行
    print(f"\n{'-'*86}")
    print("第 4 步  MuJoCo 中用计算力矩控制执行")
    ts, Qa, rms, contacts = execute(scene, robot, traj)
    env_a, self_a = trajectory_clearance(checker, Qa[::5])
    print(f"  跟踪误差 RMS {rms:.3f} mrad（同刻、控制前；控制器与被控对象同参数）")
    print(f"  实际走过的构型：环境最小间隙 {env_a:+.4f} m，"
          f"自碰撞最小间隙 {self_a:+.4f} m")
    print(f"  MuJoCo 实际记录到的 机械臂-障碍物 接触次数：{contacts}")
    q_end = Qa[-1]
    print(f"  终点误差 {np.abs(q_end - Q_GOAL).max()*1e3:.4f} mrad")

    print(f"\n{'-'*86}")
    print("小结")
    print(f"  直线插值：最小间隙 {env_n:+.4f} m（穿进障碍物）")
    print(f"  规划+平滑+执行：最小间隙 {env_a:+.4f} m，接触 {contacts} 次")
    print("  规划解决的是「可行性」问题，不是「精度」问题；")
    print("  精度由后面的时间参数化与控制器负责。\n")

    return {
        "naive_clearance": env_n,
        "planned_clearance": env_a,
        "contacts": contacts,
        "rms_mrad": rms,
        "duration": traj.duration,
        "smooth": smooth,
    }


# ------------------------------------------------------------------ 录像


def record(out: str | None = None, seed: int = 0):
    scene, robot, checker = build()
    planner = RRTConnect(checker, robot.q_lower, robot.q_upper,
                         RRTConnectConfig(step_size=0.30, max_iters=4000))
    res = planner.plan(Q_START, Q_GOAL, seed=seed)
    if not res.success:
        raise RuntimeError(f"规划失败：{res.reason}")
    smooth = shortcut_smooth(res.path, checker, iters=300, seed=seed)
    traj = JointPathTrajectory(smooth, V_MAX, A_MAX, J_MAX)

    reg = DynamicsRegressor(PANDA_NOHAND_XML)
    m, d = scene.model, robot.data
    n_sub = max(int(round(DT / m.opt.timestep)), 1)
    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_START
    mujoco.mj_forward(m, d)

    hold = 1.0                                   # 首尾各停一会儿便于看清
    total = hold + traj.duration + hold
    n_steps = int(total / DT)
    frame_every = max(int(round(1.0 / (FPS * DT))), 1)
    renderer = mujoco.Renderer(m, height=720, width=1280)
    frames = []
    for k in range(n_steps):
        t = float(np.clip(k * DT - hold, 0.0, traj.duration))
        q = d.qpos[robot.qpos_idx].copy()
        v = d.qvel[robot.qvel_idx].copy()
        qd, vd, ad = traj(t)
        aq = ad + 40.0 * (vd - v) + 400.0 * (qd - q)
        tau = reg.mass_matrix(q) @ aq + reg.bias(q, v)
        d.ctrl[:N_ARM] = np.clip(tau, -robot.tau_limit, robot.tau_limit)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        if k % frame_every == 0:
            renderer.update_scene(d, camera="planning")
            frames.append(renderer.render())
    renderer.close()

    if out is None:
        out = video_path("planning")
    written = write_video(frames, out, fps=FPS)
    print(f"[video] {len(frames)} 帧 @ {FPS} fps")
    for f in written:
        print(f"        -> {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="渲染 mp4 到 media/videos/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.video:
        record(a.out, a.seed)
    else:
        main()
