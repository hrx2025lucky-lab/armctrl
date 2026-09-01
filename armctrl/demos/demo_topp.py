"""三种时间参数化在同一条 RRT 路径上的对照。

要回答的问题
------------
规划器给出的是**几何路径**（走哪儿）。把它变成可执行轨迹（什么时候到哪）
有不同做法，各自的代价是什么？

    A  逐段 S 曲线            jerk 有界，但每个路径点都要停
    B  折线直接上 TOPP-RA     不停，但拐角处曲率极大，速度被迫压到很低
    C  圆弧倒角 + TOPP-RA     不停，且拐角可以带速通过 —— MoveIt 的做法

流程与安全性
------------
倒角会**改变几何路径**（圆弧总是往转弯内侧切进去，而障碍常常就在内侧），
所以 C 方案必须重新做碰撞检测，不能沿用规划阶段的结论。
本 demo 对每种方案都用比规划分辨率密得多的采样独立复核间隙，
并在 MuJoCo 中实际执行、统计真实接触次数。

用法
----
    export PYTHONPATH=.
    python armctrl/demos/demo_topp.py
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel
from armctrl.identification.regressor import DynamicsRegressor
from armctrl.planning.scene import (
    Obstacle, PANDA_XML, build_panda_scene, movable_geoms, self_collision_pairs,
)
from armctrl.planning.collision import CollisionChecker
from armctrl.planning.rrt import (
    RRTConnect, RRTConnectConfig, path_length, shortcut_smooth,
)
from armctrl.planning.path_timing import JointPathTrajectory
from armctrl.planning.blend import blend_corners, blend_and_verify
from armctrl.planning.topp import ToppraTrajectory, analyze, TimingReport


N_ARM = 7
DT = 0.002
Q_START = np.array([-0.85, 0.35, 0.0, -1.70, 0.0, 2.05, 0.79])
Q_GOAL = np.array([0.85, 0.35, 0.0, -1.70, 0.0, 2.05, 0.79])
OBSTACLES = [
    Obstacle("pillar", "box", (0.07, 0.07, 0.45), (0.52, 0.0, 0.45)),
    Obstacle("beam", "box", (0.30, 0.42, 0.035), (0.52, 0.0, 0.97)),
]
V_MAX = np.full(N_ARM, 1.2)
A_MAX = np.full(N_ARM, 3.0)
J_MAX = np.full(N_ARM, 30.0)
# 规划裕度必须比执行裕度大：倒角总是往转弯内侧切进去，
# 若规划时就贴着执行裕度走，倒角一动就越界，圆弧会被全部拒绝。
# 差额 (PLAN_MARGIN − EXEC_MARGIN) 就是留给倒角的预算。
PLAN_MARGIN = 0.075
EXEC_MARGIN = 0.030
MAX_DEVIATION = 0.06


def build():
    """返回场景、机械臂，以及两个裕度不同的碰撞检测器。

    checker_plan  规划与平滑用，裕度大，给后续倒角留余量
    checker_exec  最终验收用，裕度是真正要求的下限
    """
    scene = build_panda_scene(OBSTACLES, with_visuals=True)
    robot = ArmModel(model=scene.model, ee_body="attachment",
                     arm_joints=[f"joint{i}" for i in range(1, N_ARM + 1)])
    env = movable_geoms(scene.model, scene.robot_geoms, robot.joint_ids)
    pairs = self_collision_pairs(scene.model, scene.robot_geoms)

    def mk(margin):
        return CollisionChecker(
            scene.model, robot.qpos_idx, env, scene.obstacle_geoms, pairs,
            safety_margin=margin, self_margin=0.005,
            distmax=0.30, resolution=0.02)

    return scene, robot, mk(PLAN_MARGIN), mk(EXEC_MARGIN)


def traj_clearance(checker, traj, dt=0.01) -> float:
    """沿时间轨迹独立复核最小间隙（只看与环境障碍物的间隙）。"""
    _, Q, _, _ = traj.sample(dt)
    return float(min(checker.clearances(q)[0] for q in Q))


def execute(scene, robot, traj, kp=400.0, kd=40.0, settle=0.4):
    """MuJoCo 中用计算力矩控制执行，返回 (跟踪 RMS mrad, 接触次数)。"""
    reg = DynamicsRegressor(PANDA_XML)
    m, d = scene.model, robot.data
    n_sub = max(int(round(DT / m.opt.timestep)), 1)
    obstacles, robots = set(scene.obstacle_geoms), set(scene.robot_geoms)

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = traj(0.0)[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    err2, n, contacts = 0.0, 0, 0
    for k in range(int((traj.duration + settle) / DT)):
        t = k * DT
        q = d.qpos[robot.qpos_idx].copy()
        v = d.qvel[robot.qvel_idx].copy()
        qd, vd, ad = traj(min(t, traj.duration))
        err2 += float(np.sum((qd - q) ** 2))       # 同刻、控制前
        n += N_ARM
        aq = ad + kd * (vd - v) + kp * (qd - q)
        tau = reg.mass_matrix(q) @ aq + reg.bias(q, v)
        d.ctrl[:N_ARM] = np.clip(tau, -robot.tau_limit, robot.tau_limit)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if ((g1 in robots and g2 in obstacles) or (g2 in robots and g1 in obstacles)):
                contacts += 1
    return float(np.sqrt(err2 / n) * 1e3), contacts


def main(seed: int = 0):
    scene, robot, checker, checker_exec = build()
    print("\n" + "=" * 104)
    print("时间参数化对照：逐段 S 曲线  vs  TOPP-RA（折线 / 圆弧倒角）")
    print("=" * 104)

    planner = RRTConnect(checker, robot.q_lower, robot.q_upper,
                         RRTConnectConfig(step_size=0.30, max_iters=6000))
    res = planner.plan(Q_START, Q_GOAL, seed=seed)
    if not res.success:
        raise RuntimeError(res.reason)
    smooth = shortcut_smooth(res.path, checker, iters=300, seed=seed)
    print(f"\n裕度设定：规划 {PLAN_MARGIN*1e3:.0f} mm，执行验收 {EXEC_MARGIN*1e3:.0f} mm，"
          f"差额 {(PLAN_MARGIN-EXEC_MARGIN)*1e3:.0f} mm 留给倒角")
    print(f"RRT-Connect（seed={seed}）：{res.iterations} 次迭代，"
          f"原始路径长 {path_length(res.path):.3f} rad")
    print(f"短路平滑后：{len(smooth)} 个路径点，长度 {path_length(smooth):.3f} rad")

    # ---------------------------------------------------------------- A
    a_traj = JointPathTrajectory(smooth, V_MAX, A_MAX, J_MAX)
    a_rep = analyze(a_traj, "A 逐段 S 曲线", path_length(smooth),
                    clearance=traj_clearance(checker_exec, a_traj))

    # ---------------------------------------------------------------- B
    raw = blend_corners(smooth, max_deviation=0.0)
    b_traj = ToppraTrajectory(raw, V_MAX, A_MAX)
    b_rep = analyze(b_traj, "B 折线 + TOPP-RA", raw.length,
                    clearance=traj_clearance(checker_exec, b_traj))

    # ---------------------------------------------------------------- C
    # 倒角只需满足执行裕度；规划时多留的那部分正是给它用的
    blended, dev_used, blend_clear = blend_and_verify(
        smooth, checker_exec, max_deviation=MAX_DEVIATION)
    c_traj = ToppraTrajectory(blended, V_MAX, A_MAX)
    c_rep = analyze(c_traj, "C 倒角 + TOPP-RA", blended.length,
                    deviation=max((c["deviation"] for c in blended.corners), default=0.0),
                    clearance=traj_clearance(checker_exec, c_traj))

    print(f"\n倒角：预算 {MAX_DEVIATION} rad，实际采用 {dev_used:.4f} rad，"
          f"{blended.n_blended}/{len(blended.corners)} 个拐角完成倒角，"
          f"倒角后路径复核最小间隙 {blend_clear*1e3:.2f} mm")
    print(f"TOPP-RA 降额系数：B={b_traj.scale:.4f}（{b_traj.attempts} 次迭代）  "
          f"C={c_traj.scale:.4f}（{c_traj.attempts} 次迭代）")
    print("  降额是因为 TOPP-RA 只在离散网格上施加约束，直接求解出来的轨迹")
    print("  实测峰值会超限几个百分点；这里按实测反推降额直到真的达标。")

    print(f"\n{'-'*104}")
    print(TimingReport.header())
    for r in (a_rep, b_rep, c_rep):
        print(r.row())

    print(f"\n{'-'*104}")
    print("MuJoCo 中实际执行")
    print(f"{'方案':<22}{'跟踪 RMS/mrad':>16}{'机械臂-障碍物接触':>20}")
    for name, tr in (("A 逐段 S 曲线", a_traj), ("B 折线 + TOPP-RA", b_traj),
                     ("C 倒角 + TOPP-RA", c_traj)):
        rms, contacts = execute(scene, robot, tr)
        print(f"{name:<22}{rms:>16.3f}{contacts:>20}")

    print(f"\n{'-'*104}")
    print("怎么读这张表")
    print(f"  · A 的 jerk 有界（≤{J_MAX[0]:.0f}），代价是在 {a_rep.stops} 个路径点各停一次，"
          f"总时长 {a_rep.duration:.2f} s。")
    print(f"  · B 不停，但折线拐角处曲率极大，TOPP-RA 只能把速度压到很低才不越加速度限，"
          f"所以并不快（{b_rep.duration:.2f} s）。")
    print(f"  · C 把拐角倒成圆弧后才真正快起来（{c_rep.duration:.2f} s，"
          f"比 A 快 {(1-c_rep.duration/a_rep.duration)*100:.1f}%），"
          f"代价是路径偏离最多 {c_rep.deviation*1e3:.1f} mm，且 jerk 无界。")
    print("  · 结论不是「谁更好」，而是「约束是什么」：要抑制结构模态就选 A，"
          "要节拍就选 C，且必须重新做碰撞检测。\n")


if __name__ == "__main__":
    main()
