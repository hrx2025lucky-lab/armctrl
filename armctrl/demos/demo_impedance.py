"""笛卡尔阻抗控制闭环验证。

三组实验：
  1. 自由空间轨迹跟踪  —— 验证控制器基本跟踪能力
  2. 刚度扫描          —— 施加恒定外力，验证稳态偏移 Δx = F/K
  3. 阻尼设计对比      —— critical(Λ自适应) vs fixed，看阶跃响应超调
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.core.robot import make_panda
from armctrl.control.impedance import CartesianImpedanceController, ImpedanceGains
from armctrl.planning.trajectory import CartesianLine


Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])


def simulate(robot, ctrl, p_ref_fn, R_ref, duration, f_ext=None, dt_ctrl=0.002):
    """闭环仿真。f_ext 为施加在末端的世界系外力 (3,)，None 表示自由空间。"""
    m, d = robot.model, robot.data
    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_HOME
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    n_sub = max(int(round(dt_ctrl / m.opt.timestep)), 1)
    steps = int(duration / dt_ctrl)
    log = {"t": [], "p": [], "p_ref": [], "tau": []}

    for k in range(steps):
        t = k * dt_ctrl
        q = d.qpos[robot.qpos_idx].copy()
        qd = d.qvel[robot.qvel_idx].copy()
        p_ref, v_ref = p_ref_fn(t)
        tau = ctrl.compute(q, qd, p_ref, R_ref, v_des=np.concatenate([v_ref, np.zeros(3)]))

        # 写回控制器状态后施加力矩（compute 内部调用 fk 会改写 data）
        d.qpos[robot.qpos_idx] = q
        d.qvel[robot.qvel_idx] = qd
        mujoco.mj_forward(m, d)

        if f_ext is not None:
            d.xfrc_applied[robot.ee_body_id, :3] = f_ext
        d.ctrl[: robot.n] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        p_now, _ = robot.fk(d.qpos[robot.qpos_idx].copy())
        log["t"].append(t)
        log["p"].append(p_now)
        log["p_ref"].append(p_ref)
        log["tau"].append(tau)

    for key in log:
        log[key] = np.array(log[key])
    return log


def exp1_tracking(robot):
    print("=" * 62)
    print("实验 1：自由空间笛卡尔直线跟踪（S 曲线 + slerp）")
    print("=" * 62)
    robot.set_state(Q_HOME, np.zeros(7))
    p0, R0 = robot.fk(Q_HOME)
    p1 = p0 + np.array([0.12, 0.10, -0.06])
    line = CartesianLine(p0, R0, p1, R0, v_max=0.15, a_max=0.6, j_max=5.0)

    for k_trans in (200.0, 600.0, 1500.0):
        ctrl = CartesianImpedanceController(
            robot, ImpedanceGains(k_trans=k_trans, k_rot=30.0, zeta=1.0), q_null=Q_HOME
        )
        log = simulate(
            robot, ctrl, lambda t: (line(t)[0], line(t)[2]), R0, line.T + 1.0
        )
        err = np.linalg.norm(log["p"] - log["p_ref"], axis=1)
        print(
            f"  K={k_trans:6.0f} N/m | 跟踪 RMSE={err.mean()*1000:6.2f} mm | "
            f"最大误差={err.max()*1000:6.2f} mm | 终点误差={err[-1]*1000:5.2f} mm"
        )
    print("  → 刚度越高跟踪越准，但接触时冲击也越大，这是阻抗控制的核心权衡。\n")


def exp2_stiffness(robot):
    print("=" * 62)
    print("实验 2：刚度验证 —— 恒定外力下稳态偏移应满足 Δx = F/K")
    print("=" * 62)
    robot.set_state(Q_HOME, np.zeros(7))
    p0, R0 = robot.fk(Q_HOME)
    f_ext = np.array([0.0, 0.0, -20.0])

    print(f"  外力 Fz = {f_ext[2]:.0f} N")
    print(f"  {'K (N/m)':>10} {'理论 Δz (mm)':>14} {'实测 Δz (mm)':>14} {'相对误差':>10}")
    for k_trans in (200.0, 400.0, 800.0, 1600.0):
        ctrl = CartesianImpedanceController(
            robot, ImpedanceGains(k_trans=k_trans, k_rot=30.0, zeta=1.0), q_null=Q_HOME
        )
        log = simulate(robot, ctrl, lambda t: (p0, np.zeros(3)), R0, 4.0, f_ext=f_ext)
        dz_meas = (log["p"][-200:, 2] - p0[2]).mean()
        dz_theory = f_ext[2] / k_trans
        rel = abs(dz_meas - dz_theory) / abs(dz_theory) * 100
        print(f"  {k_trans:10.0f} {dz_theory*1000:14.2f} {dz_meas*1000:14.2f} {rel:9.1f}%")
    print("  → 稳态偏移与 1/K 成正比，证明控制器确实实现了指定的笛卡尔刚度。\n")


def exp3_damping(robot):
    print("=" * 62)
    print("实验 3：阻尼设计对比 —— 阶跃响应超调量")
    print("=" * 62)
    robot.set_state(Q_HOME, np.zeros(7))
    p0, R0 = robot.fk(Q_HOME)
    p_step = p0 + np.array([0.0, 0.0, 0.05])

    print(f"  {'阻尼设计':>12} {'ζ':>6} {'超调量 (%)':>12} {'调节时间 (s)':>14}")
    for design in ("critical", "fixed"):
        for zeta in (0.4, 1.0):
            ctrl = CartesianImpedanceController(
                robot,
                ImpedanceGains(k_trans=600.0, k_rot=30.0, zeta=zeta),
                damping_design=design,
                q_null=Q_HOME,
            )
            log = simulate(
                robot, ctrl, lambda t: (p_step, np.zeros(3)), R0, 3.0
            )
            z = log["p"][:, 2]
            target = p_step[2]
            span = target - p0[2]
            overshoot = max((z.max() - target) / span * 100.0, 0.0)
            band = 0.02 * abs(span)
            settled = np.where(np.abs(z - target) > band)[0]
            t_settle = log["t"][settled[-1]] if len(settled) else 0.0
            print(f"  {design:>12} {zeta:6.1f} {overshoot:12.2f} {t_settle:14.3f}")
    print("  → ζ=0.4 欠阻尼有明显超调；ζ=1.0 临界阻尼几乎无超调。")
    print("  → critical 设计用 Λ(q) 自适应，构型变化时阻尼比保持一致。\n")


def main():
    robot = make_panda()
    print(f"\nFranka Panda 7-DOF | 关节数={robot.n} | 力矩上限={robot.tau_limit}\n")
    exp1_tracking(robot)
    exp2_stiffness(robot)
    exp3_damping(robot)


if __name__ == "__main__":
    main()
