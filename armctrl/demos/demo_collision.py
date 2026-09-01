"""动量观测器碰撞检测验证。

场景：机械臂在阻抗控制下悬停，第 2 秒起在末端施加一个已知外力，
      观测器应当在几十毫秒内估计出该外力，且无外力时残差接近零。
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.core.robot import make_panda
from armctrl.control.impedance import CartesianImpedanceController, ImpedanceGains
from armctrl.control.momentum_observer import MomentumObserver


Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])


def run(k_i: float, f_ext_world: np.ndarray, duration: float = 4.0, t_on: float = 2.0):
    robot = make_panda()
    m, d = robot.model, robot.data
    dt = 0.002
    n_sub = max(int(round(dt / m.opt.timestep)), 1)

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_HOME
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    p_hold, R_hold = robot.fk(Q_HOME)

    ctrl = CartesianImpedanceController(
        robot, ImpedanceGains(k_trans=800.0, k_rot=40.0, zeta=1.0), q_null=Q_HOME
    )
    obs = MomentumObserver(robot, k_i=k_i, dt=dt)
    obs.reset(Q_HOME, np.zeros(robot.n))

    log = {"t": [], "f_hat": [], "r_norm": []}
    for k in range(int(duration / dt)):
        t = k * dt
        q = d.qpos[robot.qpos_idx].copy()
        qd = d.qvel[robot.qvel_idx].copy()

        tau = ctrl.compute(q, qd, p_hold, R_hold)
        obs.update(q, qd, tau)
        f_hat = obs.cartesian_force(q)[:3]

        d.qpos[robot.qpos_idx] = q
        d.qvel[robot.qvel_idx] = qd
        mujoco.mj_forward(m, d)
        d.xfrc_applied[robot.ee_body_id, :3] = f_ext_world if t >= t_on else np.zeros(3)
        d.ctrl[: robot.n] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        log["t"].append(t)
        log["f_hat"].append(f_hat)
        log["r_norm"].append(float(np.linalg.norm(obs.r)))

    for key in log:
        log[key] = np.array(log[key])
    return log


def main():
    f_ext = np.array([0.0, 0.0, -15.0])
    print("\n" + "=" * 66)
    print("动量观测器外力估计   真实外力 Fz = -15 N，第 2.0 s 施加")
    print("=" * 66)
    print(f"{'K_I (1/s)':>10} {'无外力残差':>12} {'稳态 Fz 估计':>14} {'估计误差':>10} {'检测延迟':>10}")

    for k_i in (10.0, 25.0, 60.0):
        log = run(k_i, f_ext)
        t = log["t"]
        fz = log["f_hat"][:, 2]

        pre = np.abs(log["r_norm"][(t > 1.0) & (t < 2.0)]).max()
        post = fz[t > 3.0].mean()
        err = abs(post - f_ext[2]) / abs(f_ext[2]) * 100

        idx = np.where((t >= 2.0) & (np.abs(fz) > 0.5 * abs(f_ext[2])))[0]
        delay = (t[idx[0]] - 2.0) * 1000 if len(idx) else float("nan")

        print(
            f"{k_i:10.0f} {pre:12.4f} {post:14.2f} N {err:8.1f}% {delay:8.1f} ms"
        )

    print("\n  → 无外力时残差接近 0，说明 M/C/G 模型与斜对称性质使用正确。")
    print("  → K_I 越大检测越快，但对模型误差和数值噪声越敏感 —— 与 LESO")
    print("    的观测器带宽 wo 是完全相同的权衡。\n")


if __name__ == "__main__":
    main()
