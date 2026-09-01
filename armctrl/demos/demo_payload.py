"""未知负载突变实验：什么时候固定增益控制器会失效，自适应控制的价值在哪。

前一个基准（demo_benchmark.py）里 PD 表现很好，因为轨迹慢、重力补偿准确。
这会误导人以为"自适应控制没用"。本实验制造一个真实工程中最常见的情形：

    t = 8 s 时末端突然挂上一个 2 kg 的未知负载（控制器不知道）

此时：
    PD          没有积分作用，负载重力变成常值扰动 → 出现稳态误差
    CTC(旧模型) 模型里没有这个负载 → 前馈补偿不足，同样有稳态误差
    MBAC        负载改变了末端连杆的等效质量，属于参数变化 → 在线重新估计并补偿
    RBF-NN      把负载当作未知动力学的一部分 → 网络重新逼近

这才是自适应控制真正的用武之地：不是"精度更高"，而是"模型变了还能自己修正"。
"""

from __future__ import annotations

import numpy as np
import mujoco
import pinocchio as pin

from armctrl.identification.regressor import DynamicsRegressor
from armctrl.control.adaptive import (
    SlotineLiAdaptiveController,
    RBFAdaptiveController,
    fit_centers_to_trajectory,
)
from armctrl.demos.demo_benchmark import (
    PANDA_XML,
    Q_HOME,
    N_ARM,
    DT,
    make_arm,
    traj_full,
)

T_END = 20.0
T_PAYLOAD = 8.0
PAYLOAD_MASS = 2.0            # kg，控制器完全不知道


def simulate_payload(controller_fn, label: str, q_offset: np.ndarray):
    robot = make_arm()
    m, d = robot.model, robot.data
    n_sub = max(int(round(DT / m.opt.timestep)), 1)

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_HOME + q_offset
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    errs, ts = [], []
    for k in range(int(T_END / DT)):
        t = k * DT
        qd_f, vd_f, ad_f = traj_full(t, m.nv)
        q_f = d.qpos[: m.nv].copy()
        v_f = d.qvel[: m.nv].copy()

        tau = robot.saturate_torque(controller_fn(q_f, v_f, qd_f, vd_f, ad_f, DT)[:N_ARM])
        d.ctrl[:N_ARM] = tau

        # 未知负载：以末端受到的重力形式施加
        if t >= T_PAYLOAD:
            d.xfrc_applied[robot.ee_body_id, :3] = [0.0, 0.0, -PAYLOAD_MASS * 9.81]

        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        errs.append(qd_f[:N_ARM] - d.qpos[robot.qpos_idx])
        ts.append(t)

    errs, ts = np.array(errs), np.array(ts)

    def rmse(mask):
        return float(np.sqrt(np.mean(np.linalg.norm(errs[mask], axis=1) ** 2)))

    return {
        "label": label,
        "before": rmse((ts >= 5.0) & (ts < 8.0)),        # 挂载前稳态
        "shock": rmse((ts >= 8.0) & (ts < 10.0)),        # 挂载瞬间冲击
        "after": rmse(ts >= 15.0),                        # 挂载后重新稳定
    }


def main():
    reg = DynamicsRegressor(PANDA_XML)
    nv = reg.nv
    q_offset = np.array([0.10, -0.12, 0.08, 0.10, -0.15, 0.10, -0.20])
    lam, kd = 12.0, 70.0
    kp_ctc, kd_ctc = 400.0, 40.0

    print("\n" + "=" * 78)
    print(f"未知负载突变实验  |  t = {T_PAYLOAD:.0f} s 时末端挂上 {PAYLOAD_MASS:.0f} kg 未知负载")
    print("=" * 78)

    results = []

    kp_pd, kd_pd = 250.0, 30.0
    results.append(
        simulate_payload(
            lambda q, v, qd, vd, ad, dt: kp_pd * (qd - q) + kd_pd * (vd - v) + reg.gravity(q),
            "PD + 重力补偿（固定增益）",
            q_offset,
        )
    )

    def ctc_ctrl(q, v, qd, vd, ad, dt):
        aq = ad + kd_ctc * (vd - v) + kp_ctc * (qd - q)
        return reg.mass_matrix(q) @ aq + reg.bias(q, v)

    results.append(simulate_payload(ctc_ctrl, "CTC（模型不含负载）", q_offset))

    mbac = SlotineLiAdaptiveController(reg, N_ARM, lam=lam, kd=kd, gamma=0.5, use_base=True)
    results.append(
        simulate_payload(
            lambda q, v, qd, vd, ad, dt: mbac.compute(q, v, qd, vd, ad, dt)[0],
            "MBAC（Slotine-Li 自适应）",
            q_offset,
        )
    )

    C, W = fit_centers_to_trajectory(lambda t: traj_full(t, nv), T_END, n_centers=240, lam=lam)
    rbf = RBFAdaptiveController(N_ARM, C, W, lam=lam, kd=kd, gamma=200.0, sigma=0.02)
    results.append(
        simulate_payload(
            lambda q, v, qd, vd, ad, dt: rbf.compute(q, v, qd, vd, ad, dt)[0],
            "RBF-NN 自适应",
            q_offset,
        )
    )

    print("\n" + "-" * 78)
    print(f"{'控制器':<28}{'挂载前 5–8s':>14}{'冲击 8–10s':>14}{'恢复后 15–20s':>15}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['label']:<28}{r['before']*1000:>11.2f} mrad"
            f"{r['shock']*1000:>11.2f} mrad{r['after']*1000:>12.2f} mrad"
        )
    print("-" * 78)

    print("\n负载引起的稳态精度损失（恢复后 ÷ 挂载前）：")
    for r in results:
        print(f"  {r['label']:<28}{r['after'] / max(r['before'], 1e-12):>8.1f} ×")

    print("\n  固定增益控制器无法消除未知负载带来的稳态误差；")
    print("  自适应控制器把负载视为参数变化并在线重新估计，误差回落到接近原水平。\n")


if __name__ == "__main__":
    main()
