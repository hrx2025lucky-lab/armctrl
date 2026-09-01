"""MPC 关键实验：力矩约束绑定时，前馈限幅 vs 显式约束优化。

实验设计
--------
Franka Panda 腕部三个关节的额定力矩只有 12 N·m，比前四个关节（87 N·m）小得多。
设计一条**故意超出腕部力矩能力**的激进轨迹，制造约束绑定的工况。

对比三个控制器（增益按各自结构合理设定，均使用同一套精确模型）：

    CTC        计算力矩控制。约束只能在输出端硬限幅（clip）。
               饱和期间实际处于开环，闭环极点被破坏。
    CTC+饱和补偿  在 CTC 基础上做一个简单的抗饱和：饱和时冻结误差累积。
               这是工业上常见的补救手段，用来说明"事后补救"的能力上限。
    MPC        把 |M(q)a + h| <= tau_max 作为线性不等式写进 QP，
               预测未来 N 步后只执行第一步。

指标
----
    跟踪 RMSE              越小越好
    最大跟踪误差            峰值偏离
    饱和样本占比            控制量撞上限的时间比例
    限幅前后力矩差的 L1 范数  衡量"被硬砍掉了多少控制量"

预期（也是 MPC 的核心价值）：
CTC 在饱和期间会被硬砍，产生额外超调；MPC 会提前减速，虽然名义跟踪指令
完不成，但**始终工作在可行域内**，误差峰值和恢复速度更好。
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel
from armctrl.identification.regressor import DynamicsRegressor
from armctrl.control.mpc import JointSpaceMPC
from armctrl.assets import panda_nohand_xml

PANDA_XML = panda_nohand_xml()
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
DT = 0.002
MPC_DT = 0.01
N_ARM = 7
T_END = 8.0


def make_arm() -> ArmModel:
    return ArmModel(
        xml_path=PANDA_XML,
        ee_body="attachment",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
    )


def aggressive_traj(t: float, scale: float):
    """基座温和、腕部激进的正弦轨迹。

    Panda 腕部三关节额定力矩仅 12 N·m（基座四关节是 87 N·m），且惯量小、
    可以做高频运动，因此把幅值和频率集中加在腕部，可以在基座远未饱和时
    单独让腕部撞上力矩上限，使"约束绑定"成为唯一变量。

    scale = 1.0 时腕部所需峰值力矩约为额定值的 1.34 倍（必然饱和），
    基座峰值约 29 N·m（远低于 87），且所有关节角都在行程内。
    """
    wa = 1.0 * scale
    ww = 9.0 if scale >= 0.999 else 9.0 * np.sqrt(scale)
    amp = np.array([0.25, 0.15, 0.20, 0.15, wa, wa * 0.8, wa * 1.1])
    w = np.array([0.6, 0.45, 0.7, 0.5, ww, ww * 0.9, ww * 1.1])
    q = Q_HOME + amp * np.sin(w * t)
    v = amp * w * np.cos(w * t)
    a = -amp * w**2 * np.sin(w * t)
    return q, v, a


def run(controller: str, scale: float, mpc_kw: dict | None = None):
    robot = make_arm()
    reg = DynamicsRegressor(PANDA_XML)
    m, d = robot.model, robot.data
    n_sub = max(int(round(DT / m.opt.timestep)), 1)
    tau_lim = robot.tau_limit

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_HOME
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    mpc = None
    if controller == "mpc":
        # 权重按**等效闭环带宽**标定，使 MPC 与 CTC 可公平对比。
        # 实测 MPC 的隐含反馈增益（单位惯量、无约束激活的小误差区）：
        #   w_pos=6e3, w_vel=60,  w_acc=2e-2, w_rate=3e-1  ->  kp=42.4,  ωn=6.5  rad/s
        #   w_pos=5e4, w_vel=300, w_acc=1e-4, w_rate=1e-3  ->  kp=841,   ωn=29.0 rad/s
        # CTC 用 kp=900, kd=60 -> ωn=30.0 rad/s。
        # 最初用第一组权重时 MPC 带宽仅 6.5 rad/s，低于轨迹频率 9 rad/s，
        # 根本无法跟踪，导致对比结论完全无效。这是标定错误，不是 MPC 的性质。
        kw = dict(horizon=8, dt=MPC_DT, w_pos=5e4, w_vel=300.0,
                  w_acc=1e-4, w_rate=1e-3)
        kw.update(mpc_kw or {})
        mpc = JointSpaceMPC(
            N_ARM, tau_lim, robot.q_lower, robot.q_upper,
            v_limit=np.full(N_ARM, 4.0), **kw,
        )

    kp, kd = 900.0, 60.0
    errs, sat_flags, clip_l1, ts = [], [], [], []
    int_freeze = np.zeros(N_ARM)
    mpc_every = max(int(round(MPC_DT / DT)), 1)
    tau_hold = np.zeros(N_ARM)

    for k in range(int(T_END / DT)):
        t = k * DT
        q = d.qpos[robot.qpos_idx].copy()
        v = d.qvel[robot.qvel_idx].copy()
        qd_, vd_, ad_ = aggressive_traj(t, scale)

        if controller == "mpc":
            if k % mpc_every == 0:
                Nh = mpc.N
                qs, vs, as_ = [], [], []
                for i in range(1, Nh + 1):
                    qq, vv, aa = aggressive_traj(t + i * MPC_DT, scale)
                    qs.append(qq); vs.append(vv); as_.append(aa)
                M = reg.mass_matrix(q)
                h = reg.bias(q, v)
                tau_raw, _, _ = mpc.compute(
                    q, v, np.array(qs), np.array(vs), np.array(as_), M, h
                )
                tau_hold = tau_raw
            tau_raw = tau_hold
        else:
            aq = ad_ + kd * (vd_ - v) + kp * (qd_ - q)
            if controller == "ctc_aw":
                # 抗饱和：上一步饱和的关节，本步降低比例项贡献
                aq = aq - kp * int_freeze
            tau_raw = reg.mass_matrix(q) @ aq + reg.bias(q, v)

        tau = np.clip(tau_raw, -tau_lim, tau_lim)
        sat = np.abs(tau_raw) > tau_lim * 1.0001
        if controller == "ctc_aw":
            int_freeze = np.where(sat, (qd_ - q) * 0.5, 0.0)

        d.ctrl[:N_ARM] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        errs.append(qd_ - d.qpos[robot.qpos_idx])
        sat_flags.append(sat)
        clip_l1.append(np.abs(tau_raw - tau).sum())
        ts.append(t)

    errs = np.array(errs)
    sat_flags = np.array(sat_flags)
    ts = np.array(ts)
    mask = ts > 1.0                       # 跳过初始瞬态

    return {
        "rmse": float(np.sqrt(np.mean(np.linalg.norm(errs[mask], axis=1) ** 2))),
        "emax": float(np.linalg.norm(errs[mask], axis=1).max()),
        "sat_frac": float(sat_flags[mask].any(axis=1).mean()),
        "sat_joint_frac": float(sat_flags[mask].mean()),
        "clip_l1": float(np.mean(np.array(clip_l1)[mask])),
        "fail": 0 if mpc is None else mpc.solve_failures,
    }


def main():
    print("\n" + "=" * 84)
    print("MPC 关键实验：力矩约束绑定时的表现")
    print(f"Franka Panda 7-DOF | 腕部力矩上限仅 12 N·m | 轨迹时长 {T_END:.0f} s")
    print("=" * 84)

    labels = {
        "ctc": "CTC（输出端硬限幅）",
        "ctc_aw": "CTC + 简单抗饱和",
        "mpc": "MPC（约束写进 QP）",
    }

    for scale, tag in [(0.25, "温和：力矩需求远低于上限，约束不绑定"),
                       (1.00, "激进：腕部力矩需求达额定值 1.34 倍，约束绑定")]:
        print(f"\n{'-'*84}")
        print(f"轨迹强度 scale = {scale:.2f}   （{tag}）")
        print(f"{'-'*84}")
        print(f"{'控制器':<24}{'RMSE':>12}{'最大误差':>12}{'饱和时间占比':>14}"
              f"{'被砍掉力矩':>14}")
        rows = {}
        for c in ("ctc", "ctc_aw", "mpc"):
            r = run(c, scale)
            rows[c] = r
            print(f"{labels[c]:<24}{r['rmse']*1000:>9.2f} mrad"
                  f"{r['emax']*1000:>9.2f} mrad"
                  f"{r['sat_frac']*100:>12.1f} %"
                  f"{r['clip_l1']:>11.3f} N·m")
        if rows["mpc"]["fail"]:
            print(f"  ! MPC 求解失败 {rows['mpc']['fail']} 次")
        base = rows["ctc"]
        print(f"\n  MPC 相对 CTC：RMSE {base['rmse']/rows['mpc']['rmse']:.2f} ×，"
              f"最大误差 {base['emax']/rows['mpc']['emax']:.2f} ×")

    print(f"\n{'-'*84}")
    print("说明：CTC 在饱和期间被硬砍掉的控制量无处补偿，闭环极点被破坏；")
    print("      MPC 把力矩上限作为不等式约束，求解结果本身就在可行域内，")
    print("      因此'被砍掉力矩'一列应接近 0。")
    print("      约束不绑定时两者应当接近——MPC 的价值只在约束起作用时体现。\n")


if __name__ == "__main__":
    main()
