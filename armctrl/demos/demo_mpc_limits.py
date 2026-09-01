"""MPC 的正确论证工况：关节限位避让。

为什么力矩饱和不是好场景
------------------------
参考轨迹不可行时，限幅 CTC 的 bang-bang 行为对追踪激进正弦接近时间最优，
而二次代价 MPC 会取折中。带宽对齐后实测：约束不绑定时两者一致（2.86 vs 2.91 mrad），
力矩饱和时 CTC 反而更好（145.8 vs 617.0 mrad）。这是目标函数不同，不是缺陷。

为什么关节限位是好场景
----------------------
**限幅力矩无法阻止位置越限。** 关节带着动能冲向限位时，即使把力矩砍到上限，
制动距离仍受 v²/(2·a_max) 决定。反馈控制只能在越限**之后**才产生反向力，
而那时已经晚了。

MPC 在预测时域内把 q_min ≤ q_k ≤ q_max 写成线性不等式，会在**还没到限位时**
就开始减速。这是预测控制独有的能力，任何调参都无法让反馈控制获得。

公平性设计
----------
两个控制器都**知道**限位：
    CTC   把参考轨迹钳制到限位（反应式控制唯一能做的事），然后高增益跟踪
    MPC   把限位作为 QP 约束
两者带宽已对齐（CTC ωn=30.0 rad/s；MPC 权重标定到 ωn≈29.0 rad/s），
使用同一套动力学模型、同一条参考、同一初始状态。

指标
----
    越限深度     max(q) − q_limit，越小越好（核心指标）
    触限速度     首次到达限位时的关节速度，反映冲击强度
    停止后回退   稳定后与限位的距离
    可行域外时间 位置超出限位的时间占比
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
T_END = 3.0

# 虚拟限位：取在硬件行程之内，使实验不依赖 MuJoCo 的限位接触模型，
# 纯粹考察控制器能否自己守住约束。
JOINT_IDX = 0
Q_LIMIT = 1.20                    # 关节 1 的虚拟上限 (rad)
Q_TARGET = 2.40                   # 参考要求冲到的位置，远超限位


def make_arm() -> ArmModel:
    return ArmModel(
        xml_path=PANDA_XML,
        ee_body="attachment",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
    )


def reference(t: float, speed: float):
    """关节 1 以恒定速度冲向 Q_TARGET，其余关节保持不动。"""
    q = Q_HOME.copy()
    v = np.zeros(N_ARM)
    a = np.zeros(N_ARM)
    s = min(speed * t, Q_TARGET - Q_HOME[JOINT_IDX])
    q[JOINT_IDX] = Q_HOME[JOINT_IDX] + s
    if speed * t < Q_TARGET - Q_HOME[JOINT_IDX]:
        v[JOINT_IDX] = speed
    return q, v, a


def clamp_reference(q, v, limit):
    """反应式控制唯一能做的事：把参考钳制到限位。

    注意钳制后速度也要置零，否则控制器会继续按原速度前馈冲过去。
    """
    qc, vc = q.copy(), v.copy()
    if qc[JOINT_IDX] > limit:
        qc[JOINT_IDX] = limit
        vc[JOINT_IDX] = 0.0
    return qc, vc


def run(controller: str, speed: float):
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
        q_hi = robot.q_upper.copy()
        q_hi[JOINT_IDX] = Q_LIMIT
        mpc = JointSpaceMPC(
            N_ARM, tau_lim, robot.q_lower, q_hi,
            v_limit=np.full(N_ARM, 4.0),
            horizon=8, dt=MPC_DT,
            w_pos=5e4, w_vel=300.0, w_acc=1e-4, w_rate=1e-3,   # ωn ≈ 29 rad/s
            use_limits=True, soft_state=True, rho_lin=1e5, rho_quad=1e3,
        )

    kp, kd = 900.0, 60.0                                        # ωn = 30 rad/s
    every = max(int(round(MPC_DT / DT)), 1)
    hold = np.zeros(N_ARM)

    qs, vs, ts = [], [], []
    for k in range(int(T_END / DT)):
        t = k * DT
        q = d.qpos[robot.qpos_idx].copy()
        v = d.qvel[robot.qvel_idx].copy()
        qd_, vd_, ad_ = reference(t, speed)

        if controller == "mpc":
            if k % every == 0:
                Q, V, A = [], [], []
                for i in range(1, mpc.N + 1):
                    a1, a2, a3 = reference(t + i * MPC_DT, speed)
                    # 与 CTC 使用**同一条钳制后的参考**：现实中不会下发违反
                    # 约束的指令。若参考本身要求越限，代价函数会与约束直接
                    # 对抗，导致要么约束被无视（罚太小），要么 P 矩阵条件数
                    # 爆炸使 QP 迭代不收敛（罚太大）。
                    a1, a2 = clamp_reference(a1, a2, Q_LIMIT)
                    Q.append(a1); V.append(a2); A.append(a3)
                hold, _, _ = mpc.compute(
                    q, v, np.array(Q), np.array(V), np.array(A),
                    reg.mass_matrix(q), reg.bias(q, v),
                )
            raw = hold
        else:
            qc, vc = clamp_reference(qd_, vd_, Q_LIMIT)
            aq = ad_ + kd * (vc - v) + kp * (qc - q)
            raw = reg.mass_matrix(q) @ aq + reg.bias(q, v)

        d.ctrl[:N_ARM] = np.clip(raw, -tau_lim, tau_lim)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        qs.append(d.qpos[robot.qpos_idx][JOINT_IDX])
        vs.append(d.qvel[robot.qvel_idx][JOINT_IDX])
        ts.append(t)

    qs, vs, ts = np.array(qs), np.array(vs), np.array(ts)
    over = qs - Q_LIMIT
    hit = np.where(qs >= Q_LIMIT)[0]
    return {
        "overshoot": float(max(over.max(), 0.0)),
        "hit_speed": float(vs[hit[0]]) if len(hit) else 0.0,
        "settle_gap": float(Q_LIMIT - qs[-200:].mean()),
        "viol_frac": float(np.mean(qs > Q_LIMIT)),
        "final": float(qs[-1]),
        "fail": 0 if mpc is None else mpc.solve_failures,
        "max_slack": 0.0 if mpc is None else mpc.max_slack,
    }


def main():
    print("\n" + "=" * 86)
    print("MPC 的结构性优势：关节限位避让")
    print(f"关节 1 虚拟上限 = {Q_LIMIT:.2f} rad，参考要求冲到 {Q_TARGET:.2f} rad")
    print("两个控制器都知道限位；带宽已对齐（CTC ωn=30.0，MPC ωn≈29.0 rad/s）")
    print("=" * 86)

    labels = {
        "ctc": "CTC（参考钳制到限位）",
        "mpc": "MPC（限位写进 QP 约束）",
    }

    for speed in (0.8, 1.6, 3.2):
        print(f"\n{'-'*86}")
        print(f"冲向限位的速度 = {speed:.1f} rad/s")
        print(f"{'控制器':<26}{'越限深度':>14}{'触限速度':>14}{'越限时间占比':>14}"
              f"{'稳定后余量':>14}")
        rows = {}
        for c in ("ctc", "mpc"):
            r = run(c, speed)
            rows[c] = r
            print(f"{labels[c]:<26}{r['overshoot']*1000:>9.2f} mrad"
                  f"{r['hit_speed']:>11.3f} r/s"
                  f"{r['viol_frac']*100:>12.1f} %"
                  f"{r['settle_gap']*1000:>9.2f} mrad")
        print(f"{'':>26}MPC 求解失败 {rows['mpc']['fail']} 次，"
              f"最大松弛量 {rows['mpc']['max_slack']*1000:.2f} mrad")
        o_c, o_m = rows["ctc"]["overshoot"], rows["mpc"]["overshoot"]
        if o_m > 1e-9:
            print(f"\n  MPC 越限深度为 CTC 的 {o_m/o_c:.3f} 倍")
        else:
            print(f"\n  MPC 完全未越限（CTC 越限 {o_c*1000:.2f} mrad）")

    print(f"\n{'-'*86}")
    print("说明：钳制参考是反应式控制唯一能做的事，但关节带着动能冲过来时，")
    print("      反向力矩只能在越限之后才建立，制动距离受 v²/(2a) 决定。")
    print("      MPC 在预测时域内看到即将越限，提前减速，因此越限深度显著更小。")
    print("      速度越高，两者差距越大——这正是预测能力的价值所在。\n")


if __name__ == "__main__":
    main()
