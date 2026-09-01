"""控制律谱系统一基准评测。

在同一平台（Franka Panda 7-DOF）、同一轨迹、同一初始误差下对比：

    PD + 重力补偿   无模型基线，只靠反馈纠偏
    CTC             计算力矩控制，需要精确模型
    CTC(失配)       模型参数有 30% 误差时的退化，说明为什么需要自适应
    MBAC            Slotine-Li 模型自适应，结构已知参数未知，在线估参数
    RBF-NN          神经网络自适应，结构未知，在线逼近未知动力学

三把尺子：跟踪 RMSE、控制力矩范数、误差收敛过程。
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.assets import panda_nohand_xml
from armctrl.core.robot import ArmModel
from armctrl.identification.regressor import DynamicsRegressor
from armctrl.control.adaptive import (
    SlotineLiAdaptiveController,
    RBFAdaptiveController,
    fit_centers_to_trajectory,
)

PANDA_XML = panda_nohand_xml()

def make_arm() -> ArmModel:
    """无夹爪的纯 7 自由度 Panda：执行器数与关节数一致，消除夹爪耦合干扰。"""
    return ArmModel(
        xml_path=PANDA_XML,
        ee_body="attachment",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
    )


Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
DT = 0.002
T_END = 20.0
N_ARM = 7


def trajectory(t: float):
    """正弦关节轨迹，各关节频率不同以提供持续激励。"""
    amp = np.array([0.35, 0.25, 0.30, 0.25, 0.40, 0.30, 0.45])
    w = np.array([0.6, 0.45, 0.75, 0.5, 0.9, 0.7, 1.1])
    q = Q_HOME + amp * np.sin(w * t)
    v = amp * w * np.cos(w * t)
    a = -amp * w**2 * np.sin(w * t)
    return q, v, a


def traj_full(t: float, nv: int):
    """扩展到完整 nv 维（夹爪关节保持不动）。"""
    q, v, a = trajectory(t)
    Q, V, A = np.zeros(nv), np.zeros(nv), np.zeros(nv)
    Q[:N_ARM], V[:N_ARM], A[:N_ARM] = q, v, a
    return Q, V, A


def simulate(controller_fn, label: str, q_offset: np.ndarray):
    """通用闭环仿真。controller_fn(q, v, qd, vd, ad, dt) -> tau(nv,)。"""
    robot = make_arm()
    m, d = robot.model, robot.data
    n_sub = max(int(round(DT / m.opt.timestep)), 1)

    mujoco.mj_resetData(m, d)
    d.qpos[robot.qpos_idx] = Q_HOME + q_offset
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    errs, taus, ts = [], [], []
    for k in range(int(T_END / DT)):
        t = k * DT
        qd_f, vd_f, ad_f = traj_full(t, m.nv)
        q_f = d.qpos[: m.nv].copy()
        v_f = d.qvel[: m.nv].copy()

        # 夹爪关节不参与本次跟踪任务：把其期望设为当前状态，使误差为零。
        # 否则模型基控制器会通过耦合惯量 M[:7, 7:] 向手臂注入虚假力矩。
        qd_f[N_ARM:] = q_f[N_ARM:]
        vd_f[N_ARM:] = v_f[N_ARM:]
        ad_f[N_ARM:] = 0.0

        tau_f = controller_fn(q_f, v_f, qd_f, vd_f, ad_f, DT)
        tau = robot.saturate_torque(tau_f[:N_ARM])

        d.ctrl[:N_ARM] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

        errs.append(qd_f[:N_ARM] - d.qpos[robot.qpos_idx])
        taus.append(tau)
        ts.append(t)

    errs = np.array(errs)
    taus = np.array(taus)
    ts = np.array(ts)

    def rmse(mask):
        return float(np.sqrt(np.mean(np.linalg.norm(errs[mask], axis=1) ** 2)))

    return {
        "label": label,
        "w1": rmse(ts < 4.0),                       # 初始收敛 / 学习期
        "w2": rmse((ts >= 4.0) & (ts < 10.0)),      # 过渡期
        "w3": rmse(ts >= 10.0),                     # 稳态
        "tau_norm": float(np.mean(np.linalg.norm(taus, axis=1))),
        "tau_max": float(np.abs(taus).max()),
        "errs": errs,
        "ts": ts,
    }


def main():
    reg = DynamicsRegressor(PANDA_XML, n_arm=N_ARM)
    robot = make_arm()
    nv = reg.nv
    pi_true = reg.true_parameters()

    q_offset = np.array([0.10, -0.12, 0.08, 0.10, -0.15, 0.10, -0.20])
    lam, kd = 12.0, 70.0
    kp_ctc, kd_ctc = 400.0, 40.0          # 临界阻尼 ζ = 1.0

    print("\n" + "=" * 80)
    print("机械臂控制律谱系统一基准  |  Franka Panda 7-DOF  |  MuJoCo")
    print(f"轨迹 20 s 正弦跟踪，初始误差 ‖e(0)‖ = {np.linalg.norm(q_offset):.3f} rad")
    print("=" * 80)

    results = []

    # ---------- 1. PD + 重力补偿 ----------
    kp_pd, kd_pd = 250.0, 30.0

    def pd_ctrl(q, v, qd, vd, ad, dt):
        return kp_pd * (qd - q) + kd_pd * (vd - v) + reg.gravity(q)

    results.append(simulate(pd_ctrl, "PD + 重力补偿", q_offset))

    # ---------- 2. CTC（精确模型） ----------
    def ctc_ctrl(q, v, qd, vd, ad, dt):
        aq = ad + kd_ctc * (vd - v) + kp_ctc * (qd - q)
        return reg.mass_matrix(q) @ aq + reg.bias(q, v)

    results.append(simulate(ctc_ctrl, "CTC（精确模型 + 摩擦）", q_offset))

    # ---------- 2b. CTC（忽略摩擦） ----------
    import pinocchio as pin

    def ctc_nofric(q, v, qd, vd, ad, dt):
        aq = ad + kd_ctc * (vd - v) + kp_ctc * (qd - q)
        h_rigid = pin.rnea(reg.model, reg.data, q, v, np.zeros(nv))
        return reg.mass_matrix(q) @ aq + h_rigid

    results.append(simulate(ctc_nofric, "CTC（刚体模型，忽略摩擦）", q_offset))

    # ---------- 3. CTC（模型失配 30%） ----------
    scale = 0.7

    def ctc_bad(q, v, qd, vd, ad, dt):
        aq = ad + kd_ctc * (vd - v) + kp_ctc * (qd - q)
        return scale * (reg.mass_matrix(q) @ aq + reg.bias(q, v))

    results.append(simulate(ctc_bad, "CTC（模型失配 30%）", q_offset))

    # ---------- 4. MBAC（Slotine-Li） ----------
    mbac = SlotineLiAdaptiveController(
        reg, N_ARM, lam=lam, kd=kd, gamma=0.5, use_base=True
    )
    print(f"\nMBAC 自适应维数：标准参数 {reg.n_par} → 基参数集 {mbac.rank}")

    def mbac_ctrl(q, v, qd, vd, ad, dt):
        tau, _ = mbac.compute(q, v, qd, vd, ad, dt)
        return tau

    results.append(simulate(mbac_ctrl, "MBAC（Slotine-Li 自适应）", q_offset))

    # ---------- 5. RBF-NN 自适应 ----------
    C, W = fit_centers_to_trajectory(
        lambda t: traj_full(t, nv), T_END, n_centers=240, lam=lam
    )
    print(f"RBF 网络：{C.shape[0]} 个高斯中心，输入维数 {C.shape[1]}，"
          f"宽度范围 [{W.min():.2f}, {W.max():.2f}]")

    rbf = RBFAdaptiveController(
        N_ARM, C, W, lam=lam, kd=kd, gamma=200.0, sigma=0.02
    )

    def rbf_ctrl(q, v, qd, vd, ad, dt):
        tau7, _ = rbf.compute(q, v, qd, vd, ad, dt)
        out = np.zeros(nv)
        out[:N_ARM] = tau7[:N_ARM] if tau7.size >= N_ARM else tau7
        return out

    results.append(simulate(rbf_ctrl, "RBF-NN 自适应", q_offset))

    # ---------- 汇总 ----------
    print("\n" + "-" * 80)
    print("跟踪 RMSE 随时间分段（mrad）—— 自适应控制器应当逐段变好")
    print("-" * 80)
    print(f"{'控制器':<26}{'0–4s 学习期':>13}{'4–10s 过渡':>12}{'10–20s 稳态':>13}"
          f"{'平均‖τ‖':>11}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<26}{r['w1']*1000:>10.2f}   {r['w2']*1000:>9.2f}   "
            f"{r['w3']*1000:>10.2f}   {r['tau_norm']:>8.2f} Nm"
        )
    print("-" * 80)

    idx = {r["label"][:6]: r for r in results}
    ceil_ = results[1]["w3"]          # CTC 精确模型，性能上限
    mismatch = results[3]["w3"]       # CTC 模型失配，固定模型的代价

    print("\n关键对比：")
    print(f"  摩擦建模的价值      CTC 精确 {ceil_*1000:.2f} → 忽略摩擦 "
          f"{results[2]['w3']*1000:.2f} mrad，退化 {results[2]['w3']/ceil_:.1f} ×")
    print(f"  固定模型失配的代价  CTC 精确 {ceil_*1000:.2f} → 失配 30% "
          f"{mismatch*1000:.2f} mrad，退化 {mismatch/ceil_:.1f} ×")
    print("\n  自适应控制从零模型知识出发（π̂(0)=0、Ŵ(0)=0），对比失配固定模型：")
    for r in results[4:]:
        print(f"    {r['label']:<26}{r['w3']*1000:>7.2f} mrad   "
              f"优于失配 CTC {mismatch/r['w3']:>5.1f} ×   "
              f"距精确 CTC 上限 {r['w3']/ceil_:>4.1f} ×")

    print("\n学习过程（学习期 → 稳态 误差下降）：")
    for r in results:
        print(f"  {r['label']:<26}{r['w1']/max(r['w3'],1e-12):>8.1f} ×")

    print("\nMBAC 参数估计范数 ‖π̂‖ = %.3f（真值 ‖π‖ = %.3f）"
          % (np.linalg.norm(mbac.pi_hat), np.linalg.norm(pi_true)))
    print("  跟踪收敛不等于参数收敛：π̂ → π 还需轨迹满足持续激励（PE）条件。")
    print("RBF 权重范数 ‖Ŵ‖ = %.3f（σ-modification 保证有界，防止权重漂移）\n"
          % np.linalg.norm(rbf.W))


if __name__ == "__main__":
    main()
