#!/usr/bin/env python
"""独立复测 `01_阻抗控制.md` §十「K=50 那 33% 偏差」里的全部数字。

为什么要重测
------------
那一节写了 6 个数（下沉 404 mm、离地 9.4 mm、接触点 4 个、地面扛 9.78 N、
弹簧扛 20.3 N、qfrc_constraint 4.121 N·m），但**一个都没有测试守着**——
它们是三轮以前手工跑出来贴进文档的，之后场景改过好几次。
第四轮审核报告直接点名：ncon 实际是 3、qfrc_constraint 实际是 4.2579。

oracle 独立性
------------
- 地面托力**不读** MuJoCo 的 `qfrc_constraint`，而是逐个接触调
  `mj_contactForce` 拿接触力，自己转到世界系再取 z 分量求和。
- 「弹簧扛多少」**不读**控制器内部的 `f_cmd`，而是用胡克定律 K·Δz 现算。
- 两条路径只共享「牛顿第三定律」这一个前提，于是
  `外力 30 N == 地面托力 + 弹簧力` 才是一句有内容的校验。

用法：
    cd <repo>
    PYTHONPATH=. MUJOCO_GL=egl python tools/verification/probe_k50_contact.py
"""
from __future__ import annotations

import mujoco
import numpy as np

from armctrl.tuner.scenes import SCENES

_IMP = [s for s in SCENES if s.name == "impedance"][0]


def roll(seconds: float, **overrides):
    """跑一段仿真到稳态，返回 (scene, 最后一帧遥测)。"""
    sc = _IMP()
    sc.build()
    for key, val in overrides.items():
        sc.set(key, val)
    sc.reset()
    d, r = sc.data, sc.robot
    sub = max(1, int(round(sc.dt / sc.model.opt.timestep)))
    tel = {}
    for k in range(int(seconds / sc.dt)):
        t = k * sc.dt
        q = d.qpos[r.qpos_idx][:7]
        v = d.qvel[r.qvel_idx][:7]
        d.ctrl[r.arm_actuator_ids] = sc.control(t, q, v)
        for _ in range(sub):
            mujoco.mj_step(sc.model, d)
        tel = sc.telemetry(t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
    return sc, tel


def contact_report(sc):
    """逐个接触算世界系接触力，返回 (接触数, 世界 z 方向合力, 明细)。"""
    m, d = sc.model, sc.data
    fz_total, rows = 0.0, []
    buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, buf)
        # buf 前三项是接触系下的 (法向, 切向1, 切向2)；frame 是 3x3 行主序
        f_world = c.frame.reshape(3, 3).T @ buf[:3]
        fz_total += float(f_world[2])
        def nm(g):
            return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                    or f"geom#{int(g)}(body={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(g)])) or int(m.geom_bodyid[int(g)])})")
        rows.append((nm(c.geom1), nm(c.geom2),
                     float(c.dist), float(buf[0]), float(f_world[2])))
    return d.ncon, fz_total, rows


def main() -> None:
    K, FZ = 50.0, -30.0
    sc, tel = roll(8.0, k_trans=K, fz=FZ, force_mode="constant",
                   damping="critical")
    d, r = sc.data, sc.robot
    q = d.qpos[r.qpos_idx][:7]

    dz_mm = float(tel["dz"])
    theory_mm = abs(FZ) / K * 1e3
    ncon, fz_world, rows = contact_report(sc)

    print("=" * 68)
    print(f"K = {K:g} N/m, Fz = {FZ:g} N, 恒力模式, 8 s 到稳态")
    print("=" * 68)
    print(f"理论下沉  F/K            = {theory_mm:8.2f} mm")
    print(f"实测下沉  dz             = {dz_mm:8.2f} mm  "
          f"(缺口 {100*(1-abs(dz_mm)/theory_mm):.1f} %)")
    print()
    print(f"接触点数  ncon           = {ncon}")
    for g1, g2, dist, fn, fw in rows:
        print(f"    {g1:>28} ↔ {g2:<28} dist={dist:+.5f}  "
              f"|f_n|={fn:7.4f}  f_world_z={fw:+8.4f}")
    print(f"地面托力  Σf_world_z     = {fz_world:8.4f} N   (独立 oracle)")
    print(f"弹簧承担  K·|dz|         = {K*abs(dz_mm)*1e-3:8.4f} N   (胡克定律)")
    print(f"两者之和                 = {fz_world + K*abs(dz_mm)*1e-3:8.4f} N   "
          f"应 ≈ |Fz| = {abs(FZ):.1f} N")
    print()
    print(f"qfrc_constraint 范数     = "
          f"{float(np.linalg.norm(d.qfrc_constraint[r.qvel_idx][:7])):8.4f} N·m")
    print(f"qfrc_constraint 全量范数 = "
          f"{float(np.linalg.norm(d.qfrc_constraint)):8.4f} N·m")
    print()
    p, _R = r.fk(q)
    print(f"末端(法兰)高度 z         = {float(p[2])*1e3:8.2f} mm")

    # ---- 独立 oracle：把接触力用接触点雅可比搬到关节空间，和 qfrc_constraint 对照
    m = sc.model
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    tau_c = np.zeros(m.nv)
    buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, buf)
        f_world = c.frame.reshape(3, 3).T @ buf[:3]
        # 接触力作用在「非世界」的那个刚体上；floor 属于 world(0)
        b1 = int(m.geom_bodyid[int(c.geom1)])
        b2 = int(m.geom_bodyid[int(c.geom2)])
        body, sign = (b2, +1.0) if b1 == 0 else (b1, -1.0)
        mujoco.mj_jac(m, d, jacp, jacr, c.pos, body)
        tau_c += sign * (jacp.T @ f_world)
    res = float(np.linalg.norm(tau_c[r.qvel_idx][:7]))
    print(f"接触力搬到关节空间 ‖Jᵀf‖ = {res:8.4f} N·m   (独立 oracle)")
    print(f"最小接触间隙 min(dist)   = "
          f"{min([x[2] for x in rows], default=float('nan'))*1e3:8.2f} mm")
    J = r.jacobian(q)
    print(f"雅可比条件数 κ(J)        = {np.linalg.cond(J):8.3f}")
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"奇异值                   = "
          + " ".join(f"{s:.4f}" for s in sv))
    # z 方向是否落在 Jᵀ 的零空间里：看 Jᵀ 对单位 z 力旋量的响应
    w = np.zeros(6)
    w[2] = 1.0
    print(f"‖Jᵀ·ẑ‖ (为 0 才是使不上力) = {np.linalg.norm(J.T @ w):8.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
