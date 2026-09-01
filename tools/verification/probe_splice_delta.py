#!/usr/bin/env python
"""独立复核 `replan` 场景的 `splice_dp` / `splice_dv` 到底在量什么。

审核指控
--------
第四轮审核说这两个读数「量错了 delta」。看实现：

    dp   = p_des[k] - p_des[k-1]              # 一拍之内指令位置走了多远
    vdes = dp / dt
    splice_dp = max_k ‖dp‖                    # 全程最大值
    splice_dv = max_k ‖vdes[k] - vdes[k-1]‖   # 全程最大值

标签写的是「**切换瞬间**的跳变」，实现取的却是**全程每拍位移的最大值**——
名义轨迹自己走得快的那一拍同样会被算进去。

oracle 独立性
------------
判据不看 `splice_dp` 自己，而是换两个完全独立的问题：

1. **零切换判据**：一次重规划都不发生时，「切换跳变」按定义必须是 0。
2. **地板判据**：把每拍位移的分布画出来，看极值到底比"名义运动地板"高多少。
   若极值只和地板差不到一倍，那它就不是跳变，是正常走路。

用法：
    cd <repo>
    PYTHONPATH=. python tools/verification/probe_splice_delta.py
"""
from __future__ import annotations

import mujoco
import numpy as np

from armctrl.tuner.scenes import SCENES

_RP = [s for s in SCENES if s.name == "replan"][0]


def roll(seconds=6.0, **overrides):
    sc = _RP()
    sc.build()
    for k, v in overrides.items():
        sc.set(k, v)
    sc.reset()
    d, r, m = sc.data, sc.robot, sc.model
    sub = max(1, int(round(sc.dt / m.opt.timestep)))
    pdes, switches, splice_on, tel = [], [], [], {}
    n_prev = 0
    for k in range(int(seconds / sc.dt)):
        t = k * sc.dt
        d.ctrl[r.arm_actuator_ids] = sc.control(
            t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
        for _ in range(sub):
            mujoco.mj_step(m, d)
        tel = sc.telemetry(t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
        pdes.append(np.asarray(sc._p_des, float).copy())
        # 直接问场景：这一拍是不是正处在 splice 过渡里
        splice_on.append(float(sc._splice_t0) >= 0.0)
        n_now = int(tel.get("n_replan", 0))
        if n_now > n_prev:
            switches.append(k)
            n_prev = n_now
    return tel, np.asarray(pdes), switches, np.asarray(splice_on), sc.dt


def summarize(tag, **kw):
    tel, pdes, switches, son, dt = roll(**kw)
    dp = np.linalg.norm(np.diff(pdes, axis=0), axis=1) * 1e3
    son = son[1:]                                  # 和 dp 对齐
    floor = float(np.percentile(dp[~son], 95)) if (~son).any() else float("nan")
    kmax = int(np.argmax(dp))
    print(f"\n--- {tag} ---")
    print(f"  n_replan = {int(tel.get('n_replan', 0))}   "
          f"splice 过渡中的拍数 = {int(son.sum())}")
    print(f"  读数 splice_dp = {tel['splice_dp']:9.3f} mm    "
          f"splice_dv = {tel['splice_dv']:10.1f} mm/s")
    print(f"  独立复算 max|dp| = {dp.max():9.3f} mm  (对上则公式确认)")
    print(f"  ⭐ 极值在第 {kmax} 拍，该拍是否处于 splice 过渡中："
          f"{'是' if son[kmax] else '❌ 否'}")
    print(f"  非过渡段的名义地板(95%) = {floor:8.3f} mm    "
          f"极值/地板 = {dp.max()/floor:.2f} 倍")
    if son.any():
        print(f"  过渡段内 max|dp| = {dp[son].max():9.3f} mm   "
              f"非过渡段 max|dp| = {dp[~son].max():9.3f} mm")
    return tel, dp, son


def main() -> None:
    print("=" * 72)
    print("replan 场景：splice_dp / splice_dv 到底在量什么")
    print("=" * 72)
    summarize("默认（splice_s=0.18）")
    summarize("硬切换（splice_s=0）", splice_s=0.0)
    summarize("过渡拉满（splice_s=0.6）", splice_s=0.6)
    tel, dp, son = summarize("⭐ 障碍物停住（obs_speed=0）", obs_speed=0.0)

    n = int(tel.get("n_replan", 0))
    print("\n" + "=" * 72)
    print("判决")
    print("-" * 72)
    if n != 0:
        print(f"⚠️  这一档触发了 {n} 次重规划，零切换判据用不上。")
    elif tel["splice_dp"] == 0.0 and tel["splice_dv"] == 0.0:
        print("✅ 零切换 ⇒ splice_dp / splice_dv 都恰好是 0。")
        print("   读数确实只统计「切换造成的」那部分，名义运动没有混进来。")
        print(f"   对照：同一段里 p_des 每拍最多走 {dp.max():.3f} mm——")
        print("   旧实现差分的正是它，所以永远有一个消不掉的地板。")
    else:
        print(f"❌ 零切换，但读数仍报 splice_dp = {tel['splice_dp']:.3f} mm、"
              f"splice_dv = {tel['splice_dv']:.1f} mm/s。")
        print("   「切换跳变」在没有切换时按定义必须是 0 ⇒ 量错了对象。")

    print("\n" + "-" * 72)
    print("splice_s 扫描（修好之后的真实跳变）")
    print("-" * 72)
    print(f"{'splice_s':>10} | {'splice_dp (mm)':>15} | {'splice_dv (mm/s)':>17}")
    for ss in (0.0, 0.06, 0.18, 0.40, 0.60):
        t2, _, _, _, _ = roll(6.0, splice_s=ss)
        print(f"{ss:>10.2f} | {t2['splice_dp']:>15.3f} | {t2['splice_dv']:>17.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
