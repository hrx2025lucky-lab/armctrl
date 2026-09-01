#!/usr/bin/env python
"""批次 D-2 变异证明：把 13 号 §7.4-7.6 与 test_servo_resonance 的新守卫逐条改坏。"""
import io, subprocess, sys, os
#: 本文件位于 <repo>/tools/verification/，仓库根是上上级
ROOT = str(pathlib.Path(__file__).resolve().parents[2])
DOC = os.path.join(ROOT, "armctrl_讲解/13_伺服环路与振动抑制.md")
TST = os.path.join(ROOT, "armctrl/tests/test_servo_resonance.py")

MUT = [
 (DOC, "| `probe_cl` $K_{vp}$=15 | 2.93 | 4.88 | 3.42 | 5.37 | **7.81** | 3.42 | 4.39 | **4.88** |",
       "| `probe_cl` $K_{vp}$=15 | 3.91 | 3.91 | 3.91 | 3.91 | 3.91 | 3.91 | 3.91 | **0.00** |",
  "M1 谎称闭环读数其实很稳（极差 0）"),
 (DOC, "| `probe_cl` $K_{vp}$=15 | 2.93 | 4.88 | 3.42 | 5.37 | **7.81** | 3.42 | 4.39 | **4.88** |",
       "| `probe_cl` $K_{vp}$=15 | 2.93 | 4.88 | 3.42 | 5.37 | **7.81** | 3.42 | 4.39 | **1.20** |",
  "M2 极差列与本行读数对不上"),
 (DOC, "旧解释：「主导极点换人了，看到的是**伺服环自己的闭环带宽峰**」。",
       "结论：主导极点换人了，看到的是**伺服环自己的闭环带宽峰**。",
  "M3 把错解释复活成正文结论"),
 (DOC, "现已作废，不要再引用", "实测真值",
  "M4 撤掉「已作废」标记"),
 (DOC, "**两根独立的轴都指向同一个答案：那个峰是反谐振，不是什么\"伺服带宽峰\"。**",
       "**那个峰的来源尚不清楚。**",
  "M5 删掉反谐振结论"),
 (DOC, "| 0.60 | 5.10 | 3.02 | 4.88 / 5.37 / 4.88 | 2.93 / 2.93 / 3.42 |",
       "| 0.60 | 5.10 | 3.02 | 4.88 / 5.37 / 4.88 | 4.88 / 5.37 / 4.88 |",
  "M6 谎称闭环峰也随 J_m 变（即它是机械谐振）"),
 (DOC, "| 0.15 | 8.75 | 3.02 | 8.30 / 9.28 / 8.79 | 2.93 / 2.93 / 3.42 |",
       "| 0.15 | 8.75 | 3.02 | 2.93 / 2.93 / 3.42 | 2.93 / 2.93 / 3.42 |",
  "M7 把开环列也改成不随 J_m 变（对照组失效）"),
 (DOC, "| 0.60 | 5.10 | 3.02 |", "| 0.60 | 5.10 | 5.10 |",
  "M8 让 f_ares 列随 J_m 变（违反 f_ares=sqrt(Ks/Jl)）"),
 (DOC, "| 1600 | 13.09 | 6.03 | 13.18 \u00d73 | **6.35** | $f_{ares}$ |",
       "| 1600 | 13.09 | 6.03 | 13.18 \u00d73 | **12.70** | $f_{res}$ |",
  "M9 K_s 表里闭环列改成跟 f_res 走"),
 (DOC, "| **60** | **2.93 Hz** | | \u2705 \u5c31\u662f\u5b83\uff08\u5dee 1 \u683c\uff0c\u5206\u8fa8\u7387 0.24 Hz\uff09|",
       "| **60** | **6.59 Hz** | | \u2705 \u5c31\u662f\u5b83\uff08\u5dee 1 \u683c\uff0c\u5206\u8fa8\u7387 0.24 Hz\uff09|",
  "M10 因果表 Kvp=60 的实测值改坏"),
 (DOC, "| **0** | **6.59 Hz** | \u2705 \u5c31\u662f\u5b83 | |",
       "| **0** | **3.15 Hz** | \u2705 \u5c31\u662f\u5b83 | |",
  "M11 因果表 Kvp=0 的实测值改坏"),
 (DOC, "\u90a3\u4e2a\u5cf0\u662f\u53cd\u8c10\u632f", "\u90a3\u4e2a\u5cf0\u662f\u4e2a\u8c1c",
  "M12 抽掉「那个峰是反谐振」这句结论"),
]

def run():
    r = subprocess.run([PY, "-m", "unittest",
                        "armctrl.tests.test_servo_resonance", "-q"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ".",
                            "PYTHONDONTWRITEBYTECODE": "1"})
    return "OK" in r.stderr.splitlines()[-1] if r.stderr.strip() else False

base = {f: io.open(f, encoding="utf-8").read() for f in {m[0] for m in MUT}}
assert run(), "基线就不绿，先修基线"
killed = survived = 0
for path, old, new, name in MUT:
    src = base[path]
    n = src.count(old)
    if n != 1:
        print(f"  ⚠️ 跳过 {name}：锚点出现 {n} 次"); survived += 1; continue
    io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
    ok = run()
    io.open(path, "w", encoding="utf-8").write(src)
    if ok:
        print(f"  ❌ 存活 {name}"); survived += 1
    else:
        print(f"  ✅ 杀死 {name}"); killed += 1
for f, src in base.items():
    io.open(f, "w", encoding="utf-8").write(src)
print(f"\n杀死 {killed} / 存活 {survived}")
sys.exit(1 if survived else 0)
