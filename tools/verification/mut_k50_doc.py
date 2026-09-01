#!/usr/bin/env python
"""变异验证：`01_阻抗控制.md` §10 那张实测清单，改坏任何一格都必须测试红。

为什么要单独验这一组
--------------------
§10 的守护测试（`TestTheContactBreakdownAtK50IsReproducible`）
是**从 Markdown 现场解析期望值**的。这种写法有个特有的失败模式：
解析没匹配上 → 拿到空表 → 断言全部跳过 → 测试永远全绿。
所以必须专门把**文档里的数字**逐个改坏，确认测试真的会红。

⭐ 这里变异的是**文档**，不是测试文件——放松断言永远能过，证明不了任何事。
   变异体清一色用「这一节历史上写错过的那个旧值」，
   于是这组扫描顺便回答了一个具体问题：
   **如果当年就有这组测试，第四轮审核挑出来的错还会不会漏过去？**

用法：
    cd <repo>
    PYTHONPATH=. python tools/verification/mut_k50_doc.py
"""
from __future__ import annotations

import os
import io
import pathlib
import subprocess
import sys

# --- 仓库内自举：不依赖任何绝对路径 -------------------------------
#: 本文件位于 <repo>/tools/verification/，仓库根是上上级
ROOT = pathlib.Path(__file__).resolve().parents[2]
#: 优先用当前解释器（在 venv 里跑就是 venv 的），保证子进程和父进程一致
PY = sys.executable
DOC = "armctrl_讲解/01_阻抗控制.md"
TESTS = ["armctrl.tests.test_impedance_scene."
         "TestTheContactBreakdownAtK50IsReproducible"]

#: (编号, 说明, 原文, 替换)
MUTANTS: list[tuple[str, str, str, str]] = [
    ("D1", "⭐ 接触点数改回历史错值 3 → 4",
     "| 接触点数 `ncon` | 3 |", "| 接触点数 `ncon` | 4 |"),
    ("D2", "⭐ qfrc_constraint 改回历史错值 4.2611 → 4.121",
     "| `qfrc_constraint` | 4.2611 N·m |", "| `qfrc_constraint` | 4.121 N·m |"),
    ("D3", "地面托力改回历史错值 9.88 → 9.78",
     "| 地面托力 | 9.88 N |", "| 地面托力 | 9.78 N |"),
    ("D4", "弹簧承担改回历史错值 20.19 → 20.3",
     "| 弹簧承担 $K\\Delta z$ | 20.19 N |",
     "| 弹簧承担 $K\\Delta z$ | 20.3 N |"),
    ("D5", "实测下沉改错 403.71 → 420.00",
     "| 实测下沉 $\\Delta z$ | 403.71 mm |",
     "| 实测下沉 $\\Delta z$ | 420.00 mm |"),
    ("D6", "末端离地改错 9.42 → 12.00",
     "| 末端离地高度 | 9.42 mm |", "| 末端离地高度 | 12.00 mm |"),
    ("D7", "独立雅可比那一行改错 4.2611 → 4.5000",
     "$ | 4.2611 N·m | 自己用接触点雅可比搬一遍",
     "$ | 4.5000 N·m | 自己用接触点雅可比搬一遍"),
    ("D8", "条件数改回历史错值 10.727 → 10.8",
     "条件数只有 **10.727**", "条件数只有 **10.8**"),
    ("D9", "‖Jᵀẑ‖ 改成 0（等于宣称「真的是奇异」，结论会反过来）",
     "\\rVert = 0.5234 \\ne 0", "\\rVert = 0.0 \\ne 0"),
    ("D10", "⭐ 行标签漂移：「接触点数」改叫「碰撞点数」（测试必须发现少了一行）",
     "| 接触点数 `ncon` |", "| 碰撞点数 `ncon` |"),
    ("D11", "⭐⭐ 把表格拆成普通文字（模拟「解析不到 → 断言全跳过 → 静默全绿」）",
     "| 地面托力 | 9.88 N |", "地面托力 9.88 N"),
]


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        [PY, "-m", "unittest", *TESTS, "-q"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ".",
             "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=1800)
    return r.returncode == 0, r.stdout + r.stderr



def check_baseline() -> bool:
    """⭐ 变异前先跑一次**未改动**的测试，必须全绿。

    为什么这一步不能省：变异验证的逻辑是「改坏 -> 测试红 -> 记为被杀」。
    但「测试红」有两种原因——**真的被变异抓到**，或者**环境本来就是坏的**。
    第二种会让每个变异体都「被杀」，报告一片漂亮的全绿，实际什么都没验证。

    实测踩过：在一份新 clone 里跑，因为没找到 mujoco_menagerie，
    所有变异体都报「被杀」，而失败原因清一色是 ParseXML 找不到模型。
    """
    print("基线自检：先跑一次未改动的测试 ...", flush=True)
    ok, out = run_tests()
    if ok:
        print("基线 ✅ 全绿，开始变异")
        print("", flush=True)
        return True
    first = next((l for l in out.splitlines()
                  if l.startswith(("FAIL:", "ERROR:"))), "")
    print("")
    print("❌ 基线就不是绿的，变异验证无意义 —— 先把环境/测试修好。")
    print(f"   首个失败：{first}")
    if "ParseXML" in out or "menagerie" in out:
        print("   看起来是缺 Panda 模型。先确认环境：")
        print("     PYTHONPATH=. python -c "
              "'from armctrl.assets import require_panda_xml;"
              " print(require_panda_xml())'")
    return False

def main() -> int:
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}
    mutants = [m for m in MUTANTS if only is None or m[0] in only]
    print(f"§10 文档变异验证：{len(mutants)} 个变异体\n" + "=" * 64, flush=True)
    if not check_baseline():
        return 2

    path = ROOT / DOC
    src = path.read_text(encoding="utf-8")
    killed, survived = [], []

    for tag, desc, old, new in mutants:
        n = src.count(old)
        if n != 1:
            print(f"{tag} ⚠️  锚点出现 {n} 次，跳过 —— {desc}", flush=True)
            survived.append((tag, desc, f"锚点 {n} 次"))
            continue
        try:
            io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
            ok, out = run_tests()
        finally:
            io.open(path, "w", encoding="utf-8").write(src)
        if ok:
            print(f"{tag} ❌ 存活 —— {desc}", flush=True)
            survived.append((tag, desc, "测试仍然全绿"))
        else:
            first = next((l for l in out.splitlines()
                          if l.startswith(("FAIL:", "ERROR:"))), "?")
            print(f"{tag} ✅ 被杀 —— {desc}\n       {first}", flush=True)
            killed.append(tag)

    print("=" * 64)
    print(f"杀死 {len(killed)} / 存活 {len(survived)}")
    for tag, desc, why in survived:
        print(f"  存活 {tag}: {desc}  ({why})")
    return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
