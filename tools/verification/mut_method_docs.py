#!/usr/bin/env python
"""变异验证：`docs/控制方法/` 的守护测试（test_method_docs.py）真的会红吗。

为什么要验这一组
----------------
这批守护测试是**从 Markdown 解析**的。这种写法有个特有的失败模式：
解析没匹配上 → 拿到空集合 → 断言全部跳过 → 测试永远全绿。

所以必须把**文档**逐条改坏，确认测试真的会红。

⭐ 这里变异的是**文档**，不是测试文件——放松断言永远能过，证明不了任何事。

用法：
    cd <repo>
    PYTHONPATH=. python tools/verification/mut_method_docs.py
    PYTHONPATH=. python tools/verification/mut_method_docs.py --only=M01,M05
"""
from __future__ import annotations

import io
import os
import pathlib
import subprocess
import sys

# --- 仓库内自举：不依赖任何绝对路径 -------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable

TESTS = ["armctrl.tests.test_method_docs"]

D = "docs/控制方法"

#: (编号, 说明, 相对路径, 原文, 替换)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    ("M01", "⭐ 编造一个不存在的函数名",
     f"{D}/01_笛卡尔阻抗与导纳.md",
     "## 六、边界与局限",
     "调用 `ThisFunctionDoesNotExist()` 即可。\n\n## 六、边界与局限"),

    ("M02", "⭐ 引用一个不存在的源文件",
     f"{D}/02_动量观测器与碰撞检测.md",
     "## 六、边界与局限",
     "详见 `armctrl/control/no_such_file.py`。\n\n## 六、边界与局限"),

    ("M03", "⭐⭐ 把文档写的默认值改错（代码没变，文档变了）",
     f"{D}/11_逆运动学与奇异性.md",
     "- 默认 `sigma0_pos=0.030707`；",
     "- 默认 `sigma0_pos=0.999999`；"),

    ("M04", "引用一个不存在的测试模块",
     f"{D}/03_自适应控制.md",
     "## 六、边界与局限",
     "跑 `armctrl.tests.test_nonexistent_module` 验证。\n\n## 六、边界与局限"),

    ("M05", "⭐ 删掉「怎么用在机械臂上」这一节（骨架被破坏）",
     f"{D}/04_模型预测控制MPC.md",
     "## 三、怎么用在机械臂上",
     "## 三、随便写点什么"),

    ("M06", "⭐⭐ 本机绝对路径又混进来（别人照抄会失败）",
     f"{D}/05_迭代学习控制ILC.md",
     "## 六、边界与局限",
     "```bash\ncd /home/someuser/project && python x.py\n```\n\n## 六、边界与局限"),

    ("M07", "⭐ 写作过程自述又混进来",
     f"{D}/06_模态分析与振动抑制.md",
     "## 六、边界与局限",
     "本次核对的真实符号：全部存在。\n\n## 六、边界与局限"),

    ("M08", "整篇文档被删（篇数少一）",
     f"{D}/08_视觉伺服.md",
     "# 视觉伺服", "# 视觉伺服（占位）"),   # 见下：M08 用改名模拟
]

#: M08 不是内容变异，单独处理：把文件挪走再挪回
RENAME_MUTANT = ("M08", "⭐ 整篇文档消失（篇数守护该报警）",
                 f"{D}/08_视觉伺服.md")


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

    没有基线，「测试红了」既可能是被变异抓到，也可能是环境本来就坏的——
    后者会让每个变异体都「被杀」，报告一片漂亮的全绿，实际什么都没验证。
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
    return False


def main() -> int:
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}

    content = [m for m in MUTANTS if m[0] != "M08"
               and (only is None or m[0] in only)]
    do_rename = only is None or RENAME_MUTANT[0] in only

    total = len(content) + (1 if do_rename else 0)
    print(f"控制方法文档守护 · 变异验证：{total} 个变异体")
    print("=" * 64, flush=True)
    if not check_baseline():
        return 2

    killed, survived = [], []

    for tag, desc, rel, old, new in content:
        path = ROOT / rel
        src = path.read_text(encoding="utf-8")
        n = src.count(old)
        if n != 1:
            print(f"{tag} ⚠️  锚点在 {rel} 中出现 {n} 次，跳过 —— {desc}",
                  flush=True)
            survived.append((tag, desc, f"锚点 {n} 次"))
            continue
        try:
            io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
            ok, out = run_tests()
        finally:
            io.open(path, "w", encoding="utf-8").write(src)
        _report(tag, desc, ok, out, killed, survived)

    if do_rename:
        tag, desc, rel = RENAME_MUTANT
        path = ROOT / rel
        tmp = path.with_suffix(".md.bak")
        try:
            path.rename(tmp)
            ok, out = run_tests()
        finally:
            tmp.rename(path)
        _report(tag, desc, ok, out, killed, survived)

    print("=" * 64)
    print(f"杀死 {len(killed)} / 存活 {len(survived)}")
    for tag, desc, why in survived:
        print(f"  存活 {tag}: {desc}  ({why})")
    return 0 if not survived else 1


def _report(tag, desc, ok, out, killed, survived) -> None:
    if ok:
        print(f"{tag} ❌ 存活 —— {desc}", flush=True)
        survived.append((tag, desc, "测试仍然全绿"))
    else:
        first = next((l for l in out.splitlines()
                      if l.startswith(("FAIL:", "ERROR:"))), "?")
        print(f"{tag} ✅ 被杀 —— {desc}\n       {first}", flush=True)
        killed.append(tag)


if __name__ == "__main__":
    sys.exit(main())
