#!/usr/bin/env python
"""批次 E-2 / P1-09 变异验证：把柔度补偿逐条改坏，看测试会不会红。

⭐ 只变异**被守护的产物**（场景代码、讲解文档），
   绝不变异测试文件本身——把断言放松永远会通过，证明不了任何事。
   （元教训 #44。）

⭐⭐ 本组的头号变异体是 M03：**两端符号同时翻**。
   旧代码两边共用同一个 ``deflect``，这个变异体在数学上完全等价，
   8/8 测试全绿——审核报告 P1-09 抓的就是它。
   修复之后它必须被杀掉，否则这轮修复等于没做。

用法：
    cd <repo>
    PYTHONPATH=. MUJOCO_GL=egl <venv-python> \\
        tools/verification/mut_e2_ilc.py
"""
from __future__ import annotations

import io
import pathlib
import subprocess
import sys

# --- 仓库内自举：不依赖任何绝对路径 -------------------------------
#: 本文件位于 <repo>/tools/verification/，仓库根是上上级
ROOT = pathlib.Path(__file__).resolve().parents[2]
#: 优先用当前解释器（在 venv 里跑就是 venv 的），保证子进程和父进程一致
PY = sys.executable
TEST = "armctrl.tests.test_ilc_replan"

SCN = "armctrl/tuner/scenes.py"
DOC = "armctrl_讲解/11_轨迹精度补偿.md"

COMP_ON = """        if self.choice("comp") == "on":
            qd_cmd = qd_cmd + defl_comp"""
PLANT = """        q_eff = q - defl_plant + geom + noise"""

#: (编号, 说明, 文件, 原文, 替换)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    # ---------- ⭐ 核心：两端必须是两个不同的量 ----------
    ("M01", "被控对象退回复用补偿器的量（还原 P1-09 那个 bug）", SCN,
     "        defl_plant = self.compliance.deflection(self._tau)",
     "        defl_plant = defl_comp"),
    ("M02", "补偿器偷看被控对象的真实力矩（作弊：τ 而非 bias）", SCN,
     "        defl_comp = self.compliance.deflection(tau_nom)",
     "        defl_comp = self.compliance.deflection(\n"
     "            self.reg.mass_matrix(q) @ ad + tau_nom)"),
    ("M03", "⭐⭐ 两端符号同时翻（旧代码抓不住的那个）", SCN,
     COMP_ON + "\n",
     """        if self.choice("comp") == "on":
            qd_cmd = qd_cmd - defl_comp
"""),
    ("M04", "只翻补偿端符号", SCN,
     "            qd_cmd = qd_cmd + defl_comp",
     "            qd_cmd = qd_cmd - defl_comp"),
    ("M05", "只翻被控对象端符号", SCN,
     PLANT,
     "        q_eff = q + defl_plant + geom + noise"),
    ("M06", "补偿开关失效：无论选什么都补", SCN,
     '        if self.choice("comp") == "on":\n            qd_cmd = qd_cmd + defl_comp',
     "        if True:\n            qd_cmd = qd_cmd + defl_comp"),
    ("M07", "补偿开关失效：无论选什么都不补", SCN,
     '        if self.choice("comp") == "on":\n            qd_cmd = qd_cmd + defl_comp',
     "        if False:\n            qd_cmd = qd_cmd + defl_comp"),

    # ---------- 惯性形变读数 ----------
    ("M08", "惯性形变读数恒为 0（假装没有建不了模的部分）", SCN,
     "        self._defl_extra = float(np.linalg.norm(defl_plant - defl_comp)) * 1e3",
     "        self._defl_extra = 0.0"),
    ("M09", "惯性形变读成 plant 全量（没减掉可建模那部分）", SCN,
     "float(np.linalg.norm(defl_plant - defl_comp)) * 1e3",
     "float(np.linalg.norm(defl_plant)) * 1e3"),
    ("M10", "惯性形变漏了 1e3 单位换算（rad 当 mrad 报）", SCN,
     "float(np.linalg.norm(defl_plant - defl_comp)) * 1e3",
     "float(np.linalg.norm(defl_plant - defl_comp))"),
    ("M11", "可建模形变读数改用 plant 力矩（两个读数变成同一个）", SCN,
     "        deflect = float(np.linalg.norm(\n"
     "            self.compliance.deflection(self.reg.bias(q, v)))) * 1e3",
     "        deflect = float(np.linalg.norm(\n"
     "            self.compliance.deflection(self._tau))) * 1e3"),

    # ---------- 被控对象力矩来源 ----------
    ("M12", "被控对象用未裁剪的 τ（忽略力矩饱和）", SCN,
     "        defl_plant = self.compliance.deflection(self._tau)",
     "        defl_plant = self.compliance.deflection(tau)"),
    ("M13", "柔度模型除法写成乘法", "armctrl/control/compliance.py",
     "PLACEHOLDER_WILL_BE_PATCHED", "X"),
]


def run_tests(workdir: pathlib.Path) -> tuple[bool, str]:
    r = subprocess.run(
        [PY, "-m", "unittest", TEST, "-q"],
        cwd=workdir, capture_output=True, text=True,
        env={"PYTHONPATH": ".", "MUJOCO_GL": "egl", "PATH": "/usr/bin:/bin",
             "HOME": str(pathlib.Path.home()), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=1800)
    out = r.stdout + r.stderr
    return r.returncode == 0, out


def main() -> int:
    mutants = [m for m in MUTANTS if m[0] != "M13"]
    print(f"批次 E-2 / P1-09 变异验证：{len(mutants)} 个变异体\n" + "=" * 62)
    killed, survived = [], []
    originals = {f: (ROOT / f).read_text(encoding="utf-8")
                 for f in {m[2] for m in mutants}}

    for tag, desc, rel, old, new in mutants:
        src = originals[rel]
        n = src.count(old)
        if n != 1:
            print(f"{tag} ⚠️  锚点在 {rel} 中出现 {n} 次，跳过 —— {desc}")
            survived.append((tag, desc, f"锚点 {n} 次"))
            continue
        path = ROOT / rel
        try:
            io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
            ok, out = run_tests(ROOT)
        finally:
            io.open(path, "w", encoding="utf-8").write(src)
        if ok:
            print(f"{tag} ❌ 存活 —— {desc}")
            survived.append((tag, desc, "测试仍然全绿"))
        else:
            first = next((l for l in out.splitlines()
                          if l.startswith(("FAIL:", "ERROR:"))), "?")
            print(f"{tag} ✅ 被杀 —— {desc}\n       {first}")
            killed.append(tag)

    print("=" * 62)
    print(f"杀死 {len(killed)} / 存活 {len(survived)}")
    for tag, desc, why in survived:
        print(f"  存活 {tag}: {desc}  ({why})")
    return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
