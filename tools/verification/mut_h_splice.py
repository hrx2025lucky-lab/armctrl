#!/usr/bin/env python
"""批次 H 变异验证：把 splice/跳变读数改回出问题的写法，看测试红不红。

⭐ 三个变异体直接就是**修复前的真实代码**。所以这组扫描回答的是一个
   很具体的问题：**如果早有这些测试，批次 H 这三个缺陷会不会漏过去？**

用法：
    cd <repo>
    PYTHONPATH=. python tools/verification/mut_h_splice.py
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
SCN = "armctrl/tuner/scenes.py"
TESTS = ["armctrl.tests.test_ilc_replan.TestReplanSceneSplice",
         "armctrl.tests.test_ilc_replan.TestQuinticSplice",
         "armctrl.tests.test_ilc_replan.TestTheSpliceSweepTableIsReproducible"]

#: 文档变异：把 §2.2 扫描表逐格改坏，确认那张表真的被守着
DOC = "armctrl_讲解/12_动态避障.md"
DOC_MUTANTS: list[tuple[str, str, str, str]] = [
    ("HD1", "⭐ 硬切换位置跳变改回历史错值 199.776 → 220.0",
     "| 0（硬切换，旧实现） | **199.776 mm** |",
     "| 0（硬切换，旧实现） | **220.0 mm** |"),
    ("HD2", "⭐ 默认档速度跳变改回修复前的 71 → 120",
     "| 0.18 s（默认） | **4.161 mm** | 71 mm/s |",
     "| 0.18 s（默认） | **4.161 mm** | 120 mm/s |"),
    ("HD3", "0.60 档位置跳变改错 1.249 → 2.5",
     "| 0.60 s | 1.249 mm | 6 mm/s |", "| 0.60 s | 2.5 mm | 6 mm/s |"),
    ("HD4", "⭐⭐ 把表格拆成普通文字（模拟「解析不到 → 静默全绿」）",
     "| 0.06 s | 12.449 mm | 637 mm/s |", "0.06 s 时 12.449 mm / 637 mm/s"),
]

DIFF_OFF = """        off = self._detour_at(t)
        if self._off_prev is not None:
            doff = off - self._off_prev
            voff = doff / self.dt
            self._splice_dp = max(self._splice_dp, float(np.linalg.norm(doff)))
            self._splice_dv = max(self._splice_dv,
                                  float(np.linalg.norm(voff - self._voff_prev)))
            self._voff_prev = voff
        self._off_prev = np.asarray(off, float).copy()"""

DIFF_PDES = """        off = self._p_des
        if self._off_prev is not None:
            doff = off - self._off_prev
            voff = doff / self.dt
            self._splice_dp = max(self._splice_dp, float(np.linalg.norm(doff)))
            self._splice_dv = max(self._splice_dv,
                                  float(np.linalg.norm(voff - self._voff_prev)))
            self._voff_prev = voff
        self._off_prev = np.asarray(off, float).copy()"""

EASE_NEW = """        want = 1.0 if (self._status == 0 and self._pending is None
                       and self._splice_t0 < 0.0) else 0.0
        rate = self.dt / self.DECAY_EASE_S
        self._decay_ease = float(np.clip(
            self._decay_ease + np.clip(want - self._decay_ease, -rate, rate),
            0.0, 1.0))
        if self._splice_t0 < 0.0:
            self._detour = self._detour * (1.0 - (1.0 - 0.999) * self._decay_ease)"""

EASE_OLD = """        if self._status == 0 and self._pending is None and self._splice_t0 < 0.0:
            self._detour = self._detour * 0.999"""

V0_NEW = """            self._splice_v0 = (np.zeros(3) if self._off_prev is None
                               else (_from - self._off_prev) / self.dt)"""
V0_OLD = """            self._splice_v0 = np.zeros(3)"""

MUTANTS: list[tuple[str, str, str, str]] = [
    ("H1", "⭐⭐ 跳变读数改回差分 p_des（名义运动混进来，产生消不掉的地板）",
     DIFF_OFF, DIFF_PDES),
    ("H2", "⭐ 偏置回落改回满速接管（`*0.999`，过渡结束那一拍速度阶跃 ~100 mm/s）",
     EASE_NEW, EASE_OLD),
    ("H3", "⭐ splice 起点速度强制写回 0（正在回落时切换 → 速度突降）",
     V0_NEW, V0_OLD),
    ("H4", "缓入时长拉到 0.001 s（几乎等于不缓入）",
     "    DECAY_EASE_S = 0.15", "    DECAY_EASE_S = 0.001"),
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
            only = {x.strip() for x in a.split("=", 1)[1].split(",") if x.strip()}
    mutants = [m for m in MUTANTS if only is None or m[0] in only]
    print(f"批次 H 变异验证：{len(mutants)} 实现 + "
          f"{len([m for m in DOC_MUTANTS if only is None or m[0] in only])} 文档\n"
          + "=" * 64, flush=True)
    if not check_baseline():
        return 2
    killed, survived = [], []
    doc_m = [m for m in DOC_MUTANTS if only is None or m[0] in only]
    for tag, desc, old, new in mutants + [
            (t, d, o, nw) for t, d, o, nw in doc_m]:
        rel = SCN if (tag, desc, old, new) in mutants else DOC
        path = ROOT / rel
        src = path.read_text(encoding="utf-8")
        n = src.count(old)
        if n != 1:
            print(f"{tag} ⚠️  锚点在 {rel} 中出现 {n} 次，跳过 —— {desc}", flush=True)
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
