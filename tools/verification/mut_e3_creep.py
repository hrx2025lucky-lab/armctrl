#!/usr/bin/env python
"""批次 E-3 变异验证：把抓取蠕变/摩擦锥相关的实现逐条改坏，看测试红不红。

⭐ 只变异**被守护的产物**（场景代码、control/grasp.py、讲解文档），
   绝不变异测试文件——把断言放松永远会通过，证明不了任何事（元教训 #44）。

⭐⭐ 本组的头号变异体是 M01：**把场景默认 impratio 直接设成 10**。
   旧测试（`速率 < 0.10 mm/s`）对它毫无感觉——蠕变变好了当然还是小于阈值。
   新测试必须抓住它，因为「默认配置不达标」这件事被显式钉住了。
   如果 M01 存活，说明这一批的"诚实记录"根本没生效。

用法：
    cd <repo>
    PYTHONPATH=. python tools/verification/mut_e3_creep.py
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
#: 一轮约 2 分钟。前三个是本批次新写的场景级测试；
#: `test_grasp` 是 grasp.py 的单元测试模块——第一轮漏掉它，
#: 结果 M09（有效摩擦取 max）在场景里是**等价变异体**（两侧摩擦被强制相等，
#: min 和 max 恒等），只有单元测试能杀。⭐ 教训：变异扫描的测试集
#: 必须覆盖「被变异的那个函数的所有调用方」，否则活口是假的。
TESTS = ["armctrl.tests.test_grasp_scene.TestCreepAgainstTheTaskSpec",
         "armctrl.tests.test_grasp_scene.TestTheCriterionCatchesRealConeViolation",
         "armctrl.tests.test_grasp_scene.TestTheFrictionSliderActuallyReachesBothSides",
         "armctrl.tests.test_grasp"]

SCN = "armctrl/tuner/scenes.py"
GRP = "armctrl/control/grasp.py"

FRICTION = """        m = self.model
        m.geom_friction[self.obj_gid, 0] = float(mu)
        for g in self._pad_geoms:
            m.geom_friction[g, 0] = float(mu)"""

#: (编号, 说明, 文件, 原文, 替换)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    ("M01", "⭐⭐ 场景默认 impratio 偷偷改成 10（旧测试完全抓不住）", SCN,
     FRICTION,
     FRICTION + "\n        m.opt.impratio = 10.0"),
    ("M02", "⭐ 接触变软：impratio 调到 0.2，蠕变必然恶化", SCN,
     FRICTION,
     FRICTION + "\n        m.opt.impratio = 0.2"),
    ("M03", "关掉 noslip 迭代（求解器不再抑制切向漂移）", SCN,
     FRICTION,
     FRICTION + "\n        m.opt.noslip_iterations = 0"),
    ("M04", "指垫摩擦不跟着改（滑块看起来在动，其实被 1.0 覆盖）", SCN,
     FRICTION,
     """        m = self.model
        m.geom_friction[self.obj_gid, 0] = float(mu)"""),
    ("M05", "工件摩擦不跟着改（只改指垫）", SCN,
     FRICTION,
     """        m = self.model
        for g in self._pad_geoms:
            m.geom_friction[g, 0] = float(mu)"""),
    ("M06", "⭐ 闭式漏掉接触面数 n（N ≥ mg/μ，算出来大一倍）", GRP,
     "    return float(safety) * load / (float(n_contacts) * mu)",
     "    return float(safety) * load / mu"),
    ("M07", "闭式把摩擦系数写到分子上", GRP,
     "    return float(safety) * load / (float(n_contacts) * mu)",
     "    return float(safety) * load * mu / float(n_contacts)"),
    ("M08", "闭式漏掉重力（只算加速度项）", GRP,
     "    load = float(mass) * (float(gravity) + float(accel))",
     "    load = float(mass) * float(accel)"),
    ("M09", "有效摩擦取 max 而不是 min（两侧不一致时会高估）", GRP,
     "    return min(mus) if mus else 0.0",
     "    return max(mus) if mus else 0.0"),
]


def run_tests(workdir: pathlib.Path = ROOT) -> tuple[bool, str]:
    r = subprocess.run(
        [PY, "-m", "unittest", *TESTS, "-q"],
        cwd=workdir, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ".",
             "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=3600)
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
    #: 支持 `--only M09,M04` 只跑指定变异体，省去整批 18 分钟的重跑
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}
    mutants = [m for m in MUTANTS if only is None or m[0] in only]
    print(f"批次 E-3 变异验证：{len(mutants)} 个变异体\n" + "=" * 64, flush=True)
    killed, survived = [], []
    if not check_baseline():
        return 2
    originals = {f: (ROOT / f).read_text(encoding="utf-8")
                 for f in {m[2] for m in mutants}}

    for tag, desc, rel, old, new in mutants:
        src = originals[rel]
        n = src.count(old)
        if n != 1:
            print(f"{tag} ⚠️  锚点在 {rel} 中出现 {n} 次，跳过 —— {desc}",
                  flush=True)
            survived.append((tag, desc, f"锚点 {n} 次"))
            continue
        path = ROOT / rel
        try:
            io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
            ok, out = run_tests(ROOT)
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
