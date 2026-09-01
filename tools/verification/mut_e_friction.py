#!/usr/bin/env python
"""批次 E / P1-05 变异验证：把修复逐条改坏，看测试会不会红。

⭐ 只变异**被守护的产物**（库代码、场景代码、讲解文档），
   绝不变异测试文件本身——把断言放松永远会通过，证明不了任何事。
   （这条教训来自批次 D-2，见 00_总索引.md 元教训 #44。）

用法：
    cd <repo>
    PYTHONPATH=. MUJOCO_GL=egl <venv-python> \\
        tools/verification/mut_e_friction.py
"""
from __future__ import annotations

import os
import io
import pathlib
import shutil
import subprocess
import sys
import tempfile

# --- 仓库内自举：不依赖任何绝对路径 -------------------------------
#: 本文件位于 <repo>/tools/verification/，仓库根是上上级
ROOT = pathlib.Path(__file__).resolve().parents[2]
#: 优先用当前解释器（在 venv 里跑就是 venv 的），保证子进程和父进程一致
PY = sys.executable

LIB = "armctrl/control/momentum_observer.py"
SCN = "armctrl/tuner/scenes.py"
DOC = "armctrl_讲解/10_拖动示教.md"

#: (编号, 说明, 文件, 原文, 替换)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    # ---------- 判据本身 ----------
    ("M01", "退回只看 tanh 的旧界（丢掉 f_v 和 D）", LIB,
     "    return (fc * np.tanh(u / tanh_scale) + (fv + damping) * u) / (fc + fv * u)",
     "    return np.tanh(u / tanh_scale) + 0.0 * (fc + fv + damping) * u"),
    ("M02", "把控制器阻尼 D 从分子里拿掉", LIB,
     "return (fc * np.tanh(u / tanh_scale) + (fv + damping) * u) / (fc + fv * u)",
     "return (fc * np.tanh(u / tanh_scale) + fv * u) / (fc + fv * u)"),
    ("M03", "把粘滞项 f_v 从分母里拿掉", LIB,
     "return (fc * np.tanh(u / tanh_scale) + (fv + damping) * u) / (fc + fv * u)",
     "return (fc * np.tanh(u / tanh_scale) + (fv + damping) * u) / fc"),
    ("M04", "天花板写回 1.0（恢复「η≥1 没救」的老错）", LIB,
     "    return 1.0 + damping / fv",
     "    return 1.0"),
    ("M05", "天花板漏掉 f_v 归一化（写成 1 + D）", LIB,
     "    return 1.0 + damping / fv",
     "    return 1.0 + damping"),

    # ---------- 死区求解 ----------
    ("M06", "η≥1 直接返回 inf（不看阻尼）", LIB,
     "    if e >= _passivity_ceiling(fv, damping):",
     "    if e >= 1.0:"),
    ("M07", "二分求上确界时取下界（少给一点死区）", LIB,
     "        if _passivity_g(mid, fc, fv, damping, a) < e:\n"
     "            lo = mid\n        else:\n            hi = mid\n    return hi",
     "        if _passivity_g(mid, fc, fv, damping, a) < e:\n"
     "            lo = mid\n        else:\n            hi = mid\n    return lo * 0.98"),
    # ⚠️ M08 曾经是 bad.max() -> bad.min()。实测 4000 组随机参数下
    #    g(u) **严格单调递增**，违规集合是一个区间，min/max 会被二分
    #    收敛到同一个交点 —— 这是**等价变异体**，杀不掉是对的，不是洞。
    #    换成一个真会出错的版本：直接返回违规集合的最小值，不做二分。
    ("M08", "违规区间取最小值且不二分（死区严重不足）", LIB,
     "    lo, hi = float(bad.max()), top * 1.05",
     "    return float(bad.min())\n    lo, hi = float(bad.max()), top * 1.05"),

    # ---------- 功率扫描窗口 ----------
    ("M09", "扫描上限退回硬截断 v_max（低报 5.26 倍）", LIB,
     "    hi = max(float(v_max), 5.0 * deadband, 1.05 * fc * eta / c if c > 0 else 0.0)",
     "    hi = float(v_max)"),
    # ⚠️ M10 是**等价变异体**：注能区间必然落在 u <= f_c·η/c 之内
    #    （η<1 时更是只在 u <= a·atanh(η) 这一小段），
    #    所以第三项已经包住一切，5·deadband 是纯粹的防御性冗余。
    #    保留它是为了万一以后有人改坏第三项 —— 但它杀不掉，如实记下来。
    ("M10", "扫描上限漏掉 5·deadband（大死区时全瞎）", LIB,
     "    hi = max(float(v_max), 5.0 * deadband, 1.05 * fc * eta / c if c > 0 else 0.0)",
     "    hi = max(float(v_max), 1.05 * fc * eta / c if c > 0 else 0.0)"),
    ("M11", "发散配置不报 inf，闷头报个有限值", LIB,
     "    if c <= 0.0 and eta > 0.0:\n        return float(\"inf\"), float(\"inf\")",
     "    if c <= 0.0 and eta > 0.0:\n        c = 1.0"),
    ("M12", "去掉双尺度网格（近原点分辨率丢失）", LIB,
     "    u = np.unique(np.concatenate([np.linspace(0.0, near, n),\n"
     "                                  np.linspace(near, hi, n)]))",
     "    u = np.linspace(0.0, hi, n)"),
    # ⚠️ M13 曾经是 1.05 -> 0.5。真峰在 u* = f_c(η−1)/(2c)，
    #    而 0.5·f_c·η/c > u* 恒成立（等价于 η−1 < η），
    #    所以 0.5 仍然包住峰 —— **等价变异体**。
    #    0.05 才真的不够：η ≥ 1.111 时窗口就短于真峰位置。
    ("M13", "把安全系数 1.05 改成 0.05（窗口真的不够长）", LIB,
     "1.05 * fc * eta / c if c > 0 else 0.0",
     "0.05 * fc * eta / c if c > 0 else 0.0"),

    # ---------- 向后兼容 ----------
    ("M14", "两参数形式也偷偷用紧界（破坏保守性契约）", LIB,
     "    if fv == 0.0 and damping == 0.0:\n        return float(np.tanh(db / a))",
     "    if fv == 0.0 and damping == 0.0:\n        return float(np.tanh(db / a)) * 1.02"),
    ("M15", "两参数死区形式退化（不再返回 inf）", LIB,
     "    if fv == 0.0 and damping == 0.0:\n"
     "        return float(\"inf\") if e >= 1.0 else a * float(np.arctanh(e))",
     "    if fv == 0.0 and damping == 0.0:\n"
     "        return a * float(np.arctanh(min(e, 0.999)))"),

    # ---------- 场景接线 ----------
    ("M16", "场景退回不带阻尼的死区计算", SCN,
     "        need = passivity_deadband(\n"
     "            self.get(\"comp_ratio\"), self.TANH_SCALE,\n"
     "            fc=self.get(\"fc\"), fv=self.get(\"fv\"), damping=self.get(\"damping\"))",
     "        need = passivity_deadband(self.get(\"comp_ratio\"), self.TANH_SCALE)"),
    ("M17", "场景只在 η 变时重算死区（改阻尼没反应）", SCN,
     "        if key in (\"comp_ratio\", \"passive\", \"damping\", \"fc\", \"fv\"):\n"
     "            self._apply_deadband()",
     "        if key in (\"comp_ratio\", \"passive\"):\n            self._apply_deadband()"),
    ("M18", "f_c 写成不存在的属性名（静默失效）", SCN,
     "            self.teach.fc = np.full(N_ARM, self.get(\"fc\"))",
     "            self.teach.friction_coulomb = np.full(N_ARM, self.get(\"fc\"))"),
    ("M19", "f_v 写成不存在的属性名（静默失效）", SCN,
     "            self.teach.fv = np.full(N_ARM, self.get(\"fv\"))",
     "            self.teach.friction_viscous = np.full(N_ARM, self.get(\"fv\"))"),
    ("M20", "代价太大时偷偷放大死区假装安全", SCN,
     "        if not np.isfinite(need) or need > self.DEADBAND_CAP:\n"
     "            self.teach.vel_deadband = base\n            return",
     "        if not np.isfinite(need):\n"
     "            self.teach.vel_deadband = base\n            return"),
    ("M21", "上限读数退回保守界", SCN,
     "        emax = max_passive_ratio(db, self.TANH_SCALE, fc=self.get(\"fc\"),\n"
     "                                 fv=self.get(\"fv\"), damping=self.get(\"damping\"))",
     "        emax = max_passive_ratio(db, self.TANH_SCALE)"),

    # ---------- 讲解文档 ----------
    ("M22", "文档 η 表：把 1.2 那行改回旧的低报值", DOC,
     "| 1.20 | **+1.00e-1 W** | ❌ |", "| 1.20 | +1.90e-2 W | ❌ |"),
    ("M23", "文档 η 表：临界值写回保守界", DOC,
     "| **0.4625015** | **0（临界，真界）** | ✅ |",
     "| **0.4621172** | **0（临界，真界）** | ✅ |"),
    ("M24", "文档死区表：最小死区写回旧值", DOC,
     "| **2.193574e-3** | **0** | ✅ |", "| **2.197000e-3** | **0** | ✅ |"),
    ("M25", "文档死区表：把「还差一点点」那行谎称已被动", DOC,
     "| 2.150e-3 | +3.42e-5 W | ❌（**还差一点点**）|",
     "| 2.150e-3 | 0 W | ✅ |"),
    ("M26", "文档反例表：天花板写回 1.0", DOC,
     "| 1.2 | 8.0 | 17.0 | **50.633 mrad/s** | ✅ 被动 |",
     "| 1.2 | 8.0 | 1.0 | **50.633 mrad/s** | ✅ 被动 |"),
    ("M27", "文档反例表：死区数字改小（就不再被动了）", DOC,
     "| 1.2 | 8.0 | 17.0 | **50.633 mrad/s** | ✅ 被动 |",
     "| 1.2 | 8.0 | 17.0 | **20.000 mrad/s** | ✅ 被动 |"),
    ("M28", "文档反例表：谎称 D=0 时也有救", DOC,
     "| 1.2 | **0** | **1.0** | — | ❌ 真的没救 |",
     "| 1.2 | **0** | **1.0** | 99.9 mrad/s | ✅ 被动 |"),
    ("M29", "文档撤回：删掉「这句话也是错的」", DOC,
     "### 4.2.5 ⚠️⚠️ 「$\\eta\\ge1$ 就没救」——这句话也是错的",
     "### 4.2.5 ⚠️ 保护不是万能的"),
    ("M30", "文档 4.2.2：删掉完整 g(u)，退回只看 tanh", DOC,
     "$$\\boxed{\\;\\eta \\;\\le\\; g(u):=\\frac{f_c\\tanh(u/a)+(f_v+D)\\,u}{f_c+f_v u}\n"
     "\\qquad \\forall\\,u>d_b\\;}$$",
     "$$\\boxed{\\;\\eta \\;\\le\\; \\tanh(u/a)\\qquad \\forall\\,u>d_b\\;}$$"),
    ("M31", "文档：删掉撤回出处「第五轮审核（P1-05）」", DOC,
     "第五轮审核（P1-05）抓出来的", "我自己复查时发现的"),
    ("M32", "文档 4.2.4：死区表里删掉旧公式对照行", DOC,
     "| 2.197e-3（旧公式给的）| 0 | ✅（多开 0.17 %，浪费但安全）|\n", ""),
]

TEST = "armctrl.tests.test_friction_passivity"


def run_tests(workdir: pathlib.Path = ROOT) -> tuple[bool, str]:
    r = subprocess.run(
        [PY, "-m", "unittest", TEST, "-q"],
        cwd=workdir, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ".",
             "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
             "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=900)
    out = r.stdout + r.stderr
    return r.returncode == 0, out



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
    print(f"批次 E / P1-05 变异验证：{len(MUTANTS)} 个变异体\n" + "=" * 62)
    killed, survived = [], []
    if not check_baseline():
        return 2
    originals = {f: (ROOT / f).read_text(encoding="utf-8")
                 for f in {m[2] for m in MUTANTS}}

    for tag, desc, rel, old, new in MUTANTS:
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
