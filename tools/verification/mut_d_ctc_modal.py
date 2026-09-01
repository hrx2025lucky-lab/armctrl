"""Batch D / P0-02 收尾：CTC 模态读数的变异验证。

每个变异体都是一次"把 bug 放回去"的尝试。测试必须**全部抓住**。
"""
import pathlib
import io
import os
import subprocess
import sys

#: 本文件位于 <repo>/tools/verification/，仓库根是上上级
ROOT = str(pathlib.Path(__file__).resolve().parents[2])
SCENES = os.path.join(ROOT, "armctrl/tuner/scenes.py")
DOC = os.path.join(ROOT, "armctrl_讲解/02_计算力矩.md")

REAL = ("                Mr = self._reg_mass(Q_HOME, self._pi_hat())[\n"
        "                    np.ix_(self._WN_JOINTS, self._WN_JOINTS)]\n"
        "                mf = modal_frequencies(Ms, kp_eff * Mr)")

MUTANTS = [
    # ---- 代码变异：把 P0-02 的 bug 放回去 --------------------------------
    (SCENES, REAL,
     "                mf = np.full(len(self._WN_JOINTS), "
     "np.sqrt(max(kp_eff, 0.0)))",
     "M1 退回写死的 √Kp（原始 bug）"),
    (SCENES, REAL,
     "                Mr = self._reg_mass(Q_HOME, self._pi_hat())[\n"
     "                    np.ix_(self._WN_JOINTS, self._WN_JOINTS)]\n"
     "                mf = modal_frequencies(Mr, kp_eff * Mr)",
     "M2 被控对象也用控制器模型（同模型自证）"),
    (SCENES, REAL,
     "                Mr = self._reg_mass(Q_HOME, self._pi_hat())[\n"
     "                    np.ix_(self._WN_JOINTS, self._WN_JOINTS)]\n"
     "                mf = modal_frequencies(Mr, kp_eff * Ms)",
     "M3 把 M_true 和 M_reg 弄反"),
    (SCENES, REAL,
     "                Mr = self._reg_mass(Q_HOME, self.pi_true)[\n"
     "                    np.ix_(self._WN_JOINTS, self._WN_JOINTS)]\n"
     "                mf = modal_frequencies(Ms, kp_eff * Mr)",
     "M4 用真参数建 M_reg（模型误差滑块又失效）"),
    (SCENES, "        M = np.column_stack(cols)\n        return 0.5 * (M + M.T)",
     "        M = np.column_stack(cols)\n"
     "        return np.diag(np.diag(0.5 * (M + M.T)))",
     "M5 M_reg 只取对角线（丢掉惯性耦合）"),
    (SCENES, "                mf = modal_frequencies(Ms, kp_eff)\n            else:",
     "                mf = modal_frequencies(Ms, kp_eff * self._reg_mass(\n"
     "                    Q_HOME, self._pi_hat())[np.ix_(\n"
     "                        self._WN_JOINTS, self._WN_JOINTS)])\n            else:",
     "M6 让 PD 也吃模型误差（本该不动的动了）"),

    # ---- 文档变异：表格数字被改坏 ---------------------------------------
    (DOC, "| 19.586 rad/s | 1.0188 |", "| 19.500 rad/s | 1.0188 |",
     "M7 §6.2d 10% 行的最慢模态改坏"),
    (DOC, "| 18.731 rad/s | 1.0604 |", "| 18.731 rad/s | 1.0700 |",
     "M8 §6.2d 30% 行的离散度改坏"),
    (DOC, "| 12.498 rad/s（不变） | 4.4804（不变） |",
     "| 12.600 rad/s（不变） | 4.4804（不变） |",
     "M9 §6.2d PD 行改坏"),
    (DOC, "| **20.00 rad/s** | **1.00×** |", "| **22.00 rad/s** | **1.00×** |",
     "M10 §6.3 CTC 最慢模态改坏"),
    (DOC, "但真实数值是 **1.0000225**", "但真实数值是 **1.0000300**",
     "M11 残差数字改坏（正文那一处）"),
    (DOC, "带一点点脏的数（1.0000225）", "带一点点脏的数（1.0000300）",
     "M11b 残差数字改坏（旁注那一处——测 #33 有没有真修好）"),
    (DOC, "| 模态离散度 | 1.000022 |", "| 模态离散度 | 1.000099 |",
     "M11c 连续性表首格改坏"),
    (DOC, "全是 1.000022495", "全是 1.000099495",
     "M11d Kp 不变性那句的数字改坏"),
    (DOC, "= 4.797\\times10^{-5}", "= 4.900\\times10^{-5}",
     "M11e ‖ΔM‖ 实测值改坏"),

    # ---- 文档变异：撤回声明被删 -----------------------------------------
    (DOC, "修复前它是**假设**，修复后它是**测量**", "这个数一直是测出来的",
     "M12 删掉「从假设变成测量」的撤回声明"),
    (DOC, "1. ⭐ **模型准确时，CTC 的模态离散度等于 1.00。**",
     "1. ⭐ **CTC 的模态离散度精确等于 1.00。**",
     "M13 退回「无条件解耦」的旧说法"),

    # ---- 自由振动表：窗长不变性那一套 -----------------------------------
    (DOC, "| **1 / 1 / 1 / 1 / 1** | **19.96** | 0.01 |",
     "| **1 / 1 / 1 / 1 / 1** | **19.90** | 0.01 |",
     "M16 CTC 0% 的稳定峰位置改坏"),
    (DOC, "| **1 / 1 / 1 / 1 / 1** | **18.88** | 0.02 |",
     "| **2 / 2 / 2 / 2 / 2** | **18.88** | 0.02 |",
     "M17 CTC 60% 的峰数改坏（谎称有两个峰）"),
    (DOC, "| 12 / 11 / 9 / 6 / 11 | **13.26** 、**26.57** | 0.05 / 0.02 |",
     "| 12 / 11 / 9 / 6 / 11 | **13.26** 、**26.57** | 0.05 / 0.09 |",
     "M18 PD 五窗极差改坏"),
    (DOC, "| 12 / 11 / 9 / 6 / 11 |", "| 11 / 11 / 9 / 6 / 11 |",
     "M19 PD 各窗峰数改坏"),
    (DOC, "**判断一个峰是不是真的，最简单的办法：换个窗长再看它动不动。**",
     "**峰就是峰，直接读出来就行。**",
     "M20 删掉「换窗长验峰」这条纪律"),
    (DOC, "**0.76 和 0.60**", "**0.05 和 0.02**",
     "M21 谎称假峰也很稳"),

    # ---- 测试自身的守卫：那条"锁住 bug"的断言不能回来 -------------------
    (os.path.join(ROOT, "armctrl/tests/test_ctc_modal.py"),
     "            self.assertNotAlmostEqual(\n"
     "                tel[\"modal_spread\"], 1.0, places=6,",
     "            self.assertAlmostEqual(\n"
     "                tel[\"modal_spread\"], 1.0, places=6,",
     "M14 把「离散度不得精确为 1」翻回「必须精确为 1」（新测试自己锁 bug）"),
    (os.path.join(ROOT, "armctrl/tests/test_modal.py"),
     "        self.assertNotAlmostEqual(\n"
     "            t[\"modal_spread\"], 1.0, places=6,",
     "        self.assertAlmostEqual(\n"
     "            t[\"modal_spread\"], 1.0, places=6,",
     "M15 同上，旧测试文件（这正是它上一轮犯的错）"),
]


MODULES = ["armctrl.tests.test_ctc_modal", "armctrl.tests.test_modal"]


def run() -> bool:
    r = subprocess.run(
        [PY, "-m", "unittest", *MODULES],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ".",
             "PYTHONDONTWRITEBYTECODE": "1"})
    return r.returncode == 0



def check_baseline() -> bool:
    """⭐ 变异前先跑一次**未改动**的测试，必须全绿。

    为什么这一步不能省：变异验证的逻辑是「改坏 -> 测试红 -> 记为被杀」。
    但「测试红」有两种原因——**真的被变异抓到**，或者**环境本来就是坏的**。
    第二种会让每个变异体都「被杀」，报告一片漂亮的全绿，实际什么都没验证。

    实测踩过：在一份新 clone 里跑，因为没找到 mujoco_menagerie，
    所有变异体都报「被杀」，而失败原因清一色是 ParseXML 找不到模型。
    """
    print("基线自检：先跑一次未改动的测试 ...", flush=True)
    ok, out = run(), ""
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
    if not run():
        print("基线就不过，先修基线")
        return 1
    print("基线 ✅ 通过\n")
    killed = survived = 0
    if not check_baseline():
        return 2
    for path, old, new, label in MUTANTS:
        src = io.open(path, encoding="utf-8").read()
        n = src.count(old)
        if n != 1:
            print(f"  ⚠️ {label}: 锚点出现 {n} 次，跳过")
            survived += 1
            continue
        io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
        try:
            ok = run()
        finally:
            io.open(path, "w", encoding="utf-8").write(src)
        if ok:
            print(f"  ❌ 存活 {label}")
            survived += 1
        else:
            print(f"  ✅ 杀死 {label}")
            killed += 1
    print(f"\n杀死 {killed} / 存活 {survived}")
    return 0 if survived == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
