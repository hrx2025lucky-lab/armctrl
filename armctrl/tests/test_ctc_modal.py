"""CTC 模态频率读数的守卫（第五轮审核 P0-02 收尾）。

修复前 `TrackingScene.telemetry` 对计算力矩分支直接写
``mf = np.full(n, sqrt(kp))``——「模态离散度 = 1.00」是**填进去的**，
不是量出来的，所以「模型误差」滑块拖到底它也不动。

这里的守卫分两层：

1. **行为层**：模型误差必须能推动 CTC 的模态读数，
   而且必须**推不动** PD 的模态读数（PD 的模型只进重力补偿，不进齐次方程）。
   ⭐ 只测前者不够——把 CTC 换成任何随模型误差乱跳的量都能过。
2. **独立 oracle 层**：冻住参考、把关节推开、撒手让**真实 MuJoCo 闭环**
   自由振荡，对误差做 FFT 数频率。这条路径里没有任何矩阵特征值，
   和被测代码不共用任何计算。

⚠️ 文档表格的期望值**一律从 Markdown 里现场解析**，不写死在测试里
（元教训 #35）：期望值写死在测试里，测试就只是在和自己的常数比。
"""
import io
import os
import re
import unittest

import mujoco
import numpy as np

from armctrl.control.modal import modal_frequencies
from armctrl.tuner.scenes import SCENES, Q_HOME
from armctrl.tests._docs import requires_docs

DOC = os.path.join(os.path.dirname(__file__), "..", "..",
                   "armctrl_讲解", "02_计算力矩.md")
WN = (0, 1, 3, 5)
_TRACKING = [s for s in SCENES if s.name == "tracking"][0]


def _doc() -> str:
    return io.open(DOC, encoding="utf-8").read()


def _f(x: str) -> float:
    """文档里用的是全角减号 U+2212，float() 不认。"""
    return float(x.replace("\u2212", "-"))


# --------------------------------------------------------------------------
# 场景读数
# --------------------------------------------------------------------------
_CACHE = {}


def readout(controller: str, model_err: float, kp: float = 400.0,
            align: str = "off") -> dict:
    """跑一次场景，返回 telemetry。同一行表格的所有数字都来自这一次调用。"""
    key = (controller, model_err, kp, align)
    if key in _CACHE:
        return _CACHE[key]
    sc = _TRACKING()
    sc.build()
    sc.set("controller", controller)
    sc.set("model_err", model_err)
    sc.set("kp", kp)
    sc.set("align", align)
    sc.reset()
    d, r = sc.data, sc.robot
    q = d.qpos[r.qpos_idx][:7]
    v = d.qvel[r.qvel_idx][:7]
    sc.control(0.0, q, v)
    _CACHE[key] = sc.telemetry(0.0, q, v)
    return _CACHE[key]


class TestTheModalReadoutIsMeasuredNotAsserted(unittest.TestCase):
    """回归守卫：这个读数必须是算出来的，不能是填出来的。"""

    def test_model_error_moves_the_ctc_modal_spread(self):
        """⭐ 就是这一条抓住了原来的 bug：拖满模型误差，读数纹丝不动。"""
        base = readout("ctc", 0.0)["modal_spread"]
        worst = readout("ctc", 60.0)["modal_spread"]
        self.assertGreater(
            worst - base, 0.05,
            "模型误差 0→60% 必须显著改变 CTC 的模态离散度；"
            "如果没变，说明这个读数又被写死成常数了")

    def test_model_error_slows_the_ctc_slowest_mode_monotonically(self):
        got = [readout("ctc", e)["modal_min"] for e in (0.0, 10.0, 30.0, 60.0)]
        for a, b in zip(got, got[1:]):
            self.assertLess(b, a, f"模型越错最慢模态应该越慢，实测 {got}")

    def test_model_error_does_not_move_the_pd_modal_readout(self):
        """⭐⭐ 反向守卫：PD 的模型只进重力补偿，进不了齐次方程。

        少了这一条，把 CTC 那行换成任何随模型误差乱跳的量都能过测试。
        """
        base = readout("pd_gravity", 0.0)
        worst = readout("pd_gravity", 60.0)
        self.assertAlmostEqual(base["modal_min"], worst["modal_min"], places=6)
        self.assertAlmostEqual(base["modal_spread"], worst["modal_spread"],
                               places=6)

    def test_an_exact_model_gives_sqrt_kp_and_a_spread_just_above_one(self):
        """⭐⭐ 「差一点点不等于 1」正是"真的在量"的签名。

        写死的 ``np.full(n, sqrt(kp))`` 会让离散度**精确**等于 1.0。
        真去解 det(Kp·M_reg − ω²·M_true)=0 就不会：regressor 模型和
        MuJoCo 之间本来就有 ‖ΔM‖≈4.8e-5 的差距，它会漏进读数里，
        给出 1.0000225。这个残差和 §6.2b 在**另一条路径**上量到的
        5.1e-4 是同一个来源。
        """
        spreads = []
        for kp in (100.0, 400.0, 900.0):
            tel = readout("ctc", 0.0, kp=kp)
            self.assertAlmostEqual(tel["modal_min"], np.sqrt(kp), delta=1e-3)
            self.assertAlmostEqual(tel["modal_spread"], 1.0, delta=1e-3)
            spreads.append(tel["modal_spread"])
            self.assertNotAlmostEqual(
                tel["modal_spread"], 1.0, places=6,
                msg="离散度精确等于 1.0，说明它又变回写死的常数了")
        self.assertLess(max(spreads) - min(spreads), 1e-9,
                        f"离散度应当与 Kp 无关（理论性质），实测 {spreads}")

    def test_the_ctc_branch_solves_a_different_equation_than_pd(self):
        """CTC 用 det(Kp·M_reg − ω²·M_true)，PD 用 det(Kp − ω²·M_true)。

        独立复算一遍（不读场景的中间量，只用公开的 robot / regressor）。
        """
        sc = _TRACKING()
        sc.build()
        sc.set("controller", "ctc")
        sc.set("model_err", 60.0)
        sc.set("kp", 400.0)
        sc.set("align", "off")
        sc.reset()
        Ms = sc.robot.mass_matrix(Q_HOME)[np.ix_(WN, WN)]
        pd_like = modal_frequencies(Ms, 400.0)
        got = readout("ctc", 60.0)["modal_min"]
        self.assertGreater(
            abs(got - pd_like[0]), 1.0,
            "CTC 的模态不应该等于 PD 的模态——它多了一个 M_reg")


@requires_docs
class TestTheDocModalTablesAreReproducible(unittest.TestCase):
    """文档里的每张模态表都必须能被重新跑出来，且**整行来自同一次实验**。"""

    def test_the_model_error_sweep_table_matches(self):
        rows = re.findall(
            r"^\|\s*计算力矩\s*\|\s*(\d+)%\s*\|\s*([\d.\u2212-]+)\s*rad/s\s*\|"
            r"\s*\*{0,2}([\d.]+)\*{0,2}\s*\|",
            _doc(), re.M)
        self.assertGreaterEqual(len(rows), 4, "§6.2d 的模型误差扫描表没解析到")
        for err, mn, sp in rows:
            tel = readout("ctc", float(err))          # ⭐ 一次调用供整行
            self.assertAlmostEqual(tel["modal_min"], _f(mn), delta=2e-3,
                                   msg=f"模型误差 {err}% 的最慢模态对不上")
            self.assertAlmostEqual(tel["modal_spread"], _f(sp), delta=5e-4,
                                   msg=f"模型误差 {err}% 的离散度对不上")

    def test_the_pd_row_of_the_sweep_table_matches(self):
        m = re.search(
            r"^\|\s*PD\+重力补偿\s*\|[^|]*\|\s*([\d.]+)\s*rad/s[^|]*\|"
            r"\s*([\d.]+)[^|]*\|",
            _doc(), re.M)
        self.assertIsNotNone(m, "§6.2d 的 PD 行没解析到")
        tel = readout("pd_gravity", 0.0)
        self.assertAlmostEqual(tel["modal_min"], _f(m.group(1)), delta=2e-3)
        self.assertAlmostEqual(tel["modal_spread"], _f(m.group(2)), delta=5e-4)

    def test_the_headline_modal_table_matches(self):
        rows = re.findall(
            r"^\|\s*(PD\+重力补偿|计算力矩)\s*\|[^|]*\|"
            r"\s*\*{0,2}([\d.]+)\s*rad/s\*{0,2}\s*\|"
            r"\s*\*{0,2}([\d.]+)×\*{0,2}\s*\|",
            _doc(), re.M)
        self.assertEqual(len(rows), 2, "§6.3 的模态表没解析到两行")
        for name, mn, sp in rows:
            tel = readout("ctc" if name == "计算力矩" else "pd_gravity", 0.0)
            self.assertAlmostEqual(tel["modal_min"], _f(mn), delta=6e-3,
                                   msg=f"{name} 的最慢模态对不上")
            self.assertAlmostEqual(tel["modal_spread"], _f(sp), delta=6e-3,
                                   msg=f"{name} 的离散度对不上")

    def test_the_doc_states_that_the_ctc_row_used_to_be_hardcoded(self):
        """撤回声明必须留在文档里，否则读者会以为这个数一直是测的。"""
        d = _doc()
        self.assertIn("np.full", d, "§6.2d 应当贴出修复前那行代码")
        self.assertIn("修复前它是**假设**，修复后它是**测量**", d)

    def test_every_stated_residual_matches_the_measurement(self):
        """⭐ 元教训 #33 第四次：不能用 assertIn 查这个数字。

        `1.0000225` 在文中出现 **两次**（正文一次、旁注一次）。
        只做子串检查的话，改坏其中一处，另一处会替它顶包，测试照过。
        正确做法：把**每一处**都解析出来，逐个和当场实测的数比。
        """
        d = _doc()
        # (?!0) 把"假如写死会是 1.000000000"那句排除掉——它讲的是另一个数。
        stated = re.findall(r"1\.0000(?!0)\d+", d)
        self.assertGreaterEqual(
            len(stated), 3,
            "残差在正文、旁注、连续性表里都出现过，应当至少解析到 3 处")
        got = readout("ctc", 0.0, kp=400.0)["modal_spread"]
        for i, v in enumerate(stated):
            decimals = len(v.split(".")[1])
            tol = 0.51 * 10.0 ** (-decimals)      # 按文档写出的精度定容差
            self.assertAlmostEqual(
                _f(v), got, delta=tol,
                msg=f"文档里第 {i + 1} 处残差 {v} 与实测 {got:.9f} 对不上")

    def test_the_measured_residual_matches_the_model_gap(self):
        """残差必须真的来自 ‖M_true − M_reg‖，不是随便一个小数。"""
        d = _doc()
        m = re.search(r"M_{true}-M_{reg}\\rVert_F = ([\d.]+)\\times10\^\{-(\d+)\}", d)
        self.assertIsNotNone(m, "§6.2d 应当写出 ‖ΔM‖ 的实测值")
        stated = float(m.group(1)) * 10.0 ** (-int(m.group(2)))
        sc = _TRACKING()
        sc.build()
        sc.set("controller", "ctc")
        sc.set("model_err", 0.0)
        sc.reset()
        gap = np.linalg.norm(
            sc.robot.mass_matrix(Q_HOME)[np.ix_(WN, WN)]
            - sc._reg_mass(Q_HOME, sc._pi_hat())[np.ix_(WN, WN)])
        self.assertAlmostEqual(gap, stated, delta=5e-8,
                               msg=f"文档写 {stated:.4e}，实测 {gap:.4e}")

    def test_the_doc_does_not_claim_unconditional_decoupling(self):
        """「CTC 离散度精确等于 1.00」不能再无条件地说。"""
        d = _doc()
        self.assertNotIn("CTC 的模态离散度精确等于 1.00", d)
        self.assertIn("模型准确时", d)


class TestFreeVibrationConfirmsTheModes(unittest.TestCase):
    """独立 oracle：不解特征值，改用「撒手让它振，数它振多快」。

    ⚠️ 关键纪律：**同一个实验要在多个窗长下各做一次**。
    「找局部极大值」会把重阻尼模态的宽包切成好几个假峰，
    切在哪里取决于窗有多长。真峰几乎不动，假峰跟着窗长跑。
    """

    WINDOWS = (4.0, 5.0, 6.0, 8.0, 10.0)
    CASES = (("ctc", 0.0), ("ctc", 30.0), ("ctc", 60.0), ("pd_gravity", 0.0))
    LABEL = {("ctc", 0.0): "CTC 误差 0%", ("ctc", 30.0): "CTC 误差 30%",
             ("ctc", 60.0): "CTC 误差 60%", ("pd_gravity", 0.0): "PD 误差 0%"}

    @classmethod
    def setUpClass(cls):
        cls.peaks = {c: [cls._peaks(*cls._free_response(*c, seconds=t))
                         for t in cls.WINDOWS]
                     for c in cls.CASES}

    # ---------------- 仿真与频谱 ----------------
    @staticmethod
    def _free_response(controller, model_err, kp=400.0, kd=2.0,
                       delta=0.02, seconds=5.0):
        """冻住参考 → 推开关节 → 撒手，返回真实闭环的误差时间序列。"""
        sc = _TRACKING()
        sc.build()
        sc.set("controller", controller)
        sc.set("model_err", model_err)
        sc.set("kp", kp)
        sc.set("kd", kd)          # 故意取小，阻尼比 ~0.05，振动才看得清
        sc.set("amp", 0.0)        # 参考冻结在 Q_HOME
        sc.set("align", "off")
        sc.reset()
        d, r, m = sc.data, sc.robot, sc.model
        q0 = d.qpos[r.qpos_idx][:7].copy()
        for j in WN:
            d.qpos[r.qpos_idx[j]] += delta
        mujoco.mj_forward(m, d)
        sub = max(1, int(round(sc.dt / m.opt.timestep)))
        rows = []
        for k in range(int(seconds / sc.dt)):
            q = d.qpos[r.qpos_idx][:7]
            v = d.qvel[r.qvel_idx][:7]
            rows.append((q0 - q)[list(WN)].copy())
            d.ctrl[r.arm_actuator_ids] = sc.control(k * sc.dt, q, v)
            for _ in range(sub):
                mujoco.mj_step(m, d)
        return np.array(rows), sc.dt

    @staticmethod
    def _peaks(series, dt, floor=0.02):
        """零填充 FFT + 抛物线插值，返回显著峰的角频率（rad/s），按频率升序。"""
        x = series - series.mean(0)
        nfft = 1 << 16
        power = (np.abs(np.fft.rfft(x, nfft, axis=0)) ** 2).sum(1)
        freq = np.fft.rfftfreq(nfft, dt)
        df = freq[1] - freq[0]
        found = []
        for i in range(2, len(power) - 1):
            if (power[i] > power[i - 1] and power[i] >= power[i + 1]
                    and power[i] > floor * power.max()):
                a, b, c = power[i - 1], power[i], power[i + 1]
                shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
                found.append(2 * np.pi * (freq[i] + shift * df))
        return sorted(found)

    def _stable(self, case, rel=0.02, max_spread=0.2):
        """在所有窗长里都出现、且漂移 < max_spread 的峰 → [(频率, 极差)]。"""
        per_window = self.peaks[case]
        out = []
        for w in per_window[0]:
            hits = []
            for pk in per_window[1:]:
                near = [x for x in pk if abs(x - w) / w < rel]
                if not near:
                    break
                hits.append(min(near, key=lambda x: abs(x - w)))
            if len(hits) == len(per_window) - 1:
                vals = [w] + hits
                spread = max(vals) - min(vals)
                if spread < max_spread:
                    out.append((float(np.mean(vals)), float(spread)))
        return sorted(out)

    def _predicted(self, controller, model_err, kp=400.0):
        sc = _TRACKING()
        sc.build()
        sc.set("controller", controller)
        sc.set("model_err", model_err)
        sc.set("kp", kp)
        sc.set("align", "off")
        sc.reset()
        Ms = sc.robot.mass_matrix(Q_HOME)[np.ix_(WN, WN)]
        if controller == "pd_gravity":
            return modal_frequencies(Ms, kp)
        Mr = sc._reg_mass(Q_HOME, sc._pi_hat())[np.ix_(WN, WN)]
        return modal_frequencies(Ms, kp * Mr)

    # ---------------- 结论 ----------------
    def test_ctc_rings_at_exactly_one_frequency_in_every_window(self):
        """⭐⭐⭐ "解耦"最硬的证据：五个窗长都只数出 1 个峰。"""
        for err in (0.0, 30.0, 60.0):
            counts = [len(p) for p in self.peaks[("ctc", err)]]
            self.assertEqual(
                counts, [1] * len(self.WINDOWS),
                f"CTC 误差 {err}% 应当在每个窗长下都只有一个峰，实测 {counts}")

    def test_the_single_ctc_peak_does_not_move_with_the_window(self):
        """真峰不随窗长跑——这是"这个峰是真的"的判据。"""
        for err in (0.0, 30.0, 60.0):
            stable = self._stable(("ctc", err))
            self.assertEqual(len(stable), 1,
                             f"CTC 误差 {err}% 应当只有一个稳定峰")
            self.assertLess(stable[0][1], 0.1,
                            f"CTC 误差 {err}% 的峰漂移过大：{stable[0]}")

    def test_model_error_slows_the_measured_ringing(self):
        """⭐ 特征值说它会变慢，秒表也说它变慢了。"""
        got = [self._stable(("ctc", e))[0][0] for e in (0.0, 30.0, 60.0)]
        for a, b in zip(got, got[1:]):
            self.assertLess(b, a - 0.1,
                            f"模型越错实测振动应越慢，实测 {got}")

    def test_the_measured_peak_lies_inside_the_predicted_band(self):
        for err in (0.0, 30.0, 60.0):
            lo, hi = self._predicted("ctc", err)[[0, -1]]
            got = self._stable(("ctc", err))[0][0]
            self.assertGreater(got, lo - 1.5,
                               f"误差 {err}%：实测 {got:.2f} 低于预测带 {lo:.2f}")
            self.assertLess(got, hi + 1.5,
                            f"误差 {err}%：实测 {got:.2f} 高于预测带 {hi:.2f}")

    def test_pd_peak_count_is_window_dependent_so_most_of_them_are_fake(self):
        """⭐⭐ PD 数出的峰数随窗长变 ⇒ 那些高频"峰"是切碎的宽包，不是模态。"""
        counts = [len(p) for p in self.peaks[("pd_gravity", 0.0)]]
        self.assertGreater(
            max(counts) - min(counts), 2,
            f"PD 的峰数本该随窗长明显变化（重阻尼模态被切碎），实测 {counts}")
        self.assertGreater(max(counts), 5, f"实测 {counts}")

    def test_pd_has_two_genuine_low_frequency_modes(self):
        """PD 的 4.48 离散度不是算错——两个真峰摆在那里。"""
        stable = [f for f, _ in self._stable(("pd_gravity", 0.0)) if f < 30.0]
        self.assertEqual(len(stable), 2,
                         f"PD 低频段应有 2 个稳定峰，实测 {stable}")
        pred = self._predicted("pd_gravity", 0.0)
        for got, want in zip(stable, (pred[0], pred[2])):
            self.assertLess(abs(got - want) / want, 0.08,
                            f"实测 {got:.2f} 与预测 {want:.2f} 差得太多")
        self.assertGreater(stable[1] / stable[0], 1.8,
                           f"PD 的实测频率跨度应明显大于 1，实测 {stable}")

    @requires_docs
    def test_the_doc_rejected_peaks_are_really_unstable(self):
        """⭐⭐⭐ 文档说 48.50 / 51.50 是假峰、极差 0.76 / 0.60 —— 必须重测得到。

        没有这条，"哪些峰被淘汰、淘汰得有多干脆"就成了随口一说：
        文档可以把假峰的极差改成和真峰一样小（那样"换窗长"这招就失去说服力），
        而没有任何测试会红。元教训 #41 的判别力**本身**需要被守住。
        """
        d = _doc()
        m = re.search(
            r"PD 那一行里\s*([\d.]+)\s*和\s*([\d.]+)[^*]*\*\*"
            r"([\d.]+)\s*和\s*([\d.]+)\*\*", d)
        self.assertIsNotNone(m, "§6.2d 的假峰说明被删了或改了写法")
        f_lo, f_hi = float(m.group(1)), float(m.group(2))
        sp_lo, sp_hi = float(m.group(3)), float(m.group(4))

        allp = self._stable(("pd_gravity", 0.0), max_spread=1e9)
        got = {f: sp for f, sp in allp}
        for f_doc, sp_doc in ((f_lo, sp_lo), (f_hi, sp_hi)):
            near = [f for f in got if abs(f - f_doc) < 1.0]
            self.assertTrue(
                near, f"文档说 {f_doc} 每个窗长都出现，实测却没有：{sorted(got)}")
            sp = got[near[0]]
            self.assertGreaterEqual(
                sp, 0.2,
                f"{f_doc} 的五窗极差 {sp:.2f} < 0.2 —— 它其实是真峰，"
                f"文档不该把它当假峰淘汰")
            self.assertAlmostEqual(
                sp, sp_doc, delta=0.15,
                msg=f"文档说 {f_doc} 的极差是 {sp_doc}，实测 {sp:.2f}")

        real = [sp for f, sp in allp if sp < 0.2]
        self.assertTrue(real, "PD 一个真峰都没有？实验坏了")
        self.assertGreater(
            min(sp_lo, sp_hi) / max(max(real), 1e-9), 5.0,
            "假峰的极差没有比真峰大一个数量级 —— "
            "「换窗长验峰」这招在这组数据上其实分不开真假，文档不该那样写")

    @requires_docs
    def test_the_doc_free_vibration_table_is_reproducible(self):
        """⭐ 文档那张表必须能被重跑出来，且**整行来自同一组实验**。"""
        d = _doc()
        for case in self.CASES:
            label = self.LABEL[case]
            m = re.search(
                r"^\|\s*" + re.escape(label) + r"\s*\|([^|]*)\|([^|]*)\|"
                r"([^|]*)\|([^|]*)\|", d, re.M)
            self.assertIsNotNone(m, f"§6.2d 的表里找不到「{label}」这一行")
            want_counts = [int(x) for x in re.findall(r"\d+", m.group(2))]
            self.assertEqual([len(p) for p in self.peaks[case]], want_counts,
                             f"{label}：各窗长峰数与文档不符")
            stable = self._stable(case)
            want_f = [_f(x) for x in re.findall(r"[\d.]+", m.group(3))]
            want_s = [_f(x) for x in re.findall(r"[\d.]+", m.group(4))]
            self.assertEqual(len(stable), len(want_f),
                             f"{label}：稳定峰个数与文档不符")
            for (got_f, got_s), wf, ws in zip(stable, want_f, want_s):
                self.assertAlmostEqual(got_f, wf, delta=6e-3,
                                       msg=f"{label}：稳定峰位置对不上")
                self.assertAlmostEqual(got_s, ws, delta=6e-3,
                                       msg=f"{label}：五窗极差对不上")

    @requires_docs
    def test_the_doc_warns_that_peak_counting_is_window_dependent(self):
        """这个陷阱必须留在文档里——真机振动分析踩的是同一个坑。"""
        d = _doc()
        self.assertIn("换个窗长再看它动不动", d)
        self.assertIn("48.50", d, "被淘汰的假峰应当明写出来")
        self.assertIn("51.50", d)


if __name__ == "__main__":
    unittest.main()
