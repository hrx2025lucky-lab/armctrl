"""教学层（Param.help / effect、lesson 卡片、Markdown）与源码实现的一致性锁。

为什么单独立一个文件
--------------------
第五轮独立审核暴露了一个反复出现的缺陷类型：**同一件事在代码里是对的，
在界面/讲义里是错的**。四轮审核里它出现了五次（自适应符号、倒角预算、
f_meas 的信号、vib_amp 的信号、动量观测器延迟）。根因是：
Param.help / Param.effect / lesson 卡片这一层**从来没有任何测试覆盖**，
改了代码没人提醒去改文案。

本文件的原则
------------
1. **先用数值独立测出源码的真实行为**，再要求文案与之一致。
   纯字符串比对只能锁住「文案没被改」，锁不住「文案说的对不对」。
2. 每条断言的失败信息里写明**期望的实测值**，让后来人知道该改哪边。
3. 反向变异（把文案改回旧的错误说法）必须让测试失败。
"""

from __future__ import annotations

import inspect
import pathlib
import re
import unittest
from pathlib import Path

import numpy as np

from armctrl.tests._docs import requires_docs

REPO = Path(__file__).resolve().parents[2]

#: ⚠️ 2026-09-01：`armctrl` 被拆成独立仓库 `armctrl_project/`，
#: 但 `TARGET_岗位目标导向.md` 和 `面试问答/` 留在了原来的
#: `motion control/`（它们讲的是求职定位，不属于代码仓库）。
#: 这些测试要**跨仓库**核对文档，所以路径必须两处都找。
#:
#: ⭐ 拆仓那天这一整个文件（37 项防漂移守护）连同源码一起消失了，
#: 只剩一个过期的 .pyc —— discover 还能"发现"它，于是报的是
#: ImportError 而不是"少了 37 项测试"。**守护测试自己也会掉。**
_SIBLING = REPO.parent / "motion control"

#: ⭐ 讲义（`armctrl_讲解/`）不随公开仓库分发。本文件里几乎每条测试都要读它，
#: 所以整个文件都挂 `requires_docs`：缺讲义就整体 skip，而不是报一堆红。
#: ⚠️ 但「守护的守护」那一组是例外——它防的正是「测试文件本身消失」，
#: 必须在没有讲义时也照跑（见文件末尾）。


def _find(rel: str) -> Path:
    """先在本仓库找，找不到就去兄弟目录 `motion control/` 找。"""
    here = REPO / rel
    return here if here.exists() else (_SIBLING / rel)


def _need(rel: str) -> Path:
    """同 `_find`，但**找不到就跳过这条测试**，而不是报错。

    ⭐ 为什么要有这个：`TARGET_岗位目标导向.md` 和 `面试问答/` 是
    **求职私人材料**，不会进公开仓库。但本文件里大部分测试守的是
    `armctrl_讲解/`（它在仓库里），这部分对 clone 下来的人同样有价值。

    以前的做法是把整个文件 gitignore 掉——代价是**公开仓库一条文档守护都没有**，
    而且拆仓那天正是因为它没被跟踪才整个丢失（见 §批次 I）。
    改成「缺私人材料就跳过那几条」，两边都能拿到该拿的保护。
    """
    path = _find(rel)
    if not path.exists():
        raise unittest.SkipTest(
            f"需要 {rel}，它属于求职私人材料，不在公开仓库里 —— 跳过这条")
    return path


def _text(path: str) -> str:
    return _need(path).read_text(encoding="utf-8")


@requires_docs
class TestAdaptiveSignConvention(unittest.TestCase):
    """P0-01：自适应控制律的符号，必须和误差约定成对出现且与源码一致。"""

    def test_production_controller_uses_plus_signs(self):
        """独立数值判定：源码用的到底是加号还是减号。

        不读任何字符串，只看数值行为。构造 s > 0 的状态，
        看反馈力矩是否与 s 同号（加号约定），以及参数是否被更新。
        """
        import mujoco

        from armctrl.control.adaptive import SlotineLiAdaptiveController
        from armctrl.tuner.scenes import AdaptiveScene

        sc = AdaptiveScene()
        sc.build()
        ctrl = SlotineLiAdaptiveController(sc.reg, 7, lam=10.0, kd=60.0,
                                           gamma=0.5, use_base=True)
        ctrl.reset()

        nv = sc.reg.nv
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = np.zeros(nv)
        # e = q_des - q = 0，de = v_des - v > 0  ⇒  s = de > 0
        q_des, v_des, a_des = q.copy(), np.full(nv, 0.10), np.zeros(nv)

        tau, s = ctrl.compute(q, v, q_des, v_des, a_des, dt=1e-3)
        self.assertTrue(np.all(s > 0), f"构造失败：期望 s>0，实得 {s}")

        # β̂ 初值为 0 ⇒ Yb@β = 0 ⇒ tau 全部来自反馈项。
        # 加号约定：tau = +kd*s，与 s 同号。
        self.assertTrue(
            np.all(tau > 0),
            f"源码反馈项与 s 同号（即 τ = Yπ̂ + Kd·s）。实测 τ={tau}，s={s}。"
            "若此处变为异号，说明 adaptive.py 改成了 e = q - q_d 约定，"
            "则 scenes.py / lessons.py 的公式必须同步改成减号。",
        )
        self.assertGreater(
            float(np.linalg.norm(ctrl.beta)), 0.0,
            "一步之后参数应已被更新；为 0 说明更新律没生效。",
        )

    def test_source_law_is_plus_plus(self):
        """直接锁住 adaptive.py 的源码形态，防止被改成减号后文案又对上。"""
        from armctrl.control.adaptive import SlotineLiAdaptiveController

        src = inspect.getsource(SlotineLiAdaptiveController.compute)
        self.assertIn("e = q_des - q", src,
                      "误差约定应为 e = q_des - q；改了就必须同步改所有教学文案。")
        self.assertRegex(src, r"tau\s*=\s*Yb\s*@\s*self\.beta\s*\+\s*self\.kd\s*\*\s*s",
                         "控制律应为 τ = Yβ̂ + Kd·s（加号）。")
        self.assertRegex(src, r"self\.beta\s*\+=\s*self\.gamma\s*\*\s*\(\s*Yb\.T\s*@\s*s\s*\)",
                         "更新律应为 β̂ += Γ·Ybᵀs（加号）。")

    def test_gui_law_matches_source(self):
        """AdaptiveScene 的 ControlLaw 文案必须写加号，并写明 e = q_d − q。"""
        from armctrl.tuner.scenes import AdaptiveScene

        law = AdaptiveScene.law

        # ① 界面上直接显示的公式区：只允许出现加号写法
        self.assertIn("e = q_d − q", law.formula,
                      "公式区必须先声明误差约定，否则加减号无从判断。")
        self.assertIn("π̂̇ = +Γ·Yᵀ·s", law.formula,
                      "更新律应为加号（与 adaptive.py 的 β += γ·Ybᵀs 一致）。")
        self.assertIn("π̂ + Kd·s", law.formula,
                      "控制律应为加号（与 adaptive.py 的 τ = Yβ̂ + kd·s 一致）。")
        self.assertNotIn("π̂̇ = −Γ", law.formula,
                         "公式区不得出现 π̂̇ = −ΓYᵀs（与源码相反）。")

        # ② 框图方块同样只能是加号
        blocks = " ".join(b.label + " " + (b.detail or "") for b in law.blocks)
        self.assertIn("π̂̇ = +ΓYᵀs", blocks)
        self.assertNotIn("π̂̇ = −ΓYᵀs", blocks)
        self.assertNotIn("Y·π̂ − Kd·s", blocks)

        # ③ note 允许（并且必须）提到镜像约定，但要成对出现：
        #    提到减号时必须同时写出 e = q − q_d 这个前提。
        note = law.note or ""
        self.assertIn("e = q_d − q", note, "note 必须声明本场景的误差约定。")
        self.assertIn("e = q − q_d", note,
                      "note 必须交代镜像约定的存在，否则读者看到别处的减号会以为是错的。")
        self.assertIn("混着用才错", note)

    def test_lesson_law_matches_source(self):
        """自适应 lesson 卡片同样必须是加号，并声明约定。"""
        from armctrl.tuner import lessons

        blob = _lesson_blob(lessons, "adaptive")
        self.assertIn(r"+\,\Gamma\, Y^{\mathsf T} s", blob,
                      "lesson 的更新律应为 +ΓYᵀs。")
        self.assertNotIn(r"-\,\Gamma\, Y^{\mathsf T} s", blob,
                         "lesson 里不得再出现 −ΓYᵀs。")
        self.assertIn("e=q_d-q", blob.replace(" ", "").replace("$", ""),
                      "lesson 必须写明 e = q_d − q 这个前提。")

    def test_mirrored_convention_doc_is_left_alone(self):
        """面试问答/03 用镜像约定 e = q − q_d，其减号是对的，不许被「统一」掉。"""
        doc = _text("面试问答/03_二自由度机械臂_自适应控制.md")
        self.assertIn("q̃ = q − q_d", doc)
        self.assertIn("θ̂̇ = −Γ·Yᵀ·s", doc)
        self.assertIn("绝不能混用", doc,
                      "该篇必须保留「两套约定各自自洽但不能混用」的提示。")


@requires_docs
class TestObserverDetectionDelay(unittest.TestCase):
    """P1-01：`1/K_I` 是时间常数，不是检测延迟。"""

    def test_first_crossing_formula(self):
        """独立闭式：阶跃 u 下，残差首次越过阈值 T 的时刻。

        r(t) = u(1 − e^{−K_I t})  ⇒  t = −ln(1 − T/|u|)/K_I
        用直接数值积分一阶环节验证闭式，不调用任何场景。
        """
        for k_i in (15.0, 30.0, 60.0):
            for ratio in (0.1, 0.395, 0.632, 0.8):
                u, thr = 20.0, 20.0 * ratio
                t_closed = -np.log(1.0 - ratio) / k_i
                dt, r, t = 1e-5, 0.0, 0.0
                while r <= thr and t < 5.0:
                    r += k_i * (u - r) * dt
                    t += dt
                self.assertAlmostEqual(
                    t, t_closed, delta=2e-4,
                    msg=f"K_I={k_i}, T/|u|={ratio}: 数值 {t:.5f}s vs 闭式 {t_closed:.5f}s",
                )

    def test_delay_equals_time_constant_only_at_one_ratio(self):
        """只有 T/|u| = 1 − 1/e 时，检测延迟才恰好等于 1/K_I。"""
        k_i = 30.0
        tau_c = 1.0 / k_i
        special = 1.0 - 1.0 / np.e                      # ≈0.632
        self.assertAlmostEqual(-np.log(1 - special) / k_i, tau_c, places=9)
        # 场景默认工况 T/|u| ≈ 0.395（实测 |r|max≈20.2 N·m、阈值 8 N·m）
        default_ratio = 8.0 / 20.24
        t_default = -np.log(1 - default_ratio) / k_i
        self.assertLess(
            t_default, 0.6 * tau_c,
            f"默认工况下延迟应明显小于 1/K_I：闭式 {t_default*1e3:.1f} ms "
            f"vs 1/K_I = {tau_c*1e3:.1f} ms。文档若仍写「K_I=30 → 33 ms」即为错。",
        )
        self.assertAlmostEqual(t_default * 1e3, 16.8, delta=1.5)

    def test_delay_depends_on_force_and_threshold(self):
        """同一个 K_I，延迟随外力幅值和阈值大幅变化——所以它不是 K_I 的固有属性。"""
        k_i = 30.0
        delays = {}
        for u in (11.89, 20.24, 28.93, 38.76):          # 对应 25/40/60/80 N 实测残差
            delays[u] = -np.log(1 - 8.0 / u) / k_i * 1e3
        vals = list(delays.values())
        self.assertGreater(max(vals) / min(vals), 4.0,
                           f"同 K_I 下延迟应至少差 4 倍，实得 {delays}")
        # 阈值高于残差峰值时根本检测不到
        self.assertGreaterEqual(30.0 / 21.33, 1.0,
                                "阈值 30 N·m 高于残差峰值 ⇒ 永不检出，此时「延迟」无定义。")

    def test_docs_do_not_claim_delay_equals_one_over_ki(self):
        """全库不得再出现「检测延迟 = 1/K_I」或「K_I=30 → 33 ms」。"""
        from armctrl.tuner import lessons
        from armctrl.tuner.scenes import ObserverScene

        blobs = {
            "04_动量观测器.md": _text("armctrl_讲解/04_动量观测器.md"),
            "lessons.py": _lesson_blob(lessons, "observer"),
            "scenes.py(verify)": " ".join(ObserverScene.lesson.verify or []),
            "scenes.py(params)": _param_blob(ObserverScene),
        }
        for name, blob in blobs.items():
            flat = blob.replace(" ", "")
            self.assertNotIn("检测延迟≈1/K_I", flat, f"{name} 仍把 1/K_I 当检测延迟")
            self.assertNotIn("延迟≈1/K_I", flat, f"{name} 仍把 1/K_I 当检测延迟")
            self.assertNotIn("延迟≈$1/K_I$", flat, f"{name} 仍把 1/K_I 当检测延迟")
            self.assertNotIn("K_I=30对应约33ms", flat, f"{name} 仍写 33 ms（实测 16 ms）")
            self.assertNotIn("$K_I=30$对应约**33ms**", flat, f"{name} 仍写 33 ms")

    def test_observer_doc_states_the_real_formula(self):
        """讲义必须给出带 T/|u| 的闭式、对照表和实测值。"""
        doc = _text("armctrl_讲解/04_动量观测器.md")
        flat = re.sub(r"[\s\\]", "", doc)

        # ① 闭式必须原样在场（只删掉这一处就应当失败）
        self.assertIn("t_{text{detect}}=frac{-ln!bigl(1-T/|u|bigr)}{K_I}", flat,
                      "讲义必须写出 t_detect = -ln(1-T/|u|)/K_I 这条闭式。")
        # ② 必须点明「只有 T/|u|=1-1/e 时才等于 1/K_I」
        self.assertIn("1-1/mathrme", flat,
                      "必须说明只有 T/|u| = 1-1/e 时延迟才等于时间常数。")
        # ③ 必须给出默认工况的实测值，且明确否定 33 ms
        self.assertIn("16.0ms", flat, "应给出默认工况实测 16.0 ms。")
        self.assertIn("33.3ms", flat, "应把 1/K_I = 33.3 ms 并排列出来做对照。")
        # ④ 必须有「阈值一动延迟就变」的反例
        self.assertIn("44.0ms", flat, "应给出阈值 16 N·m 时的 44 ms 反例。")
        self.assertIn("永远检不到", flat, "应给出阈值高于残差峰时检不到的极端例子。")


@requires_docs
class TestPlanningBlendBudget(unittest.TestCase):
    """P1-08：倒角吃掉 60%、保留 40%，UI/lesson 曾经写反。"""

    def test_implementation_reserves_forty_percent(self):
        import inspect as _i
        from armctrl.tuner.scenes import PlanningScene

        src = _i.getsource(PlanningScene)
        self.assertIn('safety_margin=self.get("margin") * 0.4', src,
                      "实现应把倒角检查器的裕度降到 40%（即允许吃掉 60%）。")

    def test_ui_and_lesson_agree_with_implementation(self):
        from armctrl.tuner import lessons
        from armctrl.tuner.scenes import PlanningScene

        blobs = {
            "scenes.py": _explain_blob(PlanningScene) + _param_blob(PlanningScene),
            "lessons.py": _lesson_blob(lessons, "planning"),
            "06_路径规划.md": _text("armctrl_讲解/06_路径规划.md"),
        }
        for name, blob in blobs.items():
            flat = blob.replace(" ", "")
            if "吃掉规划裕度" not in flat:
                continue
            self.assertIn("吃掉规划裕度的60%", flat,
                          f"{name}：实现保留 40%，文案必须写「吃掉 60%」")
            self.assertNotIn("吃掉规划裕度的40%", flat,
                             f"{name}：写反了，实现是吃掉 60%")


@requires_docs
class TestServoReadoutLabels(unittest.TestCase):
    """P3：伺服场景 Param/Readout 说明与实际信号不符。"""

    def _servo_src(self):
        import inspect as _i
        from armctrl.tuner.scenes import ServoScene
        return _i.getsource(ServoScene)

    def test_fft_buffer_holds_spring_deflection(self):
        src = self._servo_src()
        self.assertIn("self._vel_buf[self._vel_i] = self.joint.theta_m - theta_l", src,
                      "FFT 缓冲存的是弹簧形变 δ = θ_m − θ_l。")

    def test_vib_amp_uses_motor_side_velocity(self):
        src = self._servo_src()
        self.assertIn("omega_m = self.joint.omega_m", src)
        self.assertRegex(src, r"hp\s*=\s*omega_m\s*-\s*self\._svf_lp",
                         "vib_amp 的带通输入是电机侧速度 ω_m。")

    def test_readout_help_matches_actual_signal(self):
        from armctrl.tuner.scenes import ServoScene

        helps = {r.key: (r.help or "") for r in ServoScene.readouts}
        self.assertIn("弹簧形变", helps["f_meas"],
                      "f_meas 是对弹簧形变 δ 做 FFT，不是对负载侧速度。")
        self.assertNotIn("负载侧速度", helps["f_meas"])
        self.assertIn("电机侧速度", helps["vib_amp"],
                      "vib_amp 取的是电机侧速度的谐振带通。")
        self.assertNotIn("负载侧速度", helps["vib_amp"])

    def test_probe_amp_help_covers_both_probe_modes(self):
        from armctrl.tuner.scenes import ServoScene

        src = self._servo_src()
        self.assertIn('("probe", "probe_cl")', src,
                      "注入噪声在 probe 与 probe_cl 两种模式下都生效。")
        p = {p.key: p for p in ServoScene.params}["probe_amp"]
        blob = (p.help or "") + (p.effect or "")
        self.assertIn("probe_cl", blob,
                      "probe_amp 的说明必须交代它对 probe_cl 也生效。")
        self.assertNotIn("只在「运动模式 = probe」时起作用", blob,
                         "旧文案声称只对 probe 生效，与代码的 (probe, probe_cl) 不符。")
        self.assertIn("两种模式下都起作用", blob,
                      "说明必须正面写出「两种模式下都起作用」。")

    def test_peak_search_floor_is_documented_as_such(self):
        """`_hp_fc` 已不参与峰值搜索，注释不得再称它为 FFT 下沿。"""
        src = self._servo_src()
        # 全类范围内都不允许再把 _hp_fc 说成 FFT 分析带下沿：
        # 峰值搜索现在用固定常数 PEAK_SEARCH_FLOOR_HZ。
        self.assertNotIn("FFT 分析带下沿", src,
                         "_hp_fc 已不参与峰值搜索，不得再称它为 FFT 分析带下沿。")
        self.assertIn("PEAK_SEARCH_FLOOR_HZ", src)
        i = src.rfind("self._hp_fc = float(np.sqrt")   # control() 里那一处
        self.assertGreater(i, 0)
        head = src[max(0, i - 500):i]
        self.assertIn("不参与任何判定", head,
                      "赋值处必须注明 _hp_fc 现在只是参考刻度、不参与峰值判定。")

    def test_servo_docstring_reduction_ratio_direction(self):
        from armctrl.control import servo

        doc = servo.TwoMassJoint.__doc__ or ""
        flat = doc.replace(" ", "")
        self.assertIn("乘以减速比的平方", flat,
                      "折算到关节（输出）侧是乘 n²；旧文案写「除过」与紧跟的公式自相矛盾。")
        self.assertNotIn("已经除过减速比的平方", flat)


@requires_docs
class TestZmpAnswer(unittest.TestCase):
    """Q-07：ZMP 处仍有竖直自由力矩，且 ZMP 本身是动力学概念而非准静态。"""

    DOC = "面试问答/07_逐际动力实习问答.md"

    def test_free_moment_is_stated(self):
        doc = _text(self.DOC)
        self.assertIn("自由力矩", doc,
                      "必须说明 ZMP 处仍剩一个绕竖直轴的自由力矩 τ_z。")
        self.assertIn("free moment", doc)
        self.assertIn("水平力矩消失", doc.replace(" ", ""),
                      "必须点明 ZMP 只让*水平*力矩为零。")
        self.assertNotIn("等效成\"作用在某一点上的一个力\"", doc,
                         "旧说法把合旋量简化成一个力，丢掉了自由力矩。")

    def test_not_called_quasi_static(self):
        doc = _text(self.DOC)
        i = doc.find("### Q6")
        j = doc.find("### Q7", i + 1)
        sec = doc[i:j if j > 0 else len(doc)]
        # ⚠️ 不能只搜「不是准静态」：表头「是不是准静态」去掉空格后也含这四个字，
        #    会造成假阳性（这条测试第一版就栽在这里）。必须锁住完整的那句断言。
        self.assertIn("ZMP 本身是完全动力学的概念，不是准静态的", sec,
                      "ZMP 是动力学概念，必须原样保留这句明确否定。")
        self.assertNotIn("它也是**准静态**的", sec,
                         "旧的错误说法「ZMP 是准静态的」不得复活。")
        self.assertIn("真正「准静态」的是另外两样东西", sec,
                      "必须把「准静态」的帽子交还给静稳定判据与 LIPM 近似。")
        self.assertIn("角动量变化率", sec,
                      "ZMP 方程含角动量变化率 L̇，必须写出来。")
        for token in ("L̇_y", "L̇_x", "LIPM"):
            self.assertIn(token, sec,
                          "必须给出含 L̇ 的完整 ZMP 表达式，并说明 LIPM 是额外近似。")

    def test_static_criterion_is_the_quasi_static_one(self):
        """把「准静态」的帽子扣回它该待的地方：质心投影判据。"""
        doc = _text(self.DOC)
        flat = doc.replace(" ", "")
        self.assertIn("静稳定判据", flat)
        self.assertIn("质心投影落在支撑多边形内", flat)


@requires_docs
class TestAdaptiveParameterScale(unittest.TestCase):
    """P3：自适应参数规模注释不能混口径。"""

    def test_docstring_states_model_for_each_number(self):
        from armctrl.control import adaptive

        doc = adaptive.__doc__ or ""
        self.assertNotIn("99 个，但其中只有 58 个可辨识", doc,
                         "99/58 是对不上任何一种工况的旧口径，不能当项目级数字。")
        for token in ("117", "70", "67", "62"):
            self.assertIn(token, doc,
                          "docstring 必须同时说明独立/联动/固定三种夹爪工况的秩。")

    def test_declared_ranks_are_reproducible(self):
        """独立 SVD 复核 docstring 里的 70/67/62，不调用生产的投影函数。"""
        from armctrl.tuner.scenes import AdaptiveScene

        sc = AdaptiveScene()
        sc.build()
        reg = sc.reg
        m = reg.reg.model
        lo = np.where(np.isfinite(m.lowerPositionLimit),
                      m.lowerPositionLimit, -np.pi)[: reg.n_arm]
        up = np.where(np.isfinite(m.upperPositionLimit),
                      m.upperPositionLimit, np.pi)[: reg.n_arm]
        f_lo = np.where(np.isfinite(m.lowerPositionLimit),
                        m.lowerPositionLimit, 0.0)[reg.n_arm:]
        f_up = np.where(np.isfinite(m.upperPositionLimit),
                        m.upperPositionLimit, 0.04)[reg.n_arm:]
        saved = reg.finger_q.copy()

        def rank_for(mode, samples=250, seed=0):
            rng = np.random.default_rng(seed)
            rows = []
            for _ in range(samples):
                if mode == "independent":
                    reg.finger_q = rng.uniform(f_lo, f_up)
                elif mode == "coupled":
                    reg.finger_q = np.full_like(saved, rng.uniform(f_lo[0], f_up[0]))
                else:
                    reg.finger_q = saved.copy()
                rows.append(reg.regressor(rng.uniform(lo, up),
                                          rng.uniform(-1.5, 1.5, reg.n_arm),
                                          rng.uniform(-3.0, 3.0, reg.n_arm)))
            reg.finger_q = saved
            s = np.linalg.svd(np.vstack(rows), compute_uv=False)
            r = int(np.sum(s > s[0] * 1e-8))
            return r, s

        for mode, want in (("independent", 70), ("coupled", 67), ("fixed", 62)):
            r, s = rank_for(mode)
            self.assertEqual(r, want, f"{mode} 工况的结构秩应为 {want}，实得 {r}")
            # 秩必须由真实断崖决定，不能是阈值凑出来的
            self.assertGreater(s[r - 1] / s[r], 1e6,
                               f"{mode}: 第 {r} 与第 {r+1} 个奇异值之间应有明显断崖")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _param_blob(scene_cls) -> str:
    out = []
    for p in getattr(scene_cls, "params", []):
        out += [p.help or "", p.effect or ""]
    for r in getattr(scene_cls, "readouts", []):
        out.append(r.help or "")
    return " ".join(out)


def _explain_blob(scene_cls) -> str:
    """把场景的 Lesson 各段 + ControlLaw 文案拼成一坨。"""
    parts = []
    les = getattr(scene_cls, "lesson", None)
    if les is not None:
        parts += [les.problem, les.idea, les.how, les.caveat]
        parts += list(les.watch) + list(les.verify)
    law = getattr(scene_cls, "law", None)
    if law is not None:
        parts += [law.formula, law.note or ""]
    return " ".join(p for p in parts if p)


def _lesson_blob(lessons_mod, scene_name: str) -> str:
    """把某个场景的所有 lesson 卡片文本拼成一坨，用于文案断言。"""
    out = []
    for step in lessons_mod.STEPS_BY_SCENE[scene_name]:
        out += [step.title or "", step.teach or "", getattr(step, "teach2", "") or ""]
        out += list(getattr(step, "math", None) or [])
        out += list(getattr(step, "watch", None) or [])
    return " ".join(out)


@requires_docs
class TestLongDocsHaveAReadingGuide(unittest.TestCase):
    """可读性：超长的讲解篇必须告诉读者「先读哪几节」。

    元教训 16 说过：**没有测试覆盖的那一层一定会烂**。
    「导读」正是这样一层——它不影响任何代码，所以最容易在后续
    编辑中被挤掉或者和正文脱节。这里给它装上防护。
    """

    LIMIT = 800          # 超过这么多行就必须有导读

    def _long_docs(self):
        for path in sorted((REPO / "armctrl_讲解").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            if text.count("\n") >= self.LIMIT:
                yield path.name, text

    def test_every_long_doc_tells_the_reader_where_to_start(self):
        found = list(self._long_docs())
        self.assertGreaterEqual(len(found), 2, "应该至少有两篇超长文档")
        for name, text in found:
            with self.subTest(doc=name):
                if name == "00_总索引.md":
                    # 总索引本身就是导读，它的入口是「建议按这个顺序」
                    self.assertIn("建议按这个顺序", text)
                    continue
                self.assertRegex(
                    text, r"别从头啃|这一篇怎么读",
                    f"{name} 有 {text.count(chr(10))} 行却没有导读，"
                    f"读者不知道该从哪读起")
                self.assertIn("第一遍", text, f"{name} 的导读缺少分轮次的读法")
                self.assertRegex(
                    text, r"跳过",
                    f"{name} 的导读没说哪些可以跳过——"
                    f"只说读什么不说不读什么，等于没减负")

    def test_the_reading_guide_points_at_sections_that_exist(self):
        """导读里提到的章节号，正文里必须真有。"""
        heads = re.compile(r"^#{2,3}\s*(§?[一二三四五六七八九十]+[、之]?)",
                           re.M)
        for name, text in self._long_docs():
            if name == "00_总索引.md" or "第一遍" not in text:
                continue          # 缺导读由上一条测试报错，这里不重复报
            titles = "".join(heads.findall(text))
            guide = text.split("第一遍", 1)[1].split("---", 1)[0]
            for ref in set(re.findall(r"§([一二三四五六七八九十]+)", guide)):
                with self.subTest(doc=name, section=ref):
                    self.assertIn(
                        ref, titles,
                        f"{name} 的导读指向 §{ref}，正文里却没有这一节")


@requires_docs
class TestTheJDCoverageCountIsActuallyCounted(unittest.TestCase):
    """岗位 B 工业方向的覆盖度，必须是「数出来的」而不是「记出来的」。

    2026-08-31 一次复查里发现三处数错：总索引写「9/12」（实际 8），
    TARGET 写「11 项命中，4 项空白」（实际 13/1/3），而且两张表
    **都漏掉了 JD 里逐字点名的「工艺」**。
    这条测试的独立依据是 TARGET 里引用的 **JD 原文本身**——
    从原文切出条目清单，再回头查每一条在覆盖表里有没有交代。
    """

    TARGET = _find("TARGET_岗位目标导向.md")   # 缺失时由 setUpClass 跳过
    INDEX = REPO / "armctrl_讲解" / "00_总索引.md"

    @staticmethod
    def _items(text, head, tail):
        m = re.search(re.escape(head) + r"：(.+?)" + tail, text, re.S)
        assert m, f"JD 原文里找不到「{head}」这一段"
        return [x for x in re.split(r"[、\n\s]+", m.group(1)) if x]

    def _jd_items(self):
        text = self.TARGET.read_text(encoding="utf-8")
        ctrl = self._items(text, "**运动控制算法**", r"等算法")
        plan = self._items(text, "**运动规划算法**", r"等\n")
        return ctrl, plan

    def test_the_jd_still_names_seventeen_algorithms(self):
        ctrl, plan = self._jd_items()
        self.assertEqual(len(ctrl), 12,
                         f"JD 运动控制算法应为 12 项，解析到 {len(ctrl)}：{ctrl}")
        self.assertEqual(len(plan), 5,
                         f"JD 运动规划算法应为 5 项，解析到 {len(plan)}：{plan}")

    def _coverage_region(self, doc):
        """只取「岗位 B 覆盖表」那一段——整篇搜索会被别处的同名词蒙混过去。"""
        text = doc.read_text(encoding="utf-8")
        if doc.name == "TARGET_岗位目标导向.md":
            start = "armctrl 对岗位 B 工业机器人方向的覆盖度"
            end = "JD 逐字点名"
        else:
            start = "### 6.5 岗位 B"
            end = "### 6.6"
        self.assertIn(start, text, f"{doc.name} 找不到覆盖表的起点")
        body = text.split(start, 1)[1]
        self.assertIn(end, body, f"{doc.name} 找不到覆盖表的终点")
        region = body.split(end, 1)[0]
        # 只认表格行：更正说明那段引文里也会提到条目名，会把漏算蒙混过去
        return "\n".join(ln for ln in region.splitlines()
                         if ln.lstrip().startswith("|"))

    def test_every_named_algorithm_is_accounted_for_in_both_docs(self):
        """JD 点名的每一项，两份覆盖表里都必须出现——漏掉就是漏算。"""
        ctrl, plan = self._jd_items()
        # 覆盖表里用的是简称，这里给出等价写法
        alias = {"动力学建模与辨识": ("动力学建模与辨识", "动力学建模辨识"),
                 "伺服环路控制": ("伺服环路控制", "伺服环路"),
                 "轨迹规划算法": ("轨迹规划",),
                 "路径规划算法": ("路径规划",),
                 "插补算法": ("插补",),
                 "时间最优规划": ("时间最优",),
                 "安全停机": ("安全停机",),
                 "故障诊断": ("故障诊断",)}
        for doc in (self.TARGET, self.INDEX):
            region = self._coverage_region(doc)
            for item in ctrl + plan:
                with self.subTest(doc=doc.name, item=item):
                    names = alias.get(item, (item,))
                    self.assertTrue(
                        any(n in region for n in names),
                        f"{doc.name} 的岗位 B 覆盖表里没有交代 JD 点名的「{item}」")

    def test_the_stale_miscounts_are_gone_everywhere(self):
        """旧的错数字不许留在任何一份对外文档里。"""
        stale = ["9 / 12", "11 项命中", "10/14 项", "4 项空白"]
        docs = [self.TARGET, self.INDEX]
        docs += sorted(_find("面试问答").glob("*.md"))   # 私人材料，没有就不加
        for doc in docs:
            text = doc.read_text(encoding="utf-8")
            for bad in stale:
                with self.subTest(doc=doc.name, stale=bad):
                    # 更正说明里可以引用旧值，但必须带「旧」字样在同一行
                    for line in text.splitlines():
                        if bad in line and not re.search(r"旧|更正|不可再", line):
                            self.fail(
                                f"{doc.name} 仍在用已知错误的覆盖度数字「{bad}」：\n  {line}")


@requires_docs
class TestTheScaleNumbersAreCurrent(unittest.TestCase):
    """⭐ 文档里的"规模数字"必须是**现在**跑出来的，不是上次记得的。

    ⚠️ 第五轮审核点名了这一类问题：行数、测试数、场景数、文档篇数
    散落在好几份 md 里，改完代码没人记得同步，于是越写越假。
    修法不是"下次记得改"，是**让它错了就红**。
    """

    @classmethod
    def setUpClass(cls):
        cls.target = _need("TARGET_岗位目标导向.md").read_text(encoding="utf-8")
        cls.index = (REPO / "armctrl_讲解" / "00_总索引.md").read_text(
            encoding="utf-8")

    @staticmethod
    def _py_lines():
        total = 0
        for f in (REPO / "armctrl").rglob("*.py"):
            total += len(f.read_text(encoding="utf-8").splitlines())
        return total

    @staticmethod
    def _test_files():
        return len(list((REPO / "armctrl" / "tests").glob("test_*.py")))

    @staticmethod
    def _test_count():
        loader = unittest.TestLoader()
        suite = loader.discover(str(REPO / "armctrl" / "tests"),
                                top_level_dir=str(REPO))
        n = 0
        stack = [suite]
        while stack:
            item = stack.pop()
            if isinstance(item, unittest.TestSuite):
                stack.extend(item)
            else:
                n += 1
        return n

    def test_the_line_count_is_current(self):
        want = self._py_lines()
        m = re.search(r"Python 代码 \*\*(\d+) 行\*\*", self.target)
        self.assertIsNotNone(m, "TARGET 里找不到行数，格式改了？")
        got = int(m.group(1))
        self.assertEqual(
            got, want,
            f"TARGET 写着 {got} 行，实际 {want} 行。"
            f"口径：cat $(find armctrl -name '*.py') | wc -l")

    def test_the_test_file_count_is_current(self):
        want = self._test_files()
        m = re.search(r"(\d+) 个测试文件", self.target)
        self.assertIsNotNone(m, "TARGET 里找不到测试文件数")
        self.assertEqual(int(m.group(1)), want,
                         f"TARGET 写着 {m.group(1)} 个测试文件，实际 {want} 个")

    def test_the_test_count_is_current_in_both_docs(self):
        """⭐ 测试总数在两份文档里都出现，两处都要对。

        ⚠️ 这里**不跑**测试，只用 unittest 的 loader 数一遍，
        所以它自己不会拖慢套件（也不会递归调用自己）。
        """
        want = self._test_count()
        m = re.search(r"\*\*(\d+) 项全过\*\*", self.target)
        self.assertIsNotNone(m, "TARGET 里找不到测试项数")
        self.assertEqual(
            int(m.group(1)), want,
            f"TARGET 写着 {m.group(1)} 项，loader 数出 {want} 项")
        # ⚠️ 锚点不能挂在某一个**历史批次**的句子上（原来挂的是「批次 E-1
        # 之后 N 项」）：那句话陈述的是那一批**当时**的数，后面每加一批测试
        # 就得回去把历史改成假的。改用一个专门的「当前全量」行做锚点。
        m2 = re.search(r"当前全量 (\d+) 项全过", self.index)
        self.assertIsNotNone(
            m2, "总索引里找不到「当前全量 N 项全过」这一行")
        self.assertEqual(
            int(m2.group(1)), want,
            f"总索引写着 {m2.group(1)} 项，loader 数出 {want} 项")
        m3 = re.search(r"\*\*(\d+) 项测试 \+ (\d+) 场景\*\*|"
                       r"\| \*\*仿真验证\*\* \| (\d+) 项测试 \+ (\d+) 场景",
                       self.index)
        self.assertIsNotNone(m3, "总索引的能力表里找不到测试数")
        got = int(next(g for g in m3.groups() if g))
        self.assertEqual(got, want,
                         f"总索引能力表写着 {got} 项，loader 数出 {want} 项")

    def test_the_scene_count_is_current(self):
        from armctrl.tuner.scenes import SCENES
        want = len(SCENES)
        for name, text in (("TARGET", self.target), ("总索引", self.index)):
            for m in re.finditer(r"(\d+) 个场景|(\d+) 场景", text):
                got = int(m.group(1) or m.group(2))
                self.assertEqual(
                    got, want,
                    f"{name} 里写着 {got} 个场景，实际注册了 {want} 个")


# ======================================================================
# 批次 F 验收合同：schema、入口、复现资料
#
# 审核第五轮 §11「批次 F」的三条验收标准，逐条做成测试：
#   ① 全 12 场景「声明的读数键」必须真的被 telemetry() 产出
#   ② 直接运行每个测试文件，收集数必须等于 discovery 的收集数
#   ③ 文档里引用的每个脚本必须真实存在
#
# 这三条以前只是"人肉检查过一次"，所以会漂。现在每次提交都会被重跑。
# ======================================================================


@requires_docs
class TestEveryDeclaredReadoutIsActuallyProduced(unittest.TestCase):
    """⭐ 面板上声明了一个读数，代码却从不产出它 —— 用户看到的是空白。

    这类 bug 静悄悄：没有异常、没有报错，只是那一栏永远不显示。
    因为读数是学习时唯一的观察窗口，空白栏等于"这节课没法上"。
    """

    def test_all_scenes_produce_every_key_they_declare(self):
        import mujoco
        from armctrl.tuner.scenes import SCENES

        self.assertGreaterEqual(len(SCENES), 12,
                                f"只注册了 {len(SCENES)} 个场景")
        broken = []
        for cls in SCENES:
            sc = cls()
            model = sc.build()
            sc.reset()
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(0.0, q, v)
            for _ in range(3):
                mujoco.mj_step(model, sc.data)
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            try:
                tel = sc.telemetry(0.0, q, v)
            except TypeError:
                tel = sc.telemetry()
            missing = sorted({r.key for r in sc.readouts} - set(tel))
            if missing:
                broken.append(f"{cls.__name__} 缺 {missing}")
        self.assertEqual(broken, [], "有场景声明了读数却不产出：\n" +
                         "\n".join(broken))

    def test_declared_sources_are_all_legal(self):
        """读数来源只能取 READOUT_SOURCES 里的值，否则前端分色会错。"""
        from armctrl.tuner.base import READOUT_SOURCES
        from armctrl.tuner.scenes import SCENES
        for cls in SCENES:
            for r in cls().readouts:
                self.assertIn(r.resolved_source(), READOUT_SOURCES,
                              f"{cls.__name__}.{r.key} 的来源非法")


@requires_docs
class TestRunningAFileDirectlyFindsTheSameTests(unittest.TestCase):
    """⭐ `unittest.main()` 写在文件中部 ⇒ 直接运行时只跑到那一行为止。

    实际踩过：`test_ilc_replan.py` 有一个 `unittest.main()` 卡在中间，
    `python -m unittest <file>` 收集到 39 项，discovery 收集到 43 项。
    **两条路数出来的数不一样，就说明有一条在骗你。**
    """

    def test_no_test_file_has_a_mid_file_unittest_main(self):
        root = pathlib.Path(__file__).resolve().parent
        offenders = []
        for f in sorted(root.glob("test_*.py")):
            lines = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "unittest.main()":
                    rest = [l for l in lines[i + 1:] if l.strip()]
                    if rest:
                        offenders.append(
                            f"{f.name}:{i + 1} 之后还有 {len(rest)} 行代码")
        self.assertEqual(offenders, [],
                         "unittest.main() 不在文件末尾，直接运行会漏测：\n" +
                         "\n".join(offenders))

    def test_each_file_collects_the_same_count_both_ways(self):
        """逐文件加载 vs 整目录 discovery，两边总数必须相等。"""
        root = pathlib.Path(__file__).resolve().parent
        loader = unittest.TestLoader()

        per_file = 0
        for f in sorted(root.glob("test_*.py")):
            per_file += loader.loadTestsFromName(
                f"armctrl.tests.{f.stem}").countTestCases()
        whole = loader.discover(
            str(root), top_level_dir=str(root.parent.parent)).countTestCases()
        self.assertEqual(per_file, whole,
                         f"逐文件收集 {per_file} 项，discovery 收集 {whole} 项")


@requires_docs
class TestEveryScriptTheDocsMentionExists(unittest.TestCase):
    """⭐ 文档里写「跑这个脚本复现」，脚本却不存在 —— 等于没法复现。

    验证脚本存在两个地方：仓库里，或者会话的 files/ 目录（不入库的产物）。
    两边都找不到，就是文档在开空头支票。
    """

    #: ⭐ 这些脚本原来只存在于「会话目录」（一个跟机器绑定的绝对路径），
    #: 于是 clone 下来的人读到文档里「见 probe_kin.py」会一头雾水，
    #: 而这条测试在别人机器上必然失败。现已全部收进仓库。
    SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[2] / "tools" / "verification"

    def test_referenced_verify_and_mutation_scripts_are_real(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        docs = sorted((root / "armctrl_讲解").glob("*.md"))
        docs.append(_find("TARGET_岗位目标导向.md"))   # 私人材料，可能不存在
        pat = re.compile(
            r"\b((?:verify|probe|mut|day\d)_[A-Za-z0-9_.\-]*\.py)")
        missing = []
        seen = set()
        for d in docs:
            if not d.exists():
                continue
            for name in pat.findall(d.read_text(encoding="utf-8")):
                if name in seen:
                    continue
                seen.add(name)
                if not ((root / name).exists()
                        or (self.SCRIPT_DIR / name).exists()):
                    missing.append(f"{name}（{d.name} 引用）")
        self.assertEqual(missing, [],
                         "文档引用的脚本不存在：\n" + "\n".join(missing))
        self.assertGreater(len(seen), 5,
                           f"只扫到 {len(seen)} 个脚本引用，正则大概率失效了")


# ==================================== 守护测试自己也会掉（2026-09-01 拆仓事故）


class TestTheGuardsThemselvesAreStillHere(unittest.TestCase):
    """⭐⭐ 防的是「守护测试文件被整个弄丢」这件事。

    2026-09-01 把 `armctrl` 拆成独立仓库时，**这个文件没被带过去**，
    只在 `__pycache__` 里剩了一个过期的 `.pyc`。
    后果特别隐蔽：`unittest discover` 报的是 ImportError，
    看起来像「某个模块坏了」，而**真实损失是 37 项防漂移守护全没了**。

    ⚠️ 这里有个鸡生蛋问题：**这个类自己也在这个文件里**，
    文件没了它也跟着没。所以它防不住「整个文件消失」，
    只能防住「文件还在、里面的守护被删空」。
    真正防「文件消失」的是最后那条：把**期望的测试文件清单**
    写进 TARGET，少一个就红 —— 而那份清单在**另一个仓库**里。
    """

    REQUIRED_CLASSES = (
        "TestTheScaleNumbersAreCurrent",
        "TestEveryScriptTheDocsMentionExists",
        "TestEveryDeclaredReadoutIsActuallyProduced",
        "TestRunningAFileDirectlyFindsTheSameTests",
        "TestTheJDCoverageCountIsActuallyCounted",
    )

    def test_every_required_guard_class_is_present(self):
        src = Path(__file__).read_text(encoding="utf-8")
        for name in self.REQUIRED_CLASSES:
            self.assertIn(
                f"class {name}", src,
                f"守护类 {name} 不见了 —— 是被删了还是拆仓时漏带了？")

    def test_no_stale_pyc_shadows_a_missing_test_module(self):
        """⭐ `__pycache__` 里有 .pyc、源码却没了 = 那个模块已经悄悄消失。

        这正是拆仓那天的现场：`test_teaching_consistency.cpython-310.pyc`
        还在，`test_teaching_consistency.py` 已经没了。
        """
        tests_dir = Path(__file__).resolve().parent
        cache = tests_dir / "__pycache__"
        if not cache.is_dir():
            self.skipTest("没有 __pycache__，无从检查")
        orphans = []
        for pyc in cache.glob("test_*.cpython-*.pyc"):
            stem = pyc.name.split(".")[0]
            if not (tests_dir / f"{stem}.py").exists():
                orphans.append(stem)
        self.assertEqual(
            orphans, [],
            f"这些测试模块只剩 .pyc、源码已经没了：{orphans}。"
            f"discover 会报 ImportError，掩盖掉「少了一批测试」这个真实损失")

    def test_the_test_file_inventory_in_target_is_current(self):
        """⭐ 把「有哪些测试文件」写进 TARGET，少一个就红。

        ⚠️ 这条的清单存在**另一个仓库**（`motion control/`），
        所以即使本仓库整个 `tests/` 被删光，它仍然能报警。
        """
        want = sorted(f.name for f in
                      (REPO / "armctrl" / "tests").glob("test_*.py"))
        target = _need("TARGET_岗位目标导向.md").read_text(encoding="utf-8")
        m = re.search(r"测试文件清单（自动核对）：`([^`]+)`", target)
        self.assertIsNotNone(
            m, "TARGET 里找不到「测试文件清单（自动核对）」那一行")
        got = sorted(x.strip() for x in m.group(1).split(",") if x.strip())
        self.assertEqual(
            got, want,
            f"测试文件清单对不上。\n  少了：{sorted(set(want) - set(got))}"
            f"\n  多了：{sorted(set(got) - set(want))}")



if __name__ == "__main__":
    unittest.main()
