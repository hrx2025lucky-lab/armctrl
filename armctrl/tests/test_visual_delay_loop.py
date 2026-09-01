"""视觉伺服闭环延迟拓扑的独立验证（第四轮审核 P1-06）。

被验证的命题：**延迟真的进了反馈回路**，而不是只延迟了参考。

独立 oracle：一个不含 MuJoCo、不含项目代码的纯数值积分

    sp(k+1) = sp(k) + λ·Δt·(p − sp(k−d)),   d = τ/Δt

它的连续极限 ṡ = λ(p − s(t−τ)) 的稳定边界是 **λτ = π/2**
（把 s = e^{jωt} 代入得 jω = −λ e^{−jωτ}，取模得 ω=λ，
 取相角得 ωτ = π/2 ⇒ λτ = π/2）。
"""

import unittest

import numpy as np


PI_OVER_2 = float(np.pi / 2.0)


def delayed_loop_ptp(lam, tau, dt=2.0e-3, secs=20.0, target=1.0):
    """独立 oracle：把延迟反馈积分器直接跑一遍，返回末段峰峰值。

    ⭐ 全程只用 numpy，不 import 任何 armctrl 模块——
    这样它就不可能"用被测代码的中间量证明被测代码"。
    """
    d = int(round(tau / dt))
    n = int(secs / dt)
    hist = [0.0] * (d + 1)
    sp = 0.0
    rec = np.empty(n)
    for k in range(n):
        sp = sp + lam * dt * (target - hist[0])
        hist.pop(0)
        hist.append(sp)
        rec[k] = sp
        if not np.isfinite(sp) or abs(sp) > 1e30:
            return float("inf")
    return float(np.ptp(rec[n // 2:]))


class TestDelayedLoopOracle(unittest.TestCase):
    """oracle 自身必须先站得住：它得能分辨稳定与不稳定。"""

    def test_converges_well_below_the_boundary(self):
        for lt in (0.5, 1.0, 1.4):
            with self.subTest(lam_tau=lt):
                ptp = delayed_loop_ptp(lt / 0.05, 0.05)
                self.assertLess(
                    ptp, 1e-3,
                    f"λτ={lt}（< π/2={PI_OVER_2:.4f}）本应收敛，"
                    f"末段峰峰却有 {ptp:.4g}")

    def test_diverges_just_above_the_boundary(self):
        for lt in (1.6, 1.7, 2.0):
            with self.subTest(lam_tau=lt):
                ptp = delayed_loop_ptp(lt / 0.05, 0.05)
                self.assertGreater(
                    ptp, 1e3,
                    f"λτ={lt}（> π/2={PI_OVER_2:.4f}）本应发散，"
                    f"末段峰峰只有 {ptp:.4g}")

    def test_boundary_is_pi_over_2_to_one_percent(self):
        """⭐ 二分找边界，必须落在 π/2 的 1% 以内。"""
        tau = 0.05
        lo, hi = 1.0, 3.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if delayed_loop_ptp(mid / tau, tau) > 1e2:
                hi = mid
            else:
                lo = mid
        boundary = 0.5 * (lo + hi)
        rel = abs(boundary - PI_OVER_2) / PI_OVER_2
        self.assertLess(
            rel, 0.01,
            f"实测边界 λτ={boundary:.4f}，理论 π/2={PI_OVER_2:.4f}，"
            f"差 {rel:.2%}（应 <1%）")

    def test_boundary_scales_with_tau_not_with_lambda_alone(self):
        """⭐ 变异证明：边界是 λ·τ 的**乘积**，不是 λ 单独说了算。

        若某个实现把延迟漏在环外，λ 一大照样稳，这条就会失效。
        """
        # 同一个 λτ、不同的 (λ, τ) 组合，结论必须一致
        for lt, pairs in ((1.0, [(20.0, 0.05), (10.0, 0.10), (5.0, 0.20)]),
                          (2.0, [(40.0, 0.05), (20.0, 0.10), (10.0, 0.20)])):
            with self.subTest(lam_tau=lt):
                verdicts = [delayed_loop_ptp(l, t) > 1e3 for l, t in pairs]
                self.assertEqual(
                    len(set(verdicts)), 1,
                    f"λτ={lt} 的三种 (λ,τ) 组合结论不一致：{verdicts}，"
                    "说明稳定性不是只由乘积 λτ 决定")


class TestSceneFeedbackIsActuallyDelayed(unittest.TestCase):
    """⭐ P1-06 本体：反馈端必须取「相机那一帧时刻」的**真实末端位置**。"""

    def test_source_uses_delayed_actual_tcp(self):
        import inspect
        from armctrl.tuner.scenes import GraspScene
        src = inspect.getsource(GraspScene.control)
        self.assertIn(
            "self._tcp_delayed(t - age)", src,
            "反馈端没有取延迟的真实末端位置——延迟只作用在参考上，"
            "λτ 失稳在构造上就演示不出来")
        self.assertNotIn(
            "self.servo.step(self._sp, p_des", src,
            "还在用当拍设定点当反馈")
        self.assertNotIn(
            "self._sp_delayed", src,
            "还在用**内部设定点**历史当反馈：那会把机械臂整个摘出环路，"
            "实测 |sp-TCP| 达 82 mm RMS，得到的边界只是软件积分器的性质")

    def test_tcp_delayed_returns_an_older_value(self):
        from armctrl.tuner.scenes import GraspScene
        sc = GraspScene.__new__(GraspScene)
        from collections import deque
        sc._tcp_hist = deque(maxlen=GraspScene.SP_HIST_MAX)
        sc._p = np.array([9.0, 9.0, 9.0])
        for k in range(100):
            sc._tcp_hist.append((k * 0.002, np.array([float(k), 0.0, 0.0])))
        got = sc._tcp_delayed(0.198 - 0.050)   # 回看 50 ms = 25 拍
        self.assertAlmostEqual(
            got[0], 74.0, places=6,
            msg=f"回看 50 ms 应取到第 74 拍的值，实际取到 {got[0]}")

    def test_tcp_delayed_falls_back_when_history_is_short(self):
        from armctrl.tuner.scenes import GraspScene
        from collections import deque
        sc = GraspScene.__new__(GraspScene)
        sc._tcp_hist = deque(maxlen=GraspScene.SP_HIST_MAX)
        sc._p = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(sc._tcp_delayed(-5.0), [1.0, 2.0, 3.0]),
                        "历史为空时应退化成当前值，而不是抛异常")
        sc._tcp_hist.append((10.0, np.array([7.0, 0.0, 0.0])))
        self.assertAlmostEqual(sc._tcp_delayed(0.0)[0], 7.0,
                               msg="历史不够早时应退化成最早的一条")


class TestSceneOscillationGrowsWithLatency(unittest.TestCase):
    """⭐ 端到端判别实验：固定 λ，只加大 τ，**真实末端**的振荡必须显著变大。

    ⚠️ 这条测试第五轮被改过两处，两处都是「量错了对象/窗口」：

    1. 原来量的是 ``sc._sp``——**内部设定点**。P1-06 的整个要点就是
       设定点不是物理量（实测它和真实末端差 82 mm RMS）。现在量 ``sc._p``。
    2. 原来只跑 8 s 且不锁阶段，样本里绝大部分是**行进段**，
       行进位移把振荡淹没了。现在锁在对准阶段跑 20 s，并丢掉前 40%。

    修好之后实测（λ=8，相机 120 Hz）：

        量 真实末端：τ=0.02 → 0.058 mm，τ=0.20 → 0.906 mm（**15.7×**）
        量 内部设定点：0.530 → 2.874 mm（5.4×，被牵引绳污染）
        λ=15 量设定点时甚至是 0.91×（顶到饱和，完全失去分辨力）
    """

    LAM = 8.0

    @staticmethod
    def _osc(lam, lat, secs=20.0, win=51):
        import mujoco
        from armctrl.tuner.scenes import GraspScene
        sc = GraspScene()
        model = sc.build()
        sc.set("vs_gain", lam)
        sc.set("vs_latency", lat)
        sc.set("vs_rate", 120.0)
        sc.set("closed_loop", "visual")
        sc.reset()
        n_sub = max(1, int(round(sc.dt / model.opt.timestep)))
        t = 0.0
        tcp = []
        for _ in range(int(secs / sc.dt)):
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
            sc._phase = sc.PH_ALIGN            # 锁住阶段，避免"退出条件"污染判据
            sc._align_hold = 0
            for _ in range(n_sub):
                mujoco.mj_step(model, sc.data)
            t += sc.dt
            tcp.append(sc._p.copy())           # ⭐ 真实末端，不是设定点
        if len(tcp) < 3 * win:
            return None
        a = np.array(tcp)[int(len(tcp) * 0.4):]   # 丢掉行进段
        ker = np.ones(win) / win
        trend = np.stack([np.convolve(a[:, i], ker, mode="valid")
                          for i in range(3)], axis=1)
        resid = a[win // 2:win // 2 + len(trend)] - trend
        return float(np.max(np.abs(resid))) * 1e3

    def test_latency_actually_reaches_the_loop(self):
        lo = self._osc(self.LAM, 0.02)
        hi = self._osc(self.LAM, 0.20)
        self.assertIsNotNone(lo, "短延迟工况没能进入对准阶段")
        self.assertIsNotNone(hi, "长延迟工况没能进入对准阶段")
        ratio = hi / max(lo, 1e-9)
        self.assertGreater(
            ratio, 4.0,
            f"τ 从 0.02 加到 0.20 s，真实末端振荡只从 {lo:.3f} 变到 {hi:.3f} mm "
            f"（{ratio:.2f}×，修好后实测 15.7×）。"
            "比值掉到 1 附近说明延迟没有真正进入反馈回路")


class TestCameraRateIsNotQuantizedAway(unittest.TestCase):
    """⭐ P1-06 附带 bug：滑条写 120 Hz，环里实际只跑 100 Hz。

    根因：旧代码 ``if t - _last < period: return; _last = t``——
    每帧都把基准重新对齐到**实际**采样时刻，而采样只能落在 2 ms 控制栅格上，
    于是 8.333 ms 的周期被系统性地撑成 10 ms。

    修复前实测：120→100.0 Hz(−16.7%)、90→83.3、60→55.6、30→29.4 Hz。
    修复后实测：119.99 / 89.97 / 59.99 / 29.99 Hz，帧间隔在 8～10 ms 抖动。
    """

    DT = 0.002

    @staticmethod
    def _sample_times(rate_hz, dt, T=2.0):
        """只驱动采样判据本身，不需要 MuJoCo。"""
        from armctrl.control.visual_servo import ObjectPoseSensor
        sc = ObjectPoseSensor.__new__(ObjectPoseSensor)
        sc._last_sample_t = -1e9
        sc._next_sample_t = -1e9
        period = 1.0 / rate_hz
        ts, t = [], 0.0
        while t < T:
            if sc._next_sample_t < -1e8:
                sc._next_sample_t = t
            if t >= sc._next_sample_t:
                sc._last_sample_t = t
                sc._next_sample_t += period
                if sc._next_sample_t <= t:
                    sc._next_sample_t = t + period
                ts.append(t)
            t = round(t + dt, 9)
        return np.asarray(ts)

    def test_effective_rate_matches_the_slider(self):
        for hz in (120.0, 90.0, 60.0, 30.0):
            with self.subTest(hz=hz):
                ts = self._sample_times(hz, self.DT)
                eff = 1.0 / float(np.mean(np.diff(ts)))
                self.assertLess(
                    abs(eff - hz) / hz, 0.01,
                    f"要 {hz:.0f} Hz，实际 {eff:.2f} Hz。"
                    f"修复前 120 Hz 只能跑到 100.0 Hz（慢 16.7%），"
                    f"任何『帧率对稳定性的影响』的结论都会跟着错")

    def test_source_accumulates_absolute_time(self):
        import inspect
        from armctrl.control.visual_servo import ObjectPoseSensor
        src = inspect.getsource(ObjectPoseSensor.update)
        self.assertIn("self._next_sample_t += period", src,
                      "没有累加绝对时刻，帧率会被控制周期系统性拖慢")
        self.assertNotIn("if t - self._last_sample_t < period", src,
                         "还在用『上次实际采样时刻 + 周期』，帧率必然偏低")


class TestSaturationsCannotBeUsedAsAStabilityTest(unittest.TestCase):
    """⭐⭐⭐ 第五轮最重要的一条：有饱和时『收没收敛』不是稳定性判据。

    这里用的是**纯代数**递推 e(k+1)=e(k)−λΔt·e(k−d)，不碰 MuJoCo，
    所以「谁稳谁不稳」有解析答案，可以拿来检验判据本身有没有分辨能力。
    """

    DT = 0.002
    D = 25                                    # τ = 50 ms

    @classmethod
    def _boundary(cls, d):
        return 2.0 * d * np.sin(np.pi / (4.0 * d + 2.0))

    @classmethod
    def _spectral_radius(cls, lam, d):
        n = d + 1
        A = np.zeros((n, n))
        A[0, 0] = 1.0
        A[0, -1] = -lam * cls.DT
        for i in range(1, n):
            A[i, i - 1] = 1.0
        return float(np.max(np.abs(np.linalg.eigvals(A))))

    @classmethod
    def _final_error(cls, lam, d, T=60.0, e0=0.05,
                     deadband=1e-3, step_max=4e-3):
        n = int(T / cls.DT)
        h = np.zeros(n + d + 1)
        h[: d + 1] = e0
        for k in range(d, n + d):
            ed = h[k - d]
            if abs(ed) < deadband:
                dlt = 0.0
            else:
                dlt = -lam * cls.DT * ed
                if abs(dlt) > step_max:
                    dlt = np.sign(dlt) * step_max
            h[k + 1] = h[k] + dlt
        return abs(float(h[-1]))

    def test_closed_form_matches_numerical_root(self):
        """闭式边界与数值二分求 ρ=1 必须一致——两条独立路径。"""
        for d in (5, 10, 25, 60):
            with self.subTest(d=d):
                lam_a = self._boundary(d) / (d * self.DT)
                lo, hi = 0.1, 5000.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if self._spectral_radius(mid, d) < 1.0:
                        lo = mid
                    else:
                        hi = mid
                lam_n = 0.5 * (lo + hi)
                self.assertLess(
                    abs(lam_n - lam_a) / lam_a, 1e-9,
                    f"d={d}: 闭式 {lam_a:.6f} vs 数值 {lam_n:.6f}")

    def test_discrete_boundary_tends_to_pi_over_two(self):
        self.assertLess(self._boundary(5), self._boundary(500),
                        "离散边界应随 d 单调趋近 π/2")
        self.assertAlmostEqual(
            self._boundary(200000), np.pi / 2, delta=1e-5,
            msg="d→∞ 时离散边界必须收敛到 π/2 = 1.5707963")
        self.assertLess(
            self._boundary(25), np.pi / 2,
            "离散边界永远比 π/2 **小**；以前把这 2% 归因于『相机半帧滞后』是错的")

    def test_final_error_cannot_tell_stable_from_unstable(self):
        """⭐ 判据自检：给一个**解析上确定发散**的工况，看『末值』报不报。"""
        d = self.D
        lam_crit = self._boundary(d) / (d * self.DT)

        lam_div = 35.0
        self.assertGreater(
            self._spectral_radius(lam_div, d), 1.0,
            f"前提失效：λ={lam_div} 应当在解析上发散（λ_crit={lam_crit:.3f}）")

        e_stable = self._final_error(30.0, d)      # ρ=0.99927，稳定
        e_critical = self._final_error(30.8, d)    # ρ≈1.00000，临界
        e_divergent = self._final_error(lam_div, d)  # ρ=1.00359，必然发散

        self.assertLess(
            e_divergent, e_critical,
            f"⭐ 这正是要锁住的反直觉现象：必然发散的 λ={lam_div} 末值 "
            f"{e_divergent * 1e3:.3f} mm，反而**小于**临界点的 "
            f"{e_critical * 1e3:.3f} mm。一旦这条不成立，说明饱和参数被改过，"
            "§7.4b′ 的撤回理由需要重新核对")
        self.assertLess(
            e_divergent, 10e-3,
            f"必然发散的工况末值只有 {e_divergent * 1e3:.3f} mm——"
            "所以『收没收敛』完全没有分辨能力")
        self.assertLess(e_stable, 2e-3, "λ=30 确实稳定，末值应在死区量级")


from armctrl.tests._docs import requires_docs


class TestRetractedClaimsStayRetracted(unittest.TestCase):
    """⭐ 锁住第五轮的撤回：那批数字不能悄悄回到文档或界面里。"""

    BAD = ["1.46", "63.5 s", "τ_mech", "tau_mech"]

    def _read(self, rel):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        return (root / rel).read_text(encoding="utf-8")

    def test_slider_no_longer_quotes_the_retracted_boundary(self):
        from armctrl.tuner.scenes import GraspScene
        blob = ""
        for p in GraspScene.params:
            if getattr(p, "key", "") == "vs_gain":
                blob = f"{p.help or ''}{p.effect or ''}"
        self.assertTrue(blob, "找不到 vs_gain 滑条")
        self.assertNotIn(
            "1.46", blob,
            "滑条还在写已撤回的『实测边界 λτ∈[1.46,1.50]』——"
            "那是用没有分辨能力的判据量出来的")
        self.assertIn("2d·sin(π/(4d+2))", blob,
                      "应当给出离散边界的闭式，而不是只写 π/2")
        self.assertIn(
            "本场景<b>测不出</b>可信的边界", blob,
            "滑条必须明说本场景量不出可信边界（三重饱和 + 关掉饱和就撞物理极限）；"
            "只提一句『有饱和』不够——旧文案也提了饱和，却仍然给出了边界数字")
        self.assertIn(
            "饱和只给『有界』，不给『稳定』", blob,
            "必须写明饱和不提供稳定性，否则读者会拿『没跑飞』当稳定")

    @requires_docs
    def test_doc_carries_an_explicit_retraction(self):
        doc = self._read("armctrl_讲解/07_抓取与视觉伺服.md")
        self.assertIn("撤回声明", doc, "§7.4b′ 必须有明确的撤回声明")
        self.assertIn("2d\\,\\sin\\!\\frac{\\pi}{4d+2}", doc,
                      "撤回之后要给出经过验证的离散边界闭式")
        head = doc[doc.index("### 7.4b′"): doc.index("### 7.4c")]
        self.assertIn(
            "82.0 mm", head,
            "要写明旧拓扑下『内部命令 vs 真实末端』的实测差距")
        self.assertNotIn(
            "标定点，≈14.8", head,
            "已撤回的 τ_mech 标定—预测表不得留在文档里")

    @requires_docs
    def test_index_meta_lesson_marks_the_prediction_as_false_success(self):
        doc = self._read("armctrl_讲解/00_总索引.md")
        self.assertIn("可证伪预测是必要条件，不是充分条件", doc,
                      "元教训 12 必须说明『预测中了也可能是假的』")
        self.assertIn('"有界"不等于"稳定"', doc,
                      "元教训 20 缺失")


if __name__ == "__main__":
    unittest.main()
