"""伺服谐振测量的**独立**验证（第四轮审核 P1-07）。

本文件的核心纪律：**绝不用被测代码的中间量去验证它自己**。

三个独立 oracle：
  1. `swept_sine_peak`  —— 扫频法。真的驱动 `TwoMassJoint` 一遍遍跑，
     记录每个频率下的稳态形变幅值，取最大的那个频率。
     它**只用 `TwoMassJoint.step`**，完全不碰 `resonance_frequency` 公式。
  2. `state_matrix_peak` —— 状态矩阵特征值。纯线性代数，不跑仿真。
  3. `brute_notch_response` —— 直接对陷波器喂正弦，量幅度衰减。

被验证的对象：
  * `resonance_frequency` / `antiresonance_frequency` 解析公式
  * `ServoScene._peak_freq` 的 FFT 峰值搜索（含固定搜索下限）
  * 陷波器**打在速度误差上**（PI 之前）这个实现事实
"""

import re
import unittest

import numpy as np

from armctrl.tests._docs import requires_docs
from armctrl.control.servo import (
    CascadeServo,
    NotchFilter,
    TwoMassJoint,
    antiresonance_frequency,
    resonance_frequency,
)


# ---------------------------------------------------------------- oracle 1
def swept_sine_peak(k_s, j_m, j_l, d_s=0.5,
                    f_lo=0.5, f_hi=60.0, n_f=200, dt=2.0e-4):
    """扫频 oracle（Hz）：逐个频率驱动真实对象，返回形变幅值最大的频率。

    ⭐ 不使用任何解析公式——只调用 `TwoMassJoint.step`，
    负载侧自己积分成一个纯惯量。`TwoMassJoint` 只建模电机+弹簧，
    负载是外部给的（真实场景里是 MuJoCo 机械臂）。
    """
    freqs = np.linspace(f_lo, f_hi, n_f)
    amps = np.empty(n_f)
    for i, f in enumerate(freqs):
        j = TwoMassJoint(j_m=j_m, k_s=k_s, d_s=d_s)
        j.reset()
        theta_l, omega_l = 0.0, 0.0
        cycles = max(30, int(6 * f))
        n_step = int((cycles / f) / dt)
        hold = int(0.6 * n_step)
        peak = 0.0
        for k in range(n_step):
            t = k * dt
            tau_s = j.step(np.sin(2.0 * np.pi * f * t), theta_l, omega_l, dt)
            omega_l += tau_s / j_l * dt
            theta_l += omega_l * dt
            if k > hold:
                peak = max(peak, abs(j.theta_m - theta_l))
        amps[i] = peak
    return float(freqs[int(np.argmax(amps))])


def _res_hz(k_s, j_m, j_l):
    """⚠️ `resonance_frequency` 返回的是 **rad/s**，不是 Hz。"""
    return resonance_frequency(k_s, j_m, j_l) / (2.0 * np.pi)


def _ares_hz(k_s, j_l):
    return antiresonance_frequency(k_s, j_l) / (2.0 * np.pi)


# ---------------------------------------------------------------- oracle 2
def state_matrix_peak(k_s, j_m, j_l):
    """状态矩阵特征值 oracle：纯线性代数，不跑仿真、不用解析公式。

    状态 x = [theta_m, theta_l, omega_m, omega_l]。
    """
    a = np.zeros((4, 4))
    a[0, 2] = 1.0
    a[1, 3] = 1.0
    a[2, 0] = -k_s / j_m
    a[2, 1] = +k_s / j_m
    a[3, 0] = +k_s / j_l
    a[3, 1] = -k_s / j_l
    ev = np.linalg.eigvals(a)
    w = np.max(np.abs(ev.imag))
    return w / (2.0 * np.pi)


class TestSweptSineOracleAgreesWithFormula(unittest.TestCase):
    """oracle 1（跑仿真）与解析公式必须一致——两条完全独立的路径。"""

    CASES = [
        (800.0, 0.30, 0.60),
        (400.0, 0.30, 0.60),
        (2000.0, 0.10, 0.60),
    ]

    def test_swept_sine_matches_resonance_formula(self):
        for k_s, j_m, j_l in self.CASES:
            with self.subTest(k_s=k_s, j_m=j_m):
                meas = swept_sine_peak(k_s, j_m, j_l)
                theory = _res_hz(k_s, j_m, j_l)
                rel = abs(meas - theory) / theory
                self.assertLess(
                    rel, 0.03,
                    f"K_s={k_s} J_m={j_m}: 扫频实测 {meas:.3f} Hz 与解析式 "
                    f"{theory:.3f} Hz 相差 {rel:.1%}（应 <2%）")

    def test_state_matrix_matches_resonance_formula(self):
        for k_s, j_m, j_l in self.CASES:
            with self.subTest(k_s=k_s, j_m=j_m):
                meas = state_matrix_peak(k_s, j_m, j_l)
                theory = _res_hz(k_s, j_m, j_l)
                self.assertAlmostEqual(
                    meas, theory, places=6,
                    msg=f"特征值 {meas:.6f} vs 公式 {theory:.6f}")

    def test_oracle_can_tell_resonance_from_antiresonance(self):
        """⭐ 变异证明：oracle 必须能区分谐振与反谐振。

        若 oracle 是坏的（比如峰找错位置），这个断言就会失效。
        """
        for k_s, j_m, j_l in self.CASES:
            with self.subTest(k_s=k_s):
                f_res = _res_hz(k_s, j_m, j_l)
                f_ares = _ares_hz(k_s, j_l)
                self.assertGreater(
                    f_res, f_ares * 1.1,
                    "谐振应显著高于反谐振，否则这个工况区分不出对错")
                meas = swept_sine_peak(k_s, j_m, j_l)
                self.assertGreater(
                    abs(meas - f_ares) / f_ares, 0.10,
                    f"扫频峰 {meas:.2f} Hz 落在了反谐振 {f_ares:.2f} Hz 附近，"
                    "说明测的是错的对象")


class TestPeakSearchFloorIsIndependent(unittest.TestCase):
    """⭐ P1-07(f)：峰值搜索窗口不能由"被验证的量"自己定义。"""

    def test_floor_is_a_plain_constant(self):
        from armctrl.tuner.scenes import ServoScene
        floor = ServoScene.PEAK_SEARCH_FLOOR_HZ
        self.assertIsInstance(floor, float)
        self.assertGreater(floor, 0.0)
        self.assertLess(
            floor, 3.0,
            f"搜索下限 {floor} Hz 太高，会把低刚度工况的真实峰挡在外面")

    def test_floor_is_below_every_reachable_resonance(self):
        """搜索下限必须低于整个参数空间里最低的谐振频率。"""
        from armctrl.tuner.scenes import ServoScene
        floor = ServoScene.PEAK_SEARCH_FLOOR_HZ
        j_l = 0.60
        lowest = min(
            _res_hz(k_s, j_m, j_l)
            for k_s in (120.0, 6000.0)
            for j_m in (0.10, 0.40)
        )
        self.assertLess(
            floor, lowest,
            f"搜索下限 {floor} Hz 高于最低可达谐振 {lowest:.2f} Hz")

    def test_old_self_locating_window_would_have_been_wrong(self):
        """⭐ 变异证明：旧窗口 sqrt(f_ares*f_res) 会随参数漂移 16 倍。

        一个"随被测量一起移动"的搜索窗口，测不出被测量测错了。
        """
        combos = [(120.0, 0.40), (6000.0, 0.10)]
        j_l = 0.60
        old = [np.sqrt(_ares_hz(k, j_l) * _res_hz(k, jm, j_l))
               for k, jm in combos]
        self.assertGreater(
            max(old) / min(old), 5.0,
            f"旧窗口下限在参数空间里从 {min(old):.2f} 变到 {max(old):.2f} Hz，"
            "它是跟着被测量走的——这正是自证循环")


class TestNotchActsOnVelocityError(unittest.TestCase):
    """⭐ P1-07(d)：陷波器打在**速度误差**上（PI 之前），不是控制器输出。"""

    @staticmethod
    def _loop(with_notch, f_notch=10.0):
        loop = CascadeServo(kpp=20.0, kvp=10.0, kvi=5.0, tau_max=1e6)
        if with_notch:
            loop.notch = NotchFilter(f_notch, depth=0.05, width=0.6, dt=1e-3)
        loop.reset()
        return loop

    def test_notch_changes_the_recorded_velocity_error(self):
        """若陷波在 PI 之后，`v_err` 就该是**未滤波**的原始误差。

        实际实现里 `v_err` 记的是滤波**之后**的值 ⇒ 陷波在 PI 之前。
        """
        dt = 1e-3
        raw, filt = [], []
        lo_a, lo_b = self._loop(False), self._loop(True)
        for k in range(4000):
            t = k * dt
            omega_m = np.sin(2.0 * np.pi * 10.0 * t)
            lo_a.update(0.0, 0.0, 0.0, omega_m, dt)
            lo_b.update(0.0, 0.0, 0.0, omega_m, dt)
            if t > 2.0:
                raw.append(lo_a.v_err)
                filt.append(lo_b.v_err)
        amp_raw = float(np.max(np.abs(raw)))
        amp_filt = float(np.max(np.abs(filt)))
        self.assertGreater(
            amp_raw / max(amp_filt, 1e-12), 3.0,
            f"陷波开启后 v_err 幅值 {amp_filt:.4f} 应远小于未滤波 {amp_raw:.4f}；"
            "若二者相同，说明陷波被放到了 PI 之后")

    def test_integrator_never_sees_the_resonant_component(self):
        """⭐ 关键区别：陷波在 PI 前，积分器就攒不到谐振分量。"""
        dt = 1e-3
        lo_a, lo_b = self._loop(False), self._loop(True)
        for k in range(6000):
            t = k * dt
            omega_m = 2.0 * np.sin(2.0 * np.pi * 10.0 * t) + 0.05
            lo_a.update(0.0, 0.0, 0.0, omega_m, dt)
            lo_b.update(0.0, 0.0, 0.0, omega_m, dt)
        self.assertLess(
            abs(lo_b.v_err), abs(lo_a.v_err),
            "陷波回路的瞬时速度误差应更小")

    def test_notch_is_wired_into_the_error_path_in_source(self):
        """结构性断言：源码里 notch 必须出现在 `self.kvp * err` **之前**。"""
        import inspect
        src = inspect.getsource(CascadeServo.update)
        i_notch = src.index("self.notch(err)")
        i_pi = src.index("self.kvp * err")
        self.assertLess(
            i_notch, i_pi,
            "陷波调用出现在 PI 之后了——文档描述的信号路径与实现不符")


class TestProbeModeIsHonestlyLabelled(unittest.TestCase):
    """⭐ P1-07(e)：`probe` 旁路伺服环，所以"峰与 Kvp 无关"是构造性的。

    这里只验证**两个模式确实不同**（一个旁路、一个不旁路），
    真实的频率对比在讲解文档里给出（跑一次要十几分钟，不适合放进单元测试）。
    """

    def test_both_probe_modes_exist(self):
        from armctrl.tuner.scenes import ServoScene
        choices = next(p for p in ServoScene.params
                       if p.key == "move").choices
        self.assertIn("probe", choices)
        self.assertIn("probe_cl", choices,
                      "缺少闭环探针模式，就无法证明开环结论是构造性的")

    def test_open_loop_probe_bypasses_the_servo(self):
        """源码结构：`probe` 分支旁路伺服，`probe_cl` 不旁路。"""
        import inspect
        from armctrl.tuner.scenes import ServoScene
        src = inspect.getsource(ServoScene.control)
        self.assertIn('inject = _mv in ("probe", "probe_cl")', src,
                      "两个模式都应注入噪声")
        self.assertIn('probe = _mv == "probe"', src,
                      "只有开环 probe 才旁路伺服环")
        i_inject = src.index("if inject:")
        i_bypass = src.index("if probe:")
        self.assertLess(i_inject, i_bypass,
                        "噪声注入应先于旁路判断，否则闭环模式收不到激励")


# =====================================================================
# 第五轮·批次 D-2：高 Kvp 下的闭环峰是**反谐振**，不是"伺服带宽峰"
# =====================================================================
#
# 旧笔记（13 号 §7）曾写：闭环峰 6.35 → 2.93 Hz，"降了 54%"，
# 解释成"主导极点换人，看到的是伺服环自己的闭环带宽峰"。
# 两处都不成立：
#   * 那张表每格只读了一次，而 f_meas 会随读取时刻在 2.93~7.81 Hz 之间跳；
#   * 那个峰对 J_m **完全免疫**、随 K_s 走 —— 是反谐振 sqrt(K_s/J_l)。
#
# 下面这组测试用**纯两质量 ODE**（不启动 MuJoCo、不碰 ServoScene._peak_freq）
# 独立复现这件事，几毫秒就能跑完，所以它可以放进单元测试
# —— 这正是当年"跑一次十几分钟、只好写进文档"留下的缺口。


def _closed_loop_peak(k_s, j_m, j_l, kvp, *, closed, seconds=40.0,
                      dt=2e-3, seed=0, d_s=0.6, kpp=20.0, kvi=12.0,
                      floor_hz=1.0, nfft=2048):
    """白噪声激励两质量系统，返回弹簧形变 δ 的频谱主峰（Hz）。

    `closed=False` 时电机力矩里**没有伺服环**（对应 `probe`），
    `closed=True` 时串级伺服在环（对应 `probe_cl`）。
    除此之外两条路径逐行相同。
    """
    joint = TwoMassJoint(j_m=j_m, k_s=k_s, d_s=d_s)
    joint.reset()
    servo = CascadeServo(kpp=kpp, kvp=kvp, kvi=kvi)
    rng = np.random.default_rng(seed)
    th_l = om_l = 0.0
    rec = []
    for _ in range(int(seconds / dt)):
        tau = 2.0 * rng.standard_normal()
        if closed:
            tau += servo.update(0.0, 0.0, joint.theta_m, joint.omega_m, dt)
        tau_s = joint.step(tau, th_l, om_l, dt)
        om_l += tau_s / j_l * dt          # 负载：半隐式，与 joint.step 同格式
        th_l += om_l * dt
        rec.append(joint.theta_m - th_l)
    x = np.asarray(rec[len(rec) // 4:])   # 丢掉前 1/4 的暂态
    n = nfft
    win = np.hanning(n)
    acc, k = None, 0
    for i in range(0, len(x) - n + 1, n // 2):
        seg = x[i:i + n]
        pw = np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2
        acc = pw if acc is None else acc + pw
        k += 1
    power = acc / k
    freqs = np.fft.rfftfreq(n, dt)
    band = freqs >= floor_hz
    pb, fb = power[band], freqs[band]
    inner = np.where((pb[1:-1] > pb[:-2]) & (pb[1:-1] > pb[2:]))[0] + 1
    if inner.size == 0:
        return 0.0
    return float(fb[int(inner[int(np.argmax(pb[inner]))])])


class TestHighGainPeakIsTheAntiresonance(unittest.TestCase):
    """⭐⭐⭐ 第五轮·批次 D-2：把"高增益闭环峰是什么"这件事钉死。

    判别手法就是 13 号 §3.1 教的那一招：**只改 J_m**。
    f_res 依赖 J_m，f_ares 不依赖。频率跟谁走，那个峰就是谁。
    """

    K_S, J_L = 400.0, 1.0
    TWO_PI = 2.0 * np.pi

    def _f_res(self, j_m):
        return resonance_frequency(self.K_S, j_m, self.J_L) / self.TWO_PI

    def _f_ares(self):
        return antiresonance_frequency(self.K_S, self.J_L) / self.TWO_PI

    def test_open_loop_peak_tracks_the_resonance(self):
        """开环（伺服不在环）：峰必须跟着 f_res 走 —— 这是对照组。"""
        for j_m in (0.15, 0.30, 0.60):
            f = _closed_loop_peak(self.K_S, j_m, self.J_L, 45.0, closed=False)
            self.assertAlmostEqual(
                f, self._f_res(j_m), delta=0.6,
                msg=f"J_m={j_m}：开环峰 {f:.2f} 应当 ≈ f_res "
                    f"{self._f_res(j_m):.2f}")

    def test_open_loop_peak_really_moves_with_j_m(self):
        """反向守卫：若开环峰也纹丝不动，上一条就变成恒真了。"""
        got = [_closed_loop_peak(self.K_S, j, self.J_L, 45.0, closed=False)
               for j in (0.15, 0.60)]
        self.assertGreater(
            abs(got[0] - got[1]), 1.5,
            f"J_m 翻两番开环峰只动了 {abs(got[0]-got[1]):.2f} Hz —— "
            f"对照组失效，J_m 判别法在这组参数下没有分辨力")

    def test_high_gain_closed_loop_peak_is_immune_to_j_m(self):
        """⭐⭐⭐ 高 Kvp 闭环峰对 J_m 免疫 ⇒ 它不可能是机械谐振。"""
        got = [_closed_loop_peak(self.K_S, j, self.J_L, 60.0, closed=True)
               for j in (0.15, 0.30, 0.60)]
        self.assertLess(
            max(got) - min(got), 1.0,
            f"J_m 三档下闭环峰 {[round(x,2) for x in got]} 极差过大 —— "
            f"它不像反谐振")
        for j_m, f in zip((0.15, 0.30, 0.60), got):
            self.assertAlmostEqual(
                f, self._f_ares(), delta=0.8,
                msg=f"J_m={j_m}：闭环峰 {f:.2f} 应当 ≈ f_ares "
                    f"{self._f_ares():.2f}（而不是 f_res "
                    f"{self._f_res(j_m):.2f}）")
            self.assertGreater(
                abs(f - self._f_res(j_m)), 0.8,
                f"J_m={j_m}：闭环峰 {f:.2f} 同时也贴着 f_res —— "
                f"这组参数分不开两者，换 J_m 再测")

    def test_the_answer_is_not_an_artefact_of_the_search_floor(self):
        """⭐⭐⭐ 排除竞争假设：那个峰会不会只是「测不到更低」？

        第五轮审核报告点出：场景里 FFT 分辨率 0.48828 Hz、搜索下限 2.0 Hz、
        且只取内部极大值 ⇒ **最低可报告频率恰好就是 2.9297 Hz**。
        而高增益下场景长期报 2.93 —— 这完全可能是**边界钳制**，
        跟"它是反谐振"是两个不同的解释。

        排除办法（元教训 #42）：**把下限挪一挪，看答案动不动。**
        真峰不随下限变；钳制值会紧贴下限一起走。
        """
        base = _closed_loop_peak(self.K_S, 0.30, self.J_L, 60.0, closed=True,
                                 floor_hz=1.0)
        for floor in (0.4, 0.7, 1.5):
            f = _closed_loop_peak(self.K_S, 0.30, self.J_L, 60.0, closed=True,
                                  floor_hz=floor)
            self.assertAlmostEqual(
                f, base, delta=0.3,
                msg=f"搜索下限 1.0→{floor} Hz 时峰从 {base:.2f} 变成 {f:.2f} "
                    f"—— 说明读数被下限钳住了，不是真峰")
        step = 1.0 / (2048 * 2e-3)
        self.assertGreater(
            (base - 1.0) / step, 3.0,
            f"峰 {base:.2f} 距搜索下限只有 {(base-1.0)/step:.1f} 格，太近")

    def test_finer_resolution_does_not_move_the_answer(self):
        """⭐⭐ 第二条：把频率分辨率提高一倍，答案也不该动。"""
        a = _closed_loop_peak(self.K_S, 0.30, self.J_L, 60.0, closed=True,
                              nfft=2048)
        b = _closed_loop_peak(self.K_S, 0.30, self.J_L, 60.0, closed=True,
                              nfft=4096, seconds=80.0)
        self.assertAlmostEqual(
            a, b, delta=0.3,
            msg=f"分辨率翻倍后峰从 {a:.2f} 变成 {b:.2f} —— 读数不可信")

    def test_high_gain_closed_loop_peak_scales_with_stiffness(self):
        """第二根独立轴：f_ares = sqrt(K_s/J_l) ⇒ K_s 翻 4 倍，峰翻 2 倍。"""
        lo = _closed_loop_peak(400.0, 0.30, self.J_L, 60.0, closed=True)
        hi = _closed_loop_peak(1600.0, 0.30, self.J_L, 60.0, closed=True)
        self.assertGreater(lo, 0.0)
        self.assertAlmostEqual(
            hi / lo, 2.0, delta=0.35,
            msg=f"K_s 400→1600，峰 {lo:.2f}→{hi:.2f}，比值 {hi/lo:.2f}，"
                f"应当 ≈2（反谐振随 sqrt(K_s) 走）")

    def test_a_weak_servo_still_shows_the_resonance(self):
        """伺服整体放软时，闭环峰回到机械谐振 —— 「随增益换模态」的另一端。

        这里把 `kpp`、`kvi` 也一并调小，是为了得到一个干净的"伺服基本不在"端点。
        因果轴到底是不是 `kvp`，由下一条测试单独回答。
        """
        f = _closed_loop_peak(self.K_S, 0.30, self.J_L, 0.1, closed=True,
                              kpp=0.5, kvi=0.0)
        self.assertAlmostEqual(
            f, self._f_res(0.30), delta=0.8,
            msg=f"弱伺服闭环峰 {f:.2f} 应当仍是机械谐振 "
                f"{self._f_res(0.30):.2f}")

    def test_kvp_is_the_causal_axis_not_the_position_loop(self):
        """⭐⭐ 把 Kvp 单独归零（位置环、积分照旧），峰立刻回到机械谐振。

        这一条是「换模态是 $K_{vp}$ 干的」这句话的**因果证据**：
        同一组 kpp/kvi 下，只动 kvp 就能让峰在 f_res 和 f_ares 之间来回。
        没有它，"高增益让电机变成速度源"就只是一个说得通的故事。
        """
        f0 = _closed_loop_peak(self.K_S, 0.30, self.J_L, 0.0, closed=True)
        f1 = _closed_loop_peak(self.K_S, 0.30, self.J_L, 60.0, closed=True)
        self.assertAlmostEqual(
            f0, self._f_res(0.30), delta=0.8,
            msg=f"Kvp=0 时峰 {f0:.2f} 应当就是机械谐振 "
                f"{self._f_res(0.30):.2f}（位置环和积分并不足以按住电机）")
        self.assertAlmostEqual(
            f1, self._f_ares(), delta=0.8,
            msg=f"同样的 kpp/kvi，只把 Kvp 提到 60，峰 {f1:.2f} "
                f"应当移到反谐振 {self._f_ares():.2f}")


@requires_docs
class TestTheServoDocRetractedTheOldTable(unittest.TestCase):
    """⭐ 13 号 §7.4–7.6 必须保留撤回声明，且不得复活旧说法。"""

    @staticmethod
    def _doc():
        import io
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "armctrl_讲解", "13_伺服环路与振动抑制.md")
        with io.open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_the_wrong_explanation_is_gone(self):
        d = self._doc()
        i = d.find("旧解释：")
        self.assertGreater(i, 0, "撤回段落被删了")
        self.assertNotIn("闭环带宽峰", d[:i],
                         "「伺服环自己的闭环带宽峰」这个错解释又被写回正文了")

    def test_the_54_percent_claim_is_marked_retracted(self):
        d = self._doc()
        for k in ("54%", "现已作废"):
            self.assertIn(k, d, f"缺少「{k}」——撤回记录不完整")
        i54 = d.find("54%")
        self.assertIn("现已作废", d[max(0, i54 - 900):i54 + 200],
                      "「54%」出现的地方附近没有作废标记")

    def test_the_doc_states_the_antiresonance_conclusion(self):
        d = self._doc()[self._doc().find("### 7.5"):]
        self.assertIn("那个峰是反谐振", d,
                      "§7.5 没有明确写出「那个峰是反谐振」这句结论")
        self.assertIn("K_s/J_l", d.replace("\\sqrt{K_s/J_l}", "K_s/J_l"),
                      "没有写出反谐振公式")

    # ---- 下面三条：把 §7.5 的三张表**当成数据**来查 -------------------
    #
    # 这三张表来自真实 ServoScene，跑一遍要好几分钟，不适合每次单测都重跑。
    # 但表**内部**是否自洽、是否真的支持"跟 f_ares 走"这个结论，
    # 完全可以从表自己的数字加解析公式查出来 —— 而这恰恰是最容易被改坏的地方。

    @staticmethod
    def _rows(doc, header_key):
        out = []
        for ln in doc.splitlines():
            if not ln.startswith("| ") or header_key in ln or "---" in ln:
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            nums = []
            for c in cells:
                got = re.findall(r"\d+\.?\d*", c.replace("*", ""))
                nums.append([float(g) for g in got])
            if len(nums) >= 4 and nums[0] and nums[1] and nums[2]:
                out.append(nums)
        return out

    def test_the_j_m_table_actually_supports_the_conclusion(self):
        """⭐⭐⭐ J_m 表：开环列必须随 J_m 变，闭环列必须不变且贴 f_ares。"""
        d = self._doc()
        d = d[d.find("### 7.5"):]
        seg = d[d.find("| $J_m$ |"):]
        seg = seg[:seg.find("\n\n")]
        rows = self._rows(seg, "$J_m$")
        self.assertEqual(len(rows), 3, f"J_m 表应有 3 行，解析到 {len(rows)}")
        f_res = [r[1][0] for r in rows]
        f_ares = [r[2][0] for r in rows]
        open_loop = [float(np.median(r[3])) for r in rows]
        closed = [float(np.median(r[4])) for r in rows]

        self.assertLess(max(f_ares) - min(f_ares), 1e-6,
                        f"表里 f_ares 列本该恒定，却是 {f_ares}")
        self.assertGreater(max(f_res) - min(f_res), 2.0,
                           f"表里 f_res 列本该随 J_m 大幅变化，却是 {f_res}")
        for fr, fo in zip(f_res, open_loop):
            self.assertLess(abs(fo - fr), 1.0,
                            f"开环实测 {fo} 与该行 f_res {fr} 差太多")
        self.assertLess(
            max(closed) - min(closed), 1.0,
            f"⭐ 闭环列 {closed} 随 J_m 变了 —— 那它就不是反谐振，"
            f"§7.5 的结论不成立")
        for fr, fc in zip(f_res, closed):
            self.assertLess(
                abs(fc - f_ares[0]), abs(fc - fr),
                f"闭环实测 {fc} 离 f_res {fr} 比离 f_ares {f_ares[0]} 更近 —— "
                f"表本身不支持「跟 f_ares 走」")

    def test_the_k_s_table_actually_supports_the_conclusion(self):
        """⭐⭐ K_s 表：闭环列必须每一行都更靠近 f_ares 而不是 f_res。"""
        d = self._doc()
        d = d[d.find("### 7.5"):]
        seg = d[d.find("| $K_s$ |"):]
        seg = seg[:seg.find("\n\n")]
        rows = self._rows(seg, "$K_s$")
        self.assertEqual(len(rows), 3, f"K_s 表应有 3 行，解析到 {len(rows)}")
        for r in rows:
            k_s, f_res, f_ares = r[0][0], r[1][0], r[2][0]
            closed = float(np.median(r[4]))
            self.assertLess(
                abs(closed - f_ares), abs(closed - f_res),
                f"K_s={k_s}：闭环 {closed} 离 f_res {f_res} 更近，"
                f"「跟 f_ares」这一列的判定与数字矛盾")
        ratio = [r[2][0] for r in rows]
        self.assertAlmostEqual(
            ratio[2] / ratio[0], 2.0, delta=0.1,
            msg=f"K_s 400→1600 时 f_ares 应当翻倍，表里是 {ratio}")

    def test_the_causal_table_is_reproducible(self):
        """⭐⭐⭐ 「只动 Kvp」那张表必须能被 toy oracle 当场重跑出来。"""
        d = self._doc()
        seg = d[d.find("#### 还差一步"):]
        seg = seg[:seg.find("> ⭐ 这就是")]
        m0 = re.search(r"\|\s*\*\*0\*\*\s*\|\s*\*\*([\d.]+) Hz\*\*", seg)
        m1 = re.search(r"\|\s*\*\*60\*\*\s*\|\s*\*\*([\d.]+) Hz\*\*", seg)
        self.assertIsNotNone(m0, "因果表 Kvp=0 那行没了")
        self.assertIsNotNone(m1, "因果表 Kvp=60 那行没了")
        got0 = _closed_loop_peak(400.0, 0.30, 1.0, 0.0, closed=True)
        got1 = _closed_loop_peak(400.0, 0.30, 1.0, 60.0, closed=True)
        self.assertAlmostEqual(
            float(m0.group(1)), got0, delta=0.3,
            msg=f"文档写 Kvp=0 → {m0.group(1)} Hz，实测 {got0:.2f}")
        self.assertAlmostEqual(
            float(m1.group(1)), got1, delta=0.3,
            msg=f"文档写 Kvp=60 → {m1.group(1)} Hz，实测 {got1:.2f}")
        self.assertGreater(
            got0 - got1, 2.0,
            f"只动 Kvp 峰只移了 {got0-got1:.2f} Hz —— 因果结论站不住")

    def test_the_doc_time_series_table_shows_real_spread(self):
        """文档那张「同一次运行不同时刻」的表，极差列必须真的很大。

        否则"只读一次不算数"这个论点就没有证据支撑。
        """
        d = self._doc()
        m = re.search(r"\|\s*`probe_cl` \$K_\{vp\}\$=15\s*\|([^\n]*)\|", d)
        self.assertIsNotNone(m, "§7.4 的时间序列表被删或改了写法")
        cells = [c.strip().strip("*") for c in m.group(1).split("|")]
        nums = [float(c) for c in cells if re.fullmatch(r"[\d.]+", c)]
        self.assertGreaterEqual(len(nums), 8, f"这一行只解析出 {nums}")
        readings, spread = nums[:-1], nums[-1]
        self.assertAlmostEqual(
            max(readings) - min(readings), spread, delta=0.05,
            msg=f"极差列 {spread} 与该行读数 {readings} 对不上")
        self.assertGreater(
            spread, 2.0,
            f"极差只有 {spread} Hz，撑不起「读数会乱跳」这个论点")


if __name__ == "__main__":
    unittest.main()
