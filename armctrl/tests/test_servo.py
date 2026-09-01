"""伺服环路与振动抑制的数学不变量测试。

oracle 独立性
------------
- 谐振/反谐振频率：不调用 `resonance_frequency`，而是对双质量系统的
  **状态矩阵直接做特征值分解**。两条路（解析公式 vs 数值特征值）
  完全不共享代码，互相印证。
- 陷波深度：用**连续域传递函数在 s=jω 处的复数值**独立算，
  再与离散滤波器对正弦稳态的实测增益比对。
- 输入整形：用残余振动闭式 V(ω,ζ) 独立算，并另用**一个真实的
  二阶系统数值积分**验证残余振幅确实被压下去。
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.control.servo import (
    CascadeServo,
    InputShaper,
    NotchFilter,
    TwoMassJoint,
    antiresonance_frequency,
    residual_vibration,
    resonance_frequency,
)


def two_mass_eigenfreqs(k_s, d_s, j_m, j_l):
    """双质量系统的振动频率——用状态矩阵特征值独立求。

    状态 x = [θ_m, θ_l, ω_m, ω_l]。开环（τ_m = 0）时

        ω̇_m = [−K(θ_m−θ_l) − D(ω_m−ω_l)] / J_m
        ω̇_l = [ K(θ_m−θ_l) + D(ω_m−ω_l)] / J_l

    非零特征值的虚部就是阻尼振荡频率；D→0 时它就是无阻尼谐振频率。
    这条路完全不经过任何解析公式。
    """
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2, 0], A[2, 1] = -k_s / j_m, k_s / j_m
    A[2, 2], A[2, 3] = -d_s / j_m, d_s / j_m
    A[3, 0], A[3, 1] = k_s / j_l, -k_s / j_l
    A[3, 2], A[3, 3] = d_s / j_l, -d_s / j_l
    ev = np.linalg.eigvals(A)
    im = np.abs(ev.imag)
    return float(im.max())


def locked_load_eigenfreq(k_s, d_s, j_m, j_l):
    """负载被**夹死**（θ_l≡0）时电机自己的振动频率 = 反谐振？不是。

    反谐振是「电机被夹死、负载自己晃」：J_l θ̈_l = −K θ_l − D θ̇_l，
    无阻尼时 ω = sqrt(K/J_l)。这条也是独立求的。
    """
    A = np.array([[0.0, 1.0], [-k_s / j_l, -d_s / j_l]])
    return float(np.abs(np.linalg.eigvals(A).imag).max())


class TestResonanceFormulas(unittest.TestCase):
    """两个频率公式必须与状态矩阵特征值一致。"""

    CASES = [
        (800.0, 0.05, 0.5),
        (3000.0, 0.02, 1.2),
        (150.0, 0.10, 0.1),
        (12000.0, 0.30, 4.0),
    ]

    def test_resonance_matches_eigenvalue(self):
        for k_s, j_m, j_l in self.CASES:
            with self.subTest(k_s=k_s, j_m=j_m, j_l=j_l):
                got = resonance_frequency(k_s, j_m, j_l)
                ref = two_mass_eigenfreqs(k_s, 0.0, j_m, j_l)
                self.assertAlmostEqual(got, ref, delta=1e-6 * ref)

    def test_antiresonance_matches_eigenvalue(self):
        for k_s, j_m, j_l in self.CASES:
            with self.subTest(k_s=k_s, j_l=j_l):
                got = antiresonance_frequency(k_s, j_l)
                ref = locked_load_eigenfreq(k_s, 0.0, j_m, j_l)
                self.assertAlmostEqual(got, ref, delta=1e-6 * ref)

    def test_resonance_always_above_antiresonance(self):
        """ω_res/ω_ares = sqrt(1+J_l/J_m) > 1 恒成立。

        这条是设计上的硬结论：谐振永远在反谐振**上面**。
        """
        for k_s, j_m, j_l in self.CASES:
            r = resonance_frequency(k_s, j_m, j_l)
            a = antiresonance_frequency(k_s, j_l)
            self.assertGreater(r, a)
            self.assertAlmostEqual(r / a, np.sqrt(1.0 + j_l / j_m), places=9)

    def test_antiresonance_ignores_motor_inertia(self):
        """换电机不改反谐振——这是区分两个频率最快的实验。

        ⚠️ 这条原来写的是 ``a1 = f(800, 0.5); a2 = f(800, 0.5)`` 两次**完全相同的
        调用**，比较必然成立（第四轮审核 T-07）。
        ``antiresonance_frequency`` 的签名里根本没有 ``j_m``，
        所以"它不依赖 j_m"在函数层面是**结构上不可能违反**的，测了等于没测。

        真正该测的是**物理**：把负载锁死时的特征频率
        ``locked_load_eigenfreq(k_s, 0, j_m, j_l)`` 在 j_m 变化 100 倍时
        必须纹丝不动，而 ``resonance_frequency`` 必须明显变化。
        """
        k_s, j_l = 800.0, 0.5
        a_ref = antiresonance_frequency(k_s, j_l)
        r_prev = None
        for j_m in (0.005, 0.02, 0.2, 0.5):
            # 反谐振：换 100 倍的电机惯量也不动
            self.assertAlmostEqual(
                locked_load_eigenfreq(k_s, 0.0, j_m, j_l), a_ref,
                delta=1e-9 * a_ref,
                msg=f"j_m={j_m} 时反谐振变了——它本该只由 k_s 和 j_l 决定")
            # 谐振：必须随 j_m 单调变化（对照组，证明判据有区分能力）
            r = resonance_frequency(k_s, j_m, j_l)
            if r_prev is not None:
                self.assertLess(r, r_prev,
                                "电机惯量变大，谐振频率却没下降——对照组失效")
            r_prev = r
        # 首尾两端至少差一个量级，确保上面的"不动"不是因为都不动
        self.assertGreater(resonance_frequency(k_s, 0.005, j_l)
                           / resonance_frequency(k_s, 0.5, j_l), 5.0)

    def test_rejects_nonpositive(self):
        for bad in ((0.0, 0.1, 0.1), (10.0, 0.0, 0.1), (10.0, 0.1, -1.0)):
            with self.assertRaises(ValueError):
                resonance_frequency(*bad)


class TestTwoMassIntegration(unittest.TestCase):
    """自由振荡的实测频率必须落在解析谐振频率上。"""

    def test_free_oscillation_frequency(self):
        j_m, j_l, k_s = 0.05, 0.5, 800.0
        joint = TwoMassJoint(j_m=j_m, k_s=k_s, d_s=0.0)
        dt = 2e-5
        # 负载在这里也自己积分，构成完整的双质量系统
        joint.reset(theta=1e-3, omega=0.0)
        th_l, om_l = 0.0, 0.0
        hist = []
        for _ in range(200000):
            tau_s = joint.step(0.0, th_l, om_l, dt)
            om_l += tau_s / j_l * dt
            th_l += om_l * dt
            hist.append(joint.theta_m - th_l)
        x = np.asarray(hist) - np.mean(hist)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), dt)
        f_peak = float(freqs[int(np.argmax(spec))])
        f_theory = resonance_frequency(k_s, j_m, j_l) / (2 * np.pi)
        self.assertAlmostEqual(f_peak, f_theory, delta=0.02 * f_theory)

    def test_energy_does_not_blow_up(self):
        """半隐式欧拉对弹簧-质量系统不应注入能量。"""
        j_m, j_l, k_s = 0.05, 0.5, 800.0
        joint = TwoMassJoint(j_m=j_m, k_s=k_s, d_s=0.0)
        dt = 1e-4
        joint.reset(theta=1e-3)
        th_l, om_l = 0.0, 0.0
        e0 = None
        emax = 0.0
        for _ in range(400000):
            tau_s = joint.step(0.0, th_l, om_l, dt)
            om_l += tau_s / j_l * dt
            th_l += om_l * dt
            e = (0.5 * j_m * joint.omega_m ** 2 + 0.5 * j_l * om_l ** 2
                 + 0.5 * k_s * (joint.theta_m - th_l) ** 2)
            if e0 is None:
                e0 = e
            emax = max(emax, e)
        self.assertLess(emax / e0, 1.02, "能量增长超过 2%，积分格式不稳")


class TestNotchFilter(unittest.TestCase):
    """凹口深度 = ζ_n/ζ_d，且离散实现要对得上连续域。"""

    def test_depth_equals_zeta_ratio(self):
        for depth in (0.02, 0.1, 0.3, 1.0):
            n = NotchFilter(freq_hz=30.0, depth=depth, width=0.7, dt=1e-3)
            self.assertAlmostEqual(n.gain_at(30.0), depth, places=9)

    def test_passband_untouched(self):
        """远离凹口的频率增益应当接近 1，否则就不是陷波是低通了。"""
        n = NotchFilter(freq_hz=30.0, depth=0.05, width=0.7, dt=1e-3)
        self.assertGreater(n.gain_at(2.0), 0.97)
        self.assertGreater(n.gain_at(300.0), 0.97)

    def test_discrete_matches_continuous_at_notch(self):
        """喂正弦，测稳态幅值，与连续域 |N(jω)| 比。

        独立性：这里完全不看滤波器系数，只看「输入正弦 → 输出正弦」
        的幅值比，是把它当黑盒测的。
        """
        f0, dt = 30.0, 2e-4
        for depth in (0.05, 0.2):
            n = NotchFilter(freq_hz=f0, depth=depth, width=0.7, dt=dt)
            t = np.arange(0, 4.0, dt)
            y = np.array([n(np.sin(2 * np.pi * f0 * s)) for s in t])
            tail = y[len(y) // 2:]
            got = float(np.max(np.abs(tail)))
            self.assertAlmostEqual(got, depth, delta=0.05 * depth + 2e-3)

    def test_notch_lands_on_design_frequency_at_coarse_rate(self):
        """⭐ 凹口必须真的落在设计频率上——低采样率下这是预畸变的功劳。

        为什么用粗采样率：普通双线性变换会把频率轴按 tan 压缩，
        偏差量约为 ω₀T/2 的三阶项。f₀·dt 很小时 tan(x)≈x，看不出差别；
        只有当 f₀ 接近奈奎斯特时才暴露。真实伺服驱动器正是这种工况：
        控制周期 0.125~1 ms，机械谐振几百 Hz，f₀·dt 一点都不小。

        oracle 独立性：这里**不看滤波器系数**，而是把它当黑盒——
        逐个频率喂纯正弦，量稳态输出幅值，找幅值最低的那个频率。
        """
        dt = 1e-3
        for f0 in (200.0, 350.0):
            n = NotchFilter(freq_hz=f0, depth=0.05, width=0.7, dt=dt)

            def amp(f):
                n.reset()
                t = np.arange(0, 2.0, dt)
                y = [n(np.sin(2 * np.pi * f * s)) for s in t]
                return float(np.max(np.abs(y[len(y) // 2:])))

            scan = np.linspace(0.6 * f0, min(1.3 * f0, 0.49 / dt), 71)
            f_min = float(scan[int(np.argmin([amp(f) for f in scan]))])
            self.assertAlmostEqual(
                f_min, f0, delta=0.02 * f0,
                msg=f"设计 {f0} Hz，实测凹口却在 {f_min:.1f} Hz "
                    f"（偏 {100 * (f_min - f0) / f0:+.1f}%）")

    def test_rejects_above_nyquist(self):
        with self.assertRaises(ValueError):
            NotchFilter(freq_hz=900.0, depth=0.1, width=0.7, dt=1e-3)

    def test_rejects_bad_depth(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                NotchFilter(freq_hz=30.0, depth=bad, width=0.7, dt=1e-3)


class TestInputShaper(unittest.TestCase):
    """整形器必须在设计频率上把残余振动打到 0，且幅值和为 1。"""

    def test_amplitudes_sum_to_one(self):
        """否则终点位置会偏——这是整形器的硬约束。"""
        for kind in ("zv", "zvd"):
            for f in (5.0, 20.0, 60.0):
                a, _ = InputShaper.impulses(kind, f, 0.03)
                self.assertAlmostEqual(float(a.sum()), 1.0, places=12)

    def test_zero_residual_at_design_frequency(self):
        for kind in ("zv", "zvd"):
            for f in (5.0, 20.0, 60.0):
                for z in (0.0, 0.02, 0.15):
                    a, t = InputShaper.impulses(kind, f, z)
                    v = residual_vibration(a, t, f, z)
                    self.assertLess(v, 1e-9, f"{kind} @ {f}Hz ζ={z} V={v}")

    def test_unshaped_has_full_vibration(self):
        """对照组：不整形时 V=1。没有这条，上面那条可能是恒零函数。"""
        a, t = InputShaper.impulses("none", 20.0, 0.02)
        self.assertAlmostEqual(residual_vibration(a, t, 20.0, 0.02), 1.0,
                               places=12)

    def test_zvd_is_more_robust_than_zv(self):
        """频率估错 ±20% 时，ZVD 的残余振动必须明显小于 ZV。

        这是 ZVD 存在的**唯一理由**（它比 ZV 多花半个周期）。
        """
        f_design, z = 20.0, 0.02
        for err in (0.8, 1.2):
            f_true = f_design * err
            a_zv, t_zv = InputShaper.impulses("zv", f_design, z)
            a_zd, t_zd = InputShaper.impulses("zvd", f_design, z)
            v_zv = residual_vibration(a_zv, t_zv, f_true, z)
            v_zd = residual_vibration(a_zd, t_zd, f_true, z)
            self.assertLess(v_zd, 0.5 * v_zv,
                            f"频率误差 {err}: ZVD {v_zd:.4f} 未显著优于 "
                            f"ZV {v_zv:.4f}")

    def test_zvd_costs_one_more_half_period(self):
        f, z = 20.0, 0.02
        lag_zv = InputShaper.impulses("zv", f, z)[1][-1]
        lag_zvd = InputShaper.impulses("zvd", f, z)[1][-1]
        self.assertAlmostEqual(lag_zvd, 2.0 * lag_zv, places=12)

    def test_shaper_suppresses_real_second_order_system(self):
        """端到端：真跑一个二阶系统，比整形前后的残余振幅。

        oracle 独立性：这里既不用 residual_vibration 也不用整形器内部量，
        而是直接数值积分 ẍ + 2ζωẋ + ω²x = ω²u，量到达后的残余摆幅。
        """
        f, z, dt = 8.0, 0.01, 2e-4
        wn = 2 * np.pi * f

        def run(kind):
            sh = InputShaper(kind, f, z, dt)
            x = v = 0.0
            peak_after = 0.0
            t_move = 0.6
            for k in range(int(3.0 / dt)):
                t = k * dt
                u = sh(1.0 if t >= 0.1 else 0.0)
                a = wn * wn * (u - x) - 2 * z * wn * v
                v += a * dt
                x += v * dt
                if t > t_move:
                    peak_after = max(peak_after, abs(x - 1.0))
            return peak_after

        raw = run("none")
        zv = run("zv")
        zvd = run("zvd")
        self.assertGreater(raw, 0.3, "对照组振动太小，实验无意义")
        self.assertLess(zv, 0.05 * raw, f"ZV 只压到 {zv / raw:.3f}")
        self.assertLess(zvd, 0.05 * raw, f"ZVD 只压到 {zvd / raw:.3f}")


class TestCascadeServo(unittest.TestCase):
    """串级本身的性质：饱和、内环滞后、陷波接入。"""

    def test_torque_is_saturated(self):
        s = CascadeServo(kpp=50.0, kvp=5.0, kvi=50.0, tau_max=10.0)
        tau = 0.0
        for _ in range(5000):
            tau = s.update(1.0, 0.0, 0.0, 0.0, 1e-3)
        self.assertLessEqual(abs(tau), 10.0 + 1e-9)

    def test_integrator_does_not_wind_up(self):
        """长期饱和后反向给指令，必须很快换向而不是先吐一大段积分。"""
        s = CascadeServo(kpp=50.0, kvp=5.0, kvi=200.0, tau_max=10.0)
        for _ in range(20000):
            s.update(1.0, 0.0, 0.0, 0.0, 1e-3)
        n = 0
        while n < 2000:
            tau = s.update(-1.0, 0.0, 0.0, 0.0, 1e-3)
            n += 1
            if tau < 0.0:
                break
        self.assertLess(n, 200, f"换向用了 {n} 步，积分饱和了")

    def test_current_loop_lag_is_first_order(self):
        """阶跃响应到 63.2% 的时间应当是 1/(2πf_i)。"""
        f_i, dt = 100.0, 1e-5
        s = CascadeServo(kpp=0.0, kvp=1.0, kvi=0.0, f_current=f_i,
                         tau_max=1e9)
        target = None
        t63 = None
        for k in range(int(0.2 / dt)):
            tau = s.update(0.0, 1.0, 0.0, 0.0, dt)
            if target is None:
                target = 1.0        # kvp*(v_cmd−ω) = 1*(1−0)
            if t63 is None and tau >= 0.632 * target:
                t63 = k * dt
        self.assertIsNotNone(t63)
        self.assertAlmostEqual(t63, 1.0 / (2 * np.pi * f_i),
                               delta=0.06 / (2 * np.pi * f_i))

    def test_notch_is_actually_in_the_loop(self):
        """装上陷波后，谐振频率的误差分量必须被削掉。"""
        dt = 2e-4
        s = CascadeServo(kpp=0.0, kvp=1.0, kvi=0.0, f_current=5000.0,
                         tau_max=1e9)
        f0 = 30.0

        def amp():
            s.reset()
            out = []
            for k in range(int(3.0 / dt)):
                w = np.sin(2 * np.pi * f0 * k * dt)
                out.append(s.update(0.0, w, 0.0, 0.0, dt))
            return float(np.max(np.abs(out[len(out) // 2:])))

        a_raw = amp()
        s.notch = NotchFilter(freq_hz=f0, depth=0.05, width=0.7, dt=dt)
        a_notch = amp()
        self.assertLess(a_notch, 0.15 * a_raw,
                        f"陷波后 {a_notch:.4f} vs 原 {a_raw:.4f}，没削下去")


if __name__ == "__main__":
    unittest.main()
