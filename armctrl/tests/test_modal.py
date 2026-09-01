"""闭环模态频率（第四轮审核 P0-02）。

oracle 独立性说明
----------------
1. **手算的闭式解。** 2×2 的广义特征值问题可以手推，测试里直接写死答案。
2. **数值特征值 vs 时域仿真。** 用独立积分的自由振动响应做 FFT，
   峰值必须落在解析模态频率上——两种**完全不同**的方法互相印证
   （审核报告的验收条件之三）。
3. **对角线法的反例。** 明确构造一个"对角线法说完全一致、真实模态差 1.73 倍"
   的例子，把两种算法的差别钉死。

⚠️ 级别：**理论级**（判据是解析解和独立数值实验）。
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.control.modal import (diagonal_frequencies, modal_frequencies,
                                   modal_spread)


def free_response_peak(M, Kp, dt=5e-4, T=120.0, x0=None):
    """独立方法：直接积分 ``M ẍ + K_p x = 0``，对响应做 FFT 取峰值频率。

    完全不碰 ``modal_frequencies``，用的是显式辛欧拉积分 + numpy 的 FFT。
    """
    M = np.asarray(M, float)
    Kp = np.asarray(Kp, float)
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    x = np.ones(n) if x0 is None else np.asarray(x0, float).copy()
    v = np.zeros(n)
    N = int(T / dt)
    rec = np.empty((N, n))
    for k in range(N):
        rec[k] = x
        v = v - Minv @ (Kp @ x) * dt
        x = x + v * dt
    # ⚠️ 不能用 rec.sum(axis=1)：反对称模态 [1,−1] 会被求和**精确抵消**，
    # 信号恒为 0，FFT 峰值落在直流上。取单个分量才两个模态都看得见。
    sig = rec[:, 0]
    sig = sig - sig.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, dt) * 2.0 * np.pi     # → rad/s
    return freqs, spec


class TestModalCounterexample(unittest.TestCase):
    """⭐⭐ 审核报告 P0-02 给的 2×2 反例，逐位复现。"""

    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    KP = np.eye(2)

    def test_diagonal_method_wrongly_reports_zero_spread(self):
        """对角线法给 [0.8165, 0.8165]，看起来"两个方向完全一致"。"""
        d = diagonal_frequencies(self.M, self.KP)
        np.testing.assert_allclose(d, [np.sqrt(2.0 / 3.0)] * 2, rtol=1e-12)
        self.assertAlmostEqual(float(d.max() / d.min()), 1.0, places=12)

    def test_true_modal_frequencies_are_one_and_one_over_sqrt_three(self):
        """⭐ 真实模态是 [0.5774, 1.0]，**手算可验**。

        M=[[2,1],[1,2]] 的特征向量是 [1,1]/√2（特征值 3）
        和 [1,−1]/√2（特征值 1）。Kp=I，所以

            ω² = 1/3 → ω = 0.5774
            ω² = 1/1 → ω = 1.0000

        两者差 **√3 = 1.732 倍**，而对角线法报告"差 0"。
        """
        w = modal_frequencies(self.M, self.KP)
        np.testing.assert_allclose(np.sort(w),
                                   [1.0 / np.sqrt(3.0), 1.0], rtol=1e-12)
        self.assertAlmostEqual(modal_spread(self.M, self.KP),
                               np.sqrt(3.0), places=12)

    def test_free_vibration_fft_confirms_the_modal_frequencies(self):
        """⭐⭐ 第二种独立方法：时域自由振动 + FFT。

        初值取 [1, −1]（正好是第二个模态），响应应当是**单频**的 1.0 rad/s。
        这条把"广义特征值"和"实际振动"钉在一起。
        """
        modes = [1.0 / np.sqrt(3.0), 1.0]
        for x0, expect in (([1.0, 1.0], modes[0]), ([1.0, -1.0], modes[1])):
            freqs, spec = free_response_peak(self.M, self.KP, x0=x0)
            peak = float(freqs[int(np.argmax(spec))])
            binw = float(freqs[1] - freqs[0])          # FFT 分辨率
            other = modes[1] if expect == modes[0] else modes[0]
            # ① 峰值必须落在正确模态的 1.5 个频率分辨率之内
            self.assertLess(abs(peak - expect), 1.5 * binw,
                            f"初值 {x0}：FFT 峰值 {peak:.4f} rad/s，"
                            f"解析模态 {expect:.4f}，分辨率 {binw:.4f}")
            # ② 而且必须离"正确的那个"明显比离"另一个"更近 —— 这才有区分能力
            self.assertLess(abs(peak - expect), 0.4 * abs(peak - other),
                            f"初值 {x0}：峰值 {peak:.4f} 分不清 "
                            f"{expect:.4f} 和 {other:.4f}")

    def test_the_two_methods_disagree_by_more_than_measurement_noise(self):
        """⭐ 变异守卫：如果有人把 modal_frequencies 换回对角线实现，必须失败。"""
        w = np.sort(modal_frequencies(self.M, self.KP))
        d = np.sort(diagonal_frequencies(self.M, self.KP))
        self.assertGreater(float(np.abs(w - d).max()), 0.2,
                           "两种算法给出几乎相同的结果，说明实现被换掉了")


class TestModalProperties(unittest.TestCase):
    """一般性质。"""

    def test_diagonal_mass_makes_both_methods_agree(self):
        """⭐ 对照组：M 是对角阵（关节之间无耦合）时，两法必须一致。

        这条说明对角线法**不是"总是错"**，而是"丢掉耦合项"。
        没有耦合的时候它是对的——这正好定位了它错在哪。
        """
        M = np.diag([2.0, 5.0, 0.5])
        Kp = np.diag([10.0, 10.0, 10.0])
        np.testing.assert_allclose(np.sort(modal_frequencies(M, Kp)),
                                   np.sort(diagonal_frequencies(M, Kp)),
                                   rtol=1e-12)

    def test_scalar_and_vector_and_matrix_kp_agree(self):
        M = np.array([[3.0, 0.4], [0.4, 1.0]])
        a = modal_frequencies(M, 7.0)
        b = modal_frequencies(M, np.array([7.0, 7.0]))
        c = modal_frequencies(M, 7.0 * np.eye(2))
        np.testing.assert_allclose(a, b, rtol=1e-12)
        np.testing.assert_allclose(a, c, rtol=1e-12)

    def test_frequencies_scale_as_sqrt_of_stiffness(self):
        """ω ∝ √Kp：刚度翻 4 倍，频率翻 2 倍。"""
        M = np.array([[2.0, 0.7], [0.7, 1.5]])
        w1 = modal_frequencies(M, 1.0)
        w4 = modal_frequencies(M, 4.0)
        np.testing.assert_allclose(w4, 2.0 * w1, rtol=1e-12)

    def test_frequencies_scale_as_inverse_sqrt_of_mass(self):
        """ω ∝ 1/√M：质量翻 4 倍，频率减半。"""
        M = np.array([[2.0, 0.7], [0.7, 1.5]])
        np.testing.assert_allclose(modal_frequencies(4.0 * M, 1.0),
                                   0.5 * modal_frequencies(M, 1.0), rtol=1e-12)

    def test_output_is_sorted_ascending(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            A = rng.normal(0, 1, (4, 4))
            M = A @ A.T + 4.0 * np.eye(4)
            w = modal_frequencies(M, 3.0)
            self.assertTrue(np.all(np.diff(w) >= -1e-12), f"没有升序：{w}")


class TestTrackingSceneModalReadouts(unittest.TestCase):
    """⭐ 场景级：修完之后读数讲的是不是真话。

    ⚠️ 这一组是审核报告 P0-02 验收条件的落点：
    "重新生成 32.7 vs 20 等带宽数字"。
    """

    @staticmethod
    def _tel(**params):
        from armctrl.tuner.scenes import TrackingScene
        sc = TrackingScene()
        sc.build()
        sc.reset()
        for k, v in params.items():
            sc.set(k, v)
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc._wn_tick = 0
        sc.control(0.0, q, v)
        return sc.telemetry(0.0, q, v)

    def test_ctc_is_decoupled_but_not_perfectly(self):
        """⭐⭐ CTC 解耦，但**不是精确的 0**——这一条本身就是 P0-02 的证据。

        修复前 ``_measure_wn`` 用**控制器自己的模型**当被控对象，
        于是 CTC 分支代数上**必然**得到 ``wn_spread == 0.000``。
        那个漂亮的 0 不是测出来的，是**算术恒等式**。

        现在被控对象换成 MuJoCo（``mj_fullM`` / ``qfrc_bias``），
        控制器仍用自己的 regressor + π̂，两边彻底分开。
        于是残余离散度变成 **5.1e-4 rad/s**——很小，但**不是零**，
        它量的是 regressor 模型和 MuJoCo 之间的真实差距（‖ΔM‖ ≈ 4.8e-5）。

        ⭐ **一个恒等于 0 的指标，往往不是"做得完美"，而是"根本没在量"。**

        ⚠️⚠️ 2026-08-31（第五轮批次 D）：**这个测试自己犯了同一个错。**
        它对 ``wn_spread`` 正确地要求了"必须 > 1e-9"，
        却在**上一行**对 ``modal_spread`` 要求"精确等于 1.0（6 位）"——
        而当时 `scenes.py` 里 CTC 的模态频率就是写死的 ``np.full(√Kp)``。
        ⭐⭐⭐ **也就是说，这一行断言正在把 bug 锁住。**
        同一个方法里，一行在防这个病，另一行在保护这个病。

        现在 ``modal_spread`` 也真的解 $\det(K_pM_{reg}-\omega^2M_{true})=0$ 了，
        残余是 **1.0000225**（同样来自 ‖ΔM‖ ≈ 4.8e-5）。
        两个读数于是用**同一条纪律**验收：接近理论值，但**不等于**理论值。
        """
        t = self._tel(controller="ctc")
        # 等效增益路径：接近 0，但不能精确是 0
        self.assertLess(t["wn_spread"], 1e-2,
                        "CTC 本该基本解耦")
        self.assertGreater(t["wn_spread"], 1e-9,
                           "离散度精确为 0 —— 说明被控对象又变回控制器自己的模型了")
        # 模态路径：接近 1，但不能精确是 1（同一条纪律）
        self.assertAlmostEqual(t["modal_spread"], 1.0, delta=1e-3)
        self.assertNotAlmostEqual(
            t["modal_spread"], 1.0, places=6,
            msg="模态离散度精确等于 1 —— 说明它又变回写死的 √Kp 了")

    def test_pd_looks_faster_by_median_but_its_slowest_mode_is_slower(self):
        """⭐⭐ **本轮最值得记的一个数字修正。**

        旧笔记拿"PD 等效带宽 32.7 rad/s vs CTC 20 rad/s"说事，
        听上去 PD 更快。但那是**逐关节等效增益的中位数**。

        真实的最慢模态（决定整体响应速度的那个）：

        | 控制器 | 中位等效增益 | **最慢模态** | 模态离散度 |
        |---|---|---|---|
        | CTC | 20.00 | **20.00** | 1.00 |
        | PD + 重力补偿 | 32.71 | **12.50** | 4.48 |

        ⭐ **PD 最慢的方向比 CTC 慢 38%，不是快 64%。**
        """
        ctc = self._tel(controller="ctc")
        pd = self._tel(controller="pd_gravity", align="off")
        self.assertGreater(pd["wn_med"], ctc["wn_med"],
                           "按中位等效增益，PD 应当看起来更快")
        self.assertLess(pd["modal_min"], ctc["modal_min"],
                        f"PD 最慢模态 {pd['modal_min']:.2f} 却不比 CTC "
                        f"{ctc['modal_min']:.2f} 慢——该重查这个对照")
        self.assertGreater(pd["modal_spread"], 3.0,
                           f"PD 模态离散度只有 {pd['modal_spread']:.2f}，"
                           "本该是 4.48 量级")

    def test_gain_alignment_cannot_fix_the_modal_spread(self):
        """⭐ 对齐只能挪中位数，**改不了离散度**——这正是 CTC 存在的理由。"""
        on = self._tel(controller="pd_gravity", align="on")
        off = self._tel(controller="pd_gravity", align="off")
        self.assertAlmostEqual(on["modal_spread"], off["modal_spread"], places=6)
        self.assertLess(on["wn_med"], off["wn_med"],
                        "对齐之后中位等效增益本该被拉下来")


if __name__ == "__main__":
    unittest.main()
