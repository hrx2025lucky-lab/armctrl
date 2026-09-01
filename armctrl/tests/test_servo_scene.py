"""ServoScene 的**测量管线与振动抑制**行为测试。

这一组测的不是 `control/servo.py` 里的公式（那些在 test_servo.py），
而是把它们接进 MuJoCo 闭环之后的行为：

1. probe 辨识模式量出来的谐振频率，是不是真的等于双质量系统的机械谐振；
2. 这个读数是不是**只由机械决定**——换伺服增益不该让它变；
3. 陷波滤波器和输入整形器，是不是真的把谐振带里的振动压下去了。

被它们钉死的四个真实缺陷
------------------------
1. **拿 ω_m 做 FFT**：ω_m/τ_m 的传递函数带一个零点（反谐振），
   频谱是「先一个凹口再一个峰」，凹口比峰更显眼，argmax 会挑到凹口边缘。
   必须改测弹簧形变 δ = θ_m − θ_l，它的传递函数只有极点没有零点。
2. **邻居关节用普通 PD 锁**：带宽只有 4.8 Hz，比谐振还低，邻居跟着一起晃，
   J4 感受到的有效惯量不再是 M[j,j]，f_meas 系统性偏 16%。
3. **单次 FFT**：每根谱线的方差等于均值、信噪比恒为 1，峰不够尖时
   argmax 挑到的是噪声。必须做 Welch 分段平均。
4. **振动读数用一阶高通**：外环位置环的低频晃动（≈0.9 Hz）和阶跃尖角
   大量漏过去，导致「陷波打开后谐振带能量降到 12.5%，读数反而升到 113%」
   ——读数和真实效果相反。必须换成中心在 f_res 的二阶带通。

oracle 独立性
------------
- 谐振频率用**状态矩阵的特征值**独立算（下面的 `two_mass_resonance_hz`），
  不调用 `control/servo.py` 的 `resonance_frequency()`，也不读场景的
  `_resonance_hz()`。两者只共享「双质量系统的物理定义」这一个前提。
- 振动强度用测试自己对 ω_m 序列做的 FFT 带内能量算，
  不读场景的 `_vib` / `vib_amp`。
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from armctrl.tuner.scenes import SCENES

_SERVO = [s for s in SCENES if s.name == "servo"][0]


def two_mass_resonance_hz(j_m, j_l, k_s):
    """双质量系统的机械谐振频率（独立 oracle，走状态矩阵特征值）。

    运动方程（无阻尼）：
        J_m·θ̈_m = −K_s·(θ_m − θ_l)
        J_l·θ̈_l = +K_s·(θ_m − θ_l)
    取状态 x = [θ_m, θ_l, ω_m, ω_l]，写成 ẋ = A·x 后求 A 的特征值。
    非零那一对纯虚特征值 ±jω 的模就是谐振角频率。

    这条路径与解析公式 √(K_s(J_m+J_l)/(J_m·J_l)) **没有共享任何代码**：
    它只用了牛顿第二定律和 numpy 的特征值分解。
    """
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2, 0] = -k_s / j_m
    A[2, 1] = +k_s / j_m
    A[3, 0] = +k_s / j_l
    A[3, 1] = -k_s / j_l
    w = np.abs(np.linalg.eigvals(A))
    return float(np.max(w) / (2.0 * np.pi))


def peak_tolerance(f_res, seg=1024, dt=0.002):
    """这个测量方法**本身**能达到的精度，作为容差依据。

    FFT 的频率分辨率是 Δf = 1/(N·dt) = 1/(1024×0.002) = 0.488 Hz。
    不做插值时峰位精度就是 ±0.5 个谱线宽（插值对不对称峰有害，见 scenes.py）。
    所以容许的相对误差
    随频率变化：频率越低，一个谱线占的比例越大。
        tol = max(2%, 0.7·Δf / f_res)
    2% 的下限留给阻尼峰移和半隐式欧拉的离散化偏差。

    ⭐ 这不是「调到能过为止」的数字，而是从采样定理推出来的上限：
    低于它的差别，这套测量方法**在原理上就分辨不出来**。
    高频工况因此被卡得比低频严得多（20 Hz 处只有 2%）。
    """
    df = 1.0 / (seg * dt)
    return max(0.02, 0.7 * df / f_res)


def roll_scene(scene, seconds, record_omega_m=False):
    """把场景跑 `seconds` 秒，可选地记录电机侧速度序列。"""
    d, r = scene.data, scene.robot
    n = int(seconds / scene.dt)
    buf = np.zeros(n) if record_omega_m else None
    t = 0.0
    for k in range(n):
        t = k * scene.dt
        q = d.qpos[r.qpos_idx][:7]
        v = d.qvel[r.qvel_idx][:7]
        d.ctrl[r.arm_actuator_ids] = scene.control(t, q, v)
        mujoco.mj_step(scene.model, d)
        if record_omega_m:
            buf[k] = scene.joint.omega_m
    return t, buf


def make_scene(**overrides):
    sc = _SERVO()
    for key, val in overrides.items():
        sc.set(key, val)
    sc.build()
    sc.reset()
    return sc


def measured_resonance(**overrides):
    """跑 probe 辨识模式，返回 (量出来的 f_meas, oracle 算出来的 f_res)。"""
    overrides = dict(overrides)
    overrides["move"] = "probe"
    sc = make_scene(**overrides)
    t, _ = roll_scene(sc, 22.0)
    d, r = sc.data, sc.robot
    tel = sc.telemetry(t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
    oracle = two_mass_resonance_hz(sc.joint.j_m, sc._j_load(), sc.get("stiffness"))
    return float(tel["f_meas"]), oracle


def resonant_band_energy(**overrides):
    """point 阶跃模式下，电机速度频谱在谐振带内的能量（独立 oracle）。

    带宽取 (0.75~1.35)·f_res：足够宽以容纳阻尼展宽和整形器的频率误差，
    又足够窄以排除外环位置环（≈0.9 Hz）和奈奎斯特附近的数值噪声。
    """
    overrides = dict(overrides)
    overrides.setdefault("move", "point")
    sc = make_scene(**overrides)
    _, buf = roll_scene(sc, 14.0, record_omega_m=True)
    n0 = int(3.0 / sc.dt)                     # 丢掉起步暂态
    x = buf[n0:] - buf[n0:].mean()
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), sc.dt)
    f_res = two_mass_resonance_hz(sc.joint.j_m, sc._j_load(), sc.get("stiffness"))
    band = (freqs > 0.75 * f_res) & (freqs < 1.35 * f_res)
    return float(power[band].sum())


class TestResonanceOracle(unittest.TestCase):
    """先证明 oracle 自己是对的（否则后面全白搭）。"""

    def test_eigenvalue_oracle_matches_hand_derivation(self):
        # 手推：ω = √(K_s·(J_m+J_l)/(J_m·J_l))，这里用具体数字独立核对
        j_m, j_l, k_s = 0.3, 1.114, 800.0
        hand = np.sqrt(k_s * (j_m + j_l) / (j_m * j_l)) / (2.0 * np.pi)
        self.assertAlmostEqual(two_mass_resonance_hz(j_m, j_l, k_s), hand, places=9)

    def test_oracle_scales_as_sqrt_stiffness(self):
        # 刚度乘 4，频率应当正好乘 2
        a = two_mass_resonance_hz(0.3, 1.114, 400.0)
        b = two_mass_resonance_hz(0.3, 1.114, 1600.0)
        self.assertAlmostEqual(b / a, 2.0, places=9)


class TestProbeIdentification(unittest.TestCase):
    """probe 辨识模式量出来的，必须是**机械**谐振。"""

    def test_probe_measures_mechanical_resonance(self):
        """跨多个工况都要对得上。

        只测默认工况是不够的：默认点上「测 ω_m 还是测 δ」「邻居硬锁还是软锁」
        「Welch 平均还是单次 FFT」恰好都能蒙对，只有把 K_s、J_m 拉开
        才分得出来。
        """
        cases = [
            ("默认", {}),
            ("K_s=200 很软", dict(stiffness=200.0)),
            ("K_s=400 软传动", dict(stiffness=400.0)),
            ("K_s=1600 硬传动", dict(stiffness=1600.0)),
            ("K_s=3000", dict(stiffness=3000.0)),
            ("K_s=6000 很硬", dict(stiffness=6000.0)),
            ("J_m=0.05 大惯量比", dict(j_motor=0.05)),
            ("J_m=0.10", dict(j_motor=0.10)),
            ("J_m=0.20", dict(j_motor=0.20)),
            ("J_m=0.40 小惯量比", dict(j_motor=0.40)),
        ]
        for tag, kw in cases:
            with self.subTest(tag):
                f_meas, f_res = measured_resonance(**kw)
                self.assertGreater(f_meas, 0.0, f"{tag}：probe 模式没量到任何峰")
                rel = abs(f_meas - f_res) / f_res
                tol = peak_tolerance(f_res)
                self.assertLess(rel, tol,
                                f"{tag}：量到 {f_meas:.2f} Hz，"
                                f"独立 oracle 给 {f_res:.2f} Hz，"
                                f"差 {rel * 100:.1f}%（容差 {tol * 100:.1f}%）")

    def test_reading_is_independent_of_servo_gain(self):
        """换速度环增益不该改变机械谐振——这是区分「真谐振」和
        「滤波器主导的极限环」的判据。极限环的频率会跟着增益跑。"""
        low, _ = measured_resonance(kvp=3.0)
        high, _ = measured_resonance(kvp=45.0)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, 0.0)
        rel = abs(high - low) / low
        self.assertLess(rel, 0.03,
                        f"Kvp 3→45 让读数从 {low:.2f} 变到 {high:.2f} Hz "
                        f"（差 {rel * 100:.1f}%），说明量到的不是机械谐振")

    def test_reading_tracks_stiffness_as_sqrt(self):
        """刚度乘 4，量出来的频率应当接近乘 2。"""
        soft, _ = measured_resonance(stiffness=400.0)
        stiff, _ = measured_resonance(stiffness=1600.0)
        self.assertGreater(soft, 0.0)
        self.assertGreater(stiff, 0.0)
        self.assertAlmostEqual(stiff / soft, 2.0, delta=0.25,
                               msg=f"K_s 400→1600 时读数 {soft:.2f}→{stiff:.2f} Hz，"
                                   f"比值 {stiff / soft:.2f} 偏离 2.0 太多")


class TestVibrationSuppression(unittest.TestCase):
    """陷波与输入整形必须真的把谐振带里的能量压下去。"""

    @classmethod
    def setUpClass(cls):
        cls.base = resonant_band_energy()
        cls.notch = resonant_band_energy(notch="on")
        cls.zv = resonant_band_energy(shaper="zv")
        cls.zvd = resonant_band_energy(shaper="zvd")

    def test_baseline_actually_rings(self):
        """基线必须真的有谐振，否则「压下去了」毫无意义。"""
        quiet = resonant_band_energy(move="hold")
        self.assertGreater(self.base, 20.0 * quiet,
                           "阶跃指令没有激起谐振，抑制实验不成立")

    def test_notch_cuts_resonant_band(self):
        ratio = self.notch / self.base
        self.assertLess(ratio, 0.40,
                        f"陷波打开后谐振带能量只降到 {ratio * 100:.1f}%")

    def test_zv_shaper_cuts_resonant_band(self):
        ratio = self.zv / self.base
        self.assertLess(ratio, 0.40,
                        f"ZV 整形后谐振带能量只降到 {ratio * 100:.1f}%")

    def test_zvd_is_at_least_as_deep_as_zv(self):
        """ZVD 多用一个脉冲换鲁棒性，在频率估得准时抑制深度不应更差。"""
        self.assertLessEqual(self.zvd, self.zv * 1.2,
                             f"ZVD({self.zvd:.3e}) 明显比 ZV({self.zv:.3e}) 还差")

    def test_readout_agrees_with_independent_spectrum(self):
        """调参台上显示的 `vib_amp` 读数，必须和独立频谱给出**同样的排序**。

        钉住的缺陷：原来 `_vib` 用一阶高通（截止 6.3 Hz），外环位置环
        ≈0.9 Hz 的晃动和阶跃尖角大量漏过去，于是「陷波打开后谐振带能量
        降到 12.5%、读数反而升到 113%」——使用者照着读数调，会把对的
        设置当成错的。
        """
        def readout(**kw):
            kw = dict(kw)
            kw.setdefault("move", "point")
            sc = make_scene(**kw)
            d, r = sc.data, sc.robot
            n0 = int(3.0 / sc.dt)
            acc = []
            for k in range(int(14.0 / sc.dt)):
                t = k * sc.dt
                d.ctrl[r.arm_actuator_ids] = sc.control(
                    t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
                mujoco.mj_step(sc.model, d)
                if k >= n0:
                    acc.append(sc._vib)
            return float(np.mean(acc))

        base = readout()
        for tag, kw in (("陷波", dict(notch="on")),
                        ("ZV", dict(shaper="zv")),
                        ("ZVD", dict(shaper="zvd"))):
            self.assertLess(readout(**kw), base,
                            f"{tag} 明明把谐振带能量压下去了，vib_amp 读数却没降")


if __name__ == "__main__":
    unittest.main()
