"""轨迹精度补偿（ILC + 关节柔度）与动态避障（在线重规划）的测试。

oracle 独立性说明
----------------
判据分三类，都不读被测实现的中间量：

1. **闭式收敛理论。**ILC 的误差每遍乘上收缩因子 ``|1 − L·P|``，
   这是从更新律解析推出来的，与实现无关。测试用一个**独立搭的一阶被控对象**
   跑多遍，把实测的逐遍误差比与它对照。
2. **外部可观测事实。**柔度补偿对不对，只看「补偿后的末端误差是不是比
   不补偿小」——不看补偿量本身算得对不对。符号写反时误差会**翻倍**，
   这一条立刻失败。
3. **几何真值。**重规划的前瞻检测用解析距离独立复算，
   不读监督器内部缓存。

⚠️ 全部为**同模型自洽**级别：被控对象是我们自己搭的简化模型，不是真机。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

from armctrl.control.ilc import (IterativeLearningController,
                                 JointComplianceModel, zero_phase_lowpass)
from armctrl.planning.replan import MovingSphere, ReplanSupervisor


class FirstOrderPlant:
    """独立搭的一阶被控对象，**不引用 armctrl 的任何控制代码**。

        y[i] = plant_gain · u[i − delay] + d[i]

    ``d`` 是**每遍都一样**的扰动——它就是 ILC 要学掉的那个「重复性误差」，
    物理上对应关节柔度形变、几何误差这类东西。

    ⚠️ 这一项不能少：没有扰动时 y ≡ ref，误差从第 0 遍就是零，
    「学习」无从谈起。（这个坑是写测试时踩到的：最初漏了 d，
    全部收敛判据都退化成 0/0。）

    ``delay`` 是纯延迟，用来检验 ILC 的「时间超前」那一项确实必要。
    """

    def __init__(self, n_steps: int, n_joints: int, gain: float = 1.0,
                 delay: int = 0, disturbance: np.ndarray | None = None):
        self.n_steps, self.n_joints = n_steps, n_joints
        self.gain, self.delay = float(gain), int(delay)
        if disturbance is None:
            t = np.linspace(0.0, 1.0, n_steps)
            disturbance = np.stack(
                [0.02 * np.sin(2 * np.pi * (j + 1) * t) + 0.01
                 for j in range(n_joints)], axis=1)
        self.d = np.asarray(disturbance, float).reshape(n_steps, n_joints)

    def run(self, cmd: np.ndarray) -> np.ndarray:
        y = self.d.copy()
        for i in range(self.n_steps):
            j = i - self.delay
            if j >= 0:
                y[i] += self.gain * cmd[j]
        return y


class TestZeroPhaseFilter(unittest.TestCase):
    """Q-filter 的两条性质：低通、且**零相位**。"""

    def test_passes_constant_unchanged(self):
        """常值信号过低通不该有任何变化（直流增益为 1）。"""
        x = np.full((200, 3), 2.5)
        self.assertLess(np.abs(zero_phase_lowpass(x, 0.1) - 2.5).max(), 1e-9)

    def test_attenuates_high_frequency_more_than_low(self):
        """高频衰减必须比低频狠——这是「低通」的定义。"""
        n = 400
        t = np.arange(n)
        lo = np.sin(2 * np.pi * t / 200.0).reshape(n, 1)
        hi = np.sin(2 * np.pi * t / 6.0).reshape(n, 1)
        keep_lo = np.std(zero_phase_lowpass(lo, 0.2)) / np.std(lo)
        keep_hi = np.std(zero_phase_lowpass(hi, 0.2)) / np.std(hi)
        self.assertGreater(keep_lo, 0.8, "低频不该被滤掉")
        self.assertLess(keep_hi, 0.2, "高频没被滤掉，不是低通")

    def test_is_zero_phase(self):
        """⭐ 零相位：滤波后峰值位置不移动。

        独立判据——用互相关的峰值位置判断有没有整体平移，
        不看滤波器是怎么实现的。
        """
        n = 400
        x = np.zeros((n, 1))
        x[150:250, 0] = np.hanning(100)          # 一个对称的包
        y = zero_phase_lowpass(x, 0.15)
        # 质心位置不应移动（对称信号过零相位滤波仍对称）
        idx = np.arange(n)
        c_x = float((idx * x[:, 0]).sum() / x[:, 0].sum())
        c_y = float((idx * y[:, 0]).sum() / y[:, 0].sum())
        self.assertLess(abs(c_x - c_y), 0.5,
                        f"峰值移动了 {abs(c_x - c_y):.2f} 步，不是零相位")

    def test_causal_filter_would_shift(self):
        """反证：只做**单向**滤波一定会移位。

        这一条不测被测实现，而是说明「为什么必须反向再滤一次」。
        """
        n, a = 400, 0.15
        x = np.zeros(n)
        x[150:250] = np.hanning(100)
        out, acc = np.empty(n), x[0]
        for i in range(n):
            acc += a * (x[i] - acc)
            out[i] = acc
        idx = np.arange(n)
        shift = abs(float((idx * out).sum() / out.sum())
                    - float((idx * x).sum() / x.sum()))
        self.assertGreater(shift, 2.0,
                           "单向滤波居然没有移位，说明这条反证构造错了")


class TestIlcConvergence(unittest.TestCase):
    """ILC 的收敛行为，判据全部来自闭式收缩因子。"""

    N, J = 300, 2

    def _repeat(self, plant, ilc, ref, n_trials):
        rms = []
        for _ in range(n_trials):
            y = plant.run(ref + ilc.correction())
            e = ref - y
            rms.append(ilc.update(e))
        return rms

    def test_error_shrinks_by_the_theoretical_contraction_factor(self):
        """⭐ 逐遍误差比应当接近 |1 − L·P|。

        取 L=0.5、P=1 ⇒ 因子 0.5：每遍误差应当**减半**。
        这个数字完全由更新律解析给出，不涉及实现细节。

        ⚠️ **适用范围**：这里 ``q_alpha=1.0``，即 Q=I、完全不滤波，
        此时完整递推 ``e_{k+1}=(I−Q)(r−d)+Q(I−LP)e_k`` 的仿射项
        ``(I−Q)(r−d)`` 恰好为零，误差才真的按 ρ^k 衰减到 0。
        ``q_alpha<1`` 时不成立，见
        :meth:`test_q_filter_leaves_a_nonzero_residual_floor`。
        """
        ref = np.tile(np.sin(np.linspace(0, 4 * np.pi, self.N)), (self.J, 1)).T
        plant = FirstOrderPlant(self.N, self.J, gain=1.0, delay=0)
        ilc = IterativeLearningController(self.N, self.J, gain=0.5,
                                          q_alpha=1.0, lead=0, limit=10.0)
        rms = self._repeat(plant, ilc, ref, 6)
        theory = IterativeLearningController.contraction_factor(0.5, 1.0)
        self.assertAlmostEqual(theory, 0.5, places=12)
        for k in range(1, 5):
            self.assertAlmostEqual(rms[k] / rms[k - 1], theory, delta=0.02,
                                   msg=f"第 {k} 遍误差比与理论收缩因子不符")

    def test_perfect_gain_converges_in_one_trial(self):
        """L·P = 1 ⇒ 因子 0 ⇒ 一遍到位。

        ⚠️ **「到位」不等于「归零」**：这里 ``q_alpha=1.0``（Q=I），
        仿射项 ``(I−Q)(r−d)`` 才恰好为零，所以能断言 rms[1]≈0。
        ``q_alpha<1`` 时一遍之后会**冻在残差地板上**，
        见 :meth:`test_q_filter_leaves_a_nonzero_residual_floor`。
        """
        ref = np.tile(np.linspace(0, 1, self.N), (self.J, 1)).T
        plant = FirstOrderPlant(self.N, self.J, gain=2.0)
        ilc = IterativeLearningController(self.N, self.J, gain=0.5,
                                          q_alpha=1.0, lead=0, limit=10.0)
        rms = self._repeat(plant, ilc, ref, 3)
        self.assertLess(rms[1], 1e-9 + 1e-6 * rms[0],
                        f"应当一遍学完，实际 {rms[0]:.3e} → {rms[1]:.3e}")
        # ⭐ 这组取值（L=0.5、P=2）能区分 |1−LP| 与 |LP|：前者 0，后者 1。
        # 变异测试暴露出的洞：早先只用 L=0.5、P=1，两个公式碰巧都等于 0.5，
        # 把 contraction_factor 写成 |LP| 也照样通过。
        self.assertAlmostEqual(
            IterativeLearningController.contraction_factor(0.5, 2.0), 0.0,
            places=12, msg="收缩因子公式应为 |1 − L·P|")
        self.assertAlmostEqual(
            IterativeLearningController.contraction_factor(0.25, 2.0), 0.5,
            places=12)

    def test_q_filter_leaves_a_nonzero_residual_floor(self):
        """⭐⭐ 完整递推 e_{k+1} = (I−Q)(r−d) + Q(I−LP)e_k 的仿射项确实存在。

        文档早期版本只写 ``Q(I−LP)e_k``，于是得出「误差指数衰减到 0」——**错的**。

        判据独立性：取 L·P = 1 让收缩因子 ``Q(I−LP)`` 恰好归零，递推**一步到位**，
        解析预测 ``e_1 = (I−Q)(r−d)`` 且此后**恒定不变**。
        预测值由 :func:`zero_phase_lowpass` 直接作用在扰动上算出，
        与 ILC 实现的任何中间量无关。
        """
        n, alpha = 256, 0.15
        t = np.arange(n)
        # 高频可重复扰动：Q 挡得住的频段，(I−Q) 几乎原样保留
        d = np.stack([np.sin(2 * np.pi * 100 * t / n)] * self.J, axis=1)
        ref = np.zeros((n, self.J))
        plant = FirstOrderPlant(n, self.J, gain=1.0, delay=0, disturbance=d)
        ilc = IterativeLearningController(n, self.J, gain=1.0, q_alpha=alpha,
                                          lead=0, limit=1e9)
        rms = self._repeat(plant, ilc, ref, 12)

        predicted = float(np.sqrt(np.mean(
            (d - zero_phase_lowpass(d, alpha)) ** 2)))
        self.assertGreater(predicted, 0.5 * rms[0],
                           "判据自检：高频扰动应当几乎学不掉，否则本例选错了频段")
        self.assertAlmostEqual(rms[1], predicted, places=9,
                               msg="一遍之后的误差应当等于 (I−Q)(r−d)")
        self.assertAlmostEqual(rms[11], predicted, places=9,
                               msg="此后应当冻住不动，而不是继续衰减到 0")
        self.assertGreater(rms[11], 1e-3,
                           "残差地板必须非零——这正是文档漏掉的那一项")

    def test_q_filter_has_unit_dc_gain_so_low_frequency_is_fully_learnable(self):
        """⭐ 与上一条配对，证明判据有判别力而不是"恒报非零"。

        :func:`zero_phase_lowpass` 用 ``acc = sig[0]`` 初始化，
        所以**直流增益恰为 1**，``(I−Q)`` 把直流完全抹掉。
        于是同一套装置换成直流扰动，残差必须回到 0。
        """
        n, alpha = 256, 0.15
        d = np.full((n, self.J), 0.03)                  # 纯直流可重复扰动
        ref = np.zeros((n, self.J))
        plant = FirstOrderPlant(n, self.J, gain=1.0, delay=0, disturbance=d)
        ilc = IterativeLearningController(n, self.J, gain=1.0, q_alpha=alpha,
                                          lead=0, limit=1e9)
        rms = self._repeat(plant, ilc, ref, 4)
        self.assertAlmostEqual(float(np.abs(
            zero_phase_lowpass(np.ones((n, 1)), alpha) - 1.0).max()), 0.0,
            places=12, msg="Q 的直流增益应当恰为 1")
        self.assertLess(rms[1], 1e-12 + 1e-9 * rms[0],
                        f"直流分量应当被学干净，实际 {rms[0]:.3e} → {rms[1]:.3e}")

    def test_affine_recursion_matches_the_scalar_counterexample(self):
        """⭐ 审核报告的验收反例：P=L=1, Q=0.5, r−d=1, e0=1 ⇒ e1=0.5。

        文档的残缺递推 ``e_{k+1}=Q(I−LP)e_k`` 会预测 e1=0，与之矛盾。
        这里直接对**递推式本身**做数值验证，并核对解析不动点。
        """
        Q, P, L, rd = 0.5, 1.0, 1.0, 1.0
        e = 1.0
        seq = [e]
        for _ in range(30):
            e = (1 - Q) * rd + Q * (1 - P * L) * e
            seq.append(e)
        self.assertAlmostEqual(seq[1], 0.5, places=12,
                               msg="完整递推必须给出 e1 = 0.5")
        fixed = (1 - Q) * rd / (1 - Q * (1 - P * L))
        self.assertAlmostEqual(fixed, 0.5, places=12)
        self.assertAlmostEqual(seq[-1], fixed, places=12,
                               msg="应当验证渐近误差，而不只是相邻两遍的收缩")
        self.assertNotAlmostEqual(seq[1], 0.0, places=6,
                                  msg="残缺递推预测的 0 必须被排除")

    def test_too_large_gain_diverges(self):
        """⭐ L·P > 2 ⇒ 因子 > 1 ⇒ **发散**。

        「学习增益越大越好」是错的，这条测试把它钉死。
        """
        ref = np.tile(np.sin(np.linspace(0, 4 * np.pi, self.N)), (self.J, 1)).T
        plant = FirstOrderPlant(self.N, self.J, gain=1.0)
        ilc = IterativeLearningController(self.N, self.J, gain=2.5,
                                          q_alpha=1.0, lead=0, limit=1e9)
        rms = self._repeat(plant, ilc, ref, 5)
        self.assertGreater(rms[-1], rms[0],
                           "增益超过 2/P 却没有发散，收敛条件判断错了")
        theory = IterativeLearningController.contraction_factor(2.5, 1.0)
        self.assertGreater(theory, 1.0)

    def test_lead_compensation_helps_when_plant_has_delay(self):
        """⭐ 被控对象有纯延迟时，**时间超前**必须能改善收敛。

        对照组：同样的增益与滤波，只把 lead 设成 0。
        """
        ref = np.tile(np.sin(np.linspace(0, 6 * np.pi, self.N)), (self.J, 1)).T
        delay = 5

        def run(lead):
            plant = FirstOrderPlant(self.N, self.J, gain=1.0, delay=delay)
            ilc = IterativeLearningController(self.N, self.J, gain=0.5,
                                              q_alpha=0.6, lead=lead, limit=10.0)
            return self._repeat(plant, ilc, ref, 8)[-1]

        with_lead = run(delay)
        without = run(0)
        self.assertLess(with_lead, without,
                        f"超前补偿反而更差：{with_lead:.4e} vs {without:.4e}")

    def test_q_filter_limits_what_can_be_learned(self):
        """⭐ Q-filter 的代价：**高频误差学不掉**。

        参考信号是纯高频时，滤得越狠残差越大。
        这不是缺陷，是这个方法的固有上限——测试把它写成明确的判据。
        """
        ref = np.tile(np.sin(np.linspace(0, 60 * np.pi, self.N)), (self.J, 1)).T
        out = {}
        for a in (1.0, 0.05):
            plant = FirstOrderPlant(self.N, self.J, gain=1.0)
            ilc = IterativeLearningController(self.N, self.J, gain=0.5,
                                              q_alpha=a, lead=0, limit=10.0)
            out[a] = self._repeat(plant, ilc, ref, 10)[-1]
        self.assertLess(out[1.0], out[0.05],
                        "滤得更狠却学得更好，与 Q-filter 的作用相反")

    def test_correction_is_clipped(self):
        """限幅必须真的生效，否则发散时会把指令送到天上。"""
        ref = np.tile(np.ones(self.N) * 100.0, (self.J, 1)).T
        plant = FirstOrderPlant(self.N, self.J, gain=0.01)
        ilc = IterativeLearningController(self.N, self.J, gain=1.0,
                                          q_alpha=1.0, lead=0, limit=0.2)
        self._repeat(plant, ilc, ref, 10)
        self.assertLessEqual(float(np.abs(ilc.correction()).max()), 0.2 + 1e-12)

    def test_convergence_ratio_reports_first_to_last(self):
        ref = np.tile(np.sin(np.linspace(0, 4 * np.pi, self.N)), (self.J, 1)).T
        plant = FirstOrderPlant(self.N, self.J, gain=1.0)
        ilc = IterativeLearningController(self.N, self.J, gain=0.5,
                                          q_alpha=1.0, lead=0, limit=10.0)
        rms = self._repeat(plant, ilc, ref, 5)
        self.assertAlmostEqual(ilc.convergence_ratio(), rms[-1] / rms[0], places=12)


class TestJointCompliance(unittest.TestCase):
    """关节柔度模型与前馈补偿。"""

    def test_deflection_is_torque_over_stiffness(self):
        """闭式判据：Δq = τ/k。"""
        m = JointComplianceModel(stiffness=2.0e4, n=7)
        tau = np.array([50.0, -30.0, 0.0, 20.0, 0.0, 5.0, -5.0])
        self.assertLess(np.abs(m.deflection(tau) - tau / 2.0e4).max(), 1e-15)

    def test_stiffer_joint_bends_less(self):
        """刚度翻倍，弯曲量减半。"""
        tau = np.full(7, 40.0)
        soft = JointComplianceModel(1.0e4).deflection(tau)
        hard = JointComplianceModel(2.0e4).deflection(tau)
        self.assertLess(np.abs(hard - soft / 2.0).max(), 1e-15)

    def test_compensation_reduces_error_and_wrong_sign_doubles_it(self):
        """⭐ 只看外部可观测事实：补偿后误差应当变小。

        再给一个**符号写反**的对照——它会让误差**翻倍**。
        这一条锁住的正是「往哪个方向补」这个最容易写反的地方。
        """
        m = JointComplianceModel(1.5e4)
        q_cmd = np.linspace(-0.5, 0.5, 7)
        tau = np.array([60.0, -40.0, 10.0, 35.0, -5.0, 8.0, -3.0])
        deflect = tau / 1.5e4

        # 被控对象：实际到达的角度 = 指令 − 弯曲量
        def actual(cmd):
            return cmd - deflect

        err_none = np.abs(actual(q_cmd) - q_cmd).max()
        err_comp = np.abs(actual(m.compensate(q_cmd, tau)) - q_cmd).max()
        err_wrong = np.abs(actual(q_cmd - deflect) - q_cmd).max()

        self.assertLess(err_comp, 1e-15, "正确补偿后应当基本无误差")
        self.assertGreater(err_none, 1e-6)
        self.assertAlmostEqual(err_wrong / err_none, 2.0, delta=1e-9,
                               msg="符号写反时误差应当正好翻倍")

    def test_rejects_non_positive_stiffness(self):
        with self.assertRaises(ValueError):
            JointComplianceModel(stiffness=0.0)


class TestMovingObstacle(unittest.TestCase):
    """移动障碍物的运动学。"""

    def test_stays_within_span(self):
        """往复运动不得超出给定行程。"""
        o = MovingSphere(np.zeros(3), 0.1, np.array([0.0, 1.0, 0.0]), span=0.6)
        offs = [float(o.position_at(t)[1]) for t in np.linspace(0, 10, 2000)]
        self.assertLessEqual(max(offs), 0.3 + 1e-9)
        self.assertGreaterEqual(min(offs), -0.3 - 1e-9)

    def test_zero_velocity_stays_put(self):
        o = MovingSphere(np.array([1.0, 2.0, 3.0]), 0.1, np.zeros(3))
        for t in (0.0, 1.7, 5.5):
            self.assertLess(np.abs(o.position_at(t)
                                   - np.array([1.0, 2.0, 3.0])).max(), 1e-15)

    def test_traverses_full_span_over_a_period(self):
        """一个周期内应当走完整个行程（峰峰值 = span）。"""
        o = MovingSphere(np.zeros(3), 0.1, np.array([0.0, 0.5, 0.0]), span=0.8)
        offs = [float(o.position_at(t)[1]) for t in np.linspace(0, 6.4, 4000)]
        self.assertAlmostEqual(max(offs) - min(offs), 0.8, delta=0.01)


class TestReplanSupervisor(unittest.TestCase):
    """触发逻辑：迟滞、冷却、预算，以及前瞻检测的正确性。"""

    def test_lookahead_uses_obstacle_future_position(self):
        """⭐ 障碍物必须按**各自时刻**取位置，不能拿当前位置比未来轨迹。

        构造：末端不动；障碍物此刻很远，但正朝末端飞来。
        如果实现拿的是「当前障碍物位置」，就会漏检。
        """
        sup = ReplanSupervisor(horizon=2.0, samples=80, clearance=0.05)
        obs = MovingSphere(np.zeros(3), 0.10,
                           np.array([1.0, 0.0, 0.0]), span=1.6)

        def ee(_t):
            return np.array([0.75, 0.0, 0.0])

        gap_now = float(np.linalg.norm(ee(0) - obs.position_at(0))) - obs.radius
        gap_min, _ = sup.min_clearance(ee, obs, 0.0)
        self.assertGreater(gap_now, 0.4, "构造有误：此刻本就该很远")
        self.assertLess(gap_min, 0.05,
                        "前瞻没有用障碍物的未来位置，快速移动的障碍会被漏检")

    def test_min_clearance_matches_independent_scan(self):
        """用一个独立的密采样扫描复算最小间隙。"""
        sup = ReplanSupervisor(horizon=1.5, samples=64, clearance=0.05)
        obs = MovingSphere(np.array([0.5, 0.0, 0.3]), 0.12,
                           np.array([0.0, 1.0, 0.0]), span=0.5)

        def ee(t):
            return np.array([0.5, 0.25 * np.sin(2.0 * t), 0.3])

        got, _ = sup.min_clearance(ee, obs, 0.3)
        ref = min(float(np.linalg.norm(ee(t) - obs.position_at(t))) - obs.radius
                  for t in np.linspace(0.3, 0.3 + 1.5, 4000))
        self.assertLess(got - ref, 0.02, "前瞻采样太稀，漏掉了更近的时刻")
        self.assertGreaterEqual(got, ref - 1e-9)

    def test_hysteresis_requires_consecutive_detections(self):
        """⭐ 单次检测不触发——单次受采样相位影响很大，会误报。"""
        sup = ReplanSupervisor(clearance=0.05, trigger_count=3, cooldown=0.0)
        sup.reset()
        for i in range(2):
            fire, why = sup.should_replan(0.01, t_now=float(i))
            self.assertFalse(fire, f"第 {i+1} 次就触发了：{why}")
        fire, why = sup.should_replan(0.01, t_now=2.0)
        self.assertTrue(fire, why)

    def test_streak_resets_when_safe(self):
        """中间只要安全一次，迟滞计数必须清零。"""
        sup = ReplanSupervisor(clearance=0.05, trigger_count=3, cooldown=0.0)
        sup.reset()
        sup.should_replan(0.01, 0.0)
        sup.should_replan(0.01, 0.1)
        sup.should_replan(0.50, 0.2)                 # 安全
        fire, _ = sup.should_replan(0.01, 0.3)
        self.assertFalse(fire, "安全一次之后计数没有清零")

    def test_cooldown_blocks_immediate_retrigger(self):
        sup = ReplanSupervisor(clearance=0.05, trigger_count=1, cooldown=1.0)
        sup.reset()
        self.assertTrue(sup.should_replan(0.01, 0.0)[0])
        sup.commit(0.0, "test", True)
        fire, why = sup.should_replan(0.01, 0.5)
        self.assertFalse(fire, why)
        self.assertIn("冷却", why)
        self.assertTrue(sup.should_replan(0.01, 1.6)[0])

    def test_budget_aborts_after_max_replans(self):
        """⭐ 必须有兜底：超预算就放弃，不能陷入死循环。"""
        sup = ReplanSupervisor(clearance=0.05, trigger_count=1, cooldown=0.0,
                               max_replans=3)
        sup.reset()
        for i in range(3):
            self.assertTrue(sup.should_replan(0.01, float(i))[0])
            sup.commit(float(i), "test", success=False)
        self.assertTrue(sup.aborted)
        fire, why = sup.should_replan(0.01, 10.0)
        self.assertFalse(fire)
        self.assertIn("放弃", why)
        self.assertEqual(sup.n_failed, 3)

    def test_replan_start_is_ahead_by_planning_time(self):
        """⭐ 规划起点必须是「算完时机器人会在的地方」，不是当前位置。"""
        sup = ReplanSupervisor(plan_time=0.15)
        self.assertAlmostEqual(sup.replan_start_time(2.0), 2.15, places=12)
        self.assertGreater(sup.replan_start_time(0.0), 0.0,
                           "规划起点没有前移，切换瞬间会出现状态跳变")


# ======================================================== P0-06 场景级全链路
#
# 上面那些测试只覆盖 ReplanSupervisor 这个**纯逻辑**类，
# 而第四轮审核发现的两个缺陷都在**场景层**（指令是怎么被执行的）：
#
#   问题 A：``sup.aborted`` 只是不再重规划，指令位置照样沿名义轨迹跑。
#           实测放弃后末端又走了 638 mm、关节速度均值 1.48 rad/s，
#           文档写的「机器人停下来而不是乱撞」并不成立。
#   问题 B：新绕行偏置是 ``self._detour = 新值`` **一步到位**，指令位置直接跳。
#           实测硬切换 ‖Δp_des‖ 最大 220 mm。而当时叫 ``jump`` 的读数
#           量的其实是「切换后残余间隙」，跟跳变毫无关系——自证循环。
#
# 下面这些测试的判据全部是**外部可观测量**：指令位置序列、关节速度、
# 末端位移。不读任何被测实现的中间变量作为"正确答案"。


def _run_replan_scene(seconds, **params):
    """跑一段 ReplanScene，返回场景对象和逐拍记录。

    只用外部可观测量：每一拍的指令位置 p_des、真实末端位置 p、关节速度。
    """
    import mujoco

    from armctrl.tuner.scenes import ReplanScene

    sc = ReplanScene()
    model = sc.build()
    for k, v in params.items():
        sc.set(k, v)
    sc.reset()
    n_sub = max(1, int(round(sc.dt / model.opt.timestep)))
    rec = {"t": [], "p_des": [], "p": [], "qd": [], "aborted": []}
    for _ in range(int(seconds / sc.dt)):
        t = sc.data.time
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
        for _ in range(n_sub):
            mujoco.mj_step(model, sc.data)
        rec["t"].append(t)
        rec["p_des"].append(sc._p_des.copy())
        rec["p"].append(sc.robot.fk(sc.data.qpos[sc.robot.qpos_idx])[0].copy())
        rec["qd"].append(float(np.linalg.norm(sc.data.qvel[sc.robot.qvel_idx])))
        rec["aborted"].append(bool(sc.sup.aborted))
    for k in ("t", "p_des", "p", "qd"):
        rec[k] = np.asarray(rec[k])
    rec["aborted"] = np.asarray(rec["aborted"])
    return sc, rec


#: 一定会超预算的组合：障碍物又大又快，预算只有 3 次。
ABORT_CASE = dict(max_replans=3, obs_radius=0.16, obs_speed=1.0,
                  det_noise=0.0, cooldown=0.3)
#: 会重规划好几次但不超预算的组合，用来看切换是否连续。
SPLICE_CASE = dict(max_replans=25, obs_radius=0.14, obs_speed=0.9,
                   det_noise=0.0)


class TestQuinticSplice(unittest.TestCase):
    """五次多项式过渡本身的数学性质（不需要仿真器）。"""

    @staticmethod
    def _s(tau):
        return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)

    def test_endpoints_and_zero_end_velocity(self):
        """⭐ s(0)=0, s(1)=1, 且两端斜率为 0 —— 位置和速度都接得上。"""
        self.assertAlmostEqual(self._s(0.0), 0.0, places=14)
        self.assertAlmostEqual(self._s(1.0), 1.0, places=14)
        h = 1e-6
        self.assertAlmostEqual((self._s(h) - self._s(0.0)) / h, 0.0, places=8)
        self.assertAlmostEqual((self._s(1.0) - self._s(1.0 - h)) / h, 0.0,
                               places=8)

    def test_is_monotone_so_the_offset_never_overshoots(self):
        """过渡过程单调，绕行量不会先冲过头再退回来。"""
        vals = self._s(np.linspace(0.0, 1.0, 401))
        self.assertTrue(np.all(np.diff(vals) >= -1e-15))
        self.assertLessEqual(float(vals.max()), 1.0 + 1e-12)

    def test_a_linear_ramp_would_fail_the_velocity_test(self):
        """对照组：线性过渡位置连续但**速度不连续**，两端斜率是 1 不是 0。"""
        h = 1e-6
        lin = lambda tau: tau
        self.assertAlmostEqual((lin(h) - lin(0.0)) / h, 1.0, places=8)


class TestReplanSceneSafeStop(unittest.TestCase):
    """P0-06 问题 A：超预算放弃之后，机器人必须真的停下来。"""

    def test_abort_actually_stops_the_robot(self):
        """⭐ 放弃后关节速度必须收敛到 0，位移必须有界。"""
        sc, rec = _run_replan_scene(9.0, stop_s=0.4, **ABORT_CASE)
        self.assertTrue(rec["aborted"].any(), "这组参数本该超预算，测试前提失效")
        self.assertAlmostEqual(sc._clock_rate, 0.0, places=12,
                               msg="轨迹时钟没有降到 0，说明限减速停车没生效")
        self.assertLess(rec["qd"][-1], 1e-3,
                        "放弃后机器人还在动——安全兜底形同虚设")
        # 指令位置在最后 1 秒内必须完全不动
        tail = rec["p_des"][-int(1.0 / sc.dt):]
        self.assertLess(float(np.abs(tail - tail[0]).max()), 1e-9,
                        "指令位置还在变，机器人没有真正停住")

    def test_stop_distance_is_bounded_and_stops_growing(self):
        """⭐ 停车位移必须收敛：后半段不能再增长。"""
        sc, rec = _run_replan_scene(9.0, stop_s=0.4, **ABORT_CASE)
        i0 = int(np.argmax(rec["aborted"]))
        p0 = rec["p"][i0]
        travel = np.linalg.norm(rec["p"] - p0, axis=1)[i0:]
        half = len(travel) // 2
        self.assertLess(float(travel.max()), 0.30,
                        "放弃后位移超过 300 mm，停车距离失控")
        self.assertLess(float(travel[half:].max() - travel[half:].min()), 5e-3,
                        "位移还在变化，说明根本没停（旧实现在 250~590 mm 间振荡）")

    def test_stop_travel_readout_matches_an_independent_recount(self):
        """⭐ 「放弃后位移」这一栏必须是真数，不能写死。

        判据是**外部独立复算**：拿逐拍记录下来的末端位置自己算一遍
        ``‖p_now − p_abort‖``，再和读数对照。读数如果被改成常数，
        或者量错了对象，这一条立刻失败。
        """
        sc, rec = _run_replan_scene(9.0, stop_s=0.4, **ABORT_CASE)
        i0 = int(np.argmax(rec["aborted"]))
        expect = float(np.linalg.norm(rec["p"][-1] - rec["p"][i0])) * 1e3
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        got = sc.telemetry(sc.data.time, q, v)["stop_travel"]
        self.assertGreater(got, 1.0, "放弃后位移读数恒为 0，这一栏是死数")
        self.assertLess(abs(got - expect), 5.0,
                        f"读数 {got:.1f} mm 与独立复算 {expect:.1f} mm 对不上")

    def test_longer_stop_time_means_longer_stop_distance(self):
        """⭐ 单调性对照：停车时间调大，停车距离必须变长（物理必然）。"""
        results = []
        for stop_s in (0.4, 1.5):
            sc, rec = _run_replan_scene(9.0, stop_s=stop_s, **ABORT_CASE)
            i0 = int(np.argmax(rec["aborted"]))
            # ⚠️ 必须从放弃那一刻开始切片。对整段取 max 会把放弃**之前**的
            # 正常运动也算进去，两组结果会一模一样，判据就失去区分能力了。
            travel = np.linalg.norm(rec["p"][i0:] - rec["p"][i0], axis=1)
            results.append(float(travel.max()))
        self.assertGreater(results[1], results[0] * 1.2,
                           f"停车时间翻了近 4 倍，停车距离却没明显变长：{results}")


class TestReplanSceneSplice(unittest.TestCase):
    """P0-06 问题 B：切换到新路径必须位置、速度都连续。"""

    def test_splice_removes_the_commanded_position_jump(self):
        """⭐ 开了过渡时间，指令位置的单拍跳变必须很小。"""
        sc, rec = _run_replan_scene(6.0, splice_s=0.18, **SPLICE_CASE)
        self.assertGreater(sc.sup.n_replans, 0, "没触发重规划，测试前提失效")
        step = np.linalg.norm(np.diff(rec["p_des"], axis=0), axis=1)
        self.assertLess(float(step.max()), 0.02,
                        "指令位置单拍跳了 20 mm 以上，splice 没起作用")

    def test_hard_switch_control_group_really_does_jump(self):
        """⭐ 对照组（过渡时间调 0 = 旧的硬切换）必须被判据抓到。

        这一条是**变异证明**：如果它不失败，说明上面那条测试根本没有区分能力。
        """
        sc, rec = _run_replan_scene(6.0, splice_s=0.0, **SPLICE_CASE)
        self.assertGreater(sc.sup.n_replans, 0, "没触发重规划，测试前提失效")
        step = np.linalg.norm(np.diff(rec["p_des"], axis=0), axis=1)
        self.assertGreater(float(step.max()), 0.05,
                           "硬切换居然没跳变？说明判据量错了对象")

    def test_telemetry_separates_clearance_from_jump(self):
        """⭐ 读数必须名副其实：jump 是「切换后残余间隙」，跳变另有其人。

        旧版本把 ``jump`` 映射到 ``_worst_post``（间隙），标签却写「位置跳变」，
        于是不管切换多硬，这个数都不会变大 —— 典型的自证循环。

        判据分两层，缺一不可：
        1. ``splice_dp`` 必须对「过渡时间」**敏感**（硬切换 → 开过渡至少降 5 倍）；
        2. ``splice_dp`` 不能只是 ``jump`` 的复制品（两者量的是不同的东西）。
        只查第一层会漏掉"复制品"变异，只查第二层会漏掉"量错对象"变异。
        """
        tels = {}
        for ss in (0.0, 0.18):
            sc, _ = _run_replan_scene(6.0, splice_s=ss, **SPLICE_CASE)
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            tels[ss] = (sc.telemetry(sc.data.time, q, v), sc)
        for key in ("splice_dp", "splice_dv", "stop_travel", "clock_rate"):
            self.assertIn(key, tels[0.0][0], f"读数 {key} 缺失")
        hard, soft = tels[0.0][0], tels[0.18][0]
        self.assertAlmostEqual(
            hard["jump"],
            tels[0.0][1]._worst_post * 1e3
            if np.isfinite(tels[0.0][1]._worst_post) else 0.0,
            places=9, msg="jump 不再是「切换后残余间隙」，标签又对不上了")
        self.assertGreater(hard["splice_dp"], 50.0,
                           "硬切换下位置跳变读数没反应，读数量错了对象")
        self.assertLess(soft["splice_dp"] * 5.0, hard["splice_dp"],
                        "开了过渡时间跳变没明显变小 —— 这个读数对 splice_s "
                        f"不敏感，很可能量错了对象：{hard['splice_dp']:.2f} → "
                        f"{soft['splice_dp']:.2f} mm")
        self.assertNotAlmostEqual(
            soft["splice_dp"], soft["jump"], places=6,
            msg="splice_dp 和 jump 数值完全相同 —— 跳变读数只是间隙的复制品")
        # 速度跳变同样要真的会动：硬切换是「一拍走完整段偏置」，
        # 等效速度 = 跳变 / dt，量级在万 mm/s；开了过渡后掉到百 mm/s。
        self.assertGreater(hard["splice_dv"], 1e4,
                           "硬切换下速度跳变读数没反应，这一栏是死数")
        self.assertLess(soft["splice_dv"] * 10.0, hard["splice_dv"],
                        f"速度跳变对 splice_s 不敏感："
                        f"{hard['splice_dv']:.0f} → {soft['splice_dv']:.0f} mm/s")

    # ------------------------------------------------ 批次 H：跳变读数量对了吗

    def test_a_run_without_any_switch_reports_exactly_zero_jump(self):
        """⭐⭐ 语义判据：一次重规划都没发生时，「切换跳变」必须**恰好**是 0。

        为什么这条最关键
        ----------------
        原实现差分的是**指令位置 p_des**：

            dp = p_des[k] - p_des[k-1]        # 一拍走了多远
            splice_dp = max_k ‖dp‖            # 全程最大值

        但 ``p_des = 名义轨迹 + 绕行偏置``，名义轨迹**本来就在走**。
        于是这个读数有一个**永远消不掉的地板**：实测障碍物速度调 0
        （全程零重规划）时，它照样报 1.466 mm / 4.6 mm/s。
        标签写着「切换跳变」，量的却是「每拍最大位移」。

        ⭐ 这条判据不看任何实现细节，纯粹是语义：**没有切换 ⇒ 跳变为 0。**
        它也解释了为什么原来那批测试全绿却没发现问题——
        它们只验「硬切换 vs 开过渡」的**比值**，而地板在两边都在，
        比值照样漂亮。
        """
        sc, rec = _run_replan_scene(6.0, obs_speed=0.0)
        self.assertEqual(sc.sup.n_replans, 0,
                         "障碍物没动却触发了重规划，这条测试的前提失效")
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        tel = sc.telemetry(sc.data.time, q, v)
        # 前提要有区分力：名义轨迹本身确实在动，地板不是 0
        step = np.linalg.norm(np.diff(rec["p_des"], axis=0), axis=1)
        self.assertGreater(
            float(step.max()) * 1e3, 0.5,
            "名义轨迹几乎没动，这条测试区分不了「量对」和「量错」")
        self.assertEqual(
            tel["splice_dp"], 0.0,
            f"零切换却报出 {tel['splice_dp']:.3f} mm 的「切换位置跳变」；"
            f"同段名义运动每拍最大 {float(step.max())*1e3:.3f} mm "
            f"—— 读数把名义运动算进去了")
        self.assertEqual(
            tel["splice_dv"], 0.0,
            f"零切换却报出 {tel['splice_dv']:.1f} mm/s 的「切换速度跳变」")

    def test_a_longer_transition_keeps_reducing_the_velocity_jump(self):
        """⭐ 过渡时间越长，速度跳变必须**持续**变小，不能卡在某个地板上。

        修之前实测：0.18 s → 99.9、0.40 s → 100.0、0.60 s → 99.9 mm/s，
        **完全不动**。原因不是 splice 不好，而是紧接着的「偏置慢慢回落」
        在过渡结束那一拍**满速接管**，速度从 0 一步跳到
        ``0.001·‖detour‖/dt ≈ 99.9 mm/s``。
        也就是说「位置、速度、jerk 全连续」这句话当时并不成立。

        ⚠️ 判据用**三档单调**而不是「小于某个阈值」：
        阈值是可以靠调参蒙混的，单调性不行——它要求每加长一次过渡都真的更好。
        """
        got = {}
        for ss in (0.18, 0.40, 0.60):
            sc, _ = _run_replan_scene(6.0, splice_s=ss, **SPLICE_CASE)
            self.assertGreater(sc.sup.n_replans, 0,
                               f"splice_s={ss} 没触发重规划，前提失效")
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            got[ss] = float(sc.telemetry(sc.data.time, q, v)["splice_dv"])
        self.assertLess(got[0.40], got[0.18] * 0.6,
                        f"过渡从 0.18 拉到 0.40 s，速度跳变几乎没降：{got}")
        self.assertLess(got[0.60], got[0.40] * 0.75,
                        f"过渡从 0.40 拉到 0.60 s，速度跳变几乎没降：{got}")

    def test_the_offset_velocity_has_no_step_when_the_decay_takes_over(self):
        """⭐ 偏置回落**接管的那一刻**不许有速度台阶（独立于遥测读数）。

        oracle 独立性：**完全不看** ``splice_dv``，而是自己把每一拍的
        绕行偏置记下来、做两次差分，直接检查偏置速度的逐拍变化量。
        遥测读数万一又量错了对象，这条测试仍然有效。
        """
        import mujoco

        from armctrl.tuner.scenes import ReplanScene

        sc = ReplanScene()
        model = sc.build()
        for k, v in dict(splice_s=0.6, **SPLICE_CASE).items():
            sc.set(k, v)
        sc.reset()
        d, r = sc.data, sc.robot
        sub = max(1, int(round(sc.dt / model.opt.timestep)))
        offs = []
        for k in range(int(6.0 / sc.dt)):
            t = k * sc.dt
            d.ctrl[r.arm_actuator_ids] = sc.control(
                t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
            for _ in range(sub):
                mujoco.mj_step(model, d)
            offs.append(np.asarray(sc._detour_at(t), float).copy())
        self.assertGreater(sc.sup.n_replans, 0, "没触发重规划，前提失效")
        offs = np.asarray(offs)
        vel = np.diff(offs, axis=0) / sc.dt
        dv = np.linalg.norm(np.diff(vel, axis=0), axis=1) * 1e3   # mm/s 每拍
        # 前提：偏置真的动过，否则全 0 也能"通过"
        self.assertGreater(np.linalg.norm(offs, axis=1).max() * 1e3, 50.0,
                           "偏置几乎没动过，这条测试没有区分力")
        self.assertLess(
            float(dv.max()), 20.0,
            f"偏置速度出现了 {dv.max():.1f} mm/s 的单拍台阶——"
            f"「位置、速度全连续」这句话不成立")

    def test_normal_running_matches_the_old_trajectory_exactly(self):
        """⭐ 回归保护：不放弃、不切换时，新旧公式必须逐点相同。

        轨迹时钟倍率恒为 1 ⇒ ``_planned(t) == _nominal(t) + _detour``。
        """
        import mujoco  # noqa: F401  （确保仿真依赖可用）

        from armctrl.tuner.scenes import ReplanScene
        sc = ReplanScene()
        sc.build()
        sc.set("obs_speed", 0.0)      # 障碍物不动 ⇒ 永不挡路 ⇒ 不会重规划
        sc.reset()
        for t in (0.0, 0.37, 1.5, 4.2):
            np.testing.assert_allclose(sc._planned(t),
                                       sc._nominal(t) + sc._detour,
                                       rtol=0, atol=1e-15)


# ======================================================================
# P1-09：场景与控制器的柔度符号约定必须是同一套
#
# ``JointComplianceModel`` 的约定是「实际 = 指令 − 形变」（上面
# TestJointCompliance 已经锁死）。但 IlcScene 里原来写的是
# ``q_eff = q + deflect``，符号正好反过来。
# 后果：打开柔度补偿**反而让误差变大**（实测 2.551 → 2.875 mrad）。
#
# 下面这些 oracle 全部只看外部可观测量（场景自己报的 RMS 读数），
# 不读补偿项、不读 deflect 的中间值。
# ======================================================================


def _run_ilc_trial(stiffness, comp, freq=0.4):
    """跑 IlcScene 一整遍，返回 (第一遍 RMS, 形变读数)。

    关掉噪声、几何误差和学习增益，让**柔度成为唯一的可变误差源**——
    否则补偿的效果会被别的东西盖住，看不出符号对不对。
    """
    import mujoco
    from armctrl.tuner.scenes import IlcScene

    sc = IlcScene()
    sc.set("stiffness", stiffness)          # restart=True，必须在 build 前设
    model = sc.build()
    sc.set("comp", comp)
    sc.set("noise", 0.0)
    sc.set("geom_err", 0.0)
    sc.set("gain", 0.0)
    sc.set("freq", freq)
    sc.reset()

    n_sub = max(1, int(round(sc.dt / model.opt.timestep)))
    t = 0.0
    for _ in range(sc.n_steps + 2):
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
        for _ in range(n_sub):
            mujoco.mj_step(model, sc.data)
        t += sc.dt
    q = sc.data.qpos[sc.robot.qpos_idx].copy()
    v = sc.data.qvel[sc.robot.qvel_idx].copy()
    tel = sc.telemetry(t, q, v)
    return tel["rms"], tel["deflect"]


class TestIlcSceneComplianceSign(unittest.TestCase):
    """场景级：补偿必须**减小**误差，而不是放大。"""

    SOFT = 3.0e3        # 形变 ≈ 10.2 mrad，远大于跟踪误差地板
    STIFF = 1.0e5       # 形变 ≈ 0.31 mrad，远小于地板

    def test_turning_compensation_on_must_reduce_the_error_not_raise_it(self):
        """⭐ 这是能杀死符号错误的核心判据。

        符号写反时补偿会把形变**叠加**而不是抵消，比值 > 1。
        """
        off, deflect = _run_ilc_trial(self.SOFT, "off")
        on, _ = _run_ilc_trial(self.SOFT, "on")

        self.assertGreater(deflect, 5.0,
                           f"软关节形变应有 ~10 mrad，实测 {deflect:.3f}——"
                           f"场景参数变了，本用例的前提不成立")
        self.assertLess(on, 0.6 * off,
                        f"打开柔度补偿后 RMS = {on:.4f} mrad，关闭时 "
                        f"{off:.4f} mrad。补偿没起到作用（预期 ≈ 1.45 vs 4.18，"
                        f"即降到 35% 左右）。符号写反时这里会 > 1。")

    def test_compensation_cannot_beat_the_tracking_error_floor(self):
        """诚实边界：补偿只能消掉形变那一部分，消不掉跟踪误差本身。"""
        stiff_off, _ = _run_ilc_trial(self.STIFF, "off")
        soft_on, _ = _run_ilc_trial(self.SOFT, "on")
        self.assertGreater(soft_on, 0.5 * stiff_off,
                           f"补偿后 RMS = {soft_on:.4f} mrad 竟低于刚性地板 "
                           f"{stiff_off:.4f} mrad 的一半，说明有别的东西在"
                           f"抵消误差，不是柔度补偿在起作用")

    def test_the_benefit_vanishes_when_the_joint_is_already_stiff(self):
        """对照组：关节很硬时形变本来就可忽略，补不补几乎一样。

        这一条排除了「补偿其实是靠改变增益把误差压下去」的可能——
        真那样的话硬关节上也会有明显收益。
        """
        off, deflect = _run_ilc_trial(self.STIFF, "off")
        on, _ = _run_ilc_trial(self.STIFF, "on")
        self.assertLess(deflect, 1.0,
                        f"硬关节形变应 < 1 mrad，实测 {deflect:.3f}")
        self.assertLess(abs(on - off) / off, 0.10,
                        f"硬关节上补偿带来 {abs(on - off) / off * 100:.1f}% 的"
                        f"变化（预期 < 1%）——那它压的就不是柔度")

    def test_scene_matches_the_unit_model_sign_convention(self):
        """把场景实测的「补偿收益」和柔度模型的闭式预测**定量**对上。

        ⚠️ 先统一量纲，这一步最容易出错：

        * ``deflect`` 读数 = ``‖Δq‖``，7 个关节的 **L2 范数**；
        * ``rms``     读数 = ``sqrt(mean(e²))``，跨时间和关节的**逐元素**均方根。

        所以同一个物理量在两个读数里差一个 ``sqrt(7)``：

            形变对 rms 的贡献 = ‖Δq‖ / √7

        闭式预测（完全独立于场景代码）：

            未补偿² ≈ (‖Δq‖/√7)² + 地板²
            已补偿² ≈ 地板²
            ⇒ √(未补偿² − 已补偿²) ≈ ‖Δq‖/√7
        """
        off, deflect = _run_ilc_trial(self.SOFT, "off")
        on, _ = _run_ilc_trial(self.SOFT, "on")
        removed = np.sqrt(max(off ** 2 - on ** 2, 0.0))
        predicted = deflect / np.sqrt(7.0)      # 7 = N_ARM

        ratio = removed / predicted
        # 实测 3.917 / 3.845 = 1.019。留 ±25% 余量给「整遍平均 vs 末时刻瞬时」
        self.assertGreater(ratio, 0.75,
                           f"补偿消掉 {removed:.3f} mrad，闭式预测 "
                           f"{predicted:.3f} mrad，比值 {ratio:.3f}（预期 ≈1.02）")
        self.assertLess(ratio, 1.25,
                        f"补偿消掉 {removed:.3f} mrad，闭式预测 "
                        f"{predicted:.3f} mrad，比值 {ratio:.3f}（预期 ≈1.02）")


# ======================================================================
# P1-09 第二刀：补偿器和被控对象**不能共用同一个形变量**
#
# 第五轮审核指出：旧代码里 ``deflect`` 算一次，两边都用——
#   补偿端  qd_cmd = qd_cmd + deflect
#   对象端  q_eff  = q      - deflect
# 于是**把两处符号一起翻掉，误差分毫不变**，8/8 测试照样全绿。
# 上面 TestIlcSceneComplianceSign 抓得住「一端错」，抓不住「两端同错」。
#
# 物理上这也是假的：真正压弯关节的是**实际传过去的力矩** τ
# （含惯性项 M(q)q̈、PD 修正、饱和裁剪），而补偿器在下指令之前
# 只拿得到重力+科氏项 bias(q,v)。二者之差 = M(q)q̈/k，建不了模。
#
# 修复后这两个量真的不一样了，于是出现一个**旧代码造不出来的signature**：
#   「可建模形变几乎不随频率变化，但补偿后的残差随频率暴涨。」
# 下面全部只看场景对外报的读数，不读 control() 里的中间变量。
# ======================================================================


def _run_ilc_sweep(stiffness, comp, freq):
    """跑一遍，返回 (末时刻 RMS, 整遍可建模形变 RMS, 整遍惯性形变 RMS)。"""
    import mujoco
    from armctrl.tuner.scenes import IlcScene

    sc = IlcScene()
    sc.set("stiffness", stiffness)
    model = sc.build()
    sc.set("comp", comp)
    sc.set("noise", 0.0)
    sc.set("geom_err", 0.0)
    sc.set("gain", 0.0)
    sc.set("freq", freq)
    sc.reset()

    n_sub = max(1, int(round(sc.dt / model.opt.timestep)))
    t = 0.0
    modelled, inertial, plant, sat = [], [], [], []
    for _ in range(sc.n_steps + 2):
        q = sc.data.qpos[sc.robot.qpos_idx].copy()
        v = sc.data.qvel[sc.robot.qvel_idx].copy()
        sc.data.ctrl[sc.robot.arm_actuator_ids] = sc.control(t, q, v)
        for _ in range(n_sub):
            mujoco.mj_step(model, sc.data)
        tel = sc.telemetry(t, q, v)
        modelled.append(tel["deflect"])
        inertial.append(tel["defl_extra"])
        plant.append(tel["defl_plant"])
        sat.append(tel["tau_max_pct"])
        t += sc.dt
    q = sc.data.qpos[sc.robot.qpos_idx].copy()
    v = sc.data.qvel[sc.robot.qvel_idx].copy()
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    lim = float(np.linalg.norm(sc.robot.tau_limit)) / stiffness * 1e3
    return (sc.telemetry(t, q, v)["rms"], rms(modelled), rms(inertial),
            max(plant), max(sat), lim)


class TestCompensatorAndPlantAreNotTheSameQuantity(unittest.TestCase):
    """⭐⭐ P1-09：补偿端知道的 ≠ 被控对象真实发生的。"""

    SOFT = 3.0e3
    FREQS = (0.2, 0.4, 0.8, 1.6)

    @classmethod
    def setUpClass(cls):
        cls.data = {}
        cls.raw = {}
        for f in cls.FREQS:
            off = _run_ilc_sweep(cls.SOFT, "off", f)
            on = _run_ilc_sweep(cls.SOFT, "on", f)
            cls.data[f] = (off[0], on[0], on[1], on[2])
            cls.raw[f] = on

    def test_modelled_deflection_barely_moves_with_frequency(self):
        """地基事实：可建模形变 bias(q,v)/k 几乎不随频率变。

        它只跟姿态有关（重力项主导），轨迹快慢基本不影响。
        实测 10.445 → 10.683 mrad，涨幅 2.3%。
        这条是下一条测试的**对照组**：有了它，才能说
        「残差涨了 11 倍」不可能是可建模那部分造成的。
        """
        vals = [self.data[f][2] for f in self.FREQS]
        spread = (max(vals) - min(vals)) / min(vals)
        self.assertLess(spread, 0.15,
                        f"可建模形变随频率变化 {spread:.1%}，"
                        f"实测各频率 {[round(v, 3) for v in vals]}，预期 ≈2%")

    def test_the_unmodelled_inertial_part_explodes_with_frequency(self):
        """⭐ 惯性形变 M(q)q̈/k ∝ ω²，频率翻倍应该涨约 4 倍。

        实测 0.537 → 1.310 → 4.141 → 14.352 mrad
        （相邻比 2.44 / 3.16 / 3.47，越来越接近理论的 4）。
        低频不到 4 是因为 PD 修正项也混在 τ 里，它不随 ω² 走。
        """
        vals = [self.data[f][3] for f in self.FREQS]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(b / a, 2.0,
                               f"频率翻倍惯性形变只涨 {b / a:.2f} 倍，"
                               f"实测序列 {[round(v, 3) for v in vals]}")
        self.assertGreater(vals[-1] / vals[0], 20.0,
                           f"0.2→1.6 Hz 惯性形变只涨 {vals[-1] / vals[0]:.1f} 倍，"
                           f"预期 ≈26.7")

    def test_at_high_frequency_inertia_beats_what_the_model_knows(self):
        """⭐⭐ 1.6 Hz 时，建不了模的部分**超过**可建模的部分。

        这就是柔度前馈的天花板：再准的重力模型也补不掉它，
        剩下的只能交给 ILC 一遍遍学。
        实测 14.352 > 10.683 mrad（1.34 倍）。
        """
        _, _, modelled, inertial = self.data[1.6]
        self.assertGreater(inertial, modelled,
                           f"1.6 Hz 时惯性形变 {inertial:.3f} mrad 没有超过 "
                           f"可建模形变 {modelled:.3f} mrad")
        low = self.data[0.2]
        self.assertLess(low[3], 0.2 * low[2],
                        f"0.2 Hz 时惯性形变 {low[3]:.3f} mrad 相对 "
                        f"可建模 {low[2]:.3f} mrad 不够小，低频应当可忽略")

    def test_compensation_benefit_decays_as_the_trajectory_speeds_up(self):
        """⭐⭐⭐ 决定性判据：补偿收益必须**随频率单调衰减**。

        旧代码两端共用同一个 ``deflect``，补偿在数学上是精确抵消，
        收益跟频率基本无关。只有当两端真的是两个不同的量时，
        才会出现这条衰减曲线。

        实测改善率 77.9% → 62.5% → 31.5% → 4.9%。
        """
        gains = []
        for f in self.FREQS:
            off, on, _, _ = self.data[f]
            self.assertLess(on, off,
                            f"{f} Hz 打开补偿反而更差：{off:.4f} → {on:.4f}")
            gains.append(1.0 - on / off)
        for a, b in zip(gains, gains[1:]):
            self.assertLess(b, a,
                            f"补偿收益没有随频率下降："
                            f"{[f'{g:.1%}' for g in gains]}")
        self.assertGreater(gains[0], 0.6,
                           f"0.2 Hz 补偿收益只有 {gains[0]:.1%}，预期 ≈78%")
        self.assertLess(gains[-1], 0.20,
                        f"1.6 Hz 补偿收益还有 {gains[-1]:.1%}，预期 ≈5%")

    def test_the_residual_grows_even_though_the_modelled_part_does_not(self):
        """把上面两条钉在一起：残差涨 11 倍，可建模那部分只涨 2%。

        所以残差的增长**不可能**来自可建模形变——这是一条
        纯外部可观测的反证，不需要看代码一眼。
        """
        on_lo = self.data[0.2][1]
        on_hi = self.data[1.6][1]
        mod_lo = self.data[0.2][2]
        mod_hi = self.data[1.6][2]
        self.assertGreater(on_hi / on_lo, 5.0,
                           f"补偿后残差只涨 {on_hi / on_lo:.1f} 倍，预期 ≈11.5")
        self.assertLess(mod_hi / mod_lo, 1.2,
                        f"可建模形变涨了 {mod_hi / mod_lo:.2f} 倍，预期 ≈1.02")


class TestStiffnessSweepStillBehaves(unittest.TestCase):
    """刚度扫描：越硬 → 形变越小 → 补偿收益越小。"""

    def test_stiffer_joints_leave_less_for_compensation_to_fix(self):
        """实测改善率 62.5% / 41.7% / 13.8% / 1.6% / 0.4%。"""
        gains = []
        for k in (3.0e3, 1.5e4, 1.0e5):
            off = _run_ilc_sweep(k, "off", 0.4)[0]
            on = _run_ilc_sweep(k, "on", 0.4)[0]
            gains.append(1.0 - on / off)
        for a, b in zip(gains, gains[1:]):
            self.assertLess(b, a,
                            f"刚度变大补偿收益没有变小："
                            f"{[f'{g:.1%}' for g in gains]}")
        self.assertGreater(gains[0], 0.5)
        self.assertLess(gains[-1], 0.05)




class TestPlantDeflectionRespectsTheTorqueLimit(unittest.TestCase):
    """⭐ 硬物理不变量：弹簧传的力矩不可能超过电机能出的上限。

    关节柔度形变 Δq = τ/k，而 τ 是**裁剪后**的实际力矩。
    所以 ‖Δq‖ ≤ ‖τ_limit‖/k，这是一条与实现无关的天花板。

    这条测试是变异体 M12「被控对象用未裁剪的 τ」的专杀器：
    0.8 Hz 以上力矩已经打到 100% 饱和（实测峰值 tau_max_pct = 100.0），
    此时未裁剪的 τ 会算出一个**物理上不可能**的形变。
    """

    def test_deflection_never_exceeds_torque_limit_over_stiffness(self):
        for freq in (0.8, 1.6):
            _, _, _, plant, sat, lim = _run_ilc_sweep(3.0e3, "on", freq)
            self.assertGreater(sat, 99.0,
                               f"{freq} Hz 力矩峰值只有 {sat:.1f}%，"
                               f"没进入饱和，这条测试就失去意义了")
            self.assertLessEqual(plant, lim * 1.001,
                                 f"{freq} Hz 实际形变 {plant:.3f} mrad 超过了"
                                 f"物理上限 ‖τ_limit‖/k = {lim:.3f} mrad —— "
                                 f"说明被控对象用的不是裁剪后的力矩")

    def test_below_saturation_the_limit_is_not_even_close(self):
        """对照组：0.2 Hz 不饱和，形变离上限还很远——上一条不是恒真。"""
        _, _, _, plant, sat, lim = _run_ilc_sweep(3.0e3, "on", 0.2)
        self.assertLess(sat, 90.0,
                        f"0.2 Hz 就已经 {sat:.1f}% 饱和了，对照组不成立")
        self.assertLess(plant, 0.9 * lim,
                        f"0.2 Hz 形变 {plant:.3f} 已经贴到上限 {lim:.3f}")


# ================================================ 批次 H：§2.2 那张扫描表


class TestTheSpliceSweepTableIsReproducible(unittest.TestCase):
    """`12_动态避障.md` 里 `splice_s` 扫描表的每一格都必须能重跑出来。

    为什么需要这一组
    ----------------
    这张表的上一版（220.0 / 110 005 / 13.8 / 702 / 4.8 / 120 / 2.5 / 120）
    **没有任何测试守着**，是手工跑一次贴上去的；批次 H 复现时发现
    没有一个对得上。刚把新数字写进去，如果还是不守，过几轮又会烂掉
    （元教训 #58：「跑一次贴上来的数字，寿命只到下一次改代码为止」）。

    ⚠️ 期望值**从 Markdown 现场解析**，不写死在测试里——
    写死等于把文档漏在守护范围之外。
    """

    #: 文档那张表用的是场景**默认参数**，只改 splice_s
    ROWS = (0.0, 0.06, 0.18, 0.40, 0.60)

    @classmethod
    def setUpClass(cls):
        cls.doc = cls._parse()
        cls.got = {}
        for ss in cls.ROWS:
            sc, _ = _run_replan_scene(6.0, splice_s=ss)
            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            tel = sc.telemetry(sc.data.time, q, v)
            cls.got[ss] = (float(tel["splice_dp"]), float(tel["splice_dv"]))

    @staticmethod
    def _parse():
        """解析 「| splice_s | dp mm | dv mm/s |」三列。"""
        doc = (Path(__file__).resolve().parents[2]
               / "armctrl_讲解" / "12_动态避障.md").read_text(encoding="utf-8")
        body = doc.split("实测效果（`splice_s` 从 0 起逐步调大", 1)[1]
        body = body.split("\n\n", 3)[0] + body.split("\n\n", 3)[1]
        out = {}
        for line in body.splitlines():
            if not line.startswith("|") or line.count("|") < 4:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            nums = []
            for c in cells[:3]:
                # 去掉 **加粗**、千分位空格、以及「（默认）」这类中文注解
                t = c.replace("*", "").replace("\u00a0", " ")
                m = re.search(r"\d[\d ]*(?:\.\d+)?", t)
                nums.append(float(m.group(0).replace(" ", "")) if m else None)
            if nums[0] is not None and nums[1] is not None and nums[2] is not None:
                out[round(nums[0], 3)] = (nums[1], nums[2])
        assert out, "§2.2 的 splice 扫描表解析不出来——表格格式变了？"
        return out

    def test_the_table_still_has_every_row(self):
        for ss in self.ROWS:
            self.assertIn(round(ss, 3), self.doc,
                          f"文档表里缺 splice_s={ss} 这一行")

    def test_every_position_jump_matches(self):
        for ss in self.ROWS:
            want = self.doc[round(ss, 3)][0]
            got = self.got[ss][0]
            self.assertAlmostEqual(
                got, want, delta=max(0.02, want * 0.02),
                msg=f"splice_s={ss}: 文档写 {want} mm，实测 {got:.3f} mm")

    def test_every_velocity_jump_matches(self):
        for ss in self.ROWS:
            want = self.doc[round(ss, 3)][1]
            got = self.got[ss][1]
            self.assertAlmostEqual(
                got, want, delta=max(1.0, want * 0.05),
                msg=f"splice_s={ss}: 文档写 {want} mm/s，实测 {got:.1f} mm/s")

    def test_the_table_is_monotone_in_both_columns(self):
        """⭐ 表格自身的形状也要对：过渡越长，两列都必须越小。

        这条挡的是「某一格被改成一个碰巧也能通过公差的怪数」。
        """
        dp = [self.got[ss][0] for ss in self.ROWS]
        dv = [self.got[ss][1] for ss in self.ROWS]
        self.assertEqual(dp, sorted(dp, reverse=True),
                         f"位置跳变没有随过渡时间单调下降：{dp}")
        self.assertEqual(dv, sorted(dv, reverse=True),
                         f"速度跳变没有随过渡时间单调下降：{dv}")



if __name__ == "__main__":
    unittest.main()
