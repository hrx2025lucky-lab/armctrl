"""广义动量观测器的直接测试（第四轮审核 T-04 补的覆盖缺口）。

⚠️ 补这个文件之前，``armctrl/tests`` 里**没有任何一处**直接命中
``MomentumObserver``。它只被 ``ObserverScene`` 间接跑过，
而场景测试看的是"画面上有没有东西"，不是"残差对不对"。

oracle 独立性说明
----------------
被测对象的设计方程是

    ṙ = K_I (τ_ext − r)

这是一个**一阶线性系统**，阶跃响应有闭式解

    r(t) = τ_ext · (1 − e^{−K_I t})

⭐ 判据就是这条闭式解，**不读观测器的任何中间量**（不读 integral、不读 p0）。

为了让这条闭式解**严格**成立，这里不用 Panda，而是自己搭一个
**常惯量、无重力、无科氏力**的最简被控对象：

    M = 常数  ⇒  Ṁ = 0, C = 0, g = 0  ⇒  β = 0

于是观测器方程退化成纯粹的一阶低通，理论值可以手算到任意精度。
用真实机械臂反而会把"模型对不对"和"观测器对不对"两件事混在一起。

⚠️ 级别：**理论级**（判据是独立解析解），但被控对象是简化模型，
不是真机标定。
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.control.momentum_observer import MomentumObserver


class LinearPlant:
    """常惯量、无重力、无科氏力的最简"机械臂"。

    只实现 ``MomentumObserver`` 真正会调用的四个接口。
    故意不继承 ArmModel：**被测对象不该依赖真机模型才能验证**。
    """

    def __init__(self, mass_matrix: np.ndarray, jacobian: np.ndarray | None = None):
        self.M = np.asarray(mass_matrix, float)
        self.n = self.M.shape[0]
        self._J = (np.asarray(jacobian, float) if jacobian is not None
                   else np.eye(6, self.n))

    def mass_matrix(self, q):
        return self.M.copy()

    def gravity(self, q):
        return np.zeros(self.n)

    def coriolis_times_qd(self, q, qd):
        return np.zeros(self.n)

    def jacobian(self, q):
        return self._J.copy()


def simulate(plant, obs, tau_ext, seconds, tau_cmd=None, friction=None):
    """用**独立**的前向欧拉推进真实动力学，同时喂给观测器。

    真实动力学：``M q̈ = τ_cmd + τ_ext − τ_friction``
    观测器只知道 ``τ_cmd``，所以 τ_ext 和摩擦对它都是"外部的"。
    """
    n = plant.n
    dt = obs.dt
    q = np.zeros(n)
    qd = np.zeros(n)
    tau_cmd = np.zeros(n) if tau_cmd is None else np.asarray(tau_cmd, float)
    ts, rs = [], []
    Minv = np.linalg.inv(plant.M)
    for k in range(int(round(seconds / dt))):
        t = k * dt
        ext = tau_ext(t) if callable(tau_ext) else tau_ext
        fric = np.zeros(n) if friction is None else friction(qd)
        rs.append(obs.update(q, qd, tau_cmd).copy())
        ts.append(t)
        qdd = Minv @ (tau_cmd + ext - fric)
        qd = qd + qdd * dt
        q = q + qd * dt
    return np.asarray(ts), np.asarray(rs)


class TestMomentumObserverStepResponse(unittest.TestCase):
    """⭐ 核心：残差必须是外力矩的一阶低通。"""

    K_I = 25.0
    DT = 1e-4

    def _obs(self, plant, k_i=None):
        return MomentumObserver(plant, k_i=k_i or self.K_I, dt=self.DT)

    def test_zero_external_torque_leaves_only_a_quantifiable_euler_spike(self):
        """⭐⭐ 没有外力时残差**不是恒 0**，而是一个可精确预测的离散化尖峰。

        我本来想断言"残差恒为 0"，结果实测 1.75e−3。查下来不是 bug，
        是**前向欧拉的一步滞后**：观测器先把积分推进一步，再用**尚未更新**的
        动量去算残差，于是力矩阶跃会漏进来一拍。手推递推式：

            r₁ = −K_I·τ·Δt,   r_{k+1} = r_k (1 − K_I Δt)

        所以峰值 **正好等于 K_I·|τ|·Δt**，之后按 1/K_I 的时间常数衰减掉。

        实测 12 组参数（K_I ∈ {25,50} × Δt ∈ {2e−3,1e−3,1e−4} × τ ∈ {0.7,7}）
        的实测峰值 / 预测值比值**全部等于 1.000**。

        ⚠️ **真机含义**：用生产默认值 K_I=25、Δt=2 ms，
        一次 τ 的力矩阶跃会产生 **5% τ** 的假残差。
        Panda 的 87 N·m 上限对应 **4.35 N·m** 的虚假"外力"——
        碰撞阈值如果定得比这还低，**一加速就会误报碰撞**。
        减小 Δt 可线性压低它，但真正的对策是：阈值必须实测无负载本底来定。
        """
        plant = LinearPlant(np.diag([1.0]))
        for k_i in (25.0, 50.0):
            for dt in (2e-3, 1e-4):
                for tau in (0.7, 7.0):
                    obs = MomentumObserver(plant, k_i=k_i, dt=dt)
                    t, r = simulate(plant, obs, np.zeros(1), 8.0 / k_i,
                                    tau_cmd=np.array([tau]))
                    peak = float(np.abs(r).max())
                    self.assertAlmostEqual(
                        peak, k_i * tau * dt, delta=1e-9 * k_i * tau * dt,
                        msg=f"K_I={k_i} dt={dt} tau={tau}: 峰值 {peak:.4e} "
                            f"不等于 K_I·τ·Δt = {k_i * tau * dt:.4e}")
                    # 尖峰方向与驱动力矩**相反**（是滞后造成的，不是真外力）
                    self.assertLess(float(r[:, 0].min()), 0.0)
                    # 并且必须衰减掉：5 个时间常数后残留 < 峰值的 2%
                    self.assertLess(float(np.abs(r[-1, 0])), 0.02 * peak,
                                    "尖峰没有衰减，说明不是一步滞后而是真的漂了")

    def test_a_constant_velocity_glide_produces_no_residual_at_all(self):
        """对照组：匀速滑行（无力矩、无外力）时残差必须**真的**是 0。

        这一条排除"上面那个尖峰其实是观测器本身有偏"的可能——
        没有力矩阶跃，就没有尖峰。
        """
        plant = LinearPlant(np.diag([2.0, 1.0, 0.5]))
        obs = MomentumObserver(plant, k_i=self.K_I, dt=self.DT)
        qd0 = np.array([0.9, -0.4, 1.1])
        obs.reset(np.zeros(3), qd0)
        r_max = 0.0
        q = np.zeros(3)
        for _ in range(3000):                    # 恒速：τ_cmd=0, τ_ext=0
            r_max = max(r_max, float(np.abs(obs.update(q, qd0, np.zeros(3))).max()))
            q = q + qd0 * self.DT
        self.assertLess(r_max, 1e-9,
                        f"匀速滑行时残差 {r_max:.3e}，本该恒为 0")

    def test_step_response_matches_the_closed_form_low_pass(self):
        """⭐ r(t) = τ_ext (1 − e^{−K t})，逐点对照解析解。"""
        plant = LinearPlant(np.diag([2.0, 1.0, 0.5]))
        obs = self._obs(plant)
        tau_ext = np.array([5.0, -3.0, 1.5])
        t, r = simulate(plant, obs, tau_ext, 0.5)
        ref = tau_ext[None, :] * (1.0 - np.exp(-self.K_I * t))[:, None]
        err = float(np.abs(r - ref).max())
        self.assertLess(err, 5e-3 * float(np.abs(tau_ext).max()),
                        f"与一阶低通闭式解最大偏差 {err:.4e}")

    def test_a_wrong_gain_would_fail_the_same_oracle(self):
        """⭐ 变异对照：把闭式解里的增益改错，判据必须失败。

        不做这一步，上面那条测试可能只是"曲线大致像"而已。
        """
        plant = LinearPlant(np.diag([2.0, 1.0, 0.5]))
        obs = self._obs(plant)
        tau_ext = np.array([5.0, -3.0, 1.5])
        t, r = simulate(plant, obs, tau_ext, 0.5)
        wrong = tau_ext[None, :] * (1.0 - np.exp(-2.0 * self.K_I * t))[:, None]
        self.assertGreater(float(np.abs(r - wrong).max()),
                           5e-3 * float(np.abs(tau_ext).max()),
                           "增益差一倍居然也能通过——判据没有区分能力")

    def test_gain_sets_the_bandwidth_so_larger_k_settles_faster(self):
        """⭐ K_I 就是带宽：到达 63.2% 的时刻应当等于 1/K_I。"""
        plant = LinearPlant(np.diag([1.0]))
        tau_ext = np.array([4.0])
        for k_i in (10.0, 25.0, 60.0):
            obs = MomentumObserver(plant, k_i=k_i, dt=self.DT)
            t, r = simulate(plant, obs, tau_ext, 6.0 / k_i)
            idx = int(np.argmax(r[:, 0] >= 0.632 * tau_ext[0]))
            self.assertAlmostEqual(t[idx], 1.0 / k_i, delta=3.0 * self.DT,
                                   msg=f"K_I={k_i} 的时间常数不是 1/K_I")

    def test_residual_converges_to_the_true_external_torque(self):
        """稳态残差必须等于真实外力矩本身（直流增益为 1）。"""
        plant = LinearPlant(np.diag([2.0, 1.0, 0.5]))
        obs = self._obs(plant)
        tau_ext = np.array([5.0, -3.0, 1.5])
        _, r = simulate(plant, obs, tau_ext, 1.5)   # ≫ 5 个时间常数
        np.testing.assert_allclose(r[-1], tau_ext, rtol=2e-3, atol=1e-3)

    def test_a_non_diagonal_inertia_does_not_break_it(self):
        """惯量矩阵耦合（非对角）时结论不变——p = M q̇ 里 M 必须真的被用上。"""
        M = np.array([[2.0, 0.6, 0.1],
                      [0.6, 1.4, 0.3],
                      [0.1, 0.3, 0.9]])
        plant = LinearPlant(M)
        obs = self._obs(plant)
        tau_ext = np.array([2.0, 4.0, -1.0])
        t, r = simulate(plant, obs, tau_ext, 0.5)
        ref = tau_ext[None, :] * (1.0 - np.exp(-self.K_I * t))[:, None]
        self.assertLess(float(np.abs(r - ref).max()), 5e-3 * 4.0)


class TestMomentumObserverReset(unittest.TestCase):
    """p0 / reset 的行为（外部可观测，不读 integral）。"""

    def test_reset_with_state_uses_that_state_as_the_baseline(self):
        """⭐ 从**非零速度**启动时，残差仍必须从 0 出发。

        如果 p0 没有被正确锁存，第一拍就会凭空冒出 K_I·M·q̇ 的残差——
        真机上表现为"一上电就报碰撞"。
        """
        plant = LinearPlant(np.diag([2.0, 1.0]))
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-4)
        qd0 = np.array([1.3, -0.8])
        obs.reset(np.zeros(2), qd0)
        r0 = obs.update(np.zeros(2), qd0, np.zeros(2))
        self.assertLess(float(np.abs(r0).max()), 1e-9,
                        f"带初速度启动时第一拍残差 {r0}，本该是 0")

    def test_lazy_latch_when_reset_is_called_without_state(self):
        """不带参数 reset 时，第一次 update 必须自动把当前动量锁成基线。"""
        plant = LinearPlant(np.diag([2.0, 1.0]))
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-4)
        obs.reset()
        qd0 = np.array([1.3, -0.8])
        r0 = obs.update(np.zeros(2), qd0, np.zeros(2))
        self.assertLess(float(np.abs(r0).max()), 1e-9,
                        f"延迟锁存失效，第一拍残差 {r0}")

    def test_reset_clears_a_previously_accumulated_residual(self):
        """跑出残差后 reset，必须回到干净状态。"""
        plant = LinearPlant(np.diag([2.0, 1.0]))
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-4)
        simulate(plant, obs, np.array([5.0, 5.0]), 0.3)
        self.assertGreater(float(np.abs(obs.r).max()), 1.0)   # 前提：确实有残差
        obs.reset(np.zeros(2), np.zeros(2))
        self.assertLess(float(np.abs(obs.r).max()), 1e-15)
        r0 = obs.update(np.zeros(2), np.zeros(2), np.zeros(2))
        self.assertLess(float(np.abs(r0).max()), 1e-12)


class TestMomentumObserverDetection(unittest.TestCase):
    """阈值判定：越阈时刻必须与解析解一致。"""

    def test_first_crossing_time_matches_the_analytic_prediction(self):
        """⭐ 由 r(t)=τ(1−e^{−Kt}) 反解：t* = −ln(1 − thr/τ) / K。"""
        k_i, dt = 25.0, 1e-4
        plant = LinearPlant(np.diag([1.0]))
        tau_ext, thr = 8.0, 4.0
        obs = MomentumObserver(plant, k_i=k_i, dt=dt)
        t, r = simulate(plant, obs, np.array([tau_ext]), 0.4)
        fired = [tt for tt, rr in zip(t, r)
                 if abs(float(rr[0])) > thr]
        self.assertTrue(fired, "残差已超阈值却从未触发")
        expect = -np.log(1.0 - thr / tau_ext) / k_i
        self.assertAlmostEqual(fired[0], expect, delta=5.0 * dt,
                               msg=f"越阈时刻 {fired[0]:.5f} vs 解析 {expect:.5f}")

    def test_higher_threshold_fires_later(self):
        """单调性：阈值调高，触发必须变晚（不然阈值这个旋钮是假的）。"""
        k_i, dt = 25.0, 1e-4
        plant = LinearPlant(np.diag([1.0]))
        obs = MomentumObserver(plant, k_i=k_i, dt=dt)
        t, r = simulate(plant, obs, np.array([8.0]), 0.4)
        times = []
        for thr in (2.0, 4.0, 6.0):
            hit = [tt for tt, rr in zip(t, r) if abs(float(rr[0])) > thr]
            times.append(hit[0])
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])

    def test_per_joint_threshold_vector_is_honoured(self):
        """⭐ 逐关节阈值：只有**越阈的那个关节**能让 detect 变真。"""
        plant = LinearPlant(np.diag([1.0, 1.0, 1.0]))
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-4)
        simulate(plant, obs, np.array([0.0, 6.0, 0.0]), 1.0)
        self.assertGreater(abs(float(obs.r[1])), 5.0)
        self.assertLess(abs(float(obs.r[0])), 1e-6)
        # 关节 2 的阈值放得很高 ⇒ 不该报警
        self.assertFalse(obs.detect(np.array([1.0, 100.0, 1.0])))
        # 关节 2 的阈值压低 ⇒ 必须报警
        self.assertTrue(obs.detect(np.array([100.0, 5.0, 100.0])))
        # 标量阈值等价于全关节同阈值
        self.assertTrue(obs.detect(5.0))
        self.assertFalse(obs.detect(100.0))


class TestMomentumObserverCartesianMapping(unittest.TestCase):
    """r → 笛卡尔外力的映射 F ≈ (Jᵀ)⁺ r。"""

    def test_recovers_the_cartesian_force_that_generated_the_residual(self):
        """⭐ 构造 r = Jᵀ F_true，再映射回去必须还原 F_true。

        这是一条**可逆性**判据：正向用 Jᵀ 造，反向用伪逆解，
        两边用的是同一个 J 但**不同的运算**，所以不是自证循环。
        """
        rng = np.random.default_rng(20260828)
        J = rng.normal(size=(6, 7))              # 满行秩，几乎必然
        plant = LinearPlant(np.eye(7), jacobian=J)
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-3)
        F_true = np.array([1.0, -2.0, 3.0, 0.4, -0.5, 0.6])
        obs.r = J.T @ F_true
        F_hat = obs.cartesian_force(np.zeros(7))
        np.testing.assert_allclose(F_hat, F_true, rtol=1e-9, atol=1e-9)

    def test_a_residual_in_the_nullspace_maps_to_zero_force(self):
        """⭐ 边界：落在 Jᵀ 零空间里的残差**映射不出任何笛卡尔力**。

        真机含义：某些内部关节力矩在末端根本看不出来，
        所以"末端测不到力"不等于"机器人没被碰"。
        """
        rng = np.random.default_rng(7)
        J = rng.normal(size=(6, 7))
        plant = LinearPlant(np.eye(7), jacobian=J)
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-3)
        # Jᵀ 是 7×6，其左零空间维度 = 7 − 6 = 1
        u, s, vt = np.linalg.svd(J.T)
        null_dir = u[:, -1]                      # Jᵀ 的列空间之外
        self.assertLess(float(np.abs(J @ null_dir).max()), 1e-9)
        obs.r = 10.0 * null_dir
        F_hat = obs.cartesian_force(np.zeros(7))
        self.assertLess(float(np.abs(F_hat).max()), 1e-9,
                        f"零空间残差映射出了非零外力 {F_hat}")


class TestMomentumObserverBoundaries(unittest.TestCase):
    """⭐ 边界：观测器分不清"真外力"和"没建模的内部力"。"""

    def test_unmodelled_friction_shows_up_as_fake_external_torque(self):
        """⭐⭐ 这是动量观测器最重要的一条边界。

        观测器只知道 ``τ_cmd``。任何**没写进模型**的力矩——摩擦、
        减速器损耗、错的重力参数——都会被它算成"外力"。
        所以真机上阈值不能按理论噪声定，必须实测无负载时的残差本底。
        """
        plant = LinearPlant(np.diag([1.0]))
        obs = MomentumObserver(plant, k_i=25.0, dt=1e-4)
        # 完全没有外力，只有黏滞摩擦；同时给一个恒定驱动力矩让它动起来
        _, r = simulate(plant, obs, np.zeros(1), 1.5,
                        tau_cmd=np.array([2.0]),
                        friction=lambda qd: 0.8 * qd)
        self.assertGreater(float(np.abs(r[-1]).max()), 0.5,
                           "未建模摩擦居然没进入残差——那这条边界就不存在了")
        # 而且符号是负的：摩擦阻碍运动，等效于一个反向的"外力"
        self.assertLess(float(r[-1][0]), 0.0)

    def test_gravity_model_error_also_leaks_into_the_residual(self):
        """重力参数标错，同样会被当成外力——所以重力补偿必须先标准。"""
        plant = LinearPlant(np.diag([1.0]))

        class WrongGravity(LinearPlant):
            def gravity(self, q):
                return np.array([1.5])           # 观测器以为有 1.5 N·m 重力

        wrong = WrongGravity(np.diag([1.0]))
        obs = MomentumObserver(wrong, k_i=25.0, dt=1e-4)
        _, r = simulate(plant, obs, np.zeros(1), 1.0, tau_cmd=np.zeros(1))
        self.assertGreater(float(np.abs(r[-1]).max()), 1.0,
                           "重力模型错了 1.5 N·m，残差却没反应")
        self.assertAlmostEqual(float(r[-1][0]), 1.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
