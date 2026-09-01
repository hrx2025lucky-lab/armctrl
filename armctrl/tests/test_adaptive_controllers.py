"""自适应控制器的直接测试（第四轮审核 T-09 补的覆盖缺口）。

⚠️ 补这个文件之前，``armctrl/tests`` 里**没有任何一处**直接命中
``SlotineLiAdaptiveController`` 或 ``RBFAdaptiveController``。
它们只被 ``AdaptiveScene`` 间接跑过——而场景测试证明不了自适应律的**符号**。

⭐ 符号为什么是重中之重：第四轮审核的 P0-01 就是自适应律符号写反
（4 个文件 11 处）。符号反了以后，跟踪误差**不是收敛而是发散**，
但如果只跑几百步、只看"曲线在动"，是看不出来的。

oracle 独立性说明
----------------
1. **合成的参数线性系统。**自己搭一个 ``τ = m·a + d·v`` 的一自由度对象，
   它的动力学**严格**满足 ``τ = Y(q,q̇,v_ref,a_ref) π*``，
   于是 Slotine-Li 理论的全部前提都精确成立，``π*`` 是已知真值。
   判据是"``‖s‖`` 收不收敛"和"``β̂`` 收不收敛到 ``π*``"——
   **外部可观测量**，不读控制器的任何中间量。
2. **σ-modification 的闭式解。**令 ``s = 0``，权重方程退化成
   ``Ẇ = −γσW``，闭式解 ``W(t) = W₀ e^{−γσt}``。
3. **内建变异对照。**每条关键判据都配一个"把符号改反"的对照，
   对照必须失败。判据抓不到变异就等于没测。

⚠️ 级别：**理论级**（判据是独立解析解/已知真值），
但被控对象是简化模型，不是真机。
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.control.adaptive import (RBFAdaptiveController,
                                      SlotineLiAdaptiveController,
                                      fit_centers_to_trajectory)


class MassDamperRegressor:
    """一自由度质量-阻尼系统的回归器：``τ = m·a_ref + d·v_ref``。

    ⭐ 这个对象**严格**是参数线性的，所以 ``π* = [m, d]`` 是精确真值。
    用真实 Panda 反而会把"回归器对不对"和"自适应律对不对"混在一起。
    """

    nv = 1
    n_par = 2

    @staticmethod
    def slotine_li_regressor(q, qd, v_ref, a_ref):
        return np.array([[float(a_ref[0]), float(v_ref[0])]])

    def base_parameter_projection(self, samples=300):
        return np.eye(self.n_par), self.n_par


TRUE_PI = np.array([2.5, 0.8])          # 真实的 [质量, 阻尼]


def run_slotine_li(ctrl, ref, seconds=6.0, dt=1e-3, flip_sign=False,
                   q0=0.0, v0=0.0):
    """闭环仿真。``ref(t)`` 返回 (q_des, v_des, a_des)。

    真实对象：``m q̈ + d q̇ = τ``，其中 m、d 取自 ``TRUE_PI``。
    ``flip_sign=True`` 时把自适应律的符号改反——**变异对照**。
    """
    m, d = TRUE_PI
    q, v = np.array([q0]), np.array([v0])
    ts, ss, betas = [], [], []
    for k in range(int(round(seconds / dt))):
        t = k * dt
        qd_, vd_, ad_ = ref(t)
        if flip_sign:
            before = ctrl.beta.copy()
            tau, s = ctrl.compute(q, v, qd_, vd_, ad_, dt)
            ctrl.beta[:] = before - (ctrl.beta - before)     # 符号反过来
        else:
            tau, s = ctrl.compute(q, v, qd_, vd_, ad_, dt)
        ts.append(t)
        ss.append(float(np.linalg.norm(s)))
        betas.append(ctrl.beta.copy())
        acc = (float(tau[0]) - d * float(v[0])) / m
        v = v + acc * dt
        q = q + v * dt
        if not np.isfinite(q[0]) or abs(q[0]) > 1e6:         # 已经炸了
            break
    return np.asarray(ts), np.asarray(ss), np.asarray(betas)


def sine_ref(amp=0.6, w=3.0):
    def ref(t):
        return (np.array([amp * np.sin(w * t)]),
                np.array([amp * w * np.cos(w * t)]),
                np.array([-amp * w * w * np.sin(w * t)]))
    return ref


def constant_ref(target=0.5):
    """常值目标：**激励不足**（没有持续激励 PE）。"""
    def ref(t):
        return np.array([target]), np.zeros(1), np.zeros(1)
    return ref


def make_sl(gamma=8.0, kd=12.0, lam=6.0, **kw):
    return SlotineLiAdaptiveController(
        MassDamperRegressor(), n_ctrl=1, lam=lam, kd=kd, gamma=gamma,
        use_base=False, **kw)


class TestSlotineLiStructure(unittest.TestCase):
    """控制律的代数结构（不跑闭环）。"""

    def test_control_law_is_regressor_times_estimate_plus_damping(self):
        """⭐ τ = Y β̂ + K_D s，逐项独立复算。"""
        c = make_sl(gamma=0.0)          # 关掉自适应，只看控制律本身
        c.beta[:] = np.array([1.7, -0.4])
        q, v = np.array([0.2]), np.array([-0.3])
        qd_, vd_, ad_ = np.array([0.5]), np.array([0.1]), np.array([0.7])
        tau, s = c.compute(q, v, qd_, vd_, ad_, dt=1e-3)
        e = qd_ - q
        de = vd_ - v
        s_ref = de + c.lam * e
        v_ref = vd_ + c.lam * e
        a_ref = ad_ + c.lam * de
        Y = MassDamperRegressor.slotine_li_regressor(q, v, v_ref, a_ref)
        np.testing.assert_allclose(s, s_ref, atol=1e-12)
        np.testing.assert_allclose(tau, Y @ c.beta + c.kd * s_ref, atol=1e-12)

    def test_adaptation_direction_is_gamma_Y_transpose_s(self):
        """⭐⭐ 自适应律符号：β̂ 必须朝 **+Γ Yᵀs** 走（P0-01 就是这里写反的）。"""
        c = make_sl(gamma=3.0)
        c.beta[:] = 0.0
        q, v = np.array([0.0]), np.array([0.0])
        qd_, vd_, ad_ = np.array([1.0]), np.array([1.0]), np.array([0.0])
        dt = 1e-3
        _, s = c.compute(q, v, qd_, vd_, ad_, dt)
        e = qd_ - q
        v_ref = vd_ + c.lam * e
        a_ref = ad_ + c.lam * (vd_ - v)
        Y = MassDamperRegressor.slotine_li_regressor(q, v, v_ref, a_ref)
        np.testing.assert_allclose(c.beta, 3.0 * (Y.T @ s).ravel() * dt,
                                   rtol=1e-12, atol=1e-15)
        # 这一步 s>0 且 Y 的两列都 >0 ⇒ 两个参数都必须**增大**
        self.assertGreater(float(s[0]), 0.0)
        self.assertTrue(np.all(Y > 0.0))
        self.assertTrue(np.all(c.beta > 0.0),
                        f"β̂ 走反了：{c.beta}（自适应律符号写反的典型症状）")

    def test_reset_clears_the_estimate(self):
        c = make_sl(gamma=5.0)
        run_slotine_li(c, sine_ref(), seconds=0.5)
        self.assertGreater(float(np.linalg.norm(c.beta)), 1e-6)
        c.reset()
        self.assertLess(float(np.linalg.norm(c.beta)), 1e-15)

    def test_projection_bounds_the_estimate(self):
        """参数投影必须真的把估计压在球内。"""
        c = make_sl(gamma=5e4, proj_bound=3.0)
        run_slotine_li(c, sine_ref(amp=1.5, w=8.0), seconds=1.0)
        self.assertLessEqual(float(np.linalg.norm(c.beta)), 3.0 + 1e-9)


class TestSlotineLiClosedLoop(unittest.TestCase):
    """闭环行为：判据全是外部可观测量。"""

    def test_tracking_error_converges(self):
        """⭐ 滤波误差 ‖s‖ 必须收敛。"""
        c = make_sl()
        t, s, _ = run_slotine_li(c, sine_ref(), seconds=8.0)
        head = float(np.max(s[: len(s) // 8]))
        tail = float(np.max(s[-len(s) // 8:]))
        self.assertLess(tail, 0.2 * head,
                        f"‖s‖ 没有收敛：前段峰值 {head:.4f} → 后段 {tail:.4f}")

    def test_flipping_the_adaptation_sign_makes_it_diverge(self):
        """⭐⭐ 变异对照：自适应律符号反过来，闭环必须**发散**。

        这条测试的存在意义就是守住 P0-01。
        如果它通不过（即反符号也收敛），说明上面那条收敛测试没有区分能力。
        """
        c = make_sl()
        t, s, _ = run_slotine_li(c, sine_ref(), seconds=8.0, flip_sign=True)
        head = float(np.max(s[: max(1, len(s) // 8)]))
        tail = float(np.max(s[-max(1, len(s) // 8):]))
        self.assertGreater(tail, 5.0 * head,
                           f"符号反了居然还收敛：{head:.4f} → {tail:.4f}，"
                           "说明这个判据抓不到符号错误")

    def test_persistent_excitation_makes_parameters_converge(self):
        """⭐ 有持续激励时，估计必须收敛到**真值** [m, d]。"""
        c = make_sl(gamma=25.0, kd=15.0)
        _, _, b = run_slotine_li(c, sine_ref(amp=0.8, w=4.0), seconds=60.0)
        err = float(np.linalg.norm(b[-1] - TRUE_PI) / np.linalg.norm(TRUE_PI))
        self.assertLess(err, 0.15,
                        f"参数没收敛到真值：估计 {b[-1]} vs 真值 {TRUE_PI}，"
                        f"相对误差 {err:.1%}")

    def test_without_excitation_tracking_converges_but_parameters_do_not(self):
        """⭐⭐ 本项目最该记住的一条：**跟踪收敛 ≠ 参数收敛**。

        常值目标下没有持续激励，误差照样趋于 0，
        但参数估计**停在一条错误的直线上**——因为多组参数都能解释同一条数据。
        真机含义：靠"轨迹跟得准"来证明"我辨识对了"，是站不住脚的。
        """
        c = make_sl(gamma=25.0, kd=15.0)
        _, s, b = run_slotine_li(c, constant_ref(0.5), seconds=40.0)
        self.assertLess(float(np.max(s[-len(s) // 8:])), 0.05,
                        "常值目标下跟踪本该收敛")
        err = float(np.linalg.norm(b[-1] - TRUE_PI) / np.linalg.norm(TRUE_PI))
        self.assertGreater(err, 0.3,
                           f"没有持续激励，参数却收敛到了真值？估计 {b[-1]}，"
                           f"相对误差 {err:.1%} —— 该重新检查这个对照实验")


class TestRBFBasis(unittest.TestCase):
    """高斯基函数的性质（纯数学，独立可验）。"""

    def _c(self, **kw):
        C = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.5, 0.5, 0.5])
        return RBFAdaptiveController(n_ctrl=1, centers=C, widths=b, **kw)

    def test_basis_peaks_at_one_on_its_own_center(self):
        c = self._c()
        h = c.basis(np.array([1.0, 0.0]))
        self.assertAlmostEqual(float(h[1]), 1.0, places=12)
        self.assertLess(float(h[0]), 1.0)

    def test_basis_matches_the_gaussian_formula(self):
        """独立按 exp(−‖x−c‖²/2b²) 复算。"""
        c = self._c()
        x = np.array([0.3, -0.7])
        ref = np.exp(-np.sum((c.C - x) ** 2, axis=1) / (2.0 * c.b ** 2))
        np.testing.assert_allclose(c.basis(x), ref, rtol=1e-14)

    def test_basis_decays_monotonically_with_distance(self):
        c = self._c()
        vals = [float(c.basis(np.array([r, 0.0]))[0]) for r in (0.0, 0.3, 0.8, 2.0)]
        self.assertTrue(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
                        f"高斯基不是单调衰减：{vals}")


class TestRBFAdaptation(unittest.TestCase):
    """权重更新律 Ŵ̇ = Γ(h sᵀ − σŴ)。"""

    def _c(self, gamma=300.0, sigma=0.02, n_ctrl=1, w_bound=500.0):
        C = np.array([[0.0] * 4, [1.0] + [0.0] * 3])
        b = np.array([0.5, 0.5])
        return RBFAdaptiveController(n_ctrl=n_ctrl, centers=C, widths=b,
                                     gamma=gamma, sigma=sigma, w_bound=w_bound)

    def test_sigma_modification_decays_weights_by_the_closed_form(self):
        """⭐ s=0 时 Ẇ = −γσW，闭式解 W(t) = W₀ e^{−γσt}，逐点对照。"""
        gamma, sigma, dt = 2.0, 0.5, 1e-4
        c = self._c(gamma=gamma, sigma=sigma)
        c.W[:] = np.array([[1.0], [-2.0]])
        W0 = c.W.copy()
        zero = np.zeros(1)
        # 目标与状态完全一致 ⇒ e=0, de=0 ⇒ s=0
        n = 3000
        for _ in range(n):
            c.compute(zero, zero, zero, zero, zero, dt)
        ref = W0 * np.exp(-gamma * sigma * n * dt)
        np.testing.assert_allclose(c.W, ref, rtol=2e-3,
                                   err_msg="σ-modification 的衰减率对不上闭式解")

    def test_a_wrong_leak_rate_would_fail_the_same_oracle(self):
        """⭐ 变异对照：泄漏率差一倍，上面那条判据必须抓到。"""
        gamma, sigma, dt = 2.0, 0.5, 1e-4
        c = self._c(gamma=gamma, sigma=sigma)
        c.W[:] = np.array([[1.0], [-2.0]])
        W0 = c.W.copy()
        zero = np.zeros(1)
        n = 3000
        for _ in range(n):
            c.compute(zero, zero, zero, zero, zero, dt)
        wrong = W0 * np.exp(-2.0 * gamma * sigma * n * dt)
        self.assertGreater(float(np.abs(c.W - wrong).max()), 1e-3,
                           "泄漏率差一倍也能通过——判据没有区分能力")

    def test_weight_update_follows_h_times_s_outer_product(self):
        """⭐ 一步更新必须等于 Γ(h sᵀ − σŴ)Δt，符号不能反。"""
        c = self._c(gamma=1.0, sigma=0.0)
        q, v = np.zeros(1), np.zeros(1)
        qd_, vd_, ad_ = np.array([1.0]), np.zeros(1), np.zeros(1)
        dt = 1e-3
        x = np.concatenate([q, v, vd_ + c.lam * (qd_ - q),
                            ad_ + c.lam * (vd_ - v)])
        h = c.basis(x)
        _, s = c.compute(q, v, qd_, vd_, ad_, dt)
        np.testing.assert_allclose(c.W, np.outer(h, s) * dt,
                                   rtol=1e-12, atol=1e-15)
        self.assertGreater(float(s[0]), 0.0)
        self.assertTrue(np.all(c.W > 0.0),
                        f"权重走反了：{c.W.ravel()}")

    def test_weight_norm_is_bounded(self):
        c = self._c(gamma=1e6, sigma=0.0, w_bound=4.0)
        q, v = np.zeros(1), np.zeros(1)
        for _ in range(50):
            c.compute(q, v, np.array([2.0]), np.zeros(1), np.zeros(1), 1e-3)
        self.assertLessEqual(float(np.linalg.norm(c.W)), 4.0 + 1e-9)

    def test_only_the_controlled_joints_get_a_torque(self):
        """⭐ 状态是 nv 维，但输出力矩只能有 n_ctrl 维（夹爪关节不参与）。"""
        C = np.zeros((2, 12))
        c = RBFAdaptiveController(n_ctrl=2, centers=C, widths=np.ones(2))
        q = np.zeros(3)
        tau, s = c.compute(q, q, np.array([1.0, 2.0, 3.0]), q, q, 1e-3)
        self.assertEqual(tau.shape, (2,))
        self.assertEqual(s.shape, (2,))

    def test_reset_clears_the_weights(self):
        c = self._c()
        c.compute(np.zeros(1), np.zeros(1), np.array([1.0]),
                  np.zeros(1), np.zeros(1), 1e-3)
        self.assertGreater(float(np.linalg.norm(c.W)), 0.0)
        c.reset()
        self.assertLess(float(np.linalg.norm(c.W)), 1e-15)


def run_rbf(ctrl, seconds=6.0, dt=1e-3, amp=0.5, w=2.5, m=1.0, d=0.5,
            nonlin=6.0):
    """RBF 闭环仿真。真实对象带一个 RBF 事先**不知道**的非线性 g(q)=nonlin·sin(q)。

    判据只看外部可观测量：跟踪误差。
    """
    q, v = np.array([0.0]), np.array([0.0])
    errs = []
    for k in range(int(round(seconds / dt))):
        t = k * dt
        qd_ = np.array([amp * np.sin(w * t)])
        vd_ = np.array([amp * w * np.cos(w * t)])
        ad_ = np.array([-amp * w * w * np.sin(w * t)])
        tau, _ = ctrl.compute(q, v, qd_, vd_, ad_, dt)
        errs.append(abs(float(qd_[0] - q[0])))
        acc = (float(tau[0]) - d * float(v[0]) - nonlin * np.sin(float(q[0]))) / m
        v = v + acc * dt
        q = q + v * dt
        if not np.isfinite(q[0]) or abs(q[0]) > 1e6:
            errs.extend([1e6] * 10)
            break
    return np.asarray(errs)


def make_rbf(gamma=300.0, sigma=0.02, kd=8.0, lam=4.0, amp=0.5, w=2.5):
    """用**生产代码自己的** ``fit_centers_to_trajectory`` 沿轨迹布中心。

    ⚠️ 我第一版是在 (q, q̇) 平面上手工铺 5×5 网格、其余两维填 0，
    结果学习几乎完全无效（误差只降 1~3%）。原因是基函数的输入
    ``x = [q; q̇; q̇ᵣ; q̈ᵣ]`` 是 **4 维**，而 ``q̇ᵣ``、``q̈ᵣ`` 在跟踪正弦时
    幅值好几个单位，离"填 0"的中心极远，``exp(−‖x−c‖²/2b²)`` 直接趋于 0——
    **整张网络是死的**。这正是 ``fit_centers_to_trajectory`` 存在的理由。
    """
    def traj(t):
        return (np.array([amp * np.sin(w * t)]),
                np.array([amp * w * np.cos(w * t)]),
                np.array([-amp * w * w * np.sin(w * t)]))
    C, widths = fit_centers_to_trajectory(traj, t_end=2 * np.pi / w,
                                          n_centers=120, lam=lam)
    return RBFAdaptiveController(n_ctrl=1, centers=C, widths=widths,
                                 lam=lam, kd=kd, gamma=gamma, sigma=sigma)


class TestRBFClosedLoop(unittest.TestCase):
    """RBF 的闭环行为。

    ⚠️ 补这一组是因为：只测更新律的代数结构，**抓不到鲁棒项 K_D s 的符号错误**。
    我把生产代码的 ``+ self.kd * s_ctrl`` 改成 ``-`` 之后，
    上面 11 条 RBF 结构测试**全部照常通过**。必须有闭环判据才守得住。
    """

    def test_the_robust_term_alone_stabilizes_the_plant(self):
        """⭐⭐ 关掉学习（Γ=0）后只剩 τ = K_D s，这是一个 PD，必须**稳定**。

        符号反了就变成正反馈，一定发散。这条就是守 K_D 符号的。
        """
        e = run_rbf(make_rbf(gamma=0.0), seconds=6.0)
        self.assertLess(float(np.max(e[-len(e) // 6:])), 0.5,
                        f"Γ=0 时纯 PD 都不稳定，末段误差 {np.max(e[-len(e)//6:]):.3f}"
                        "（鲁棒项 K_D s 符号写反的典型症状）")

    def test_learning_beats_the_fixed_gain_baseline(self):
        """⭐ 打开学习后，稳态跟踪误差必须**明显好于**同增益的纯 PD。

        这是 RBF 存在的全部理由：把未知非线性 g(q) 学出来前馈掉。

        实测（K_D=8, Λ=4, Γ=300, 12 s 末段 25% 平均 |e|）：

        | 配置 | 平均误差 | 相对纯 PD |
        |---|---|---|
        | 纯 PD（Γ=0） | 0.01108 rad | 1.00 |
        | 开学习 Γ=300 | 0.00060 rad | **0.054**（好 18 倍）|
        | 开学习 Γ=1000 | 0.00040 rad | 0.036 |

        ⭐ 顺带一个重要经验：增益越硬，学习的收益越小
        （K_D=60 时只好 3.3 倍）。**高增益会掩盖"没学会"**——
        只报"误差 0.6 mm"而不报增益，这个数字没有意义。
        """
        base = run_rbf(make_rbf(gamma=0.0), seconds=12.0)
        learn = run_rbf(make_rbf(gamma=300.0), seconds=12.0)
        b = float(np.mean(base[-len(base) // 4:]))
        l = float(np.mean(learn[-len(learn) // 4:]))
        self.assertLess(l, 0.15 * b,
                        f"学习没带来改善：纯 PD 末段平均误差 {b:.5f} rad，"
                        f"开学习 {l:.5f} rad（比值 {l / b:.3f}，实测应为 0.054）")

    def test_centers_must_cover_the_reference_dimensions_to_learn(self):
        """⭐⭐ 反例守卫：中心只铺 (q, q̇)、把 q̇ᵣ/q̈ᵣ 两维填 0 时，网络是**死的**。

        这条记录的是我自己踩的坑，也是 RBF 落地最常见的失败模式：
        基函数输出恒 ≈ 0 ⇒ 权重学不动 ⇒ 表面上"跑通了"，实则退化成纯 PD。
        """
        grid = np.linspace(-0.8, 0.8, 5)
        C = np.array([[a, b, 0.0, 0.0] for a in grid
                      for b in np.linspace(-2, 2, 5)])
        dead = RBFAdaptiveController(n_ctrl=1, centers=C,
                                     widths=np.full(C.shape[0], 0.6),
                                     lam=4.0, kd=8.0, gamma=300.0)
        e_dead = run_rbf(dead, seconds=12.0)
        e_ok = run_rbf(make_rbf(gamma=300.0), seconds=12.0)
        m_dead = float(np.mean(e_dead[-len(e_dead) // 4:]))
        m_ok = float(np.mean(e_ok[-len(e_ok) // 4:]))
        self.assertGreater(m_dead, 5.0 * m_ok,
                           "中心没覆盖参考维度居然也学得动？该重查这个对照")
        # 基函数确实几乎全灭
        x = np.array([0.3, 1.0, 1.2, -3.0])
        self.assertLess(float(dead.basis(x).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
