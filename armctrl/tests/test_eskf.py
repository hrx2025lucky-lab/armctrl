"""含姿态的误差状态卡尔曼滤波（ESKF）测试。

oracle 独立性说明
----------------
本文件的判据分三类，都不读被测实现的中间量：

1. **流形上的数值雅可比。**
   ``transition_matrix`` 手写的 F 只出现在协方差传播里，状态估计本身用不到它。
   所以 F 写错时状态照样跑得好好的，**只有协方差是错的**——这种错误极难靠
   观察发现。这里用 ``propagate_nominal``（纯函数）在流形上做中心差分：
   对名义状态施加小扰动 → 传播 → 用 ``local_coordinates`` 求回误差 →
   与 ``F·δx`` 比。**中心差分本身与 F 的推导完全无关**。

2. **闭式物理量。**
   比如陀螺零偏 b_g 恒定时姿态角按 ``θ = b_g·t`` 线性增长，
   加速度计零偏按 ``½·b·t²`` 平方增长；这些是从物理直接写出来的，
   不涉及滤波器实现。

3. **结构性判据。**
   偏航不可观来自 ``[u]ₓ`` 的零空间是 u 自己——这是反对称矩阵的代数性质，
   用 ``numpy.linalg`` 的秩/零空间独立验证，不看滤波器怎么说。

变异测试结果（17 个植入缺陷，抓到 16 个）
-------------------------------------
逐个植入下列缺陷并重跑本文件，括号里是失败的测试数：

F[θ,θ] 退化成单位阵 (3)｜F[θ,θ] 符号反 (3)｜F[θ,bg] 丢右雅可比 (2)｜
F[θ,bg] 符号反 (5)｜F[v,θ] 丢 skew (3)｜F[v,θ] 符号反 (4)｜F[p,θ] 漏二阶块 (1)｜
重力量测 H_θ 符号反 (1)｜重力量测漏 b_a 通道 (1)｜传播漏 +g (2)｜
陀螺零偏没扣 (3)｜误差重置雅可比退化 (1)｜Joseph 退化 (1)｜
陀螺噪声未按 dt 换算 (1)｜过程噪声漏姿态通道 (1)｜右雅可比二阶项符号反 (4)

⚠️ **唯一没抓到的是一个等价变异**：把 :func:`so3_exp` 小角度分支里的
``c1 = 1 − θ²/6`` 改成 ``c1 = 1``。该分支只在 ``θ < 1e-6`` 时启用，
此时两者相差 ``θ²/6·θ ≈ 1.7e-19``，而双精度分辨率是 2.2e-16——
**相差三个数量级，双精度下不可能观测到**。

这不是测试覆盖的缺口，是变异测试文献里明确要求排除的 equivalent mutant。
记在这里是为了下次有人重跑变异测试时不必再查一遍。

⚠️ 全部为**同模型自洽**级别：被估的"真值"来自同一套仿真，不是真机。
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.core.kinematics import skew, so3_exp, so3_log, so3_right_jacobian
from armctrl.estimation.eskf import (IDX_BA, IDX_BG, IDX_P, IDX_TH, IDX_V,
                                     ErrorStateKalmanFilter)

G_VEC = np.array([0.0, 0.0, -9.81])


def _random_state(rng) -> tuple:
    return (rng.normal(0, 0.5, 3), rng.normal(0, 0.3, 3),
            so3_exp(rng.normal(0, 0.6, 3)),
            rng.normal(0, 0.05, 3), rng.normal(0, 0.01, 3))


class _TruthTrajectory:
    """⭐ 独立的真值轨迹（第四轮审核 T-06 新增）。

    这个类**完全不认识 ESKF**：它自己积分一条 (R, v, p) 真值轨迹，
    然后按 IMU 的物理定义吐出理想读数

        比力  f = Rᵀ (a_world − g)
        角速度 ω（体系）

    滤波器只能拿到 ``f`` 和 ``ω``（还有可选的真值位置量测），
    **拿不到任何真值姿态**。这样才谈得上"独立判据"。

    ⚠️ 旧版测试是 ``f = kf.R.T @ (acc - g)`` —— 用滤波器**自己的估计**造读数，
    再喂回滤波器。这叫**同模型自洽**：滤波器姿态整个错了，实验也照样漂亮。
    """

    def __init__(self, *, rotate: bool, translate: bool, R0=None):
        self.rotate = rotate
        self.translate = translate
        self.R = np.eye(3) if R0 is None else np.asarray(R0, float).copy()
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.t = 0.0

    def accel_world(self) -> np.ndarray:
        if not self.translate:
            return np.zeros(3)
        t = self.t
        return np.array([0.8 * np.sin(3 * t), 0.6 * np.cos(2.2 * t),
                         0.5 * np.sin(1.7 * t)])

    def omega_body(self) -> np.ndarray:
        if not self.rotate:
            return np.zeros(3)
        t = self.t
        return np.array([0.6 * np.sin(2 * t), 0.5 * np.cos(1.3 * t),
                         0.4 * np.sin(0.7 * t)])

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """推进真值一步，返回这一步的理想 IMU 读数 ``(f, ω)``。"""
        a = self.accel_world()
        w = self.omega_body()
        f = self.R.T @ (a - G_VEC)          # ← 用**真值**姿态，不是估计
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.R = self.R @ so3_exp(w * dt)
        self.t += dt
        return f, w


class TestSO3Helpers(unittest.TestCase):
    """先锁住 SO(3) 工具本身，后面的推导全都建立在它们之上。"""

    def test_exp_log_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(400):
            w = rng.normal(0, 1, 3)
            w = w / max(np.linalg.norm(w), 1e-30) * rng.uniform(1e-9, 3.0)
            self.assertLess(np.linalg.norm(so3_log(so3_exp(w)) - w), 1e-12)

    def test_exp_is_a_rotation(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            R = so3_exp(rng.normal(0, 1, 3) * rng.uniform(0, 3))
            self.assertLess(np.linalg.norm(R @ R.T - np.eye(3)), 1e-12)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_right_jacobian_matches_its_definition(self):
        """J_r 的定义：Exp(ω+δ) ≈ Exp(ω)·Exp(J_r(ω)·δ)。

        独立判据——只用指数映射，不看 so3_right_jacobian 怎么实现的。
        """
        rng = np.random.default_rng(5)
        for _ in range(200):
            w = rng.normal(0, 1, 3) * rng.uniform(0.01, 2.5)
            d = rng.normal(0, 1, 3) * 1e-7
            lhs = so3_exp(w + d)
            rhs = so3_exp(w) @ so3_exp(so3_right_jacobian(w) @ d)
            self.assertLess(np.linalg.norm(lhs - rhs) / np.linalg.norm(d), 1e-6)

    def test_skew_null_space_is_the_vector_itself(self):
        """⭐ 偏航不可观的**代数根源**：[u]ₓ·u = 0。"""
        rng = np.random.default_rng(6)
        for _ in range(200):
            u = rng.normal(0, 1, 3)
            self.assertLess(np.linalg.norm(skew(u) @ u), 1e-12)
            self.assertEqual(np.linalg.matrix_rank(skew(u)), 2)


class TestTransitionJacobian(unittest.TestCase):
    """⭐ 流形上的数值雅可比：锁死 F 的每一个块。"""

    DT = 0.002
    EPS = 1e-6

    def _numeric_F(self, kf, state, f, w, dt) -> np.ndarray:
        """中心差分求 ∂δx⁺/∂δx，全程只用 propagate_nominal / retract /
        local_coordinates，**不碰 transition_matrix**。"""
        base = kf.propagate_nominal(state, f, w, dt)
        F = np.zeros((kf.N, kf.N))
        for j in range(kf.N):
            e = np.zeros(kf.N)
            e[j] = self.EPS
            plus = kf.propagate_nominal(kf.retract(state, e), f, w, dt)
            minus = kf.propagate_nominal(kf.retract(state, -e), f, w, dt)
            F[:, j] = (kf.local_coordinates(base, plus)
                       - kf.local_coordinates(base, minus)) / (2 * self.EPS)
        return F

    def _check(self, w_scale: float, seed: int, tol: float = 2e-5) -> None:
        rng = np.random.default_rng(seed)
        kf = ErrorStateKalmanFilter(gravity=G_VEC)
        for _ in range(12):
            state = _random_state(rng)
            f = rng.normal(0, 4.0, 3)
            w = rng.normal(0, w_scale, 3)
            Fa = kf.transition_matrix(state, f, w, self.DT)
            Fn = self._numeric_F(kf, state, f, w, self.DT)
            self.assertLess(np.abs(Fa - Fn).max(), tol,
                            f"F 与数值雅可比不符，最大偏差 "
                            f"{np.abs(Fa - Fn).max():.3e}")

    def test_slow_rotation(self):
        self._check(w_scale=0.2, seed=11)

    def test_fast_rotation(self):
        """⭐ 转得快时 F[θ,θ] = Exp(−ω̂dt) 与单位阵的差别才显出来。"""
        self._check(w_scale=25.0, seed=12)

    def test_large_timestep_exposes_second_order_blocks(self):
        """dt 放大到 20 ms，位置行的 ½dt² 块和 J_r 都更容易被抓到。"""
        rng = np.random.default_rng(13)
        kf = ErrorStateKalmanFilter(gravity=G_VEC)
        dt = 0.02
        for _ in range(10):
            state = _random_state(rng)
            f = rng.normal(0, 4.0, 3)
            w = rng.normal(0, 15.0, 3)
            Fa = kf.transition_matrix(state, f, w, dt)
            Fn = self._numeric_F(kf, state, f, w, dt)
            self.assertLess(np.abs(Fa - Fn).max(), 5e-5)

    def test_bias_rows_are_identity(self):
        """两个零偏都建模成随机游走，均值不变 ⇒ 对应行必须是单位阵。"""
        kf = ErrorStateKalmanFilter(gravity=G_VEC)
        rng = np.random.default_rng(14)
        F = kf.transition_matrix(_random_state(rng), rng.normal(0, 4, 3),
                                 rng.normal(0, 2, 3), self.DT)
        self.assertLess(np.abs(F[IDX_BA, :] - np.eye(kf.N)[IDX_BA, :]).max(), 1e-15)
        self.assertLess(np.abs(F[IDX_BG, :] - np.eye(kf.N)[IDX_BG, :]).max(), 1e-15)


class TestGravityMeasurement(unittest.TestCase):
    """加速度计水平修正的雅可比与它的可观性缺口。"""

    def test_jacobian_matches_finite_difference(self):
        """H_θ = −[Rᵀg]ₓ 用数值差分独立验证。

        量测函数 h(δθ) = −(R·Exp(δθ))ᵀ·g + b_a，只用旋转本身写出来，
        与 update_gravity 的实现无关。
        """
        rng = np.random.default_rng(21)
        for _ in range(60):
            R = so3_exp(rng.normal(0, 0.8, 3))
            ba = rng.normal(0, 0.05, 3)

            def h(dth):
                return -(R @ so3_exp(dth)).T @ G_VEC + ba

            Hn = np.zeros((3, 3))
            eps = 1e-7
            for j in range(3):
                e = np.zeros(3)
                e[j] = eps
                Hn[:, j] = (h(e) - h(-e)) / (2 * eps)
            Ha = -skew(R.T @ G_VEC)
            self.assertLess(np.abs(Ha - Hn).max(), 1e-5)

    def test_yaw_is_structurally_unobservable(self):
        """⭐ 沿重力轴的姿态误差不产生任何量测变化。

        这是本篇最重要的一条：它说明「光靠 IMU 定不住方位」不是调参问题，
        是**结构性**的。
        """
        rng = np.random.default_rng(22)
        for _ in range(60):
            R = so3_exp(rng.normal(0, 0.8, 3))
            u = R.T @ G_VEC
            u = u / np.linalg.norm(u)
            H_theta = -skew(R.T @ G_VEC)
            self.assertLess(np.linalg.norm(H_theta @ u), 1e-12)
            self.assertEqual(np.linalg.matrix_rank(H_theta), 2)

    def test_roll_and_pitch_are_observable(self):
        """与偏航相对：垂直于重力轴的两个方向必须产生非零量测。"""
        rng = np.random.default_rng(23)
        R = so3_exp(rng.normal(0, 0.5, 3))
        u = R.T @ G_VEC
        u = u / np.linalg.norm(u)
        basis = np.linalg.svd(u.reshape(1, 3))[2][1:]      # 与 u 正交的两个方向
        H_theta = -skew(R.T @ G_VEC)
        for d in basis:
            self.assertGreater(np.linalg.norm(H_theta @ d), 1.0)


class TestFilterBehaviour(unittest.TestCase):
    """在合成数据上跑整条链路，判据全部来自闭式物理量。"""

    DT = 0.002

    @staticmethod
    def _kf(**kw):
        base = dict(acc_noise=0.0, gyro_noise=0.0, acc_bias_walk=0.0,
                    gyro_bias_walk=0.0, pos_noise=1e-4, gravity=G_VEC)
        base.update(kw)
        return ErrorStateKalmanFilter(**base)

    def test_static_perfect_imu_stays_put(self):
        """完美 IMU、静止：位置、速度、姿态都不该动。"""
        kf = self._kf()
        R0 = so3_exp(np.array([0.1, -0.2, 0.3]))
        kf.reset(np.array([0.4, 0.0, 0.5]), np.zeros(3), R0)
        f = -R0.T @ G_VEC                       # 静置时的比力（见 imu.py）
        for _ in range(3000):
            kf.predict(f, np.zeros(3), self.DT)
        self.assertLess(np.linalg.norm(kf.p - np.array([0.4, 0.0, 0.5])), 1e-9)
        self.assertLess(np.linalg.norm(kf.v), 1e-9)
        self.assertLess(np.linalg.norm(so3_log(R0.T @ kf.R)), 1e-12)

    def test_gyro_bias_makes_attitude_drift_linearly(self):
        """⭐ 闭式判据：陀螺零偏恒定时姿态角按 θ = b_g·t **线性**增长。

        与加速度计零偏的 ½·b·t²（平方）形成对照——两者阶数不同，
        这个测试同时锁住了「陀螺零偏进的是姿态、只积一次」这件事。
        """
        bg_true = np.array([0.0, 0.0, 0.02])            # rad/s
        kf = self._kf()
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3))
        angles = {}
        n = int(10.0 / self.DT)
        for i in range(n):
            f = -kf.R.T @ G_VEC
            kf.predict(f, bg_true, self.DT)             # 滤波器不知道有零偏
            t = round((i + 1) * self.DT, 6)
            if t in (2.0, 4.0, 8.0):
                angles[t] = float(np.linalg.norm(so3_log(kf.R)))
        # 线性 ⇒ 时间翻倍角度翻倍
        self.assertAlmostEqual(angles[4.0] / angles[2.0], 2.0, delta=0.02)
        self.assertAlmostEqual(angles[8.0] / angles[4.0], 2.0, delta=0.02)
        self.assertAlmostEqual(angles[2.0], 0.02 * 2.0, delta=1e-3)

    def test_levelling_pulls_roll_pitch_back(self):
        """给一个错的初始姿态，只用加速度计修正，横滚俯仰应当被拉回。"""
        kf = self._kf(lev_noise=0.05)
        tilt = np.array([0.08, -0.06, 0.0])             # 约 4.6° / 3.4°
        kf.reset(np.zeros(3), np.zeros(3), so3_exp(tilt))
        f_true = -np.eye(3).T @ G_VEC                   # 真姿态是单位阵
        before = float(np.linalg.norm(so3_log(kf.R)))
        for _ in range(4000):
            kf.predict(f_true, np.zeros(3), self.DT)
            kf.update_gravity(f_true)
        after = float(np.linalg.norm(so3_log(kf.R)))
        self.assertLess(after, 0.1 * before,
                        f"水平修正没起作用：{before:.4f} → {after:.4f}")

    def test_levelling_cannot_fix_yaw(self):
        """⭐ 同样的装置，把误差换成纯偏航，修正**不应该**有效果。

        与上一条构成对照：同一段代码、同一组参数，只换误差方向，
        结论完全相反。这排除了「是不是滤波器整体不работает」的可能。
        """
        kf = self._kf(lev_noise=0.05)
        yaw = np.array([0.0, 0.0, 0.08])
        kf.reset(np.zeros(3), np.zeros(3), so3_exp(yaw))
        f_true = -np.eye(3).T @ G_VEC
        before = float(np.linalg.norm(so3_log(kf.R)))
        for _ in range(4000):
            kf.predict(f_true, np.zeros(3), self.DT)
            kf.update_gravity(f_true)
        after = float(np.linalg.norm(so3_log(kf.R)))
        self.assertGreater(after, 0.9 * before,
                           f"偏航居然被修掉了，说明可观性判断错了："
                           f"{before:.4f} → {after:.4f}")

    def test_attitude_update_does_fix_yaw(self):
        """引入外部姿态参考后，偏航立刻可修——证明上一条不是滤波器坏了。"""
        kf = self._kf()
        kf.reset(np.zeros(3), np.zeros(3), so3_exp(np.array([0.0, 0.0, 0.08])))
        before = float(np.linalg.norm(so3_log(kf.R)))
        for _ in range(2000):
            kf.predict(-kf.R.T @ G_VEC, np.zeros(3), self.DT)
            kf.update_attitude(np.eye(3))
        after = float(np.linalg.norm(so3_log(kf.R)))
        self.assertLess(after, 0.05 * before)

    def test_yaw_uncertainty_grows_without_heading_reference(self):
        """⭐ 只有位置 + 重力修正时，偏航方向的标准差必须**单调涨**。"""
        kf = self._kf(gyro_noise=1e-3, gyro_bias_walk=1e-5, lev_noise=0.05)
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3))
        f = -G_VEC
        samples = []
        for i in range(6000):
            kf.predict(f, np.zeros(3), self.DT)
            kf.update_gravity(f)
            if i % 10 == 0:
                kf.update_position(np.zeros(3))
            if i in (1000, 3000, 5999):
                samples.append(kf.yaw_uncertainty_deg())
        self.assertLess(samples[0], samples[1])
        self.assertLess(samples[1], samples[2])

    def _run_excitation(self, *, rotate: bool, translate: bool,
                        position: bool, n: int = 4000) -> tuple[int, float]:
        """按给定激励跑一段，返回 (可观性秩, 偏航标准差 °)。

        ⚠️ **本函数在第四轮审核（T-06）后重写过。** 旧版有两处自证循环：

        1. ``f = kf.R.T @ (acc - g)`` —— 用**滤波器自己估计的姿态** 造 IMU 读数；
        2. ``kf.update_position(kf.p)`` —— 把**滤波器自己估计的位置**当量测喂回去，
           新息恒为 0，等于什么都没修正。

        这两条合起来意味着：**即使滤波器的姿态整个错了，实验也照样"自洽"**，
        测不出任何东西。

        现在改为**独立积分一条真值轨迹** ``(R_true, v_true, p_true)``：
        IMU 读数由**真值姿态**生成，位置量测取**真值位置**，
        滤波器和真值之间没有任何信息回流。
        """
        truth = _TruthTrajectory(rotate=rotate, translate=translate)
        kf = self._kf(gyro_noise=1e-4, lev_noise=0.05)
        kf.reset(truth.p.copy(), truth.v.copy(), truth.R.copy())
        for i in range(n):
            f, w = truth.step(self.DT)
            kf.predict(f, w, self.DT)
            if i % 5 == 0:
                kf.update_gravity(f)
            if position and i % 10 == 0:
                kf.update_position(truth.p)          # ← 真值，不是 kf.p
        return kf.observability_matrix_rank(), kf.yaw_uncertainty_deg()

    def test_absolute_position_is_what_makes_yaw_observable(self):
        """⭐ 本模块最重要的一条，而且是**实测推翻我最初判断**得到的。

        我原本断言「只有位置 + 重力修正 ⇒ 秩 14，缺偏航」。扫描之后发现：
        真正决定偏航可不可观的是**有没有世界系绝对位置量测**，
        而不是有没有外部姿态参考。

        道理：位置轨迹在世界系、加速度计读数在体系，两者必须自洽，
        一比对就把完整姿态（含偏航）定死了。

        ⭐ 边界：真实浮动基座**没有**世界系绝对位置（腿部运动学只给相对位移），
        所以那里偏航仍然不可观。
        """
        rank_no_pos, _ = self._run_excitation(
            rotate=True, translate=True, position=False)
        rank_pos, _ = self._run_excitation(
            rotate=True, translate=True, position=True)
        self.assertLess(rank_no_pos, 15,
                        "没有绝对位置量测却满秩，说明可观性判据算错了")
        self.assertEqual(rank_pos, 15,
                         "有绝对位置量测 + 充分激励时应当满秩")

    def _yaw_error_run(self, *, position: bool, translate: bool = True,
                       yaw_err_deg: float = 10.0, n: int = 6000
                       ) -> tuple[float, float]:
        """⭐⭐ T-06 的核心判据：**注入一个已知的偏航误差，看它到底被不被修正**。

        滤波器的初始姿态被人为绕 z 轴转了 ``yaw_err_deg``；
        真值轨迹完全独立地跑，IMU 读数由真值姿态生成。
        返回 (初始偏航误差°, 末尾偏航误差°)——两者都是**真值与估计之差**，
        不读滤波器自报的任何协方差。

        这条判据能杀死"滤波器整个姿态都错了"这类错误，
        而旧版的同模型自洽实验对此完全免疫。
        """
        truth = _TruthTrajectory(rotate=False, translate=translate)
        kf = self._kf(gyro_noise=1e-4, lev_noise=0.05)
        yaw0 = np.deg2rad(yaw_err_deg)
        R_bad = so3_exp(np.array([0.0, 0.0, yaw0])) @ truth.R
        kf.reset(truth.p.copy(), truth.v.copy(), R_bad,
                 th_std=np.deg2rad(30.0))
        e0 = self._yaw_err_deg(truth.R, kf.R)
        for i in range(n):
            f, w = truth.step(self.DT)
            kf.predict(f, w, self.DT)
            if i % 5 == 0:
                kf.update_gravity(f)
            if position and i % 10 == 0:
                kf.update_position(truth.p)
        return e0, self._yaw_err_deg(truth.R, kf.R)

    @staticmethod
    def _yaw_err_deg(R_true, R_est) -> float:
        """绕世界系 z 轴的姿态误差角（度），真值 vs 估计。"""
        dR = R_est @ R_true.T
        return abs(float(np.rad2deg(np.arctan2(dR[1, 0], dR[0, 0]))))

    def test_position_measurements_actually_correct_an_injected_yaw_error(self):
        """⭐⭐ 本文件最强的一条判据（第四轮审核 T-06 新增）。

        给滤波器一个**故意转歪 10° 的初始偏航**，然后：

        - **有**世界系绝对位置量测 ⇒ 偏航误差必须被**压下去**；
        - **没有**位置量测 ⇒ 重力修正只能定住横滚/俯仰，
          偏航**修不动**（这正是"绕重力方向旋转不改变重力读数"的直接后果）。

        为什么这条比看秩更硬：秩是滤波器**自己算的**，
        而这里比的是**真值与估计之差**——滤波器没法自己给自己打分。
        """
        e0_no, e1_no = self._yaw_error_run(position=False)
        e0_yes, e1_yes = self._yaw_error_run(position=True)
        self.assertAlmostEqual(e0_no, 10.0, places=6)
        self.assertAlmostEqual(e0_yes, 10.0, places=6)
        self.assertLess(e1_yes, 0.3 * e0_yes,
                        f"有绝对位置量测却修不动偏航：{e0_yes:.2f}° → {e1_yes:.2f}°")
        self.assertGreater(e1_no, 0.7 * e0_no,
                           f"没有位置量测偏航却自己好了：{e0_no:.2f}° → {e1_no:.2f}°，"
                           "该重查这个对照实验")
        self.assertLess(e1_yes, 0.3 * e1_no)

    def test_yaw_stays_unobservable_without_horizontal_acceleration(self):
        """⭐ 边界：**光有位置量测还不够，必须有水平加速度**。

        真值完全静止时，位置量测虽然在，但对偏航没有任何约束——
        因为"位置一直是 0"这件事，绕 z 转多少度都成立。

        真机含义：浮动基座机器人站着不动时，偏航会慢慢漂，
        **这不是滤波器坏了，是信息本来就不存在。**
        """
        e0, e1 = self._yaw_error_run(position=True, translate=False)
        self.assertGreater(e1, 0.7 * e0,
                           f"静止无激励时偏航居然被修正了：{e0:.2f}° → {e1:.2f}°")

    def test_excitation_is_required_even_with_position(self):
        """⭐ 又一次持续激励条件：静止时即使有位置量测也补不满秩。"""
        rank_static, _ = self._run_excitation(
            rotate=False, translate=False, position=True)
        rank_moving, _ = self._run_excitation(
            rotate=False, translate=True, position=True)
        self.assertLess(rank_static, 15)
        self.assertEqual(rank_moving, 15)

    def test_horizontal_motion_pins_yaw_tightest(self):
        """水平平移直接产生水平加速度，对偏航的约束最强。"""
        _, yaw_static = self._run_excitation(
            rotate=False, translate=False, position=True)
        _, yaw_trans = self._run_excitation(
            rotate=False, translate=True, position=True)
        self.assertLess(yaw_trans, 0.2 * yaw_static)

    def test_covariance_stays_symmetric_positive_semidefinite(self):
        """Joseph 形式 + 重置雅可比之后，P 必须仍是对称半正定。"""
        rng = np.random.default_rng(31)
        kf = self._kf(acc_noise=5e-3, gyro_noise=1e-3, lev_noise=0.1)
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3))
        for i in range(2000):
            f = -kf.R.T @ G_VEC + rng.normal(0, 0.05, 3)
            kf.predict(f, rng.normal(0, 0.02, 3), self.DT)
            if i % 5 == 0:
                kf.update_gravity(f)
            if i % 20 == 0:
                kf.update_position(rng.normal(0, 1e-3, 3))
            self.assertLess(np.abs(kf.P - kf.P.T).max(), 1e-12)
            self.assertGreater(float(np.min(np.linalg.eigvalsh(kf.P))), -1e-12)

    def test_rotation_stays_orthonormal_over_long_run(self):
        """长时间传播后名义旋转仍应是严格的旋转矩阵。"""
        rng = np.random.default_rng(32)
        kf = self._kf()
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3))
        for _ in range(20000):
            kf.predict(-kf.R.T @ G_VEC, rng.normal(0, 1.0, 3), self.DT)
        self.assertLess(np.abs(kf.R @ kf.R.T - np.eye(3)).max(), 1e-12)
        self.assertAlmostEqual(float(np.linalg.det(kf.R)), 1.0, places=12)

    def test_retract_and_local_coordinates_are_inverse(self):
        """两个流形工具必须严格互逆，否则数值雅可比测试本身就不可信。"""
        rng = np.random.default_rng(33)
        kf = self._kf()
        for _ in range(200):
            s = _random_state(rng)
            dx = rng.normal(0, 0.05, kf.N)
            back = kf.local_coordinates(s, kf.retract(s, dx))
            self.assertLess(np.abs(back - dx).max(), 1e-11)


class TestCovarianceConsistency(unittest.TestCase):
    """⭐ 协方差是否**可信**。

    状态估计有个隐蔽的失效模式：**估计值看着挺准，但协方差是错的**。
    协方差错了滤波器就会误判该信谁，长期会发散或者拒绝有用量测。
    而它不影响单次运行的估计值，**光看误差曲线永远发现不了**。

    这一组测试专门盯协方差，判据全部独立于实现。
    """

    DT = 0.002

    def test_exact_reset_jacobian_is_the_inverse_right_jacobian(self):
        """先把「精确答案」用数值差分钉死，再拿实现去比。

        重置映射：注入 dθ 后名义姿态变成 R·Exp(dθ)，真实姿态不变，于是

            R·Exp(δθ) = R·Exp(dθ)·Exp(δθ')  ⇒  δθ' = Log(Exp(−dθ)·Exp(δθ))

        对 δθ 在 0 处求偏导。用李群的标准结果 Log(Exp(x)Exp(δ)) ≈ x + J_r(x)⁻¹δ，
        精确雅可比应当是 **J_r(−dθ)⁻¹**。

        这里用中心差分独立确认这一点（实测最大偏差 2.4e-9）。
        """
        rng = np.random.default_rng(41)
        for _ in range(80):
            a = rng.normal(0, 1, 3) * rng.uniform(0.01, 0.6)
            Gn = np.zeros((3, 3))
            eps = 1e-7
            for j in range(3):
                e = np.zeros(3)
                e[j] = eps
                Gn[:, j] = (so3_log(so3_exp(-a) @ so3_exp(e))
                            - so3_log(so3_exp(-a) @ so3_exp(-e))) / (2 * eps)
            G_exact = np.linalg.inv(so3_right_jacobian(-a))
            self.assertLess(np.abs(G_exact - Gn).max(), 1e-6)

    def test_reset_jacobian_first_order_form_is_accurate_to_second_order(self):
        """⭐ 实现用的是一阶截断 I − ½[dθ]ₓ，它相对精确式的误差必须是 O(|dθ|²)。

        **容差是推出来的，不是拍的。**把 J_r(−a)⁻¹ 按 |a| 展开，
        二阶项的系数是 **1/12**：

            J_r(−a)⁻¹ = I − ½[a]ₓ + (1/12)[a]ₓ² + O(|a|³)

        实测扫描 |a| = 0.05 … 0.6，偏差 / |a|² 稳定在 **0.0833–0.0838**，
        与 1/12 = 0.08333 吻合。所以判据取 0.09·|a|²（留 8% 余量）。

        这条测试同时说明：**保留一阶截断是合理的工程取舍**——
        重置量 dθ 通常很小（毫弧度量级），二阶误差可以忽略；
        但如果哪天有人把 ½ 写错或者整块删掉，这条会立刻失败。
        """
        rng = np.random.default_rng(42)
        ratios = []
        for _ in range(400):
            a = rng.normal(0, 1, 3)
            a = a / np.linalg.norm(a) * rng.uniform(0.02, 0.6)
            G_exact = np.linalg.inv(so3_right_jacobian(-a))
            G_impl = np.eye(3) - 0.5 * skew(a)
            mag2 = float(np.linalg.norm(a)) ** 2
            r = float(np.abs(G_exact - G_impl).max()) / mag2
            self.assertLess(r, 0.09, "二阶误差超出 1/12 的理论上界")
            ratios.append(r)
        # 单个样本的比值取决于 a 的方向，可以小于 1/12；
        # 但**最坏方向**必须逼近 1/12，否则说明实现根本不是这个一阶截断。
        self.assertGreater(max(ratios), 0.08,
                           f"最坏比值只有 {max(ratios):.5f}，"
                           f"与理论系数 1/12 = 0.08333 不符")

    def test_levelling_can_estimate_accelerometer_bias(self):
        """⭐ 重力量测的 H_ba 通道必须真的把零偏估出来。

        单靠水平修正，倾角与零偏是**分不开的**（一个小倾斜和一个小零偏
        产生一样的读数）。所以这里额外开外部姿态量测把姿态钉死，
        剩下的偏差就只能归给零偏——这时零偏必须收敛。

        判据是与**注入的真值**比，独立于滤波器。
        """
        ba_true = np.array([0.05, -0.03, 0.02])
        kf = ErrorStateKalmanFilter(
            acc_noise=1e-3, gyro_noise=1e-5, acc_bias_walk=0.0,
            gyro_bias_walk=0.0, lev_noise=0.02, gravity=G_VEC)
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3))
        f_meas = -np.eye(3).T @ G_VEC + ba_true      # 静置 + 真实零偏
        for _ in range(8000):
            kf.predict(f_meas, np.zeros(3), self.DT)
            kf.update_attitude(np.eye(3))            # 把姿态钉死
            kf.update_gravity(f_meas)
        err = float(np.linalg.norm(kf.ba - ba_true))
        self.assertLess(err, 0.2 * float(np.linalg.norm(ba_true)),
                        f"零偏没被估出来：估计 {kf.ba} vs 真值 {ba_true}")

    def _nees_trial(self, seed: int, n: int = 1500) -> float:
        """跑一次带真噪声的仿真，返回姿态通道的 NEES。

        NEES = δᵀ·P⁻¹·δ。若协方差建模正确，它的期望值等于自由度数（这里 3）。
        真值轨迹是我们自己造的，噪声是我们自己加的，
        **P 是滤波器自己报的**——两边独立，所以这个比值能检验 P 对不对。
        """
        rng = np.random.default_rng(seed)
        sa, sg = 2.0e-2, 5.0e-3
        kf = ErrorStateKalmanFilter(
            acc_noise=sa, gyro_noise=sg, acc_bias_walk=0.0,
            gyro_bias_walk=0.0, pos_noise=1e-3, lev_noise=0.05, gravity=G_VEC)
        R_true = np.eye(3)
        p_true = np.zeros(3)
        v_true = np.zeros(3)
        kf.reset(p_true, v_true, R_true, th_std=1e-4, ba_std=1e-4, bg_std=1e-4)
        for i in range(n):
            t = i * self.DT
            w_true = np.array([0.5 * np.sin(2 * t), 0.4 * np.cos(1.7 * t), 0.3])
            a_true = np.array([0.6 * np.sin(3 * t), 0.5 * np.cos(2 * t), 0.0])
            f_true = R_true.T @ (a_true - G_VEC)
            # 真值推进
            p_true = p_true + v_true * self.DT + 0.5 * a_true * self.DT ** 2
            v_true = v_true + a_true * self.DT
            R_true = R_true @ so3_exp(w_true * self.DT)
            # 带噪声的量测
            f_m = f_true + rng.normal(0, sa / np.sqrt(self.DT), 3)
            w_m = w_true + rng.normal(0, sg / np.sqrt(self.DT), 3)
            kf.predict(f_m, w_m, self.DT)
            if i % 10 == 0:
                kf.update_position(p_true + rng.normal(0, 1e-3, 3))
        d = so3_log(kf.R.T @ R_true)
        P_th = kf.P[IDX_TH, IDX_TH]
        return float(d @ np.linalg.solve(P_th, d))

    def test_attitude_covariance_is_consistent(self):
        """⭐ 蒙特卡洛一致性检验（NEES）：协方差既不能过于自信也不能过于保守。

        跑 60 次独立试验，姿态误差的 NEES 平均值应当接近自由度 3。

        * **平均值远大于 3** → 滤波器**过于自信**（P 太小）。
          这是最危险的情况：它会拒绝有用的量测然后发散。
          过程噪声漏项、噪声密度没按 dt 换算、Joseph 形式退化，
          都会表现成这个。
        * 平均值远小于 3 → 过于保守，没充分利用信息。

        卡方分布 3 自由度、60 次平均，95% 区间约 [2.4, 3.7]；
        这里放宽到 [1.5, 6.0]，只抓量级性的错误。
        """
        vals = [self._nees_trial(seed) for seed in range(60)]
        mean = float(np.mean(vals))
        self.assertGreater(mean, 1.5,
                           f"NEES 均值 {mean:.2f} 过小：协方差过于保守")
        self.assertLess(mean, 6.0,
                        f"NEES 均值 {mean:.2f} 过大：⭐ 滤波器过于自信，"
                        f"协方差被低估了")


class TestCovarianceUpdateForm(unittest.TestCase):
    """协方差更新的两处结构：Joseph 形式与误差重置变换。

    ⭐ **诚实说明（这两条都是变异测试逼出来的）**

    **一、Joseph 形式与 (I−KH)P 在最优增益下数学恒等。**
    可以两行推出来：

    .. code-block:: text

        Pj = P − KHP − PHᵀKᵀ + K(HPHᵀ+R)Kᵀ = P − KHP − PHᵀKᵀ + KSKᵀ
        KSKᵀ = PHᵀS⁻¹·S·S⁻¹HP = PHᵀS⁻¹HP = KHP
        ⇒ Pj = P − PHᵀKᵀ = (P − KHP)ᵀ = Psᵀ，而 Pj 对称 ⇒ Pj = Ps

    实测 500 组随机良态问题，最大相对差 **6.1e-16**（机器精度）。

    所以**任何统计一致性测试都区分不了这两种写法**——我一开始就是这么试的，
    变异测试直接从眼皮底下溜过去。Joseph 的价值不在估计正确性，
    而在**有限精度下的数值结构保持**：它对任意增益都保持对称半正定，
    简化式只在增益恰好最优时才对。

    区别只在**病态**时显现，所以下面用条件数 1e24 的近奇异先验。

    **二、误差重置变换的效应是二阶的。**
    G_θ = I − ½[dθ]ₓ 让 P 相对变化约 |dθ|²/12：注入 0.02 rad 只有 0.01%，
    0.4 rad 才到 4%。蒙特卡洛一致性检验在实际可跑的样本数下**噪声高于它**，
    所以同样必须用直接比对。

    两条测试都带一个**守护断言**：万一换了设置两个候选值又变得没区别，
    测试会主动报告自己失效，而不是默默通过。
    """

    N = ErrorStateKalmanFilter.N

    @classmethod
    def _ill_conditioned_position_prior(cls) -> np.ndarray:
        """位置块病态、其余块干净、**块间零相关**的先验。

        为什么要块间零相关：位置量测的增益是 K = P·Hᵀ·S⁻¹，
        而 P·Hᵀ 取的正是 P 的**位置那几列**。如果存在 p–θ 交叉相关，
        一次位置修正会顺带产生非零的姿态修正量 dx_θ，
        于是误差重置变换 G 也被触发——两个结构就混在一起分不开了。
        （这一条是被自己的测试失败逼出来的：最初用满阵先验，
        位置更新竟然带出 0.03 量级的姿态修正。）

        位置块用条件数 1e24 的近奇异阵来逼出 Joseph 与简化式的数值差别。
        """
        rng = np.random.default_rng(5)
        U, _, _ = np.linalg.svd(rng.standard_normal((3, 3)))
        P = np.zeros((cls.N, cls.N))
        P[IDX_P, IDX_P] = U @ np.diag([1e6, 1e-6, 1e-18]) @ U.T
        for blk in (IDX_V, IDX_TH, IDX_BA, IDX_BG):
            P[blk, blk] = np.eye(3) * 1e-6
        return 0.5 * (P + P.T)

    def test_position_update_uses_joseph_form(self):
        """病态先验 + 位置量测：实现必须贴 Joseph、远离简化式。

        位置量测的 H 只碰 δp，注入的 δθ 为零，所以重置变换是单位阵，
        这条测试能**单独**隔离出 Joseph 那一步。
        """
        kf = ErrorStateKalmanFilter(pos_noise=1e-10, gravity=G_VEC)
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3))
        kf.P = self._ill_conditioned_position_prior()
        P0 = kf.P.copy()

        H = np.zeros((3, self.N))
        H[:, IDX_P] = np.eye(3)
        Rm = (kf.pos_noise ** 2) * np.eye(3)
        K = P0 @ H.T @ np.linalg.inv(H @ P0 @ H.T + Rm)
        IKH = np.eye(self.N) - K @ H
        joseph = IKH @ P0 @ IKH.T + K @ Rm @ K.T
        joseph = 0.5 * (joseph + joseph.T)
        simple = 0.5 * ((IKH @ P0) + (IKH @ P0).T)

        kf.update_position(np.array([0.1, -0.2, 0.3]))

        scale = max(float(np.abs(joseph).max()), 1.0)
        d_j = float(np.abs(kf.P - joseph).max()) / scale
        d_s = float(np.abs(kf.P - simple).max()) / scale
        self.assertGreater(d_s, 1e-9,
                           f"两种写法在本设置下没有区分度（差 {d_s:.3e}），"
                           "测试失去意义，请换更病态的先验")
        self.assertLess(d_j, 1e-12,
                        f"协方差更新与 Joseph 形式不符（差 {d_j:.3e}）")
        self.assertLess(d_j, 1e-3 * d_s)

    def test_attitude_update_applies_the_error_reset_transform(self):
        """⭐ 大姿态修正后，P 必须再经过 G·P·Gᵀ 这一步。

        构造一个**很大**的修正（约 0.5 rad）把二阶效应放大到可测，
        然后把「做了重置变换」和「没做」两个候选值都独立算出来比。
        """
        kf = ErrorStateKalmanFilter(gravity=G_VEC)
        kf.reset(np.zeros(3), np.zeros(3), np.eye(3),
                 th_std=0.35, ba_std=1e-6, bg_std=1e-6, p_std=1e-6, v_std=1e-6)
        P0 = kf.P.copy()

        R_meas = so3_exp(np.array([0.30, -0.25, 0.35]))   # 离名义姿态很远
        sigma = 0.02
        H = np.zeros((3, self.N))
        H[:, IDX_TH] = np.eye(3)
        Rm = (sigma ** 2) * np.eye(3)
        y = so3_log(np.eye(3).T @ R_meas)
        K = P0 @ H.T @ np.linalg.inv(H @ P0 @ H.T + Rm)
        dx = K @ y
        IKH = np.eye(self.N) - K @ H
        joseph = IKH @ P0 @ IKH.T + K @ Rm @ K.T
        no_reset = 0.5 * (joseph + joseph.T)
        G = np.eye(self.N)
        G[IDX_TH, IDX_TH] = np.eye(3) - 0.5 * skew(dx[IDX_TH])
        with_reset = G @ no_reset @ G.T
        with_reset = 0.5 * (with_reset + with_reset.T)

        kf.update_attitude(R_meas, sigma=sigma)

        scale = max(float(np.abs(with_reset).max()), 1e-12)
        d_with = float(np.abs(kf.P - with_reset).max()) / scale
        d_without = float(np.abs(kf.P - no_reset).max()) / scale
        self.assertGreater(
            float(np.linalg.norm(dx[IDX_TH])), 0.3,
            "构造的修正太小，二阶效应测不出来")
        self.assertGreater(
            d_without, 1e-6,
            f"做不做重置变换没有区分度（差 {d_without:.3e}），测试失去意义")
        self.assertLess(
            d_with, 1e-12,
            f"协方差没有经过误差重置变换（差 {d_with:.3e}）")


class TestGravityLeakage(unittest.TestCase):
    """⭐ 重力泄漏：本模块相对 kalman.py 最本质的新增内容。"""

    def test_one_degree_tilt_leaks_about_017_horizontal(self):
        """闭式判据：姿态误差 δ 产生的水平伪加速度是 g·sin δ。

        1° → 9.81·sin(1°) = 0.1712 m/s²。这个数完全由三角函数决定，
        不涉及任何滤波器实现。
        """
        g = 9.81
        for deg in (0.5, 1.0, 2.0, 5.0):
            rad = np.radians(deg)
            R_err = so3_exp(np.array([rad, 0.0, 0.0]))
            f_true = -np.eye(3).T @ np.array([0.0, 0.0, -g])
            a_wrong = R_err @ f_true + np.array([0.0, 0.0, -g])
            horiz = float(np.linalg.norm(a_wrong[:2]))
            self.assertAlmostEqual(horiz, g * np.sin(rad), places=6)
        self.assertAlmostEqual(9.81 * np.sin(np.radians(1.0)), 0.1712, places=4)

    def test_position_error_from_tilt_grows_quadratically(self):
        """伪加速度是常值 ⇒ 位置误差 ½·a·t²，时间翻倍误差翻四倍。"""
        kf = ErrorStateKalmanFilter(
            acc_noise=0.0, gyro_noise=0.0, acc_bias_walk=0.0,
            gyro_bias_walk=0.0, gravity=G_VEC)
        tilt = np.radians(1.0)
        kf.reset(np.zeros(3), np.zeros(3), so3_exp(np.array([tilt, 0.0, 0.0])))
        f_true = -np.eye(3).T @ G_VEC           # 真姿态是单位阵，滤波器以为倾斜 1°
        errs = {}
        for i in range(int(8.0 / 0.002)):
            kf.predict(f_true, np.zeros(3), 0.002)
            t = round((i + 1) * 0.002, 6)
            if t in (2.0, 4.0):
                errs[t] = float(np.linalg.norm(kf.p[:2]))
        self.assertAlmostEqual(errs[4.0] / errs[2.0], 4.0, delta=0.05)
        # 与 ½·(g·sinδ)·t² 的闭式值对照
        self.assertAlmostEqual(
            errs[2.0], 0.5 * 9.81 * np.sin(tilt) * 4.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
