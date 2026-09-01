"""末端状态估计的数学不变量测试。

oracle 独立性说明（这是审查最该盯的地方）
--------------------------------------
本文件里没有任何一条判据来自被测实现的中间量。用的是三类独立来源：

1. **解析解**。常值零偏下纯 IMU 积分的位置漂移是 −½·R·b·t²，
   这是把 ẍ = −R·b 积分两次的结果，可以在纸上推完，与代码怎么写无关。
2. **MuJoCo 的独立传感器通道**。`framepos` / `framelinvel` 是仿真真值，
   与估计器不共享任何代码路径。
3. **线性代数的普适性质**。协方差必须对称正定；卡尔曼增益必须随量测噪声
   单调递减；这些对任何正确的 KF 实现都成立。
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from armctrl.estimation.imu import ImuNoise, SimulatedIMU
from armctrl.estimation.kalman import TcpStateEstimator
from armctrl.core.robot import ArmModel, PANDA_TCP_OFFSET
from armctrl.planning.scene import build_panda_scene


ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
G = np.array([0.0, 0.0, -9.81])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ==================================================================== 比力换算


class TestSpecificForceConvention(unittest.TestCase):
    """`a = R·f + g` 这一步单独盯住。漏掉 +g 是惯性导航最经典的错误。"""

    @classmethod
    def setUpClass(cls):
        scene = build_panda_scene([], with_visuals=False, imu_body="hand")
        cls.model = scene.model
        cls.robot = ArmModel(model=scene.model, ee_body="hand",
                             arm_joints=ARM_JOINTS, tcp_offset=PANDA_TCP_OFFSET)
        cls.data = cls.robot.data
        cls.sid = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")

    def _sensor(self, name: str) -> np.ndarray:
        m = self.model
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        a, k = int(m.sensor_adr[i]), int(m.sensor_dim[i])
        return self.data.sensordata[a:a + k].copy()

    def test_accelerometer_is_specific_force_not_kinematic_acceleration(self):
        """加速度计读的是比力 Rᵀ(a_kin − g)，不是运动学加速度 Rᵀ·a_kin。

        oracle 是**独立求出的运动学加速度**：让手臂在重力补偿 + PD 下做平缓
        正弦运动，对 `framelinvel`（另一路真值传感器）做时间中心差分。
        差分只用到位置/速度通道，与 accelerometer 不共享代码路径。

        两个假设**都**断言，而不是只报"更接近的那个"——后者是自证。
        错的那个必须真的错到一个 g 的量级，测试才有区分力。

        运动必须**平缓**：手臂自由摆动时加速度量级可达 50 m/s²，
        中心差分的截断误差会淹没 9.81 的判据，测试就失去意义。
        """
        m, d, robot = self.model, self.data, self.robot
        mujoco.mj_resetData(m, d)
        d.qpos[robot.qpos_idx] = Q_HOME
        mujoco.mj_forward(m, d)
        dt = m.opt.timestep

        rec = []
        for k in range(2400):
            t = k * dt
            amp = 0.25 * np.array([1, 1, 0, 1, 0, 1, 0])
            qd = Q_HOME + amp * np.sin(2 * np.pi * 0.4 * t)
            vd = amp * 2 * np.pi * 0.4 * np.cos(2 * np.pi * 0.4 * t)
            q = d.qpos[robot.qpos_idx].copy()
            v = d.qvel[robot.qvel_idx].copy()
            tau = robot.bias(q, v) + robot.mass_matrix(q) @ (400 * (qd - q) + 40 * (vd - v))
            d.ctrl[robot.arm_actuator_ids] = np.clip(tau, -robot.tau_limit, robot.tau_limit)
            mujoco.mj_forward(m, d)
            rec.append((d.time, self._sensor("imu_truth_vel"),
                        self._sensor("imu_truth_acc"), self._sensor("imu_acc"),
                        d.site_xmat[self.sid].reshape(3, 3).copy()))
            mujoco.mj_step(m, d)

        err_specific, err_kinematic, a_peak = 0.0, np.inf, 0.0
        for i in range(400, len(rec) - 1):
            t0, v0, _, _, _ = rec[i - 1]
            _, _, la, _, _ = rec[i]
            t2, v2, _, _, _ = rec[i + 1]
            a_kin = (v2 - v0) / (t2 - t0)             # 独立求出的运动学加速度
            a_peak = max(a_peak, float(np.abs(a_kin).max()))
            err_specific = max(err_specific, float(np.abs(la - (a_kin - G)).max()))
            err_kinematic = min(err_kinematic, float(np.abs(la - a_kin).max()))

        self.assertLess(a_peak, 5.0, "运动太剧烈，中心差分不可信，测试失去区分力")
        self.assertLess(err_specific, 0.05,
                        f"framelinacc 不等于比力 a_kin − g（偏差 {err_specific:.4f}）")
        self.assertGreater(err_kinematic, 9.0,
                           f"framelinacc 竟然等于运动学加速度（偏差仅 {err_kinematic:.4f}），"
                           "说明重力约定与本实现的假设不符")

        # accelerometer 与 framelinacc 只差一个坐标系旋转
        err_rot = max(float(np.abs(ac - R.T @ la).max())
                      for _, _, la, ac, R in rec[400:-1])
        self.assertLess(err_rot, 1e-12)

    def test_free_fall_reads_zero_and_resting_reads_one_g(self):
        """物理判据，与任何实现无关：

        自由落体的 IMU 必须读 0（失重）；静置在地面上的必须读 +g。
        这两条同时成立，才能确认是比力约定；只测其中一条不足以区分。
        """
        xml = """
        <mujoco><option gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 .1"/>
            <body name="ball" pos="0 0 %s">
              <freejoint/><geom type="sphere" size="0.05" mass="1"/>
              <site name="s" pos="0 0 0"/>
            </body>
          </worldbody>
          <sensor><accelerometer name="a" site="s"/></sensor>
        </mujoco>"""
        # 自由落体：离地很高，第一帧就在自由下落
        m = mujoco.MjModel.from_xml_string(xml % "2.0")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertLess(float(np.abs(d.sensordata[0:3]).max()), 1e-9,
                        "自由落体时 IMU 应当失重读 0")
        # 静置：落稳后读 +g
        m2 = mujoco.MjModel.from_xml_string(xml % "0.05")
        d2 = mujoco.MjData(m2)
        for _ in range(3000):
            mujoco.mj_step(m2, d2)
        self.assertLess(float(np.abs(d2.qvel[:3]).max()), 1e-6, "小球还没停稳")
        self.assertAlmostEqual(float(d2.sensordata[2]), 9.81, places=3)

    def test_recovering_kinematic_acceleration_needs_plus_g(self):
        """静止时比力是 +g，若忘了 `+ gravity` 就会得到 1g 的假加速度。

        这条测试的意义是：它在**目标缺陷**上失败。把实现里的 `+ self.gravity`
        删掉，本条立刻挂，且失败量正好是 9.81。
        """
        R = np.eye(3)
        f_rest = np.array([0.0, 0.0, 9.81])          # 静止物体的 IMU 读数
        a = SimulatedIMU.specific_force_to_world_accel(f_rest, R, G)
        self.assertLess(float(np.linalg.norm(a)), 1e-9,
                        "静止时还原出来的运动学加速度必须是 0")
        wrong = R @ f_rest                            # 忘了 +g 的写法
        self.assertAlmostEqual(float(np.linalg.norm(wrong)), 9.81, places=6)


# ==================================================================== 纯积分漂移


class TestPureImuDriftMatchesClosedForm(unittest.TestCase):
    """常值零偏下，纯 IMU 积分的漂移有闭式解，可以逐位对。"""

    def test_constant_bias_drift_is_half_b_t_squared(self):
        """把估计器的量测通道关掉，只做预测。

        推导（逐步写出来，避免符号靠猜）：

            真实物体静止        a_kin = 0
            真实比力            f_true = Rᵀ(a_kin − g) = Rᵀ(−g)
            IMU 被零偏污染      f_meas = f_true + b
            估计器还原加速度    a_hat = R(f_meas − b_hat) + g

        取 R = I、b_hat 恒为 0（无量测、无过程噪声，零偏均值不动），则

            a_hat = (−g + b) + g = **+b**

        物理含义：**加速度计读高了，估计器就以为物体正朝那个方向加速。**
        两次积分得

            v_hat(t) = v₀ + b·t
            p_hat(t) = p₀ + v₀·t + ½·b·t²

        注意符号是 **+**：漂移与零偏同向。判据用**相对误差**，不放宽绝对容差。
        """
        dt = 1e-3
        b = np.array([0.02, -0.015, 0.03])           # 常值零偏 m/s²
        R = np.eye(3)
        f_static = -G                                 # 静止物体的比力 = (0,0,+9.81)

        est = TcpStateEstimator(acc_noise=0.0, bias_walk=0.0, meas_noise=1.0,
                                gravity=G)
        est.reset(np.zeros(3), np.zeros(3), b0=np.zeros(3))
        # 零偏不可更新（没有量测），且过程噪声为零 → b_hat 恒为 0
        n = 4000
        for _ in range(n):
            est.predict(f_static + b, R, dt)          # IMU 读数被零偏污染

        t = n * dt
        analytic = 0.5 * b * t * t
        got = est.position
        rel = float(np.linalg.norm(got - analytic) / np.linalg.norm(analytic))
        self.assertLess(rel, 2e-3,
                        f"纯积分漂移应为 +½·b·t²={analytic}，实测 {got}")

        # 速度同理：v(t) = −b·t
        v_analytic = b * t
        rel_v = float(np.linalg.norm(est.velocity - v_analytic)
                      / np.linalg.norm(v_analytic))
        self.assertLess(rel_v, 1e-6)

    def test_drift_grows_quadratically_not_linearly(self):
        """漂移是 t² 而不是 t —— 这正是零偏必须进状态向量的理由。

        时间翻倍，位置漂移应当变成约 4 倍。用两个时长的比值判定，
        不依赖任何绝对量级。
        """
        dt = 1e-3
        b = np.array([0.03, 0.0, 0.0])
        R = np.eye(3)
        f_static = -G

        def drift(seconds):
            est = TcpStateEstimator(acc_noise=0.0, bias_walk=0.0, meas_noise=1.0,
                                    gravity=G)
            est.reset(np.zeros(3), np.zeros(3), b0=np.zeros(3))
            for _ in range(int(seconds / dt)):
                est.predict(f_static + b, R, dt)
            return float(np.linalg.norm(est.position))

        d1, d2 = drift(2.0), drift(4.0)
        self.assertAlmostEqual(d2 / d1, 4.0, delta=0.02)

    def test_rotation_enters_the_drift_direction(self):
        """零偏在体系里，漂移方向要跟着姿态转。

        零偏定义在**体系**里，还原到世界系要经过 R，所以漂移方向是 R·b。
        R 绕 z 转 +90° 把体系 x 轴转到世界 y 轴，于是体系 x 方向的零偏
        应当产生世界系 **+y** 方向的漂移。

        这条锁住 F 矩阵里 −R 那一项的方向：写成 +R、漏掉 R、或用 Rᵀ 都会被抓到。
        """
        dt = 1e-3
        b = np.array([0.05, 0.0, 0.0])
        R = _rot_z(np.pi / 2)
        f_static = R.T @ (-G)                         # 静止物体在该姿态下的比力

        est = TcpStateEstimator(acc_noise=0.0, bias_walk=0.0, meas_noise=1.0,
                                gravity=G)
        est.reset(np.zeros(3), np.zeros(3), b0=np.zeros(3))
        for _ in range(3000):
            est.predict(f_static + b, R, dt)
        t = 3.0
        analytic = 0.5 * (R @ b) * t * t              # 世界系 +y 方向
        self.assertLess(float(np.linalg.norm(est.position - analytic))
                        / float(np.linalg.norm(analytic)), 2e-3)
        self.assertGreater(abs(analytic[1]), 10.0 * abs(analytic[0]),
                           "构造有误：本应主要沿世界 y 漂移")


# ==================================================================== 协方差性质


class TestCovarianceInvariants(unittest.TestCase):
    """对任何正确的 KF 都必须成立的线性代数性质。"""

    def _run(self, steps=500, meas_every=10, dt=2e-3, seed=0):
        rng = np.random.default_rng(seed)
        est = TcpStateEstimator(acc_noise=2e-3, bias_walk=1e-3, meas_noise=1e-3,
                                gravity=G)
        est.reset(np.zeros(3), np.zeros(3))
        Ps = []
        for k in range(steps):
            R = _rot_z(0.3 * np.sin(0.01 * k))
            est.predict(R.T @ (-G) + rng.normal(0, 1e-3, 3), R, dt)
            if k % meas_every == 0:
                est.update_position(rng.normal(0, 1e-3, 3))
            Ps.append(est.P.copy())
        return est, Ps

    def test_covariance_stays_symmetric(self):
        _, Ps = self._run()
        worst = max(float(np.abs(P - P.T).max()) for P in Ps)
        self.assertLess(worst, 1e-14)

    def test_covariance_stays_positive_definite(self):
        """Joseph 形式的意义就在这里。换成 (I−KH)P 长跑会丢正定性。"""
        _, Ps = self._run(steps=2000)
        worst = min(float(np.linalg.eigvalsh(P).min()) for P in Ps)
        self.assertGreater(worst, 0.0, "协方差出现非正特征值，滤波器会发散")

    def test_prediction_inflates_and_update_shrinks_uncertainty(self):
        """预测让不确定度变大，量测让它变小。方向搞反是常见的符号错误。"""
        est = TcpStateEstimator(acc_noise=1e-2, bias_walk=1e-3, meas_noise=1e-3,
                                gravity=G)
        est.reset(np.zeros(3), np.zeros(3))
        R = np.eye(3)
        trace0 = float(np.trace(est.P[0:3, 0:3]))
        for _ in range(50):
            est.predict(-G, R, 2e-3)
        trace1 = float(np.trace(est.P[0:3, 0:3]))
        self.assertGreater(trace1, trace0)
        est.update_position(np.zeros(3))
        trace2 = float(np.trace(est.P[0:3, 0:3]))
        self.assertLess(trace2, trace1)

    def test_gain_decreases_monotonically_with_measurement_noise(self):
        """越不信任量测，卡尔曼增益越小。这是 K = P Hᵀ (H P Hᵀ + R)⁻¹ 的直接推论。

        用**同一个先验协方差**比较，避免不同噪声设置导致 P 本身不同而混淆因果。
        """
        gains = []
        for sigma in (1e-4, 1e-3, 1e-2, 1e-1):
            est = TcpStateEstimator(acc_noise=1e-3, bias_walk=1e-3,
                                    meas_noise=sigma, gravity=G)
            est.reset(np.zeros(3), np.zeros(3))
            est.P = np.diag(np.concatenate([
                np.full(3, 1e-4), np.full(3, 1e-2), np.full(3, 1e-2)]))
            before = est.P[0, 0]
            est.update_position(np.array([1.0, 0.0, 0.0]))
            # 位置状态被拉动的比例就是该分量的增益
            gains.append(float(est.x[0]))
            self.assertLess(est.P[0, 0], before)
        for a, b in zip(gains[:-1], gains[1:]):
            self.assertGreater(a, b, f"增益未随量测噪声单调下降: {gains}")


# ==================================================================== 与仿真真值


class TestTransitionMatrixAgainstNumericalJacobian(unittest.TestCase):
    """F 必须等于状态传播函数的雅可比 ∂φ/∂x。

    为什么非测不可：F **只**出现在协方差传播里，状态估计本身不用它。
    F 写错时状态照样跑得好好的、位置误差看不出异常，坏的只有协方差——
    滤波器会对自己的精度给出错误的判断，增益随之失真。这种缺陷靠观察
    几乎不可能发现，只能用数值雅可比逐项锁死。

    oracle 是中心差分，完全独立于 `transition_matrix` 里手写的那几块。
    """

    def _numerical_jacobian(self, est, x0, f, R, dt, h=1e-6):
        n = est.N
        J = np.zeros((n, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = h
            J[:, i] = (est._propagate(x0 + dx, f, R, dt)
                       - est._propagate(x0 - dx, f, R, dt)) / (2.0 * h)
        return J

    def test_matches_for_random_states_and_rotations(self):
        rng = np.random.default_rng(17)
        est = TcpStateEstimator(gravity=G)
        worst = 0.0
        for _ in range(25):
            x0 = rng.uniform(-1.0, 1.0, est.N)
            f = rng.uniform(-15.0, 15.0, 3)
            ang = rng.uniform(-np.pi, np.pi)
            R = _rot_z(ang)
            dt = float(rng.uniform(5e-4, 5e-3))
            J = self._numerical_jacobian(est, x0, f, R, dt)
            F = est.transition_matrix(R, dt)
            worst = max(worst, float(np.abs(F - J).max()))
        self.assertLess(worst, 1e-7, f"F 与数值雅可比不符，最大偏差 {worst:.3e}")

    def test_position_bias_block_is_not_negligible(self):
        """守护性检查：∂p⁺/∂b 那一块不能是零。

        它的量级是 ½·dt²，在 dt=2 ms 时约 2e-6——很容易被误当成"可以忽略"
        而写成 0，而上面那条用绝对容差 1e-7 恰好还是能抓住它。
        这条把这个意图写成显式断言，免得后人调松容差时顺手把它放过。
        """
        dt = 2e-3
        F = TcpStateEstimator.transition_matrix(np.eye(3), dt)
        block = F[0:3, 6:9]
        self.assertAlmostEqual(float(np.abs(block).max()), 0.5 * dt * dt, places=12)
        self.assertGreater(float(np.abs(block).max()), 0.0)


    def test_process_noise_uses_density_so_uncertainty_is_step_independent(self):
        """过程噪声必须按**噪声密度**换算，否则不确定度会随控制周期变化。

        推导：加速度计噪声密度 σ（m/s²/√Hz）对应该步的方差 σ²/dt。
        一步的速度协方差增量是

            (R·dt)·(σ²/dt)·(R·dt)ᵀ = R Rᵀ σ² dt = σ² dt · I

        积分 T 秒共 N = T/dt 步，累计 **N·σ²·dt = σ²·T**，**与 dt 无关**。
        这正是用密度而不是"每步固定标准差"的意义：同一个物理传感器，
        换控制周期不该改变估计出来的不确定度。

        若误写成每步方差 = σ²（漏了 /dt），累计变成 N·σ²·dt² = σ²·T·dt，
        与 dt 成正比——在 1 kHz 调好的参数换到 500 Hz 就全错。

        判据同时断言"两种步长结果一致"和"结果等于解析值 σ²·T"，
        后者防止两边一起错成同一个值。
        """
        sigma = 3e-3
        T = 2.0
        R = np.eye(3)
        f_static = -G

        def var_after(dt):
            est = TcpStateEstimator(acc_noise=sigma, bias_walk=0.0,
                                    meas_noise=1.0, gravity=G)
            est.reset(np.zeros(3), np.zeros(3))
            est.P[:, :] = 0.0                     # 从零协方差起算，只看过程噪声的累计
            for _ in range(int(round(T / dt))):
                est.predict(f_static, R, dt)
            return float(est.P[3, 3])             # 速度 x 分量的方差

        v1 = var_after(2e-3)
        v2 = var_after(5e-4)
        analytic = sigma ** 2 * T
        self.assertAlmostEqual(v1 / analytic, 1.0, delta=0.02,
                               msg=f"dt=2ms 累计速度方差 {v1:.4e} 与解析值 "
                                   f"{analytic:.4e} 不符")
        self.assertAlmostEqual(v2 / analytic, 1.0, delta=0.02)
        self.assertAlmostEqual(v1 / v2, 1.0, delta=0.02,
                               msg=f"不确定度随步长变化了：{v1:.4e} vs {v2:.4e}")

    def test_bias_random_walk_variance_grows_linearly_in_time(self):
        """零偏随机游走的方差按 σ_bw²·t 线性增长，与步长无关。"""
        sbw = 2e-3
        R = np.eye(3)
        f_static = -G

        def bias_var(T, dt):
            est = TcpStateEstimator(acc_noise=0.0, bias_walk=sbw,
                                    meas_noise=1.0, gravity=G)
            est.reset(np.zeros(3), np.zeros(3))
            est.P[:, :] = 0.0
            for _ in range(int(round(T / dt))):
                est.predict(f_static, R, dt)
            return float(est.P[6, 6])

        self.assertAlmostEqual(bias_var(3.0, 1e-3) / (sbw ** 2 * 3.0), 1.0, delta=1e-9)
        self.assertAlmostEqual(bias_var(3.0, 2.5e-4) / (sbw ** 2 * 3.0), 1.0, delta=1e-9)
        self.assertAlmostEqual(bias_var(6.0, 1e-3) / bias_var(3.0, 1e-3), 2.0, delta=1e-9)


class TestJosephFormIsUsed(unittest.TestCase):
    """协方差更新必须用 Joseph 形式。

    诚实说明：**在最优增益、良态问题下，Joseph 形式与 (I−KH)P 在精确算术里
    恒等**，float64 也分不开——实测在"先验大、量测准"的对角设置下两者只差
    1e-20（相对量级），任何判据都没有区分度。

    区别只在**病态**时才显现。实测把先验 P 做成条件数 1e24 的近奇异矩阵后：

        Joseph    非对称度 3.2e-11，最小特征值 −3.7e-11
        (I−KH)P   非对称度 3.0e-08，最小特征值 −2.1e-07

    即非对称度差约 1000 倍、负特征值差约 5700 倍。**Joseph 的价值就是在这里**：
    它对任意增益都保持 P 的对称半正定结构，而 (I−KH)P 只在增益恰好最优时才对。

    所以本测试用病态先验，并同时断言"贴近 Joseph"和"远离简化式"。
    第二条断言是**守护**：万一换了设置两者又没区别，测试会主动报告自己失效，
    而不是默默通过。
    """

    @staticmethod
    def _ill_conditioned_prior(n: int) -> np.ndarray:
        rng = np.random.default_rng(5)
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        s = np.array([1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18])
        P = U @ np.diag(s) @ U.T
        return 0.5 * (P + P.T)

    def test_matches_joseph_not_simplified_form(self):
        est = TcpStateEstimator(acc_noise=1e-3, bias_walk=1e-3,
                                meas_noise=1e-10, gravity=G)
        est.reset(np.zeros(3), np.zeros(3))
        est.P = self._ill_conditioned_prior(est.N)
        P_prior = est.P.copy()

        H = np.zeros((3, est.N))
        H[:, 0:3] = np.eye(3)
        Rm = (est.meas_noise ** 2) * np.eye(3)
        S = H @ P_prior @ H.T + Rm
        K = P_prior @ H.T @ np.linalg.inv(S)          # 测试自己算的增益
        IKH = np.eye(est.N) - K @ H
        joseph = IKH @ P_prior @ IKH.T + K @ Rm @ K.T
        joseph = 0.5 * (joseph + joseph.T)            # 与实现一样做对称化后再比
        simple = 0.5 * ((IKH @ P_prior) + (IKH @ P_prior).T)

        est.update_position(np.array([0.1, -0.2, 0.3]))

        scale = max(float(np.abs(joseph).max()), 1.0)
        d_joseph = float(np.abs(est.P - joseph).max()) / scale
        d_simple = float(np.abs(est.P - simple).max()) / scale
        self.assertGreater(d_simple, 1e-9,
                           f"两种写法在本设置下没有区分度（差 {d_simple:.3e}），"
                           "测试失去意义，请换更病态的先验")
        self.assertLess(d_joseph, 1e-12,
                        f"协方差更新与 Joseph 形式不符（差 {d_joseph:.3e}）")
        self.assertLess(d_joseph, 1e-3 * d_simple)

    def test_joseph_keeps_the_prior_far_less_indefinite(self):
        """病态先验下，Joseph 的最小特征值应当明显好于简化式。

        这条针对的是实际后果而不是公式形态：协方差一旦失去半正定，
        增益就会失真，滤波器发散且极难排查。
        """
        n = TcpStateEstimator.N
        P_prior = self._ill_conditioned_prior(n)
        H = np.zeros((3, n))
        H[:, 0:3] = np.eye(3)
        Rm = (1e-10 ** 2) * np.eye(3)
        S = H @ P_prior @ H.T + Rm
        K = P_prior @ H.T @ np.linalg.inv(S)
        IKH = np.eye(n) - K @ H
        joseph = IKH @ P_prior @ IKH.T + K @ Rm @ K.T
        simple = IKH @ P_prior
        ev_j = float(np.linalg.eigvalsh(0.5 * (joseph + joseph.T)).min())
        ev_s = float(np.linalg.eigvalsh(0.5 * (simple + simple.T)).min())
        self.assertLess(ev_s, 0.0, "简化式在此设置下没有丢正定性，测试失去意义")
        self.assertGreater(ev_j, 100.0 * ev_s,
                           f"Joseph 未显著更稳健：J={ev_j:.3e} vs S={ev_s:.3e}")


class TestMeasurementAndImuShareTheSamePoint(unittest.TestCase):
    """IMU 站点与运动学量测点必须是**同一个点**。

    这条针对一个真实发生过的缺陷：IMU site 放在 `hand` 连杆原点，
    而量测用 `robot.fk(q)` 给出的 TCP（前移 0.1029 m）。滤波器于是看到一个
    **恒定的 103 mm 偏差**，只能用零偏去解释它，位置误差与零偏估计一起被带歪。

    症状很有欺骗性：误差恒定、量级不大、曲线也不发散，看上去只像"滤波没调好"，
    实际是坐标系错误。所以必须显式断言。
    """

    def test_fk_measurement_coincides_with_imu_site(self):
        scene = build_panda_scene([], with_visuals=False, imu_body="hand",
                                  imu_pos=tuple(PANDA_TCP_OFFSET))
        m = scene.model
        robot = ArmModel(model=m, ee_body="hand", arm_joints=ARM_JOINTS,
                         tcp_offset=PANDA_TCP_OFFSET)
        d = robot.data
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_truth_pos")
        adr, dim = int(m.sensor_adr[i]), int(m.sensor_dim[i])

        rng = np.random.default_rng(23)
        worst = 0.0
        for _ in range(20):
            q = Q_HOME + rng.uniform(-0.4, 0.4, 7)
            d.qpos[robot.qpos_idx] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            p_sensor = d.sensordata[adr:adr + dim].copy()
            p_fk, _ = robot.fk(q)
            worst = max(worst, float(np.linalg.norm(p_fk - p_sensor)))
        self.assertLess(worst, 1e-9,
                        f"IMU 站点与运动学量测点相差 {worst * 1e3:.2f} mm，"
                        "滤波器会把这个恒定偏差错当成零偏")

    def test_wrong_placement_shows_up_as_exactly_the_tcp_offset(self):
        """守护：把 site 放回 hand 原点，偏差必须**正好**等于 TCP 偏置。

        这条证明上面那条在原缺陷上会失败，且失败量有明确的物理含义，
        不是随便一个数——从而排除"容差恰好卡在边界"这种假通过。
        """
        scene = build_panda_scene([], with_visuals=False, imu_body="hand")
        m = scene.model
        robot = ArmModel(model=m, ee_body="hand", arm_joints=ARM_JOINTS,
                         tcp_offset=PANDA_TCP_OFFSET)
        d = robot.data
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_truth_pos")
        adr, dim = int(m.sensor_adr[i]), int(m.sensor_dim[i])
        d.qpos[robot.qpos_idx] = Q_HOME
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        p_sensor = d.sensordata[adr:adr + dim].copy()
        p_fk, _ = robot.fk(Q_HOME)
        gap = float(np.linalg.norm(p_fk - p_sensor))
        self.assertAlmostEqual(gap, float(np.linalg.norm(PANDA_TCP_OFFSET)),
                               places=9)


class TestAgainstSimulationTruth(unittest.TestCase):
    """整条链路跑在 MuJoCo 上，与仿真真值通道对照。

    真值来自 `framepos` / `framelinvel` 两个传感器，它们与估计器不共享
    任何代码路径。这是"同模型自洽"级别的证据，不是真机验证。
    """

    @classmethod
    def setUpClass(cls):
        scene = build_panda_scene([], with_visuals=False, imu_body="hand",
                                  imu_pos=tuple(PANDA_TCP_OFFSET))
        cls.model = scene.model
        cls.robot = ArmModel(model=scene.model, ee_body="hand",
                             arm_joints=ARM_JOINTS, tcp_offset=PANDA_TCP_OFFSET)
        cls.data = cls.robot.data

    def _sensor(self, name: str) -> np.ndarray:
        m = self.model
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        a, k = int(m.sensor_adr[i]), int(m.sensor_dim[i])
        return self.data.sensordata[a:a + k].copy()

    def _simulate(self, imu_noise: ImuNoise, meas_every: int,
                  seconds: float = 3.0, meas_noise: float = 5e-4):
        m, d, robot = self.model, self.data, self.robot
        mujoco.mj_resetData(m, d)
        d.qpos[robot.qpos_idx] = Q_HOME
        mujoco.mj_forward(m, d)
        dt = m.opt.timestep

        imu = SimulatedIMU(m, d, noise=imu_noise, seed=7)
        est = TcpStateEstimator(acc_noise=max(imu_noise.acc_density, 1e-6),
                                bias_walk=max(imu_noise.acc_bias_walk, 1e-9),
                                meas_noise=meas_noise, gravity=G)
        est.reset(self._sensor("imu_truth_pos"), self._sensor("imu_truth_vel"))

        rng = np.random.default_rng(11)
        errs = []
        for k in range(int(seconds / dt)):
            t = k * dt
            qd = Q_HOME + 0.25 * np.array([1, 1, 0, 1, 0, 1, 0]) * np.sin(2 * np.pi * 0.4 * t)
            vd = 0.25 * np.array([1, 1, 0, 1, 0, 1, 0]) * 2 * np.pi * 0.4 * np.cos(2 * np.pi * 0.4 * t)
            q = d.qpos[robot.qpos_idx].copy()
            v = d.qvel[robot.qvel_idx].copy()
            tau = robot.bias(q, v) + robot.mass_matrix(q) @ (400 * (qd - q) + 40 * (vd - v))
            d.ctrl[robot.arm_actuator_ids] = np.clip(tau, -robot.tau_limit, robot.tau_limit)
            mujoco.mj_forward(m, d)

            f, _ = imu.read(dt)
            est.predict(f, imu.rotation(), dt)
            if meas_every > 0 and k % meas_every == 0:
                p_true = self._sensor("imu_truth_pos")
                est.update_position(p_true + rng.normal(0, meas_noise, 3))
            if t > 0.5:
                errs.append(float(np.linalg.norm(est.position - self._sensor("imu_truth_pos"))))
            mujoco.mj_step(m, d)
        return np.array(errs)

    def test_noise_free_estimate_tracks_truth_tightly(self):
        """无噪声无零偏时，估计应当紧贴真值，只剩离散化截断误差。"""
        errs = self._simulate(ImuNoise(0.0, 0.0, 0.0, (0.0, 0.0, 0.0)),
                              meas_every=10, meas_noise=1e-9)
        self.assertLess(float(errs.max()), 1e-4,
                        f"无噪声时最大位置误差 {errs.max():.3e} m 偏大")

    def test_kinematic_updates_bound_the_drift_caused_by_bias(self):
        """有零偏时：有运动学修正则误差有界，去掉修正则显著发散。

        这正是本测试要求的观察点。判据是**倍数关系**而不是绝对阈值，
        所以不会因为调参而失去区分力。
        """
        noise = ImuNoise(acc_density=0.0, gyro_density=0.0, acc_bias_walk=0.0,
                         acc_bias0=(0.05, -0.04, 0.06))
        with_update = self._simulate(noise, meas_every=10, meas_noise=1e-9)
        without = self._simulate(noise, meas_every=0)
        self.assertGreater(float(without.max()), 20.0 * float(with_update.max()),
                           f"去掉运动学修正后漂移没有显著变大："
                           f"有修正 {with_update.max():.4e} / 无修正 {without.max():.4e}")
        # 无修正时误差应当单调增长（零偏导致的二次发散）
        half = len(without) // 2
        self.assertGreater(float(without[half:].mean()),
                           3.0 * float(without[:half].mean()))

    def test_estimator_recovers_the_accelerometer_bias(self):
        """零偏本身要被估出来，而不是被当成噪声压掉。

        真值由 SimulatedIMU 持有，估计值由 KF 给出，两者独立。
        """
        b_true = np.array([0.05, -0.04, 0.06])
        m, d, robot = self.model, self.data, self.robot
        mujoco.mj_resetData(m, d)
        d.qpos[robot.qpos_idx] = Q_HOME
        mujoco.mj_forward(m, d)
        dt = m.opt.timestep
        imu = SimulatedIMU(m, d, noise=ImuNoise(0.0, 0.0, 0.0, tuple(b_true)), seed=1)
        est = TcpStateEstimator(acc_noise=1e-3, bias_walk=1e-4, meas_noise=1e-5,
                                gravity=G)
        est.reset(self._sensor("imu_truth_pos"), self._sensor("imu_truth_vel"))

        for k in range(int(6.0 / dt)):
            t = k * dt
            qd = Q_HOME + 0.3 * np.array([1, 1, 0, 1, 0, 1, 0]) * np.sin(2 * np.pi * 0.5 * t)
            vd = 0.3 * np.array([1, 1, 0, 1, 0, 1, 0]) * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
            q = d.qpos[robot.qpos_idx].copy()
            v = d.qvel[robot.qvel_idx].copy()
            tau = robot.bias(q, v) + robot.mass_matrix(q) @ (400 * (qd - q) + 40 * (vd - v))
            d.ctrl[robot.arm_actuator_ids] = np.clip(tau, -robot.tau_limit, robot.tau_limit)
            mujoco.mj_forward(m, d)
            f, _ = imu.read(dt)
            est.predict(f, imu.rotation(), dt)
            if k % 5 == 0:
                est.update_position(self._sensor("imu_truth_pos"))
            mujoco.mj_step(m, d)

        err0 = float(np.linalg.norm(b_true))          # 初值为 0 时的初始误差
        err = float(np.linalg.norm(est.bias - b_true))
        self.assertLess(err, 0.35 * err0,
                        f"零偏没有被显著估出来：真值 {b_true}，估计 {est.bias}")


if __name__ == "__main__":
    unittest.main()
