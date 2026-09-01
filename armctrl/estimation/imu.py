"""IMU 模型：把 MuJoCo 给出的理想惯性量弄脏成真实传感器的样子。

MuJoCo 的 accelerometer 是什么，先说清楚
--------------------------------------
这一点决定了整个估计器的正确性，所以是实测确认的，不是查文档抄的：

1. **自由落体的小球**，accelerometer 读 `(0, 0, 0)`；
2. **静置在地面上的小球**，读 `(0, 0, +9.81)`；
3. 让 Panda 在重力补偿 + PD 下做平缓正弦运动（运动学加速度量级 1.05 m/s²），
   用 `framelinvel` 的**时间中心差分**独立求出运动学加速度 `a_kin`，则
   `framelinacc == a_kin − g` 的最大偏差为 **0.0040**（差分截断误差量级），
   而 `framelinacc == a_kin` 的偏差为 **9.8113**（正好一个 g）。
4. `accelerometer == R_siteᵀ · framelinacc`，最大偏差 **1.4e-14**。

结论：**MuJoCo 的 accelerometer 输出的就是真实 IMU 的比力（specific force）**

    f_body = R_bodyᵀ · ( a_kin − g )

于是要从 IMU 还原世界系的运动学加速度，必须**把重力加回去**：

    a_kin = R_body · f_body + g

漏掉这个 `+ g` 是惯性导航里最经典的错误之一：静止时 IMU 读 +9.81，
不加回去就会被当成"末端正在以 1g 向上加速"，位置估计几秒内就飞了。

本模块只负责"弄脏"
----------------
理想值来自 MuJoCo 传感器，本类只叠加两类误差，不做任何坐标或重力换算：

    白噪声      每次采样独立，用噪声密度 σ_n（单位 m/s²/√Hz）描述，
                离散标准差 = σ_n / √dt。数据手册就是按密度给的。
    零偏随机游走 b_{k+1} = b_k + w·√dt，w ~ N(0, σ_bw²)。
                零偏是低频的、缓慢漂移的，滤波器压不掉，只能**估**出来。

为什么零偏必须单独建模
--------------------
白噪声在积分中会互相抵消：**速度**误差按 √t 增长（这才是随机游走），
再积一次进位置则是 **t^(3/2)**——Var[p] = q·t³/3。注意别把 √t 记到位置上。
而零偏是**常值偏移**，两次积分后位置误差按 **½·b·t²** 增长——这是平方发散。
所以任何认真的惯性估计都要把零偏放进状态向量里估计，而不是当噪声处理。
`TcpStateEstimator` 正是这么做的，本模块的 `true_bias` 提供真值用于对照。

（指数经蒙特卡洛拟合验证：速度 t^0.4995、位置 t^1.4987，4000 样本；
 脚本 `verify_A2_imu_error_propagation.py`。）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class ImuNoise:
    """IMU 噪声参数。

    acc_density    加速度计噪声密度，m/s²/√Hz。消费级 MEMS 约 1e-3 量级
    gyro_density   陀螺噪声密度，rad/s/√Hz
    acc_bias_walk  加速度计零偏随机游走强度，m/s²/√s
    acc_bias0      加速度计初始零偏（真值），m/s²。三分量
    gyro_bias_walk 陀螺零偏随机游走强度，rad/s/√s
    gyro_bias0     ⭐ 陀螺初始零偏（真值），rad/s。三分量

    陀螺零偏单独列出来是因为它的后果和加速度计零偏完全不同：
    加速度计零偏经**两次**积分进位置（½·b·t²），陀螺零偏经**一次**积分进姿态
    （θ ≈ b_g·t），而姿态误差又会把重力泄漏成伪加速度、再积两次进位置。
    所以一个很小的陀螺零偏最终对位置的影响是 **t³** 量级。
    0.01 rad/s（约 0.57 °/s，很普通的消费级指标）积 10 秒就是 5.7°，
    对应 0.98 m/s² 的伪加速度。
    """

    acc_density: float = 2.0e-3
    gyro_density: float = 1.0e-4
    acc_bias_walk: float = 1.0e-3
    acc_bias0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_walk: float = 1.0e-5
    gyro_bias0: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SimulatedIMU:
    """挂在某个 site 上的模拟 IMU。

    需要模型里已经有名为 `acc_sensor` / `gyro_sensor` 的传感器，
    并且它们绑定在同一个 site 上（见 `armctrl.planning.scene.add_imu`）。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 site: str = "imu_site",
                 acc_sensor: str = "imu_acc", gyro_sensor: str = "imu_gyro",
                 noise: ImuNoise | None = None, seed: int = 0):
        self.model = model
        self.data = data
        self.noise = noise or ImuNoise()
        self.rng = np.random.default_rng(seed)

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site 不存在: {site}")
        self._acc = self._sensor_slice(acc_sensor, 3)
        self._gyro = self._sensor_slice(gyro_sensor, 3)
        self.reset()

    def _sensor_slice(self, name: str, dim: int) -> slice:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"传感器不存在: {name}")
        if int(self.model.sensor_dim[sid]) != dim:
            raise ValueError(f"传感器 {name} 维度应为 {dim}")
        adr = int(self.model.sensor_adr[sid])
        return slice(adr, adr + dim)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.true_bias = np.asarray(self.noise.acc_bias0, dtype=float).copy()
        self.true_gyro_bias = np.asarray(self.noise.gyro_bias0, dtype=float).copy()

    # ------------------------------------------------------------ 读数

    def rotation(self) -> np.ndarray:
        """IMU 所在 site 的姿态 R（世界 ← 体）。"""
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def ideal(self) -> tuple[np.ndarray, np.ndarray]:
        """无噪声无零偏的理想读数 (比力_体系, 角速度_体系)。"""
        return (self.data.sensordata[self._acc].copy(),
                self.data.sensordata[self._gyro].copy())

    def read(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """带噪声与零偏的一次采样。

        必须传 dt：噪声密度换成离散标准差要除以 √dt，采样越快单次噪声越大，
        但积分同样步数覆盖的时间也越短，两者的净效果才是物理上正确的。
        写成"每步固定标准差"会让噪声强度随控制周期变化，是常见的错误。
        """
        f_ideal, w_ideal = self.ideal()
        n = self.noise
        sa = n.acc_density / np.sqrt(dt)
        sg = n.gyro_density / np.sqrt(dt)
        f = f_ideal + self.true_bias + self.rng.normal(0.0, sa, 3)
        w = w_ideal + self.true_gyro_bias + self.rng.normal(0.0, sg, 3)
        # 零偏随机游走要放在读数**之后**推进：本次读数用的是本次采样时刻的零偏
        self.true_bias = self.true_bias + self.rng.normal(
            0.0, n.acc_bias_walk * np.sqrt(dt), 3)
        self.true_gyro_bias = self.true_gyro_bias + self.rng.normal(
            0.0, n.gyro_bias_walk * np.sqrt(dt), 3)
        return f, w

    # ------------------------------------------------------------ 工具

    @staticmethod
    def specific_force_to_world_accel(f_body: np.ndarray, R: np.ndarray,
                                      gravity: np.ndarray) -> np.ndarray:
        """比力 → 世界系运动学加速度：a = R·f + g。

        单独抽成一个函数是为了让这条换算能被测试**单独**盯住。
        它是整条链路里最容易搞错、错了又不容易看出来的一步。
        """
        return R @ np.asarray(f_body, dtype=float) + np.asarray(gravity, dtype=float)
