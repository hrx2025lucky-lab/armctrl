"""末端状态估计：IMU 与运动学的互补融合（线性卡尔曼滤波）。

要估什么，为什么这么选
--------------------
状态向量 9 维：

    x = [ p(3) ; v(3) ; b_a(3) ]
        末端位置  末端速度  加速度计零偏

**姿态不在状态里**，这是有意的，也是本场景与真实足式机器人最本质的差别之一。
固定基座机械臂的末端姿态由关节编码器经正运动学**精确给出**，没有任何不确定性，
把它塞进状态里去"估计"是自欺欺人。真实的浮动基座没有这种奢侈品：
基座姿态只能靠陀螺积分 + 加速度计的重力方向来估，而姿态误差会把重力泄漏进
水平加速度通道，是足式状态估计里最主要的误差源。**本实现完全没有这个问题。**

正因为姿态已知，比力可以直接转到世界系，过程模型对状态是**线性**的，
用标准卡尔曼滤波即可，不需要 EKF。这让实现可以被闭式地验证。

过程模型
--------
由 IMU 比力还原世界系运动学加速度（`+ g` 这一步见 imu.py 的说明）：

    a = R·(f_meas − b_a) + g

于是（R 在一个采样周期内视为常值）

    p⁺ = p + v·dt + ½·a·dt²
    v⁺ = v + a·dt
    b⁺ = b                       零偏是随机游走，均值不变

对状态求偏导得到状态转移矩阵（注意 ∂a/∂b_a = −R）

        ⎡ I   I·dt   −½·R·dt² ⎤
    F = ⎢ 0   I      −R·dt    ⎥
        ⎣ 0   0       I       ⎦

过程噪声有两个来源，必须分开建模：

    加速度计白噪声 → 经 G = [½R dt² ; R dt ; 0] 传播，Q₁ = G·σ_a²·Gᵀ
    零偏随机游走   → Q₂ = σ_bw²·dt·I，只作用在零偏块上

量测模型
--------
关节编码器 + 正运动学给出末端位置，作为位置量测：

    z = p + n,   H = [ I  0  0 ],   R_meas = σ_p²·I

这是本场景里"运动学约束"的类比。真实足式机器人对应的是
「支撑足在世界系静止 ⇒ 由足到基座的运动学链反推基座位置」。

两者的差别必须说清楚（这是类比的边界，不是等价）
--------------------------------------------
1. **可观性完全不同。**
   机械臂末端位置光靠编码器就**完全可观、且无漂移**——正运动学是确定映射。
   浮动基座的水平位置与偏航角**本质不可观**（没有绝对参考），
   再好的滤波器也只能减缓漂移，不能消除。这是最根本的区别。
2. **约束的性质不同。**
   正运动学**恒成立**；而"支撑足静止"是**间歇的、需要判断的**，
   踩空或打滑就失效，接触判据错一次就注入一个大的伪量测。
   本实现用 `kinematic_available` 这个开关**人为**制造间歇性，
   才能演示那个问题——它是模拟出来的，不是平台自带的。
3. **没有重力泄漏。**
   姿态已知 ⇒ 比力到世界系的旋转没有误差。浮动基座的姿态误差 δθ 会产生
   约 g·δθ 的水平伪加速度，1° 的姿态误差就是 0.17 m/s²，两次积分后非常可观。

因此本场景能演示的是：**IMU 高频但会漂 + 运动学低频但准确 ⇒ 互补融合**，
以及噪声、过程噪声、量测噪声、零偏、量测间歇性各自的影响。
**不能**演示浮动基座特有的不可观性与重力泄漏。
"""

from __future__ import annotations

import numpy as np


class TcpStateEstimator:
    """末端位置/速度/加速度计零偏的线性卡尔曼滤波。

    参数
    ----
    acc_noise      加速度计噪声密度 σ_a，m/s²/√Hz
    bias_walk      零偏随机游走强度 σ_bw，m/s²/√s
    meas_noise     运动学位置量测标准差 σ_p，m
    gravity        重力矢量（世界系），默认 (0,0,−9.81)
    """

    N = 9

    def __init__(self, acc_noise: float = 2.0e-3, bias_walk: float = 1.0e-3,
                 meas_noise: float = 1.0e-3,
                 gravity=(0.0, 0.0, -9.81)):
        self.acc_noise = float(acc_noise)
        self.bias_walk = float(bias_walk)
        self.meas_noise = float(meas_noise)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3).copy()
        self.reset(np.zeros(3), np.zeros(3))

    # ------------------------------------------------------------ 状态存取

    def reset(self, p0, v0, b0=None,
              p_std: float = 1e-3, v_std: float = 1e-2,
              b_std: float = 0.1) -> None:
        """重置。b_std 给零偏一个**较大**的初始不确定度，否则滤波器会
        固执地认为零偏就是 0，需要很久才肯改口。"""
        self.x = np.zeros(self.N)
        self.x[0:3] = np.asarray(p0, dtype=float)
        self.x[3:6] = np.asarray(v0, dtype=float)
        if b0 is not None:
            self.x[6:9] = np.asarray(b0, dtype=float)
        self.P = np.diag(np.concatenate([
            np.full(3, p_std ** 2), np.full(3, v_std ** 2), np.full(3, b_std ** 2)]))
        self.innovation = np.zeros(3)
        self.n_update = 0

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def bias(self) -> np.ndarray:
        return self.x[6:9].copy()

    def std(self) -> np.ndarray:
        """各状态分量的标准差 √diag(P)。用于对比"滤波器以为自己多准"
        与"实际有多准"——两者长期严重不符就说明噪声参数设得不对。"""
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))

    # ------------------------------------------------------------ 预测

    def _propagate(self, x, f_body, R, dt: float) -> np.ndarray:
        """状态传播 x⁺ = φ(x)，**纯函数**，不碰 self。

        单独抽出来有两个用处：
          1. `predict` 用它推进状态；
          2. 测试可以对它做**数值雅可比**，与下面手写的 F 逐项对照。
             F 只出现在协方差传播里，状态估计本身不用它，所以 F 写错时
             状态照样跑得好好的，只有协方差是错的——这种错误极难靠观察发现，
             必须用数值雅可比锁死。
        """
        x = np.asarray(x, dtype=float)
        f = np.asarray(f_body, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        a = R @ (f - x[6:9]) + self.gravity          # 关键：+ g 把重力加回去
        out = np.empty_like(x)
        out[0:3] = x[0:3] + x[3:6] * dt + 0.5 * a * dt * dt
        out[3:6] = x[3:6] + a * dt
        out[6:9] = x[6:9]                            # 零偏均值不变（随机游走）
        return out

    @staticmethod
    def transition_matrix(R, dt: float) -> np.ndarray:
        """状态转移矩阵 F = ∂φ/∂x。

            ⎡ I   I·dt   −½·R·dt² ⎤
            ⎢ 0   I      −R·dt    ⎥
            ⎣ 0   0       I       ⎦

        右上两块来自 ∂a/∂b_a = −R：零偏估计偏大，还原出的加速度就偏小。
        """
        R = np.asarray(R, dtype=float).reshape(3, 3)
        I3 = np.eye(3)
        F = np.eye(TcpStateEstimator.N)
        F[0:3, 3:6] = I3 * dt
        F[0:3, 6:9] = -0.5 * R * dt * dt
        F[3:6, 6:9] = -R * dt
        return F

    def predict(self, f_body, R, dt: float) -> None:
        """用 IMU 比力推进一步。

        f_body  IMU 测到的比力（体系）
        R       IMU 体系到世界系的旋转，由编码器正运动学给出，视为已知
        """
        R = np.asarray(R, dtype=float).reshape(3, 3)
        dt = float(dt)

        self.x = self._propagate(self.x, f_body, R, dt)
        F = self.transition_matrix(R, dt)

        # 加速度计白噪声经同一条路径传播到位置与速度
        G = np.zeros((self.N, 3))
        G[0:3, :] = 0.5 * R * dt * dt
        G[3:6, :] = R * dt
        var_a = (self.acc_noise / np.sqrt(dt)) ** 2      # 密度 → 该步的方差
        Q = G @ (var_a * np.eye(3)) @ G.T
        Q[6:9, 6:9] += (self.bias_walk ** 2) * dt * np.eye(3)

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)               # 抑制舍入带来的非对称
        self.F = F

    # ------------------------------------------------------------ 更新

    def update_position(self, p_meas) -> np.ndarray:
        """用运动学给出的末端位置做一次修正，返回新息。"""
        z = np.asarray(p_meas, dtype=float).reshape(3)
        H = np.zeros((3, self.N))
        H[:, 0:3] = np.eye(3)
        Rm = (self.meas_noise ** 2) * np.eye(3)

        y = z - H @ self.x
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        # Joseph 形式而不是 (I−KH)P。后者在数值上不保证对称正定，
        # 长时间运行会让 P 失去正定性，滤波器发散且很难查。
        IKH = np.eye(self.N) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.innovation = y
        self.n_update += 1
        return y
