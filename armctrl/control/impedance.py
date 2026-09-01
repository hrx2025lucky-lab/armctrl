"""柔顺控制：笛卡尔阻抗、导纳、混合力位。

阻抗控制（力矩级）
    τ = Jᵀ[ K(x_d − x) + D(ẋ_d − ẋ) ] + g(q) + C(q,q̇)q̇ + τ_null

    机器人表现为一个可编程的质量-弹簧-阻尼系统。不需要力传感器，
    但要求关节可力矩控制。重力补偿必须准确，否则稳态存在下垂偏差。

    阻尼按临界阻尼设计：D = 2ζ √(Λ K)，Λ 为任务空间惯量 (J M⁻¹ Jᵀ)⁻¹。
    这样刚度变化时阻尼自动跟随，避免欠阻尼振荡或过阻尼迟滞。

导纳控制（位置级）
    ẍ = M_d⁻¹ [ F_ext − D_d ẋ − K_d (x − x_0) ]

    外力经虚拟导纳模型积分成参考位移，再交给内层高刚度位置环跟踪。
    需要力传感器，适合本身刚性很高、无法力矩控制的工业臂。

混合力位控制（Raibert–Craig）
    用选择矩阵 S 把任务空间分成互斥的两组方向：
        S      方向做位置控制
        I − S  方向做力控制
    同一方向不能同时控力和控位（环境约束已经决定了其中一个）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armctrl.core.robot import ArmModel
from armctrl.core.kinematics import rot_to_axis_angle_error, OrientationErrorTracker


@dataclass
class ImpedanceGains:
    """六维笛卡尔刚度/阻尼。前 3 维平移 (N/m)，后 3 维旋转 (Nm/rad)。"""

    k_trans: float = 400.0
    k_rot: float = 30.0
    zeta: float = 1.0            # 阻尼比，1.0 为临界阻尼
    k_null: float = 5.0          # 零空间刚度
    d_null: float = 2.0

    def stiffness(self) -> np.ndarray:
        return np.diag([self.k_trans] * 3 + [self.k_rot] * 3)


class CartesianImpedanceController:
    """笛卡尔阻抗控制器。

    阻尼有两种设计方式：
        damping_design="critical"   D = 2ζ√(ΛK)，随构型自适应（推荐）
        damping_design="fixed"      D = 2ζ√K，忽略惯量，简单但构型相关性差
    """

    def __init__(
        self,
        robot: ArmModel,
        gains: ImpedanceGains | None = None,
        damping_design: str = "critical",
        q_null: np.ndarray | None = None,
        track_orientation: bool = True,
        ori_tracker: OrientationErrorTracker | None = None,
        dt: float | None = None,
    ):
        self.robot = robot
        self.gains = gains or ImpedanceGains()
        self.damping_design = damping_design
        self.q_null = q_null
        # 姿态误差的 SO(3) 连续 lift。θ≈π 时主值 log 两值，无状态调用会产生
        # 2π 跳变，实测导致限幅后力矩差达 194 N·m（符号整体翻转）。
        # 缺省 tracker **只做分支连续性**：不限速、不限幅，因此输出始终满足
        # exp(skew(e_o)) = R_des·R_curᵀ，不引入任何跟踪滞后。
        # 需要限速时请自己构造 OrientationErrorTracker(max_rate=..., dt=真实dt) 传入。
        if ori_tracker is not None:
            self.ori_tracker = ori_tracker
        elif track_orientation:
            self.ori_tracker = OrientationErrorTracker(dt=dt)
        else:
            self.ori_tracker = None
        self.last_error = np.zeros(6)
        self.last_wrench = np.zeros(6)
        self.last_tau_raw = np.zeros(robot.n)

    def reset(self) -> None:
        """episode 切换时**必须**调用：清空姿态 lift 的分支历史。"""
        if self.ori_tracker is not None:
            self.ori_tracker.reset()

    def _orientation_error(self, R_cur, R_des, dt=None) -> np.ndarray:
        if self.ori_tracker is None:
            return rot_to_axis_angle_error(R_cur, R_des)
        return self.ori_tracker.update(R_cur, R_des, dt=dt)

    def _damping(self, K: np.ndarray, q: np.ndarray) -> np.ndarray:
        if self.damping_design == "fixed":
            return 2.0 * self.gains.zeta * np.sqrt(K)
        # D = 2ζ Λ^{1/2} K^{1/2}，用对称平方根保证正定
        Lam = self.robot.task_space_inertia(q)
        Lam = 0.5 * (Lam + Lam.T)
        w, V = np.linalg.eigh(Lam)
        Lam_sqrt = V @ np.diag(np.sqrt(np.maximum(w, 1e-9))) @ V.T
        return 2.0 * self.gains.zeta * Lam_sqrt @ np.sqrt(K)

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        p_des: np.ndarray,
        R_des: np.ndarray,
        v_des: np.ndarray | None = None,
        f_ff: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """返回**饱和后**的关节力矩 τ。f_ff 为任务空间前馈力/力矩（6 维）。

        诊断用的未饱和量保存在 self.last_wrench / self.last_tau_raw —— 只检查
        饱和后的力矩会掩盖控制律本身的跳变。
        dt 仅在 tracker 开启限速时使用。
        """
        p, R = self.robot.fk(q)
        J = self.robot.jacobian(q)
        x_dot = J @ qd

        e = np.concatenate([p_des - p, self._orientation_error(R, R_des, dt=dt)])
        e_dot = (np.zeros(6) if v_des is None else v_des) - x_dot

        K = self.gains.stiffness()
        D = self._damping(K, q)
        wrench = K @ e + D @ e_dot
        if f_ff is not None:
            wrench = wrench + f_ff

        tau = J.T @ wrench
        # 重力与科氏补偿：让闭环动力学只剩期望的阻抗特性
        tau = tau + self.robot.bias(q, qd)

        if self.q_null is not None:
            tau = tau + self._null_space_torque(q, qd, J)
        self.last_error = e.copy()
        self.last_wrench = wrench.copy()
        self.last_tau_raw = tau.copy()
        return self.robot.saturate_torque(tau)

    def _null_space_torque(self, q, qd, J) -> np.ndarray:
        """零空间 PD，把冗余自由度拉向 q_null，不影响末端。"""
        M = self.robot.mass_matrix(q)
        Minv = np.linalg.inv(M)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-9 * np.eye(6))
        J_bar = Minv @ J.T @ Lam                      # 动力学一致伪逆
        N = np.eye(self.robot.n) - J.T @ J_bar.T      # 零空间投影
        tau_null = self.gains.k_null * (self.q_null - q) - self.gains.d_null * qd
        return N @ tau_null


class AdmittanceController:
    """导纳控制：外力 → 参考轨迹修正量。

    虚拟模型 M_d ẍ + D_d ẋ + K_d (x − x_0) = F_ext
    输出修正后的位置指令，交给外部的高刚度位置环。
    """

    def __init__(self, m_d=2.0, k_d=200.0, zeta=1.0, dt=0.002):
        self.m_d = float(m_d)
        self.k_d = float(k_d)
        self.d_d = 2.0 * zeta * np.sqrt(m_d * k_d)
        self.dt = float(dt)
        self.dx = np.zeros(3)
        self.dv = np.zeros(3)

    def reset(self) -> None:
        self.dx[:] = 0.0
        self.dv[:] = 0.0

    def step(self, f_ext: np.ndarray) -> np.ndarray:
        """输入外力（3 维），返回相对标称轨迹的位置偏移。"""
        acc = (f_ext - self.d_d * self.dv - self.k_d * self.dx) / self.m_d
        self.dv = self.dv + acc * self.dt          # 半隐式欧拉，比显式稳定
        self.dx = self.dx + self.dv * self.dt
        return self.dx.copy()


class HybridForcePositionController:
    """混合力位控制（Raibert–Craig 选择矩阵）。

    selection 为 6 维 0/1 向量：1 = 该方向做位置控制，0 = 做力控制。
    典型接触任务（沿水平面滑动）：位置控 x,y 与全部姿态，力控 z。
        selection = [1, 1, 0, 1, 1, 1]
    """

    def __init__(
        self,
        robot: ArmModel,
        selection: np.ndarray,
        kp_pos: float = 600.0,
        kd_pos: float = 40.0,
        kp_force: float = 0.6,
        ki_force: float = 4.0,
        dt: float = 0.002,
        track_orientation: bool = True,
        ori_tracker: OrientationErrorTracker | None = None,
    ):
        self.robot = robot
        self.S = np.diag(np.asarray(selection, float))
        self.S_bar = np.eye(6) - self.S
        self.kp_pos, self.kd_pos = kp_pos, kd_pos
        self.kp_force, self.ki_force = kp_force, ki_force
        self.dt = dt
        self.f_int = np.zeros(6)
        if ori_tracker is not None:
            self.ori_tracker = ori_tracker
        elif track_orientation:
            self.ori_tracker = OrientationErrorTracker(dt=dt)
        else:
            self.ori_tracker = None
        self.last_error = np.zeros(6)
        self.last_wrench = np.zeros(6)
        self.last_tau_raw = np.zeros(robot.n)

    def reset(self) -> None:
        """episode 切换时**必须**调用：清空力积分与姿态 lift 的分支历史。"""
        self.f_int[:] = 0.0
        if self.ori_tracker is not None:
            self.ori_tracker.reset()

    def _orientation_error(self, R_cur, R_des, dt=None) -> np.ndarray:
        if self.ori_tracker is None:
            return rot_to_axis_angle_error(R_cur, R_des)
        return self.ori_tracker.update(R_cur, R_des, dt=dt)

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        p_des: np.ndarray,
        R_des: np.ndarray,
        f_des: np.ndarray,
        f_meas: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        p, R = self.robot.fk(q)
        J = self.robot.jacobian(q)
        x_dot = J @ qd

        e = np.concatenate([p_des - p, self._orientation_error(R, R_des, dt=dt)])
        w_pos = self.kp_pos * e - self.kd_pos * x_dot

        # 力控方向用 PI：比例项快速响应，积分项消除接触刚度带来的稳态误差
        e_f = f_des - f_meas
        self.f_int += e_f * self.dt
        self.f_int = np.clip(self.f_int, -50.0, 50.0)
        w_force = f_des + self.kp_force * e_f + self.ki_force * self.f_int

        wrench = self.S @ w_pos + self.S_bar @ w_force
        tau = J.T @ wrench + self.robot.bias(q, qd)
        self.last_error = e.copy()
        self.last_wrench = wrench.copy()
        self.last_tau_raw = tau.copy()
        return self.robot.saturate_torque(tau)
