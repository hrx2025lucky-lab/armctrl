"""轨迹规划：多项式插值、S 曲线、笛卡尔插补。

关节空间
    五次多项式   位置/速度/加速度边界条件全部可指定，加速度连续，
                 但 jerk 不连续（阶跃），高速时激励结构振动。
    S 曲线       七段式，jerk 有界且分段常值，速度/加速度/jerk 三层限幅。
                 工业机械臂的标准做法，正是为了抑制振动。

笛卡尔空间
    位置         直线 / 圆弧插补
    姿态         四元数 slerp（球面线性插值），保证角速度恒定且不经过
                 欧拉角奇异，是姿态插补的正确做法。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 关节空间


def quintic(q0, qf, T, t, qd0=0.0, qdf=0.0, qdd0=0.0, qddf=0.0):
    """五次多项式插值，返回 (q, qd, qdd)。

    边界条件 6 个 → 6 个系数唯一确定：
        q(0)=q0, q̇(0)=qd0, q̈(0)=qdd0
        q(T)=qf, q̇(T)=qdf, q̈(T)=qddf
    """
    t = np.clip(np.atleast_1d(t), 0.0, T)
    a0, a1, a2 = q0, qd0, qdd0 / 2.0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    A = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
    b = np.array([qf - a0 - a1 * T - a2 * T2, qdf - a1 - 2 * a2 * T, qddf - 2 * a2])
    a3, a4, a5 = np.linalg.solve(A, b)

    q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    qd = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    qdd = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
    return q, qd, qdd


class SCurveProfile:
    """七段式 S 曲线（jerk 有界）速度规划，静止到静止。

    七段依次为：加加速 / 匀加速 / 减加速 / 匀速 / 加减速 / 匀减速 / 减减速。
    与五次多项式的关键差别：jerk 分段常值且有界，因此不会激励高频结构模态；
    五次多项式的 jerk 在起终点是阶跃。
    """

    def __init__(self, distance: float, v_max: float, a_max: float, j_max: float):
        self.d = abs(float(distance))
        self.sign = 1.0 if distance >= 0 else -1.0
        self.v_max, self.a_max, self.j_max = v_max, a_max, j_max
        self._plan()

    def _plan(self) -> None:
        v, a, j, d = self.v_max, self.a_max, self.j_max, self.d
        if d < 1e-12:
            self.Tj = self.Ta = self.Tv = self.T = 0.0
            self.v_peak = self.a_peak = 0.0
            return

        # 能否达到 a_max：加速段需要存在匀加速段，即 Ta >= 2Tj
        # 由 v_peak = j·Tj·(Ta − Tj) 且 Tj = a/j 得 Ta = Tj + v/a，
        # 故 Ta >= 2Tj  <=>  v/a >= a/j  <=>  v·j >= a²
        if v * j >= a * a:
            Tj = a / j                      # 加加速段时长
            Ta = Tj + v / a                 # 总加速段时长（含两段 jerk + 匀加速段）
        else:
            Tj = np.sqrt(v / j)
            Ta = 2.0 * Tj
        d_acc = v * Ta                      # 加速+减速段总位移（= 2·s_acc）

        if d >= d_acc:
            Tv = (d - d_acc) / v            # 匀速段
            v_peak = v
        else:
            # 三角形工况：达不到 v_max，需重新求解峰值速度
            # 加速段位移 s_acc = a_pk·Ta·(Ta − Tj)/2，无匀速时 2·s_acc = d，
            # 即 a·Ta² − a·Tj·Ta − d = 0，取正根
            Tv = 0.0
            if d * j * j >= 2.0 * a**3:     # 三角形工况下仍能达到 a_max
                Tj = a / j
                Ta = 0.5 * (Tj + np.sqrt(Tj * Tj + 4.0 * d / a))
                v_peak = a * (Ta - Tj)
            else:
                Tj = (d / (2.0 * j)) ** (1.0 / 3.0)
                Ta = 2.0 * Tj
                v_peak = j * Tj * Tj
        self.Tj, self.Ta, self.Tv = Tj, Ta, Tv
        self.T = 2.0 * Ta + Tv
        self.v_peak = v_peak
        self.a_peak = j * Tj

    def __call__(self, t: float):
        """返回 (s, sd, sdd) —— 位移、速度、加速度。"""
        j, Tj, Ta, Tv, T = self.j_max, self.Tj, self.Ta, self.Tv, self.T
        if T <= 0.0:
            return 0.0, 0.0, 0.0
        t = float(np.clip(t, 0.0, T))
        a_pk, v_pk = self.a_peak, self.v_peak

        def acc_phase(tt):
            """加速段 [0, Ta] 内的 (s, v, a)。"""
            if tt < Tj:
                return j * tt**3 / 6.0, 0.5 * j * tt**2, j * tt
            if tt < Ta - Tj:
                dt = tt - Tj
                s0 = j * Tj**3 / 6.0
                v0 = 0.5 * j * Tj**2
                return s0 + v0 * dt + 0.5 * a_pk * dt**2, v0 + a_pk * dt, a_pk
            dt = tt - (Ta - Tj)
            tc = Ta - 2.0 * Tj
            s0 = j * Tj**3 / 6.0 + (0.5 * j * Tj**2) * tc + 0.5 * a_pk * tc**2
            v0 = 0.5 * j * Tj**2 + a_pk * tc
            return (
                s0 + v0 * dt + 0.5 * a_pk * dt**2 - j * dt**3 / 6.0,
                v0 + a_pk * dt - 0.5 * j * dt**2,
                a_pk - j * dt,
            )

        s_acc, _, _ = acc_phase(Ta)
        if t <= Ta:
            s, v, a = acc_phase(t)
        elif t <= Ta + Tv:
            s, v, a = s_acc + v_pk * (t - Ta), v_pk, 0.0
        else:
            # 减速段用加速段的时间镜像
            tt = T - t
            s_m, v_m, a_m = acc_phase(tt)
            s, v, a = self.d - s_m, v_m, -a_m
        return self.sign * s, self.sign * v, self.sign * a


# ---------------------------------------------------------------- 笛卡尔空间


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 单位四元数 (w,x,y,z)。"""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array(
            [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
        )
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    if i == 1:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array(
        [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """四元数球面线性插值。s ∈ [0,1]，角速度恒定。"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:                     # 取短程弧
        q1, dot = -q1, -dot
    if dot > 0.9995:                  # near-parallel 退化为线性插值
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)


class CartesianLine:
    """笛卡尔直线 + 姿态 slerp，时间律由 S 曲线提供。"""

    def __init__(self, p0, R0, p1, R1, v_max=0.3, a_max=1.0, j_max=8.0):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.q0, self.q1 = quat_from_matrix(R0), quat_from_matrix(R1)
        self.L = float(np.linalg.norm(self.p1 - self.p0))
        self.profile = SCurveProfile(max(self.L, 1e-9), v_max, a_max, j_max)
        self.T = self.profile.T

    def __call__(self, t: float):
        s, sd, _ = self.profile(t)
        u = s / max(self.L, 1e-12)
        direction = (self.p1 - self.p0) / max(self.L, 1e-12)
        p = self.p0 + direction * s
        v = direction * sd
        R = quat_to_matrix(slerp(self.q0, self.q1, float(np.clip(u, 0.0, 1.0))))
        return p, R, v


class CartesianArc:
    """三点圆弧插补：起点 → 中间点 → 终点，姿态用 slerp。"""

    def __init__(self, p0, p_mid, p1, R0, R1, v_max=0.3, a_max=1.0, j_max=8.0):
        p0, p_mid, p1 = map(lambda x: np.asarray(x, float), (p0, p_mid, p1))
        # 由三点求圆心：两条弦的中垂面与所在平面的交
        v1, v2 = p_mid - p0, p1 - p0
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            raise ValueError("三点共线，无法确定圆弧")
        n = n / n_norm
        A = np.vstack([v1, v2, n])
        b = np.array([v1 @ (p0 + p_mid) / 2.0, v2 @ (p0 + p1) / 2.0, n @ p0])
        self.center = np.linalg.solve(A, b)
        self.radius = float(np.linalg.norm(p0 - self.center))
        self.n = n

        e1 = (p0 - self.center) / self.radius
        e2 = np.cross(n, e1)
        self.e1, self.e2 = e1, e2
        ang = np.arctan2((p1 - self.center) @ e2, (p1 - self.center) @ e1)
        ang_mid = np.arctan2((p_mid - self.center) @ e2, (p_mid - self.center) @ e1)
        if ang < 0:
            ang += 2 * np.pi
        if ang_mid < 0:
            ang_mid += 2 * np.pi
        self.total_angle = ang if ang_mid <= ang else ang - 2 * np.pi

        arc_len = abs(self.total_angle) * self.radius
        self.profile = SCurveProfile(arc_len, v_max, a_max, j_max)
        self.T = self.profile.T
        self.q0, self.q1 = quat_from_matrix(R0), quat_from_matrix(R1)
        self.arc_len = arc_len

    def __call__(self, t: float):
        s, sd, _ = self.profile(t)
        u = s / max(self.arc_len, 1e-12)
        ang = self.total_angle * u
        p = self.center + self.radius * (np.cos(ang) * self.e1 + np.sin(ang) * self.e2)
        tangent = self.radius * (-np.sin(ang) * self.e1 + np.cos(ang) * self.e2)
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        R = quat_to_matrix(slerp(self.q0, self.q1, float(np.clip(u, 0.0, 1.0))))
        return p, R, tangent * sd
