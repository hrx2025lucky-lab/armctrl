"""浮动体姿态与位置的**误差状态卡尔曼滤波**（ESKF）。

这个模块补的是 `kalman.py` 明确绕开的那一块：**姿态解算**。

为什么要单独做一个
------------------
`kalman.py` 的状态是 [p, v, b_a]，**姿态被当成已知精确值**——
固定基座机械臂上这是成立的（关节编码器 + 正运动学给出的末端姿态没有不确定性）。

但真实浮动基座（人形躯干、四足机体）没有这种奢侈品：

* 姿态**只能靠陀螺积分 + 加速度计的重力方向**来估；
* 而姿态误差会把重力**泄漏**进水平加速度通道：
  ``a = R·f + g`` 里 R 错 1°，就凭空多出 ``9.81·sin(1°) ≈ 0.17 m/s²``
  的伪加速度，两次积分十秒就是 8.5 米。

**这是足式状态估计最主要的误差源**，也是 `kalman.py` 完全没有的问题。

为什么是「误差状态」而不是直接估姿态
--------------------------------
姿态住在 SO(3) 这个**流形**上，不是向量空间。直接把旋转矩阵的 9 个数或者
四元数的 4 个数塞进卡尔曼滤波的状态向量会出两个问题：

1. **过参数化** —— 9 个数只有 3 个自由度，协方差矩阵必然奇异；
   四元数 4 个数带一个单位范数约束，线性更新会破坏这个约束。
2. **线性化点不对** —— 大姿态角下线性化误差很大。

ESKF 的办法是把状态劈成两半：

* **名义状态**（nominal）：``p, v, R, b_a, b_g``，**在流形上精确演化**，不做线性化；
* **误差状态**（error）：``δx = [δp, δv, δθ, δb_a, δb_g]``，**永远是小量**，
  住在切空间（普通的 15 维向量空间），卡尔曼滤波只处理它。

两者用**局部扰动**约定连起来（本实现取右乘/体系扰动）：

.. code-block:: text

    p_true = p + δp        v_true = v + δv
    R_true = R · Exp(δθ)   ← 右乘：δθ 是**体坐标系**下的小转角
    b_true = b + δb

每次更新完就把误差**注入**名义状态并清零（这一步叫 injection / reset），
所以误差永远在零点附近，线性化始终有效。

状态与误差状态的分块
------------------
误差状态 15 维，下标固定：

===========  ========  =========================================
 索引         符号      含义
===========  ========  =========================================
 0:3          δp        位置误差（世界系）
 3:6          δv        速度误差（世界系）
 6:9          δθ        **姿态误差**（体系小转角，rad）
 9:12         δb_a      加速度计零偏误差
 12:15        δb_g      **陀螺零偏误差**
===========  ========  =========================================

传播（每个 IMU 周期）
--------------------
名义状态在流形上精确推进::

    a_b = f_meas − b_a                    体系比力，扣掉零偏
    a_w = R·a_b + g                       转到世界系并加回重力
    p⁺  = p + v·dt + ½·a_w·dt²
    v⁺  = v + a_w·dt
    ω̂   = ω_meas − b_g                    体系角速度，扣掉零偏
    R⁺  = R · Exp(ω̂·dt)                   ← 在 SO(3) 上走一步，不是加法

误差状态的转移矩阵 F（推导见 :meth:`transition_matrix` 的 docstring）::

        ⎡ I   I·dt   −½·R·[a_b]ₓ·dt²   −½·R·dt²   0            ⎤
        ⎢ 0   I      −R·[a_b]ₓ·dt      −R·dt      0            ⎥
    F = ⎢ 0   0       Exp(−ω̂·dt)       0         −J_r(ω̂dt)·dt ⎥
        ⎢ 0   0       0                 I         0            ⎥
        ⎣ 0   0       0                 0         I            ⎦

⭐ 注意 ``F[θ,θ] = Exp(−ω̂·dt)``，**不是单位阵**。转得越快，上一时刻的姿态
误差在体坐标系里被「转走」得越多。写成 I 在慢速下看不出区别，快速旋转时协方差就错了。

量测
----
本实现提供三种，可以单独开关，用来对照它们各自能定住什么：

1. :meth:`update_position` —— 运动学位置。``H = [I 0 0 0 0]``。
2. :meth:`update_gravity` —— ⭐ **加速度计水平修正（levelling）**。
   准静态时 ``f ≈ −Rᵀg + b_a``，所以加速度计读数指出了「重力朝哪」，
   由此定住**横滚与俯仰**。
3. :meth:`update_attitude` —— 外部姿态量测（本平台用正运动学的真姿态），
   仅用于对照「有没有外部航向参考」的差别。

⭐ 偏航角：什么时候定不住，什么时候又能定住
--------------------------------------
这一节的结论**被实测修正过一次**，值得原样记下来。

**结构性事实（可以纯代数证明）。**
``update_gravity`` 的量测雅可比是 ``H_θ = −[Rᵀg]ₓ``。
反对称矩阵 ``[u]ₓ`` 的零空间就是 u 自己，所以沿 ``Rᵀg`` 方向的姿态误差
（绕重力轴转，即偏航）**不产生任何量测变化**。

    ⇒ **单靠加速度计的重力方向，永远定不住偏航。**这一条没有例外。

**但整条链路不止这一路量测。**最初我据此断言「只有位置 + 重力修正时可观性
秩最多 14」。**实测把它推翻了**——扫描六种激励条件（4000 步，2 s）：

===============================  ======  ==============
 条件                              秩      偏航标准差
===============================  ======  ==============
 静止　　　　　　＋ 位置量测        11       5.41°
 只转不平移　　　＋ 位置量测        15       2.29°
 只平移不转　　　＋ 位置量测        15       0.07°
 又转又平移　　　＋ 位置量测        15       0.05°
 又转又平移　　　无位置量测          9       0.97°
 只平移不转　　　无位置量测          9       0.76°
===============================  ======  ==============

真正起决定作用的是**世界系绝对位置量测**，不是姿态参考：

* 有位置量测 + 有激励 → 秩 **15**，偏航可观。
  道理是「位置轨迹（世界系）」与「加速度计读数（体系）」必须自洽，
  两者一比对就把**完整**姿态（含偏航）定死了。
  水平平移最有效（偏航标准差压到 0.07°），因为它直接产生水平加速度。
* 没有位置量测 → 秩 **9**，怎么激励都补不满。
* 静止 → 秩 **11**，即使有位置量测也不够：**没有激励就没有信息**
  （又一次持续激励条件，见 adaptive 那一篇）。

⭐ **这恰恰是与真实浮动基座的又一处本质差异。**
本平台的位置量测来自机械臂正运动学，是**世界系绝对位置**；
而足式机器人只能靠支撑足运动学做**相对**位移积分，**没有绝对位置**。
所以真机上偏航**仍然不可观**，必须引入磁力计 / 视觉 / 激光定位。

:meth:`observability_matrix_rank` 与 :meth:`yaw_uncertainty_deg`
把这两件事做成了可以直接读的数。

与真实足式机器人的边界（不要把类比说成等价）
------------------------------------------
本模块**确实**做了姿态估计与重力泄漏，这两点比 `kalman.py` 前进了一大步。
但仍有差别：

* 位置量测这里来自机械臂正运动学，**恒可用、无漂移**；
  足式机器人靠支撑足运动学，**间歇、且水平位置本质不可观**。
* 本模块在仿真里跑，IMU 噪声是我们自己加的高斯白噪声 + 随机游走；
  真实 MEMS 还有温漂、标度因数误差、轴间不正交、振动耦合。
"""

from __future__ import annotations

import numpy as np

from armctrl.core.kinematics import skew, so3_exp, so3_log, so3_right_jacobian

#: 误差状态各分块的下标，避免代码里到处写魔法数字
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_BA = slice(9, 12)
IDX_BG = slice(12, 15)


class ErrorStateKalmanFilter:
    """15 维误差状态卡尔曼滤波：位置、速度、**姿态**、加速度计零偏、**陀螺零偏**。

    参数
    ----
    acc_noise      加速度计噪声密度 σ_a，m/s²/√Hz
    gyro_noise     陀螺噪声密度 σ_g，rad/s/√Hz
    acc_bias_walk  加速度计零偏随机游走 σ_ba，m/s²/√s
    gyro_bias_walk 陀螺零偏随机游走 σ_bg，rad/s/√s
    pos_noise      运动学位置量测标准差 σ_p，m
    lev_noise      加速度计水平修正的**基础**量测标准差 σ_lev，m/s²
    lev_gate       动态门限，见 :meth:`update_gravity`
    gravity        重力矢量（世界系），默认 (0, 0, −9.81)
    """

    N = 15

    def __init__(self, acc_noise: float = 2.0e-3, gyro_noise: float = 1.0e-4,
                 acc_bias_walk: float = 1.0e-3, gyro_bias_walk: float = 1.0e-5,
                 pos_noise: float = 5.0e-4, lev_noise: float = 0.2,
                 lev_gate: float = 0.5, gravity=(0.0, 0.0, -9.81)):
        self.acc_noise = float(acc_noise)
        self.gyro_noise = float(gyro_noise)
        self.acc_bias_walk = float(acc_bias_walk)
        self.gyro_bias_walk = float(gyro_bias_walk)
        self.pos_noise = float(pos_noise)
        self.lev_noise = float(lev_noise)
        self.lev_gate = float(lev_gate)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3).copy()
        self.reset(np.zeros(3), np.zeros(3), np.eye(3))

    # ------------------------------------------------------------ 状态存取

    def reset(self, p0, v0, R0, ba0=None, bg0=None, *,
              p_std: float = 1e-3, v_std: float = 1e-2, th_std: float = 0.05,
              ba_std: float = 0.1, bg_std: float = 0.01) -> None:
        """重置名义状态与协方差。

        零偏的初始标准差要给**大**一些，否则滤波器会固执地认为零偏就是 0，
        要很久才肯改口。姿态同理：th_std 默认 0.05 rad ≈ 2.9°。
        """
        self.p = np.asarray(p0, dtype=float).reshape(3).copy()
        self.v = np.asarray(v0, dtype=float).reshape(3).copy()
        self.R = np.asarray(R0, dtype=float).reshape(3, 3).copy()
        self.ba = (np.zeros(3) if ba0 is None
                   else np.asarray(ba0, dtype=float).reshape(3).copy())
        self.bg = (np.zeros(3) if bg0 is None
                   else np.asarray(bg0, dtype=float).reshape(3).copy())
        self.P = np.diag(np.concatenate([
            np.full(3, p_std ** 2), np.full(3, v_std ** 2),
            np.full(3, th_std ** 2), np.full(3, ba_std ** 2),
            np.full(3, bg_std ** 2)]))
        self.n_pos = 0
        self.n_lev = 0
        self.n_att = 0
        #: 可观性分析用：自本次 reset 起的状态转移积 Φ(k) = F_{k−1}·…·F₀，
        #: 以及每次量测贡献的一块 H·Φ。见 observability_matrix_rank。
        self._Phi = np.eye(self.N)
        self._O_rows: list[np.ndarray] = []

    def nominal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
        """名义状态的一份拷贝 (p, v, R, b_a, b_g)。"""
        return (self.p.copy(), self.v.copy(), self.R.copy(),
                self.ba.copy(), self.bg.copy())

    def std(self) -> np.ndarray:
        """√diag(P)。滤波器**自己以为**的精度，用来和实际误差对照。"""
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))

    @property
    def euler_deg(self) -> np.ndarray:
        """名义姿态的 ZYX 欧拉角（偏航/俯仰/横滚，度）。仅用于显示。"""
        R = self.R
        pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
        if abs(R[2, 0]) < 0.999999:
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
        else:                                   # 万向节死锁附近，横滚并入偏航
            yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
            roll = 0.0
        return np.degrees([yaw, pitch, roll])

    # ------------------------------------------------------------ 流形工具

    @staticmethod
    def retract(state, dx) -> tuple:
        """把误差状态**注入**名义状态：流形上的「加法」。

        ⭐ 姿态那一项是 ``R·Exp(δθ)`` 而不是 ``R + δθ``。
        写成加法得到的矩阵不再是旋转矩阵（不正交），几步之后就发散。
        """
        p, v, R, ba, bg = state
        dx = np.asarray(dx, dtype=float).reshape(ErrorStateKalmanFilter.N)
        return (p + dx[IDX_P], v + dx[IDX_V], R @ so3_exp(dx[IDX_TH]),
                ba + dx[IDX_BA], bg + dx[IDX_BG])

    @staticmethod
    def local_coordinates(state_a, state_b) -> np.ndarray:
        """:meth:`retract` 的逆：求 b 相对 a 的误差状态。

        姿态那一项是 ``Log(Rₐᵀ·R_b)``，与 retract 的右乘约定严格配对。
        测试用它做流形上的数值雅可比。
        """
        pa, va, Ra, baa, bga = state_a
        pb, vb, Rb, bab, bgb = state_b
        dx = np.zeros(ErrorStateKalmanFilter.N)
        dx[IDX_P] = pb - pa
        dx[IDX_V] = vb - va
        dx[IDX_TH] = so3_log(Ra.T @ Rb)
        dx[IDX_BA] = bab - baa
        dx[IDX_BG] = bgb - bga
        return dx

    # ------------------------------------------------------------ 传播

    def propagate_nominal(self, state, f_meas, w_meas, dt: float) -> tuple:
        """名义状态传播，**纯函数**，不碰 self。

        单独抽出来是为了让测试能对它做**流形上的数值雅可比**，
        与手写的 F 逐项对照。F 只出现在协方差传播里，状态估计本身用不到它，
        所以 F 写错时状态照样跑得好好的，只有协方差是错的——
        这种错误极难靠观察发现，必须靠数值雅可比锁死。
        """
        p, v, R, ba, bg = state
        f = np.asarray(f_meas, dtype=float).reshape(3)
        w = np.asarray(w_meas, dtype=float).reshape(3)
        a_b = f - ba
        a_w = R @ a_b + self.gravity
        return (p + v * dt + 0.5 * a_w * dt * dt,
                v + a_w * dt,
                R @ so3_exp((w - bg) * dt),
                ba.copy(), bg.copy())

    def transition_matrix(self, state, f_meas, w_meas, dt: float) -> np.ndarray:
        """误差状态转移矩阵 F = ∂δx⁺/∂δx。

        逐块推导（记 ``a_b = f − b_a``，``ω̂ = ω − b_g``，右乘扰动
        ``R_true = R·Exp(δθ)``）：

        **速度行。** 真实速度的增量用真实量算：

        .. code-block:: text

            v⁺_true = v+δv + [ R·Exp(δθ)·(a_b − δb_a) + g ]·dt
                    ≈ v+δv + [ R(I+[δθ]ₓ)(a_b − δb_a) + g ]·dt
                    ≈ v⁺ + δv + ( R·[δθ]ₓ·a_b − R·δb_a )·dt

        再用 ``[δθ]ₓ·a_b = −[a_b]ₓ·δθ`` 得

        .. code-block:: text

            F[v,θ]  = −R·[a_b]ₓ·dt
            F[v,ba] = −R·dt

        **位置行。** 同一项经 ``½·dt²`` 传下来，所以是速度行乘 ``½·dt``。

        **姿态行。** 由 ``R⁺_true = R_true·Exp((ω̂−δb_g)·dt)`` 出发：

        .. code-block:: text

            R⁺·Exp(δθ⁺) = R·Exp(δθ)·Exp(ω̂·dt − δb_g·dt)
            Exp(δθ⁺)    = Exp(−ω̂dt)·Exp(δθ)·Exp(ω̂dt)·Exp(−J_r(ω̂dt)·δb_g·dt)

        用伴随恒等式 ``R·Exp(u)·Rᵀ = Exp(R·u)`` 把中间三项并起来：

        .. code-block:: text

            F[θ,θ]  = Exp(−ω̂·dt)
            F[θ,bg] = −J_r(ω̂·dt)·dt

        ⭐ ``F[θ,θ]`` 不是单位阵；``F[θ,bg]`` 的 ``J_r`` 也不是可有可无的
        装饰——两处都能被数值雅可比抓到。
        """
        _p, _v, R, ba, bg = state
        f = np.asarray(f_meas, dtype=float).reshape(3)
        w = np.asarray(w_meas, dtype=float).reshape(3)
        a_b = f - ba
        w_hat = (w - bg) * dt

        I3 = np.eye(3)
        F = np.eye(self.N)
        F[IDX_P, IDX_V] = I3 * dt
        F[IDX_P, IDX_TH] = -0.5 * R @ skew(a_b) * dt * dt
        F[IDX_P, IDX_BA] = -0.5 * R * dt * dt
        F[IDX_V, IDX_TH] = -R @ skew(a_b) * dt
        F[IDX_V, IDX_BA] = -R * dt
        F[IDX_TH, IDX_TH] = so3_exp(-w_hat)
        F[IDX_TH, IDX_BG] = -so3_right_jacobian(w_hat) * dt
        return F

    def process_noise(self, state, w_meas, dt: float) -> np.ndarray:
        """过程噪声 Q。四个来源分开建模，走各自的传播路径。

        * 加速度计白噪声 n_a：和 δb_a 走同一条路，但**符号相反**
          （零偏是被减掉的，噪声是被加上的）
        * 陀螺白噪声 n_g：经 ``J_r(ω̂dt)·dt`` 进姿态
        * 两个零偏的随机游走：直接落在各自的块上，方差 ``σ²·dt``

        噪声密度换成单步方差要 ``σ²/dt``（密度的单位带 √Hz）。
        写成「每步固定标准差」会让噪声强度随控制周期变化，是常见错误。
        """
        _p, _v, R, _ba, bg = state
        w = np.asarray(w_meas, dtype=float).reshape(3)
        w_hat = (w - bg) * dt

        G = np.zeros((self.N, 12))
        G[IDX_P, 0:3] = 0.5 * R * dt * dt
        G[IDX_V, 0:3] = R * dt
        G[IDX_TH, 3:6] = so3_right_jacobian(w_hat) * dt
        G[IDX_BA, 6:9] = np.eye(3)
        G[IDX_BG, 9:12] = np.eye(3)

        var = np.concatenate([
            np.full(3, self.acc_noise ** 2 / dt),
            np.full(3, self.gyro_noise ** 2 / dt),
            np.full(3, self.acc_bias_walk ** 2 * dt),
            np.full(3, self.gyro_bias_walk ** 2 * dt)])
        return G @ np.diag(var) @ G.T

    def predict(self, f_meas, w_meas, dt: float) -> None:
        """用一帧 IMU 数据（比力 + 角速度）推进一步。"""
        dt = float(dt)
        state = self.nominal()
        F = self.transition_matrix(state, f_meas, w_meas, dt)
        Q = self.process_noise(state, w_meas, dt)
        self.p, self.v, self.R, self.ba, self.bg = self.propagate_nominal(
            state, f_meas, w_meas, dt)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self._Phi = F @ self._Phi
        self._orthonormalize()

    def _orthonormalize(self) -> None:
        """把名义旋转拉回 SO(3)。

        ``R·Exp(ω dt)`` 在精确算术下保持正交，但浮点乘法每步引入 ~1e-16 的
        偏差，1 kHz 跑十分钟是 6×10⁵ 步，会累积到看得见。
        取极分解的正交因子（SVD 的 U·Vᵀ）代价很低，一劳永逸。
        """
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        if np.linalg.det(self.R) < 0:            # 保证是旋转不是反射
            U[:, -1] *= -1.0
            self.R = U @ Vt

    # ------------------------------------------------------------ 更新

    def _apply(self, y, H, Rm) -> np.ndarray:
        """通用的卡尔曼更新 + 误差注入 + 重置。返回新息 y。

        三步缺一不可：

        1. 解出误差状态 δx = K·y；
        2. **注入**：把 δx 通过 retract 加进名义状态（姿态是右乘 Exp）；
        3. **重置**：误差清零，并且**协方差也要跟着变换**——
           误差的定义点动了，它的协方差自然要跟着转。
           姿态块的重置雅可比是 ``I − ½[δθ]ₓ``。
        """
        H = np.atleast_2d(np.asarray(H, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        self.p, self.v, self.R, self.ba, self.bg = self.retract(self.nominal(), dx)
        self._orthonormalize()

        # Joseph 形式：结构上保证对称半正定，见 kalman.py 的说明
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T

        # 误差重置的协方差变换。dt 很小时接近单位阵，但一次大修正后不可忽略。
        G = np.eye(self.N)
        G[IDX_TH, IDX_TH] = np.eye(3) - 0.5 * skew(dx[IDX_TH])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)
        self._O_rows.append(H @ self._Phi)
        return y

    def update_position(self, p_meas) -> np.ndarray:
        """运动学位置量测。``H = [I 0 0 0 0]``。"""
        z = np.asarray(p_meas, dtype=float).reshape(3)
        H = np.zeros((3, self.N))
        H[:, IDX_P] = np.eye(3)
        self.n_pos += 1
        return self._apply(z - self.p, H, (self.pos_noise ** 2) * np.eye(3))

    def update_gravity(self, f_meas) -> np.ndarray:
        """⭐ 加速度计水平修正（levelling）：靠「重力朝哪」定住横滚与俯仰。

        **原理。**准静态时世界系加速度约为零，于是比力就只剩重力的反作用：

        .. code-block:: text

            f ≈ −Rᵀ·g + b_a

        把 ``R_true = R·Exp(δθ)`` 代进去线性化：

        .. code-block:: text

            h(δ) = −Exp(−δθ)·Rᵀg + b_a + δb_a
                 ≈ −Rᵀg + [δθ]ₓ·Rᵀg + b_a + δb_a
                 = h(0) − [Rᵀg]ₓ·δθ + δb_a

        所以 ``H_θ = −[Rᵀg]ₓ``，``H_ba = I``。

        ⭐ **偏航为什么定不住**：``[u]ₓ`` 的零空间是 u 自己。
        δθ 沿 ``Rᵀg``（绕重力轴，即偏航）时 ``H_θ·δθ = 0``，量测毫无反应。
        这是**结构性不可观**，不是调参能解决的。

        **准静态假设不成立时怎么办。**机器人真的在加速时，比力里混进了运动
        加速度，这一路量测就是错的。这里按 ``|‖f‖ − ‖g‖|`` 这个**与姿态无关**
        的量做门限与噪声膨胀：偏离越大越不信它。

        这个判据很粗糙——自由落体加匀速旋转能骗过它。真机上通常还要结合
        角速度大小、足底力等多个信号。**这是本实现明确的简化。**
        """
        f = np.asarray(f_meas, dtype=float).reshape(3)
        g_norm = float(np.linalg.norm(self.gravity))
        excess = abs(float(np.linalg.norm(f)) - g_norm)
        if excess > self.lev_gate * g_norm:      # 动得太厉害，这一帧不用
            return np.zeros(3)

        h = -self.R.T @ self.gravity + self.ba
        H = np.zeros((3, self.N))
        H[:, IDX_TH] = -skew(self.R.T @ self.gravity)
        H[:, IDX_BA] = np.eye(3)
        # 噪声随动态程度膨胀：静止时信它，加速时不信
        sigma2 = self.lev_noise ** 2 + excess ** 2
        self.n_lev += 1
        return self._apply(f - h, H, sigma2 * np.eye(3))

    def update_attitude(self, R_meas, sigma: float = 1e-3) -> np.ndarray:
        """外部姿态量测（本平台用正运动学给的真姿态）。

        新息取 ``Log(Rᵀ·R_meas)``，与右乘扰动约定配对，一阶下 ``H_θ = I``。

        ⭐ 它的作用是**对照**：打开它等于「有外部航向参考」，
        偏航立刻可观；关掉它就退回「只有 IMU」，偏航必然漂。
        真机上这一路对应视觉里程计 / 磁力计 / 激光定位。
        """
        R_meas = np.asarray(R_meas, dtype=float).reshape(3, 3)
        y = so3_log(self.R.T @ R_meas)
        H = np.zeros((3, self.N))
        H[:, IDX_TH] = np.eye(3)
        self.n_att += 1
        return self._apply(y, H, (sigma ** 2) * np.eye(3))

    # ------------------------------------------------------------ 可观性

    def observability_matrix_rank(self, tol: float = 1e-9) -> int:
        """把「哪些状态其实看不见」变成一个能直接读的整数。

        ⭐ **必须把状态转移带进来。**只把各次量测的雅可比 H 堆起来算秩是不够的：
        速度和陀螺零偏**从来没有被任何一路量测直接观测过**，它们是通过动力学
        与被测量耦合才变得可观的。只堆 H 得到的秩最多是 6，会严重低估。

        标准做法是**离散可观性矩阵**：

        .. code-block:: text

            O = ⎡ H₀·Φ(0)  ⎤       Φ(k) = F_{k−1}·…·F₀   （从上次 reset 到该时刻）
                ⎢ H₁·Φ(1)  ⎥
                ⎣    ⋮      ⎦

        它的数值秩就是可观子空间的维数。

        典型结果（实测，见 tests/test_eskf.py 与模块 docstring 的表）：

        ==============================  ====
         条件                             秩
        ==============================  ====
         静止　　　　　＋ 位置量测         11
         有平移或转动　＋ 位置量测         15
         有平移或转动　无位置量测           9
        ==============================  ====

        两条读法：

        * **没有绝对位置量测，怎么激励都补不满**（9 < 15）——
          真实浮动基座正是这种情况，所以那里偏航不可观。
        * **有位置量测但静止，也补不满**（11 < 15）——持续激励条件。

        ⚠️ 秩是数值判断，靠奇异值相对阈值截断；接近临界时会抖动。
        它用来**说明结构**，不适合当作严格的工程判据。
        """
        if not self._O_rows:
            return 0
        M = np.vstack(self._O_rows)
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > tol * max(float(s[0]), 1.0)))

    def yaw_uncertainty_deg(self) -> float:
        """偏航方向的姿态标准差（度）。

        取姿态协方差块在 ``Rᵀg`` 方向（重力轴，即偏航轴的体系表示）上的投影：

            σ_yaw = √( uᵀ · P_θθ · u ),   u = Rᵀg / ‖Rᵀg‖

        没有外部航向参考时**这个数会一直涨**，是「偏航不可观」最直接的证据。
        """
        u = self.R.T @ self.gravity
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            return float("nan")
        u = u / n
        var = float(u @ self.P[IDX_TH, IDX_TH] @ u)
        return float(np.degrees(np.sqrt(max(var, 0.0))))
