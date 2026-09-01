"""动力学回归矩阵：τ = Y(q, q̇, q̈) · π

刚体动力学对惯性参数是线性的：

    M(q)q̈ + C(q,q̇)q̇ + g(q) = Y(q, q̇, q̈) · π

其中 π 为每个连杆的 10 个标准惯性参数
    [m, m·cx, m·cy, m·cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
再加上每个关节的转子惯量（armature），共 11n 个参数。

这个 Y 同时服务三件事：
    参数辨识    π̂ = argmin ‖τ − Yπ‖²
    自适应控制  τ = Y_SL π̂ − K r,  π̂̇ = Γ Y_SLᵀ r
    模型前馈    用辨识后的 π̂ 喂计算力矩控制

Slotine-Li 回归矩阵
-------------------
自适应控制需要的不是 Y(q,q̇,q̈)，而是

    Y_SL · π = M(q) q̈ᵣ + C(q,q̇) q̇ᵣ + g(q)

注意科氏项是**实测速度 q̇ 与参考速度 q̇ᵣ 的混合**，无法直接调用标准回归矩阵。
利用 Christoffel 形式下 C(q,a)b = C(q,b)a 的对称性，以及极化恒等式

    C(q,q̇)q̇ᵣ = ½[ C(q,q̇+q̇ᵣ)(q̇+q̇ᵣ) − C(q,q̇)q̇ − C(q,q̇ᵣ)q̇ᵣ ]

记 A = Y(q,0,q̈ᵣ)、B₊ = Y(q,q̇+q̇ᵣ,0)、B₁ = Y(q,q̇,0)、B₂ = Y(q,q̇ᵣ,0)、G = Y(q,0,0)，
可得只需 5 次标准回归矩阵调用的精确构造

    Y_SL = A + ½(B₊ − B₁ − B₂) + ½G

使用参考速度 q̇ᵣ = q̇d + Λe 而非实测 q̈ 是 Slotine-Li 的关键：
避免了对含噪声的关节加速度求导。
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import mujoco


class DynamicsRegressor:
    """基于 Pinocchio 的动力学回归矩阵。

    参数向量 π 分四段：
        连杆惯性参数 10n   [m, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        转子惯量     n     armature，对 q̈ 线性
        粘性摩擦     n     Fv，对 q̇ 线性
        库仑摩擦     n     Fc，对 sign(q̇) 线性

    摩擦项必须进入参数向量：真实机器人（以及 MuJoCo 模型中的 dof_damping /
    dof_frictionloss）都存在关节摩擦，若模型只有刚体动力学，"精确模型"的
    计算力矩控制仍会残留可观的跟踪误差。这也是自适应控制的价值所在——
    摩擦难以先验建模，但可以在线估计。
    """

    def __init__(self, xml_path: str, n_arm: int = 7):
        self.model = pin.buildModelFromMJCF(xml_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.n_arm = n_arm
        self.n_link_par = 10 * (self.model.njoints - 1)
        self.n_par = self.n_link_par + 3 * self.nv

        # 从 MuJoCo 模型读取真实摩擦，作为辨识真值
        mj = mujoco.MjModel.from_xml_path(xml_path)
        self.fv_true = np.array(mj.dof_damping[: self.nv]).copy()
        self.fc_true = np.array(mj.dof_frictionloss[: self.nv]).copy()

    # ---------------- 参数向量 ----------------

    def true_parameters(self) -> np.ndarray:
        """从模型读出真值 π，用于辨识精度评估。"""
        link = np.concatenate(
            [self.model.inertias[i].toDynamicParameters() for i in range(1, self.model.njoints)]
        )
        return np.concatenate(
            [link, np.asarray(self.model.armature).copy(), self.fv_true, self.fc_true]
        )

    # ---------------- 标准回归矩阵 ----------------

    def _base_regressor(self, q, v, a) -> np.ndarray:
        """Y_link(q,v,a)，不含 armature 部分。"""
        return pin.computeJointTorqueRegressor(self.model, self.data, q, v, a).copy()

    def regressor(self, q, v, a) -> np.ndarray:
        """完整回归矩阵 Y，满足
        Y·π = M(q)a + C(q,v)v + g(q) + armature⊙a + Fv⊙v + Fc⊙sign(v)。"""
        n, k = self.nv, self.n_link_par
        Y = np.zeros((n, self.n_par))
        Y[:, :k] = self._base_regressor(q, v, a)
        Y[:, k : k + n] = np.diag(a)
        Y[:, k + n : k + 2 * n] = np.diag(v)
        Y[:, k + 2 * n :] = np.diag(np.tanh(200.0 * v))   # sign 的光滑近似
        return Y

    # ---------------- Slotine-Li 回归矩阵 ----------------

    def slotine_li_regressor(self, q, v, v_ref, a_ref) -> np.ndarray:
        """Y_SL·π = M(q)a_ref + C(q,v)v_ref + g(q) + armature⊙a_ref。"""
        z = np.zeros(self.nv)
        A = self._base_regressor(q, z, a_ref)
        Bp = self._base_regressor(q, v + v_ref, z)
        B1 = self._base_regressor(q, v, z)
        B2 = self._base_regressor(q, v_ref, z)
        G = self._base_regressor(q, z, z)

        n, k = self.nv, self.n_link_par
        Y = np.zeros((n, self.n_par))
        Y[:, :k] = A + 0.5 * (Bp - B1 - B2) + 0.5 * G
        Y[:, k : k + n] = np.diag(a_ref)
        # 粘性摩擦用参考速度 q̇ᵣ，使闭环误差动力学保持 M ṡ + (C+Fv+K_D) s = −Y π̃
        Y[:, k + n : k + 2 * n] = np.diag(v_ref)
        Y[:, k + 2 * n :] = np.diag(np.tanh(200.0 * v))
        return Y

    # ---------------- 参考实现（用于验证） ----------------

    def rnea(self, q, v, a) -> np.ndarray:
        """完整逆动力学，含摩擦。Pinocchio 的 rnea 已包含 armature。"""
        return (
            pin.rnea(self.model, self.data, q, v, a)
            + self.fv_true * v
            + self.fc_true * np.tanh(200.0 * v)
        )

    def mass_matrix(self, q) -> np.ndarray:
        """质量矩阵 M(q)。Pinocchio 的 crba 已包含 armature 对角项。"""
        return pin.crba(self.model, self.data, q).copy()

    def coriolis_times(self, q, v, w) -> np.ndarray:
        """C(q,v)·w。用 Christoffel 对称性由极化恒等式得到。"""
        z = np.zeros(self.nv)
        g = pin.rnea(self.model, self.data, q, z, z)
        c = lambda x: pin.rnea(self.model, self.data, q, x, z) - g
        return 0.5 * (c(v + w) - c(v) - c(w))

    def gravity(self, q) -> np.ndarray:
        z = np.zeros(self.nv)
        return pin.rnea(self.model, self.data, q, z, z).copy()

    def bias(self, q, v) -> np.ndarray:
        """h(q,q̇) = C(q,q̇)q̇ + g(q) + 摩擦。"""
        return self.rnea(q, v, np.zeros(self.nv))

    # ---------------- 基参数集 ----------------

    def base_parameter_projection(self, samples: int = 400, seed: int = 0):
        """求基参数集（可辨识参数）。

        标准惯性参数中有相当一部分对关节力矩没有任何影响，或只以固定线性组合
        出现（例如固定在基座上的连杆质量），导致 Y 列不满秩、最小二乘病态。
        做法：在关节空间随机采样堆叠 Y，做 SVD，取数值秩对应的右奇异向量
        张成的行空间作为基参数子空间。

        返回 (P, rank)：P ∈ R^{n_par × rank}，满足 Y·π = (Y·P)·(Pᵀπ)。
        """
        rng = np.random.default_rng(seed)
        lo = np.where(np.isfinite(self.model.lowerPositionLimit), self.model.lowerPositionLimit, -np.pi)
        up = np.where(np.isfinite(self.model.upperPositionLimit), self.model.upperPositionLimit, np.pi)

        stack = []
        for _ in range(samples):
            q = rng.uniform(lo, up)
            v = rng.uniform(-1.5, 1.5, self.nv)
            a = rng.uniform(-3.0, 3.0, self.nv)
            stack.append(self.regressor(q, v, a))
        W = np.vstack(stack)

        _, s, Vt = np.linalg.svd(W, full_matrices=False)
        rank = int(np.sum(s > s[0] * 1e-8))
        return Vt[:rank].T, rank


class LockedFingerArmView:
    """把「手臂 + 平行夹爪」的整机动力学，降成手臂 7 自由度的接口。

    为什么需要它
    ------------
    调参台统一用带夹爪的 `panda.xml`，Pinocchio 建出来是 9 自由度
    （7 个手臂关节 + 2 个手指移动副）。但所有手臂控制律（计算力矩、阻抗、
    自适应）都写在 7 维上，直接把 9×9 的 M 和 9 维的 h 喂进去会维度不符。

    降阶的数学依据（自己推，不是"取个子块凑合"）
    -------------------------------------------
    把广义坐标分成手臂 a 与手指 f 两块，整机方程为

        ⎡M_aa  M_af⎤ ⎡q̈_a⎤   ⎡h_a⎤   ⎡τ_a⎤
        ⎢          ⎥ ⎢    ⎥ + ⎢   ⎥ = ⎢   ⎥
        ⎣M_fa  M_ff⎦ ⎣q̈_f⎦   ⎣h_f⎦   ⎣τ_f⎦

    夹爪由腱传动的**位置伺服**驱动，抓取过程中开口宽度保持在指令值上，
    即 q̇_f = 0、q̈_f = 0。代入第一行：

        M_aa q̈_a + h_a(q, [q̇_a; 0]) = τ_a

    于是手臂看到的质量矩阵**精确等于** M 的左上 7×7 块，偏置力精确等于
    h 的前 7 个分量——不需要 Schur 补 M_aa − M_af M_ff⁻¹ M_fa。
    Schur 补对应的是手指**自由无约束**的情形（τ_f = 0 时消去 q̈_f），
    与位置伺服夹爪不是同一个物理假设。两者的差别就是"手指被夹住"与
    "手指松开随便晃"的差别。

    边界（必须诚实说明）
    ------------------
    1. 位置伺服不是理想约束。伺服刚度有限，夹持力很大时手指会被顶开，
       q̈_f ≠ 0，此时左上块只是近似。抓取场景里夹持力接近伺服能力上限时
       这个近似会退化。
    2. 手指张开宽度变化时 M_aa 会变（手指质量的位置变了）。本类每次调用都用
       **当前**宽度重新计算，不缓存，所以宽度变化被如实反映。
    3. 手上抓着的工件不在模型里。工件质量对手臂而言是未建模负载，
       会体现为重力补偿残差——这正是抓取场景要展示的现象之一。
    """

    def __init__(self, reg: DynamicsRegressor, n_arm: int = 7):
        self.reg = reg
        self.n_arm = int(n_arm)
        self.nv_full = reg.nv
        #: 对外表现成 n_arm 自由度，使它能直接顶替 DynamicsRegressor 传给
        #: SlotineLiAdaptiveController 这类只认 nv / n_par 的控制器。
        self.nv = self.n_arm
        self.n_par = reg.n_par
        if self.nv_full < self.n_arm:
            raise ValueError(f"模型自由度 {self.nv_full} 少于手臂自由度 {self.n_arm}")
        self.n_finger = self.nv_full - self.n_arm
        #: 手指关节位置，由外部按夹爪指令宽度写入（单指开度，单位 m）
        self.finger_q = np.zeros(self.n_finger)

    def set_finger_positions(self, q_finger) -> None:
        """更新手指关节位置。传标量表示两指同宽。"""
        q_finger = np.atleast_1d(np.asarray(q_finger, dtype=float))
        if q_finger.size == 1:
            self.finger_q = np.full(self.n_finger, float(q_finger[0]))
        elif q_finger.size == self.n_finger:
            self.finger_q = q_finger.astype(float).copy()
        else:
            raise ValueError(f"手指自由度为 {self.n_finger}，收到 {q_finger.size} 个值")

    def _expand(self, x_arm, fill: float | np.ndarray = 0.0) -> np.ndarray:
        out = np.empty(self.nv_full)
        out[: self.n_arm] = np.asarray(x_arm, dtype=float)
        out[self.n_arm:] = fill
        return out

    def _arm_rows(self, X: np.ndarray) -> np.ndarray:
        return X[: self.n_arm].copy()

    # ---------------- 动力学量 ----------------

    def mass_matrix(self, q_arm) -> np.ndarray:
        """手指锁定时手臂看到的 7×7 质量矩阵 M_aa。"""
        M = self.reg.mass_matrix(self._expand(q_arm, self.finger_q))
        return M[: self.n_arm, : self.n_arm].copy()

    def bias(self, q_arm, v_arm) -> np.ndarray:
        """手指锁定（q̇_f = 0）时手臂看到的偏置力 h_a = C q̇ + g + 摩擦。"""
        h = self.reg.bias(self._expand(q_arm, self.finger_q), self._expand(v_arm, 0.0))
        return self._arm_rows(h)

    def gravity(self, q_arm) -> np.ndarray:
        return self._arm_rows(self.reg.gravity(self._expand(q_arm, self.finger_q)))

    def coriolis_times(self, q_arm, v_arm, w_arm) -> np.ndarray:
        return self._arm_rows(self.reg.coriolis_times(
            self._expand(q_arm, self.finger_q),
            self._expand(v_arm, 0.0), self._expand(w_arm, 0.0)))

    def rnea(self, q_arm, v_arm, a_arm) -> np.ndarray:
        """手指锁定时的手臂逆动力学 τ_a = M_aa a_a + h_a。"""
        return self._arm_rows(self.reg.rnea(
            self._expand(q_arm, self.finger_q),
            self._expand(v_arm, 0.0), self._expand(a_arm, 0.0)))

    # ---------------- 回归矩阵 ----------------
    #
    # 回归矩阵按行对应关节。手指锁定时手臂那 7 行仍然严格满足
    #     τ_a = Y[0:7, :] · π
    # 其中 π 是**整机**参数向量（含手掌与两根手指的惯性参数）。
    # 参数向量不做裁剪：手部质量确实影响手臂力矩，把它从 π 里删掉就不再是
    # "线性于参数"的精确表示了，自适应律的 Lyapunov 推导会失去前提。

    def regressor(self, q_arm, v_arm, a_arm) -> np.ndarray:
        return self._arm_rows(self.reg.regressor(
            self._expand(q_arm, self.finger_q),
            self._expand(v_arm, 0.0), self._expand(a_arm, 0.0)))

    def slotine_li_regressor(self, q_arm, v_arm, v_ref_arm, a_ref_arm) -> np.ndarray:
        return self._arm_rows(self.reg.slotine_li_regressor(
            self._expand(q_arm, self.finger_q), self._expand(v_arm, 0.0),
            self._expand(v_ref_arm, 0.0), self._expand(a_ref_arm, 0.0)))

    def true_parameters(self) -> np.ndarray:
        return self.reg.true_parameters()

    def base_parameter_projection(self, samples: int = 400, seed: int = 0,
                                  finger_mode: str = "fixed"):
        """手指锁定条件下的基参数集。只返回 (P, rank)，奇异值见 base_parameter_svd。"""
        P, rank, _ = self.base_parameter_svd(samples, seed, finger_mode)
        return P, rank

    #: 允许的夹爪激励工况。默认 "fixed"，因为所有场景建好后都调用
    #: set_finger_positions() 把手指钉死，运行期间不再动。
    FINGER_MODES = ("fixed", "coupled", "independent")

    def base_parameter_svd(self, samples: int = 400, seed: int = 0,
                           finger_mode: str = "fixed", tol: float = 1e-8):
        """结构可辨识子空间：返回 (P, rank, 奇异值)。

        不能直接复用整机的投影：整机版本假设 9 个关节的力矩都可观测，
        而这里只有手臂 7 行可用。可辨识子空间**更小**，用整机的投影会把
        实际激励不到的参数方向也当成可辨识的，自适应律会沿这些方向漂移。

        ⚠️ 第五轮审核 P0-08：手指的激励工况直接决定 rank，必须显式声明。

        ==============  ====  ================================================
        finger_mode     rank  含义
        ==============  ====  ================================================
        ``"fixed"``       62  手指保持当前 finger_q 不动。**这才是所有场景
                              的实际工况**（build() 里 set_finger_positions()
                              之后就再没动过），因此是默认值。
        ``"coupled"``     67  平行夹爪同宽联动——真实夹爪能做到的最强激励。
        ``"independent"`` 70  两指各自独立随机。**物理上不可能**，平行夹爪
                              只有一个自由度。仅供讲解对照，说明「换一个
                              做不到的工况，秩会虚高 8 维」。
        ==============  ====  ================================================

        rank 是**结构秩**：在关节限位内全域随机取位形，力矩对哪些参数组合
        完全没有响应。它与「某条具体轨迹激励得够不够」是两回事，后者请看
        ``AdaptiveScene`` 的 ``rank_traj`` / ``pe_cond`` 读数。
        """
        if finger_mode not in self.FINGER_MODES:
            raise ValueError(
                f"finger_mode 只能是 {self.FINGER_MODES}，收到 {finger_mode!r}")
        rng = np.random.default_rng(seed)
        lo = np.where(np.isfinite(self.reg.model.lowerPositionLimit),
                      self.reg.model.lowerPositionLimit, -np.pi)[: self.n_arm]
        up = np.where(np.isfinite(self.reg.model.upperPositionLimit),
                      self.reg.model.upperPositionLimit, np.pi)[: self.n_arm]
        f_lo = np.where(np.isfinite(self.reg.model.lowerPositionLimit),
                        self.reg.model.lowerPositionLimit, 0.0)[self.n_arm:]
        f_up = np.where(np.isfinite(self.reg.model.upperPositionLimit),
                        self.reg.model.upperPositionLimit, 0.04)[self.n_arm:]

        saved = self.finger_q.copy()
        stack = []
        for _ in range(samples):
            if finger_mode == "independent":
                self.finger_q = rng.uniform(f_lo, f_up)
            elif finger_mode == "coupled":
                self.finger_q = np.full_like(saved,
                                             rng.uniform(f_lo[0], f_up[0]))
            # "fixed"：什么都不做，保持 saved
            q = rng.uniform(lo, up)
            v = rng.uniform(-1.5, 1.5, self.n_arm)
            a = rng.uniform(-3.0, 3.0, self.n_arm)
            stack.append(self.regressor(q, v, a))
        self.finger_q = saved

        W = np.vstack(stack)
        _, s, Vt = np.linalg.svd(W, full_matrices=False)
        rank = int(np.sum(s > s[0] * tol))
        return Vt[:rank].T, rank, s
