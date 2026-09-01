"""自适应控制：Slotine-Li 模型自适应（MBAC）与 RBF 神经网络自适应。

两者都属于控制理论中的「学习控制」，与强化学习有本质区别：
    自适应控制  在线更新参数/权重，更新律由 Lyapunov 函数反推，稳定性可证，
                无需数据集、无需离线训练，上电即可运行。
    强化学习    通过采样试错优化策略，需要奖励函数与大量 rollout，无稳定性保证。

一、Slotine-Li 模型自适应（MBAC）
--------------------------------
适用：动力学结构已知（能写出回归矩阵），参数未知。

    滤波误差    s = ė + Λe          e = q_d − q
    参考速度    q̇ᵣ = q̇_d + Λe      （于是 s = q̇ᵣ − q̇）
    参考加速度  q̈ᵣ = q̈_d + Λė
    控制律      τ = Y(q,q̇,q̇ᵣ,q̈ᵣ)·π̂ + K_D·s
    自适应律    π̂̇ = Γ·Yᵀ·s

Lyapunov 设计：取 V = ½sᵀM(q)s + ½π̃ᵀΓ⁻¹π̃（π̃ = π̂ − π），闭环为
M ṡ + C s + K_D s = −Y π̃。求导并利用 Ṁ − 2C 的斜对称性，得

    V̇ = −sᵀK_D s + π̃ᵀ(Γ⁻¹π̂̇ − Yᵀs)

自适应律 π̂̇ = ΓYᵀs 正是为消去符号不定的交叉项而设计，于是 V̇ = −sᵀK_D s ≤ 0，
所有信号有界，再由 Barbalat 引理得 s → 0，跟踪误差渐近收敛。

两点必须区分：
    跟踪收敛 ≠ 参数收敛。π̂ → π 还需轨迹满足持续激励（PE）条件。
    渐近收敛（MBAC，动力学严格线性于参数）vs 一致最终有界 UUB（RBF，存在逼近误差）。

关于参数维数（⭐ 三套口径必须分开说，混成一个数字就是错的）：

本项目模型共 117 个参数 = 9 个刚体 × 10 个惯性参数（90）+ 9 个关节 × 3 个
摩擦/传动参数（27）。**结构可辨识秩取决于夹爪能不能动、怎么动**
（独立 SVD 实测，400 次随机采样，奇异值断崖均在 10¹² 倍以上）：

| 夹爪工况 | 结构秩 | 说明 |
|---|---:|---|
| 双指各自独立随机运动 | 70 | ⚠️ 平行夹爪**做不到**的工况 |
| 双指同宽联动（真实平行夹爪） | 67 | 只多出一个开合自由度 |
| **夹爪固定**（AdaptiveScene 实际工况） | **62** | 夹爪成为末端连杆的刚性延伸 |

直接对全部 117 个参数做自适应会病态，因此支持在基参数子空间中自适应：
π̂ = P β̂，只更新 β̂。

⚠️ 报秩之前必须先说清楚是哪种机械结构、哪些关节受控、加了什么约束。
纯 7 轴无手模型是另一套口径（7×10 + 7×3 = 91 个参数），不能和上表混着说。

二、RBF 神经网络自适应
----------------------
适用：连动力学结构都写不出。用高斯基网络在线逼近未知项。

    控制律      τ = Ŵᵀ h(x) + K_D·s
    权重更新    Ŵ̇ = Γ(h sᵀ − σŴ)

σ-modification（泄漏项）用于抑制权重漂移：仅有 Ŵ̇ = Γh sᵀ 时，在逼近误差和
噪声下权重可能缓慢无界增长，加入 −σŴ 后可保证权重有界（代价是引入小的稳态偏差）。

维数灾难的处理：输入 x = [q; q̇; q̇ᵣ; q̈ᵣ] 为 4n = 28 维，在全空间铺高斯中心
需要指数级数量。但机器人实际只在期望轨迹的邻域内运动，该区域是 28 维空间中的
一条一维曲线。因此中心沿期望轨迹采样布置，宽度按相邻中心间距自适应确定，
用几百个中心即可覆盖真实工作区域。
"""

from __future__ import annotations

import numpy as np

from armctrl.identification.regressor import DynamicsRegressor


class SlotineLiAdaptiveController:
    """Slotine-Li 模型自适应控制器（7 自由度）。

    参数
    ----
    lam         Λ，滤波误差增益，决定误差流形收敛速度
    kd          K_D，鲁棒反馈增益
    gamma       Γ，自适应增益（标量或对角）
    use_base    是否在基参数子空间中自适应（推荐 True）
    pi_init     参数初值。None 表示从 0 起估（最保守，展示完整学习过程）
    """

    def __init__(
        self,
        reg: DynamicsRegressor,
        n_ctrl: int,
        lam: float = 10.0,
        kd: float = 60.0,
        gamma: float = 0.05,
        use_base: bool = True,
        pi_init: np.ndarray | None = None,
        proj_bound: float = 200.0,
    ):
        self.reg = reg
        self.n = n_ctrl
        self.nv = reg.nv
        self.lam = lam
        self.kd = kd
        self.proj_bound = proj_bound

        if use_base:
            self.P, self.rank = reg.base_parameter_projection(samples=300)
        else:
            self.P, self.rank = np.eye(reg.n_par), reg.n_par

        self.gamma = gamma
        pi0 = np.zeros(reg.n_par) if pi_init is None else pi_init.copy()
        self.beta = self.P.T @ pi0            # 基参数坐标下的估计
        self.beta0_norm = float(np.linalg.norm(self.beta))

    def reset(self) -> None:
        self.beta[:] = 0.0

    @property
    def pi_hat(self) -> np.ndarray:
        return self.P @ self.beta

    def compute(self, q, qd_vel, q_des, v_des, a_des, dt: float):
        """返回 (tau, s)。q/qd_vel 为完整 nv 维状态。"""
        e = q_des - q
        de = v_des - qd_vel
        s = de + self.lam * e
        v_ref = v_des + self.lam * e
        a_ref = a_des + self.lam * de

        Y = self.reg.slotine_li_regressor(q, qd_vel, v_ref, a_ref)
        Yb = Y @ self.P                        # 投影到基参数子空间

        tau = Yb @ self.beta + self.kd * s

        # 自适应律 β̂̇ = Γ Ybᵀ s
        self.beta += self.gamma * (Yb.T @ s) * dt
        # 参数投影：限制估计范数，防止 PE 不足时参数缓慢漂移
        nrm = np.linalg.norm(self.beta)
        if nrm > self.proj_bound:
            self.beta *= self.proj_bound / nrm
        return tau, s


class RBFAdaptiveController:
    """RBF 神经网络自适应控制器（7 自由度，中心沿轨迹布置 + σ-modification）。

    参数
    ----
    centers     (n_neurons, dim) 高斯中心，由 fit_centers_to_trajectory 生成
    widths      (n_neurons,) 高斯宽度
    gamma       权重自适应增益
    sigma       σ-modification 泄漏系数，抑制权重漂移
    """

    def __init__(
        self,
        n_ctrl: int,
        centers: np.ndarray,
        widths: np.ndarray,
        lam: float = 10.0,
        kd: float = 60.0,
        gamma: float = 300.0,
        sigma: float = 0.02,
        w_bound: float = 500.0,
    ):
        self.n = n_ctrl
        self.C = np.asarray(centers, float)
        self.b = np.asarray(widths, float)
        self.m = self.C.shape[0]
        self.lam = lam
        self.kd = kd
        self.gamma = gamma
        self.sigma = sigma
        self.w_bound = w_bound
        self.W = np.zeros((self.m, n_ctrl))

    def reset(self) -> None:
        self.W[:] = 0.0

    def basis(self, x: np.ndarray) -> np.ndarray:
        d2 = np.sum((self.C - x) ** 2, axis=1)
        return np.exp(-d2 / (2.0 * self.b**2))

    def compute(self, q, qd_vel, q_des, v_des, a_des, dt: float):
        e = q_des - q
        de = v_des - qd_vel
        s = de + self.lam * e
        v_ref = v_des + self.lam * e
        a_ref = a_des + self.lam * de

        x = np.concatenate([q, qd_vel, v_ref, a_ref])
        h = self.basis(x)

        # 只对前 n 个受控关节输出力矩（夹爪关节不参与）
        s_ctrl = s[: self.n]
        tau = self.W.T @ h + self.kd * s_ctrl

        # Ŵ̇ = Γ(h sᵀ − σŴ)：第二项为泄漏，保证权重有界
        self.W += self.gamma * (np.outer(h, s_ctrl) - self.sigma * self.W) * dt
        nrm = np.linalg.norm(self.W)
        if nrm > self.w_bound:
            self.W *= self.w_bound / nrm
        return tau, s_ctrl


def fit_centers_to_trajectory(
    traj_fn,
    t_end: float,
    n_centers: int = 240,
    lam: float = 10.0,
    jitter: float = 0.12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """沿期望轨迹采样布置高斯中心。

    机器人只在期望轨迹邻域运动，因此不需要在 4n 维空间全域铺中心。
    做法：沿轨迹等时间采样，构造 x = [q; q̇; q̇ᵣ; q̈ᵣ]（跟踪良好时 q≈q_d、q̇≈q̇_d），
    再加入小幅随机扰动覆盖轨迹邻域。宽度取为到最近邻中心距离的若干倍。
    """
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, t_end, n_centers)
    cs = []
    for t in ts:
        qd, vd, ad = traj_fn(t)
        x = np.concatenate([qd, vd, vd, ad])
        x = x + jitter * rng.standard_normal(x.size) * (np.abs(x) + 0.5)
        cs.append(x)
    C = np.array(cs)

    # 宽度按最近邻距离设定，保证基函数之间有适度重叠
    d = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    widths = np.clip(1.5 * nn, 0.5, 20.0)
    return C, widths
