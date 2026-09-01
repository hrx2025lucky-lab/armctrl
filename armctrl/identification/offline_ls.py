"""离线批量辨识：和在线自适应律解**同一份数据**，但解的是另一个问题。

为什么单独写一个模块
--------------------
``adaptive.py`` 里的 Slotine-Li 更新律是**在线**的：它的设计目标是让**跟踪
误差**在李雅普诺夫意义下收敛，参数收敛只是「激励恰好充分时顺带的副产品」。
``03_自适应控制.md`` §七实测证明这个副产品基本拿不到——跑满 600 秒，可辨识
子空间偏差只从 84% 降到 74%，把增益调大 10 倍则直接发散。

本模块解的是另一个问题：**给定一段轨迹上记录的 (Y, τ)，求让力矩残差最小的
参数**。这就是辨识本身，没有稳定性约束，可以放心用强力数值方法。
工业界（以及 ETH 的 PACE 论文）做参数辨识走的都是这条路。

数值上的三个讲究
----------------
1. **不建 Gramian**。``G = AᵀA`` 把条件数**平方**：κ(A)~1e8 时 κ(G)~1e16
   已经顶到双精度极限。这里用**增量 QR**，一次只保留 (r+1)×(r+1) 的 R 因子，
   内存与样本数无关。
2. **先投影到基参数子空间再回归**。直接对 117 维做最小二乘是病态的
   （55 个方向没有信息），解会沿零空间漂走。投影到 P 的 62 列上之后，
   问题才是良态的。
3. **把 τ 一起放进 QR**。解 ``min‖Aβ−τ‖`` 时把 τ 当成 A 的第 r+1 列一起
   分解，R 的右上角直接给出右端项，右下角标量就是残差范数——一次分解
   同时拿到解和残差，不用再回头扫一遍数据。

只有编码器的话怎么办
--------------------
本模块要求外部提供 τ。真机上如果没有力矩传感器，就只能像 PACE 那样改成
「用位置轨迹的重演误差当代价」，再用无梯度优化器（CMA-ES）去搜——那是另
一套方法，不是最小二乘。两者的取舍见 ``03_自适应控制.md`` §八之二。
"""

from __future__ import annotations

import numpy as np

__all__ = ["OfflineBaseLS", "mechanical_power", "potential_power", "copper_loss"]


class OfflineBaseLS:
    """基参数坐标下的增量最小二乘。

    用法::

        ls = OfflineBaseLS(P)
        for (Y, tau) in 数据流:
            ls.add(Y, tau)
        beta, info = ls.solve()
        pi_hat = P @ beta

    Parameters
    ----------
    P : (n_par, r) 基参数投影矩阵，来自 ``base_parameter_svd``。
    ridge : 吉洪诺夫正则系数 λ。解变成 ``min‖Aβ−τ‖² + λ²‖β‖²``。
        默认 0（不正则）。**λ>0 会引入偏差**，只在条件数实在太差时用，
        且必须在结论里声明用了多大的 λ。
    """

    def __init__(self, P: np.ndarray, ridge: float = 0.0):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2:
            raise ValueError(f"P 必须是二维矩阵，收到 shape={P.shape}")
        if ridge < 0.0:
            raise ValueError(f"ridge 不能为负，收到 {ridge}")
        self.P = P
        self.n_par, self.r = P.shape
        self.ridge = float(ridge)
        self.reset()

    def reset(self) -> None:
        self._R = np.zeros((self.r + 1, self.r + 1))
        self.rows = 0
        self.tau_sq = 0.0

    def add(self, Y: np.ndarray, tau: np.ndarray) -> None:
        """折进一批样本。Y 形状 (k, n_par)，tau 形状 (k,)。"""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        tau = np.atleast_1d(np.asarray(tau, dtype=float)).ravel()
        if Y.shape[1] != self.n_par:
            raise ValueError(
                f"Y 的列数应为 {self.n_par}，收到 {Y.shape[1]}")
        if Y.shape[0] != tau.size:
            raise ValueError(
                f"Y 的行数 {Y.shape[0]} 与 tau 长度 {tau.size} 对不上")
        A = np.hstack([Y @ self.P, tau.reshape(-1, 1)])
        self._R = np.linalg.qr(np.vstack([self._R, A]), mode="r")
        self.rows += Y.shape[0]
        self.tau_sq += float(tau @ tau)

    def solve(self):
        """返回 ``(beta, info)``。样本不足时 beta 为 None。

        info 字典::

            rows      已折进的行数
            rank      数据实际张开的维数（≤ r）
            cond      log10 条件数，只在张开的维数内算
            residual  ‖Aβ−τ‖ / ‖τ‖，相对力矩残差
        """
        info = {"rows": self.rows, "rank": 0, "cond": float("nan"),
                "residual": float("nan")}
        if self.rows < self.r:
            return None, info

        R = self._R
        A = R[: self.r, : self.r]
        b = R[: self.r, self.r]
        s = np.linalg.svd(A, compute_uv=False)
        rank = int(np.sum(s > s[0] * 1e-10)) if s[0] > 0 else 0
        info["rank"] = rank
        if rank:
            info["cond"] = float(np.log10(s[0] / s[rank - 1]))
        if rank < self.r:
            # 数据没张满：用最小范数解，别去求一个奇异矩阵的逆
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
        elif self.ridge > 0.0:
            aug = np.vstack([A, self.ridge * np.eye(self.r)])
            rhs = np.concatenate([b, np.zeros(self.r)])
            beta = np.linalg.lstsq(aug, rhs, rcond=None)[0]
        else:
            beta = np.linalg.solve(A, b)

        # QR 的右下角标量 = 残差范数（最小范数/正则解时不再成立，重算）
        if rank == self.r and self.ridge == 0.0:
            resid = abs(float(R[self.r, self.r]))
        else:
            resid = float(np.linalg.norm(A @ beta - b))
        if self.tau_sq > 0.0:
            info["residual"] = resid / np.sqrt(self.tau_sq)
        return beta, info


# ----------------------------------------------------------------------
# PACE 论文（arXiv:2509.06342）的能耗模型，式 11~14。
#
# ⚠️ 三项里只有两项能在本项目里精确算出来：
#   P_mech 和 P_pot 只需要仿真里已有的量；
#   P_el（铜损）需要每个关节的**相电阻 R、力矩常数 k_i、减速比 r**，
#   这三个数只能查电机数据手册。Franka Panda 的这些参数不公开，
#   所以这里把 copper_loss 写成必须显式传参的纯函数——
#   **不提供默认值，就不会有人不小心把编出来的数字当成实测。**
# ----------------------------------------------------------------------

def mechanical_power(tau, qd, k_regen: float = 0.0) -> float:
    """机械功率，PACE 式 13。

    ``k_regen`` 是能量回馈效率：制动时电机发的电有多少回到母线。
    论文里 ANYmal 取 0（不回馈），Tytan / Minimal 取 0.3。
    取 0 表示**制动能量全部当热耗掉**，这是保守（偏大）的估计。
    """
    if not 0.0 <= k_regen <= 1.0:
        raise ValueError(f"k_regen 应在 [0,1]，收到 {k_regen}")
    p = float(np.asarray(tau, dtype=float) @ np.asarray(qd, dtype=float))
    return p if p > 0.0 else k_regen * p


def potential_power(masses, vz, g: float = 9.81) -> float:
    """重力功率 ``Σ m_b g v_{b,z}``，PACE 式 14。

    往上抬为正（在耗能），往下落为负。注意它**不是**损耗——
    势能是可逆的，PACE 把它算进 P_total 是为了让奖励反映
    「把质心抬高本身就要花电」。
    """
    m = np.asarray(masses, dtype=float)
    v = np.asarray(vz, dtype=float)
    if m.shape != v.shape:
        raise ValueError(f"masses {m.shape} 与 vz {v.shape} 形状不一致")
    return float(np.sum(m * g * v))


def copper_loss(tau, resistance, torque_const, gear_ratio) -> float:
    """铜损（焦耳损耗），PACE 式 12：``Σ τ_j² R_j / (r_j² k_{i,j}²)``。

    ⚠️ **三个电机常数没有默认值，必须由调用方从数据手册提供。**
    随手编一组数会让输出看起来像瓦特，实际上毫无意义。

    只有 τ 是控制算法能左右的；R、k_i、r 是选型阶段定死的。
    这也是为什么很多论文直接用 ``Στ²`` 当能耗代理——
    在**同一台机器人**上它和铜损只差一个常数。
    但跨机器人比较时这个常数不一样，``Στ²`` 就没有可比性了，
    PACE 批评的正是这一点。
    """
    tau = np.asarray(tau, dtype=float)
    R = np.asarray(resistance, dtype=float)
    ki = np.asarray(torque_const, dtype=float)
    r = np.asarray(gear_ratio, dtype=float)
    if not (tau.shape == R.shape == ki.shape == r.shape):
        raise ValueError(
            f"四个输入形状必须一致：tau{tau.shape} R{R.shape} "
            f"k_i{ki.shape} r{r.shape}")
    if np.any(ki == 0.0) or np.any(r == 0.0):
        raise ValueError("力矩常数和减速比不能为 0（会除零）")
    return float(np.sum(tau ** 2 * R / (r ** 2 * ki ** 2)))
