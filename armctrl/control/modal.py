"""耦合机械臂的闭环模态频率。

为什么需要这个模块（第四轮审核 P0-02）
-----------------------------------
PD + 重力补偿的闭环误差方程是**耦合**的：

    M(q)·ë + K_d·ė + K_p·e = 0

它的固有频率来自**广义特征值问题** ``det(K_p − ω²M) = 0``，
也就是矩阵 ``M⁻¹K_p`` 的特征值开方。

⚠️ 项目里原来用的是

    ω_j = sqrt(K_p · [M⁻¹]_jj)          # 只取对角线

这**不是模态频率**，只是"把第 j 个关节单独拎出来、假装其他关节不动"
时的等效频率。M 的非对角项（关节之间的惯性耦合）被整个丢掉了。

**反例（审核报告给的，本模块的测试逐位复现）**：

    M = [[2, 1],
         [1, 2]],   K_p = I

    对角线法：sqrt(diag(M⁻¹)) = [0.8165, 0.8165]  → 离散度 1.000（"完全一致"）
    真实模态：                  [1.0000, 0.5774]  → 离散度 1.732

对角线法**报告离散度为 0**，而真实的两个模态差了 1.73 倍。
在 Panda 上实测：对角线法给最慢模态 17.76 rad/s、离散度 3.11；
真实模态最慢只有 **12.50 rad/s**、离散度 **4.48**。
→ **对角线法把最慢的那个模态高报了 42%，把离散度低报了 31%。**

这为什么危险：**最慢的模态决定系统整体的响应速度**，
高报它 = 以为系统比实际快 42%。
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

__all__ = ["modal_frequencies", "diagonal_frequencies", "modal_spread"]


def modal_frequencies(M, Kp) -> np.ndarray:
    """闭环 ``M ë + K_p e = 0`` 的模态固有频率（rad/s），从小到大排序。

    解广义特征值问题 ``K_p v = ω² M v``。

    参数
    ----
    M   (n, n) 质量矩阵，必须对称正定
    Kp  (n, n) 刚度矩阵，或标量 / (n,) 向量（会展开成对角阵）

    返回
    ----
    (n,) 频率数组，**升序**。``[0]`` 就是最慢的模态，决定整体响应速度。
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    Kp = np.asarray(Kp, dtype=float)
    if Kp.ndim == 0:
        Kp = float(Kp) * np.eye(n)
    elif Kp.ndim == 1:
        Kp = np.diag(Kp)
    Msym = 0.5 * (M + M.T)
    ev = sla.eigvalsh(0.5 * (Kp + Kp.T), Msym)
    return np.sqrt(np.maximum(ev, 0.0))


def diagonal_frequencies(M, Kp) -> np.ndarray:
    """⚠️ **旧的、错的**逐关节算法：``sqrt(K_p · [M⁻¹]_jj)``。

    保留它**只是为了在测试和文档里当反面教材**，不要在新代码里用。
    它忽略 M 的非对角项，即关节之间的惯性耦合。
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    Kp = np.asarray(Kp, dtype=float)
    if Kp.ndim == 0:
        kp_diag = np.full(n, float(Kp))
    elif Kp.ndim == 1:
        kp_diag = Kp
    else:
        kp_diag = np.diag(Kp)
    return np.sqrt(np.maximum(kp_diag * np.diag(np.linalg.inv(M)), 0.0))


def modal_spread(M, Kp) -> float:
    """模态离散度 = 最快模态 / 最慢模态。

    1.0 表示所有模态同速（这正是计算力矩想达到的效果）；
    越大表示各方向快慢差异越悬殊，越难用**一个**标量增益调好。
    """
    w = modal_frequencies(M, Kp)
    lo = float(w[0])
    return float(w[-1]) / lo if lo > 1e-12 else float("inf")
