"""折线路径的圆弧倒角：把 C⁰ 的折线变成 C¹ 的可跟踪路径。

为什么需要它
------------
RRT 给出的是折线。在拐点处切线方向是**跳变**的：

    q(s) = q_a + u·s，拐角前后 u 不同

于是 q̇ = u·ṡ 在拐点处跳变，除非 ṡ = 0。这就是"逐段 S 曲线必须在每个路径点
停一下"的根本原因——不停就意味着加速度冲激，任何有界力矩的机器人都执行不了。

倒角把拐角换成一段圆弧，切线连续变化，机器人就能**不减速地拐过去**。
MoveIt 默认的时间参数化（Kunz–Stilman TOTG）第一步做的正是这件事。

几何推导（自己推的，不引用现成公式）
------------------------------------
设拐点 q₁，入段单位方向 u₁ = (q₁−q₀)/‖·‖，出段单位方向 u₂ = (q₂−q₁)/‖·‖。
转角 θ 满足 cos θ = u₁·u₂。

从 q₁ 沿两侧各退回距离 L，得到切点

    A = q₁ − L·u₁,     B = q₁ + L·u₂

要求圆弧在 A 处切于 u₁、在 B 处切于 u₂。圆心 C 必须同时到两条直线等距，
所以落在角平分线上。记 q₁ 处两条射线 (−u₁) 与 (u₂) 的夹角为 φ = π − θ，
圆心到 q₁ 的距离 d 与半径 r 满足

    L = d·cos(φ/2),    r = d·sin(φ/2)

代入 φ/2 = π/2 − θ/2，即 cos(φ/2) = sin(θ/2)、sin(φ/2) = cos(θ/2)：

    d = L / sin(θ/2),    r = L / tan(θ/2)

圆弧对 q₁ 的最大偏离就是"圆心到拐点的距离减去半径"：

    δ = d − r = L·(1 − cos(θ/2)) / sin(θ/2)

反解出给定偏离预算 δ 时能退回多远：

    L = δ·sin(θ/2) / (1 − cos(θ/2))

圆弧的参数化：取 e₂ = u₁，e₁ = (A − C)/r，则

    P(s) = C + r·cos(s/r)·e₁ + r·sin(s/r)·e₂,   s ∈ [0, r·θ]

可以直接验证 (A−C)·u₁ = 0 且 ‖A−C‖ = r（见 tests/test_topp.py 的独立核对），
所以 e₁ ⟂ e₂ 且都是单位向量；P(0) = A，切线 P′(0) = e₂ = u₁，
弧长恰为 r·θ。

安全性提醒
----------
**倒角会改变几何路径**，最大偏离 δ 是有界的但不是零。原路径无碰撞
不代表倒角后仍无碰撞，必须用碰撞检测重新核验（见 blend_and_verify）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Line:
    q0: np.ndarray
    u: np.ndarray
    length: float

    def at(self, s: float) -> np.ndarray:
        return self.q0 + self.u * s

    def tangent(self, s: float) -> np.ndarray:
        return self.u


@dataclass
class _Arc:
    center: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    radius: float
    length: float

    def at(self, s: float) -> np.ndarray:
        a = s / self.radius
        return self.center + self.radius * (np.cos(a) * self.e1 + np.sin(a) * self.e2)

    def tangent(self, s: float) -> np.ndarray:
        a = s / self.radius
        return -np.sin(a) * self.e1 + np.cos(a) * self.e2


class BlendedPath:
    """折线倒角后的路径，按弧长参数化，处处 C¹。

    属性
    ----
    length      总弧长
    corners     每个拐角的 (转角 θ, 退回距离 L, 半径 r, 实际偏离 δ)
    n_blended   实际做了倒角的拐点个数
    """

    def __init__(self, segments, corners):
        self.segments = segments
        self.corners = corners
        self._starts = np.concatenate([[0.0], np.cumsum([s.length for s in segments])])
        self.length = float(self._starts[-1])
        self.n_blended = sum(1 for c in corners if c["blended"])

    def _locate(self, s: float):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self._starts, s, side="right") - 1)
        i = min(max(i, 0), len(self.segments) - 1)
        return self.segments[i], s - self._starts[i]

    def __call__(self, s: float) -> np.ndarray:
        seg, ds = self._locate(s)
        return seg.at(ds)

    def tangent(self, s: float) -> np.ndarray:
        seg, ds = self._locate(s)
        return seg.tangent(ds)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """等弧长采样 n 个点，返回 (弧长数组, 构型数组)。"""
        ss = np.linspace(0.0, self.length, int(n))
        return ss, np.array([self(s) for s in ss])


def blend_corners(waypoints, max_deviation: float = 0.05,
                  min_angle: float = 1e-3,
                  max_angle: float = np.pi - 1e-2) -> BlendedPath:
    """把折线的拐角换成圆弧。

    max_deviation   允许的最大偏离（关节空间的欧氏距离，单位 rad）。
                    设成 0 表示不倒角，退化回原折线。
    min_angle       转角小于它就认为是共线，不需要倒角。
    max_angle       转角大于它（接近 180°，原路返回）无法倒角：
                    此时半径 r = L/tan(θ/2) → 0，圆弧退化成一个点，
                    机器人在这里**必须停**。保留尖角并如实标记。
    """
    pts = [np.asarray(p, dtype=float) for p in waypoints]
    if len(pts) < 2:
        raise ValueError("至少需要两个路径点")

    dirs, lens = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            continue
        dirs.append(d / L)
        lens.append(L)
    if not dirs:
        raise ValueError("路径退化：所有点重合")
    # 去掉重复点后重建顶点序列
    verts = [pts[0]]
    for u, L in zip(dirs, lens):
        verts.append(verts[-1] + u * L)

    n_corner = len(dirs) - 1
    cut = np.zeros(n_corner)
    corners = []
    for i in range(n_corner):
        u1, u2 = dirs[i], dirs[i + 1]
        theta = float(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        info = dict(theta=theta, L=0.0, radius=0.0, deviation=0.0, blended=False)
        if max_deviation > 0 and min_angle < theta < max_angle:
            h = theta / 2.0
            # L = δ·sin(θ/2)/(1−cos(θ/2))，再受限于两侧线段各自的一半
            L = max_deviation * np.sin(h) / (1.0 - np.cos(h))
            L = min(L, lens[i] / 2.0, lens[i + 1] / 2.0)
            r = L / np.tan(h)
            info.update(L=L, radius=r,
                        deviation=L * (1.0 - np.cos(h)) / np.sin(h), blended=True)
            cut[i] = L
        corners.append(info)

    segments = []
    for i, (u, L) in enumerate(zip(dirs, lens)):
        head = cut[i - 1] if i > 0 else 0.0            # 上一个拐角吃掉的长度
        tail = cut[i] if i < n_corner else 0.0         # 下一个拐角吃掉的长度
        start = verts[i] + u * head
        straight = L - head - tail
        if straight > 1e-12:
            segments.append(_Line(start, u, straight))
        if i < n_corner and corners[i]["blended"]:
            u1, u2 = dirs[i], dirs[i + 1]
            q1 = verts[i + 1]
            Lc, r = corners[i]["L"], corners[i]["radius"]
            theta = corners[i]["theta"]
            bis = u2 - u1
            bis = bis / np.linalg.norm(bis)            # 指向转弯内侧的角平分线
            d = Lc / np.sin(theta / 2.0)
            center = q1 + d * bis
            A = q1 - Lc * u1
            e1 = (A - center) / r
            segments.append(_Arc(center, e1, u1.copy(), r, r * theta))
    return BlendedPath(segments, corners)


def blend_and_verify(waypoints, checker, max_deviation: float = 0.05,
                     samples_per_rad: float = 200.0,
                     shrink: float = 0.5, attempts: int = 5):
    """倒角并用碰撞检测复核；若倒角后碰撞就把偏离预算折半重试。

    倒角是**改变几何路径**的操作，原路径合法不蕴含倒角后合法——
    圆弧总是往转弯内侧切进去，而障碍物往往正好在内侧。
    因此这里不"相信"倒角，而是每次都重新核验。

    返回 (BlendedPath, 实际使用的 max_deviation, 复核得到的最小间隙)。
    复核失败到最后会退回不倒角的折线（deviation = 0）。
    """
    dev = float(max_deviation)
    for _ in range(int(attempts)):
        path = blend_corners(waypoints, max_deviation=dev)
        n = max(int(np.ceil(path.length * samples_per_rad)), 2)
        _, Q = path.sample(n)
        worst = min(checker.clearances(q)[0] for q in Q)
        if worst > checker.safety_margin:
            return path, dev, float(worst)
        dev *= shrink
    path = blend_corners(waypoints, max_deviation=0.0)
    n = max(int(np.ceil(path.length * samples_per_rad)), 2)
    _, Q = path.sample(n)
    worst = min(checker.clearances(q)[0] for q in Q)
    return path, 0.0, float(worst)
