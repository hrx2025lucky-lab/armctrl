"""RRT-Connect（双向快速扩展随机树）+ 短路平滑。

算法回顾（Kuffner & LaValle, 2000）
-----------------------------------
单向 RRT 从起点长一棵树，靠随机采样把树"撑"进自由空间，等某个节点足够
靠近目标就结束。它的问题是目标只靠随机撞上，在狭窄通道里效率很低。

RRT-Connect 改两件事：

1. **双向**：起点和终点各长一棵树，每轮尝试把两棵树接上。
   两棵树相向生长，需要的采样量比单向少一个量级。

2. **贪心 CONNECT**：一棵树朝目标方向不是只走一步，而是一直走到
   撞上障碍或到达为止。自由空间大时一步就能连通。

每轮：
    q_rand ← 均匀采样
    EXTEND(树A, q_rand)  →  若长出新节点 q_new
        CONNECT(树B, q_new)  →  若到达，两树连通，规划成功
    交换 A、B                （保证两棵树轮流做"主动方"）

性质与边界
----------
* **概率完备**：有解时，采样数趋于无穷则找到解的概率趋于 1。
  没有有限步的保证，因此"失败"只能理解为"在给定预算内没找到"，
  **不能**理解为"证明无解"。
* **不最优**：RRT-Connect 不是 RRT*，路径通常明显长于最短路，
  所以后面必须接平滑。
* 碰撞检测是**离散采样**的（见 collision.py），不是连续保证。

距离度量
--------
关节空间加权欧氏距离。权重的物理意义是"这个关节转 1 rad 相当于多远"，
默认全 1。对 Panda 这类近端关节带动整条臂的结构，给近端关节更大权重
更符合工作空间里的实际位移，但那样度量就依赖构型了；这里保持简单且各向同性，
把安全性交给 safety_margin，把路径质量交给后面的平滑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armctrl.planning.collision import CollisionChecker


# ------------------------------------------------------------------ 路径工具


def path_length(path, weights=None) -> float:
    """折线路径在加权关节空间中的总长。"""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0)
    if weights is not None:
        d = d * np.asarray(weights, dtype=float)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def cumulative_length(path, weights=None) -> np.ndarray:
    """每个顶点处的累计弧长，长度为 len(path)，首项为 0。"""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return np.zeros(len(path))
    d = np.diff(path, axis=0)
    if weights is not None:
        d = d * np.asarray(weights, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])


def path_interpolate(path, s: float, weights=None) -> np.ndarray:
    """按弧长参数 s（0 到总长）在折线路径上取点。"""
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) == 1:
        return path[0].copy()
    cum = cumulative_length(path, weights)
    s = float(np.clip(s, 0.0, cum[-1]))
    for i in range(len(path) - 1):
        L = cum[i + 1] - cum[i]
        if L <= 0.0:
            continue
        if s <= cum[i + 1]:
            return path[i] + (s - cum[i]) / L * (path[i + 1] - path[i])
    return path[-1].copy()


# ------------------------------------------------------------------ 树


class _Tree:
    """带父指针的节点表。最近邻用暴力搜索。

    暴力最近邻是 O(n)，n 是节点数。加 KD 树只在节点上万时才划算，
    而 RRT-Connect 在本项目的场景里通常几百个节点就连通了，
    KD 树的重建开销反而更大。
    """

    def __init__(self, root: np.ndarray, capacity: int = 256):
        self._q = np.empty((capacity, len(root)), dtype=float)
        self._parent = np.empty(capacity, dtype=int)
        self._q[0] = root
        self._parent[0] = -1
        self.size = 1

    def node(self, i: int) -> np.ndarray:
        return self._q[i]

    @property
    def nodes(self) -> np.ndarray:
        return self._q[: self.size]

    def add(self, q: np.ndarray, parent: int) -> int:
        if self.size == len(self._q):
            self._q = np.vstack([self._q, np.empty_like(self._q)])
            self._parent = np.concatenate([self._parent, np.empty_like(self._parent)])
        self._q[self.size] = q
        self._parent[self.size] = parent
        self.size += 1
        return self.size - 1

    def nearest(self, q: np.ndarray, weights=None) -> int:
        d = self.nodes - q
        if weights is not None:
            d = d * weights
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def branch(self, i: int) -> list[np.ndarray]:
        """从根到节点 i 的构型序列。"""
        out = []
        while i >= 0:
            out.append(self._q[i].copy())
            i = int(self._parent[i])
        out.reverse()
        return out


# ------------------------------------------------------------------ 规划器


@dataclass
class RRTConnectConfig:
    step_size: float = 0.30           # 单次 EXTEND 的最大步长（加权关节空间）
    max_iters: int = 4000
    limit_margin: float = 0.02        # 采样时离关节限位留的余量 (rad)
    try_direct: bool = True           # 先试一次起点到终点的直线
    weights: np.ndarray | None = None


@dataclass
class PlanResult:
    success: bool
    path: list[np.ndarray] = field(default_factory=list)
    iterations: int = 0
    nodes: int = 0
    reason: str = ""
    collision_checks: int = 0


_TRAPPED, _ADVANCED, _REACHED = 0, 1, 2


class RRTConnect:
    def __init__(self,
                 checker: CollisionChecker,
                 q_lower,
                 q_upper,
                 config: RRTConnectConfig | None = None):
        self.checker = checker
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        if np.any(self.q_upper < self.q_lower):
            raise ValueError("关节上限小于下限")
        self.cfg = config or RRTConnectConfig()
        self.n = len(self.q_lower)
        w = self.cfg.weights
        self.weights = None if w is None else np.asarray(w, dtype=float)

    # ------------------------------------------------------------ 基本操作

    def _dist(self, a, b) -> float:
        d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        if self.weights is not None:
            d = d * self.weights
        return float(np.linalg.norm(d))

    def _sample(self, rng) -> np.ndarray:
        lo = self.q_lower + self.cfg.limit_margin
        hi = self.q_upper - self.cfg.limit_margin
        lo = np.minimum(lo, hi)
        return rng.uniform(lo, hi)

    def _steer(self, tree: _Tree, from_idx: int, q_target: np.ndarray) -> tuple[int, int]:
        """从树上指定节点朝 q_target 走一步（最多 step_size）。

        返回 (状态, 结果节点索引)。ADVANCED 时新节点到目标的距离
        **恰好**减少 step_size —— 因为 q_new 落在 q_near→q_target 的线段上，
        而加权欧氏距离是范数，线段上的点满足 |q_new−q_target| = dist − step。
        这保证 _connect 的贪心循环必然在 ⌈dist/step⌉ 步内终止。
        """
        q_near = tree.node(from_idx)
        dist = self._dist(q_near, q_target)
        if dist < 1e-12:
            return _REACHED, from_idx
        ratio = min(self.cfg.step_size / dist, 1.0)
        if ratio >= 1.0:
            q_new = q_target.copy()
        else:
            q_new = q_near + ratio * (q_target - q_near)
        # 起点在限位盒内、目标点也在限位盒内，盒是凸集，
        # 因此连线不会越界；这里 clip 只是防浮点毛刺。
        q_new = np.clip(q_new, self.q_lower, self.q_upper)
        if not self.checker.segment_valid(q_near, q_new, check_start=False):
            return _TRAPPED, from_idx
        new_i = tree.add(q_new, from_idx)
        return (_REACHED if ratio >= 1.0 else _ADVANCED), new_i

    def _extend(self, tree: _Tree, q_target: np.ndarray) -> tuple[int, int]:
        """从树上离 q_target 最近的节点走一步。"""
        return self._steer(tree, tree.nearest(q_target, self.weights), q_target)

    def _connect(self, tree: _Tree, q_target: np.ndarray) -> tuple[int, int]:
        """朝 q_target 贪心地一直走，直到到达或被挡住。"""
        status, idx = self._extend(tree, q_target)
        while status == _ADVANCED:
            status, idx = self._steer(tree, idx, q_target)
        return status, idx

    # ------------------------------------------------------------ 主流程

    def plan(self, q_start, q_goal, seed: int = 0) -> PlanResult:
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        c0 = self.checker.n_config_checks

        def done(res: PlanResult) -> PlanResult:
            res.collision_checks = self.checker.n_config_checks - c0
            return res

        if np.any(q_start < self.q_lower) or np.any(q_start > self.q_upper):
            return done(PlanResult(False, reason="起点越关节限位"))
        if np.any(q_goal < self.q_lower) or np.any(q_goal > self.q_upper):
            return done(PlanResult(False, reason="终点越关节限位"))
        if not self.checker.is_valid(q_start):
            return done(PlanResult(False, reason="起点本身发生碰撞"))
        if not self.checker.is_valid(q_goal):
            return done(PlanResult(False, reason="终点本身发生碰撞"))

        if self.cfg.try_direct and self.checker.segment_valid(q_start, q_goal):
            return done(PlanResult(True, [q_start.copy(), q_goal.copy()],
                                   iterations=0, nodes=2, reason="直线可达"))

        rng = np.random.default_rng(seed)
        tree_a, tree_b = _Tree(q_start), _Tree(q_goal)
        a_is_start = True

        for it in range(1, self.cfg.max_iters + 1):
            q_rand = self._sample(rng)
            status, idx_a = self._extend(tree_a, q_rand)
            if status != _TRAPPED:
                q_new = tree_a.node(idx_a).copy()
                status_b, idx_b = self._connect(tree_b, q_new)
                if status_b == _REACHED:
                    branch_a = tree_a.branch(idx_a)          # 根A → 连接点
                    branch_b = tree_b.branch(idx_b)          # 根B → 连接点
                    branch_b.reverse()                       # 连接点 → 根B
                    path = branch_a + branch_b[1:]           # 去掉重复的连接点
                    if not a_is_start:
                        path.reverse()
                    return done(PlanResult(
                        True, path, iterations=it,
                        nodes=tree_a.size + tree_b.size, reason="双树连通"))
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start

        return done(PlanResult(False, iterations=self.cfg.max_iters,
                               nodes=tree_a.size + tree_b.size,
                               reason="达到迭代上限仍未连通"))


# ------------------------------------------------------------------ 平滑


def prune_waypoints(path, checker: CollisionChecker) -> list[np.ndarray]:
    """确定性顶点剪枝：从每个点尽量往后连，跳过中间所有点。

    路径长度只会变短或不变——用一条直线替换一段折线，
    由三角不等式，直线长度不超过折线长度。
    """
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) < 3:
        return [p.copy() for p in path]
    out = [path[0].copy()]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not checker.segment_valid(path[i], path[j],
                                                      check_start=False,
                                                      check_end=False):
            j -= 1
        out.append(path[j].copy())
        i = j
    return out


def shortcut_smooth(path,
                    checker: CollisionChecker,
                    iters: int = 200,
                    seed: int = 0,
                    weights=None,
                    min_gain: float = 1e-9) -> list[np.ndarray]:
    """短路平滑（partial shortcut）。

    每次在路径上随机取两个**弧长位置**（可以落在线段中间，不限于顶点），
    若两点之间的直线无碰撞且确实更短，就用直线替换中间那一段。

    随机取弧长位置而不是随机取顶点，是因为顶点是 RRT 采样的偶然产物，
    最优的"抄近道"起止点通常不在顶点上。

    每一步都要求 **严格变短** 才接受，所以路径长度关于迭代次数单调不增；
    加上端点始终保留、替换段总是先做碰撞检查，平滑不会把合法路径改成非法路径
    （在同一套碰撞检测分辨率下）。
    """
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) < 3:
        return [p.copy() for p in path]
    rng = np.random.default_rng(seed)

    path = prune_waypoints(path, checker)
    for _ in range(iters):
        if len(path) < 3:
            break
        cum = cumulative_length(path, weights)
        total = cum[-1]
        if total <= 0.0:
            break
        s1, s2 = np.sort(rng.uniform(0.0, total, size=2))
        if s2 - s1 < 1e-9:
            continue
        q1 = path_interpolate(path, s1, weights)
        q2 = path_interpolate(path, s2, weights)

        # 保留 s1 之前的所有顶点和 s2 之后的所有顶点，中间整段换成 q1→q2 直线。
        # cum[0] = 0 <= s1 保证 head 至少含起点；cum[-1] = total >= s2 保证 tail 至少含终点。
        head = [path[k] for k in range(len(path)) if cum[k] <= s1 + 1e-12]
        tail = [path[k] for k in range(len(path)) if cum[k] >= s2 - 1e-12]
        candidate = head + [q1, q2] + tail

        if path_length(candidate, weights) >= total - min_gain:
            continue
        # q1、q2 是线段内部的新点，规划期的采样未必落在它们身上，必须一并检查
        if not checker.segment_valid(q1, q2):
            continue
        path = candidate

    return prune_waypoints(path, checker)
