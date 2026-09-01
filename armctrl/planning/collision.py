"""关节空间碰撞检测。

为什么用有符号距离而不是"有没有接触点"
--------------------------------------
MuJoCo 的 `mj_collision` 只在几何体已经接触（或进入 margin）时才生成接触点，
返回的是布尔式信息。规划需要的是**连续量**：

    mujoco.mj_geomDistance(m, d, g1, g2, distmax, fromto)
        > 0   分离，数值就是最近间隙
        = 0   刚好接触
        < 0   已穿透，数值是穿透深度
        超过 distmax 时被截断为 distmax

有了连续间隙就能：
  1. 直接施加安全裕度（clearance > margin 才算合法），不必去改模型的 margin 字段；
  2. 报告"整条轨迹的最小间隙"这种可核对的指标；
  3. 用解析几何独立验证（球-球、球-胶囊的距离都有闭式解）。

已核对：球-球在 distmax 之内时 mj_geomDistance 与解析值逐位相同（差 <3e-17），
超出 distmax 时返回 distmax 本身（截断，不是真实距离）——所以 distmax
必须显著大于安全裕度，否则"间隙"会被截断成常数而失去分辨能力。

离散化的诚实说明
----------------
`segment_valid` 是在两个构型之间**按分辨率采样**检查，不是连续保证。
采样点之间理论上仍可能钻过极薄的障碍物。降低 resolution 或增大 safety_margin
可以降低这个风险，但不能消除它。本模块不声称提供连续碰撞检测。
"""

from __future__ import annotations

from collections import deque

import numpy as np
import mujoco


class CollisionChecker:
    """把 (关节角 → 是否合法) 这件事封装起来。

    参数
    ----
    model           已编译的 MjModel（应当已经包含障碍物）
    qpos_idx        被规划关节在 qpos 中的地址
    robot_geoms     机械臂参与碰撞的几何体 id
    obstacle_geoms  环境障碍物几何体 id
    self_pairs      需要检查的自碰撞几何体对；空表示不查自碰撞
    safety_margin   安全裕度 (m)。间隙必须严格大于它才算合法
    distmax         距离查询的截断值，必须 > safety_margin
    resolution      段检查的关节空间采样步长 (rad)，按无穷范数

    这里持有**自己的 MjData**，不会污染仿真用的 data。
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 qpos_idx,
                 robot_geoms,
                 obstacle_geoms=(),
                 self_pairs=(),
                 safety_margin: float = 0.0,
                 self_margin: float | None = None,
                 distmax: float | None = None,
                 resolution: float = 0.05):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_idx = np.asarray(qpos_idx, dtype=int)
        self.robot_geoms = list(int(g) for g in robot_geoms)
        self.obstacle_geoms = list(int(g) for g in obstacle_geoms)
        self.self_pairs = [(int(a), int(b)) for a, b in self_pairs]
        self.safety_margin = float(safety_margin)
        # 自碰撞用独立裕度：连杆的碰撞几何是凸包，相邻连杆之间本来就贴得很近
        # （Panda 在常用构型下自身最小间隙只有约 0.022 m），
        # 若和障碍物共用同一个裕度，大半工作空间会被误判成非法。
        self.self_margin = (self.safety_margin if self_margin is None
                            else float(self_margin))
        if distmax is None:
            distmax = max(4.0 * max(self.safety_margin, self.self_margin), 0.10)
        if distmax <= self.safety_margin or distmax <= self.self_margin:
            raise ValueError("distmax 必须大于安全裕度，否则间隙会被截断成常数")
        self.distmax = float(distmax)
        self.resolution = float(resolution)

        self._env_pairs = [(r, o) for r in self.robot_geoms for o in self.obstacle_geoms]
        if not self._env_pairs and not self.self_pairs:
            raise ValueError("没有任何需要检查的几何体对")
        self._fromto = np.zeros(6)
        self.n_config_checks = 0

    # -------------------------------------------------------------- 单构型

    def _scan(self, q, stop_early: bool) -> tuple[float, float]:
        """一次运动学更新，扫描全部几何体对。

        返回 (环境最小间隙, 自碰撞最小间隙)，均被 distmax 截断。
        stop_early=True 时，一旦发现已经违反对应裕度就立即返回，用于加速合法性判定。
        """
        q = np.asarray(q, dtype=float)
        d = self.data
        d.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, d)
        self.n_config_checks += 1

        best_env = best_self = self.distmax
        for g1, g2 in self._env_pairs:
            dist = mujoco.mj_geomDistance(self.model, d, g1, g2, self.distmax, self._fromto)
            if dist < best_env:
                best_env = dist
                if stop_early and best_env <= self.safety_margin:
                    return best_env, best_self
        for g1, g2 in self.self_pairs:
            dist = mujoco.mj_geomDistance(self.model, d, g1, g2, self.distmax, self._fromto)
            if dist < best_self:
                best_self = dist
                if stop_early and best_self <= self.self_margin:
                    return best_env, best_self
        return best_env, best_self

    def clearances(self, q) -> tuple[float, float]:
        """返回 (与环境障碍物的最小间隙, 自碰撞最小间隙)，均被 distmax 截断。"""
        return self._scan(q, stop_early=False)

    def clearance(self, q) -> float:
        """该构型下所有被检查几何体对的最小有符号距离。

        注意返回值被 distmax 截断：若真实最小间隙大于 distmax，返回 distmax。
        判定合法性时这没有影响（distmax > 各裕度），
        但把它当作"真实间隙"报告时必须记住这个上限。
        """
        return float(min(self._scan(q, stop_early=False)))

    def env_clearance(self, q) -> float:
        """只看机械臂与环境障碍物之间的最小间隙。"""
        return float(self._scan(q, stop_early=False)[0])

    def self_clearance(self, q) -> float:
        """只看机械臂自身连杆之间的最小间隙。"""
        return float(self._scan(q, stop_early=False)[1])

    def is_valid(self, q) -> bool:
        env, own = self._scan(q, stop_early=True)
        return env > self.safety_margin and own > self.self_margin

    # -------------------------------------------------------------- 段检查

    def n_steps(self, qa, qb) -> int:
        """该段需要切成几份。至少 1 份。"""
        delta = np.asarray(qb, dtype=float) - np.asarray(qa, dtype=float)
        return max(int(np.ceil(np.max(np.abs(delta)) / self.resolution)), 1)

    def segment_valid(self, qa, qb, check_start: bool = True,
                      check_end: bool = True) -> bool:
        """检查 qa→qb 直线段（关节空间）上的采样点是否都合法。

        内部点按**二分顺序**检查（先中点，再两侧中点……）。
        与从头到尾顺序扫描相比，采样点集合完全相同，但一段的碰撞通常
        出现在中部，二分顺序能更早返回 False，减少无谓的距离查询。
        """
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)
        if check_start and not self.is_valid(qa):
            return False
        if check_end and not self.is_valid(qb):
            return False

        n = self.n_steps(qa, qb)
        if n <= 1:
            return True
        delta = qb - qa
        queue = deque([(0, n)])
        while queue:
            lo, hi = queue.popleft()
            if hi - lo <= 1:
                continue
            mid = (lo + hi) // 2
            if not self.is_valid(qa + (mid / n) * delta):
                return False
            queue.append((lo, mid))
            queue.append((mid, hi))
        return True

    def path_clearances(self, path, samples_per_rad: float = 200.0) -> tuple[float, float]:
        """整条折线路径的最小间隙，用比规划分辨率细得多的采样独立复核。

        返回 (环境最小间隙, 自碰撞最小间隙)。
        """
        path = [np.asarray(p, dtype=float) for p in path]
        worst_env = worst_self = self.distmax
        for a, b in zip(path[:-1], path[1:]):
            L = float(np.max(np.abs(b - a)))
            n = max(int(np.ceil(L * samples_per_rad)), 1)
            for k in range(n + 1):
                e, s = self._scan(a + (k / n) * (b - a), stop_early=False)
                worst_env, worst_self = min(worst_env, e), min(worst_self, s)
        return float(worst_env), float(worst_self)

    def path_clearance(self, path, samples_per_rad: float = 200.0) -> float:
        """整条折线路径上所有几何体对的最小间隙。"""
        return float(min(self.path_clearances(path, samples_per_rad)))
