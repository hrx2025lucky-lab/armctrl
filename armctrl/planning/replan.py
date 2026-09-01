"""动态避障：执行途中监测碰撞并在线重规划。

与静态规划的区别
--------------
`rrt.py` 里的规划是**一次性**的：规划完就照着走，中途障碍物动了也不管。
真实场景（人机协作、移动的料架、另一台机器人）里这不成立。

在线重规划要回答三个问题：

1. **什么时候该重规划？** —— 太敏感会一直在重规划、机器人抖个不停；
   太迟钝就撞上了。
2. **从哪儿开始规划？** —— ⭐ 不能从「当前位置」规划。规划本身要花时间，
   等算完机器人已经跑到别处了。必须从**规划完成时机器人会在的位置**开始。
3. **算不完怎么办？** —— 必须有兜底：减速、停住、或沿旧路径继续。

本模块的做法
----------
**前瞻窗口 + 迟滞 + 预算**。

* **前瞻窗口**：只检查未来 ``horizon`` 秒的轨迹段。检查全程既浪费又没意义
  （远处的障碍物还会再动）。
* **迟滞**：连续 ``trigger_count`` 次检测到碰撞才真的触发重规划。
  单次检测受采样相位影响很大，一次就触发会**误报**。
* **冷却期**：刚重规划完的 ``cooldown`` 秒内不再触发，避免抖动。
* **预算**：累计重规划次数超过 ``max_replans`` 就放弃并报警——
  ⭐ 真机上必须有这一条，否则会陷入「规划-失败-再规划」的死循环。

⚠️ 本实现的边界（不要说成等价）
-----------------------------
* **障碍物的未来位置假设已知。**这里直接把障碍物的运动规律告诉检测器。
  真机上要靠感知 + 预测（卡尔曼/等速外推/学习模型），
  ⭐ **预测误差本身就是这类系统最主要的失效源**，本实现完全没有这一层。
* **规划耗时按名义值估计**，没有做真正的实时性保证（无 anytime 算法、
  无硬实时调度）。工业上会用 anytime RRT* 或轨迹优化（如 CHOMP）
  在固定时间预算内返回当前最好解。
* **重规划后是硬切换**，只保证起点状态连续（位置+速度），
  没有做加加速度连续的过渡段。
* 只考虑**平移**的障碍物，且用球体包络。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MovingSphere:
    """做匀速直线往复运动的球形障碍物（世界系）。

    真机上障碍物的未来位置是**预测**出来的、带误差的；
    这里给的是解析规律，属于「上帝视角」，见模块 docstring 的边界说明。
    """

    center: np.ndarray
    radius: float
    velocity: np.ndarray
    span: float = 0.6

    def position_at(self, t: float) -> np.ndarray:
        """t 时刻的球心。用三角波在 ±span/2 之间往复。"""
        v = np.asarray(self.velocity, dtype=float)
        speed = float(np.linalg.norm(v))
        if speed < 1e-12 or self.span <= 0:
            return np.asarray(self.center, dtype=float).copy()
        period = 2.0 * self.span / speed
        phase = (t % period) / period                       # 0..1
        offset = (4.0 * phase - 1.0) if phase < 0.5 else (3.0 - 4.0 * phase)
        return np.asarray(self.center, float) + v / speed * offset * self.span / 2.0


@dataclass
class ReplanEvent:
    """一次重规划的记录，用于事后复盘与界面显示。"""

    time: float
    reason: str
    lead_time: float
    success: bool


@dataclass
class ReplanSupervisor:
    """执行途中的碰撞监测与重规划触发器。

    参数
    ----
    horizon        前瞻窗口（秒）
    samples        窗口内的检查采样点数
    clearance      安全间隙（m）。末端与障碍球面之间要留这么多
    trigger_count  连续多少次检测到威胁才真的触发（迟滞）
    cooldown       触发后的冷却时间（秒）
    max_replans    重规划次数上限，超了就放弃
    plan_time      ⭐ 规划耗时的**名义估计**（秒）。用来决定从哪个未来状态起规划
    """

    horizon: float = 1.2
    samples: int = 24
    clearance: float = 0.06
    trigger_count: int = 3
    cooldown: float = 0.8
    max_replans: int = 8
    plan_time: float = 0.12

    _streak: int = field(default=0, init=False)
    _last_replan: float = field(default=-1e9, init=False)
    events: list[ReplanEvent] = field(default_factory=list, init=False)
    aborted: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._streak = 0
        self._last_replan = -1e9
        self.events.clear()
        self.aborted = False

    # ------------------------------------------------------- 碰撞前瞻

    def min_clearance(self, ee_at, obstacle: MovingSphere, t_now: float,
                      horizon: float | None = None) -> tuple[float, float]:
        """在前瞻窗口内扫描，返回 (最小间隙, 出现该间隙的时刻)。

        ⭐ **障碍物和机器人都按各自的时间演化取值**，不是拿当前障碍物位置
        去比未来的机器人位置——那是很常见的一个错误，会让快速移动的障碍物
        被完全漏检。

        ee_at(t) 给出 t 时刻末端的世界坐标（由轨迹提供）。
        """
        h = self.horizon if horizon is None else float(horizon)
        best, best_t = np.inf, t_now
        for i in range(self.samples):
            t = t_now + h * i / max(self.samples - 1, 1)
            gap = float(np.linalg.norm(ee_at(t) - obstacle.position_at(t))) \
                - obstacle.radius
            if gap < best:
                best, best_t = gap, t
        return best, best_t

    # ------------------------------------------------------- 触发判断

    def should_replan(self, gap: float, t_now: float) -> tuple[bool, str]:
        """给定当前的前瞻最小间隙，判断要不要重规划。

        返回 (是否触发, 原因)。原因串直接显示在界面上，便于理解为什么没触发。
        """
        if self.aborted:
            return False, "已放弃（超出重规划预算）"
        if gap >= self.clearance:
            self._streak = 0
            return False, "前瞻窗口内安全"
        self._streak += 1
        if t_now - self._last_replan < self.cooldown:
            return False, f"冷却中（还差 {self.cooldown - (t_now - self._last_replan):.2f} s）"
        if self._streak < self.trigger_count:
            return False, f"迟滞计数 {self._streak}/{self.trigger_count}"
        return True, "前瞻窗口内将发生碰撞"

    def commit(self, t_now: float, reason: str, success: bool) -> None:
        """登记一次重规划。失败也要登记——预算是按**尝试**算的。"""
        self._last_replan = t_now
        self._streak = 0
        self.events.append(ReplanEvent(t_now, reason, self.plan_time, success))
        if len(self.events) >= self.max_replans:
            self.aborted = True

    # ------------------------------------------------------- 起点选择

    def replan_start_time(self, t_now: float) -> float:
        """⭐ 从**哪个时刻的状态**开始规划。

        不能用当前状态：规划要花 ``plan_time`` 秒，等算完机器人已经跑过去了，
        新轨迹的起点和机器人的真实状态对不上，切换瞬间会出现速度跳变。

        正确做法是**沿旧轨迹外推**到 ``t_now + plan_time``，
        以那一时刻的位置与速度作为新规划的初始状态。
        真机上这叫「规划延迟补偿」，是在线重规划能不能用的关键一环。
        """
        return float(t_now) + self.plan_time

    @property
    def n_replans(self) -> int:
        return len(self.events)

    @property
    def n_failed(self) -> int:
        return sum(1 for e in self.events if not e.success)
