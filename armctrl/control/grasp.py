"""抓取控制：夹持力、滑移检测、夹持力自适应，以及抓取位姿生成。

这一层解决的问题
--------------
`core/gripper.py` 回答的是"怎么让夹爪出 F 牛顿的力"。本模块回答的是
**"F 该是多少"**，以及"力不够时怎么知道、怎么补"。

防滑最小夹持力（自己推，不引注释）
------------------------------
一个质量 m 的工件被两个**竖直**夹持面夹住，每面法向力 N（水平方向），
库仑摩擦系数 μ。工件重量由**摩擦力**托住：

    单面最大静摩擦    μ·N
    两面合计          2·μ·N
    需要托住的载荷    m·(g + a)      a 为竖直方向的附加加速度

不滑的条件是

    2·μ·N  ≥  m·(g + a)      ⟹      **N ≥ m·(g + a) / (2μ)**

推广到 n 个接触面就是 N ≥ m(g+a)/(n·μ)。

这条公式成立的前提（必须一起说，否则会被用错）：

1. **夹持面竖直**，法向水平。若夹持面是水平的（比如从上下夹），
   重量由法向力直接承担，摩擦根本不参与，这条公式不适用。
2. **库仑摩擦、刚体、单一 μ**。真实接触有黏滞、有接触面积效应，
   μ 还随法向力变化。
3. **只考虑平移，不考虑绕夹持轴的转动**。真实工件还要靠**扭转摩擦**
   抵抗力矩，偏心抓取时这一项可能先失效——工件会在指间转动而不是掉下去。
4. **两指法向力相等**，这对平行夹爪的内力是成立的。
5. **加速度已知**。搬运中加速会显著抬高需求：1 g 的向上加速度直接让需求翻倍。

μ 该取哪个值
-----------
**必须现场从 `contact.friction[0]` 读，不能只用工件自己的 μ。**
实测确认 MuJoCo 把两个几何体的滑动摩擦按**逐元素取最大值**合成
（μ=(0.3,0.9)→0.9；(1.2,0.2)→1.2）。指垫的 μ 比工件大时，
实际生效的是指垫的值，拿工件的 μ 去算理论值会算出偏大的需求。

真机上你不知道 m 和 μ
--------------------
所以 `min_normal_force` 只能用来做**事后对照**（仿真里我们有上帝视角），
不能直接拿来当控制律。真机的做法是 `AdaptiveGripForce`：
先给一个保守值，检测到滑移就快速加力，长时间不滑就缓慢减力，
最终稳定在"刚好不滑"的边界上方。人手拿易碎品用的也是这个策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco


# ==================================================================== 防滑判据


def min_normal_force(mass: float, mu: float, gravity: float = 9.81,
                     accel: float = 0.0, n_contacts: int = 2,
                     safety: float = 1.0) -> float:
    """防滑所需的**单面**最小法向力 N ≥ m·(g+a) / (n·μ)。

    推导见模块 docstring。`safety` 是安全系数，工程上一般取 1.5–2，
    这里默认 1.0 以便和理论值直接对照——把安全系数混进"理论值"里，
    对照就失去意义了。

    mu 必须是**实际生效**的摩擦系数（见模块说明），不是工件标称值。
    """
    mu = float(mu)
    if mu <= 0.0:
        raise ValueError("摩擦系数必须为正；μ→0 时所需夹持力发散，物理上就是夹不住")
    if n_contacts < 1:
        raise ValueError("接触面数至少为 1")
    load = float(mass) * (float(gravity) + float(accel))
    return float(safety) * load / (float(n_contacts) * mu)


def effective_friction(model: mujoco.MjModel, data: mujoco.MjData,
                       contact_ids) -> float:
    """从实际接触里读出生效的滑动摩擦系数。

    多个接触时取最小值：**最容易滑的那个面决定整体是否打滑**，
    取平均或最大都会高估抓取裕度。没有接触时返回 0。
    """
    mus = [float(data.contact[i].friction[0]) for i in contact_ids]
    return min(mus) if mus else 0.0


# ==================================================================== 滑移检测


@dataclass
class SlipReading:
    """一次滑移检测的完整结果，便于遥测和事后复盘。"""

    margin: float = 0.0          # 摩擦锥余量 |f_t| / (μ·f_n)，→1 表示濒临滑移
    slip_speed: float = 0.0      # 接触点相对切向速度 (m/s)
    normal_force: float = 0.0    # **全部**接触点的法向力之和 (N)
    tangential_force: float = 0.0
    n_contacts: int = 0
    #: **单个夹持面**的法向力 (N)，两指取较小者。
    #: 防滑公式 N ≥ mg/(2μ) 里的 N 指的就是这个量，
    #: 不是 normal_force，也不是"总力除以接触点个数"——
    #: MuJoCo 的一个指垫会产生多个接触点（方形垫最多 4 个），
    #: 除以点数得到的是单点力，比真正的面法向力小好几倍，
    #: 拿它去和理论值比会得出"力远小于理论值却没滑"的假矛盾。
    face_force: float = 0.0
    incipient: bool = False      # 摩擦锥判据：即将滑
    sliding: bool = False        # 速度判据：已经在滑


class SlipDetector:
    """两个互补的滑移判据。

    **摩擦锥余量**（预测性）：库仑定律说切向力上限是 μ·f_n。
    比值 |f_t|/(μ·f_n) 趋近 1 就是濒临滑移。它能在真滑之前报警，
    但需要知道 μ 和接触力——真机上要力/触觉传感器，且 μ 通常不准。

    **相对切向速度**（反应性）：工件真的相对指垫动起来了。
    它不需要知道 μ，但只能在滑移**已经发生**之后才报。
    真实系统里靠触觉传感器的微振动或视觉来近似。

    两个都给出来，是因为它们的工程含义不同：预测性判据用于提前加力，
    反应性判据用于确认"确实滑了"。只用其中一个都会有明显短板。
    """

    def __init__(self, gripper, object_body: str,
                 margin_threshold: float = 0.85,
                 speed_threshold: float = 2.0e-3,
                 min_normal_for_margin: float = 0.05):
        self.gripper = gripper
        self.object_body = object_body
        self.margin_threshold = float(margin_threshold)
        self.speed_threshold = float(speed_threshold)
        #: 摩擦锥余量生效所需的最小法向力 (N)，防止接触刚建立时误判滑移
        self.min_normal_for_margin = float(min_normal_for_margin)
        m = gripper.model
        self.obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if self.obj_bid < 0:
            raise ValueError(f"工件 body 不存在: {object_body}")

    def _relative_tangential_speed(self, cid: int) -> float:
        """接触点处两侧几何体的相对**切向**速度大小。

        用 `mj_objectVelocity` 分别取两个 body 在世界系的空间速度，
        换算到接触点，再投影到接触切平面。接触系第 0 轴是法向，
        1/2 轴张成切平面。
        """
        m, d = self.gripper.model, self.gripper.data
        c = d.contact[cid]
        frame = np.array(c.frame).reshape(3, 3)      # 行向量：法向, 切向1, 切向2
        b1 = int(m.geom_bodyid[int(c.geom1)])
        b2 = int(m.geom_bodyid[int(c.geom2)])

        def point_velocity(bid: int) -> np.ndarray:
            vel = np.zeros(6)
            # flg_local=0 → 世界系；返回 [角速度(3); 线速度(3)]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
            w, v = vel[0:3], vel[3:6]
            r = np.array(c.pos) - d.xipos[bid]       # 接触点相对该 body 质心
            return v + np.cross(w, r)

        dv = point_velocity(b1) - point_velocity(b2)
        return float(np.linalg.norm(frame[1:3] @ dv))

    def read(self) -> SlipReading:
        m, d = self.gripper.model, self.gripper.data
        ids = self.gripper.contacts_with(self.object_body)
        if not ids:
            return SlipReading()

        fn = ft = 0.0
        speed = 0.0
        for i in ids:
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            fn += abs(float(f6[0]))
            ft += float(np.linalg.norm(f6[1:3]))
            speed = max(speed, self._relative_tangential_speed(i))

        # 单面法向力交给 gripper 算：它按手指 body 分组求和再取两指较小者，
        # 这才是防滑公式里的 N。
        face = float(self.gripper.contact_force(self.object_body))
        mu = effective_friction(m, d, ids)
        # 法向力太小时余量没有意义：|f_t|/(μ·f_n) 在 f_n→0 时会趋于 1，
        # 被当成"濒临滑移"，自适应律就会在**接触刚建立、还没夹上**的时候
        # 一路把力加到上限——典型的绕卷（wind-up）。
        # 所以设一个下限：法向力低于它就认为"还没真正夹住"，不给滑移判据。
        # 阈值取得很小（0.05 N），只用来排除数值噪声，不会掩盖真实的轻夹持。
        if mu <= 0.0 or fn <= self.min_normal_for_margin:
            return SlipReading(normal_force=fn, tangential_force=ft,
                               n_contacts=len(ids), face_force=face,
                               slip_speed=speed,
                               sliding=(speed >= self.speed_threshold))
        margin = float(ft / (mu * fn))
        return SlipReading(
            margin=min(margin, 10.0), slip_speed=speed,
            normal_force=fn, tangential_force=ft, n_contacts=len(ids),
            face_force=face,
            incipient=(margin >= self.margin_threshold),
            sliding=(speed >= self.speed_threshold))


# ==================================================================== 力自适应


@dataclass
class AdaptiveGripConfig:
    """滑移自适应的整定参数。

    rise    检测到滑移时的加力速率 (N/s)。要快——滑移一旦开始，
            接触面从静摩擦切到动摩擦，摩擦力反而下降，不快点补会一路滑到掉。
    decay   未滑时的减力速率 (N/s)。要慢——目的是**试探**最小夹持力，
            减太快会在"滑一下、补一下"之间反复横跳，把工件磨坏。
    f_min   力的下限，防止减到零把工件放掉。
    f_max   力的上限，代表夹爪能力或工件可承受的挤压。
    """

    rise: float = 60.0
    decay: float = 2.0
    f_min: float = 1.0
    f_max: float = 80.0


class AdaptiveGripForce:
    """滑移驱动的夹持力自适应。

    为什么不能直接用 `min_normal_force`
    --------------------------------
    那条公式要 m 和 μ。真机上工件是什么、表面多滑，事先都不知道；
    就算知道标称值，沾了油污 μ 就变了。所以真实策略是**闭环试探**：

        滑了      → 快速加力（rise）
        没滑      → 缓慢减力（decay），一直试探到边界

    稳态时力会停在"刚好不滑"的上方一点点，并小幅振荡。
    这正是人手拿纸杯的策略：先轻轻捏，感觉要滑就立刻加力，
    然后再慢慢放松到不掉为止。

    仿真里我们有上帝视角，可以把 `min_normal_force` 算出来做**对照**：
    自适应收敛值应当略高于它。两者的差就是这套策略付出的"保守代价"。
    """

    def __init__(self, force0: float = 10.0,
                 config: AdaptiveGripConfig | None = None):
        self.cfg = config or AdaptiveGripConfig()
        self.reset(force0)

    def reset(self, force0: float = 10.0) -> None:
        self.force = float(np.clip(force0, self.cfg.f_min, self.cfg.f_max))
        self.n_slip_events = 0
        self._was_slipping = False

    def update(self, reading: SlipReading, dt: float) -> float:
        """按一次滑移读数推进夹持力，返回新的夹持力指令 (N)。

        判据用"濒临滑移 **或** 已在滑"的并集：只用反应性判据会等到真滑了才动，
        只用预测性判据在 μ 估计不准时会失灵。
        """
        slipping = bool(reading.incipient or reading.sliding)
        if slipping and not self._was_slipping:
            self.n_slip_events += 1
        self._was_slipping = slipping

        rate = self.cfg.rise if slipping else -self.cfg.decay
        self.force = float(np.clip(self.force + rate * float(dt),
                                   self.cfg.f_min, self.cfg.f_max))
        return self.force


# ==================================================================== 抓取位姿


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """点云的主轴与对应的方差（按方差从大到小）。

    做法是对去均值后的点做 SVD，等价于协方差矩阵的特征分解，
    但数值上更稳（不用显式构造协方差矩阵，避免条件数平方）。

    返回 (axes, variances)：axes 的**列**是主轴单位矢量。
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points 必须是 (N,3)")
    if P.shape[0] < 3:
        raise ValueError("至少需要 3 个点才能定出主轴")
    Q = P - P.mean(axis=0)
    _, s, Vt = np.linalg.svd(Q, full_matrices=False)
    var = (s ** 2) / max(P.shape[0] - 1, 1)
    return Vt.T, var


def grasp_frame_from_points(points: np.ndarray,
                            approach: np.ndarray = np.array([0.0, 0.0, -1.0])
                            ) -> tuple[np.ndarray, np.ndarray, float]:
    """由点云给出抓取位姿（简化版）。

    规则：**夹爪的开合轴取物体的最短主轴，与最长主轴垂直。**

    为什么是最短轴：
      • 开口宽度需求最小，夹爪够得着；
      • 两指沿最短方向对夹，抓取宽度小、力臂短，抵抗力矩的能力相对更强；
      • 沿最长轴去夹意味着要张开到物体全长，通常根本张不开。

    返回 (p_center, R, width)：
      p_center  抓取中心 = 点云质心
      R         抓取坐标系，列为 [开合轴, 副轴, 接近轴]
      width     沿开合轴的物体宽度（点云在该轴上的投影范围）

    **这是简化版，不是抓取规划**。它只用二阶矩，没有考虑：
    可达性与关节限位、与环境的碰撞、力封闭（force closure）判定、
    物体凹陷或把手这类局部几何、质心偏置带来的力矩。
    真实抓取位姿生成要么解析求解力封闭，要么用学习的方法。
    """
    P = np.asarray(points, dtype=float)
    axes, var = principal_axes(P)
    close_axis = axes[:, 2]                       # 方差最小 = 最短主轴
    approach = np.asarray(approach, dtype=float).reshape(3)
    n = np.linalg.norm(approach)
    if n < 1e-12:
        raise ValueError("接近方向不能是零矢量")
    approach = approach / n

    # 接近轴要与开合轴正交：把接近方向在开合轴上的分量去掉。
    # 两者接近平行时（去掉后几乎为零）说明"从这个方向接近就没法对夹"，
    # 必须报错而不是给一个数值噪声主导的方向。
    a = approach - np.dot(approach, close_axis) * close_axis
    if np.linalg.norm(a) < 1e-6:
        raise ValueError("接近方向与开合轴几乎平行，这个抓取位姿不成立")
    a = a / np.linalg.norm(a)
    second = np.cross(a, close_axis)
    second = second / np.linalg.norm(second)

    R = np.column_stack([close_axis, second, a])
    if np.linalg.det(R) < 0:                      # 保证是右手系旋转矩阵
        R[:, 1] = -R[:, 1]

    proj = P @ close_axis
    width = float(proj.max() - proj.min())
    return P.mean(axis=0), R, width


def constrained_grasp_frame(points: np.ndarray,
                            approach: np.ndarray = np.array([0.0, 0.0, -1.0]),
                            ) -> tuple[np.ndarray, np.ndarray, float]:
    """在**固定的接近方向**下生成抓取位姿。

    与 `grasp_frame_from_points` 的区别，以及为什么需要这个函数
    ---------------------------------------------------------
    那个函数无条件把**最短主轴**当开合轴，接近方向只是被正交化掉。
    物体立着时这没问题；可物体一旦被碰倒，最短主轴很可能变成**竖直**的，
    于是算出来的抓取要求「从侧面水平伸进去对夹」——而工件正躺在桌面上，
    侧面根本进不去，而且沿另一个方向对夹需要张开到工件的长边。
    对 42×42×90 mm 的工件，长边 90 mm 超过夹爪 80 mm 的行程，
    表现就是**「一倒下就再也抓不起来」**。

    本函数把接近方向钉死（例如自上而下），只在**与接近方向垂直的平面内**
    挑开合轴，所以结果永远是一个可执行的俯冲抓取。
    挑的规则仍然是"取更短的那个方向"，理由与原函数相同：开口需求最小。

    返回 (p_center, R, width)，含义与 `grasp_frame_from_points` 一致：
    R 的列为 [开合轴, 副轴, 接近轴]，width 是沿开合轴的物体宽度。

    **仍然是简化版**：只用二阶矩，不判力封闭、不查可达性与碰撞。
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points 必须是 (N,3)")
    a = np.asarray(approach, dtype=float).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("接近方向不能是零矢量")
    a = a / n

    # 在垂直于接近方向的平面里建一组正交基
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, a))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, a) * a
    u = u / np.linalg.norm(u)
    w = np.cross(a, u)

    # 把点云投到该平面，做 2×2 主轴分解，取**投影范围更小**的那个方向当开合轴。
    # 注意判据用的是"投影范围"（真实开口需求）而不是方差：
    # 方差大小与范围在均匀分布下同序，但对角点集合这类离散点云，
    # 直接比范围才是我们真正关心的"夹爪要张多大"。
    Q = P - P.mean(axis=0)
    C = np.column_stack([Q @ u, Q @ w])
    _, _, Vt = np.linalg.svd(C - C.mean(axis=0), full_matrices=False)
    cands = []
    for k in range(2):
        d2 = Vt[k]
        axis = d2[0] * u + d2[1] * w
        axis = axis / np.linalg.norm(axis)
        proj = P @ axis
        cands.append((float(proj.max() - proj.min()), axis))
    width, close_axis = min(cands, key=lambda c: c[0])

    second = np.cross(a, close_axis)
    second = second / np.linalg.norm(second)
    R = np.column_stack([close_axis, second, a])
    if np.linalg.det(R) < 0:
        R[:, 1] = -R[:, 1]
    return P.mean(axis=0), R, width


def box_corner_points(half_extents, pos=(0.0, 0.0, 0.0),
                      rot: np.ndarray | None = None) -> np.ndarray:
    """长方体的 8 个角点，用作 `grasp_frame_from_points` 的输入。

    只取角点而不是均匀采样表面：对长方体来说两者的主轴方向完全一致
    （角点集合关于三个轴对称），但角点少得多。
    """
    h = np.asarray(half_extents, dtype=float).reshape(3)
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=float)
    pts = signs * h
    if rot is not None:
        pts = pts @ np.asarray(rot, dtype=float).T
    return pts + np.asarray(pos, dtype=float).reshape(3)
