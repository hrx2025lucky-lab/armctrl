"""在线调参 UI 的框架层：参数声明、遥测声明、场景基类。

设计意图
--------
每个场景只描述五件事，UI 完全由这些声明自动生成，不用写前端：

    params    可拖的滑块 / 可选的下拉         → 左侧控制面板
    readouts  当前时刻的数值（含理论对照值）    → 画面下方数字区
    traces    随时间变化的量                  → 实时曲线
    bars      逐关节的控制量与其限幅           → 画面下方条形图
    law       控制律公式 + 每个符号的当前取值   → 右侧原理面板

再加一个运行时钩子：

    overlays()  往三维画面里叠加箭头 / 球 / 坐标架 / 轨迹拖尾

**教学重点是三处"对得上"**：

1. readouts 里的"理论对照"：例如阻抗控制同时显示"实测末端偏移"与"理论 F/K"。
   调滑块时两个数一起变、且始终对得上，才说明你理解的物理和代码里跑的是
   同一件事；对不上就说明有一方错了。
2. law 面板：公式里每个符号旁边就是它此刻的数值。滑块动 → 公式里哪一项变 →
   画面里哪个现象变，三者一眼对应，不用在脑子里做映射。
3. overlays：把控制器"心里想的量"画到画面上。外力、误差矢量、期望位姿本来
   都只是数组里的数字，画出来才能看见控制器在跟什么较劲。

为什么要有 bars
--------------
`‖τ‖` 这种标量只能告诉你"用了多大劲"，看不出是哪个关节在出力、离饱和还有多远。
力矩饱和是真机上最常见的翻车原因之一，必须能一眼看见，所以单列一类声明。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco


def hex_rgba(color: str, alpha: float = 1.0) -> np.ndarray:
    """"#rrggbb" → float32 RGBA。前端和三维画面共用同一套色号，避免两边对不上。"""
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"颜色必须是 #rrggbb 形式，收到 {color!r}")
    return np.array([int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0,
                     int(c[4:6], 16) / 255.0, alpha], dtype=np.float32)


@dataclass
class Manual:
    """手动干预状态：夹爪开合 / 关节扰动力矩 / 运动学摆位。

    为什么单独放一层，而不做成场景的 Param
    ------------------------------------
    场景的 `params` 会被 `Step` 门控（一步只露出该步要调的那几个滑块），
    而这三路是**任何场景、任何步骤都该随时能用**的手动开关，不属于任何控制律，
    所以由 Runner 直接持有，不进 `Scene.values`。

    三路各自的语义（**必须区分清楚，否则会得出错误结论**）
    -----------------------------------------------
    grip
        夹爪开口指令 (m)。走夹爪自己的腱位置伺服，**不经过手臂控制律**。
        None = 跟随场景自己的设定（抓取场景会自己接管夹爪，这时不该插手）。

    dist_joint / dist_tau
        往某个关节注入扰动力矩 (N·m)，走 `data.qfrc_applied`。
        ⚠️ 它**不是电机出力**：`qfrc_applied` 是广义外力，不经过执行器限幅，
        所以它模拟的是"外部有人在掰这个关节"，而不是"把控制量改成这个值"。
        控制量 τ 仍然由控制律算、由 `ctrl` 写入。想看"控制器出多大力"仍看条形图。
        这正是它的教学价值：给一个已知扰动，看**不同控制律各自怎么把它压回去**。

    pose_on / pose_q
        运动学摆位。打开后**不做动力学积分**，直接把 qpos 写成 pose_q。
        ⚠️ 它**不是控制**，是"把机械臂摆成这个姿势看看"，相当于用手搬。
        因此摆位时读数里的力矩、误差都没有物理意义，故意不发场景遥测。
        关掉之后物理从当前构型接着跑——这才是它真正好用的地方：
        摆一个歪姿势，关掉摆位，看控制器怎么把它拉回去。
    """

    grip: float | None = None
    dist_joint: int = 0                  # 0 = 不施加；1..7 = J1..J7
    dist_tau: float = 0.0
    pose_on: bool = False
    pose_q: list[float] = field(default_factory=list)

    def update(self, payload: dict) -> None:
        """按前端送来的字段增量更新。只认已知键，未知键直接报错而不是静默忽略。"""
        for k, v in payload.items():
            if k == "grip":
                self.grip = None if v is None else float(v)
            elif k == "dist_joint":
                self.dist_joint = int(np.clip(int(v), 0, 7))
            elif k == "dist_tau":
                self.dist_tau = float(v)
            elif k == "pose_on":
                self.pose_on = bool(v)
            elif k == "pose_q":
                self.pose_q = [float(x) for x in v]
            else:
                raise KeyError(f"未知的手动干预字段: {k}")

    def joint_torque(self, n: int = 7) -> np.ndarray:
        """展开成长度 n 的关节扰动力矩向量。dist_joint=0 时全零。"""
        tau = np.zeros(n)
        if 1 <= self.dist_joint <= n:
            tau[self.dist_joint - 1] = self.dist_tau
        return tau

    def to_json(self) -> dict:
        return dict(grip=self.grip, dist_joint=self.dist_joint,
                    dist_tau=self.dist_tau, pose_on=self.pose_on,
                    pose_q=list(self.pose_q))


@dataclass
class Param:
    """一个可调参数。

    choices 非空时渲染成下拉框（值为字符串）；否则渲染成滑块。
    restart=True 表示改动后必须重置场景（例如重新规划路径），
    因为这类参数不是控制器里能就地改的增益。
    """

    key: str
    label: str
    lo: float = 0.0
    hi: float = 1.0
    default: float | str = 0.0
    step: float = 0.0
    unit: str = ""
    help: str = ""
    choices: list[str] | None = None
    restart: bool = False
    #: 该参数对应控制律公式里的哪个符号。填了之后前端会在公式里高亮它，
    #: 拖滑块时能直接看出自己在改公式的哪一项。
    symbol: str = ""
    #: 分组标题，把一堆滑块按"控制器增益 / 被控对象 / 激励"分块，
    #: 否则十几个滑块排成一长条，分不清哪些是"我在调的控制器"、
    #: 哪些是"我在改的被控对象"。
    group: str = ""
    #: **调了它会看到什么**。整个面板里最重要的一条说明：
    #: 光说"刚度是每偏移 1 m 产生多大回复力"，人还是不知道该盯着哪儿看。
    #: 必须写成"调大 → 出现什么现象（去哪个面板看）"。
    #: 有些参数在三维画面里**根本看不出来**，那就要诚实写明"画面看不出，
    #: 要看下面的数字或曲线"，否则用户会以为是程序坏了。
    effect: str = ""

    def to_json(self) -> dict:
        d = dict(key=self.key, label=self.label, unit=self.unit, help=self.help,
                 restart=self.restart, default=self.default,
                 symbol=self.symbol, group=self.group, effect=self.effect)
        if self.choices:
            d["kind"] = "choice"
            d["choices"] = self.choices
        else:
            d["kind"] = "slider"
            d["lo"] = self.lo
            d["hi"] = self.hi
            d["step"] = self.step or (self.hi - self.lo) / 200.0
        return d


@dataclass
class Step:
    """一个教学步骤：**先讲清一件事，再动对应的那几个滑块**。

    为什么要有这个
    -------------
    把一个场景的全部讲解和全部滑块一次摊开，信息密度太高，读者不知道从哪下手：
    十几个滑块并排、几百字文案堆在一起，结果是"每个字都认识，合起来不知道该干嘛"。

    拆成步骤之后，界面**一次只显示一步**：这一步的讲解、这一步用到的公式、
    这一步该动的那两三个滑块、这一步该盯的那几个数。读完就动手，动完就看见现象，
    然后进入下一步。密度问题和"先讲原理再调参"的顺序问题一起解决了。

    字段
    ----
    title    步骤名，显示在顶部的步骤条上
    teach    零基础讲解。**一步只讲一件事**，写长了就该拆成两步
    math     LaTeX 公式列表，由 KaTeX 渲染（浏览器端，字体随本仓库自带）
    teach2   公式**之后**的补充讲解。很多时候要「先铺垫 → 给公式 → 再解释公式」，
             只有一个 teach 字段就只能把公式塞在文字中间，排版会很乱
    params   本步开放哪几个参数的 key。其余滑块**隐藏**，避免干扰
    watch    该看什么现象，一条一条列
    focus    本步重点关注的 readout key，会被放大显示在最上面
    """

    title: str
    teach: str = ""
    math: list[str] = field(default_factory=list)
    teach2: str = ""
    params: list[str] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return dict(title=self.title, teach=self.teach, math=list(self.math),
                    teach2=self.teach2, params=list(self.params),
                    watch=list(self.watch), focus=list(self.focus))


@dataclass
class Lesson:
    """一个场景的分层讲解，写给控制理论零基础的人。

    为什么拆成固定的几段
    ------------------
    原来每个场景只有一大段 intro，读者拿到的是一堆并列的句子，
    既不知道从哪读起，也不知道读完该干什么。拆开之后每段只回答一个问题，
    可以按顺序读下来：

        problem  我为什么需要这个东西？（先讲工程上的痛点，不出现公式）
        idea     它的基本想法是什么？（一个生活里的比喻）
        how      它具体怎么算？（把比喻逐项对回公式）
        watch    我该动哪个滑块、盯哪儿看？（操作 → 预期现象，一条一条列）
        verify   我怎么知道它算对了？（自己能复核的判据）
        caveat   哪里容易误解、这个演示的边界在哪
    """

    problem: str = ""
    idea: str = ""
    how: str = ""
    watch: list[str] = field(default_factory=list)
    verify: list[str] = field(default_factory=list)
    caveat: str = ""

    def to_json(self) -> dict:
        return dict(problem=self.problem, idea=self.idea, how=self.how,
                    watch=list(self.watch), verify=list(self.verify),
                    caveat=self.caveat)


#: 一个读数的**来源**。这不是装饰，是把工作纪律第 3 条（oracle 必须独立）
#: 变成界面上看得见的东西。四类的可信度完全不同：
#:
#:   measured  实测量。从仿真里读出来的物理量。
#:   theory    **独立理论值**。用解析式现算，**不读被测实现的任何中间量**。
#:             只有它才能拿来验证实现对不对。
#:   truth     仿真真值（上帝视角）。可信，但**真机上没有这一路**，
#:             所以用它得出的结论不能外推到真机。
#:   internal  被测实现自己报的量（控制器内部状态、滤波器自己的协方差…）。
#:             **绝不能用来证明该实现是对的**——那是自证。
#:             它有用，但用途是"看实现自己怎么想"，不是"看它对不对"。
READOUT_SOURCES = ("measured", "theory", "truth", "internal")


@dataclass
class Readout:
    """画面下方的一个数字。

    source        见 READOUT_SOURCES。前端按来源分色，并在图例里写明每类的含义。
                  把"独立算出来的理论值"和"实现自己报的数"混成一个颜色，
                  等于把纪律第 3 条的区分抹掉了。
    compare_with  填另一个 readout 的 key，前端自动显示两者的相对偏差，
                  省得人肉对比「实测 12.3 / 理论 12.1 到底差多少」。
    """

    key: str
    label: str
    unit: str = ""
    digits: int = 3
    theory: bool = False          # 兼容旧写法，等价于 source="theory"
    compare_with: str = ""
    help: str = ""
    source: str = ""

    def resolved_source(self) -> str:
        if self.source:
            if self.source not in READOUT_SOURCES:
                raise ValueError(f"未知的读数来源: {self.source}")
            return self.source
        return "theory" if self.theory else "measured"

    def to_json(self) -> dict:
        src = self.resolved_source()
        return dict(key=self.key, label=self.label, unit=self.unit,
                    digits=self.digits, theory=(src == "theory"),
                    source=src, compare_with=self.compare_with, help=self.help)


@dataclass
class Trace:
    """一条实时曲线。axis 相同的曲线共用一个 y 轴。"""

    key: str
    label: str
    color: str = "#4da3ff"
    axis: str = "main"
    unit: str = ""
    dashed: bool = False

    def to_json(self) -> dict:
        return dict(key=self.key, label=self.label, color=self.color,
                    axis=self.axis, unit=self.unit, dashed=self.dashed)


@dataclass
class Bars:
    """一组条形柱，显示**逐关节的控制量**。

    key       telemetry 里的键，值是长度 n 的列表
    limit_key telemetry 里的键，值是长度 n 的限幅列表；画成红色虚线
    """

    key: str
    label: str
    names: list[str] = field(default_factory=list)
    unit: str = ""
    limit_key: str = ""
    color: str = "#4da3ff"
    help: str = ""

    def to_json(self) -> dict:
        return dict(key=self.key, label=self.label, names=list(self.names),
                    unit=self.unit, limit_key=self.limit_key,
                    color=self.color, help=self.help)


@dataclass
class LawTerm:
    """控制律公式里的一个符号。

    symbol  公式里长什么样，例如 "K"
    desc    零基础的一句话解释，例如 "刚度：末端偏 1 m 产生多大回复力"
    key     绑定到 telemetry 的哪个键，前端把数值实时显示在符号旁边；
            留空表示这个符号只做解释、没有对应的标量
    """

    symbol: str
    desc: str
    key: str = ""
    unit: str = ""
    digits: int = 2

    def to_json(self) -> dict:
        return dict(symbol=self.symbol, desc=self.desc, key=self.key,
                    unit=self.unit, digits=self.digits)


@dataclass
class Block:
    """控制框图里的一个方块。前端按声明顺序从左到右排成数据流。

    这是给"控制理论基础弱"的人用的：先看懂信号从哪来、经过谁、到哪去，
    再看公式就不至于一上来就懵。
    """

    label: str
    detail: str = ""
    kind: str = "block"          # "input" | "block" | "plant" | "feedback" | "output"

    def to_json(self) -> dict:
        return dict(label=self.label, detail=self.detail, kind=self.kind)


@dataclass
class ControlLaw:
    """一个场景的控制律声明：公式 + 逐符号解释 + 框图。"""

    formula: str
    terms: list[LawTerm] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return dict(formula=self.formula, note=self.note,
                    terms=[t.to_json() for t in self.terms],
                    blocks=[b.to_json() for b in self.blocks])


# ==================================================================== 三维叠加
#
# 把控制器内部的矢量画到画面上。共同的坑：**物理量的单位不是米**。
# 45 N 的力画成 45 m 长的箭头显然不行，各场景必须自己按一个显式比例尺换算，
# 并把比例尺写进 readout 或 help，否则观众会以为箭头长度就是数值本身。


@dataclass
class Arrow:
    """一支箭头，用于画力 / 速度 / 误差这类矢量。"""

    start: np.ndarray
    end: np.ndarray
    color: str = "#ff5f56"
    width: float = 0.012
    alpha: float = 0.95


@dataclass
class Sphere:
    pos: np.ndarray
    radius: float = 0.02
    color: str = "#4da3ff"
    alpha: float = 0.9


@dataclass
class Frame:
    """坐标架：红绿蓝三根轴对应 x/y/z。用来看清姿态，而不只是位置。"""

    pos: np.ndarray
    rot: np.ndarray
    size: float = 0.09
    width: float = 0.006
    alpha: float = 1.0


@dataclass
class Trail:
    """走过的轨迹。points 为 (N,3)，按时间顺序。"""

    points: np.ndarray
    color: str = "#ffd166"
    width: float = 0.004
    alpha: float = 0.85


@dataclass
class Ghost:
    """半透明的"期望位姿"影子：目标在哪、当前差多少，一眼看出来。"""

    pos: np.ndarray
    radius: float = 0.028
    color: str = "#66bb6a"
    alpha: float = 0.35


class TrailBuffer:
    """末端轨迹的环形缓存。只存点、不存时间，画的是几何轨迹。"""

    def __init__(self, capacity: int = 260, min_step: float = 0.004):
        self.capacity = int(capacity)
        self.min_step = float(min_step)
        self._pts: list[np.ndarray] = []

    def reset(self) -> None:
        self._pts.clear()

    def push(self, p) -> None:
        p = np.asarray(p, dtype=float).reshape(3)
        # 距离上一个点太近就不存。否则末端停住不动时缓存会被同一个点填满，
        # 画出来的轨迹反而变短——看上去像"轨迹在缩回去"。
        if self._pts and np.linalg.norm(p - self._pts[-1]) < self.min_step:
            return
        self._pts.append(p.copy())
        if len(self._pts) > self.capacity:
            del self._pts[0]

    @property
    def points(self) -> np.ndarray:
        return np.array(self._pts) if self._pts else np.zeros((0, 3))


class Scene:
    """场景基类。

    生命周期：
        build()     切换到本场景时调用一次，返回 MjModel
        reset()     每次重置 / 改动 restart 参数时调用
        control()   每个控制周期调用，返回**手臂**关节力矩（长度 7）
        telemetry() 每个采样周期调用，返回 {readout/trace/bars 的 key: 值}
        overlays()  每个渲染周期调用，返回要叠加到三维画面里的几何体
    """

    name = ""
    title = ""
    intro = ""
    camera = dict(azimuth=135.0, elevation=-20.0, distance=2.2,
                  lookat=(0.25, 0.0, 0.45))
    dt = 0.002

    params: list[Param] = []
    readouts: list[Readout] = []
    traces: list[Trace] = []
    bars: list[Bars] = []
    law: ControlLaw | None = None
    lesson: Lesson | None = None
    #: 分步教学。非空时前端按步骤显示，一次只露出一步的讲解与滑块。
    steps: list[Step] = []

    #: 夹爪指令开口宽度（两指间距，m）。整机模型带标准平行夹爪，
    #: 它由腱传动的位置伺服驱动，不受手臂力矩控制律管辖，
    #: 所以由 server 每周期单独写一次执行器指令。
    #: None 表示本场景自己接管夹爪（抓取场景会这么做）。
    gripper_width: float | None = 0.08

    def __init__(self):
        self.values: dict[str, float | str] = {p.key: p.default for p in self.params}
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        #: 由 build() 填入的 ParallelJawGripper。夹爪不受手臂力矩控制律管辖，
        #: 由 server 每周期按 gripper_width 单独写一次位置指令。
        self.gripper = None
        self.note = ""

    # ------------------------------------------------------------ 参数存取

    def get(self, key: str) -> float:
        return float(self.values[key])

    def choice(self, key: str) -> str:
        return str(self.values[key])

    def set(self, key: str, value) -> bool:
        """写入参数，返回是否需要重置场景。"""
        p = next((p for p in self.params if p.key == key), None)
        if p is None:
            raise KeyError(key)
        self.values[key] = str(value) if p.choices else float(value)
        self.on_change(key)
        return p.restart

    def on_change(self, key: str) -> None:
        pass

    # ------------------------------------------------------------ 需子类实现

    def build(self) -> mujoco.MjModel:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def control(self, t: float, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def telemetry(self, t: float, q: np.ndarray, v: np.ndarray) -> dict:
        return {}

    def overlays(self) -> list:
        """返回叠加进三维画面的几何体（Arrow / Sphere / Frame / Trail / Ghost）。

        在渲染线程里调用，**不要**在这里推进任何状态：渲染频率与控制频率不同，
        把状态更新写在这里会让物理量的时间基准变得不可复现。
        需要缓存的量（例如轨迹点）在 control() 里 push，这里只读。
        """
        return []

    # ------------------------------------------------------------ 工具

    def meta(self) -> dict:
        return dict(
            name=self.name, title=self.title, intro=self.intro,
            params=[p.to_json() for p in self.params],
            readouts=[r.to_json() for r in self.readouts],
            traces=[t.to_json() for t in self.traces],
            bars=[b.to_json() for b in self.bars],
            law=self.law.to_json() if self.law else None,
            lesson=self.lesson.to_json() if self.lesson else None,
            steps=[st.to_json() for st in self.steps],
            camera=dict(self.camera, lookat=list(self.camera["lookat"])),
            values=dict(self.values),
            note=self.note,
        )


class RunningRMS:
    """在线均方根，避免存全部历史。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._s = 0.0
        self._n = 0

    def push(self, x) -> None:
        x = np.asarray(x, dtype=float)
        self._s += float(np.sum(x ** 2))
        self._n += x.size

    @property
    def value(self) -> float:
        return float(np.sqrt(self._s / self._n)) if self._n else 0.0
