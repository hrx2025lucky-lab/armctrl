"""在线调参 UI 的五个场景。

每个场景围绕**一个可以自己验证的物理结论**设计，而不是「跑起来好看」：

    impedance   静态偏移应当等于 F/K；阻尼比 ζ 决定回弹是否振荡
    tracking    PD+重力补偿 与 计算力矩 的差别只在「模型前馈」，
                模型准时 CTC 好，模型偏差大时优势消失
    adaptive    跟踪误差收敛 ≠ 参数收敛（需要持续激励 PE）
    observer    观测器带宽 K_I 换取的是「检测延迟 vs 噪声/误报」
    planning    起终点都合法不代表中间合法；平滑只压长度不改可行性

滑块旁边给出的「理论值」是**独立算出来的**（用解析式或另一套量），
不是把实测值抄一遍。两者对不上就说明有一方错了，这正是要看的东西。

统一用带夹爪的整机模型
--------------------
全部场景加载 `panda.xml`（7 关节 + 标准平行夹爪），末端连杆是 `hand`，
控制点 TCP 落在两指夹持垫之间的抓取中心。这带来三个后果，都必须显式处理：

1. 手部质量进入动力学。控制器用的模型必须跟着含手，否则会凭空多出一个
   约 0.7 kg 的未建模负载，跟踪误差会被错误地归因到控制器身上。
   为此动力学一律走 `LockedFingerArmView`（手指锁定的 7 自由度精确降阶）。
2. 执行器从 7 个变 8 个。第 8 个是腱传动的夹爪位置伺服，**不能**被手臂力矩
   控制律覆盖，由 server 按 `Scene.gripper_width` 单独写指令。
3. 两根手指是兄弟连杆，会被自碰撞检查当成一对。规划器不管夹爪开合，
   因此 `self_collision_pairs` 要传入被规划关节，把与规划无关的对排除掉。
"""

from __future__ import annotations

import time

from collections import deque

import numpy as np
import mujoco

from armctrl.core.robot import ArmModel, PANDA_TCP_OFFSET
from armctrl.core.gripper import ParallelJawGripper
from armctrl.control.impedance import CartesianImpedanceController, ImpedanceGains
from armctrl.control.modal import modal_frequencies
from armctrl.control.momentum_observer import (
    GravityCompensationTeaching, MomentumObserver,
    friction_power_margin, passivity_deadband, max_passive_ratio)
from armctrl.control.adaptive import (
    SlotineLiAdaptiveController, RBFAdaptiveController, fit_centers_to_trajectory,
)
from armctrl.identification.offline_ls import OfflineBaseLS
from armctrl.identification.regressor import DynamicsRegressor, LockedFingerArmView
from armctrl.estimation.imu import ImuNoise, SimulatedIMU
from armctrl.estimation.kalman import TcpStateEstimator
from armctrl.estimation.eskf import ErrorStateKalmanFilter
from armctrl.core.kinematics import so3_exp, so3_log
from armctrl.control.ilc import (
    IterativeLearningController, JointComplianceModel)
from armctrl.planning.replan import MovingSphere, ReplanSupervisor
from armctrl.control.grasp import (
    AdaptiveGripConfig, AdaptiveGripForce, SlipDetector, SlipReading,
    box_corner_points, constrained_grasp_frame, effective_friction,
    grasp_frame_from_points, min_normal_force,
)
from armctrl.control.visual_servo import (
    ObjectPoseSensor, PositionVisualServo, SensorSpec,
)
from armctrl.planning.scene import (
    FreeBody, Obstacle, PANDA_XML, build_panda_scene, movable_geoms,
    self_collision_pairs, WRIST_CAMERA,
)
from armctrl.planning.collision import CollisionChecker
from armctrl.planning.rrt import (
    RRTConnect, RRTConnectConfig, path_length, shortcut_smooth,
)
from armctrl.planning.path_timing import JointPathTrajectory
from armctrl.planning.blend import blend_and_verify
from armctrl.planning.topp import ToppraTrajectory, analyze
from armctrl.tuner.lessons import STEPS_BY_SCENE
from armctrl.control.servo import (
    CascadeServo, InputShaper, NotchFilter, TwoMassJoint,
    antiresonance_frequency, residual_vibration, resonance_frequency,
)
from armctrl.tuner.base import (
    Arrow, Bars, Block, ControlLaw, Frame, Ghost, LawTerm, Lesson, Param,
    Readout, RunningRMS, Scene, Sphere, Trace, Trail, TrailBuffer,
)


N_ARM = 7
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
ARM_JOINTS = [f"joint{i}" for i in range(1, N_ARM + 1)]
JOINT_NAMES = [f"J{i}" for i in range(1, N_ARM + 1)]

#: 力矢量画成箭头时的比例尺：1 N 画多少米。50 N 的力约 0.4 m 长，
#: 与 Panda 前臂长度同量级，既看得见又不会糊住整个画面。
#: 比例尺必须写进 intro，否则观众会以为箭头长度就是牛顿数本身。
FORCE_SCALE = 0.008

#: 叠加几何的配色。**不能与坐标架的 x/y/z 红绿蓝重名**：
#: 坐标架按通用约定必须是红绿蓝，如果外力也画成红色、估计力也画成蓝色，
#: 而它们又都从 TCP 出发，画面上就是一团分不清谁是谁的箭头。
#: 所以矢量类一律用坐标架用不到的品红/青/黄。
C_FORCE_TRUE = "#ff2d78"      # 真实外力（品红）
C_FORCE_EST = "#00e5ff"       # 估计外力（青）
C_ERROR = "#ffd166"           # 误差矢量 / 实际轨迹（黄）
#: 「目标 ↔ 实际」之间那根 1:1 连线。必须与 C_ERROR 区分开：轨迹本身就是黄色，
#: 连线再用黄色，画面上分不出「这是走过的路」还是「这是当前的差距」。
C_GAP = "#ff5f56"             # 目标与实际之间的间距（橙红）
C_TARGET = "#66bb6a"          # 期望位姿（绿）
C_PATH = "#4dd0e1"            # 规划路径
C_TCP = "#9fb3c8"             # 末端标记


def _make_robot(model) -> ArmModel:
    """整机模型的手臂视图：末端连杆是 hand，控制点在两指抓取中心。"""
    return ArmModel(model=model, ee_body="hand", arm_joints=ARM_JOINTS,
                    tcp_offset=PANDA_TCP_OFFSET)


def _make_arm_model(reg: DynamicsRegressor) -> LockedFingerArmView:
    """控制器用的动力学模型。手指由位置伺服锁定，手臂看到的是精确的 7 维降阶。"""
    return LockedFingerArmView(reg, n_arm=N_ARM)


#: 「目标点」标记的半径。必须**明显小于夹爪开口的一半**（指垫内表面在 ±41.5 mm），
#: 否则球会整个卡进两指之间，看起来像穿模——而 TCP 本来就定义在两指抓取中心，
#: 标记画在那里是对的，错的是尺寸。
R_TARGET_MARK = 0.013
#: 「实际点」标记的半径。必须小于 R_TARGET_MARK，否则会把目标球顶穿；
#: 但两者之和决定了「多大的误差才看得见两个球分开」，见 _pair_markers。
R_ACTUAL_MARK = 0.007


def _pair_markers(p_target, p_actual, *, c_target=C_TARGET, c_actual=C_TCP,
                  a_target=0.45) -> list:
    """画一对「目标 / 实际」标记，外加一根 1:1 的连线。

    这里有一个容易写错、而且错了之后**看不出来**的地方：如果目标球的半径
    大于实际球的半径加上典型误差，实际球就会永远埋在目标球肚子里，
    观众根本不知道画面上有两个点。原先 34 mm / 20 mm 的组合需要误差超过
    14 mm 才露头，而计算力矩在默认轨迹下的末端误差只有零点几毫米——
    于是「看两个球的距离」这条教学指令在主场景里从来没有成立过。

    现在的 13 mm / 7 mm 组合，误差超过约 6 mm 两球就分离；连线则让「这是
    两个点」这件事在完全重合时也能被看到。

    连线是 **1:1 的真实误差**，没有放大。误差小到看不见时，
    「看不见」本身就是正确的结论，不应该靠放大把它做出来。
    """
    p_t = np.asarray(p_target, float).reshape(3)
    p_a = np.asarray(p_actual, float).reshape(3)
    items = [Sphere(p_t, radius=R_TARGET_MARK, color=c_target, alpha=a_target),
             Sphere(p_a, radius=R_ACTUAL_MARK, color=c_actual, alpha=0.95)]
    if np.linalg.norm(p_t - p_a) > R_ACTUAL_MARK:
        items.append(Trail(np.vstack([p_a, p_t]), color=C_GAP,
                           width=0.0035, alpha=0.95))
    return items


def _torque_bars(help_text: str = "") -> Bars:
    """所有场景共用的关节力矩条形图声明。

    单独抽出来是因为它对每个场景都必须有：控制器最终能做的只有"给每个关节
    发多大力矩"，这七个数就是控制量本身。红线是模型里的 `actuator_forcerange`
    （前 4 关节 87 N·m，后 3 关节 12 N·m），碰到红线就是饱和，
    此刻控制律再算出多大值都发不出去，闭环特性会突然变样。
    """
    return Bars("tau_vec", "关节力矩 τ（控制量）", names=JOINT_NAMES, unit="N·m",
                limit_key="tau_limit", color="#4da3ff",
                help=help_text or "红色虚线是该关节的力矩上限，碰线即饱和。")


# ==================================================================== 1 阻抗


class ImpedanceScene(Scene):
    name = "impedance"
    steps = STEPS_BY_SCENE["impedance"]
    title = "笛卡尔阻抗控制"
    camera = dict(azimuth=141.0, elevation=-14.0, distance=1.15,
                  lookat=(0.42, 0.0, 0.42))
    intro = (
        "机械臂要保持末端停在绿色影子位置，同时你从外面推它。\n\n"
        "阻抗控制不是「顶住不动」，而是让末端表现得像一个弹簧-阻尼系统："
        "推得越用力偏得越多，松手就弹回去。"
    )
    lesson = Lesson(
        problem=(
            "让机械臂「停在某个位置」，最直觉的做法是位置控制：谁把它推歪了就用最大的劲推回去。"
            "工业机械臂几十年都这么干，因为它只跟工件打交道。\n\n"
            "但只要环境里出现<b>人</b>，或者末端要<b>贴着一个面</b>干活（打磨、装配、插孔），"
            "这套就出事了：位置控制的机械臂遇到阻挡不会退让，它会一直加大力矩，"
            "直到把人夹伤、把工件顶碎、或者自己过载停机。\n\n"
            "问题的根子是：位置控制器<b>只管位置，不管力</b>。我们需要一种能同时管住"
            "「位置偏多少」和「用多大力」的办法。"
        ),
        idea=(
            "换个思路：<b>别去规定机械臂停在哪，去规定它「有多软」。</b>\n\n"
            "想象在末端和目标点之间接了一根<b>看不见的弹簧</b>，再并联一个<b>阻尼器</b>"
            "（就是关门用的液压缓冲器那种）。这时候机械臂的行为完全由这两个元件决定：\n\n"
            "• 你不推它 → 弹簧不伸长 → 它停在目标点\n"
            "• 你推它 → 弹簧被拉长，产生一个反抗的力 → 推得越狠偏得越多\n"
            "• 你松手 → 弹簧把它拉回去，阻尼器决定它是慢慢贴回去还是来回弹几下\n\n"
            "关键在于：<b>这根弹簧是软的</b>。人撞上去，机械臂会让开；"
            "力大到一定程度它就偏开了，不会一根筋地硬顶。而弹簧有多硬，是我们自己设定的。\n\n"
            "这就叫「阻抗」——机械臂对外表现出来的「软硬性格」。"
        ),
        how=(
            "怎么在代码里造出这根不存在的弹簧？分三步。\n\n"
            "<b>第一步：算末端需要多大的力。</b>\n"
            "弹簧的规律是中学物理的胡克定律 F = K·Δx。量出末端偏离目标多远（Δx），"
            "乘上我们设定的刚度 K，就是弹簧该出的力。再加上阻尼器的力 D·Δẋ"
            "（速度越快，阻力越大）。两项加起来就是「末端此刻应该受到的力」。\n\n"
            "<b>第二步：把「末端的力」翻译成「关节的力矩」。</b>\n"
            "问题是电机装在关节上，它们只会转，不会直接在末端产生力。"
            "这里需要<b>雅可比矩阵 J</b>——它是一张「换算表」，告诉你每个关节转一点点，"
            "末端会往哪个方向移动多少。\n"
            "有个漂亮的结论：把这张换算表<b>转置</b>一下（记作 Jᵀ），"
            "它就变成了反方向的换算表——「末端要 F 这么大的力，各个关节该出多大力矩」。"
            "这一步就是公式里的 τ = Jᵀ·F。\n"
            "（为什么转置就行？因为功率守恒：末端做的功 Fᵀ·ẋ 必须等于各关节做的功 τᵀ·q̇，"
            "而 ẋ = J·q̇，代进去两边一对比就得到 τ = Jᵀ·F。这是可以自己推的。）\n\n"
            "<b>第三步：把重力「抵消掉」。</b>\n"
            "机械臂自身有几十公斤重，电机不出力它就砸下来了。所以还要额外加一项 h(q,q̇)，"
            "把重力和运动带来的科氏力都补上。补完之后，机械臂就像「没有重量」一样，"
            "只剩我们设计的那根弹簧在起作用。"
        ),
        watch=[
            "<b>先看默认状态</b>：外力 −30 N 一直开着，末端被压下去约 60 mm，"
            "黄色误差箭头（放大 4 倍画的）明显指向下方。",
            "<b>拖「平移刚度 K」到最左（50）</b>：末端<b>肉眼可见地一路掉下去</b>，"
            "一直沉到夹爪碰上地板为止（这时理论值和实测会分家，原因见第 ⑤ 段）。",
            "<b>拖「平移刚度 K」到最右（2500）</b>：末端几乎回到绿球位置，偏移只剩 12 mm。"
            "弹簧很硬，接近位置控制的行为。",
            "<b>把「外力模式」切成「方波」</b>：外力变成一开一关。这时才能看阻尼比 ζ 的作用——"
            "看下面<b>蓝色曲线</b>在力撤掉那一刻怎么回到零。",
            "<b>方波模式下把「阻尼比 ζ」拉到 0.3</b>：曲线会<b>上下振荡好几个来回</b>才停。"
            "拉到 2.0：慢吞吞地爬回去，不振荡但很迟钝。ζ=1 附近最干脆。",
            "<b>看「关节力矩」条形图</b>：外力一来 J2 和 J4 立刻扛起主要负担，它们是承重关节。"
            "把外力调到 −80 N 且 K 调大，注意后三个关节（限幅只有 12 N·m）会不会贴红线。",
        ],
        verify=[
            "「实测偏移 Δz」和「理论 F/K（稳态）」两个数应当基本重合。"
            "理论值是用胡克定律独立算的，没读控制器里的任何中间量——"
            "两者对得上，说明代码里跑的和你理解的是同一个物理。",
            "「控制器命令的末端力」应当等于外力大小。稳态下弹簧力和外力必须平衡，"
            "这是同一件事的另一个说法。",
            "刚度翻倍，偏移应当减半。实测：K=2500→12.00 mm，K=1000→30.00 mm，"
            "K=500→60.00 mm，K=150→200.0 mm，全部与理论逐位吻合。自己拖着验证这个反比关系。",
            "<b>但 K 拉到 50 时会出现约 33% 的偏差</b>（实测 −404 mm vs 理论 −600 mm）。"
            "这<b>不是 bug</b>：600 mm 的下沉会把夹爪直接压到地板上（实测 TCP 高度只剩 9.42 mm，"
            "有 3 个接触点），地面帮忙扛住了 9.88 N，弹簧只需要扛剩下的 20.19 N。"
            "此时 K·Δz = 20.19 N 仍然精确等于控制器命令的力——<b>弹簧定律没坏，"
            "坏的是「外力全部由弹簧承担」这个前提</b>。"
            "这正是真机上必须注意的事：柔顺控制一旦软到让机器人碰到别的东西，"
            "你算的那套力平衡就不再是全部。",
        ],
        caveat=(
            "⚠️ <b>方波模式下拖滑块可能「看不到反应」</b>：方波有一半时间外力是关的，"
            "此时末端本来就停在目标点，改刚度当然没有任何现象。想看刚度的效果请用<b>恒力</b>模式。\n\n"
            "⚠️ <b>这是仿真，不是真机。</b>这里控制器用的模型和被控对象是同一套参数，"
            "没有噪声、没有关节摩擦、没有通信延迟。真机上重力补偿补不准，"
            "机械臂会自己慢慢往下沉；摩擦会让它「推不动也弹不回」。"
            "所以这里的重合度不能说明真机上也这么准。\n\n"
            "⚠️ <b>姿态刚度这个滑块在本场景几乎看不出效果</b>，因为外力是纯平移的、不产生力矩。"
            "这不是程序坏了，是这个激励不激发那个自由度。"
        ),
    )
    law = ControlLaw(
        formula="τ = Jᵀ(q)·[ K·(x_d − x) + D·(ẋ_d − ẋ) ] + h(q,q̇) + τ_null",
        terms=[
            LawTerm("x_d − x", "期望位置减实际位置，就是画面里那支黄色误差箭头",
                    key="err", unit="mm", digits=2),
            LawTerm("K", "刚度：末端每偏 1 m，弹簧产生多大回复力",
                    key="k_now", unit="N/m", digits=0),
            LawTerm("D", "阻尼：抵抗速度。由 ζ 和任务空间惯量 Λ 一起定，不是独立旋钮",
                    key="d_now", unit="N·s/m", digits=1),
            LawTerm("Jᵀ", "雅可比转置。把「末端要多大的力」翻译成「每个关节出多大力矩」"),
            LawTerm("h(q,q̇)", "重力与科氏力补偿。不补的话手臂自己就垂下去了",
                    key="h_norm", unit="N·m", digits=1),
            LawTerm("τ_null", "零空间项。七自由度有冗余，用它把手肘姿态拉住而不影响末端"),
        ],
        blocks=[
            Block("期望位姿 x_d", "绿色影子球", "input"),
            Block("误差 x_d − x", "黄色箭头", "block"),
            Block("弹簧阻尼 K, D", "算出末端需要多大的力 F", "block"),
            Block("Jᵀ 映射", "末端的力 → 关节力矩", "block"),
            Block("+ h(q,q̇)", "补掉重力与科氏力", "block"),
            Block("机械臂", "MuJoCo 被控对象", "plant"),
            Block("实测 x, ẋ", "正运动学读回来", "feedback"),
        ],
        note="稳态时 ẋ=0、外力 F 与弹簧力平衡，于是 K·Δx = F，"
             "这就是「理论 F/K」那一栏的来历——它是从公式独立推出来的，"
             "不是把实测值抄一遍。",
    )
    params = [
        Param("k_trans", "平移刚度", 50, 2500, 500, unit="N/m", symbol="K",
              group="控制器",
              help="末端每偏移 1 m 产生多大回复力。越大越「硬」。",
              effect="调小 → 末端被外力压得越来越低，<b>画面上直接能看到手臂沉下去</b>；"
                     "调大 → 末端贴回绿球。刚度翻倍，偏移减半。"
                     "调到 50 会一直沉到<b>夹爪压上地板</b>，此时理论值不再适用。"),
        Param("zeta", "阻尼比 ζ", 0.1, 2.5, 1.0, unit="", symbol="→D",
              group="控制器",
              help="<1 欠阻尼会振荡，=1 临界阻尼，>1 过阻尼变迟钝。",
              effect="<b>恒力模式下看不出来</b>（画面是静止的）。请先把「外力模式」切成方波，"
                     "再看<b>下方蓝色曲线</b>：ζ=0.3 会来回振荡，ζ=2 慢吞吞爬回去。"),
        Param("k_rot", "姿态刚度", 5, 150, 35, unit="N·m/rad", group="控制器",
              help="抵抗末端转动的刚度。",
              effect="<b>本场景几乎无效果</b>：外力是纯平移的，不产生力矩，"
                     "所以激发不出姿态偏差。这不是程序坏了。"),
        Param("damping", "阻尼设计", default="critical",
              choices=["critical", "fixed"], symbol="D", group="控制器",
              help="critical 用任务空间惯量 Λ 整定 D=2ζΛ^½K^½；"
                   "fixed 忽略惯量直接用 D=2ζK^½。",
              effect="方波模式下切换，看曲线超调量的差别。fixed 忽略了「末端在不同方向上"
                     "惯量不一样」这件事，整定出来的阻尼在某些方向偏小，会多冲一点。"),
        Param("force_mode", "外力模式", default="constant",
              choices=["constant", "square"], group="外部激励",
              help="constant 恒力常开；square 方波，每周期开一半关一半。",
              effect="<b>调刚度请用 constant</b>（现象持续存在，一拖就看得见）；"
                     "<b>调阻尼比请用 square</b>（需要一次阶跃激励才看得出振荡）。"),
        Param("fz", "外力 Fz", -80, 80, -30, unit="N", group="外部激励",
              help="沿世界 z 轴施加在末端的力，负值为向下压。画面里的品红箭头。",
              effect="绝对值越大 → <b>品红箭头越长</b>，末端偏得越多，"
                     "关节力矩条也跟着变长。正负号决定往上抬还是往下压。"),
        Param("period", "方波周期", 1.0, 12.0, 5.0, unit="s", group="外部激励",
              help="只对 square 模式生效。每周期前一半开、后一半关。",
              effect="只影响方波的快慢。<b>constant 模式下这个滑块完全没作用。</b>"),
    ]
    readouts = [
        Readout("dz", "实测偏移 Δz", "mm", 2),
        Readout("dz_theory", "理论 F/K（稳态）", "mm", 2, theory=True, compare_with="dz",
                help="由稳态力平衡 K·Δz = F 独立算出，不读取控制器内部量。"
                     "它是**稳态**值：方波刚翻转的那一两秒里末端还在动，"
                     "此时与实测有偏差是正常的，看的是稳下来之后重不重合。"),
        Readout("err", "位置误差范数", "mm", 2),
        Readout("f_now", "当前外力", "N", 1),
        Readout("f_cmd", "控制器命令的末端力", "N", 1, source="internal",
                help="控制器算出的 K·Δx + D·Δẋ 的模。稳态时它应当与外力大小相等、"
                     "方向相反——这是力平衡，与 Δz=F/K 是同一件事的两种看法。"),
        Readout("tau", "力矩范数 ‖τ‖", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1,
                help="七个关节中离自己限幅最近的那个用掉了百分之多少。"),
        Readout("h_norm", "重力+科氏力补偿 ‖h‖", "N·m", 1,
                help="把手臂自身重量撑住要花的力气，与外力无关。"),
    ]
    traces = [
        Trace("dz", "实测 Δz", "#4da3ff", unit="mm"),
        Trace("dz_theory", "理论 F/K（稳态）", "#ffb74d", unit="mm", dashed=True),
        Trace("f_now", "外力 Fz", "#ff5f56", axis="f", unit="N"),
    ]
    bars = [_torque_bars()]

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.p0, self.R0 = self.robot.fk(Q_HOME)
        self.p0 = self.p0.copy()
        self.R0 = self.R0.copy()
        self.ctrl = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=self.get("k_trans"),
                                       k_rot=self.get("k_rot"),
                                       zeta=self.get("zeta")),
            damping_design=self.choice("damping"), q_null=Q_HOME, dt=self.dt,
        )
        self._tau = np.zeros(N_ARM)
        self._fz_now = 0.0
        self._p = self.p0.copy()
        self._R = self.R0.copy()
        self.trail = TrailBuffer(capacity=200, min_step=0.0015)

    def on_change(self, key):
        if key == "damping" and hasattr(self, "ctrl"):
            self.ctrl.damping_design = self.choice("damping")

    def control(self, t, q, v):
        g = self.ctrl.gains
        g.k_trans = self.get("k_trans")
        g.k_rot = self.get("k_rot")
        g.zeta = self.get("zeta")
        # 恒力模式下现象一直存在，拖刚度滑块立刻看得见效果；
        # 方波模式每次开/关都是一次阶跃激励，是看阻尼比 ζ 的唯一办法
        # （恒力下末端是静止的，ζ 根本不进入稳态解，怎么调都没有现象）。
        if self.choice("force_mode") == "square":
            on = (t % self.get("period")) < self.get("period") * 0.5
        else:
            on = True
        self._fz_now = self.get("fz") if on else 0.0
        self.data.xfrc_applied[self.robot.ee_body_id, :3] = \
            np.array([0.0, 0.0, self._fz_now])
        self._tau = self.ctrl.compute(q, v, self.p0, self.R0, dt=self.dt)
        self._p, self._R = self.robot.fk(q)
        self._h = float(np.linalg.norm(self.robot.bias(q, v)))
        self._f_cmd = float(np.linalg.norm(self.ctrl.last_wrench[:3]))
        self.trail.push(self._p)
        return self._tau

    def telemetry(self, t, q, v):
        p = self._p
        dz = (p[2] - self.p0[2]) * 1e3
        # 独立算的理论值：稳态时弹簧力平衡外力，K·Δz = F。
        # 用**当前时刻实际施加**的力，所以方波关掉时理论值回到 0。
        dz_theory = self._fz_now / self.get("k_trans") * 1e3
        lim = self.robot.tau_limit
        return dict(dz=dz, dz_theory=dz_theory,
                    err=float(np.linalg.norm(p - self.p0)) * 1e3,
                    f_now=self._fz_now, f_cmd=self._f_cmd,
                    tau=float(np.linalg.norm(self._tau)),
                    tau_vec=self._tau, tau_limit=lim,
                    tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0,
                    k_now=self.get("k_trans"),
                    d_now=2.0 * self.get("zeta") * np.sqrt(self.get("k_trans")),
                    h_norm=self._h)

    def overlays(self):
        items = [
            Ghost(self.p0, radius=R_TARGET_MARK, color=C_TARGET, alpha=0.45),
            Frame(self._p, self._R, size=0.085, width=0.005),
            Trail(self.trail.points, color=C_ERROR, width=0.005),
        ]
        if abs(self._fz_now) > 1e-9:
            # 力箭头从末端出发，指向力的方向。比例尺 FORCE_SCALE 写在 intro 里，
            # 否则观众会以为箭头长度就是牛顿数。
            # 起点沿 −y 让开一点，否则箭杆整根埋在夹爪网格里只露出个尖
            root = self._p + np.array([0.0, -0.055, 0.0])
            tip = root + np.array([0.0, 0.0, self._fz_now * FORCE_SCALE])
            items.append(Arrow(root, tip, color=C_FORCE_TRUE, width=0.016))
        # 误差矢量放大 4 倍才看得见：典型偏移只有几毫米。
        # 倍数写在 intro 与讲解里，否则观众会把箭头长度当成真实偏移。
        d = self._p - self.p0
        if np.linalg.norm(d) > 1e-4:
            items.append(Arrow(self.p0, self.p0 + d * 4.0,
                               color=C_ERROR, width=0.011))
        return items


# ============================================================ 2 PD vs 计算力矩


def _perturbation(n: int, seed: int = 12345) -> np.ndarray:
    """固定的相对扰动方向，保证「模型偏差 x%」是可复现的同一种偏差。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n)


class TrackingScene(Scene):
    name = "tracking"
    steps = STEPS_BY_SCENE["tracking"]
    title = "PD+重力补偿  vs  计算力矩"
    camera = dict(azimuth=138.0, elevation=-16.0, distance=1.35,
                  lookat=(0.40, 0.0, 0.45))
    intro = (
        "机械臂跟踪一条关节空间正弦轨迹。两个控制器的差别只有一处：<b>用不用模型前馈</b>。"
    )
    lesson = Lesson(
        problem=(
            "让机械臂「走一条指定的轨迹」，最省事的做法是 PD 控制：看它偏了多少，"
            "按比例给一个纠正力矩。这在慢速时够用。\n\n"
            "但机械臂快起来之后会出现两个新麻烦：\n"
            "• <b>惯性</b>：手臂有几十公斤，想让它突然加速，光靠「偏了才纠」永远慢半拍，"
            "误差会一直挂在那儿下不去。\n"
            "• <b>科氏力与离心力</b>：手臂在转的时候，各个连杆之间会互相「甩」。"
            "第 2 关节转得快，会给第 4 关节带来一个额外的力，而 PD 完全不知道这回事。\n\n"
            "结果就是：想让 PD 跟得准，只能把增益调得很高；增益一高又容易振荡、容易打人。"
        ),
        idea=(
            "PD 是<b>事后补救</b>：等偏了才纠。有没有办法<b>事先算好</b>？\n\n"
            "有。牛顿告诉我们 F = m·a。如果我知道这条轨迹在每一时刻要求多大的加速度，"
            "又知道机械臂有多重，我<b>直接就能把需要的力算出来</b>，根本不用等它偏。\n\n"
            "打个比方：推一辆购物车走 S 形。\n"
            "• PD 的做法：闭着眼推，感觉偏了就掰一下方向。\n"
            "• 计算力矩（CTC）的做法：睁着眼，提前知道下一秒要转弯，<b>提前发力</b>。"
            "偶尔算不准了，再用 PD 补一点点。\n\n"
            "所以 CTC = <b>模型前馈（提前算好的主力）+ PD 反馈（收拾残差的小工）</b>。"
            "PD 那一项还在，只是它现在只需要干很少的活。"
        ),
        how=(
            "机械臂的运动方程长这样（就是 F = m·a 的多关节版本）：\n\n"
            "    <b>M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ</b>\n\n"
            "逐项翻译：\n"
            "• <b>M(q)</b> 质量矩阵。它不是一个常数，而是随姿势变的——手臂伸直时「很重」，"
            "缩起来时「很轻」，就像你端一杯水，胳膊伸直比弯着费劲得多。\n"
            "• <b>C(q,q̇)·q̇</b> 科氏力与离心力。只有在动的时候才出现，各连杆互相甩的那部分。\n"
            "• <b>g(q)</b> 重力。手臂自身的重量，不出力就往下掉。\n"
            "• <b>τ</b> 电机力矩，也就是我们唯一能直接决定的东西。\n\n"
            "CTC 就是把这个方程<b>反过来用</b>：我想要的加速度是 a_d，"
            "那就直接算 τ = M·a_d + C·q̇ + g。再把 PD 拼进加速度里防止漂移，"
            "就得到 τ = M·(a_d + Kd·ė + Kp·e) + C·q̇ + g。\n\n"
            "PD+重力补偿则是它的<b>简化版</b>：只补 g(q)（不补重力手臂会直接垂下去），"
            "M 和 C 那两项完全不管，全交给反馈去追。"
        ),
        watch=[
            "<b>把「模型误差」保持 0，「轨迹频率」从 0.8 拉到 2.0 Hz</b>，"
            "在「控制器」下拉里来回切 ctc / pd_gravity："
            "看「跟踪误差 RMS」这个数，CTC 会明显更小。频率越高差距越大。",
            "<b>看画面</b>：绿色影子球是此刻的期望位置，蓝色小球是实际末端。"
            "选 pd_gravity 且频率调高时，能看到<b>蓝球总是落在绿球后面</b>——这就是「跟不上」。",
            "<b>把「模型误差」拉到 30%</b>，再切回 ctc：CTC 的优势大幅缩水。"
            "因为它前馈用的是一个<b>错的</b>模型，提前算出来的力本身就是错的。",
            "<b>这正是自适应控制存在的理由</b>——模型不准时，能不能让控制器自己边跑边修正？"
            "切到「自适应控制」那个场景看。",
            "<b>看关节力矩条形图</b>：切换控制器时，力矩的分布形状会变。"
            "CTC 的力矩更「预判」，PD 的力矩更「跟随」。",
        ],
        verify=[
            "「模型参数偏差」这一栏显示的就是你拖的那个百分比，"
            "它是<b>真的去扰动质量、惯量、摩擦这些物理参数</b>，不是简单把力矩乘个系数。"
            "这两件事不等价：乘系数只是改变了增益，扰动参数才是真正的模型失配。",
            "模型误差为 0 时，CTC 的误差应当接近于零（只剩离散化和数值残差）。"
            "如果这时误差还很大，说明模型前馈那部分算错了。",
        ],
        caveat=(
            "⚠️ <b>不能直接拿两个控制器的误差数字下结论。</b>CTC 里的 Kp 乘在加速度量纲上"
            "（外面还有个 M(q)），PD 里的 Kp 直接就是力矩量纲。同样标着 400，"
            "两者的等效带宽完全不同。<b>严格的对比必须先对齐带宽</b>，这里只做定性演示。\n\n"
            "⚠️ <b>仿真高估了 CTC。</b>模型误差为 0 时，控制器用的模型和被控对象是同一套代码，"
            "这在真机上永远做不到——真机的质量分布、摩擦、齿轮间隙都测不准。"
            "所以现实中 CTC 的优势远没有这里显示的这么大。"
        ),
    )
    law = ControlLaw(
        formula="CTC：τ = M(q)·[ a_d + Kd·ė + Kp·e ] + C(q,q̇)q̇ + g(q)\n"
                "PD ：τ = Kp·e + Kd·ė + g(q)",
        terms=[
            LawTerm("e = q_d − q", "关节角误差。曲线上那条蓝线",
                    key="err", unit="mrad", digits=2),
            LawTerm("Kp", "位置增益。误差乘它变成「要多大的加速度」",
                    key="kp_now", unit="1/s²", digits=0),
            LawTerm("Kd", "速度增益。抑制振荡", key="kd_now", unit="1/s", digits=0),
            LawTerm("a_d", "期望加速度。前馈项，PD 没有这一项",
                    key="ad_norm", unit="rad/s²", digits=2),
            LawTerm("M(q)", "质量矩阵。把「想要的加速度」换算成「需要的力矩」",
                    key="m_norm", unit="kg·m²", digits=2),
            LawTerm("C q̇ + g", "科氏力与重力。PD 只补 g，CTC 两个都补",
                    key="h_norm", unit="N·m", digits=1),
        ],
        blocks=[
            Block("正弦参考 q_d, q̇_d, q̈_d", "频率与幅值可调", "input"),
            Block("误差 e, ė", "", "block"),
            Block("PD 反馈 Kp·e + Kd·ė", "两个控制器都有", "block"),
            Block("模型前馈 M(q)·(…) + C q̇ + g", "只有 CTC 有；模型误差滑块污染的就是它",
                  "block"),
            Block("机械臂", "MuJoCo 被控对象，参数永远是真值", "plant"),
            Block("实测 q, q̇", "", "feedback"),
        ],
        note="两个控制器的 Kp/Kd 是同一组数，但含义不同：CTC 里它们乘的是"
             "加速度量纲（因为外面还有 M(q)），PD 里直接就是力矩。"
             "所以严格的对比必须先对齐等效带宽，这里只做定性演示。",
    )
    params = [
        Param("controller", "控制器", default="ctc",
              choices=["pd_gravity", "ctc"], group="控制器",
              effect="切换后看「跟踪误差 RMS」。模型准时 ctc 明显更小；"
                     "把模型误差拉到 30% 后两者会拉近。"),
        Param("kp", "位置增益", 20, 2000, 400, symbol="Kp", group="控制器",
              help="反馈刚度。",
              effect="调大 → 跟踪误差变小，但太大会开始<b>抖</b>（看曲线和力矩条）。"
                     "这就是「高增益换精度」的代价。"),
        Param("kd", "速度增益", 2, 200, 40, symbol="Kd", group="控制器",
              help="反馈阻尼。",
              effect="配合 Kp 用：Kp 大而 Kd 太小会振荡；Kd 过大则响应迟钝。"
                     "看曲线抖不抖。"),
        Param("freq", "轨迹频率", 0.1, 2.5, 0.8, unit="Hz", group="参考轨迹",
              help="越快，惯量与科氏力越重要，模型前馈的价值越明显。",
              effect="<b>画面上手臂挥动明显变快</b>。频率越高，"
                     "pd_gravity 与 ctc 的误差差距越大——这是本场景的核心现象。"),
        Param("amp", "轨迹幅值", 0.05, 0.8, 0.35, unit="rad", group="参考轨迹",
              effect="<b>画面上挥动幅度变大</b>，末端拖尾画出的圈更大。"),
        Param("align", "带宽对齐", default="off", choices=["off", "on"],
              group="控制器",
              help="把 PD 的增益整体缩放，使其等效 ωn 的中位数与 CTC 相同。",
              effect="<b>不对齐时两个控制器的误差数字不可直接比较</b>——"
                     "实测 PD 的 ωn 中位数 32.7 rad/s、CTC 恰好 20.0，"
                     "PD 反而更快，掩盖了它没有前馈的劣势。打开后再比才有意义。"),
        Param("model_err", "模型误差", 0.0, 60.0, 0.0, unit="%", group="被控对象",
              help="按固定方向扰动全部动力学参数。只影响控制器用的模型，"
                   "不影响 MuJoCo 里真实的机器人。",
              effect="<b>画面上看不出来</b>（机器人本身没变），要看「跟踪误差 RMS」变大。"
                     "这模拟的是「你以为手臂多重，跟它实际多重不一样」。"),
    ]
    readouts = [
        Readout("rms", "跟踪误差 RMS", "mrad", 3),
        Readout("err", "当前误差范数", "mrad", 3),
        Readout("ee_err", "末端位置误差", "mm", 2),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
        Readout("param_err", "模型参数偏差", "%", 1, source="internal",
                help="就是你拖的那个滑块本身，不是独立算出来的对照值。"),
        Readout("wn_med", "等效增益 ωn（中位）", "rad/s", 2, source="theory",
                help="用「隐含反馈增益测量法」现场测：冻结参考、人为偏置一个关节，"
                     "读控制器发出的力矩，再用 **MuJoCo 的** 真实 M 与 h 算闭环加速度 "
                     "q̈ = M⁻¹(τ−h)，由 q̈ = K·e 反解 K，ωn = √K。\n"
                     "⚠️ P0-02 修复：旧版这里用的是**控制器自己的模型**推进对象，"
                     "属于自证循环；现在被控对象换成 MuJoCo，两边彻底分开。\n"
                     "⚠️ 这是**逐关节等效增益**，不是模态频率——看下面两栏。"),
        Readout("wn_spread", "ωn 逐关节离散度", "rad/s", 2, source="theory",
                help="最大值减最小值。CTC 会把它压到 0（解耦），"
                     "PD 做不到——这是两者最本质的差别，比误差数字更能说明问题。"),
        Readout("modal_min", "最慢模态", "rad/s", 2, source="theory",
                help="⭐ 闭环 M ë + Kd ė + Kp e = 0 的**真实**最低固有频率，"
                     "解广义特征值 det(Kp − ω²M) = 0 得到。\n"
                     "**最慢的模态决定整体响应速度**，所以这一栏比中位数更重要。\n"
                     "⚠️ 旧版用 sqrt(Kp·[M⁻¹]_jj)（只取对角线），"
                     "实测把这个数**高报了 42%**。"),
        Readout("modal_spread", "模态离散度", "×", 2, source="theory",
                help="= 最快模态 / 最慢模态。1.0 表示完全解耦（CTC 的目标）。\n"
                     "⚠️ 反例：M=[[2,1],[1,2]]、Kp=I 时，对角线法报"
                     "「离散度 0」（看着完美），真实模态离散度是 **1.732**。\n"
                     "Panda 上实测：对角线法 3.11，真实 **4.48**。"),
    ]
    traces = [
        Trace("err", "跟踪误差", "#4da3ff", unit="mrad"),
        Trace("tau", "力矩范数", "#9ccc65", axis="tau", unit="N·m"),
    ]
    bars = [_torque_bars()]

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.pi_true = self.reg.true_parameters()
        self.pattern = _perturbation(self.pi_true.size)
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # 控制器模型里的手指位置要与夹爪指令一致，否则手部惯量算在错的构型上
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self.rms = RunningRMS()
        self._tau = np.zeros(N_ARM)
        self._mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self._e = np.zeros(N_ARM)
        self._p = self.robot.fk(Q_HOME)[0].copy()
        self._p_des = self._p.copy()
        self._h = 0.0
        self._m = 0.0
        self._ad = np.zeros(N_ARM)
        self.trail = TrailBuffer(capacity=260, min_step=0.003)

    #: 参与激励的关节（与 _mask 一致），带宽只在这些关节上测才有意义
    _WN_JOINTS = (0, 1, 3, 5)

    def _align_scale(self, q) -> float:
        """PD 增益的带宽对齐系数。

        CTC 的闭环是 q̈ = Kp·e（M 被前馈消掉），逐关节都等于 √Kp。
        PD 的闭环是 M ë + Kd ė + Kp e = 0，它的固有频率是**广义特征值**
        det(Kp − ω²M) = 0 的解，**不是**逐关节独立的频率。
        ⭐ **不可能用一个标量让 PD 完全对齐**——能做到那件事的东西就叫计算力矩。
        这里只把**中位模态**对齐，让"谁更快"这个因素从对比里去掉。

        ⚠️ P0-02 修复：旧版用 ``sqrt(Kp·[M⁻¹]_jj)``（只取对角线），
        那不是模态频率，它**丢掉了关节之间的惯性耦合**。
        反例 M=[[2,1],[1,2]], Kp=I：对角线法给 [0.816, 0.816]（离散度 1.000，
        看着"完全一致"），真实模态却是 [1.000, 0.577]（离散度 1.732）。
        实测 Panda 上对角线法把最慢模态高报了 42%。

        Kd 按 √s 缩放以保持阻尼比 ζ = Kd/(2√(Kp·M)) 不变。
        """
        Ms = self.robot.mass_matrix(q)[np.ix_(self._WN_JOINTS, self._WN_JOINTS)]
        w = modal_frequencies(Ms, 1.0)            # Kp=1 时的模态频率
        med = float(np.median(w))
        return 1.0 / max(med * med, 1e-9)         # 使 median(ωn) = √Kp

    def _measure_wn(self, q, delta: float = 2e-3):
        """隐含反馈增益测量法，返回逐关节等效 ωn。

        冻结参考、把某个关节偏置 δ，读控制器**实际发出的力矩**，
        再用**被控对象**的动力学算 q̈ = M⁻¹(τ−h)。
        误差 e = −δ，若闭环为 q̈ = K·e 则 K = −q̈/δ，ωn = √K。

        ⚠️ P0-02 修复：旧版这里用的是 ``self.reg``（**控制器自己的模型**）
        来推进"被控对象"，属于同模型自证循环——控制器和对象是同一套方程，
        计算力矩分支代数上**必然**得到期望的 q̈。
        现在改用 ``self.robot``（MuJoCo 的 ``mj_fullM`` / ``qfrc_bias``），
        控制器仍用 ``self.reg`` + 带误差的 ``pi_hat``，**两边彻底分开**。

        ⚠️ 仍需注意：这里得到的是**逐关节等效增益**，不是模态频率。
        真正的模态频率见读数「最慢模态 / 模态离散度」。
        """
        kp, kd = self.get("kp"), self.get("kd")
        if self.choice("controller") == "pd_gravity" and self.choice("align") == "on":
            s = self._align_scale(q)
            kp, kd = kp * s, kd * np.sqrt(s)
        pi = self._pi_hat()
        zero = np.zeros(N_ARM)
        out = []
        for j in self._WN_JOINTS:
            qj = q.copy()
            qj[j] += delta
            e = -delta
            if self.choice("controller") == "pd_gravity":
                g = self.reg.regressor(qj, zero, zero) @ pi
                tau = np.zeros(N_ARM)
                tau[j] = kp * e
                tau = tau + g
            else:
                aq = np.zeros(N_ARM)
                aq[j] = kp * e
                Y0 = self.reg.regressor(qj, zero, zero)
                Ya = self.reg.regressor(qj, zero, aq)
                Yh = self.reg.regressor(qj, zero, zero)
                tau = (Ya - Y0) @ pi + Yh @ pi
            # ↓↓ 被控对象用 MuJoCo，不是控制器的模型
            qdd = np.linalg.solve(self.robot.mass_matrix(qj),
                                  tau - self.robot.bias(qj, zero))
            K = -qdd[j] / delta
            out.append(np.sqrt(K) if K > 0 else np.nan)
        return np.array(out)

    def _pi_hat(self):
        return self.pi_true * (1.0 + self.get("model_err") / 100.0 * self.pattern)

    def _reg_mass(self, q, pi):
        """控制器**自己那套模型**在 q 处的质量矩阵 M_reg(q; π̂)。

        按列构造：第 i 列 = (Y(q,0,e_i) − Y(q,0,0))·π̂，
        这和 :meth:`control` 里算 ``(Ya−Y0)@pi`` 用的是同一条路径，
        所以它确实是控制器前馈时"以为"的惯量，而不是真实惯量。
        """
        zero = np.zeros(N_ARM)
        Y0 = self.reg.regressor(q, zero, zero)
        cols = []
        for i in range(N_ARM):
            a = np.zeros(N_ARM)
            a[i] = 1.0
            cols.append((self.reg.regressor(q, zero, a) - Y0) @ pi)
        M = np.column_stack(cols)
        return 0.5 * (M + M.T)

    def _ref(self, t):
        w = 2.0 * np.pi * self.get("freq")
        a = self.get("amp") * self._mask
        q = Q_HOME + a * np.sin(w * t)
        v = a * w * np.cos(w * t)
        acc = -a * w * w * np.sin(w * t)
        return q, v, acc

    def control(self, t, q, v):
        qd, vd, ad = self._ref(t)
        kp, kd = self.get("kp"), self.get("kd")
        if self.choice("controller") == "pd_gravity" and self.choice("align") == "on":
            # 对齐带宽后再比，才符合工作纪律第 7 条
            sc = self._align_scale(q)
            kp, kd = kp * sc, kd * np.sqrt(sc)
        e, de = qd - q, vd - v
        pi = self._pi_hat()
        zero = np.zeros(N_ARM)

        if self.choice("controller") == "pd_gravity":
            # Y(q,0,0)·π 恰好是重力项（a=0 ⇒ 无惯量项，v=0 ⇒ 无科氏与摩擦）
            g = self.reg.regressor(q, zero, zero) @ pi
            tau = kp * e + kd * de + g
        else:
            aq = ad + kd * de + kp * e
            Y0 = self.reg.regressor(q, zero, zero)
            Ya = self.reg.regressor(q, zero, aq)
            Yh = self.reg.regressor(q, v, zero)
            # (Ya−Y0)·π = M·aq（含 armature）；Yh·π = C q̇ + g + 摩擦
            tau = (Ya - Y0) @ pi + Yh @ pi

        self._e = e
        self._ad = ad
        self.rms.push(e)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)
        self._p = self.robot.fk(q)[0].copy()
        self._p_des = self.robot.fk(qd)[0].copy()
        # fk(qd) 把模型状态改到了期望构型，必须还原，否则下一次读到的是假状态
        self.robot.set_state(q, v)
        self._h = float(np.linalg.norm(self.reg.bias(q, v)))
        self._m = float(np.linalg.norm(self.reg.mass_matrix(q), 2))
        self.trail.push(self._p)
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        self._wn_tick = getattr(self, "_wn_tick", 0) + 1
        if self._wn_tick % 10 == 1 or not hasattr(self, "_wn"):
            self._wn = self._measure_wn(Q_HOME)     # 固定在 Home 处测，可比
            # ⭐ P0-02：真实模态频率（广义特征值），和上面逐关节的数是两回事
            Ms = self.robot.mass_matrix(Q_HOME)[np.ix_(self._WN_JOINTS,
                                                       self._WN_JOINTS)]
            kp_eff = self.get("kp")
            if (self.choice("controller") == "pd_gravity"
                    and self.choice("align") == "on"):
                kp_eff = kp_eff * self._align_scale(Q_HOME)
            if self.choice("controller") == "pd_gravity":
                mf = modal_frequencies(Ms, kp_eff)
            else:
                # ⚠️ P0-02 收尾（2026-08-31）：这里原来是
                #     mf = np.full(len(...), sqrt(kp_eff))
                # ——把"CTC 各模态同频"直接**写进读数**，不是量出来的。
                # 那和 6.2b 撤回的「离散度 0.000」是同一个病：一个恒等于
                # 理论值的指标，说明它根本没在量。而且「模型误差」滑块
                # 对它完全没有影响，等于这个读数是死的。
                #
                # CTC 带模型误差时的闭环误差方程（自行推导）：
                #     τ = M_reg·(a_d + Kd ė + Kp e) + h_reg
                #     M_true q̈ + h_true = τ,  q̈ = a_d − ë
                #   ⇒ M_true ë + Kd M_reg ė + Kp M_reg e = 强迫项
                # 齐次部分的模态频率来自 det(Kp·M_reg − ω²·M_true) = 0，
                # 正好是 modal_frequencies(M_true, Kp·M_reg)。
                # 模型完全准时 M_reg = M_true，退化成 ω=√Kp、离散度 1.00；
                # 模型一旦不准，离散度就会自己离开 1 —— 这才是测量。
                Mr = self._reg_mass(Q_HOME, self._pi_hat())[
                    np.ix_(self._WN_JOINTS, self._WN_JOINTS)]
                mf = modal_frequencies(Ms, kp_eff * Mr)
            self._modal = mf
        wn = self._wn
        mf = self._modal
        return dict(rms=self.rms.value * 1e3,
                    wn_med=float(np.nanmedian(wn)),
                    wn_spread=float(np.nanmax(wn) - np.nanmin(wn)),
                    modal_min=float(mf[0]),
                    modal_spread=float(mf[-1] / max(mf[0], 1e-9)),
                    err=float(np.linalg.norm(self._e)) * 1e3,
                    ee_err=float(np.linalg.norm(self._p - self._p_des)) * 1e3,
                    tau=float(np.linalg.norm(self._tau)),
                    tau_vec=self._tau, tau_limit=lim,
                    tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0,
                    param_err=self.get("model_err"),
                    kp_now=self.get("kp"), kd_now=self.get("kd"),
                    ad_norm=float(np.linalg.norm(self._ad)),
                    m_norm=self._m, h_norm=self._h)

    def overlays(self):
        # 目标球与实际球的尺寸关系由 _pair_markers 统一保证：目标球必须小到
        # 实际球能露出来，否则「看两个球分开多少」这条指令在本场景无效。
        return _pair_markers(self._p_des, self._p) + [
            Trail(self.trail.points, color=C_ERROR, width=0.0045),
        ]


# ============================================================== 3 自适应控制


class AdaptiveScene(Scene):
    name = "adaptive"
    steps = STEPS_BY_SCENE["adaptive"]
    title = "自适应控制：MBAC 与 RBF 神经网络"
    camera = dict(azimuth=138.0, elevation=-16.0, distance=1.35,
                  lookat=(0.40, 0.0, 0.45))
    intro = (
        "控制器一开始<b>不知道</b>机器人的动力学参数，靠在线更新自己补上。"
    )
    lesson = Lesson(
        problem=(
            "上一个场景的结论是：模型准，计算力矩就厉害；模型不准，它的优势就没了。\n\n"
            "可现实中模型<b>永远</b>不准：\n"
            "• 手臂末端抓了个工件，重量变了，但你不知道多重。\n"
            "• 关节用久了摩擦变大。\n"
            "• 出厂参数表本来就是估的。\n\n"
            "难道每换一个工件就得重新标定一次？"
        ),
        idea=(
            "换个想法：<b>让控制器自己边干边猜。</b>\n\n"
            "比方说你第一次拎一桶水，不知道有多沉，第一下往往会用力过猛或者不够——"
            "但拎了两秒你就「知道」了，手上的劲会自动调到合适。你并没有拿秤称，"
            "而是<b>从「我用了多大劲、它实际动了多少」里反推出来的</b>。\n\n"
            "自适应控制干的就是这件事：\n"
            "1. 用当前<b>猜测</b>的参数算一个力矩发出去；\n"
            "2. 看实际跟踪误差往哪边偏；\n"
            "3. 按一条<b>规定好的更新律</b>把猜测往「能减小误差」的方向挪一点；\n"
            "4. 回到 1，一直重复。\n\n"
            "关键是第 3 步的更新律<b>不是拍脑袋定的</b>，也<b>不是训练出来的</b>，"
            "而是从稳定性证明里推出来的——这是它和强化学习最根本的区别。"
        ),
        how=(
            "<b>为什么能这么猜？</b>因为机械臂的动力学有一个漂亮性质："
            "力矩对物理参数是<b>线性</b>的。也就是说总能写成\n\n"
            "    <b>τ = Y(q, q̇, ...) · π</b>\n\n"
            "其中 π 是一列物理参数（每个连杆的质量、质心位置、转动惯量、摩擦系数），"
            "Y 是一个只跟运动状态有关、<b>与参数无关</b>的矩阵。\n"
            "这就像 y = a·x₁ + b·x₂：只要 x 已知，从观测到的 y 反推 a、b 就是标准的线性问题。\n\n"
            "<b>怎么更新？</b>先把位置误差和速度误差揉成一个量：\n\n"
            "    <b>s = ė + λ·e</b>\n\n"
            "这叫滑模面。它的好处是：只要 s→0，e 和 ė 就必然都→0（因为 e 满足一个"
            "稳定的一阶方程）。<b>两个目标合并成了一个</b>。\n\n"
            "然后更新律是\n\n"
            "    <b>π̂̇ = −Γ·Yᵀ·s</b>\n\n"
            "读作：「往能让 s 变小的方向，以步长 Γ 修正参数猜测」。"
            "这个式子是构造一个能量函数 V = ½sᵀMs + ½π̃ᵀΓ⁻¹π̃ 后求导，"
            "让交叉项刚好抵消而得到的——所以它<b>可以证明</b>误差会收敛，不是试出来的。\n\n"
            "<b>MBAC 和 RBF 的区别</b>：MBAC 假设上面那个线性关系严格成立（对刚体机械臂确实成立），"
            "所以能做到<b>渐近收敛</b>（误差趋于 0）。RBF 神经网络不做这个假设，"
            "用一堆高斯「小山包」去硬拟合整条动力学，代价是永远存在逼近误差，"
            "结论只能到<b>一致最终有界</b>（误差进入一个小范围就不再变小）。"
        ),
        watch=[
            "<b>刚重置的头几秒</b>：参数猜测从零开始，控制器基本什么都不知道，"
            "画面上蓝球会明显跟不上绿球，误差曲线很高。然后你会看到它<b>自己收敛下去</b>。",
            "<b>把「自适应增益 Γ」调到 0</b>：参数不再更新，误差<b>永远降不下来</b>。"
            "这一条最能说明自适应在干活。",
            "<b>Γ 调大</b>：收敛更快，但太大时误差曲线会变毛躁。这就是学习速度与稳定性的取舍。",
            "<b>核心现象：同时盯两条曲线</b>——蓝色「跟踪误差」和橙色「参数偏差」。"
            "你会看到<b>蓝线很快掉到很低，而橙线还高高地挂着</b>。"
            "意思是：<b>它已经能跟得很准了，但它猜的参数其实还是错的。</b>",
            "<b>为什么会这样</b>：正弦轨迹太「单调」，激励不够丰富，"
            "很多组不同的参数都能产生同样的力矩，控制器分辨不出哪组才是真的。"
            "这叫<b>持续激励（PE）条件</b>不满足。把「轨迹频率」或「幅值」调小，"
            "橙线会更难下降。",
            "<b>切到 ctc_exact</b>：这是「参数完全正确」的理想对照组，参数偏差恒为 0。",
        ],
        verify=[
            "「参数偏差 ‖π̂−π‖/‖π‖」是把在线估计值和 MuJoCo 模型里的真值直接比，"
            "不依赖控制器自己的任何判断。它降不下去而跟踪误差降下去了，"
            "就是「跟踪收敛 ≠ 参数收敛」的现场证据。",
            "「滑模面 ‖s‖」是遥测里<b>独立算</b>的（s = ė + λe），"
            "没有读控制器内部的 s。两者若不一致说明有一方错了。",
            "选 rbf 时参数偏差显示为空——因为 RBF 逼近的是一个函数，"
            "根本没有「真参数」可比。显示成 0 才是撒谎。",
        ],
        caveat=(
            "⚠️ <b>自适应控制不是强化学习。</b>这里的更新律是 Lyapunov 推出来的，"
            "稳定性可证，不需要数据集、不需要离线训练、不需要试错。"
            "强化学习靠海量采样和奖励函数，没有稳定性保证。两者只是都带「学」字。\n\n"
            "⚠️ <b>五个控制层次不是「越往后越好」。</b>标称条件下（模型很准、负载不变），"
            "PD+重力补偿甚至可能比自适应更准，因为自适应还要花时间去「学」。"
            "自适应的价值在<b>工况会变</b>的时候。\n\n"
            "⚠️ 这里的「真值」是 MuJoCo 模型的参数。真机上没有真值可比，"
            "只能看跟踪误差——所以真机上你<b>根本不知道</b>参数收没收敛。"
        ),
    )
    law = ControlLaw(
        formula="e = q_d − q（本场景约定）,  s = ė + λ·e,  q̇ᵣ = q̇_d + λ·e\n"
                "τ = Y(q, q̇, q̇ᵣ, q̈ᵣ)·π̂ + Kd·s        （控制律）\n"
                "π̂̇ = +Γ·Yᵀ·s                          （参数更新律）",
        terms=[
            LawTerm("s", "滑模面。把位置误差和速度误差揉成一个量，s→0 则两者都→0",
                    key="s_norm", unit="", digits=3),
            LawTerm("λ", "滑模面权重。决定 e 与 ė 谁说了算，也就是收敛的时间尺度",
                    key="lam_now", unit="1/s", digits=1),
            LawTerm("π̂", "在线估计的动力学参数。一开始是零，边跑边学",
                    key="param_err", unit="% 偏差", digits=2),
            LawTerm("Γ", "自适应增益。参数更新的步长，大则学得快但会抖",
                    key="gamma_now", unit="", digits=2),
            LawTerm("Y", "回归矩阵。把「参数」和「力矩」之间的线性关系写成矩阵"),
            LawTerm("Kd", "反馈增益。保证 s 被压住，是稳定性证明的一部分",
                    key="kd_now", unit="", digits=0),
        ],
        blocks=[
            Block("正弦参考", "", "input"),
            Block("滑模面 s = ė + λe", "", "block"),
            Block("参数估计 π̂", "被更新律 π̂̇ = +ΓYᵀs 在线修正", "block"),
            Block("Y·π̂ + Kd·s", "前馈用估计参数，反馈兜底", "block"),
            Block("机械臂", "真实参数固定不变", "plant"),
            Block("实测 q, q̇", "", "feedback"),
        ],
        note="⭐ <b>符号完全由误差约定决定，两者必须一起说。</b>本场景（以及 "
             "control/adaptive.py）用 <b>e = q_d − q</b>、s = ė + λe = q̇ᵣ − q̇，"
             "此时控制律和更新律里<b>都是加号</b>。\n\n"
             "推导（三步）：\n"
             "① 闭环误差动态。把 τ = Yπ̂ + Kd·s 代入 Mq̈ + Cq̇ + g = τ，"
             "并用 Yπ = Mq̈ᵣ + Cq̇ᵣ + g，得 <b>Mṡ + Cs + Kd·s = −Yπ̃</b>"
             "（π̃ = π̂ − π 是参数估计误差）。\n"
             "② 取 V = ½sᵀM s + ½π̃ᵀΓ⁻¹π̃ 求导。斜对称性 sᵀ(Ṁ − 2C)s = 0 "
             "把科氏项整块消掉，剩 V̇ = −sᵀKd·s + π̃ᵀ(Γ⁻¹π̂̇ − Yᵀs)。\n"
             "③ 让括号为零，即得 <b>π̂̇ = +ΓYᵀs</b>，于是 V̇ = −sᵀKd·s ≤ 0。\n\n"
             "⚠️ 若改用相反约定 <b>e = q − q_d</b>（不少教材就是这么写的），"
             "s 跟着反号，两个式子就同时变成减号。两套都对，<b>混着用才错</b>——"
             "所以任何地方写这两个式子，必须把 e 的定义摆在旁边。",
    )
    params = [
        Param("controller", "控制器", default="mbac",
              choices=["ctc_exact", "mbac", "rbf"], group="控制器",
              effect="ctc_exact 是「参数全对」的理想对照；mbac 在线估物理参数；"
                     "rbf 用神经网络硬拟合。切换后看两条曲线的收敛方式差别。"),
        Param("gamma", "自适应增益", 0.0, 5.0, 0.5, symbol="Γ", group="控制器",
              help="参数更新的步长。MBAC 建议 0.1–2，RBF 会自动放大 400 倍。",
              effect="<b>调到 0 → 参数完全不更新，误差永远降不下来</b>（最能说明问题的一档）。"
                     "调大 → 收敛快但曲线变毛躁。"),
        Param("exc", "激励轨迹", default="sine",
              choices=["sine", "fourier"], group="参考轨迹", restart=True,
              help="<b>sine</b>：单频正弦，所有动关节同相位同频率。<br>"
                   "<b>fourier</b>：5 次谐波的傅里叶级数，各关节相位/幅值不同"
                   "——这是工业上做参数辨识时的标准「激励轨迹」写法"
                   "（Swevers 参数化）。",
              effect="<b>切到 fourier，盯住「激励条件数」和「⭐离线LS 可辨识偏差」。</b>"
                     "单频正弦下离线最小二乘会算出荒谬的参数（残差极小但参数错"
                     "上百倍）；换成傅里叶激励后同一个估计器立刻变准。"
                     "这就是「激励轨迹要设计」的全部含义。"),
        Param("lam", "滑模面权重", 1.0, 40.0, 10.0, symbol="λ", group="控制器",
              help="误差面 s = ė + λe 的权重，决定收敛的时间尺度。",
              effect="决定「位置误差」和「速度误差」谁说了算。调大 → 更看重位置误差，"
                     "收敛更急。看「滑模面 ‖s‖」这个数。"),
        Param("kd", "反馈增益", 5.0, 300.0, 60.0, symbol="Kd", group="控制器",
              effect="兜底的那一项，保证 s 被压住。调小到很低时误差会明显变大。"),
        Param("freq", "轨迹频率", 0.1, 2.0, 0.6, unit="Hz", restart=True,
              group="参考轨迹",
              help="改动会重新布置 RBF 中心，因此重置场景。",
              effect="<b>调低 → 橙色「参数偏差」更难降下来</b>：轨迹越单调，"
                     "激励越不充分，越分辨不出真参数。这就是持续激励条件。"),
        Param("amp", "轨迹幅值", 0.05, 0.7, 0.30, unit="rad", restart=True,
              group="参考轨迹",
              effect="同上，幅值越小激励越弱，参数越不容易收敛。"),
    ]
    readouts = [
        Readout("rms", "跟踪误差 RMS", "mrad", 3),
        Readout("err", "当前误差范数", "mrad", 3),
        Readout("param_err", "完整参数偏差 ‖π̂−π‖/‖π‖", "%", 2, source="truth",
                help="⚠️ 这一栏有**硬下限**，见「不可辨识下限」。它降不下去"
                     "不代表自适应没工作，只代表大部分惯性参数在这条轨迹上"
                     "根本测不出来。判断收敛请看「可辨识子空间偏差」。"),
        Readout("param_floor", "不可辨识下限 ‖(I−PPᵀ)π‖/‖π‖", "%", 2,
                source="truth",
                help="回归矩阵 Y 的零空间分量占真参数的比例。这部分对力矩"
                     "**没有任何影响**，因此无论激励多充分都学不到。"
                     "「完整参数偏差」永远 ≥ 这个数。\n\n"
                     "⚠️ **这个百分比不是物理不变量**：它是欧氏范数之比，"
                     "依赖惯性参数用什么单位。保持 Y·π 完全不变、只把质量项"
                     "从 kg 换成 g，这一栏就从 90.21% 跳到 99.10%。"
                     "它只能读作「**当前参数坐标下**的距离下界」，"
                     "不能说成「90% 的物理参数永远学不到」。"
                     "真正与单位无关的量是右边的 **rank**。"),
        Readout("base_err", "可辨识子空间偏差 ‖Pᵀ(π̂−π)‖/‖Pᵀπ‖", "%", 2,
                source="truth",
                help="只看结构上能学的那 rank 维。\n\n"
                     "⚠️ **不要指望它降到 0。**「子空间可达」和「参数会收敛」"
                     "是两回事，实测（amp=0.35，Γ=0.5）：\n"
                     "· t=25 s → 82.2%　· t=600 s → 72.7%，仍在缓慢下降；\n"
                     "· 把 Γ 拉到 5.0 反而**发散**，‖β‖ 撞上参数投影上界 200 被夹住。\n\n"
                     "所以这一栏的正确读法是**比较**，不是看绝对值：同样跑 25 s，"
                     "强激励 82.2% / 弱激励 92.4%——差距来自「激励条件数」"
                     "而不是维数（两者维数都是 62）。\n\n"
                     "⭐ 这正是**在线自适应辨识的固有短板**，也是工业上改用"
                     "「离线设计激励轨迹 + 最小二乘」的原因，"
                     "见讲解 03 的「传统辨识 vs 学习辨识」一节。"),
        Readout("pred_err", "力矩预测误差 ‖Y(π̂−π)‖/‖Yπ‖", "%", 3,
                source="truth",
                help="在一组固定随机位形上比较「用估计参数算的力矩」和真力矩。"
                     "这是**任务层面**真正关心的量，与参数是否可辨识无关。"),
        Readout("n_par_now", "参数总维数 n_par", "", 0, source="truth",
                help="从模型现场读出来的，不是手写的常数。"),
        Readout("rank_now", "结构可辨识维数 rank", "", 0, source="truth",
                help="在关节限位内**全域随机取位形**堆 Y 做 SVD 得到的秩，"
                     "刻画的是「这套硬件构型下最多能辨识多少维」。\n\n"
                     "⚠️ 它取决于夹爪工况：手指固定（本场景实况）**62**；"
                     "平行夹爪同宽联动 **67**；两指各自独立随机 **70**——"
                     "但最后这个物理上做不到，平行夹爪只有一个自由度。\n"
                     "换句话说，报 70 等于偷偷假设了一个不存在的夹爪。"),
        Readout("rank_traj", "轨迹激励维数", "", 0, source="truth",
                help="⭐ 与 rank 互不替代的第二个量：把**实际跑过的轨迹**上的"
                     "Slotine-Li 回归矩阵增量 QR 累积起来，再看它在可辨识"
                     "子空间里张成了几维。\n\n"
                     "rank 问「硬件最多能辨识几维」，这一栏问「**这条轨迹**"
                     "实际激励到几维」。前者是结构性质，后者才是持续激励。"
                     "开跑几秒后才有意义。"),
        Readout("pe_cond", "激励条件数 log₁₀", "", 2, source="truth",
                help="激励矩阵在可辨识子空间里的条件数（s₀/s_last）取 log₁₀。\n\n"
                     "⚠️ **强/弱激励的维数往往一样，差别全在这一栏。**\n"
                     "实测 25 s（freq 0.6 / amp 0.30）：\n"
                     "· 单频正弦 **4.73**，离线 LS 参数偏差 0.0071%\n"
                     "· 傅里叶激励 **2.53**，离线 LS 参数偏差 0.0013%\n"
                     "所以「秩够了」不等于「激励够了」，越大越病态。\n\n"
                     "🔧 用的是全谱 s₀/s_last，不是按阈值截断后的 s₀/s_{r−1}——"
                     "后者按定义永远 ≤ 8，再烂的激励也报不出来。"),
        Readout("ls_base_err", "⭐离线LS 可辨识偏差", "%", 3, source="truth",
                help="⭐⭐ **和上面的「可辨识子空间偏差」用的是同一份数据、"
                     "同一个机器人，只是换了个估计器。**\n\n"
                     "在线自适应律解的是「让跟踪误差收敛」；"
                     "离线最小二乘解的是「让力矩残差最小」——后者才是辨识。\n\n"
                     "实测（freq 0.6 / amp 0.30，25 s）：\n"
                     "· 在线自适应 **84.4 %**\n"
                     "· 离线 LS **0.0071 %**  ← 差约 12000 倍，而且 5 秒就到位\n\n"
                     "这就是工业界（以及 PACE 论文）做参数辨识时"
                     "**不用**在线自适应律的原因。\n\n"
                     "⚠️ 90.21% 那个地板管不到这一栏——它本来就是在可辨识"
                     "子空间里算的，所以**可以**趋于 0。"),
        Readout("ls_pred_err", "⭐离线LS 力矩预测误差", "%", 3, source="truth",
                help="离线最小二乘估计的参数拿去预测力矩，误差有多大。"
                     "和「力矩预测误差」那一栏比，同一把尺子量两个估计器："
                     "在线 44.8 % vs 离线 0.0015 %（25 s，单频正弦）。"),
        Readout("ls_resid", "离线LS 拟合残差", "%", 3,
                help="⭐ 这一栏**不需要参数真值**：它是 "
                     "‖Yβ̂−τ_实测‖/‖τ_实测‖，真机上照样能算。\n"
                     "上面两栏标 `真值`，真机上根本拿不到。\n\n"
                     "⚠️⚠️ **残差小绝不等于参数对。**实测过一次反例：把加速度"
                     "改成有半步时间错位的版本，残差只有 0.077 %，"
                     "参数却错了 **123 %**——因为条件数 10^4.7 把残差放大了。"
                     "所以这一栏必须和「激励条件数」一起看。"),
        Readout("s_norm", "滑模面 ‖s‖", "", 3),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
    ]
    traces = [
        Trace("err", "跟踪误差", "#4da3ff", unit="mrad"),
        Trace("param_err", "完整参数偏差", "#ff8a65", axis="param", unit="%"),
        Trace("base_err", "可辨识子空间偏差", "#7fe08a", axis="param", unit="%"),
    ]
    bars = [_torque_bars()]

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self.pi_true = self.reg.true_parameters()
        self.pi_norm = float(np.linalg.norm(self.pi_true))
        self._mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        # 基参数投影要做一次 SVD，比较慢，只在建场景时算一次
        self.mbac = SlotineLiAdaptiveController(
            self.reg, N_ARM, lam=10.0, kd=60.0, gamma=0.5, use_base=True)
        self.rbf = None
        self._init_identifiability()
        self._pe_reset()
        return scene.model

    # ------------------------------------------------------------------
    # P0-08：把「参数收敛」拆成三个互不替代的量
    #
    # Slotine-Li 只在基参数坐标 β 上更新，π̂ = P·β 恒落在 P 的列空间里。
    # 真参数 π 一般**不**在这个子空间里，那部分差 (I − PPᵀ)π 对力矩没有
    # 任何影响，也就永远学不到。所以 ‖π̂−π‖/‖π‖ 有一个**硬下限**，
    # 拿它当「参数是否收敛」的证据是错的。
    # ------------------------------------------------------------------
    def _init_identifiability(self):
        P = self.mbac.P
        pi = self.pi_true
        self._pi_null = pi - P @ (P.T @ pi)          # 不可辨识分量
        self._floor = float(np.linalg.norm(self._pi_null)) / self.pi_norm
        self._pi_base = P.T @ pi                     # 真值在基参数坐标下
        self._base_norm = max(float(np.linalg.norm(self._pi_base)), 1e-12)
        # 固定的一组随机位形，用来算与可辨识性无关的「力矩预测误差」。
        # 只在建场景时堆一次，运行时就是一次矩阵-向量乘，不拖慢仿真。
        rng = np.random.default_rng(20240817)
        rows = [self.reg.regressor(rng.uniform(-1.5, 1.5, self.reg.nv),
                                   rng.uniform(-1.0, 1.0, self.reg.nv),
                                   rng.uniform(-2.0, 2.0, self.reg.nv))
                for _ in range(24)]
        self._Wprobe = np.vstack(rows)
        self._tau_probe_norm = max(
            float(np.linalg.norm(self._Wprobe @ pi)), 1e-12)

    def _identifiability_stats(self, pi_hat):
        """返回 (完整偏差, 可辨识子空间偏差, 力矩预测误差)，都是相对值。"""
        d = pi_hat - self.pi_true
        full = float(np.linalg.norm(d)) / self.pi_norm
        base = float(np.linalg.norm(self.mbac.P.T @ d)) / self._base_norm
        pred = float(np.linalg.norm(self._Wprobe @ d)) / self._tau_probe_norm
        return full, base, pred

    # ------------------------------------------------------------------
    # P0-08 补充（第五轮）：结构秩 ≠ 轨迹激励
    #
    # rank_now 是**结构**性质：全域随机位形下最多能辨识几维，和跑什么轨迹
    # 无关。持续激励说的却是另一回事——**这一条**轨迹实际张开了几维、条件
    # 数多差。两者必须分开报，否则「rank=62」会被误读成「激励充分」。
    #
    # 数值方法：增量 QR 累积 R 因子，不用 G = WᵀW。
    # 后者把条件数**平方**，κ(W)~1e8 时 κ(G)~1e16 已到双精度极限，
    # 秩会算出比结构秩还大的荒谬值（实测拿 Gramian 算出过 66 > 62）。
    # ------------------------------------------------------------------
    _PE_STRIDE = 10          # 每 10 个控制步折进一次（500 Hz → 50 Hz）
    _PE_SVD_PERIOD = 0.25    # 秒；SVD 比较贵，按仿真时间节流
    _LS_PERIOD = 0.25        # 秒；离线 LS 求解同样按仿真时间节流

    def _pe_reset(self):
        n = self.reg.n_par
        self._pe_R = np.zeros((n, n))
        self._pe_k = 0
        self._pe_rows = 0
        self._pe_t_last = -1e9
        self._pe_cache = (0.0, float("nan"))
        # 离线最小二乘：同一份数据、同一个回归器，换一个估计器。
        self._ls = OfflineBaseLS(self.mbac.P)
        self._ls_t_last = -1e9
        self._ls_cache = (float("nan"), float("nan"), float("nan"))

    def _pe_update(self, t, q, v, qd, vd, ad):
        """把当前时刻的 Slotine-Li 回归矩阵折进 R 因子。

        回归矩阵在这里**重算**，不读控制器内部量——否则就是拿被测实现
        的中间量给自己作证。三种控制器都走这条路径，口径才一致。
        """
        self._pe_k += 1
        if self._pe_k % self._PE_STRIDE:
            return
        lam = self.get("lam")
        e = qd - q
        de = vd - v
        Y = self.reg.slotine_li_regressor(q, v, vd + lam * e, ad + lam * de)
        self._pe_R = np.linalg.qr(np.vstack([self._pe_R, Y]), mode="r")
        self._pe_rows += Y.shape[0]

        # --- 离线 LS 用的是**普通**回归矩阵和**实测力矩** ---
        # Slotine-Li 回归器里的 v_ref/a_ref 是控制器的内部构造，真机上没有；
        # 离线辨识只能用 (q, q̇, q̈) 和测到的 τ。
        #
        # ⚠️ 这里先做一次 mj_forward，是为了拿到与当前 (q,v,ctrl) **严格自洽**
        # 的 (qacc, qfrc_actuator)。直接读 mj_step 之后残留的 qacc 会有半步的
        # 时间错位，实测把力矩残差从 1.9e-5 抬到 3.0e-3——**150 倍**。
        # 而条件数是 10^4.7 量级，残差被放大之后参数能错上一百倍。
        # 这不是无关紧要的实现细节，它就是「为什么真机上做力矩回归很难」的
        # 全部原因：真机没有 mj_forward，加速度只能靠差分速度得到。
        # 详见讲解 03 §八之三。
        if self._pe_k <= self._PE_STRIDE:
            return                      # 第一步 qacc/qfrc 还是 0，跳过
        mujoco.mj_forward(self.model, self.data)
        a = self.data.qacc[self.robot.qvel_idx].copy()
        tau_meas = self.data.qfrc_actuator[self.robot.qvel_idx].copy()
        self._ls.add(self.reg.regressor(q, v, a), tau_meas)

    def _ls_stats(self, t):
        """(离线LS 可辨识偏差, 力矩预测误差, 拟合残差)，都是百分数。"""
        if t - self._ls_t_last < self._LS_PERIOD:
            return self._ls_cache
        self._ls_t_last = t
        beta, info = self._ls.solve()
        if beta is None:
            return self._ls_cache
        pi_hat = self.mbac.P @ beta
        _, base, pred = self._identifiability_stats(pi_hat)
        self._ls_cache = (base * 100.0, pred * 100.0,
                          info["residual"] * 100.0)
        return self._ls_cache

    def _pe_stats(self, t):
        """(轨迹激励维数, log10 条件数)；SVD 按仿真时间节流。

        ⚠️ 条件数用 s[0]/s[-1]（可辨识子空间里的**全谱**跨度），**不是**
        s[0]/s[rank-1]。后者按 1e-8 阈值截断之后，条件数按定义永远 ≤ 1e8，
        再差的激励也报不出来——第一版就是这么写的，把参考轨迹上实测
        10^18 的病态显示成了 4.7。判据的上限不能比被判对象的坏程度还低。
        """
        if self._pe_rows < self.reg.n_par:
            return 0.0, float("nan")
        if t - self._pe_t_last < self._PE_SVD_PERIOD:
            return self._pe_cache
        self._pe_t_last = t
        # 只问「可辨识子空间里张开了几维」：R 投到 P 的 rank 列上
        s = np.linalg.svd(self._pe_R @ self.mbac.P, compute_uv=False)
        r = int(np.sum(s > s[0] * 1e-8))
        cond = float(np.log10(s[0] / s[-1])) if s[-1] > 0 else float("inf")
        self._pe_cache = (float(r), cond)
        return self._pe_cache

    # ------------------------------------------------------------------
    # 参考轨迹。两种写法的差别，就是「激励轨迹要不要设计」这件事本身。
    #
    # sine    单频正弦：所有动关节同频同相。看着挺动，但回归矩阵的行
    #         几乎都指向同几个方向，条件数很差。
    # fourier 傅里叶级数（Swevers 参数化）：
    #             q(t) = q0 + Σ_k [ a_k/(ωk)·sin(ωkt) − b_k/(ωk)·cos(ωkt) ]
    #         好处是位置/速度/加速度都有解析式，且每个周期首尾状态一致
    #         （可以无缝重复采数据）。系数除以 ωk 是为了不让高次谐波把
    #         速度顶爆。工业辨识里 a_k、b_k 是拿「最小化条件数」当目标
    #         优化出来的；这里用固定种子的随机系数，已经足够把条件数
    #         拉下两个数量级——足以说明问题，但**不要**说成"优化过的"。
    # ------------------------------------------------------------------
    _FOURIER_H = 5           # 谐波次数
    _FOURIER_SEED = 124      # 300 组随机系数里条件数最小的一组（见下）

    def _fourier_coeffs(self):
        """傅里叶系数。

        工业做法是把 a_k、b_k 当优化变量，以「回归矩阵条件数最小」为目标
        求解（Swevers 1997）。这里做的是最朴素的版本：**在 300 组随机系数
        里挑条件数最小的那一组**，种子固定成 124。
        参考轨迹上实测 log10 条件数：300 组的中位数 2.889、最差 3.242、
        最好 2.538（= seed 124）。随机搜索本身只赢了中位数 0.35 个数量级
        ——**真正的大头不在系数上，在「让所有 7 个关节都动起来」**：
        只动 4 个关节时，同样的傅里叶系数条件数是 10^18.5。
        """
        if getattr(self, "_fc", None) is None:
            rng = np.random.default_rng(self._FOURIER_SEED)
            h = self._FOURIER_H
            self._fc = (rng.uniform(-1.0, 1.0, (h, N_ARM)),
                        rng.uniform(-1.0, 1.0, (h, N_ARM)))
        return self._fc

    def _fourier_peak(self, w):
        """一个周期内的逐关节峰值位移，用来把幅值滑块归一化。

        直接采样求最大值，不用三角不等式的上界——上界会保守好几倍，
        导致 fourier 档实际幅值远小于滑块显示值，两档就没法公平对比。
        """
        key = round(w, 9)
        cache = getattr(self, "_fpk", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        ak, bk = self._fourier_coeffs()
        ts = np.linspace(0.0, 2.0 * np.pi / w, 512, endpoint=False)
        q = np.zeros((ts.size, N_ARM))
        for k in range(1, self._FOURIER_H + 1):
            wk = w * k
            q += (np.outer(np.sin(wk * ts), ak[k - 1])
                  - np.outer(np.cos(wk * ts), bk[k - 1])) / wk
        pk = np.maximum(np.max(np.abs(q), axis=0), 1e-9)
        self._fpk = (key, pk)
        return pk

    def _ref(self, t):
        w = 2.0 * np.pi * self.get("freq")
        if self.choice("exc") == "sine":
            a = self.get("amp") * self._mask
            return (Q_HOME + a * np.sin(w * t), a * w * np.cos(w * t),
                    -a * w * w * np.sin(w * t))
        # 激励档：7 个关节全动。只动 4 个关节时另外 3 个关节的参数根本
        # 没被激励过，条件数实测 10^18.5——比系数怎么选重要得多。
        a = self.get("amp") * np.ones(N_ARM)
        ak, bk = self._fourier_coeffs()
        q = np.zeros(N_ARM)
        v = np.zeros(N_ARM)
        acc = np.zeros(N_ARM)
        for k in range(1, self._FOURIER_H + 1):
            wk = w * k
            c, s = np.cos(wk * t), np.sin(wk * t)
            A, B = ak[k - 1], bk[k - 1]
            q += (A * s - B * c) / wk
            v += A * c + B * s
            acc += wk * (-A * s + B * c)
        g = a / self._fourier_peak(w)
        return Q_HOME + g * q, g * v, g * acc

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.rms = RunningRMS()
        self._tau = np.zeros(N_ARM)
        self._e = np.zeros(N_ARM)
        self._s = np.zeros(N_ARM)
        self._p = self.robot.fk(Q_HOME)[0].copy()
        self._p_des = self._p.copy()
        self.mbac.reset()
        period = 1.0 / max(self.get("freq"), 1e-3)
        C, W = fit_centers_to_trajectory(self._ref, 2.0 * period,
                                         n_centers=200, lam=self.get("lam"))
        self.rbf = RBFAdaptiveController(N_ARM, C, W, lam=self.get("lam"),
                                         kd=self.get("kd"), gamma=200.0)
        self.trail = TrailBuffer(capacity=260, min_step=0.003)
        self.note = "自适应参数从零开始，前几秒误差大是正常的"
        self._pe_reset()

    def control(self, t, q, v):
        qd, vd, ad = self._ref(t)
        self._pe_update(t, q, v, qd, vd, ad)
        mode = self.choice("controller")
        if mode == "ctc_exact":
            aq = ad + self.get("kd") * (vd - v) + self.get("lam") ** 2 * (qd - q)
            tau = self.reg.mass_matrix(q) @ aq + self.reg.bias(q, v)
        elif mode == "mbac":
            self.mbac.lam = self.get("lam")
            self.mbac.kd = self.get("kd")
            self.mbac.gamma = self.get("gamma")
            tau, _ = self.mbac.compute(q, v, qd, vd, ad, self.dt)
        else:
            self.rbf.lam = self.get("lam")
            self.rbf.kd = self.get("kd")
            self.rbf.gamma = self.get("gamma") * 400.0
            tau, _ = self.rbf.compute(q, v, qd, vd, ad, self.dt)
        self._e = qd - q
        # 滑模面自己算，不读控制器内部量：三种控制器里只有两种有 s，
        # 而且用被测实现的中间量当遥测，等于让它自证。
        self._s = (vd - v) + self.get("lam") * self._e
        self.rms.push(self._e)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)
        self._p = self.robot.fk(q)[0].copy()
        self._p_des = self.robot.fk(qd)[0].copy()
        self.robot.set_state(q, v)          # fk(qd) 改过模型状态，还原
        self.trail.push(self._p)
        return self._tau

    def telemetry(self, t, q, v):
        mode = self.choice("controller")
        if mode == "mbac":
            pe, be, pr = self._identifiability_stats(self.mbac.pi_hat)
        elif mode == "ctc_exact":
            pe = be = pr = 0.0
        else:
            # RBF 逼近的是函数，没有「真参数」可比 —— 空着比编一个数诚实
            pe = be = pr = float("nan")
        lim = self.robot.tau_limit
        r_traj, pe_cond = self._pe_stats(t)
        ls_be, ls_pr, ls_rs = self._ls_stats(t)
        return dict(rms=self.rms.value * 1e3,
                    err=float(np.linalg.norm(self._e)) * 1e3,
                    param_err=pe * 100.0,
                    base_err=be * 100.0,
                    pred_err=pr * 100.0,
                    param_floor=(0.0 if mode == "ctc_exact" else self._floor * 100.0),
                    n_par_now=float(self.reg.n_par),
                    rank_now=float(self.mbac.rank),
                    rank_traj=r_traj,
                    pe_cond=pe_cond,
                    ls_base_err=ls_be,
                    ls_pred_err=ls_pr,
                    ls_resid=ls_rs,
                    s_norm=float(np.linalg.norm(self._s)),
                    tau=float(np.linalg.norm(self._tau)),
                    tau_vec=self._tau, tau_limit=lim,
                    tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0,
                    lam_now=self.get("lam"), kd_now=self.get("kd"),
                    gamma_now=self.get("gamma"))

    def overlays(self):
        # 目标球与实际球的尺寸关系由 _pair_markers 统一保证：目标球必须小到
        # 实际球能露出来，否则「看两个球分开多少」这条指令在本场景无效。
        return _pair_markers(self._p_des, self._p) + [
            Trail(self.trail.points, color=C_ERROR, width=0.0045),
        ]


# ============================================================== 4 动量观测器


class ObserverScene(Scene):
    name = "observer"
    steps = STEPS_BY_SCENE["observer"]
    title = "动量观测器：不用力传感器估外力"
    camera = dict(azimuth=141.0, elevation=-14.0, distance=1.15,
                  lookat=(0.42, 0.0, 0.42))
    intro = (
        "机械臂末端每隔几秒被一个外力撞一下。它<b>没有力传感器</b>，"
        "只能从「电机发的力矩」和「实际动起来的样子」之间的差别反推外力。"
    )
    lesson = Lesson(
        problem=(
            "协作机器人必须能感知「我撞到东西了」，否则撞到人就是事故。\n\n"
            "最直接的办法是<b>装力传感器</b>——但六维力传感器很贵（几万块），"
            "装在手腕上还会占空间、增加末端质量、多一路线缆和一个故障点。"
            "而且它只能测<b>末端</b>受的力，手肘撞到人它完全不知道。\n\n"
            "有没有办法<b>一个传感器都不加</b>，就知道机器人被外力碰了？"
        ),
        idea=(
            "有。用<b>「账对不上」</b>这个思路。\n\n"
            "打个比方：你推一个购物车，用了多大劲你自己清楚，车该跑多快你心里也有数。"
            "如果你用了同样的劲，车却明显跑慢了——不用回头看，你也知道「后面有人拽着」。\n\n"
            "机器人也一样。它知道两件事：\n"
            "• <b>我发出去了多大力矩</b>（这是自己的指令，一定知道）\n"
            "• <b>我实际动成什么样</b>（关节编码器测得到，本来就有）\n\n"
            "再加上一个动力学模型，就能算出「按理说应该动成什么样」。"
            "<b>实际和应该之间的那个差额，就是外力干的。</b>\n\n"
            "所以：不需要新传感器，只需要把已有的信息算清楚。"
        ),
        how=(
            "直接的做法是用 τ_ext = M·q̈ + C·q̇ + g − τ。但这需要<b>关节加速度 q̈</b>，"
            "而编码器只测位置，加速度得连续微分两次——噪声会被放大到没法用。\n\n"
            "De Luca 的巧办法是改用<b>广义动量</b> p = M(q)·q̇（关节空间版的「动量」，"
            "只要一次微分就够）。对它求导会得到\n\n"
            "    <b>ṗ = τ + Cᵀq̇ − g + τ_ext</b>\n\n"
            "然后构造一个残差 r，让它满足\n\n"
            "    <b>ṙ = K_I·(τ_ext − r)</b>\n\n"
            "这个式子读作：「r 去追 τ_ext，追的快慢由 K_I 决定」。它就是一个"
            "<b>一阶低通滤波器</b>——和音响上的低音旋钮是同一个数学结构。\n"
            "实现上靠积分而不是微分，所以完全避开了 q̈。\n\n"
            "<b>K_I 是唯一的旋钮，它买卖的是「快」和「稳」</b>：\n"
            "• K_I 大 → 追得快，检测延迟短，但把测量噪声也一起放大了，容易误报\n"
            "• K_I 小 → 干净、不误报，但反应慢，人已经被夹疼了才报警\n\n"
            "最后用 <b>F̂ = (Jᵀ)⁺·r</b> 把关节力矩残差翻译回「末端受了多大的力」。"
            "（Jᵀ 把末端力变成关节力矩，那么它的伪逆就是反过来。）"
        ),
        watch=[
            "<b>先看画面</b>：撞击发生时会出现两支箭头。<b>品红 = 真实施加的外力</b>，"
            "<b>青色 = 观测器估出来的力</b>，左右分开画，直接比长短。"
            "比例尺是 1 N = 8 mm，45 N 对应 36 cm。",
            "<b>把「观测器带宽 K_I」拉到最小（2）</b>：青色箭头会<b>明显变短、而且慢半拍</b>，"
            "「检测延迟」这个数会飙升到几百毫秒。慢到这个程度，人早就被夹了。",
            "<b>把 K_I 拉到 150</b>：延迟降到十几毫秒，青色箭头几乎立刻跟上品红。",
            "<b>看下面的曲线</b>：蓝线（估计）追橙色虚线（真值）。K_I 小时你能清楚看到"
            "蓝线是一条<b>被拖慢的、圆滑的</b>曲线——这就是低通滤波的样子。",
            "<b>「碰撞阈值」调到很低（比如 1）</b>：机器人会开始<b>误报</b>——"
            "手臂自己运动产生的残差就超阈值了，它会莫名其妙地变软。",
            "<b>「检测后反应」切成 none</b>：检测到碰撞也不退让，手臂硬扛。"
            "切回 compliant：一检测到就切成低刚度，<b>画面上手臂会明显被压下去、让开</b>。"
            "这就是协作机器人撞到人之后「让开」的做法。",
        ],
        verify=[
            "「真实 Fz」是直接读施加的力（仿真才有的上帝视角），"
            "「估计 Fz」完全由观测器算出。两者的偏差百分比直接显示在旁边。",
            "K_I 增大时「检测延迟」应当单调下降，且大致按 1/K_I 反比——"
            "但<b>延迟本身不等于时间常数 1/K_I</b>。阶跃外力矩 u 下残差是 "
            "r(t)=u(1−e^(−K_I·t))，越过阈值 T 的时刻是 <b>t = −ln(1−T/|u|)/K_I</b>。"
            "只有 T/|u| = 1−1/e ≈ 0.632 时两者才相等。"
            "本场景默认 T=8 N·m、残差峰≈20.2 N·m ⇒ T/|u|≈0.395，"
            "K_I=30 时<b>实测约 16 ms</b>（闭式 16.8 ms），而 1/K_I 是 33 ms——差一倍。",
            "把「碰撞阈值」从 8 调到 16，K_I 不动，「检测延迟」会从约 16 ms 涨到约 44 ms；"
            "调到 30（高于残差峰值）就<b>永远检测不到</b>。"
            "同理把外力从 40 N 调到 80 N，延迟会掉到约 10 ms。"
            "⭐ 这说明「检测延迟」是 K_I、外力大小、方向和阈值四者共同决定的，"
            "不是观测器的固有属性。",
        ],
        caveat=(
            "⚠️ <b>本场景的估计值目前有 6–9% 的稳态偏差，这是已记录未修的实现缺陷</b>"
            "（离散化时序与被动力项两处）。所以<b>只看趋势</b>（K_I 大→延迟小、箭头更贴合），"
            "不要把具体数值当作准确结果。\n\n"
            "⚠️ <b>真机上要难得多。</b>这里控制器和被控对象是同一套模型，没有噪声。"
            "真机上模型误差、关节摩擦、齿轮间隙<b>全都会变成假残差</b>——"
            "机器人明明没撞到东西，残差却不为零。所以真机上阈值要设得高得多，"
            "灵敏度也就低得多。这是这项技术最主要的工程难点。\n\n"
            "⚠️ 观测器不需要 q̈，靠的是 <b>Ṁ = C + Cᵀ</b> 这条结构恒等式。"
            "注意斜对称的是 <b>Ṁ − 2C</b>，不是「Ṁ 本身是斜对称矩阵」——这两句经常被混为一谈。"
        ),
    )
    law = ControlLaw(
        formula="r(t) = K_I·[ p(t) − p(0) − ∫₀ᵗ( τ + Cᵀq̇ − g + r )ds ]\n"
                "⇒  ṙ = K_I·(τ_ext − r)          即 r 是 τ_ext 的一阶低通\n"
                "F̂ = (Jᵀ)⁺·r                     关节力矩残差 → 笛卡尔外力",
        terms=[
            LawTerm("p = M(q)q̇", "广义动量。质量矩阵乘速度，关节空间版的「动量」",
                    key="p_norm", unit="kg·m²/s", digits=2),
            LawTerm("r", "残差。观测器估出来的关节外力矩，超阈值就判碰撞",
                    key="res", unit="N·m", digits=3),
            LawTerm("K_I", "观测器带宽。大→反应快但噪声大；小→干净但迟钝",
                    key="ki_now", unit="1/s", digits=0),
            LawTerm("τ_ext", "真实外力矩。仿真里知道，真机上正是要估的未知量"),
            LawTerm("(Jᵀ)⁺", "雅可比转置的伪逆。把关节力矩残差翻译回末端的力"),
        ],
        blocks=[
            Block("电机力矩 τ", "实际发出去的那个值，不是控制律算出的理想值", "input"),
            Block("模型项 Cᵀq̇ − g", "用名义模型算", "block"),
            Block("一阶低通 K_I", "ṙ = K_I(τ_ext − r)", "block"),
            Block("残差 r", "≈ 关节外力矩", "output"),
            Block("阈值判定", "超过就报碰撞，触发柔顺退让", "block"),
        ],
        note="观测器不需要 q̈，靠的是 Ṁ = C + Cᵀ 这一结构恒等式。"
             "注意斜对称的是 Ṁ − 2C，不是「Ṁ 本身是斜对称矩阵」——"
             "这两句话经常被混为一谈。",
    )
    params = [
        Param("k_i", "观测器带宽", 2.0, 150.0, 30.0, unit="1/s", symbol="K_I",
              group="观测器",
              help="一阶滤波的截止频率。大 → 快但噪；小 → 稳但慢。",
              effect="<b>调小 → 青色箭头明显变短、慢半拍</b>，「检测延迟」飙到几百毫秒；"
                     "调大 → 延迟降到几毫秒，两支箭头几乎重合。"
                     "⚠️ <b>时间常数 = 1/K_I，但检测延迟 ≠ 时间常数</b>："
                     "延迟 = −ln(1−T/|u|)/K_I，还要看阈值 T 和外力有多大。"
                     "默认工况 T/|u|≈0.395，K_I=30 时实测约 16 ms，不是 33 ms。"),
        Param("threshold", "碰撞阈值", 0.5, 40.0, 8.0, unit="N·m", group="观测器",
              help="残差超过它就判定为碰撞。太低会误报，太高会漏报。"
                   "⭐ 它<b>直接决定「检测延迟」读数</b>：延迟 = −ln(1−T/|u|)/K_I。",
              effect="<b>调到很低 → 开始误报</b>：没撞到也判成碰撞，手臂莫名变软。"
                     "调到很高 → 漏报，撞了也不反应。"
                     "实测（K_I=30、外力 40 N、残差峰≈20.2 N·m）："
                     "T=2 → 6 ms；T=8 → 16 ms；T=16 → 44 ms；"
                     "T=30 高于残差峰值 → <b>永远检不到</b>。"),
        Param("react", "检测后反应", default="compliant",
              choices=["none", "compliant"], group="观测器",
              help="compliant 检测到碰撞后切成低刚度柔顺退让。",
              effect="<b>切成 compliant 时画面上手臂会被明显压下去、主动让开</b>；"
                     "none 则硬扛在原位。这是本场景视觉差别最大的一个开关。"),
        Param("f_amp", "撞击力 Fz", -120.0, 0.0, -45.0, unit="N", group="外部激励",
              effect="<b>品红箭头长度直接正比于它</b>。调大后更容易越过阈值被检测到。"),
        Param("period", "撞击周期", 2.0, 12.0, 5.0, unit="s", group="外部激励",
              effect="撞击的间隔。每个周期的<b>最后 1.5 秒</b>施力，其余时间没有外力。"
                     "调小 → 撞得更频繁，不用等那么久。"),
    ]
    readouts = [
        Readout("res", "残差范数 ‖r‖", "N·m", 3),
        Readout("f_hat", "估计 Fz", "N", 2),
        Readout("f_true", "真实 Fz", "N", 2, source="truth", compare_with="f_hat",
                help="仿真真值，直接读施加的力。真机上没有这一栏。"),
        Readout("delay", "检测延迟", "ms", 1),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
        Readout("p_norm", "广义动量 ‖p‖", "kg·m²/s", 3,
                help="p = M(q)·q̇。观测器盯的就是它的变化率对不对得上账。"),
    ]
    traces = [
        Trace("f_hat", "估计 Fz", "#4da3ff", unit="N"),
        Trace("f_true", "真实 Fz", "#ffb74d", unit="N", dashed=True),
        Trace("res", "残差范数", "#ff8a65", axis="res", unit="N·m"),
    ]
    bars = [_torque_bars()]

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.p0, self.R0 = self.robot.fk(Q_HOME)
        self.p0, self.R0 = self.p0.copy(), self.R0.copy()
        self.stiff = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=900.0, k_rot=40.0, zeta=1.0),
            q_null=Q_HOME, dt=self.dt)
        self.soft = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=120.0, k_rot=15.0, zeta=1.0),
            q_null=Q_HOME, dt=self.dt)
        self.obs = MomentumObserver(self.robot, k_i=self.get("k_i"), dt=self.dt)
        self.obs.reset(Q_HOME, np.zeros(N_ARM))
        self._tau = np.zeros(N_ARM)
        self._f_hat = 0.0
        self._f_true = 0.0
        self._f_hat_vec = np.zeros(3)
        self._f_true_vec = np.zeros(3)
        self._p = self.p0.copy()
        self._R = self.R0.copy()
        self._p_norm = 0.0
        self._detected = False
        self._t_hit = None
        self._delay = 0.0

    def _force(self, t):
        per = self.get("period")
        on = (t % per) > (per - 1.5)          # 每个周期最后 1.5 s 施力
        return (np.array([0.0, 0.0, self.get("f_amp")]) if on else np.zeros(3)), on

    def control(self, t, q, v):
        self.obs.k_i = self.get("k_i")
        f, on = self._force(t)
        self.data.xfrc_applied[self.robot.ee_body_id, :3] = f

        if on and self._t_hit is None:
            self._t_hit, self._detected = t, False
        if not on:
            self._t_hit = None
            self._detected = False

        hit = self.obs.detect(self.get("threshold"))
        if hit and self._t_hit is not None and not self._detected:
            self._detected = True
            self._delay = (t - self._t_hit) * 1e3

        use_soft = hit and self.choice("react") == "compliant"
        ctrl = self.soft if use_soft else self.stiff
        self._tau = ctrl.compute(q, v, self.p0, self.R0, dt=self.dt)
        # 观测器必须拿到**实际发给电机**的力矩，否则残差里会混进饱和量
        self.obs.update(q, v, self._tau)
        self._f_hat_vec = self.obs.cartesian_force(q)[:3].copy()
        self._f_hat = float(self._f_hat_vec[2])
        self._f_true_vec = f.copy()
        self._f_true = float(f[2])
        self._p, self._R = self.robot.fk(q)
        self._p_norm = float(np.linalg.norm(self.robot.mass_matrix(q) @ v))
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        return dict(res=float(np.linalg.norm(self.obs.r)),
                    f_hat=self._f_hat, f_true=self._f_true,
                    delay=self._delay, p_norm=self._p_norm,
                    ki_now=self.get("k_i"),
                    tau=float(np.linalg.norm(self._tau)),
                    tau_vec=self._tau, tau_limit=lim,
                    tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)

    def overlays(self):
        # 本场景的主题是"估得准不准"，不是姿态，所以不画坐标架：
        # 坐标架的三根轴会和两支力箭头挤在同一个点上，反而看不清。
        items = [Sphere(self._p, radius=0.016, color=C_TCP, alpha=0.9)]
        # 两支箭头**左右分开**画，各自从 TCP 旁边一点起步。
        # 叠在一起时长度差就是估计误差，但谁压着谁看不出来；分开画才能直接比长短。
        if np.linalg.norm(self._f_true_vec) > 1e-9:
            root = self._p + np.array([0.0, -0.06, 0.0])
            items.append(Arrow(root, root + self._f_true_vec * FORCE_SCALE,
                               color=C_FORCE_TRUE, width=0.016))
        if np.linalg.norm(self._f_hat_vec) > 1e-2:
            root = self._p + np.array([0.0, 0.06, 0.0])
            items.append(Arrow(root, root + self._f_hat_vec * FORCE_SCALE,
                               color=C_FORCE_EST, width=0.016))
        return items


# ============================================================== 5 路径规划


class PlanningScene(Scene):
    name = "planning"
    steps = STEPS_BY_SCENE["planning"]
    title = "RRT-Connect 路径规划与避障"
    camera = dict(azimuth=125.0, elevation=-20.0, distance=1.85,
                  lookat=(0.34, 0.0, 0.52))
    intro = (
        "末端要从立柱左边挪到右边。<b>起点和终点各自都不碰撞，"
        "但它们之间的直线会把手臂扫过立柱</b>——这就是必须做路径规划的原因。"
    )
    lesson = Lesson(
        problem=(
            "让机械臂从 A 点走到 B 点，最直觉的做法是「直着走过去」。"
            "问题是<b>起点合法、终点合法，不代表中间合法</b>——"
            "本场景里两端都不碰撞，但直线插值会让整条手臂从立柱里穿过去。\n\n"
            "更麻烦的是：碰撞发生在<b>三维空间</b>里，而机械臂能控制的是<b>七个关节角</b>。"
            "这两者之间是高度非线性的关系，你没法直接在三维空间里画一条「绕开柱子」的线"
            "然后指望关节角也是安全的。手臂的肘部可能在你完全没注意的地方撞上去。\n\n"
            "所以必须<b>在七维关节空间里搜索</b>一条全程都不碰撞的路。"
            "而七维空间没法像迷宫那样穷举——把每个关节切成 100 份就是 10¹⁴ 个格子。"
        ),
        idea=(
            "既然穷举不可能，就<b>随机撒点连起来</b>。这就是 RRT（快速扩展随机树）。\n\n"
            "想象你在一片黑暗的树林里找路：\n"
            "1. 从起点开始，随机往一个方向<b>迈一小步</b>；\n"
            "2. 如果这一步没撞到树，就把它记下来，你的「已知安全区域」就长大了一点；\n"
            "3. 一直重复，这片安全区域会像树根一样往各个方向蔓延；\n"
            "4. 等它蔓延到终点附近，把沿途的点连起来，就是一条路。\n\n"
            "<b>RRT-Connect</b> 是加速版：<b>起点和终点各长一棵树</b>，两头对着挖，"
            "碰头就通了。比一棵树瞎长快得多。\n\n"
            "这个算法有两个必须知道的性质：\n"
            "• <b>概率完备</b>：如果真有解，采样次数足够多一定能找到。但反过来不成立——"
            "没找到<b>只能说明「预算内没找到」，不能说明无解</b>。\n"
            "• <b>不是最优</b>：随机长出来的路又绕又抖。所以后面必须接一步<b>平滑</b>："
            "反复尝试「把路径上任意两点直接连起来」，能连通就把中间那段删掉。"
        ),
        how=(
            "整条流水线分三段，<b>各管各的，不能混在一起想</b>：\n\n"
            "<b>① 几何路径（走哪条线）</b>\n"
            "RRT-Connect 输出一串关节角路径点。这一步只关心「碰不碰」，完全不管时间。\n"
            "碰撞检测用<b>有符号距离</b>：不只回答「撞没撞」，还回答「离得多近」。"
            "这样才能设<b>安全裕度</b>——把障碍物假想着「膨胀」几厘米，"
            "留出真机上标定误差和控制误差的余量。\n\n"
            "<b>② 时间律（什么时候到哪）</b>\n"
            "有了路线还要决定走多快。这里有个坑：RRT 输出的是<b>折线</b>，"
            "拐角处方向是<b>跳变</b>的。速度不能跳变（那意味着无穷大加速度），"
            "所以逐段 S 曲线只能在<b>每个路径点停一下</b>——看「中途停顿次数」这个数。\n"
            "解决办法是先把拐角<b>倒成圆弧</b>（像赛车走弯道切内线），路径变光滑了，"
            "再做时间最优参数化（TOPP-RA），机械臂就能带速拐弯。\n\n"
            "<b>③ 轨迹跟踪（真去走）</b>\n"
            "最后用上一个场景讲的计算力矩控制把这条轨迹执行出来。"
            "这一段在本场景不是重点，增益固定为 400/40。"
        ),
        watch=[
            "<b>先看画面里的青色曲线</b>——那是规划器算出来的<b>几何路径</b>（末端会经过的位置）。"
            "黄色是实际执行的拖尾。青色明显绕过了立柱，这就是规划的价值。",
            "<b>把「平滑迭代」拉到 0 然后松手</b>：青色路径立刻变得<b>曲折绕远</b>，"
            "「原始路径长」和「平滑后长度」两个数变成一样。拉回 300，路径被拉直。",
            "<b>换「随机种子」</b>：每换一个数就得到一条<b>完全不同</b>的路径。"
            "这就是随机算法——没有唯一答案。",
            "<b>把「安全裕度」调大</b>：路径绕得更开，「执行中最小间隙」变大。"
            "继续调大到某个点会<b>规划失败</b>，下面会出现橙色提示。",
            "<b>把「时间律」切成 blend_toppra</b>：「中途停顿次数」从十几掉到 <b>0</b>，"
            "「轨迹时长」明显变短，画面上手臂不再一顿一顿地走。"
            "代价是「峰值 jerk」从 30 暴涨——冲击变大，真机上会响。",
            "<b>调「关节速度上限」</b>：手臂动作快慢明显变化，但<b>青色路径一点不变</b>。"
            "这就是「几何路径」和「时间律」是两件事的直观证明。",
        ],
        verify=[
            "「执行中最小间隙」是用<b>比规划分辨率密得多</b>的采样重新算一遍的，"
            "不是把规划器自己的判断抄一遍。如果规划器漏检了，这个数会变成负的。",
            "安全裕度设为 0.03 m 时，最小间隙应当 ≥ 30 mm 左右。两者对不上就是有一方错了。",
            "切 blend_toppra 后「实际倒角偏离」应当 ≤ 你设的预算，"
            "而且倒角后会<b>自动重新做碰撞检测</b>，撞了就自动缩小预算。",
        ],
        caveat=(
            "⚠️ <b>规划失败 ≠ 无解。</b>RRT 是概率完备的，失败只能说明「这次采样预算内没找到」。"
            "换个种子、调大步长、减小裕度都可能又找到。这一点在工程上非常重要——"
            "不能因为规划器报失败就断言任务不可行。\n\n"
            "⚠️ <b>平滑只压长度，不改可行性。</b>捷径平滑每一步都会重新做碰撞检测，"
            "连不通就不删。所以它不会把一条安全路径改成危险路径。\n\n"
            "⚠️ <b>倒角只允许吃掉规划裕度的 60%</b>，剩下 40% 留作执行裕度"
            "（实现：倒角检查器用 margin×0.4）。"
            "这就是工程上「规划裕度要比执行裕度大」的由来：规划时留的余量，"
            "一部分要留给后续环节继续消耗。"
        ),
    )
    params = [
        Param("margin", "安全裕度", 0.005, 0.12, 0.03, unit="m", restart=True,
              group="碰撞检测",
              help="碰撞检测时把障碍物「膨胀」多少。",
              effect="<b>调大 → 青色规划路径明显绕得更开</b>，离柱子更远。"
                     "太大会找不到解（看下面的橙色提示语）。"
                     "松开鼠标才会重新规划。"),
        Param("step_size", "扩展步长", 0.05, 0.8, 0.30, unit="rad", restart=True,
              group="RRT 参数",
              help="随机树每次往外长多远。",
              effect="调小 → 「RRT 迭代」和「树节点数」大幅上升、规划变慢，但路径更细致；"
                     "调大 → 找得快，但容易跨过窄缝而失败。"),
        Param("smooth_iters", "平滑迭代", 0, 600, 300, step=10, restart=True,
              group="RRT 参数",
              help="捷径平滑的尝试次数。",
              effect="<b>拉到 0 → 青色路径变得明显曲折绕远</b>，「原始路径长」和"
                     "「平滑后长度」两个数会变得一样。拉大 → 路径被拉直，长度掉下来。"),
        Param("seed", "随机种子", 0, 19, 0, step=1, restart=True,
              group="RRT 参数",
              help="随机数种子。",
              effect="<b>换一个数就得到一条完全不同的路径</b>——这就是随机算法的样子。"
                     "同一个种子结果可复现。"),
        Param("v_max", "关节速度上限", 0.2, 2.5, 1.2, unit="rad/s", restart=True,
              group="时间参数化",
              effect="<b>只改走多快，不改走哪条线</b>：青色路径一模一样，"
                     "但「轨迹时长」变短、手臂动作变快。"),
        Param("a_max", "关节加速度上限", 0.5, 8.0, 3.0, unit="rad/s²", restart=True,
              group="时间参数化",
              effect="同上，只影响起停的急缓和总时长，几何路径不变。"),
        Param("time_law", "时间律", default="scurve", restart=True,
              choices=["scurve", "blend_toppra"], group="时间参数化",
              help="scurve 逐段七段 S 曲线，jerk 有界但每个路径点都要停；"
                   "blend_toppra 先把拐角倒成圆弧再做时间最优参数化，"
                   "不停顿、更快，但 jerk 无界。",
              effect="<b>切成 blend_toppra：「中途停顿次数」从十几掉到 0，"
                     "「轨迹时长」明显变短</b>，手臂能带速拐弯。"
                     "代价是「峰值 jerk」从 30 涨到很大（冲击变大）。"),
        Param("max_deviation", "倒角偏离预算", 0.0, 0.12, 0.06, unit="rad",
              restart=True, group="时间参数化",
              help="只对 blend_toppra 生效。圆弧允许离开原折线多远。",
              effect="<b>scurve 模式下完全无效果。</b>blend_toppra 下调大 → 拐得更圆更快，"
                     "但路径往转弯内侧挪（看「实际倒角偏离」），离障碍物更近。"),
    ]
    readouts = [
        Readout("iters", "RRT 迭代", "", 0),
        Readout("nodes", "树节点数", "", 0),
        Readout("raw_len", "原始路径长", "rad", 3),
        Readout("smooth_len", "平滑后长度", "rad", 3),
        Readout("waypoints", "路径点数", "", 0),
        Readout("duration", "轨迹时长", "s", 2),
        Readout("clearance", "下发轨迹最小间隙", "mm", 2, theory=True,
                help="⭐ 沿**真正下发的 q_des(t)** 按 4 ms 采样逐点复核的最小间隙。\n"
                     "blend_toppra 模式下倒角会把路径往内侧挪，"
                     "所以它**必然 ≤「平滑后路径间隙」**。\n"
                     "⚠️ 这不是机器人实际执行状态的间隙——"
                     "实际状态还差一个 PD 跟踪误差，看曲线「当前间隙」。"),
        Readout("clear_raw", "原始路径间隙", "mm", 2, theory=True,
                help="RRT 直接吐出来的折线的间隙，还没平滑、没倒角。"),
        Readout("clear_smooth", "平滑后路径间隙", "mm", 2, theory=True,
                help="捷径平滑之后的折线间隙。⚠️ **旧版本把这个数标成"
                     "「执行中最小间隙」，是错的**：blend 之后执行的不是这条线。"),
        Readout("clear_loss", "倒角吃掉的间隙", "mm", 2, theory=True,
                help="=「平滑后路径间隙」−「下发轨迹最小间隙」。\n"
                     "scurve 模式下恒为 0（不倒角）；blend_toppra 下 > 0，"
                     "调大「倒角预算」会让它变大。**这就是旧读数系统性偏乐观的量。**"),
        Readout("stops", "中途停顿次数", "", 0),
        Readout("max_jerk", "峰值 jerk", "rad/s³", 0),
        Readout("blend_dev", "实际倒角偏离", "mm", 1),
        Readout("plan_ms", "规划耗时（实测）", "ms", 0, source="measured",
                help="⭐ 真的掐表量出来的：perf_counter 包住一次规划调用。"
                     "<br>⚠️ 这是<b>本机 Python 的墙钟时间</b>，"
                     "不代表真机控制器的实时性——换机器、换负载都会变。"
                     "<br>注意别和 <code>replan</code> 场景那个同名读数混淆："
                     "那边是你自己设的假设值，不是量出来的。"),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
    ]
    bars = [_torque_bars()]
    traces = [
        Trace("clear_now", "当前间隙", "#66bb6a", unit="mm"),
        Trace("speed", "关节速度范数", "#4da3ff", axis="speed", unit="rad/s"),
    ]
    law = ControlLaw(
        formula="规划：RRT-Connect 找可行路径 → 捷径平滑 → 时间参数化\n"
                "执行：τ = M(q)·[ q̈_d + Kd·(q̇_d − q̇) + Kp·(q_d − q) ] + h(q,q̇)",
        terms=[
            LawTerm("安全裕度", "碰撞检测时把障碍物「膨胀」多少，越大越保守",
                    key="margin_now", unit="m", digits=3),
            LawTerm("下发轨迹最小间隙",
                    "沿真正下发的 q_des(t) 密采样复核。⚠️ 不是几何路径的间隙——"
                    "blend_toppra 倒角后两者不是同一条线",
                    key="clearance", unit="mm", digits=1),
            LawTerm("v_max, a_max", "时间参数化的速度/加速度上限，只改快慢不改路线"),
            LawTerm("Kp, Kd", "跟踪这条轨迹用的 PD 增益，固定 400/40，不是本场景重点"),
        ],
        blocks=[
            Block("起点 / 终点构型", "各自都不碰撞", "input"),
            Block("RRT-Connect", "双向随机树，概率完备但非最优", "block"),
            Block("捷径平滑", "只压长度，不改可行性", "block"),
            Block("时间参数化", "S 曲线 或 倒角+TOPP-RA", "block"),
            Block("计算力矩跟踪", "把轨迹变成力矩", "block"),
            Block("机械臂", "", "plant"),
        ],
        note="规划器的输出是**几何路径**（走哪条线），时间律的输出是**时间轨迹**"
             "（什么时候到哪）。这两件事必须分开想：调 v_max 只会让机械臂走得更快，"
             "路线一点不变。",
    )

    OBSTACLES = [
        Obstacle("pillar", "box", (0.07, 0.07, 0.45), (0.52, 0.0, 0.45),
                 rgba=(0.85, 0.35, 0.25, 1.0)),
        Obstacle("beam", "box", (0.30, 0.42, 0.035), (0.52, 0.0, 0.97),
                 rgba=(0.60, 0.45, 0.30, 1.0)),
    ]
    Q_START = np.array([-0.85, 0.35, 0.0, -1.70, 0.0, 2.05, 0.79])
    Q_GOAL = np.array([0.85, 0.35, 0.0, -1.70, 0.0, 2.05, 0.79])

    def build(self):
        scene = build_panda_scene(self.OBSTACLES, with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.scene_info = scene
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self.env_geoms = movable_geoms(scene.model, scene.robot_geoms,
                                       self.robot.joint_ids)
        # 传入被规划关节：两根手指是兄弟连杆，相对位姿只由 finger_joint 决定，
        # 不传的话夹爪一闭合就会被判成自碰撞，规划器认为所有构型非法。
        self.self_pairs = self_collision_pairs(scene.model, scene.robot_geoms,
                                               self.robot.joint_ids)
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = self.Q_START
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.checker = CollisionChecker(
            self.model, self.robot.qpos_idx, self.env_geoms,
            self.scene_info.obstacle_geoms, self.self_pairs,
            safety_margin=self.get("margin"), self_margin=0.005,
            distmax=0.30, resolution=0.02)
        # 倒角总是往转弯内侧切进去。若拿规划裕度原样去卡倒角，
        # 那些贴着裕度走的路径一个拐角也倒不成。这里允许倒角吃掉规划裕度的
        # 60%，剩下 40% 作为执行裕度的下限——这就是工程上
        # 「规划裕度要比执行裕度大」的由来。
        self.checker_blend = CollisionChecker(
            self.model, self.robot.qpos_idx, self.env_geoms,
            self.scene_info.obstacle_geoms, self.self_pairs,
            safety_margin=self.get("margin") * 0.4, self_margin=0.005,
            distmax=0.30, resolution=0.02)
        planner = RRTConnect(self.checker, self.robot.q_lower, self.robot.q_upper,
                             RRTConnectConfig(step_size=self.get("step_size"),
                                              max_iters=6000))
        t0 = time.perf_counter()
        res = planner.plan(self.Q_START, self.Q_GOAL, seed=int(self.get("seed")))
        self._stats = dict(iters=0, nodes=0, raw_len=0.0, smooth_len=0.0,
                           waypoints=0, duration=0.0, clearance=0.0, plan_ms=0.0,
                           clear_raw=0.0, clear_smooth=0.0, clear_loss=0.0)
        if not res.success:
            self.traj = None
            self.note = f"规划失败：{res.reason}（试试减小安全裕度或换个种子）"
            return

        smooth = shortcut_smooth(res.path, self.checker,
                                 iters=int(self.get("smooth_iters")),
                                 seed=int(self.get("seed")))
        v = np.full(N_ARM, self.get("v_max"))
        a = np.full(N_ARM, self.get("a_max"))
        blend_dev = 0.0
        if self.choice("time_law") == "blend_toppra":
            # 倒角改变了几何路径，必须重新做碰撞检测；撞了就自动缩预算
            blended, blend_dev, _ = blend_and_verify(
                smooth, self.checker_blend,
                max_deviation=self.get("max_deviation"))
            self.traj = ToppraTrajectory(blended, v, a, n_grid=1200)
            geom_len = blended.length
        else:
            self.traj = JointPathTrajectory(smooth, v, a, np.full(N_ARM, 30.0))
            geom_len = path_length(smooth)
        plan_ms = (time.perf_counter() - t0) * 1e3
        # ⭐ P0-05 修复：以前这里只算 smooth 路径的间隙，却标成「执行中最小间隙」。
        # blend_toppra 模式下真正下发的是 blended（倒角后的路径），
        # 倒角会把路径往转弯内侧挪，间隙**只会变小**——旧读数系统性偏乐观。
        # 现在分三层各算各的，标签一一对应。
        clear_raw, _ = self.checker.path_clearances(res.path,
                                                    samples_per_rad=200.0)
        clear_smooth, _ = self.checker.path_clearances(smooth,
                                                       samples_per_rad=200.0)
        clear_cmd = self._trajectory_clearance(self.traj)
        rep = analyze(self.traj, "", geom_len)
        self._stats = dict(
            iters=res.iterations, nodes=res.nodes,
            raw_len=path_length(res.path), smooth_len=geom_len,
            waypoints=len(smooth), duration=self.traj.duration,
            clearance=clear_cmd * 1e3, plan_ms=plan_ms,
            clear_raw=clear_raw * 1e3, clear_smooth=clear_smooth * 1e3,
            clear_loss=(clear_smooth - clear_cmd) * 1e3,
            stops=rep.stops, max_jerk=rep.max_jerk, blend_dev=blend_dev * 1e3)
        self._min_clear = np.inf
        self.note = ("规划成功，机械臂会在起点与终点之间来回执行"
                     if res.reason != "直线可达" else
                     "本次直线即可达（裕度较小时会出现），换个更大的裕度看绕行")
        self._tau = np.zeros(N_ARM)
        self._p = self.robot.fk(self.Q_START)[0].copy()
        self.robot.set_state(self.Q_START, np.zeros(N_ARM))
        self.trail = TrailBuffer(capacity=300, min_step=0.006)
        # 把规划出来的几何路径本身画进画面：末端沿路径的位置序列。
        # 这是"规划器决定走哪条线"最直观的证据，和执行拖尾是两回事。
        self._path_pts = self._path_points(smooth)

    def _trajectory_clearance(self, traj, dt: float = 0.004) -> float:
        """⭐ 沿**真正下发的那条轨迹**按时间采样，逐点算间隙，取最小值。

        P0-05 修复的核心。和 ``path_clearances`` 的区别：

        - ``path_clearances`` 走的是**几何路径的折线**（规划器给的那串点）；
        - 这里走的是**时间参数化之后、控制器实际读取的 q_des(t)**。

        ``blend_toppra`` 模式下两者**不是同一条线**：倒角把拐角削圆了，
        路径会往转弯内侧挪，间隙只会变小。用前者报后者，是系统性偏乐观。
        """
        _, Q, _, _ = traj.sample(dt)
        worst = np.inf
        for qq in Q:
            env, _ = self.checker.clearances(qq)
            worst = min(worst, float(env))
        return worst

    def _path_points(self, path) -> np.ndarray:
        """把关节空间路径点转成末端位置序列，只用于画线。

        逐点做正运动学会改动 robot 的模型状态，算完必须还原成仿真当前状态，
        否则下一步控制读到的是路径上某个中间构型。
        """
        q_save = self.data.qpos[self.robot.qpos_idx].copy()
        v_save = self.data.qvel[self.robot.qvel_idx].copy()
        pts = []
        for i in range(len(path) - 1):
            for s in np.linspace(0.0, 1.0, 8, endpoint=False):
                pts.append(self.robot.fk(path[i] * (1 - s) + path[i + 1] * s)[0].copy())
        pts.append(self.robot.fk(path[-1])[0].copy())
        self.robot.set_state(q_save, v_save)
        return np.array(pts)

    def control(self, t, q, v):
        if self.traj is None:
            # 规划失败就停在起点，用高增益 PD 保持
            e = self.Q_START - q
            tau = self.reg.mass_matrix(q) @ (400.0 * e - 40.0 * v) + self.reg.bias(q, v)
            self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)
            self._p = self.robot.fk(q)[0].copy()
            return self._tau
        # 往返执行：0→T 正走，T→2T 倒着走
        T = self.traj.duration
        tt = t % (2.0 * T)
        s = tt if tt <= T else 2.0 * T - tt
        qd, vd, ad = self.traj(s)
        if tt > T:
            vd, ad = -vd, -ad
        aq = ad + 40.0 * (vd - v) + 400.0 * (qd - q)
        tau = self.reg.mass_matrix(q) @ aq + self.reg.bias(q, v)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)
        self._p = self.robot.fk(q)[0].copy()
        self.trail.push(self._p)
        return self._tau

    def telemetry(self, t, q, v):
        env, _ = self.checker.clearances(q)
        out = dict(self._stats)
        lim = self.robot.tau_limit
        out["clear_now"] = env * 1e3
        out["speed"] = float(np.linalg.norm(v))
        out["tau"] = float(np.linalg.norm(self._tau))
        out["tau_vec"] = self._tau
        out["tau_limit"] = lim
        out["tau_max_pct"] = float(np.max(np.abs(self._tau) / lim)) * 100.0
        out["margin_now"] = self.get("margin")
        return out

    def overlays(self):
        items = []
        if getattr(self, "_path_pts", None) is not None and len(self._path_pts) > 1:
            # 规划出的几何路径：青色细线，全程可见
            items.append(Trail(self._path_pts, color=C_PATH, width=0.005,
                               alpha=0.75))
        items.append(Trail(self.trail.points, color=C_ERROR, width=0.0045))
        items.append(Sphere(self._p, radius=0.022, color=C_TCP, alpha=0.95))
        return items




# ============================================================== 6 状态估计


class EstimationScene(Scene):
    name = "estimation"
    steps = STEPS_BY_SCENE["estimation"]
    title = "状态估计：IMU 与运动学的互补融合"
    camera = dict(azimuth=138.0, elevation=-16.0, distance=1.35,
                  lookat=(0.40, 0.0, 0.45))
    intro = (
        "机械臂末端装了一个 IMU。要在**只信任传感器**的前提下，"
        "估计出末端此刻在哪、走多快。"
    )
    law = ControlLaw(
        formula="预测（IMU，1 kHz）  a = R·(f − b̂) + g,   p⁺ = p + v·dt + ½a·dt²\n"
                "更新（运动学，低频） z = p_fk,  K = P Hᵀ(H P Hᵀ + R)⁻¹\n"
                "状态 x = [ p ; v ; b_a ]（位置、速度、加速度计零偏）",
        terms=[
            LawTerm("‖p̂ − p‖", "位置估计误差。这是最终要看的量",
                    key="err_pos", unit="mm", digits=2),
            LawTerm("‖v̂ − v‖", "速度估计误差", key="err_vel", unit="mm/s", digits=1),
            LawTerm("b̂", "估计出来的加速度计零偏", key="bias_hat", unit="m/s²", digits=4),
            LawTerm("b", "零偏真值（仿真才有的上帝视角）", key="bias_true",
                    unit="m/s²", digits=4),
            LawTerm("σ_p", "滤波器自己认为的位置标准差 √P。与实际误差长期严重不符"
                          "就说明噪声参数设错了", key="sigma_pos", unit="mm", digits=3),
        ],
        blocks=[
            Block("IMU 比力 f（高频）", "有噪声、有零偏，积分会漂", "input"),
            Block("预测 a = R·(f − b̂) + g", "关键是 +g，把重力加回去", "block"),
            Block("运动学 z = 正运动学(q)（低频）", "准，但不是随时可用", "input"),
            Block("卡尔曼增益 K", "按两边的不确定度加权", "block"),
            Block("融合后的 p̂, v̂, b̂", "", "output"),
            Block("MuJoCo 真值", "仅用于对照，真机上没有这一路", "feedback"),
        ],
        note="IMU 快但会漂，运动学准但慢且不总有。卡尔曼滤波按各自的不确定度"
             "自动分配信任，这就是「互补融合」。零偏必须放进状态里估计——"
             "它是常值偏移，两次积分后位置误差按 ½·b·t² **平方**发散，压不掉。",
    )
    params = [
        Param("acc_density", "加速度计噪声", 0.0, 0.05, 2e-3, unit="m/s²/√Hz",
              group="IMU（被估计的一路）", symbol="σ_a",
              help="噪声密度，数据手册就是这么给的。",
              effect="调大 → <b>估计位置开始抖</b>，误差曲线变毛躁。"
                     "运动学更新还在时抖动有限；关掉更新后会迅速发散。"),
        Param("bias_x", "零偏真值 bx", -0.3, 0.3, 0.08, unit="m/s²",
              group="IMU（被估计的一路）", symbol="b",
              help="加速度计的常值偏移，真实 IMU 一定有。",
              effect="调大 → 「零偏真值」那栏跟着变，滤波器要花几秒把它<b>估出来</b>。"
                     "把运动学更新关掉，位置会按 ½·b·t² <b>平方发散</b>。"),
        Param("bias_walk", "零偏游走", 0.0, 0.02, 1e-3, unit="m/s²/√s",
              group="IMU（被估计的一路）",
              effect="零偏自己慢慢漂。调大后即使估出来了也会跟丢，"
                     "看「零偏估计」与「零偏真值」两条线的间距。"),
        Param("meas_noise", "运动学量测噪声", 1e-5, 0.02, 5e-4, unit="m",
              group="运动学（修正的一路）", symbol="R",
              help="编码器 + 正运动学给出的位置有多准。",
              effect="调大 → 滤波器不信运动学，更依赖 IMU，位置误差变大。"
                     "这就是 K = P Hᵀ(H P Hᵀ + R)⁻¹ 里 R 变大导致增益变小。"),
        Param("update_mode", "运动学何时可用", default="always",
              choices=["always", "stance", "off"], group="运动学（修正的一路）",
              help="always 恒可用（机械臂的真实情况）；"
                   "stance 只在末端速度低于阈值时可用（模拟支撑相判据）；"
                   "off 完全关掉，纯 IMU 积分。",
              effect="<b>切成 off 是本场景最重要的一次操作</b>：位置估计会肉眼可见地"
                     "飘走，原生窗口里蓝球（估计）和绿球（真值）越离越远。"),
        Param("stance_speed", "接触判据阈值", 0.005, 0.5, 0.08, unit="m/s",
              group="运动学（修正的一路）",
              help="只对 stance 模式生效：末端速度低于它才认为「站得住」。",
              effect="<b>只在 stance 模式下有效。</b>调小 → 可用的时刻变少，"
                     "漂移变大；调大 → 几乎等同 always。"),
        Param("proc_noise", "过程噪声", 1e-4, 0.05, 2e-3, unit="m/s²/√Hz",
              group="滤波器整定", symbol="Q",
              help="滤波器**以为**的 IMU 噪声。与真实噪声不一致时会失配。",
              effect="故意调得和真实噪声不一样，看「滤波器自认标准差 σ_p」与"
                     "「实际误差」如何分家——这是整定是否合理的判据。"),
        Param("freq", "运动频率", 0.0, 1.2, 0.35, unit="Hz", group="激励",
              effect="调到 0 → 末端静止。<b>静止时零偏依然可观</b>（位置量测里"
                     "零偏留下 −½bt² 的抛物线签名，与 p₀、v₀t 形状可分），"
                     "只是激励弱 → 条件数变差 → <b>收敛更慢</b>。"
                     "真正让零偏不可观的是「运动学何时可用」切到 off。"),
        Param("amp", "运动幅值", 0.0, 0.6, 0.28, unit="rad", group="激励",
              effect="同上，幅值越小激励越弱，零偏收敛越慢（不是估不出）。"),
    ]
    readouts = [
        Readout("err_pos", "位置估计误差", "mm", 2),
        Readout("sigma_pos", "滤波器自认标准差 σ_p", "mm", 3, source="internal",
                compare_with="err_pos",
                help="√P 的位置分量。它是滤波器对自己精度的判断，"
                     "与实际误差长期严重不符就说明噪声参数设错了。"),
        Readout("err_vel", "速度估计误差", "mm/s", 1),
        Readout("bias_hat", "零偏估计 b̂x", "m/s²", 4),
        Readout("bias_true", "零偏真值 bx", "m/s²", 4, source="truth",
                compare_with="bias_hat",
                help="仿真才有的上帝视角。真机上没有这一栏，"
                     "所以真机上你根本不知道零偏估准没有。"),
        Readout("update_hz", "运动学更新率", "Hz", 0),
        Readout("speed", "末端速度", "m/s", 3),
    ]
    traces = [
        Trace("err_pos", "位置误差", "#4da3ff", unit="mm"),
        Trace("sigma_pos", "滤波器自认 σ_p", "#ffb74d", unit="mm", dashed=True),
        Trace("bias_hat", "零偏估计", "#66bb6a", axis="bias", unit="m/s²"),
        Trace("bias_true", "零偏真值", "#ff2d78", axis="bias", unit="m/s²",
              dashed=True),
    ]
    bars = [_torque_bars()]

    UPDATE_EVERY = 10                 # 运动学更新周期（控制步数），50 Hz

    def build(self):
        # IMU 必须挂在**与量测同一个点**上。量测用的是 robot.fk(q) 给出的 TCP
        # （两指夹持垫之间的抓取中心），所以 site 也要放在 TCP 上。
        # 早先把 site 放在 hand 原点，与 TCP 差 0.1029 m，结果是滤波器看到一个
        # 恒定的 103 mm 偏差，只好用零偏去解释它——位置误差和零偏估计一起被带歪。
        # 这类"估计的点和量测的点不是同一个点"的坐标系错误，症状是一个
        # **恒定且恰好等于偏置量**的误差，很容易被误当成"滤波没调好"。
        scene = build_panda_scene([], with_visuals=True, imu_body="hand",
                                  imu_pos=tuple(PANDA_TCP_OFFSET))
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self._sensor_adr = {}
        for nm in ("imu_truth_pos", "imu_truth_vel"):
            i = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SENSOR, nm)
            self._sensor_adr[nm] = (int(scene.model.sensor_adr[i]),
                                    int(scene.model.sensor_dim[i]))
        self._mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        return scene.model

    def _truth(self, name):
        a, k = self._sensor_adr[name]
        return self.data.sensordata[a:a + k].copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.imu = SimulatedIMU(
            self.model, self.data,
            noise=ImuNoise(acc_density=self.get("acc_density"),
                           gyro_density=1e-4,
                           acc_bias_walk=self.get("bias_walk"),
                           acc_bias0=(self.get("bias_x"), 0.0, 0.0)),
            seed=12345)
        self.est = TcpStateEstimator(acc_noise=self.get("proc_noise"),
                                     bias_walk=max(self.get("bias_walk"), 1e-6),
                                     meas_noise=self.get("meas_noise"),
                                     gravity=self.model.opt.gravity.copy())
        # 用真值初始化：本场景要看的是**误差怎么长出来**，
        # 不是初值收敛过程。零偏仍从 0 起，因为那才是要估的东西。
        self.est.reset(self._truth("imu_truth_pos"), self._truth("imu_truth_vel"))

        self._tau = np.zeros(N_ARM)
        self._k = 0
        self._n_update = 0
        self._t = 0.0
        self._p_hat = self._truth("imu_truth_pos")
        self._p_true = self._p_hat.copy()
        self.trail_hat = TrailBuffer(capacity=300, min_step=0.002)
        self.trail_true = TrailBuffer(capacity=300, min_step=0.002)
        self._tel = dict(err_pos=0.0, sigma_pos=0.0, err_vel=0.0,
                         bias_hat=0.0, bias_true=self.get("bias_x"),
                         update_hz=0.0, speed=0.0)
        self.note = ("IMU 的零偏与噪声一开始就在起作用；"
                     "把「运动学何时可用」切成 off，看估计怎么飘走")

    def on_change(self, key):
        if not hasattr(self, "est"):
            return
        if key == "meas_noise":
            self.est.meas_noise = self.get("meas_noise")
        elif key == "proc_noise":
            self.est.acc_noise = self.get("proc_noise")
        elif key in ("acc_density", "bias_walk"):
            self.imu.noise.acc_density = self.get("acc_density")
            self.imu.noise.acc_bias_walk = self.get("bias_walk")
            self.est.bias_walk = max(self.get("bias_walk"), 1e-6)
        elif key == "bias_x":
            # 直接改零偏真值，不重置场景：这样能看到滤波器**追**上去的过程
            self.imu.true_bias[0] = self.get("bias_x")

    def _ref(self, t):
        w = 2.0 * np.pi * self.get("freq")
        a = self.get("amp") * self._mask
        return (Q_HOME + a * np.sin(w * t), a * w * np.cos(w * t),
                -a * w * w * np.sin(w * t))

    def control(self, t, q, v):
        self._t = t
        qd, vd, ad = self._ref(t)
        aq = ad + 40.0 * (vd - v) + 400.0 * (qd - q)
        tau = self.reg.mass_matrix(q) @ aq + self.reg.bias(q, v)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)

        # 传感器读数依赖 mj_forward 后的 sensordata；控制阶段 data 已是当前状态
        f_body, _ = self.imu.read(self.dt)
        self.est.predict(f_body, self.imu.rotation(), self.dt)

        p_true = self._truth("imu_truth_pos")
        v_true = self._truth("imu_truth_vel")
        speed = float(np.linalg.norm(v_true))

        mode = self.choice("update_mode")
        do_update = False
        if mode == "always":
            do_update = (self._k % self.UPDATE_EVERY == 0)
        elif mode == "stance":
            # 支撑相判据的类比：末端"站得住"（速度低）时才采信运动学。
            # 机械臂上正运动学其实恒成立，这个间歇性是**人为制造**的，
            # 目的是复现足式机器人里接触判据失效的那类问题。
            do_update = (self._k % self.UPDATE_EVERY == 0
                         and speed < self.get("stance_speed"))
        if do_update:
            # 量测由**关节编码器 + 正运动学**给出，不是直接抄真值：
            # 抄真值就成了用被测对象证明自己。这里用 robot.fk(q)，
            # 与 MuJoCo 的 framepos 传感器是两条独立通道。
            p_fk, _ = self.robot.fk(q)
            self.est.update_position(p_fk)
            self._n_update += 1

        self._p_hat = self.est.position
        self._p_true = p_true
        self.trail_hat.push(self._p_hat)
        self.trail_true.push(p_true)

        sig = self.est.std()
        self._tel = dict(
            err_pos=float(np.linalg.norm(self._p_hat - p_true)) * 1e3,
            sigma_pos=float(np.linalg.norm(sig[0:3])) * 1e3,
            err_vel=float(np.linalg.norm(self.est.velocity - v_true)) * 1e3,
            bias_hat=float(self.est.bias[0]),
            bias_true=float(self.imu.true_bias[0]),
            update_hz=(self._n_update / t if t > 0.2 else 0.0),
            speed=speed)
        self._k += 1
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        out = dict(self._tel)
        out.update(tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
                   tau_limit=lim,
                   tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)
        return out

    def overlays(self):
        # 真值球（绿）与估计球（青）走 _pair_markers 的统一尺寸约定。
        # 关掉运动学更新后估计会漂走，两球分离正是本场景要看的现象；
        # 有更新时误差只有零点几毫米，两球重合——这时「看不出差别」是正确结论。
        items = _pair_markers(self._p_true, self._p_hat,
                              c_actual=C_FORCE_EST, a_target=0.55)
        items += [
            Trail(self.trail_true.points, color=C_TARGET, width=0.004, alpha=0.6),
            Trail(self.trail_hat.points, color=C_FORCE_EST, width=0.004),
        ]
        d = self._p_hat - self._p_true
        if 1e-4 < np.linalg.norm(d) < R_TARGET_MARK:
            # 放大箭头**只在误差小到 1:1 看不出来的时候**才画。
            # 它的用途是让亚毫米级的误差有个方向指示；一旦误差已经大到两球
            # 明显分开（关掉运动学修正后能到几百毫米），5 倍放大的箭头会长到
            # 一两米、直接冲出画面，除了添乱没有任何信息——而且会让人把箭头
            # 长度误读成真实漂移量。倍数标在讲解里。
            items.append(Arrow(self._p_true, self._p_true + d * 5.0,
                               color=C_ERROR, width=0.010))
        return items




# ================================================================== 7 抓取控制


class GraspScene(Scene):
    name = "grasp"
    steps = STEPS_BY_SCENE["grasp"]
    title = "抓取：夹持力、滑移检测与力自适应"
    camera = dict(azimuth=152.0, elevation=-12.0, distance=0.72,
                  lookat=(0.52, 0.0, 0.20))
    intro = (
        "机械臂去抓桌上的方块，夹住、提起、悬停。核心问题只有一个："
        "<b>夹多大力？</b>夹太松掉下去，夹太紧压坏工件。"
    )
    law = ControlLaw(
        formula="防滑约束：2·μ·N ≥ m·(g + a)   ⟹   N_min = m·(g + a) / (2μ)\n"
                "滑移判据：摩擦锥余量 = |f_t| / (μ·f_n) → 1 即濒临滑移\n"
                "力自适应：滑则快加（rise），不滑则慢减（decay）",
        terms=[
            LawTerm("N", "实测单面法向夹持力", key="grip_force", unit="N", digits=2),
            LawTerm("N_min", "防滑所需的最小夹持力（用真实 m 与 μ 算的对照值）",
                    key="grip_theory", unit="N", digits=2),
            LawTerm("μ", "**实际生效**的摩擦系数，从接触里现场读",
                    key="mu_eff", unit="", digits=3),
            LawTerm("|f_t|/(μ f_n)", "摩擦锥余量，→1 表示快滑了",
                    key="margin", unit="", digits=3),
            LawTerm("m", "工件质量", key="obj_mass", unit="kg", digits=3),
        ],
        blocks=[
            Block("抓取位姿生成", "物体最短主轴 = 夹爪开合轴", "input"),
            Block("笛卡尔阻抗接近", "低刚度贴近，避免撞飞工件", "block"),
            Block("闭合并加载夹持力", "位置伺服过闭合 → 力", "block"),
            Block("滑移检测", "摩擦锥余量 + 相对切向速度", "feedback"),
            Block("夹持力自适应", "滑则快加、不滑则慢减", "block"),
            Block("工件", "会掉、会滑", "plant"),
        ],
        note="真机上 m 与 μ 都不知道，所以 N_min 只能事后对照，不能直接当控制律。"
             "自适应策略是闭环试探出最小夹持力——人手拿纸杯用的就是这个办法。",
    )
    params = [
        Param("mode", "夹持力策略", default="adaptive",
              choices=["fixed", "adaptive"], group="控制器",
              help="fixed 用固定夹持力；adaptive 靠滑移反馈自己找。",
              effect="<b>切成 fixed 再把夹持力调到 2 N，工件会当场滑掉</b>；"
                     "切回 adaptive，它会自己把力加上来接住。"),
        Param("grip_cmd", "目标单面夹持力", 0.3, 30.0, 6.0, unit="N",
              symbol="N", group="控制器",
              help="只对 fixed 模式生效。口径与「理论最小夹持力」一致，可直接比。",
              effect="<b>只在 fixed 模式下有效。</b>调到<b>低于</b>「理论最小夹持力」，"
                     "工件当场下滑；调到略高于它就刚好夹住——"
                     "这是对那条公式最直接的验证。"),
        Param("rise", "自适应加力速率", 5.0, 200.0, 60.0, unit="N/s",
              group="控制器",
              help="只对 adaptive 生效。检测到滑移时的加力速率。",
              effect="调小 → 滑起来补不及，工件掉下去。滑移一旦开始，"
                     "接触从静摩擦切到动摩擦、摩擦力反而下降，所以必须加得快。"),
        Param("decay", "自适应减力速率", 0.0, 20.0, 2.0, unit="N/s",
              group="控制器",
              help="不滑时缓慢减力，用来试探最小夹持力。",
              effect="调大 → 力掉得快，会在「滑一下、补一下」之间反复横跳；"
                     "调到 0 → 只加不减，最后停在很保守的高力上。"),
        Param("margin_thr", "滑移判据阈值", 0.3, 1.0, 0.85, unit="",
              group="检测", help="摩擦锥余量超过它就判定濒临滑移。",
              effect="调小 → 更早报警、更保守（力偏高）；"
                     "调到 1.0 → 等真滑了才反应。"),
        Param("slip_speed_thr", "滑移速度阈值", 2e-4, 0.02, 5e-4, unit="m/s",
              group="检测", help="工件相对指垫的切向速度超过它就算已经在滑。",
              effect="反应性判据。<b>它必须比你想抓住的最慢滑移还小</b>，"
                     "否则这条支路在构造上就不可能触发——"
                     "本场景实测的缓慢蠕变约 1.4 mm/s，所以阈值定在 0.5 mm/s。"
                     "调到 2 mm/s 试试：悬停 29 s 工件会往下爬约 40 mm，"
                     "因为检测器压根看不见这么慢的滑。调太小则易被抖动误触发。"),
        Param("obj_mass", "工件质量", 0.05, 1.2, 0.25, unit="kg", restart=True,
              symbol="m", group="被抓的东西",
              help="改动会重置场景。",
              effect="<b>调大 → 「理论最小夹持力」那一栏立刻跟着涨</b>"
                     "（正比于 m）。fixed 模式下不改力就会滑。"),
        Param("obj_mu", "工件摩擦系数", 0.15, 1.0, 0.6, unit="", restart=True,
              symbol="μ", group="被抓的东西",
              help="同时会把指垫摩擦设成同一个值，否则不生效（见讲解）。",
              effect="<b>调小 → 理论最小夹持力按 1/μ 飙升</b>，很容易滑。"
                     "这就是抓油腻工件难的原因。"),
        Param("grip_offset", "抓取高度偏移", -0.02, 0.02, 0.0, unit="m",
              restart=True, group="被抓的东西",
              help="抓取点相对工件质心沿竖直方向的偏移。",
              effect="偏离质心 → 工件在指间<b>转动</b>而不是掉下去。"
                     "防滑公式只管平移，管不了这个（见边界那一步）。"),
        Param("closed_loop", "抓取位姿来源", default="visual",
              choices=["visual", "openloop"], group="视觉伺服",
              help="visual 每帧按观测更新目标；openloop 开始前算一次就不再改。",
              effect="<b>本场景最重要的开关。</b>把「工件初始偏移」调到 0.06 再切成 "
                     "openloop：机械臂扑空，还会把工件撞倒。切回 visual 就能抓到。"),
        Param("obj_jitter", "工件初始偏移", 0.0, 0.10, 0.0, unit="m", restart=True,
              group="视觉伺服",
              help="把工件从标称位置沿 y 方向挪开，模拟「掉了之后滚到别处」。",
              effect="开环模式下这就是致命的：目标点是按标称位置算的，偏多少就差多少。"),
        Param("vs_gain", "伺服增益 λ", 0.5, 30.0, 5.0, unit="1/s",
              symbol="λ", group="视觉伺服",
              help="误差每秒被压缩的比例，闭环时间常数是 1/λ。",
              effect="调大 → 对准更快，但与延迟相乘会吃掉相位裕度。"
                     "⭐ 决定稳不稳的是<b>乘积 λ·τ</b>，不是各自的大小。"
                     "纯延迟积分环的<b>离散</b>边界是 "
                     "<b>λ·τ = 2d·sin(π/(4d+2))</b>（d = τ/Δt），"
                     "τ=50 ms 时 = <b>1.5398</b>，τ→大 时趋于 π/2 ≈ 1.5708。"
                     "⚠️ <b>这个数是纯延迟环的理论值，不是本场景的实测边界。</b>"
                     "本场景<b>测不出</b>可信的边界：伺服有死区 1 mm、单步限幅 4 mm、"
                     "设定点牵引绳 50 mm 三重饱和，会把发散压成有界振荡；"
                     "把饱和关掉又会立刻撞上工作空间/关节极限。"
                     "详见 <code>07_抓取与视觉伺服.md §7.4b′</code>（含撤回声明）。"
                     "⭐ 所以调大 λ 时看的是<b>误差还在不在衰减</b>，"
                     "绝不能看『跑没跑飞』——饱和只给『有界』，不给『稳定』。"),
        Param("vs_latency", "观测延迟", 0.0, 0.20, 0.05, unit="s",
              group="视觉伺服",
              help="从曝光到位姿可用的总延迟。真机 30–100 ms 常见。",
              effect="<b>视觉伺服最主要的性能瓶颈。</b>调大后末端会在目标附近"
                     "过冲、来回摆。它和噪声的影响完全不同：噪声让输出毛躁但不改相位，"
                     "延迟直接吃相位裕度。"),
        Param("vs_rate", "观测更新率", 5.0, 120.0, 30.0, unit="Hz",
              group="视觉伺服",
              help="相机帧率。控制周期是 500 Hz，两帧之间只能保持。",
              effect="调低 → 修正变成一跳一跳的，末端走走停停。"),
        Param("vs_noise", "观测噪声", 0.0, 0.02, 2e-3, unit="m",
              group="视觉伺服",
              help="位姿估计的位置噪声标准差。",
              effect="调大 → 目标点自己在抖，末端跟着抖。"
                     "配合死区看：噪声超过死区时才会一直修。"),
        Param("lift_height", "提升高度", 0.0, 0.25, 0.12, unit="m",
              group="任务", effect="抓住后往上提多高。"),
        Param("shake", "提起后抖动", 0.0, 6.0, 0.0, unit="m/s²",
              group="任务",
              help="给工件一个竖直方向的正弦加速度激励。",
              effect="<b>调大 → 惯性载荷让所需夹持力按 (g+a) 上升</b>，"
                     "「理论最小夹持力」跟着变。fixed 模式下会被抖掉。"),
    ]
    readouts = [
        Readout("grip_force", "实测夹持力", "N", 2),
        Readout("grip_theory", "理论最小夹持力", "N", 2, theory=True,
                compare_with="grip_force",
                help="N_min = m(g+a)/(2μ)，用真实质量与**实际生效**的摩擦系数算。"
                     "真机上这两个量都不知道，所以这一栏只能事后对照。"),
        Readout("margin", "摩擦锥余量", "", 3,
                help="|f_t|/(μ·f_n)，趋近 1 表示濒临滑移。"),
        Readout("slip_speed", "滑移速度", "mm/s", 3),
        Readout("mu_eff", "生效摩擦系数", "", 3, source="measured",
                help="从 contact.friction 现场读，是实测量不是理论值。"),
        Readout("obj_height", "工件高度", "mm", 1),
        Readout("obj_drop", "工件在指间下滑", "mm", 2,
                help="工件在**夹爪坐标系**里沿开合面滑了多少。"
                     "必须同时扣掉夹爪的平移与转动——只扣平移的话，"
                     "夹爪受载转几度就会被误读成滑移。"),
        Readout("obj_tilt", "工件在指间转动", "°", 2,
                help="转动逃逸是与平移滑移**并列的另一种失效模式**，"
                     "防滑公式完全管不了它。"),
        Readout("n_slip", "滑移事件次数", "", 0),
        Readout("obj_mass", "工件质量", "kg", 3),
        Readout("phase", "阶段", "", 0,
                help="0 搜索对准 / 1 下降 / 2 闭合加力 / 3 提起 / 4 悬停"),
        Readout("vs_err", "视觉对准误差", "mm", 2,
                help="观测到的工件位置与夹爪目标点的水平距离。"),
        Readout("vs_age", "观测新鲜度", "ms", 1, source="measured",
                help="当前用的这一帧对应的真值距今多久。它比设定的延迟还大一点，"
                     "因为两帧之间只能保持。**这才是控制器实际承受的延迟**。"),
        Readout("target_err", "目标点跟真值差", "mm", 2, source="truth",
                help="夹爪目标点与工件**真实**位置的差。仿真才有的上帝视角，"
                     "用来看开环模式差在哪里。"),
        Readout("n_retry", "重抓次数", "", 0),
        Readout("need_now", "当前姿态开口需求", "mm", 1,
                help="沿**当前夹爪开合方向**量到的工件跨度，也就是「现在这个"
                     "朝向要张多大才夹得下」。工件立着时是 42 mm；被碰倒后"
                     "沿旧方向会变成 90 mm，超过夹爪 80 mm 行程 —— "
                     "这就是触发转腕重抓的判据，比「开合轴变了」稳，"
                     "因为正方形截面的主轴方向本来就是任意的。"),
        Readout("grasp_yaw", "规划开合方位角", "°", 1,
                help="规划出的开合轴在水平面内的方位角。工件被碰倒时它会跳 90°，"
                     "手腕跟着转过去。"),
        Readout("obj_lean", "工件倾角", "°", 1, source="truth",
                help="工件自身竖轴与世界竖直方向的夹角（仿真真值）。"
                     "0° 是立着，90° 是完全躺倒。"),
    ]
    traces = [
        Trace("grip_force", "实测夹持力", "#4da3ff", unit="N"),
        Trace("grip_theory", "理论最小", "#ffb74d", unit="N", dashed=True),
        Trace("margin", "摩擦锥余量", "#ff2d78", axis="margin", unit=""),
    ]
    bars = [_torque_bars()]

    OBJ_HALF = (0.021, 0.021, 0.045)
    OBJ_POS = (0.52, 0.0, 0.045)
    #: 各阶段的持续时间（秒）
    T_APPROACH, T_DESCEND, T_GRIP, T_LIFT = 1.6, 1.4, 1.2, 1.6

    def build(self):
        self.obj = FreeBody("cube", self.OBJ_HALF, self.OBJ_POS,
                            mass=0.25, friction=0.6)
        scene = build_panda_scene([], with_visuals=True, free_bodies=[self.obj])
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        m = scene.model
        # MuJoCo 主求解器把摩擦当**软约束**解，持续切向载荷（这里是重力）下
        # 会有数值蠕变：实测夹持力从 4 N 加到 30 N（摩擦锥只用了 7%）下滑量
        # 不变，步长减半、主迭代 ×4 也不变 —— 三个排除性实验说明它既不是
        # 「力不够」也不是离散化误差，而是求解器构造性的漂移。noslip 是
        # MuJoCo 为此提供的后处理迭代，开 5 次把蠕变从 0.78 降到 0.006 mm/s。
        m.opt.noslip_iterations = 5
        self.wrist_camera_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        if self.wrist_camera_id < 0:
            raise RuntimeError(f"模型缺少夹爪相机 {WRIST_CAMERA}")
        self.obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                         self.obj.body_name())
        self.obj_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                         self.obj.geom_name())
        self.obj_qadr = int(m.jnt_qposadr[int(m.body_jntadr[self.obj_bid])])
        self._mass0 = float(m.body_mass[self.obj_bid])
        self._inertia0 = m.body_inertia[self.obj_bid].copy()
        # 指垫几何体：改摩擦时必须连它一起改（见 _apply_friction）
        fb = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
              for n in ("left_finger", "right_finger")}
        self._pad_geoms = [g for g in range(m.ngeom)
                           if int(m.geom_bodyid[g]) in fb
                           and int(m.geom_contype[g]) == 1]
        self.detector = SlipDetector(self.gripper, self.obj.body_name())
        self.sensor = ObjectPoseSensor(scene.model, self.data,
                                       self.obj.body_name(), SensorSpec(), seed=7)
        self.servo = PositionVisualServo()
        return scene.model

    def _command_grip(self, face_force: float) -> None:
        """按**单面法向力**下夹持指令。

        `set_grip_force` 的入参是**腱作动力**，而 split 腱的 wrap 系数是 0.5/0.5，
        腱力平分到两指，所以单面法向力只有腱力的一半。
        本场景对外一律用单面口径（与防滑公式 N ≥ mg/(2μ) 里的 N 同一个量），
        在这里统一乘 2 转成腱力。

        不做这层换算的话，滑块标 3 N、理论栏标 2.04 N，看上去"够了"，
        实际单面只有 1.5 N，工件照掉——一个纯粹由口径不一致造成的假矛盾。
        """
        self.gripper.set_grip_force(2.0 * float(face_force), squeeze=0.004)
        self.gripper.command_width(
            max(self.obj_width - 0.008, self.gripper.width_min))

    def _apply_mass(self, mass: float) -> None:
        """就地改工件质量，不重编译模型。

        惯量必须按同一比例缩放：对均匀密度的长方体，几何不变时惯量正比于质量。
        只改质量不改惯量会得到一个"很重但很难转"的物体，物理上不自洽。
        """
        m = self.model
        s = float(mass) / self._mass0
        m.body_mass[self.obj_bid] = float(mass)
        m.body_inertia[self.obj_bid] = self._inertia0 * s

    def _apply_friction(self, mu: float) -> None:
        """把工件与**指垫**的滑动摩擦一起设成 mu。

        为什么必须一起改：实测确认 MuJoCo 把两个几何体的滑动摩擦按
        **逐元素取最大值**合成。Panda 指垫出厂 μ = 1.0，只改工件的话，
        生效的永远是 1.0——「工件摩擦系数」这个滑块会**完全没有效果**，
        而理论最小夹持力还会按错的 μ 去算。这类"参数看起来在动、
        其实被另一侧覆盖"的问题很难从现象上看出来。
        """
        m = self.model
        m.geom_friction[self.obj_gid, 0] = float(mu)
        for g in self._pad_geoms:
            m.geom_friction[g, 0] = float(mu)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self._apply_mass(self.get("obj_mass"))
        self._apply_friction(self.get("obj_mu"))
        self.gripper.restore_factory_servo()

        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        # 工件可以被人为挪开，模拟「掉了之后滚到别处」
        obj0 = np.array(self.OBJ_POS) + np.array([0.0, self.get("obj_jitter"), 0.0])
        self.data.qpos[self.obj_qadr:self.obj_qadr + 3] = obj0
        self.data.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # 抓取位姿的**姿态与开口宽度**由工件几何算（真机上来自 CAD 或点云主轴
        # 分析）。**位置**交给视觉伺服在线更新。
        # 注意这里按工件的**实时位姿**算，而不是标称位姿：工件被碰倒之后
        # 开合轴必须跟着转，否则需要的开口会变成工件的长边（见 _replan_grasp）。
        self._replan_grasp()
        # 开环模式的目标点：按**标称位置**算一次，之后不再改——这正是它会失败的原因
        self.p_nominal = (np.array(self.OBJ_POS)
                          + np.array([0.0, 0.0, self.get("grip_offset")]))
        self.p_grasp = self.p_nominal.copy()

        self.sensor.spec = SensorSpec(rate_hz=self.get("vs_rate"),
                                      latency=self.get("vs_latency"),
                                      pos_noise=self.get("vs_noise"))
        self.sensor.reset()
        self.servo.gain = self.get("vs_gain")
        self.servo.reset()
        # 末端姿态：保持 Home 的姿态，只做位置伺服。
        # 本场景的重点是夹持力而不是姿态对准，把姿态也一起动会引入额外的失败模式。
        _, self.R_ee = self.robot.fk(Q_HOME)
        self.R_ee = self.R_ee.copy()
        self.p_home = self.robot.fk(Q_HOME)[0].copy()

        self.ctrl = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=900.0, k_rot=45.0, zeta=1.0),
            q_null=Q_HOME, dt=self.dt)
        # 自适应也按**单面**口径工作，与理论值同量纲，曲线才能直接叠着看
        self.adaptive = AdaptiveGripForce(
            0.5, AdaptiveGripConfig(rise=self.get("rise"), decay=self.get("decay"),
                                    f_min=0.2, f_max=30.0))
        self.detector.margin_threshold = self.get("margin_thr")
        self.detector.speed_threshold = self.get("slip_speed_thr")

        self.gripper_width = None            # 本场景自己接管夹爪
        self.gripper.command_width(self.gripper.width_max)
        self._tau = np.zeros(N_ARM)
        self._phase = 0
        self._t_phase = 0.0
        self._n_retry = 0
        self._vs_err = 0.0
        self._vs_age = 0.0
        self._grip = 0.0
        self._reading = SlipReading()
        self._p = self.p_home.copy()
        self._p_seen = self.p_nominal.copy()
        self._p_hold = self.p_nominal.copy()
        self._sp = self.p_home.copy()        # 视觉伺服的内部设定点
        # ⭐ P1-06：真实末端位置历史。相机那一帧里夹爪在哪，误差就得拿哪个值算。
        self._tcp_hist: deque = deque(maxlen=self.SP_HIST_MAX)
        self._align_hold = 0                 # 连续在到位窗口内的拍数
        self._tcp_speed = 0.0                # 末端水平速率（低通后）
        self._p_prev = self._p.copy()
        self._lost_count = 0
        self._obj_ref = None
        self._tilt_ref = None
        self._R_now = self.R_ee.copy()
        self._R_des = self._aligned_ee_rotation()
        self._last_replan_frame = -1
        self._need_now = self.obj_width
        self.trail = TrailBuffer(capacity=240, min_step=0.003)
        self.note = ("抓取序列：接近 → 下降 → 闭合加力 → 提起 → 悬停。"
                     "把夹持力策略切成 fixed 并把力调低，看工件怎么滑掉")

    def on_change(self, key):
        if not hasattr(self, "adaptive"):
            return
        if key == "rise":
            self.adaptive.cfg.rise = self.get("rise")
        elif key == "decay":
            self.adaptive.cfg.decay = self.get("decay")
        elif key == "margin_thr":
            self.detector.margin_threshold = self.get("margin_thr")
        elif key == "slip_speed_thr":
            self.detector.speed_threshold = self.get("slip_speed_thr")
        elif key == "vs_gain":
            self.servo.gain = self.get("vs_gain")
        elif key in ("vs_latency", "vs_rate", "vs_noise"):
            self.sensor.spec = SensorSpec(rate_hz=self.get("vs_rate"),
                                          latency=self.get("vs_latency"),
                                          pos_noise=self.get("vs_noise"))

    # ------------------------------------------------------------ 状态机

    def _obj_pose(self):
        """工件当前的位置与旋转矩阵（仿真真值）。

        真机上这一路来自视觉的 6D 位姿估计；本场景只把**位置**做成带噪声、
        带延迟的观测（见 ObjectPoseSensor），姿态直接取真值，
        因为本场景要讲的是夹持力与滑移，不是位姿估计精度。
        ⚠️ 所以"倒了还能重抓"这条结论是**同模型自洽**级别，
        真机上还要叠加姿态估计误差。
        """
        p = self.data.qpos[self.obj_qadr:self.obj_qadr + 3].copy()
        quat = self.data.qpos[self.obj_qadr + 3:self.obj_qadr + 7].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        return p, R.reshape(3, 3)

    def _replan_grasp(self) -> None:
        """按工件**当前**位姿重新生成抓取位姿（开合轴 + 开口宽度）。

        早先只在 reset 时按标称位姿算一次，且用的是"取最短主轴"的通用版本。
        工件一旦被碰倒，最短主轴变成**竖直**方向、与自上而下的接近方向平行，
        `grasp_frame_from_points` 会直接抛 ValueError——抓取位姿根本构造不出来；
        即使勉强换个轴，需要的开口也会变成工件长边 90 mm，
        超过夹爪 80 mm 的行程。表现就是"一倒下就再也抓不起来"。

        改用 `constrained_grasp_frame`：接近方向钉死为自上而下，
        只在水平面内挑更短的那个方向当开合轴，所以任何倾倒姿态下
        需要的开口都是 42 mm，始终在行程内。
        """
        p_obj, R_obj = self._obj_pose()
        pts = box_corner_points(self.OBJ_HALF, p_obj, rot=R_obj)
        self._obj_pts = pts
        _, self.R_grasp, self.obj_width = constrained_grasp_frame(
            pts, approach=np.array([0.0, 0.0, -1.0]))
        # 行程不够时必须明说，而不是让它悄悄失败之后让人以为是控制没调好
        if self.obj_width > self.gripper.width_max - 2e-3:
            self.note = (f"⚠️ 需要开口 {self.obj_width * 1000:.0f} mm，"
                         f"超过夹爪行程 {self.gripper.width_max * 1000:.0f} mm，"
                         "这个姿态下**几何上就抓不住**，与控制参数无关。")

    def _aligned_ee_rotation(self) -> np.ndarray:
        """把末端绕**世界竖直轴**转到与规划开合轴对齐，返回期望姿态。

        为什么只转这一个自由度
        --------------------
        夹爪的开合轴在末端系里是固定的（实测为末端 y 轴），
        所以"让夹爪沿某个方向对夹"只能靠转手腕。
        但如果让姿态三个自由度都去跟 R_grasp，接近方向也会跟着变，
        俯冲抓取会变成侧向插入，引入一堆与本场景无关的失败模式。
        只绕竖直轴转，就既对上了开合方向，又保持自上而下接近。

        开合轴是一条**直线**不是矢量，所以 ψ 与 ψ+π 等价；
        取绝对值最小的那个，避免手腕为了转 170° 而撞上 J7 的 ±2.897 rad 限位。
        """
        target = self.R_grasp[:, 0].copy()
        target[2] = 0.0
        n = float(np.linalg.norm(target))
        if n < 1e-9:                       # 开合轴几乎竖直，转手腕也没用
            return self.R_ee
        target /= n
        cur = self.R_ee[:, 1].copy()       # 末端 y 轴 = 夹爪开合轴
        cur[2] = 0.0
        nc = float(np.linalg.norm(cur))
        if nc < 1e-9:
            return self.R_ee
        cur /= nc
        psi = float(np.arctan2(float(np.cross(cur, target)[2]),
                               float(np.dot(cur, target))))
        psi = (psi + np.pi / 2.0) % np.pi - np.pi / 2.0     # 折到 (−90°, 90°]
        c, s = np.cos(psi), np.sin(psi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ self.R_ee

    @staticmethod
    def _width_along(pts: np.ndarray, axis: np.ndarray) -> float:
        """点云沿给定方向的跨度——也就是「夹爪要张多大才夹得下」。"""
        proj = np.asarray(pts, dtype=float) @ np.asarray(axis, dtype=float)
        return float(proj.max() - proj.min())

    def _obj_half_along(self, axis: np.ndarray) -> float:
        """工件沿给定方向的**真实半跨度**（长方体支撑函数）。

        只给画图用，所以直接取仿真真姿态：箭头画在真工件上，
        起点却按感知位姿算，两者不一致反而更误导。
        """
        R = self.data.xmat[self.obj_bid].reshape(3, 3)
        return float(np.abs(R.T @ np.asarray(axis, float)) @ np.asarray(self.OBJ_HALF))

    #: 开口需求至少要改善这么多，才值得为它转手腕（米）
    REGRASP_HYST = 5.0e-3
    #: 指间必须留出的余量：跨度超过「行程 − 余量」就认定夹不下（米）
    REGRASP_CLEARANCE = 5.0e-3

    def _track_object(self, t: float) -> None:
        """接触前持续跟踪工件位姿，必要时转腕或中止下降。

        为什么判据不能写成「开合轴变了就重规划」
        ------------------------------------
        工件立着时横截面是 42×42 的**正方形**，两个方向一样宽，
        主轴分解挑哪一个在数值上是任意的，工件稍微偏航一点结果就会翻 90°。
        用「轴变了」当判据，手腕会毫无理由地来回甩。

        改用一个**物理量**：沿**当前**开合轴，工件还有多宽。
        - 正方形截面：无论轴怎么翻都是 42 mm，判据纹丝不动 → 不会误动作。
        - 工件被碰倒：沿旧轴变成 90 mm，超过夹爪 80 mm 行程 → 必须转腕。
        这个量自带迟滞，不需要额外滤波。

        两个阶段的处置不同，因为**离工件多远**决定了转腕安不安全：
        - 对准阶段悬停在 140 mm 高处，转腕碰不到任何东西，直接转。
        - 下降阶段已经贴近了，此时大幅转腕会像扫帚一样把工件扫飞。
          所以只有在「当前姿态已经几何上夹不下」时才中止，退回对准重来。
          夹得下就继续下去——不追求最优，只要求可行。
        """
        if self._phase > self.PH_DESCEND:
            return                            # 已经接触，交给力控，不再动姿态
        if self.sensor.n_frames == self._last_replan_frame:
            return                            # 还没有新的一帧，没有新信息
        self._last_replan_frame = self.sensor.n_frames

        axis_now = self._R_des[:, 1]          # 当前夹爪开合轴（世界系）
        self._replan_grasp()                  # 更新 R_grasp / obj_width / _obj_pts
        self._need_now = self._width_along(self._obj_pts, axis_now)

        if self._phase == self.PH_ALIGN:
            if self.obj_width < self._need_now - self.REGRASP_HYST:
                self._R_des = self._aligned_ee_rotation()
        elif self._need_now > self.gripper.width_max - self.REGRASP_CLEARANCE:
            self.note = (f"⚠️ 下降途中工件姿态变了：当前开合方向上需要 "
                         f"{self._need_now * 1e3:.0f} mm，超过行程 "
                         f"{self.gripper.width_max * 1e3:.0f} mm。中止下降，"
                         "退回对准换方向重抓。")
            self._retry(t)

    #: 阶段编号：0 搜索对准 / 1 下降 / 2 闭合加力 / 3 提起 / 4 悬停
    PH_ALIGN, PH_DESCEND, PH_GRIP, PH_LIFT, PH_HOLD = 0, 1, 2, 3, 4
    ALIGN_TOL = 4.0e-3          # 水平对准到多少算够（米）——「到位窗口」
    ALIGN_TIMEOUT = 6.0         # 对不准就放弃重来，防止卡死
    #: ⭐ 「到位滤波时间」：误差必须**连续**这么多拍都在窗口内才算真到位。
    #: 这是工业伺服驱动器判「位置到达」的标准做法（In-Position Window +
    #: In-Position Time）。只看瞬时误差会在**还在晃**的时候误判到位：
    #: 设定点摆过零点的那一瞬间误差当然小，但速度是最大的。
    #: P1-06 把延迟放进反馈环之后，对准段本来就更晃，这个门必须补上——
    #: 否则会带着超调进入下降段，抓偏，悬停时蠕变从 0.135 mm 涨到 0.599 mm。
    ALIGN_HOLD_STEPS = 25       # 25 拍 @ 500 Hz = 50 ms
    #: ⭐ 「零速窗口」：到位判据的**另一半**。误差小 ≠ 停下来了——
    #: 设定点摆过零点的那一刹那误差最小，速度却最大。工业伺服的
    #: 「位置到达」信号同时要求 |误差| < 窗口 **且** |速度| < 零速阈值。
    ALIGN_SPEED_TOL = 8.0e-3    # m/s，末端水平速度低于此才算停稳
    HOVER = 0.14                # 抓取前悬停在工件上方多高

    def _perceived_object(self, t):
        """当前**认为**工件在哪。

        visual   用传感器观测（带噪声、延迟、有限更新率）
        openloop 用开始前算好的标称位置，之后再也不更新

        这一个函数就是本场景的核心对照：换掉它的返回值，
        整条控制链完全不变，成败却完全不同。
        """
        if self.choice("closed_loop") == "openloop":
            return self.p_nominal.copy(), 0.0
        p, age = self.sensor.read(t)
        if p is None:                      # 还没有任何一帧可用
            return self.p_nominal.copy(), 0.0
        return p + np.array([0.0, 0.0, self.get("grip_offset")]), age

    #: 末端位置历史缓存的最大长度（0.20 s 最大延迟 @ 500 Hz，留一倍余量）
    SP_HIST_MAX = 256

    #: 设定点允许领先真实末端的最大距离（m）。刚度 900 N/m ⇒ 力上限约 45 N。
    SP_LEAD_MAX = 0.05

    def _tcp_delayed(self, t_capture: float) -> np.ndarray:
        """取出 `t_capture` 时刻**末端真实在哪**——即那一帧相机看到的自己。

        ⭐ 这里必须是**真实末端位置**，不能是内部设定点。
        眼在手相机测的是「工件相对相机」，相机装在真实末端上，
        所以图像给出的误差是 p_obj(t−τ) − p_tcp(t−τ)，两端都是物理量。
        用设定点当反馈会把机械臂整个从环路里摘出去（实测两者相差
        82 mm RMS / 232 mm 峰值），得到的"稳定边界"只是一个纯软件积分器的性质。

        没有那么早的历史（刚开始跑）时退化成最早的一条，
        这只影响头几拍，不影响稳定边界。
        """
        if not self._tcp_hist:
            return self._p.copy()
        for t_h, p_h in reversed(self._tcp_hist):
            if t_h <= t_capture:
                return p_h.copy()
        return self._tcp_hist[0][1].copy()

    #: 判定"丢件"需要连续多少个控制周期没有接触（去抖）
    LOST_DEBOUNCE = 200                      # 0.4 s @ 500 Hz

    def _object_lost(self, dt_ph: float) -> bool:
        """工件是不是掉了 / 滑出去了。

        判据是**指尖与工件完全没有接触**，不是"工件高度低"。
        高度判据踩过一次坑：刚进入提起阶段时工件本来就还在桌面高度，
        判据在提起发生之前就触发，于是每次都在 2→3 的瞬间被判失败，
        永远抓不起来——现象是"一直重抓"，根因却是判据的时机不对。

        还要**去抖**：夹持力斜坡加载时接触会短暂断开，
        单帧无接触就判丢会误报。要求连续 0.4 秒都没有接触才算。
        """
        if dt_ph < 0.5:                      # 刚进入本阶段，给动作留出时间
            self._lost_count = 0
            return False
        if self.gripper.contacts_with(self.obj.body_name()):
            self._lost_count = 0
            return False
        self._lost_count += 1
        return self._lost_count >= self.LOST_DEBOUNCE

    def _advance(self, t, p_seen):
        """推进状态机，返回 (末端目标位置, 夹爪宽度指令或 None)。

        与早先的「按时间表走」相比，最重要的差别是**每个阶段都有失败出口**：
        对不准会超时重来，抓失败会回到对准重抓。
        开环版本没有这个能力——它只会照着时间表把动作做完，然后停在那儿。
        """
        dt_ph = t - self._t_phase
        w_open = min(self.obj_width + 0.03, self.gripper.width_max)
        above = p_seen + np.array([0.0, 0.0, self.HOVER])

        if self._phase == self.PH_ALIGN:
            # 只在水平面上对准，高度保持在悬停高度
            self._vs_err = float(np.linalg.norm((self._p - above)[:2]))
            settled = (self._vs_err < self.ALIGN_TOL
                       and self._tcp_speed < self.ALIGN_SPEED_TOL)
            if settled:
                self._align_hold += 1
            else:
                self._align_hold = 0
            if self._align_hold >= self.ALIGN_HOLD_STEPS and dt_ph > 0.6:
                self._enter(self.PH_GRIP if False else self.PH_DESCEND, t)
            elif dt_ph > self.ALIGN_TIMEOUT:
                self._enter(self.PH_ALIGN, t)          # 重新计时再试
                self._n_retry += 1
            return above, w_open

        if self._phase == self.PH_DESCEND:
            s_ = 0.5 - 0.5 * np.cos(np.pi * min(dt_ph / self.T_DESCEND, 1.0))
            if dt_ph > self.T_DESCEND:
                self._enter(self.PH_GRIP, t)
            return above + s_ * (p_seen - above), w_open

        if self._phase == self.PH_GRIP:
            if dt_ph > self.T_GRIP:
                self._enter(self.PH_LIFT, t)
            return self._p_hold, None

        if self._phase == self.PH_LIFT:
            s_ = 0.5 - 0.5 * np.cos(np.pi * min(dt_ph / self.T_LIFT, 1.0))
            if self._object_lost(dt_ph):                # 提到一半掉了 → 重抓
                self._retry(t)
                return self._p_hold, w_open
            if dt_ph > self.T_LIFT:
                self._enter(self.PH_HOLD, t)
            return self._p_hold + np.array(
                [0.0, 0.0, s_ * self.get("lift_height")]), None

        # PH_HOLD
        if self._object_lost(dt_ph):
            self._retry(t)
            return self._p_hold, w_open
        return self._p_hold + np.array([0.0, 0.0, self.get("lift_height")]), None

    def _enter(self, phase, t):
        self._phase = phase
        self._t_phase = t
        if phase == self.PH_ALIGN:
            # 每次（重新）进入对准都按工件**当前**位姿重规划抓取位姿。
            # 这是"掉了/倒了还能重抓"的关键：工件躺倒之后开合轴必须跟着转，
            # 否则重抓多少次都是照着旧方向去夹，永远抓不住。
            self._replan_grasp()
            self._R_des = self._aligned_ee_rotation()
        if phase == self.PH_GRIP:
            self._p_hold = self._p.copy()               # 就地夹，不再移动末端

    def _retry(self, t):
        """抓失败了：松开、回到对准、重来。

        这是闭环相对开环最实际的价值：**失败可以恢复**。
        开环流程走完就停在那儿，工件掉了也不会有任何反应。
        """
        self._n_retry += 1
        self._obj_ref = None
        self._tilt_ref = None
        self.adaptive.reset(0.5)
        self.gripper.restore_factory_servo()
        self._enter(self.PH_ALIGN, t)

    def control(self, t, q, v):
        # 观测位置取模型里真实眼在手相机的世界坐标，而不是在 TCP 上方虚构一个点。
        # 传感器仍直接读取工件三维位置（不处理 RGB 像素），但至少距离/视场的
        # 几何基准与用户在原生窗口切到的那台相机完全一致。
        self.sensor.update(t, self.data.cam_xpos[self.wrist_camera_id])
        p_seen, age = self._perceived_object(t)
        self._vs_age = age * 1e3
        self._p_seen = p_seen
        # 绿球标记的是「机器人**以为**该去抓哪里」，所以必须跟着感知走。
        # 早先它在 reset 里定死在标称位置，于是闭环明明精准跟上了工件，
        # 画面上标记却离工件 60 mm——机器人是对的，标记在骗人。
        # 现在开环模式下它会留在原地不动，那正是开环失效的可视化证据。
        self.p_grasp = p_seen.copy()
        # 末端水平速率，一阶低通（时间常数 ~20 ms）。差分速度很毛躁，
        # 不滤波的话零速判据会被噪声反复打断，永远凑不满到位滤波时间。
        v_raw = float(np.linalg.norm((self._p - self._p_prev)[:2])) / self.dt
        self._p_prev = self._p.copy()
        self._tcp_speed += 0.1 * (v_raw - self._tcp_speed)
        self._track_object(t)

        p_des, w_cmd = self._advance(t, p_seen)

        # 抖动激励：给末端一个竖直正弦位移，等效于工件受到 (g + a) 的载荷
        shake = self.get("shake")
        self._accel = 0.0
        if self._phase == self.PH_HOLD and shake > 0.0:
            w = 2.0 * np.pi * 1.5
            self._accel = shake * np.sin(w * t)
            p_des = p_des + np.array([0.0, 0.0, -self._accel / (w * w)])

        # 视觉伺服只在对准阶段生效：接触之后再按观测挪目标点会把工件推走。
        # 这也是真机的做法——**接触前用视觉，接触后交给力控**。
        if self._phase == self.PH_ALIGN and self.choice("closed_loop") == "visual":
            # 伺服维护**自己的设定点**并累加，不能写成「当前位置 + 增量」。
            # 后者的设定点永远只领先实际位置零点几毫米，阻抗力只有零点几牛，
            # 机械臂原地蠕动——看起来像"伺服增益太小"，其实是结构错了。
            #
            # ⭐⭐ 第五轮审核 P1-06：反馈端必须是**真实末端**，不是内部设定点。
            # 相机在 t−τ 时刻拍下的那一帧里，工件和夹爪**都是 t−τ 时刻的样子**，
            # 而"夹爪在哪"是相机实际测到的物理位置：
            #     sp(k+1) = sp(k) + λ·dt·( p_obj(t−τ) − p_tcp(t−τ) )
            # 第四轮曾把反馈端写成设定点历史 sp(k−d)：
            #     sp(k+1) = sp(k) + λ·dt·( p_obj(t−τ) − sp(k−d) )
            # 那条递推确实会因延迟失稳（离散边界 λτ = 2d·sin(π/(4d+2))），
            # 但它**整个把机械臂摘出了环路**——实测 |sp − TCP| 达 82 mm RMS、
            # 232 mm 峰值，所以那个"边界"只是一个纯软件积分器的性质，
            # 和这台机器人没关系。现在两端都是物理量，环路里含臂的动力学。
            self._tcp_hist.append((t, self._p.copy()))
            tcp_seen = self._tcp_delayed(t - age)
            self._sp = self._sp + self.servo.step(tcp_seen, p_des, self.dt)
            # ⭐ 牵引绳：限制设定点**领先真实末端**的距离。
            # 设定点是对末端误差的积分，手臂跟不上时它会一路积出去
            # （实测 λ=15 时 sp 领先 267 mm，λ=25 时 270 mm）——就是积分饱和。
            # 阻抗刚度 900 N/m，领先 50 mm 对应 45 N，这是自由段接近时的力上限。
            # ⚠️ 它是一个**饱和**，只把发散限成有界振荡，**不提供稳定性**：
            # 判"稳不稳"绝不能看"有没有跑飞"，要看小信号下振幅是涨还是消。
            lead = self._sp - self._p
            n_lead = float(np.linalg.norm(lead))
            if n_lead > self.SP_LEAD_MAX:
                self._sp = self._p + lead * (self.SP_LEAD_MAX / n_lead)
            p_des = self._sp
        else:
            self._sp = p_des.copy()          # 非对准阶段跟随，避免切回来时跳变

        self._tau = self.ctrl.compute(q, v, p_des, self._R_des, dt=self.dt)

        if w_cmd is not None:
            self.gripper.restore_factory_servo()
            self.gripper.command_width(w_cmd)
            self._grip = 0.0
            self._reading = SlipReading()
        else:
            reading = self.detector.read()
            self._reading = reading
            if self.choice("mode") == "adaptive":
                target = self.adaptive.update(reading, self.dt)
            else:
                target = self.get("grip_cmd")
            if self._phase == self.PH_GRIP:
                ramp = min((t - self._t_phase) / 0.5, 1.0)
                target = max(target * ramp, 0.3)
            self._grip = float(target)
            self._command_grip(self._grip)

        self._p, self._R_now = self.robot.fk(q)
        self._p = self._p.copy()
        self._R_now = self._R_now.copy()
        self.trail.push(self._p)
        if self._phase >= self.PH_GRIP and self._obj_ref is None:
            self._obj_ref = self._R_now.T @ (
                self.data.xpos[self.obj_bid] - self._p)
        return self._tau

    def telemetry(self, t, q, v):
        m = self.model
        r = self._reading
        mass = float(m.body_mass[self.obj_bid])
        mu = r.n_contacts and self.detector and 0.0
        ids = self.gripper.contacts_with(self.obj.body_name())
        mu_eff = effective_friction(m, self.data, ids)
        # 理论对照：用真实质量与**实际生效**的摩擦系数，加上抖动带来的惯性载荷。
        # 还没接触时读不到生效摩擦，退回用设定值——否则这一栏在接触前是空的，
        # 没法在"下手之前"就知道该给多大力。
        mu_theory = mu_eff if mu_eff > 0 else self.get("obj_mu")
        theory = min_normal_force(mass, mu_theory, accel=abs(self._accel))
        obj_p = self.data.xpos[self.obj_bid]
        drop = 0.0
        tilt = 0.0
        if self._obj_ref is not None:
            # 工件在**夹爪坐标系**里挪了多少，才是真正的滑移。
            # 必须同时扣掉夹爪的平移**和转动**：只扣平移的话，
            # 夹爪在负载下转 7.6° 就会被误读成 7 mm 的滑落，
            # 而实测切向力只有摩擦上限的 44%，根本没滑。
            rel = self._R_now.T @ (obj_p - self._p)
            drop = float((rel - self._obj_ref)[2]) * -1e3
            # 工件相对夹爪的转角：转动逃逸是与平移滑移并列的另一种失效模式
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, self.data.xquat[self.obj_bid])
            R_rel = self._R_now.T @ Rq.reshape(3, 3)
            ang = float(np.degrees(np.arccos(np.clip(
                (np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))))
            if self._tilt_ref is None:
                self._tilt_ref = ang
            tilt = abs(ang - self._tilt_ref)
        lim = self.robot.tau_limit
        return dict(
            grip_force=r.face_force,
            grip_theory=theory, margin=r.margin,
            slip_speed=r.slip_speed * 1e3, mu_eff=mu_eff,
            obj_height=float(obj_p[2]) * 1e3, obj_drop=drop, obj_tilt=tilt,
            n_slip=float(self.adaptive.n_slip_events),
            obj_mass=mass, phase=float(self._phase),
            vs_err=self._vs_err * 1e3, vs_age=self._vs_age,
            target_err=float(np.linalg.norm(
                (self._p_seen - self.data.xpos[self.obj_bid])[:2])) * 1e3,
            n_retry=float(self._n_retry),
            need_now=self._need_now * 1e3,
            grasp_yaw=float(np.degrees(np.arctan2(
                self.R_grasp[1, 0], self.R_grasp[0, 0]))),
            obj_lean=float(np.degrees(np.arccos(np.clip(
                abs(self._obj_pose()[1][2, 2]), -1.0, 1.0)))),
            tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
            tau_limit=lim,
            tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)

    def _planned_frame(self) -> np.ndarray:
        """要画出来的抓取坐标架：列为 [开合轴, 副轴, 接近轴]。

        用**正在执行**的 `_R_des` 而不是 `R_grasp`。
        `R_grasp` 是每帧重算的候选解，工件立着时横截面是正方形、
        主轴方向本来就任意，直接画它会让坐标架 90° 来回抖。
        画正在执行的方向，看到的才是「机器人打算怎么夹」。
        """
        axis = self._R_des[:, 1]
        a = np.array([0.0, 0.0, -1.0])
        second = np.cross(a, axis)
        n = float(np.linalg.norm(second))
        if n < 1e-9:
            return self.R_grasp
        return np.column_stack([axis, second / n, a])

    def overlays(self):
        items = [
            Sphere(self.p_grasp, radius=0.012, color=C_TARGET, alpha=0.5),
            Frame(self.p_grasp, self._planned_frame(), size=0.06, width=0.004),
            Trail(self.trail.points, color=C_ERROR, width=0.003),
        ]
        # 夹持力画成两支相对的箭头，长度按比例尺换算。
        # 方向取**实际**指尖开合轴（末端 y 轴），不是规划候选轴：
        # 画的是真实接触力，就该沿真实的指垫法向。
        if self._reading.n_contacts:
            L = min(self._reading.face_force * 0.012, 0.10)
            axis = self._R_now[:, 1]
            c = self.data.xpos[self.obj_bid]
            # 箭头必须落在**工件沿这个方向的真实表面**上。写死 0.045 是长半轴，
            # 而开合方向的半宽只有 21 mm，箭头会悬在工件外 24 mm 的空中，
            # 看上去像力作用在空气上。
            h = self._obj_half_along(axis)
            items.append(Arrow(c + axis * (h + L), c + axis * h,
                               color=C_FORCE_TRUE, width=0.008))
            items.append(Arrow(c - axis * (h + L), c - axis * h,
                               color=C_FORCE_TRUE, width=0.008))
        return items


# ====================================================== 8 姿态解算（ESKF）


class AttitudeScene(Scene):
    """姿态解算：陀螺积分 + 重力修正，以及偏航为什么定不住。

    与 EstimationScene 的分工：那一篇把**姿态当已知**，只估位置速度零偏；
    这一篇**把姿态也估出来**，于是重力泄漏、偏航不可观这两个足式状态估计
    最核心的问题才显现出来。
    """

    name = "attitude"
    steps = STEPS_BY_SCENE["attitude"]
    title = "姿态解算：陀螺、重力修正与偏航不可观"
    camera = dict(azimuth=138.0, elevation=-14.0, distance=1.30,
                  lookat=(0.40, 0.0, 0.45))
    intro = (
        "上一篇把<b>姿态当成已知</b>。真实浮动基座没有这种奢侈品——"
        "姿态只能靠陀螺积分 + 加速度计的重力方向估出来。"
        "这一篇把姿态也放进状态里，看会冒出什么新问题。"
    )
    law = ControlLaw(
        formula="预测  R⁺ = R·Exp((ω−b̂_g)·dt),  a = R·(f−b̂_a)+g\n"
                "重力修正  z = f,  h = −Rᵀg + b̂_a,  H_θ = −[Rᵀg]ₓ\n"
                "状态 x = [ p ; v ; R ; b_a ; b_g ]，误差 δx 15 维",
        terms=[
            LawTerm("‖δθ‖", "姿态误差总量", key="att_err", unit="°", digits=3),
            LawTerm("δθ_水平", "横滚+俯仰误差。重力修正能定住它",
                    key="tilt_err", unit="°", digits=3),
            LawTerm("δθ_偏航", "偏航误差。⭐ 重力修正**定不住**它",
                    key="yaw_err", unit="°", digits=3),
            LawTerm("b̂_g", "估计的陀螺零偏（z 轴）", key="bg_hat",
                    unit="rad/s", digits=5),
            LawTerm("b_g", "陀螺零偏真值（仿真上帝视角）", key="bg_true",
                    unit="rad/s", digits=5),
            LawTerm("g·sin δθ_水平", "**倾斜**误差泄漏出的伪加速度。"
                    "偏航误差不参与：δa = −θ×g，θ∥g 时为零",
                    key="leak", unit="m/s²", digits=4),
        ],
        blocks=[
            Block("陀螺 ω（高频）", "有噪声、有零偏，积分一次进姿态", "input"),
            Block("R⁺ = R·Exp((ω−b̂_g)dt)", "在 SO(3) 上走一步，不是加法", "block"),
            Block("加速度计 f", "准静态时指出重力朝哪", "input"),
            Block("重力修正 H_θ = −[Rᵀg]ₓ", "⭐ 零空间正是重力轴 ⇒ 偏航看不见",
                  "block"),
            Block("融合后的 R̂, b̂_g", "", "output"),
            Block("MuJoCo 真姿态", "仅用于对照，真机上没有这一路", "feedback"),
        ],
        note="陀螺零偏经**一次**积分进姿态（θ≈b_g·t，线性），"
             "而姿态误差又把重力泄漏成伪加速度、再积**两次**进位置——"
             "所以陀螺零偏对位置的影响是 t³ 量级，比加速度计零偏的 t² 更狠。",
    )
    params = [
        Param("motion", "运动方式", default="still",
              choices=["still", "rotate", "translate", "both"], restart=True,
              group="激励",
              help="still 静止不动；rotate 只转不平移；translate 只平移不转。",
              effect="<b>本场景最重要的开关之一。</b>可观性完全取决于激励："
                     "静止时秩只有 11，一动起来就能补满。看「可观性秩」那一栏。"),
        Param("gyro_bias_z", "陀螺零偏真值 b_g,z", 0.0, 0.05, 0.01,
              unit="rad/s", symbol="b_g", group="IMU",
              help="绕竖直轴的常值角速度偏差。0.01 rad/s ≈ 0.57 °/s。",
              effect="<b>关掉重力修正后，姿态会按 θ≈b_g·t 线性漂</b>。"
                     "打开修正则被估出来，看两条零偏线靠拢。"),
        Param("acc_bias_x", "加速度计零偏 b_a,x", -0.2, 0.2, 0.05,
              unit="m/s²", group="IMU",
              effect="它和倾角在重力修正里**分不开**（小倾斜和小零偏读数一样）。"
                     "只有靠运动才能拆开——又一次持续激励。"),
        Param("gyro_density", "陀螺噪声密度", 0.0, 5e-3, 2e-4,
              unit="rad/s/√Hz", group="IMU",
              effect="调大 → 姿态估计变毛躁，「偏航标准差」涨得更快。"),
        Param("acc_density", "加速度计噪声密度", 0.0, 0.05, 2e-3,
              unit="m/s²/√Hz", group="IMU",
              effect="调大 → 重力修正这一路变糙，横滚俯仰误差变大。"),
        Param("use_level", "加速度计水平修正", default="off",
              choices=["on", "off"], group="量测开关",
              help="用「重力朝哪」定住横滚与俯仰。",
              effect="<b>关掉 → 横滚俯仰立刻开始随陀螺零偏线性漂。</b>"
                     "打开 → 被拉回来，但偏航依然漂。"),
        Param("use_pos", "运动学位置量测", default="off",
              choices=["on", "off"], group="量测开关",
              help="关节编码器 + 正运动学给出的**世界系绝对位置**。",
              effect="<b>本场景最反直觉的一个开关。</b>它看起来只管位置，"
                     "实际上是<b>偏航能不能定住</b>的关键——"
                     "关掉后「可观性秩」从 15 掉到 9。原因见讲解。"),
        Param("use_heading", "外部航向参考", default="off",
              choices=["on", "off"], group="量测开关",
              help="对应真机上的磁力计 / 视觉 / 激光定位。",
              effect="打开 → 偏航立刻被钉死。用来对照「有没有外部参考」的差别。"),
        Param("tilt0", "初始倾角误差", 0.0, 10.0, 3.0, unit="°",
              restart=True, group="初始误差",
              effect="故意给滤波器一个错的初始横滚。重力修正会把它拉回来，"
                     "看「横滚俯仰误差」怎么衰减。"),
        Param("yaw0", "初始偏航误差", 0.0, 10.0, 3.0, unit="°",
              restart=True, group="初始误差",
              effect="<b>同样大小的初始误差，横滚能被拉回、偏航不能</b>——"
                     "这一对照是本场景的核心。"),
    ]
    readouts = [
        Readout("att_err", "姿态误差总量", "°", 3),
        Readout("tilt_err", "横滚+俯仰误差", "°", 3,
                help="重力修正能定住的那两个自由度。"),
        Readout("yaw_err", "偏航误差", "°", 3,
                help="⭐ 重力修正**结构性**定不住它：H_θ = −[Rᵀg]ₓ 的零空间"
                     "正是重力轴本身。"),
        Readout("yaw_sigma", "偏航标准差 σ_yaw", "°", 3, source="internal",
                compare_with="yaw_err",
                help="滤波器自己认为偏航有多不准。没有外部航向参考时它会一直涨。"),
        Readout("rank", "可观性秩", "", 0, theory=True,
                help="离散可观性矩阵 O=[H₀Φ₀; H₁Φ₁; …] 的数值秩，满秩为 15。"
                     "它把「哪些状态其实看不见」变成一个能直接读的整数。"),
        Readout("bg_hat", "陀螺零偏估计 b̂_g,z", "rad/s", 5),
        Readout("bg_true", "陀螺零偏真值", "rad/s", 5, source="truth",
                compare_with="bg_hat",
                help="仿真才有。真机上你不知道自己估得准不准。"),
        Readout("leak", "重力泄漏伪加速度", "m/s²", 4, theory=True,
                help="g·sin(δθ_水平)，由**倾斜**误差独立算出，1° 就是 0.171 m/s²。"
                     "⭐ 用的是倾斜分量而非姿态误差总量：绕重力轴的偏航误差"
                     "不产生任何重力泄漏（δa = −θ×g，θ 平行于 g 时叉乘为零）。"),
        Readout("err_pos", "位置估计误差", "mm", 2),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
    ]
    traces = [
        Trace("tilt_err", "横滚+俯仰误差", "#4da3ff", unit="°"),
        Trace("yaw_err", "偏航误差", "#ff2d78", unit="°"),
        Trace("yaw_sigma", "偏航标准差", "#ffd166", axis="sigma", unit="°"),
    ]
    bars = [_torque_bars()]

    UPDATE_EVERY = 10
    LEVEL_EVERY = 5

    def build(self):
        scene = build_panda_scene([], with_visuals=True, imu_body="hand",
                                  imu_pos=tuple(PANDA_TCP_OFFSET))
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self._sensor_adr = {}
        for nm in ("imu_truth_pos", "imu_truth_vel"):
            i = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SENSOR, nm)
            self._sensor_adr[nm] = (int(scene.model.sensor_adr[i]),
                                    int(scene.model.sensor_dim[i]))
        return scene.model

    def _truth(self, name):
        a, k = self._sensor_adr[name]
        return self.data.sensordata[a:a + k].copy()

    def _mask(self) -> np.ndarray:
        """按「运动方式」选择哪几个关节动。

        转与平移在机械臂上没法完全解耦，这里用**关节子集**近似：
        腕部关节（5/6/7）主要改末端姿态，肩肘关节（1/2/4）主要改末端位置。
        这是个近似，不是严格分解——讲解里明说了。
        """
        mode = self.choice("motion")
        if mode == "still":
            return np.zeros(N_ARM)
        if mode == "rotate":
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        if mode == "translate":
            return np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        return np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.imu = SimulatedIMU(
            self.model, self.data,
            noise=ImuNoise(acc_density=self.get("acc_density"),
                           gyro_density=self.get("gyro_density"),
                           acc_bias_walk=1e-4,
                           acc_bias0=(self.get("acc_bias_x"), 0.0, 0.0),
                           gyro_bias_walk=1e-6,
                           gyro_bias0=(0.0, 0.0, self.get("gyro_bias_z"))),
            seed=20260827)
        self.est = ErrorStateKalmanFilter(
            acc_noise=max(self.get("acc_density"), 1e-4),
            gyro_noise=max(self.get("gyro_density"), 1e-5),
            acc_bias_walk=1e-4, gyro_bias_walk=1e-6,
            pos_noise=5e-4, lev_noise=0.15,
            gravity=self.model.opt.gravity.copy())

        # 初始姿态**故意给错**：这是本场景要看的东西（能不能被拉回来）
        R_true = self.imu.rotation()
        tilt = np.radians(self.get("tilt0"))
        yaw = np.radians(self.get("yaw0"))
        R0 = R_true @ so3_exp(np.array([tilt, 0.0, yaw]))
        self.est.reset(self._truth("imu_truth_pos"),
                       self._truth("imu_truth_vel"), R0,
                       th_std=max(np.hypot(tilt, yaw), 0.02),
                       ba_std=0.15, bg_std=0.02)

        self._q_mask = self._mask()
        self._tau = np.zeros(N_ARM)
        self._k = 0
        self._t = 0.0
        self._p_hat = self._truth("imu_truth_pos")
        self._p_true = self._p_hat.copy()
        self.trail_hat = TrailBuffer(capacity=300, min_step=0.002)
        self.trail_true = TrailBuffer(capacity=300, min_step=0.002)
        self._tel = dict(att_err=0.0, tilt_err=0.0, yaw_err=0.0, yaw_sigma=0.0,
                         rank=0, bg_hat=0.0, bg_true=self.get("gyro_bias_z"),
                         leak=0.0, err_pos=0.0)
        self.note = ("默认<b>三路量测全关</b>，只有陀螺在积分——先看姿态怎么漂，"
                     "再一路一路打开")

    def on_change(self, key):
        if not hasattr(self, "est"):
            return
        if key == "gyro_bias_z":
            self.imu.true_gyro_bias[2] = self.get("gyro_bias_z")
        elif key == "acc_bias_x":
            self.imu.true_bias[0] = self.get("acc_bias_x")
        elif key == "gyro_density":
            self.imu.noise.gyro_density = self.get("gyro_density")
            self.est.gyro_noise = max(self.get("gyro_density"), 1e-5)
        elif key == "acc_density":
            self.imu.noise.acc_density = self.get("acc_density")
            self.est.acc_noise = max(self.get("acc_density"), 1e-4)

    def _ref(self, t):
        w = 2.0 * np.pi * 0.35
        a = 0.28 * self._q_mask
        return (Q_HOME + a * np.sin(w * t), a * w * np.cos(w * t),
                -a * w * w * np.sin(w * t))

    def control(self, t, q, v):
        self._t = t
        qd, vd, ad = self._ref(t)
        aq = ad + 40.0 * (vd - v) + 400.0 * (qd - q)
        tau = self.reg.mass_matrix(q) @ aq + self.reg.bias(q, v)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)

        f_body, w_body = self.imu.read(self.dt)
        self.est.predict(f_body, w_body, self.dt)

        if self.choice("use_level") == "on" and self._k % self.LEVEL_EVERY == 0:
            self.est.update_gravity(f_body)
        if self.choice("use_pos") == "on" and self._k % self.UPDATE_EVERY == 0:
            # 量测由编码器 + 正运动学给出，不抄真值——抄真值就是自证
            p_fk, _ = self.robot.fk(q)
            self.est.update_position(p_fk)
        if self.choice("use_heading") == "on" and self._k % self.UPDATE_EVERY == 0:
            self.est.update_attitude(self.imu.rotation(), sigma=2e-3)

        R_true = self.imu.rotation()
        p_true = self._truth("imu_truth_pos")
        # 姿态误差在**体坐标系**里分解：沿重力轴的分量是偏航，其余是横滚俯仰
        dth = so3_log(self.est.R.T @ R_true)
        u = self.est.R.T @ self.est.gravity
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        yaw_part = float(dth @ u)
        tilt_part = float(np.linalg.norm(dth - yaw_part * u))
        att = float(np.linalg.norm(dth))

        self._p_hat = self.est.p.copy()
        self._p_true = p_true
        self.trail_hat.push(self._p_hat)
        self.trail_true.push(p_true)

        self._tel = dict(
            att_err=np.degrees(att),
            tilt_err=np.degrees(tilt_part),
            yaw_err=abs(np.degrees(yaw_part)),
            yaw_sigma=self.est.yaw_uncertainty_deg(),
            rank=self.est.observability_matrix_rank(),
            bg_hat=float(self.est.bg[2]),
            bg_true=float(self.imu.true_gyro_bias[2]),
            # 理论对照：伪加速度 = g·sin(**倾斜**误差)，独立算，不读滤波器中间量。
            # 必须用 tilt_part 而非总误差 att：绕重力轴的偏航误差不产生重力泄漏
            # （δa = −θ×g，θ ∥ g 时叉乘为零），用 att 会把纯偏航虚报成泄漏。
            leak=float(np.linalg.norm(self.est.gravity)) * float(np.sin(tilt_part)),
            err_pos=float(np.linalg.norm(self._p_hat - p_true)) * 1e3)
        self._k += 1
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        out = dict(self._tel)
        out.update(tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
                   tau_limit=lim,
                   tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)
        return out

    def overlays(self):
        # ⭐ 两个坐标架**画在同一个点上**（都在真实位置），这样它们的夹角
        # 就是纯粹的姿态误差，一眼可读。
        # 早先把估计姿态画在估计位置上，结果关掉位置量测后位置发散好几米，
        # 估计坐标架直接飞出画面——姿态误差反而看不见了。
        # 位置误差由下面那对球 + 连线单独讲，两件事分开画。
        items = [Frame(self._p_true, self.imu.rotation(), size=0.11, width=0.006),
                 Frame(self._p_true, self.est.R, size=0.08, width=0.004, alpha=0.55)]
        # 位置误差太大时不再画那对球：连线会拉成一条冲出画面的长线，
        # 除了添乱没有信息。此时误差量级请看读数。
        if float(np.linalg.norm(self._p_hat - self._p_true)) < 0.5:
            items += _pair_markers(self._p_true, self._p_hat,
                                   c_actual=C_FORCE_EST, a_target=0.55)
            items.append(Trail(self.trail_hat.points, color=C_FORCE_EST,
                               width=0.004))
        items.append(Trail(self.trail_true.points, color=C_TARGET,
                           width=0.004, alpha=0.6))
        return items


# ====================================================== 9 拖动示教


class TeachingScene(Scene):
    """拖动示教（零力控制）：让人能用手直接把机械臂拖到想要的位姿。"""

    #: 被控对象真实摩擦里 ``tanh(v/a)`` 的尺度 a。控制器用的是 ``sign``，
    #: 两者在低速段幅值不一致——这正是 P1-05 的根源，所以提成常量，
    #: 让被动性判据能引用到**同一个** a，而不是各写各的。
    TANH_SCALE = 2e-3
    #: 死区放到这个值以上，摩擦补偿在正常示教速度下就等于没开了。
    #: 超过它宁可如实报警，也不假装"已保护"。
    DEADBAND_CAP = 0.1

    name = "teaching"
    steps = STEPS_BY_SCENE["teaching"]
    title = "拖动示教：把机械臂做成「没有重量」"
    camera = dict(azimuth=138.0, elevation=-14.0, distance=1.45,
                  lookat=(0.36, 0.0, 0.45))
    intro = (
        "不写代码、不用示教器，<b>直接用手把机械臂拖到想要的位置</b>，"
        "松手它就停在那儿。这是产线上教机器人干活最快的方式。"
    )
    law = ControlLaw(
        formula="τ = g(q) + C(q,q̇)q̇ + η·(f_c·sign(q̇) + f_v·q̇) − D·q̇\n"
                "        └── 抵消自重 ──┘   └── 抵消摩擦 ──┘      └阻尼┘",
        terms=[
            LawTerm("g + Cq̇", "重力与科氏力补偿。补不准手一松就掉",
                    key="h_norm", unit="N·m", digits=2),
            LawTerm("η", "摩擦补偿比例。⭐ 故意不补满，补过头会自己爬行",
                    key="ratio_now", unit="", digits=2),
            LawTerm("D", "小阻尼，抑制拖动时的振荡", key="damp_now",
                    unit="N·m·s", digits=2),
            LawTerm("‖τ‖", "电机实际发出的力矩", key="tau", unit="N·m", digits=2),
            LawTerm("F_人", "人手要出多大力才拖得动", key="drag_force",
                    unit="N", digits=2),
        ],
        blocks=[
            Block("人手施加的力", "在 MuJoCo 里用 Ctrl+右键拖拽产生", "input"),
            Block("g(q)+C(q,q̇)q̇", "把自重和科氏力全部抵消掉", "block"),
            Block("摩擦补偿 η·f_c·sign(q̇)", "抵消关节静摩擦，决定拖不拖得动",
                  "block"),
            Block("−D·q̇", "小阻尼，防止拖动时晃", "block"),
            Block("机械臂", "对外表现得像「没有重量」", "plant"),
        ],
        note="拖动示教是**零力控制**：目标不是让机械臂去哪，而是让它对人手"
             "「不设防」。它和阻抗控制的区别是——阻抗控制有一个期望位置，"
             "拖动示教**没有期望位置**，你把它放哪它就在哪。",
    )
    params = [
        Param("comp_ratio", "摩擦补偿比例 η", 0.0, 1.3, 0.8, symbol="η",
              group="控制器",
              help="补偿多少比例的关节摩擦。",
              effect="<b>0 → 拖起来很沉</b>（要克服全部静摩擦）；"
                     "<b>调到 1.1 以上 → 机械臂开始自己爬行</b>，"
                     "松手也不停。这就是「不能补满」的原因。"),
        Param("damping", "阻尼 D", 0.0, 8.0, 0.5, symbol="D", unit="N·m·s",
              group="控制器",
              effect="调大 → 拖起来像在糖浆里，但不会晃；"
                     "调到 0 → 轻快但松手后会来回飘。"),
        Param("fc", "关节静摩擦真值 f_c", 0.0, 6.0, 2.0, unit="N·m",
              restart=True, group="被控对象",
              help="模型里加进去的库仑摩擦，就是要被补偿掉的那个东西。",
              effect="调大 → 不补偿时更拖不动；补偿后感受差别更明显。"),
        Param("fv", "关节粘滞摩擦 f_v", 0.0, 3.0, 0.5, unit="N·m·s",
              restart=True, group="被控对象",
              effect="与速度成正比的阻力。拖得越快越明显。"),
        Param("grav_err", "重力补偿误差", -30.0, 30.0, 0.0, unit="%",
              group="被控对象",
              help="故意让控制器用错的连杆质量。",
              effect="<b>本场景最能说明问题的一个滑块。</b>"
                     "调到 ±20% → 松手后机械臂会<b>自己慢慢往下沉或往上飘</b>。"
                     "真机上重力补偿永远补不准，这就是拖动示教最主要的难点。"),
        Param("passive", "被动性保护", default="on", choices=["on", "off"],
              group="控制器",
              help="⭐ 自动把速度死区撑大到 a·atanh(η)，"
                   "保证补偿量在任何速度下都不超过真实摩擦。",
              effect="<b>关掉后看「被动性余量」会变成正数</b>——"
                     "那就是在往系统里注入能量。默认 η=0.8、死区 1e-3 时，"
                     "余量是 +6.75e-4 W，<b>并不被动</b>，"
                     "尽管 η &lt; 1。这正是「η&lt;1 就安全」这句话错在哪。"),
        Param("assist", "自动施力演示", default="on", choices=["on", "off"],
              group="演示",
              help="没有真人的手，用一个周期性外力代替。",
              effect="on 时会有一只「看不见的手」周期性推拽末端，"
                     "方便你在不动鼠标的情况下观察。"
                     "关掉后请用 <b>Ctrl+右键拖拽</b>自己上手。"),
        Param("assist_f", "演示外力大小", 0.0, 60.0, 25.0, unit="N",
              group="演示",
              effect="只在「自动施力演示」打开时有效。"),
    ]
    readouts = [
        Readout("drag_force", "拖动所需人力", "N", 2,
                help="末端要多大的力才能维持当前运动。**越小越好拖**。"),
        Readout("drift", "松手后自己走了多远", "mm", 1,
                help="⭐ 零力控制做得好不好的直接判据：松手后应当**停住**。"),
        Readout("h_norm", "重力+科氏力补偿 ‖h‖", "N·m", 2),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
        Readout("speed", "末端速度", "m/s", 3),
        Readout("crawl", "爬行指标", "mm/s", 2,
                help="无外力时的自发运动速度。补偿过头时它会明显不为零。"),
        Readout("ratio_now", "摩擦补偿比例 η", "", 2),
        Readout("eta_max", "被动性允许的最大 η", "", 4, source="theory",
                help="η_max = tanh(死区/a)。⚠️ 默认死区 1e-3、真实摩擦"
                     "tanh 尺度 2e-3 时只有 0.4621——**远小于 1**。"
                     "「η<1 就安全」是拿 sign 模型推的，被控对象是 tanh。"),
        Readout("pow_margin", "被动性余量", "mW", 4, source="theory",
                help="⭐ 全速度范围扫描出的**最坏净功率**（单关节）。"
                     "必须 ≤ 0；> 0 表示补偿在顺着运动方向推，"
                     "系统从耗散变成产能。"),
        Readout("db_now", "实际速度死区", "mrad/s", 3,
                help="打开「被动性保护」后会被自动撑到 a·atanh(η)。"),
        Readout("damp_now", "阻尼", "N·m·s", 2),
        Readout("n_wp", "已记录示教点", "", 0,
                help="拖动过程中记录的关节位形，之后可以回放。"),
    ]
    traces = [
        Trace("drag_force", "拖动所需人力", "#4da3ff", unit="N"),
        Trace("speed", "末端速度", "#ffd166", axis="spd", unit="m/s"),
    ]
    bars = [_torque_bars()]

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self._dof = self.robot.qvel_idx
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.teach = GravityCompensationTeaching(
            self.robot,
            friction_coulomb=np.full(N_ARM, self.get("fc")),
            friction_viscous=np.full(N_ARM, self.get("fv")),
            comp_ratio=self.get("comp_ratio"),
            damping=self.get("damping"))
        self._apply_deadband()
        self._tau = np.zeros(N_ARM)
        self._f_now = np.zeros(3)
        self._k = 0
        self._p = self.robot.fk(Q_HOME)[0].copy()
        self._p_release = self._p.copy()
        self._crawl = 0.0
        self.trail = TrailBuffer(capacity=320, min_step=0.004)
        self._tel = dict(drag_force=0.0, drift=0.0, h_norm=0.0, speed=0.0,
                         crawl=0.0, n_wp=0)
        self.note = ("在 MuJoCo 窗口里<b>双击末端选中，再按住 Ctrl 用右键拖</b>，"
                     "就是在用手拖它。也可以把「自动施力演示」打开让它自己动")

    def on_change(self, key):
        if not hasattr(self, "teach"):
            return
        if key == "comp_ratio":
            self.teach.comp_ratio = self.get("comp_ratio")
        elif key == "damping":
            self.teach.damping = self.get("damping")
        elif key == "fc":
            # ⚠️ 构造参数叫 friction_coulomb，**属性却叫 fc**。
            # 写成 self.teach.friction_coulomb = ... 不会报错，
            # 只会悄悄新建一个没人读的属性——静默失效。
            self.teach.fc = np.full(N_ARM, self.get("fc"))
        elif key == "fv":
            self.teach.fv = np.full(N_ARM, self.get("fv"))
        # ⚠️ 死区的紧界同时依赖 η、f_c、f_v、D **四个**量。
        # 旧代码只在 η 变时重算，于是「把阻尼调大来救回 η>1」这条
        # 真实存在的路径在界面上根本走不通——调了没反应。
        if key in ("comp_ratio", "passive", "damping", "fc", "fv"):
            self._apply_deadband()

    # ------------------------------------------------------------------
    # P1-05：让「补偿绝不超过真实摩擦」成为**结构性保证**，而不是口头承诺
    # ------------------------------------------------------------------
    def _apply_deadband(self):
        """按被动性判据设置速度死区。

        判据是死区外处处 ``η ≤ g(u)``，其中
        ``g(u) = [f_c·tanh(u/a) + (f_v+D)·u] / (f_c + f_v·u)``。
        最小死区 = 违规区间的上确界。

        ⚠️ **这里曾经写错过。** 旧版本只写 ``η ≤ tanh(u/a)``，把粘滞
        摩擦 ``f_v`` 和控制器阻尼 ``D`` 这两份白送的耗散漏掉了，于是得出
        「η ≥ 1 时任何死区都救不了」。**这句话只在 D = 0 时成立。**
        真正的天花板是 ``1 + D/f_v``：默认 ``f_v=0.5, D=0.5`` 时是 **2.0**，
        所以 ``η=1.2`` 是有救的——代价是死区要放到 1.0 rad/s。

        ⭐ 于是这里的策略不是"有没有解"，而是**解贵不贵**：
        死区超过 :data:`DEADBAND_CAP`（0.1 rad/s ≈ 5.7 °/s）就等于
        "正常示教速度下完全不补偿"，这个功能也就废了。
        这种情况下**不偷偷放大死区**，而是保持 base 并让
        「被动性余量」如实报正数——把代价摆到台面上，由人来决定。
        """
        base = 1e-3
        if self.choice("passive") == "off":
            self.teach.vel_deadband = base
            return
        need = passivity_deadband(
            self.get("comp_ratio"), self.TANH_SCALE,
            fc=self.get("fc"), fv=self.get("fv"), damping=self.get("damping"))
        if not np.isfinite(need) or need > self.DEADBAND_CAP:
            self.teach.vel_deadband = base
            return
        self.teach.vel_deadband = max(base, need)

    def _passivity_stats(self):
        """返回 (最坏净功率 mW, 允许的最大 η, 当前死区 mrad/s)。

        ⚠️ 允许上限用的是**含 f_v、D 的紧界**。默认参数下它是
        0.4625015，不是只看 tanh 得到的 0.4621172——后者偏保守，
        方向安全但当精确值引用就错了。
        """
        db = self.teach.vel_deadband
        worst, _ = friction_power_margin(
            self.get("fc"), self.get("fv"), self.get("comp_ratio"),
            self.get("damping"), db, self.TANH_SCALE)
        emax = max_passive_ratio(db, self.TANH_SCALE, fc=self.get("fc"),
                                 fv=self.get("fv"), damping=self.get("damping"))
        return worst * 1e3, emax, db * 1e3

    def _true_friction(self, v: np.ndarray) -> np.ndarray:
        """被控对象真实的关节摩擦。控制器要补的就是它。

        写在场景里而不是塞进 MuJoCo 模型，是为了让「真实摩擦」与
        「控制器以为的摩擦」能分开调——这正是本场景要演示的失配。
        """
        fc, fv = self.get("fc"), self.get("fv")
        return -(fc * np.tanh(v / self.TANH_SCALE) + fv * v)

    def _assist_force(self, t) -> np.ndarray:
        """替代真人手的周期性外力。"""
        if self.choice("assist") == "off":
            return np.zeros(3)
        amp = self.get("assist_f")
        phase = (t % 6.0) / 6.0
        if phase < 0.45:                       # 前 2.7 秒推
            s = np.sin(np.pi * phase / 0.45)
            return np.array([0.0, amp * s, 0.35 * amp * s])
        return np.zeros(3)                     # 后 3.3 秒松手，看它会不会自己走

    def control(self, t, q, v):
        tau = self.teach.compute(q, v)
        # 重力补偿误差：让控制器用**错的**重力项
        err = self.get("grav_err") / 100.0
        if abs(err) > 1e-9:
            tau = tau + err * self.robot.gravity(q)
        # 真实摩擦作用在被控对象上（控制器补的是它的估计值）
        tau = tau + self._true_friction(v)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)

        self._f_now = self._assist_force(t)
        if np.linalg.norm(self._f_now) > 1e-9:
            self.data.xfrc_applied[self.robot.ee_body_id, 0:3] = self._f_now
            self._p_release = self.robot.fk(q)[0].copy()
        else:
            self.data.xfrc_applied[self.robot.ee_body_id, :] = 0.0

        p = self.robot.fk(q)[0]
        self._p = p.copy()
        self.trail.push(p)
        J = self.robot.jacobian(q)[:3]
        speed = float(np.linalg.norm(J @ v))
        # 爬行指标：**没有外力**的时候末端还在动多快
        if np.linalg.norm(self._f_now) < 1e-9:
            self._crawl = 0.9 * self._crawl + 0.1 * speed * 1e3

        # 拖动所需人力：末端保持当前运动需要的等效力。
        # 用关节残差经 (Jᵀ)⁺ 折回末端，与施加的外力是两条独立通道。
        resid = self.robot.bias(q, v) - self.teach.compute(q, v)
        Jt_pinv = np.linalg.pinv(J.T)
        drag = float(np.linalg.norm(Jt_pinv @ resid))

        if self._k % 25 == 0 and speed > 5e-3:
            self.teach.record(q)

        pw, emax, dbn = self._passivity_stats()
        self._tel = dict(
            drag_force=drag,
            drift=float(np.linalg.norm(p - self._p_release)) * 1e3,
            h_norm=float(np.linalg.norm(self.robot.bias(q, v))),
            speed=speed, crawl=self._crawl,
            n_wp=len(self.teach.recorded),
            ratio_now=self.get("comp_ratio"), damp_now=self.get("damping"),
            pow_margin=pw, eta_max=emax, db_now=dbn)
        self._k += 1
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        out = dict(self._tel)
        out.update(tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
                   tau_limit=lim,
                   tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)
        return out

    def overlays(self):
        items = [Sphere(self._p, radius=R_ACTUAL_MARK, color=C_TCP, alpha=0.95),
                 Trail(self.trail.points, color=C_ERROR, width=0.004)]
        if np.linalg.norm(self._f_now) > 1e-9:
            root = self._p + np.array([0.0, 0.0, 0.06])
            items.append(Arrow(root, root + self._f_now * FORCE_SCALE,
                               color=C_FORCE_TRUE, width=0.014))
        else:
            # 松手位置画一个绿球：末端应当停在它附近，走远了就是补偿有问题
            items.append(Sphere(self._p_release, radius=R_TARGET_MARK,
                                color=C_TARGET, alpha=0.4))
        return items


# ================================================= 10 轨迹精度补偿（ILC）


class IlcScene(Scene):
    """轨迹精度补偿：把「每遍都一样」的误差一遍一遍学掉。"""

    name = "ilc"
    steps = STEPS_BY_SCENE["ilc"]
    title = "轨迹精度补偿：迭代学习控制"
    camera = dict(azimuth=138.0, elevation=-14.0, distance=1.40,
                  lookat=(0.38, 0.0, 0.45))
    intro = (
        "机械臂反复走同一条轨迹时，偏差<b>每一遍几乎都一样</b>。"
        "这种「重复性误差」不需要理解成因——"
        "<b>这一遍哪里偏了，下一遍就在那个时刻提前补上</b>。"
    )
    law = ControlLaw(
        formula="u_{k+1}(t) = Q · [ u_k(t) + L · e_k(t + Δ) ]\n"
                "收敛条件： ‖ Q·(I − L·P) ‖ < 1\n"
                "柔度前馈： Δq = τ_load / k_joint",
        terms=[
            LawTerm("L", "学习增益。⭐ 不是越大越好，L·P>2 会发散",
                    key="gain_now", unit="", digits=2),
            LawTerm("Δ", "时间超前量。系统有滞后，补偿要提前送",
                    key="lead_now", unit="步", digits=0),
            LawTerm("Q", "低通滤波系数。⭐ 用「不学高频」换取稳定",
                    key="q_now", unit="", digits=2),
            LawTerm("e_k", "本遍跟踪误差 RMS", key="rms", unit="mrad", digits=3),
            LawTerm("e_k/e_0", "相对第一遍的比值，越小学得越好",
                    key="ratio", unit="", digits=3),
        ],
        blocks=[
            Block("参考轨迹 r(t)", "每遍完全相同", "input"),
            Block("+ 前馈修正 u_k(t)", "上几遍学出来的", "block"),
            Block("机械臂 + 关节柔度", "产生可重复的偏差", "plant"),
            Block("记录整遍误差 e_k(t)", "", "feedback"),
            Block("学习律 u_{k+1}=Q[u_k+L·e_k(t+Δ)]", "逐遍离线更新", "block"),
        ],
        note="⭐ ILC 与自适应控制（03 篇）不是一回事：自适应学的是**物理参数**，"
             "换条轨迹仍然有效；ILC 学的是**某条轨迹上的误差信号**，"
             "换条轨迹要从头再学。它的前提是「反复走同一条轨迹」。",
    )
    params = [
        Param("gain", "学习增益 L", 0.0, 2.6, 0.6, symbol="L", group="ILC",
              help="每遍把误差的多少比例学进前馈。",
              effect="<b>调到 0 → 完全不学，误差一直不降。</b>"
                     "调到 2.2 以上 → <b>发散</b>（收敛条件 |1−L·P|<1 被破坏）。"),
        Param("q_alpha", "Q 滤波系数", 0.02, 1.0, 0.15, symbol="Q", group="ILC",
              help="低通强度，1 表示不滤。",
              effect="调小 → 更稳但<b>高频误差学不掉</b>，稳态残差卡住；"
                     "调到 1（不滤）→ 学得更彻底但容易被高频噪声带飞。"),
        Param("lead", "时间超前 Δ", 0, 24, 6, step=1, unit="步", group="ILC",
              help="补偿比误差提前多少步送出去。",
              effect="<b>调到 0 → 收敛明显变慢甚至不收敛</b>："
                     "系统有滞后，不提前送就与成因错位。"),
        Param("stiffness", "关节刚度", 3e3, 1e5, 1.5e4, unit="N·m/rad",
              restart=True, group="被控对象",
              help="减速器与连杆的柔度。工业臂典型 1e4–1e6。",
              effect="<b>调小 → 关节更软，负载下弯得更多，初始误差更大</b>。"
                     "这就是 ILC 要学掉的主要来源之一。"),
        Param("geom_err", "几何误差", 0.0, 3.0, 1.0, unit="mrad",
              restart=True, group="被控对象",
              help="关节零位与名义值的偏差（装配公差）。",
              effect="常值偏差，<b>ILC 一两遍就能学掉</b>——"
                     "它是最典型的重复性误差。"),
        Param("comp", "柔度前馈补偿", default="off", choices=["off", "on"],
              group="被控对象",
              help="用 Δq = τ/k 直接把弯曲量算出来补掉。",
              effect="<b>打开后第一遍的误差就明显变小</b>——"
                     "这是「能建模就别用学的」。ILC 负责剩下建不了模的部分。"),
        Param("noise", "不可重复噪声", 0.0, 2.0, 0.15, unit="mrad",
              group="被控对象",
              help="每遍都不一样的随机扰动。",
              effect="<b>ILC 学不掉它</b>——学的是重复性误差。"
                     "调大后收敛曲线会停在一个明显的地板上。"),
        Param("freq", "轨迹频率", 0.2, 1.6, 0.6, unit="Hz", restart=True,
              group="参考轨迹",
              effect="频率越高，柔度带来的误差越大，可学的东西也越多。"),
    ]
    readouts = [
        Readout("trial", "第几遍", "", 0),
        Readout("rms", "本遍误差 RMS", "mrad", 3),
        Readout("rms0", "第一遍误差 RMS", "mrad", 3),
        Readout("ratio", "e_k / e_0", "", 4,
                help="⭐ 学得好不好的直接判据。越小越好。"),
        Readout("theory", "理论收缩因子 |1−L·P|", "", 3, theory=True,
                compare_with="ratio_step",
                help="从更新律解析推出，不读实现中间量。"
                     "每遍**与残差地板的距离**应当大致乘上这个数——"
                     "它管收敛快慢，<b>不</b>管最终停在哪。"
                     "终点由 (I−Q)(r−d) 单独决定。"),
        Readout("ratio_step", "实测逐遍误差比", "", 3),
        Readout("floor", "不可重复噪声地板", "mrad", 3, theory=True,
                help="ILC 学不掉的部分，由噪声强度独立算出。"),
        Readout("deflect", "可建模的柔度形变", "mrad", 3,
                help="bias(q,v)/k——补偿器**知道**的那部分弯曲量。刚度越小越大。"),
        Readout("defl_plant", "真实发生的形变", "mrad", 3,
                help="τ/k——关节实际被压弯多少。τ 是**裁剪后**的力矩，"
                     "所以这个读数永远 ≤ 力矩上限/刚度。"),
        Readout("defl_extra", "建不了模的形变（惯性项）", "mrad", 3,
                help="真实传递力矩 τ/k 减去上面那一项，主要是 M(q)q̈/k。"
                     "⭐ <b>把轨迹频率从 0.2 拉到 1.6 Hz，它会超过上面那一项</b>——"
                     "这就是柔度前馈的天花板，剩下的只能交给 ILC 学。"),
        Readout("gain_now", "学习增益", "", 2),
        Readout("q_now", "Q 滤波系数", "", 2),
        Readout("lead_now", "时间超前", "步", 0),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
    ]
    traces = [
        Trace("rms", "本遍误差 RMS", "#4da3ff", unit="mrad"),
        Trace("floor", "噪声地板", "#ff8a65", unit="mrad", dashed=True),
    ]
    bars = [_torque_bars()]

    TRIAL_T = 4.0

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        self._mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # ⭐ 一遍的时长必须是轨迹周期的**整数倍**。
        # 否则每遍结束时参考正停在半途，下一遍又从起点开始 —— 参考本身
        # 在边界上不连续，每遍开头都有一个大的瞬态。
        # 而 ILC 的前提是「每一遍完全可重复」，边界不连续就破坏了这个前提：
        # 实测那样跑，第 0 遍 RMS 2.38 mrad、第 1 遍起跳到 13.18 mrad，
        # 而且 L=0（完全不学）时也一样 —— 说明和学习无关，是参考自己不连续。
        cycles = max(1, int(round(self.TRIAL_T * self.get("freq"))))
        self.trial_T = cycles / self.get("freq")
        self.n_steps = int(round(self.trial_T / self.dt))
        self.ilc = IterativeLearningController(
            self.n_steps, N_ARM, gain=self.get("gain"),
            q_alpha=self.get("q_alpha"), lead=int(self.get("lead")))
        self.compliance = JointComplianceModel(self.get("stiffness"), N_ARM)
        # 几何误差：固定的关节零位偏差，每遍完全相同 ⇒ 典型的重复性误差
        rng = np.random.default_rng(4242)
        self.geom = rng.normal(0.0, 1.0, N_ARM)
        self.geom /= max(float(np.linalg.norm(self.geom)), 1e-9)
        self.rng = np.random.default_rng(20260827)

        self._err_buf = np.zeros((self.n_steps, N_ARM))
        self._k = 0
        self._tau = np.zeros(N_ARM)
        self.trail = TrailBuffer(capacity=320, min_step=0.004)
        self._p = self.robot.fk(Q_HOME)[0].copy()
        self._p_des = self._p.copy()
        self._rms0 = float("nan")
        self._rms = float("nan")
        self._prev = float("nan")
        self._tel = dict(trial=0, rms=0.0, rms0=0.0, ratio=1.0,
                         ratio_step=1.0, theory=0.0, floor=0.0,
                         deflect=0.0, defl_extra=0.0, defl_plant=0.0)
        self.note = ("每 4 秒是一遍。看「本遍误差 RMS」<b>一遍一遍往下掉</b>，"
                     "掉到「噪声地板」就停住——那是学不掉的部分")

    def on_change(self, key):
        if not hasattr(self, "ilc"):
            return
        if key == "gain":
            self.ilc.gain = self.get("gain")
        elif key == "q_alpha":
            self.ilc.q_alpha = self.get("q_alpha")
        elif key == "lead":
            self.ilc.lead = int(self.get("lead"))

    def _ref(self, t):
        w = 2.0 * np.pi * self.get("freq")
        a = 0.30 * self._mask
        return (Q_HOME + a * np.sin(w * t), a * w * np.cos(w * t),
                -a * w * w * np.sin(w * t))

    def control(self, t, q, v):
        i = self._k % self.n_steps
        t_in = i * self.dt
        qd, vd, ad = self._ref(t_in)

        # 前馈修正叠加到参考上（这就是 ILC 学出来的东西）
        qd_cmd = qd + self.ilc.u[i]

        # 柔度前馈：用 τ/k 把弯曲量提前补掉（能建模的部分别留给学习）
        #
        # ⚠️ 这里**只能**用重力+科氏项 bias(q,v)。补偿器是在**下指令之前**
        # 算这一步的，那时候还不知道真实加速度是多少，惯性项 M(q)q̈ 拿不到。
        # 这不是偷懒，是真实控制器的处境。
        tau_nom = self.reg.bias(q, v)
        defl_comp = self.compliance.deflection(tau_nom)
        if self.choice("comp") == "on":
            qd_cmd = qd_cmd + defl_comp

        aq = ad + 40.0 * (vd - v) + 400.0 * (qd_cmd - q)
        tau = self.reg.mass_matrix(q) @ aq + tau_nom
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)

        # ⭐⭐ 被控对象侧的形变用**真正传过关节的力矩**算，
        # 也就是最终施加的 τ（含惯性项 M(q)q̈、PD 修正、以及饱和裁剪）。
        # 第五轮审核 P1-09 指出：旧代码两端复用同一个 `deflect`，
        # 于是**把两处符号一起翻掉，误差分毫不变，8/8 测试全绿**——
        # 测试只抓得住"一端错"，抓不住"两端同错"。
        # 现在补偿端知道的（bias/k）和被控对象真实发生的（τ/k）
        # 是**两个不同的量**，差的正是建不了模的惯性形变。
        defl_plant = self.compliance.deflection(self._tau)
        self._defl_extra = float(np.linalg.norm(defl_plant - defl_comp)) * 1e3
        self._defl_plant = float(np.linalg.norm(defl_plant)) * 1e3

        # 被控对象侧的「真实」偏差：柔度形变 + 几何误差 + 不可重复噪声。
        # 前两项每遍完全一样（ILC 能学掉），第三项每遍都不同（学不掉）。
        #
        # ⭐ 符号约定（第四轮审核 P1-09 修正）：
        #   柔性关节的物理模型是 k(θ_motor − θ_link) = τ_load，
        #   所以 **连杆角 = 电机角 − τ/k**，即 q_eff = q − deflect。
        #   补偿则是 qd_cmd = qd + deflect（多转一点把弯掉的补回来），
        #   两者相消 ⇒ q_eff = qd。这与 ``ComplianceModel.compensate``
        #   的约定「actual = command − deflection」是同一套。
        #   ⚠️ 旧代码写的是 q_eff = q + deflect（符号反了），
        #   后果是**打开补偿反而误差翻倍**（实测 2.551 → 2.875 mrad）。
        geom = self.geom * self.get("geom_err") * 1e-3
        noise = self.rng.normal(0.0, self.get("noise") * 1e-3, N_ARM)
        q_eff = q - defl_plant + geom + noise

        self._err_buf[i] = qd - q_eff
        self._p = self.robot.fk(q)[0].copy()
        self._p_des = self.robot.fk(qd)[0].copy()
        self.trail.push(self._p)

        if i == self.n_steps - 1:                       # 一遍跑完，学一次
            self._prev = self._rms
            self._rms = self.ilc.update(self._err_buf) * 1e3
            if not np.isfinite(self._rms0):
                self._rms0 = self._rms
            self.trail.reset()

        self._k += 1
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        # 理论收缩因子。这里的「被控对象增益」就是前馈直通增益 1
        theory = IterativeLearningController.contraction_factor(
            self.get("gain"), 1.0)
        step = (self._rms / self._prev
                if np.isfinite(self._prev) and self._prev > 0 else float("nan"))
        # 噪声地板：ILC 学不掉的部分，由噪声强度独立算出（不读 ILC 内部量）
        floor = self.get("noise")
        deflect = float(np.linalg.norm(
            self.compliance.deflection(self.reg.bias(q, v)))) * 1e3
        return dict(
            trial=self.ilc.trial,
            rms=self._rms if np.isfinite(self._rms) else 0.0,
            rms0=self._rms0 if np.isfinite(self._rms0) else 0.0,
            ratio=(self._rms / self._rms0
                   if np.isfinite(self._rms0) and self._rms0 > 0 else 1.0),
            ratio_step=step if np.isfinite(step) else 1.0,
            theory=theory, floor=floor, deflect=deflect,
            defl_extra=getattr(self, "_defl_extra", 0.0),
            defl_plant=getattr(self, "_defl_plant", 0.0),
            gain_now=self.get("gain"), q_now=self.get("q_alpha"),
            lead_now=self.get("lead"),
            tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
            tau_limit=lim,
            tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)

    def overlays(self):
        return _pair_markers(self._p_des, self._p) + [
            Trail(self.trail.points, color=C_ERROR, width=0.0045)]


# ================================================= 11 动态避障（在线重规划）


class ReplanScene(Scene):
    """动态避障：障碍物在动，执行途中监测并重规划。"""

    #: 绕行偏置「慢慢收回去」的**缓入时长**（秒）。
    #: 衰减率在这段时间内按五次多项式从 0 升到满值，
    #: 保证回落开始那一刻指令速度仍然连续（见 `control` 里的长注释）。
    DECAY_EASE_S = 0.15

    name = "replan"
    steps = STEPS_BY_SCENE["replan"]
    title = "动态避障：执行途中在线重规划"
    camera = dict(azimuth=132.0, elevation=-20.0, distance=1.95,
                  lookat=(0.40, 0.0, 0.40))
    intro = (
        "06 篇的规划是<b>一次性</b>的：算完就照着走，中途障碍物动了也不管。"
        "真实场景里障碍物会动——必须<b>边走边看，该改道就改道</b>。"
    )
    law = ControlLaw(
        formula="前瞻：  gap(t) = min_{s∈[t, t+H]} ‖p_ee(s) − p_obs(s)‖ − r_obs\n"
                "触发：  gap < clearance  连续 N 次  且不在冷却期  且未超预算\n"
                "起点：  t_start = t_now + t_plan   ← ⭐ 不是当前时刻",
        terms=[
            LawTerm("gap", "前瞻窗口内的最小间隙", key="gap", unit="mm", digits=1),
            LawTerm("H", "前瞻窗口长度", key="hz_now", unit="s", digits=2),
            LawTerm("N", "迟滞计数，连续这么多次才触发",
                    key="streak_need", unit="次", digits=0),
            LawTerm("t_plan", "规划耗时估计。⭐ 规划起点要按它前移",
                    key="plan_ms", unit="ms", digits=0),
            LawTerm("n_replan", "已重规划次数", key="n_replan", unit="", digits=0),
        ],
        blocks=[
            Block("移动障碍物", "位置随时间变化", "input"),
            Block("前瞻碰撞检测", "⭐ 机器人和障碍物都按各自时刻取值", "block"),
            Block("迟滞 + 冷却 + 预算", "防误报、防抖动、防死循环", "block"),
            Block("从 t_now+t_plan 起重规划", "补偿规划延迟", "block"),
            Block("切换到新轨迹", "", "output"),
        ],
        note="⭐ 三个最容易做错的地方：① 用当前障碍物位置去比未来轨迹（漏检）；"
             "② 从当前位置开始规划（算完就过时了，切换时状态跳变）；"
             "③ 没有预算上限（陷入「规划-失败-再规划」死循环）。",
    )
    params = [
        Param("obs_speed", "障碍物速度", 0.0, 1.2, 0.45, unit="m/s",
              group="障碍物",
              effect="<b>调大 → 更容易挡住路，重规划更频繁。</b>"
                     "调到 0 就退化成静态规划。"),
        Param("obs_radius", "障碍物半径", 0.04, 0.20, 0.10, unit="m",
              group="障碍物",
              effect="越大越难绕。太大时会看到「已放弃」——超出重规划预算。"),
        Param("horizon", "前瞻窗口 H", 0.2, 3.0, 1.2, unit="s", group="监测",
              help="只检查未来这么久的轨迹。",
              effect="<b>调小 → 发现得太晚，可能来不及绕开</b>；"
                     "调大 → 频繁被远处还会再动的障碍物触发，白白重规划。"),
        Param("clearance", "安全间隙", 0.01, 0.25, 0.06, unit="m", group="监测",
              effect="间隙低于它就算有威胁。调大 → 更早触发、更保守。"),
        Param("det_noise", "检测噪声", 0.0, 0.08, 0.02, unit="m", group="监测",
              help="障碍物位置观测的噪声标准差。真实感知一定有。",
              effect="<b>迟滞机制存在的理由。</b>调到 0 时检测完全平滑，"
                     "迟滞看不出作用；调大后单次检测开始抖，"
                     "这时把「迟滞次数」调到 1 就会看到<b>误报</b>。"),
        Param("trigger_count", "迟滞次数 N", 1, 12, 3, step=1, unit="次",
              group="监测",
              help="连续这么多次检测到威胁才真触发。",
              effect="<b>调到 1 → 开始误报</b>：单次检测受采样相位影响很大，"
                     "机器人会莫名其妙地反复改道。"),
        Param("cooldown", "冷却时间", 0.0, 3.0, 0.8, unit="s", group="监测",
              effect="<b>调到 0 → 可能连续重规划，动作发抖。</b>"),
        Param("plan_ms", "规划耗时估计", 0.0, 400.0, 120.0, unit="ms",
              group="监测",
              help="用来决定从哪个未来时刻起规划。",
              effect="<b>调到 0 = 从当前位置规划</b>，切换瞬间会有状态跳变；"
                     "看「切换速度跳变」那一栏。"),
        Param("max_replans", "重规划预算", 1, 40, 25, step=1, unit="次",
              group="监测",
              effect="超了就<b>放弃并限减速停车</b>。真机上必须有这条，"
                     "否则会陷入死循环。"),
        Param("stop_s", "放弃后停车时间", 0.05, 2.0, 0.4, unit="s",
              group="安全",
              help="超预算后把轨迹时钟从 1× 线性降到 0× 所用的时间。",
              effect="<b>调小 → 停得快但减速度大</b>（可能超力矩）；"
                     "调大 → 停得柔和但在停住前会多走一段。"
                     "看「停车后位移」这个读数。"),
        Param("splice_s", "切换过渡时间", 0.0, 0.6, 0.18, unit="s",
              group="安全",
              help="新绕行偏置用五次多项式（最小 jerk）在这段时间内接上，"
                   "保证位置和速度都连续。",
              effect="<b>调到 0 = 硬切换</b>，「切换位置跳变」立刻变成几十毫米；"
                     "调大 → 跳变归零，但让位变慢，「切换后残余间隙」会变小。"),
    ]
    readouts = [
        Readout("gap", "前瞻最小间隙", "mm", 1,
                help="⭐ 负值表示前瞻窗口内会撞上。"),
        Readout("gap_now", "当前实际间隙", "mm", 1, source="measured",
                help="此刻末端与障碍球面的真实距离，独立算出。"),
        Readout("n_replan", "已重规划次数", "", 0),
        Readout("streak", "迟滞计数", "", 0),
        Readout("streak_need", "触发所需次数", "", 0),
        Readout("status", "监测状态", "", 0,
                help="0 安全 / 1 迟滞计数中 / 2 冷却中 / 3 已触发 / 4 已放弃"),
        Readout("jump", "切换后残余间隙", "mm", 1,
                help="⭐ 新轨迹生效后一个窗口内的最小间隙。"
                     "按「生效时刻」的状态规划 ⇒ 让位量刚好够；"
                     "按「当前状态」规划（规划耗时调 0）⇒ 等生效时几何已经变了，"
                     "这个数会明显更差甚至变负。"
                     "<b>注意：这是间隙，不是跳变量</b>——真正的跳变看下面两栏。"),
        Readout("splice_dp", "切换位置跳变", "mm", 2, source="measured",
                help="⭐ 切换瞬间指令位置的真实跳变 ‖Δp_des‖。"
                     "「切换过渡时间 splice_s」大于 0 时应当接近 0；"
                     "<b>调到 0（硬切换）会直接跳几十毫米</b>——"
                     "真机上这会激起振动甚至伺服报警。"),
        Readout("splice_dv", "切换速度跳变", "mm/s", 1, source="measured",
                help="⭐ 切换瞬间指令速度的真实跳变 ‖Δṗ_des‖，由相邻两拍差分得到。"
                     "位置连续不代表速度连续，这一栏专门盯速度。"),
        Readout("stop_travel", "放弃后位移", "mm", 1, source="measured",
                help="⭐ 触发「已放弃」之后末端又走了多远。"
                     "限减速停车生效时它会收敛到一个有限值并不再增长；"
                     "旧实现里机器人<b>根本不停</b>，这个数会一直涨（实测 638 mm）。"),
        Readout("clock_rate", "轨迹时钟倍率", "×", 2,
                help="1.0 = 正常走；放弃后按「停车时间」线性降到 0.00 并保持。"),
        Readout("min_gap_ever", "全程最小间隙", "mm", 1, theory=True,
                help="用密采样独立复算，不读监测器缓存。<b>负值即真的撞了。</b>"),
        Readout("hz_now", "前瞻窗口", "s", 2, source="internal",
                help="回显左边「前瞻窗口」滑块。监测器往前看这么长时间的轨迹。"),
        Readout("plan_ms_now", "规划耗时（你设的假设值）", "ms", 0,
                source="internal",
                help="⚠️ <b>这一栏不是测出来的</b>，只是把左边「规划耗时估计」"
                     "滑块的值回显一遍，方便你确认设置生效了。"
                     "它代表「假设规划要花这么久」，用来做前瞻补偿。"
                     "<br>真正<b>测</b>出来的规划耗时在 <code>planning</code> 场景，"
                     "叫「规划耗时（实测）」——那个是 perf_counter 掐的表。"),
        Readout("tau", "力矩范数", "N·m", 2),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
    ]
    traces = [
        Trace("gap", "前瞻最小间隙", "#4da3ff", unit="mm"),
        Trace("gap_now", "当前实际间隙", "#66bb6a", unit="mm", dashed=True),
    ]
    bars = [_torque_bars()]

    A = np.array([0.35, -0.35, 0.35])
    B = np.array([0.35, 0.35, 0.35])
    CYCLE = 3.0

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.imp = CartesianImpedanceController(
            self.robot, ImpedanceGains(k_trans=900.0, k_rot=40.0, zeta=1.0))
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.sup = ReplanSupervisor(
            horizon=self.get("horizon"), clearance=self.get("clearance"),
            trigger_count=int(self.get("trigger_count")),
            cooldown=self.get("cooldown"),
            max_replans=int(self.get("max_replans")),
            plan_time=self.get("plan_ms") * 1e-3)
        self.sup.reset()
        # 静止时球停在 y=0.55，路径最远只到 y=0.35 —— **不挡路**；
        # 一动起来就沿 y 扫过整条路径。这样「障碍物速度调 0 = 退化成静态规划」
        # 这条对照才成立。
        self.obs = MovingSphere(np.array([0.35, 0.55, 0.35]),
                                self.get("obs_radius"),
                                np.array([0.0, self.get("obs_speed"), 0.0]),
                                span=0.9)
        self._detour = np.zeros(3)
        self._pending = None
        self._settle_until = -1.0
        self._post_gap = np.inf
        self._worst_post = np.inf
        self._tau = np.zeros(N_ARM)
        self._min_gap = np.inf
        self._jump = 0.0
        self._status = 0
        self._gap = 0.0
        # ---- P0-06：轨迹时钟 + 连续 splice + 诚实的跳变统计 ----
        #: 轨迹时钟。正常时与真实时间同步；超预算后倍率线性降到 0，
        #: 于是指令位置沿原路径**平滑减速**并停住（位置、速度都连续）。
        self._clock = 0.0
        self._clock_rate = 1.0
        #: splice：从 _splice_from 用五次多项式过渡到 _splice_to
        self._splice_from = np.zeros(3)
        self._splice_to = np.zeros(3)
        self._splice_t0 = -1.0
        self._splice_T = 0.0
        self._splice_dp = 0.0        # 切换造成的指令位置跳变 ‖Δp_des‖
        self._splice_dv = 0.0        # 切换造成的指令速度跳变 ‖Δṗ_des‖
        #: ⚠️ 差分的是**绕行偏置**，不是 p_des —— 见下面统计处的长注释
        self._off_prev = None
        self._voff_prev = np.zeros(3)
        #: 偏置回落的缓入系数 0↔1，限速变化以保证指令速度连续
        self._decay_ease = 0.0
        #: splice 起点处偏置的移动速度（不为 0，见 `_detour_at`）
        self._splice_v0 = np.zeros(3)
        self._abort_p = None         # 触发放弃那一刻的末端位置
        self._stop_travel = 0.0
        self._p = self.robot.fk(Q_HOME)[0].copy()
        self._p_des = self._p.copy()
        self._v_prev = np.zeros(3)
        self._det_rng = np.random.default_rng(777)
        self.trail = TrailBuffer(capacity=340, min_step=0.004)
        self.note = ("机械臂在两点之间来回走，<b>灰色球</b>横穿它的路。"
                     "看「前瞻最小间隙」变负时怎么触发改道")

    def on_change(self, key):
        if not hasattr(self, "sup"):
            return
        m = {"horizon": "horizon", "clearance": "clearance",
             "cooldown": "cooldown"}
        if key in m:
            setattr(self.sup, m[key], self.get(key))
        elif key == "trigger_count":
            self.sup.trigger_count = int(self.get("trigger_count"))
        elif key == "max_replans":
            self.sup.max_replans = int(self.get("max_replans"))
        elif key == "plan_ms":
            self.sup.plan_time = self.get("plan_ms") * 1e-3
        elif key == "obs_radius":
            self.obs.radius = self.get("obs_radius")
        elif key == "obs_speed":
            self.obs.velocity = np.array([0.0, self.get("obs_speed"), 0.0])

    def _nominal(self, t: float) -> np.ndarray:
        """名义轨迹：A、B 两点之间来回。"""
        phase = (t % self.CYCLE) / self.CYCLE
        s = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
        return self.A + (self.B - self.A) * s

    def _detour_at(self, t_real: float) -> np.ndarray:
        """当前时刻生效的绕行偏置。切换时用**五次多项式**平滑过渡。

        用五次多项式，边界条件是
        ``p(0)=from, p'(0)=v0, p''(0)=0`` 和 ``p(1)=to, p'(1)=p''(1)=0``。

        ⚠️ **起点速度 v0 不能强制写成 0。** 切换发生时，偏置往往正在
        「慢慢回落」（见 `control` 里的衰减），速度不是 0；硬把多项式的
        起点速度设成 0，指令速度就会在切换那一刻**突降到 0** ——
        实测这一下有 **95 mm/s**，恰恰是这段代码想消掉的东西。
        （批次 H 之前看不见：跳变读数差分的是 p_des，名义运动的地板盖住了它。）

        取 ``W = v0·T``、``D = to − from − W``，系数为

            c3 = 10D + 4W,  c4 = −15D − 7W,  c5 = 6D + 3W

        `v0 = 0` 时退化成经典的 ``10τ³ − 15τ⁴ + 6τ⁵``，与旧行为一致。

        旧实现是 ``self._detour = 新值`` 一步到位，指令位置直接跳。
        实测硬切换时 ``‖Δp_des‖`` 最大 **199.8 mm**，真机上足以激起振动
        甚至触发伺服报警。
        """
        if self._splice_t0 < 0.0:
            return self._detour
        if self._splice_T <= 0.0:
            return self._splice_to
        tau = (t_real - self._splice_t0) / self._splice_T
        if tau <= 0.0:
            return self._splice_from
        if tau >= 1.0:
            return self._splice_to
        T = self._splice_T
        W = self._splice_v0 * T
        D = self._splice_to - self._splice_from - W
        c3 = 10.0 * D + 4.0 * W
        c4 = -15.0 * D - 7.0 * W
        c5 = 6.0 * D + 3.0 * W
        return (self._splice_from + W * tau
                + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5)

    def _planned(self, t_traj: float, t_real: float | None = None) -> np.ndarray:
        """当前生效的轨迹 = 名义轨迹(轨迹时钟) + 绕行偏置(真实时钟)。

        两个时钟必须分开：名义轨迹沿**轨迹时钟**推进（超预算后会被减速到停），
        而 splice 过渡是按**真实时钟**计时的。正常运行时两者同步，
        ``_planned(t) == _nominal(t) + _detour``，与旧实现完全一致。
        """
        if t_real is None:
            t_real = t_traj
        return self._nominal(t_traj) + self._detour_at(t_real)

    def _detour_for(self, t_start: float) -> np.ndarray:
        """算一个绕行偏置：**从上方让开**。

        为什么是竖直方向：障碍物沿 y 扫，而名义路径本身也沿 y 走——
        在 y 上让位没有意义（球迟早扫过来），在 x 上让位会撞到工作空间边界。
        抬高是这个几何下唯一干净的逃逸方向。

        抬多少：让路径在最近点处与球心的竖直距离达到「半径 + 安全间隙 + 余量」。

        ⚠️ 这是**简化的重规划**：真正的做法是用 RRT 重新规划整条路径（见 06 篇）。
        这里做解析让位，是为了把「监测 → 迟滞 → 延迟补偿 → 切换 → 复核」
        这条链路讲清楚，**不是替代规划器**。
        """
        need = self.obs.radius + self.get("clearance") + 0.04
        # 取前瞻窗口内最贴近的那一刻来定抬升量
        best, lift = np.inf, 0.0
        for i in range(self.sup.samples):
            t = t_start + self.sup.horizon * i / max(self.sup.samples - 1, 1)
            p = self._nominal(t)
            c = self.obs.position_at(t)
            planar = float(np.linalg.norm((p - c)[:2]))
            if planar >= need:                       # 水平方向已经够远
                continue
            # 竖直方向要抬到 sqrt(need² − planar²) 才在球外
            want = float(np.sqrt(max(need ** 2 - planar ** 2, 0.0))) - (p[2] - c[2])
            if planar < best:
                best, lift = planar, want
        return np.array([0.0, 0.0, float(np.clip(lift, 0.0, 0.35))])

    def control(self, t, q, v):
        # ---- 轨迹时钟：超预算后按「停车时间」线性降到 0（限减速停车）----
        # ⭐ P0-06 修复。旧实现里 aborted 只是**不再重规划**，
        # 指令位置照样沿名义轨迹跑——实测放弃后末端又走了 638 mm，
        # 关节速度均值 1.48 rad/s。文档写的「机器人停下来而不是乱撞」并不成立。
        # 现在把整条轨迹的**时钟倍率**降到 0：位置、速度都连续，
        # 减速度有界（≈ 名义速度 / stop_s），停住后保持。
        if self.sup.aborted:
            stop_s = max(float(self.get("stop_s")), 1e-6)
            self._clock_rate = max(0.0, self._clock_rate - self.dt / stop_s)
        self._clock += self.dt * self._clock_rate

        # 前瞻检测。⭐ 机器人和障碍物都按**各自时刻**取值
        # ⭐ 检测噪声：真实感知给出的障碍物位置是带噪的。
        # 它是迟滞机制存在的理由——没有噪声时单次检测就足够可靠，
        # 迟滞看不出任何作用（实测：噪声为 0 时 N=1..8 的结果完全一样）。
        # 前瞻时把真实时刻映射回轨迹时钟（正常运行时倍率为 1，等价于旧行为）。
        sigma = self.get("det_noise")

        def _lookahead(tt: float) -> np.ndarray:
            return self._planned(self._clock + (tt - t) * self._clock_rate, tt)

        gap, _ = self.sup.min_clearance(_lookahead, self.obs, t)
        if sigma > 0:
            gap += float(self._det_rng.normal(0.0, sigma))
        self._gap = gap
        fire, why = self.sup.should_replan(gap, t)
        self._status = (4 if self.sup.aborted else
                        3 if fire else
                        2 if "冷却" in why else
                        1 if "迟滞" in why else 0)
        if fire and self._pending is None:
            # ⭐ 规划要花时间，新轨迹**不是立刻生效**，而是 plan_time 之后才生效。
            # 所以规划时必须按「生效那一刻」的状态来算，而不是当前状态。
            # plan_ms 调到 0 就退化成「按当前状态规划」，可以直接对照。
            # （只在未放弃时才会走到这里，所以此处时钟倍率必为 1，
            #   真实时刻与轨迹时刻同步，可以直接用 replan_start_time。）
            t_apply = t + self.sup.plan_time
            self._pending = (t_apply, self._detour_for(
                self.sup.replan_start_time(t)))
            self.sup.commit(t, why, success=True)

        # 长时间安全就慢慢把抬升收回去，让机器人回到名义轨迹。
        #
        # ⚠️ 这里必须**缓入**，不能一上来就按满速衰减。
        # 旧写法是 `self._detour *= 0.999`，衰减在 splice 结束的那一拍
        # **满速接管**：偏置速度从 0 一步跳到 0.001·‖detour‖/dt
        # —— 实测 **99.9 mm/s**。也就是说「连续 splice」辛辛苦苦
        # 做到的速度连续，被紧接着的这一步又打破了。
        # （批次 H 之前这个缺陷看不见：跳变读数差分的是 p_des，
        #   名义运动的地板本来就有 ~100 mm/s 量级，正好把它盖住了。）
        #
        # 改法：用和 splice 同一条五次多项式把衰减率从 0 缓升到满值，
        # 于是衰减开始那一刻的偏置速度仍然是 0，速度保持连续。
        #
        # ⚠️ 缓入系数必须是**持久的、限速变化的状态**，
        # 不能用「本轮衰减是什么时候开始的」这种时间戳：
        # 衰减只要被打断一拍（status 抖一下），时间戳就归零，
        # 缓入从头再来 ⇒ 速度从 95 mm/s 一步掉回 0，反而制造了新的跳变。
        # 改成让系数以固定速率在 0↔1 之间**爬**，它就永远是连续的。
        want = 1.0 if (self._status == 0 and self._pending is None
                       and self._splice_t0 < 0.0) else 0.0
        rate = self.dt / self.DECAY_EASE_S
        self._decay_ease = float(np.clip(
            self._decay_ease + np.clip(want - self._decay_ease, -rate, rate),
            0.0, 1.0))
        if self._splice_t0 < 0.0:
            self._detour = self._detour * (1.0 - (1.0 - 0.999) * self._decay_ease)

        if self._pending is not None and t >= self._pending[0]:
            # ⭐ 连续 splice：不是直接赋值，而是启动一段五次多项式过渡。
            # splice_s 调到 0 就退化成旧的硬切换，可以直接对照跳变读数。
            _from = self._detour_at(t).copy()
            # ⭐ 接住「此刻偏置正在以多快的速度移动」，交给多项式做起点条件
            self._splice_v0 = (np.zeros(3) if self._off_prev is None
                               else (_from - self._off_prev) / self.dt)
            self._splice_from = _from
            self._splice_to = np.asarray(self._pending[1], float).copy()
            self._splice_t0 = t
            self._splice_T = max(float(self.get("splice_s")), 0.0)
            self._pending = None
            self._settle_until = t + self.sup.horizon
            self._post_gap = np.inf          # 开始统计「切换后残余侵入」

        # splice 走完就把结果固化回 _detour，并关掉过渡
        if self._splice_t0 >= 0.0 and t - self._splice_t0 >= self._splice_T:
            self._detour = self._splice_to.copy()
            self._splice_t0 = -1.0

        self._p_des = self._planned(self._clock, t)

        # ---- 跳变统计：只差分**绕行偏置**，不差分 p_des ----
        # 指令位置 p_des = 名义轨迹(轨迹钟) + 绕行偏置(真实钟)，
        # 而「切换造成的跳变」**整个都在偏置里**——名义轨迹是连续的。
        #
        # ⚠️ 旧实现差分的是 p_des，于是名义轨迹自己走得快的那一拍
        # 也会被算成「跳变」。后果是这个读数有一个**永远消不掉的地板**：
        # 实测把障碍物速度调成 0（全程一次重规划都没有）时，
        # 它照样报 1.466 mm / 4.6 mm/s——而「切换跳变」在没有切换时
        # 按定义必须是 0。批次 H 修的就是这个。
        #
        # 改成差分偏置以后：没切换 ⇒ 偏置恒定 ⇒ 读数恒等于 0；
        # 硬切换 ⇒ 整段偏置在一拍内走完 ⇒ 读数就是那一跳的真实幅值。
        off = self._detour_at(t)
        if self._off_prev is not None:
            doff = off - self._off_prev
            voff = doff / self.dt
            self._splice_dp = max(self._splice_dp, float(np.linalg.norm(doff)))
            self._splice_dv = max(self._splice_dv,
                                  float(np.linalg.norm(voff - self._voff_prev)))
            self._voff_prev = voff
        self._off_prev = np.asarray(off, float).copy()

        p, R = self.robot.fk(q)
        self._p = p.copy()
        self.trail.push(p)

        # 放弃之后末端还走了多远——限减速停车生效时它必须收敛
        if self.sup.aborted:
            if self._abort_p is None:
                self._abort_p = p.copy()
            self._stop_travel = float(np.linalg.norm(p - self._abort_p))

        tau = self.imp.compute(q, v, self._p_des, R)
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)

        # 独立复算的真实间隙（不读监测器缓存）
        g_now = float(np.linalg.norm(p - self.obs.position_at(t))) - self.obs.radius
        self._gap_now = g_now
        self._min_gap = min(self._min_gap, g_now)
        # ⭐ 切换生效之后的一个窗口内，间隙最小压到了多少。
        # 按「生效时刻」的状态规划 ⇒ 让位量刚好够；按「当前状态」规划 ⇒
        # 等生效时几何已经变了，让位量偏小，这个数会明显更差。
        if t < self._settle_until:
            self._post_gap = min(self._post_gap, g_now)
        elif np.isfinite(self._post_gap):
            self._worst_post = min(self._worst_post, self._post_gap)
            self._post_gap = np.inf
        return self._tau

    def telemetry(self, t, q, v):
        lim = self.robot.tau_limit
        return dict(
            gap=self._gap * 1e3, gap_now=self._gap_now * 1e3,
            n_replan=self.sup.n_replans, streak=self.sup._streak,
            streak_need=self.sup.trigger_count, status=self._status,
            jump=(self._worst_post * 1e3
                  if np.isfinite(self._worst_post) else 0.0),
            splice_dp=self._splice_dp * 1e3,
            splice_dv=self._splice_dv * 1e3,
            stop_travel=self._stop_travel * 1e3,
            clock_rate=self._clock_rate,
            min_gap_ever=(self._min_gap * 1e3
                          if np.isfinite(self._min_gap) else 0.0),
            hz_now=self.sup.horizon, plan_ms_now=self.sup.plan_time * 1e3,
            tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
            tau_limit=lim,
            tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)

    def overlays(self):
        t = self.data.time
        c = self.obs.position_at(t)
        items = [Sphere(c, radius=self.obs.radius, color="#c8ccd4", alpha=0.55)]
        # 安全间隙做成一层半透明外壳：看得见「不许进入的范围」
        items.append(Sphere(c, radius=self.obs.radius + self.sup.clearance,
                            color=C_FORCE_TRUE, alpha=0.14))
        items += _pair_markers(self._p_des, self._p)
        items.append(Trail(self.trail.points, color=C_ERROR, width=0.0045))
        # 把前瞻窗口内的规划轨迹画出来（按轨迹时钟推进，停车后自然缩成一点）
        pts = np.array([self._planned(
            self._clock + self.sup.horizon * i / 16.0 * self._clock_rate,
            t + self.sup.horizon * i / 16.0) for i in range(17)])
        items.append(Trail(pts, color=C_PATH, width=0.004, alpha=0.85))
        return items


# ================================================= 12 伺服环路与振动抑制


class ServoScene(Scene):
    """伺服环路与振动抑制：机械柔度如何钳死速度环增益。"""

    name = "servo"
    steps = STEPS_BY_SCENE["servo"]
    title = "伺服环路：谐振、陷波与输入整形"
    camera = dict(azimuth=128.0, elevation=-16.0, distance=1.55,
                  lookat=(0.35, 0.0, 0.50))
    intro = (
        "前面所有场景都默认「给关节多大力矩，连杆就受多大力矩」。"
        "真实机器人里电机和连杆之间隔着减速器，<b>受力会扭</b>。"
        "而编码器装在<b>电机侧</b>——控制器根本看不见连杆在晃。"
    )
    law = ControlLaw(
        formula="J_m θ̈_m = τ_m − τ_s,   J_l θ̈_l = τ_s,   "
                "τ_s = K_s δ + D_s δ̇,  δ = θ_m − θ_l\n"
                "谐振  ω_res  = sqrt( K_s (J_m+J_l) / (J_m J_l) )\n"
                "反谐振 ω_ares = sqrt( K_s / J_l )        "
                "⭐ 与电机惯量无关",
        terms=[
            LawTerm("K_s", "传动扭转刚度。减速器 + 连杆的柔度",
                    key="stiffness_now", unit="N·m/rad", digits=0),
            LawTerm("J_m", "电机惯量（已折算到关节侧）",
                    key="j_motor_now", unit="kg·m²", digits=3),
            LawTerm("J_l", "负载惯量。⭐ 从质量矩阵实测，不是编的",
                    key="j_load", unit="kg·m²", digits=3),
            LawTerm("δ", "弹簧扭转量。电机和连杆的角度差",
                    key="twist", unit="mrad", digits=2),
            LawTerm("f_res", "理论谐振频率。速度环增益的天花板",
                    key="f_res", unit="Hz", digits=2),
            LawTerm("f_meas", "实测振动频率。⭐ 应当与 f_res 对上",
                    key="f_meas", unit="Hz", digits=2),
        ],
        blocks=[
            Block("位置指令 θ_ref", "可先经输入整形", "input"),
            Block("位置环 P（Kpp）", "+ 速度前馈", "block"),
            Block("速度环 PI（Kvp, Kvi）", "可插陷波滤波", "block"),
            Block("电流环（一阶惯性）", "带宽 f_i", "block"),
            Block("电机 J_m ─ 弹簧 K_s ─ 连杆 J_l", "会晃的地方", "plant"),
            Block("电机侧编码器 θ_m, ω_m", "⭐ 看不见连杆", "feedback"),
        ],
        note="⭐ 反馈取的是**电机侧**角度，不是连杆角度——真实伺服的编码器"
             "就装在电机上。这正是谐振能钳死增益的根本原因："
             "控制器看不见负载在晃，只能通过弹簧间接感觉到。",
    )
    params = [
        Param("kvp", "速度环比例增益 Kvp", 1.0, 120.0, 8.0, symbol="K_{vp}",
              group="伺服增益",
              help="速度误差乘这个数变成力矩指令。",
              effect="<b>⭐ 本场景的主角。</b>往上调 → 跟随变快，"
                     "但调过头会在<b>谐振频率</b>上振起来。"
                     "看「实测振动频率」是否落在「理论谐振频率」上。"),
        Param("kvi", "速度环积分增益 Kvi", 0.0, 30.0, 4.0, symbol="K_{vi}",
              group="伺服增益",
              help="消除速度稳态误差。单位是 1/s。",
              effect="调到 0 → 稳态有残差；调太大 → 低频超调、"
                     "抗扰变差。它<b>不是</b>振动的主因。"),
        Param("kpp", "位置环比例增益 Kpp", 0.5, 30.0, 4.0, symbol="K_{pp}",
              group="伺服增益",
              help="位置误差乘这个数变成速度指令。单位 1/s。",
              effect="它应当比速度环<b>慢 3~10 倍</b>。"
                     "调过头 → 位置超调、和速度环打架。"),
        Param("f_current", "电流环带宽", 15.0, 1200.0, 800.0, unit="Hz",
              group="伺服增益",
              help="内环有多快。真实驱动器 1~2 kHz。",
              effect="<b>调到 30 Hz → 跟踪误差与振动同时变差</b>，"
                     "控制律一个字没改。这就是「内环慢了拖垮外环」。"),
        Param("f_vel", "速度反馈滤波带宽", 20.0, 600.0, 120.0, unit="Hz",
              symbol="f_{vel}", group="伺服增益",
              help="编码器只测角度，速度是「两次角度相减除以时间」差分出来的，"
                   "差分会把量化噪声放大，所以必须低通滤波。"
                   "这个滤波是速度环相位滞后的头号来源，也是「增益调不上去」的真正原因。",
              effect="<b>调低（滤得狠）→ 噪声小但相位滞后大，Kvp 稍微抬高就在谐振频率上啸叫；"
                     "调高 → 能容忍更大的 Kvp，但速度反馈变毛糙。</b>"
                     "这就是伺服调试里最经典的取舍。"),
        Param("stiffness", "传动刚度 K_s", 120.0, 6000.0, 800.0,
              unit="N·m/rad", symbol="K_s", group="机械参数",
              help="减速器 + 连杆的扭转刚度。",
              effect="<b>两个频率都按平方根变</b>（刚度 ×4 → 频率 ×2）。"
                     "调小 → 更软、更容易振、频率更低。"),
        Param("j_motor", "电机惯量 J_m", 0.01, 0.40, 0.30, unit="kg·m²",
              symbol="J_m", group="机械参数",
              help="折算到关节侧的电机转动惯量（已除减速比平方）。"
                   "默认 0.30 对应「低惯量比」工况（负载/电机 ≈ 3.7），"
                   "谐振频率 9.3 Hz 正好落在速度环带宽附近 —— 这是工业上最难调的情形。",
              effect="<b>⭐ 反谐振频率完全不变，谐振频率跟着变</b>——"
                     "这是区分两个频率最快的实验。"
                     "调小 → 谐振频率升高，速度环带宽升得更快（∝1/J_m），"
                     "谐振反而被包进带宽里压住；调大 → 谐振降到带宽附近，最容易啸叫。"),
        Param("damping", "传动阻尼 D_s", 0.0, 8.0, 0.5, unit="N·m·s/rad",
              symbol="D_s", group="机械参数",
              help="弹簧自身的内阻尼。真实减速器很小。",
              effect="调大 → 振动自己衰减得快，但这在真机上"
                     "<b>不是你能调的</b>——它由材料和润滑决定。"),
        Param("notch", "陷波滤波", default="off", choices=["off", "on"],
              group="振动抑制",
              help="在速度环里挖掉谐振频率那一段。",
              effect="<b>打开后同样的 Kvp 振动明显减小</b>，"
                     "于是 Kvp 还能往上调——这就是陷波的全部价值："
                     "<b>把增益天花板抬高</b>。"),
        Param("notch_depth", "陷波深度 ζn/ζd", 0.02, 0.9, 0.08,
              group="振动抑制",
              help="凹口留下多少。0.08 表示衰减到 8%。",
              effect="调到 0.8（几乎不衰减）→ <b>振动回来</b>。"
                     "越小削得越狠，但相位代价也越大。"),
        Param("notch_width", "陷波宽度 ζd", 0.2, 2.0, 0.7,
              group="振动抑制",
              help="凹口有多宽。越大越宽越钝。",
              effect="频率估不准时调宽一点更保险，"
                     "代价是<b>影响到更多不该动的频率</b>。"),
        Param("shaper", "输入整形", default="none",
              choices=["none", "zv", "zvd"], group="振动抑制",
              help="把指令拆成几个脉冲，让振动自己抵消。",
              effect="<b>point 模式下效果最明显</b>：到位后的残余晃动消失。"
                     "zvd 比 zv 更抗频率失配，代价是延迟翻倍。"
                     "⭐ 它<b>管不了</b>反馈回路自激的振动。"),
        Param("move", "运动模式", default="point",
              choices=["point", "sine", "hold", "probe", "probe_cl"],
              group="参考轨迹",
              help="point 点位往返；sine 连续正弦；hold 定点不动；"
                   "probe <b>开环</b>辨识（旁路伺服环，量的是纯机械谐振）；"
                   "probe_cl <b>闭环</b>辨识（噪声照打，但伺服环留在回路里）。",
              effect="<b>看输入整形效果用 point</b>（有明确的「到位后」）；"
                     "<b>量机械谐振频率必须用 probe</b>——阶跃指令自己就带一堆谐波，"
                     "会把频谱搅乱，f_meas 读数在 point 模式下不可信。"
                     "真实驱动器的「自动谐振检测」用的正是 probe 这种做法。<br>"
                     "⭐ <b>probe 与 probe_cl 的对比是本场景最重要的一个实验</b>："
                     "probe 下 f_meas 与 Kvp 无关（因为伺服环被旁路了，"
                     "这是<b>构造出来的</b>，不是发现）；"
                     "probe_cl 下 f_meas <b>会随 Kvp 移动</b>——"
                     "那才是闭环阻尼峰 ω_n√(1−2ζ²)。"
                     "「机械谐振」和「闭环峰」是两个不同的东西。"),
        Param("move_amp", "运动幅度", 0.05, 0.60, 0.30, unit="rad",
              group="参考轨迹",
              effect="越大激起的振动越明显。"),
        Param("probe_amp", "辨识噪声幅度", 0.0, 6.0, 2.0, unit="N·m",
              symbol="\\sigma_{probe}", group="参考轨迹",
              help="在「运动模式 = probe」<b>和 probe_cl</b> 两种模式下都起作用："
                   "叠加在电机转矩上的白噪声标准差。"
                   "相当于用小幅随机抖动去「敲」机械结构。"
                   "两者的区别不在注入与否，而在<b>伺服环是否还闭着</b>："
                   "probe 旁路伺服，probe_cl 留在环内。",
              effect="<b>调大 → 谐振峰更突出、f_meas 更稳；调到 0 → 没有激励，"
                     "f_meas 显示「--」（测不到）。</b>"
                     "太大则关节明显抖动、可能撞限位。"),
        Param("move_period", "运动周期", 0.8, 6.0, 2.4, unit="s",
              group="参考轨迹",
              effect="point 模式下是「走过去 + 停一会」的总时长。"
                     "太短的话上一次的振动还没停就开始下一次。"),
    ]
    readouts = [
        Readout("f_res", "理论谐振频率", "Hz", 2, theory=True,
                compare_with="f_meas",
                help="由 sqrt(K_s(J_m+J_l)/(J_m J_l)) 解析算出。"
                     "⭐ 增益调高后的振动频率应当落在这里。"),
        Readout("f_meas", "实测振动频率", "Hz", 2,
                help="对<b>弹簧形变 δ = θ_m − θ_l</b> 做 Welch 平均谱找峰值"
                     "（不是对速度：δ 的传递函数只有极点没有零点，"
                     "谱上有且只有谐振这一个峰；速度谱带反谐振凹口，argmax 会挑错）。"
                     "峰不显著时显示 0（测不出来就如实说测不到）。"),
        Readout("f_ares", "理论反谐振频率", "Hz", 2, theory=True,
                help="sqrt(K_s/J_l)。⭐ 与电机惯量无关。"),
        Readout("j_load", "负载惯量 J_l", "kg·m²", 4, source="measured",
                help="从 MuJoCo 质量矩阵的对角元现场读出，随位形变化。"),
        Readout("inertia_ratio", "惯量比 J_l/J_m", "", 1,
                help="⭐ 伺服选型头号数字。工业建议 < 5~10。"),
        Readout("vib_amp", "振动幅值", "mrad/s", 2,
                help="<b>电机侧速度 ω_m</b> 经中心在 f_res、Q=2 的二阶带通后的 RMS，"
                     "即「谐振带内的振动能量」。"
                     "⭐ 看电机侧是因为谐振模态下两侧反向摆动、幅值比 J_m/(J_m+J_l)，"
                     "负载侧摆得更小；真实伺服调试看的也是电机编码器。"
                     "⭐ 判断「振没振」就看它。"),
        Readout("err", "跟踪误差", "mrad", 2,
                help="指令角度与负载侧实际角度之差。"),
        Readout("twist", "弹簧扭转量 δ", "mrad", 2,
                help="电机角度 − 连杆角度。刚度越小越大。"),
        Readout("theta_m", "电机侧角度", "rad", 4,
                help="控制器唯一能看到的量。"),
        Readout("theta_l", "负载侧角度", "rad", 4,
                help="我们真正想控制的量，但编码器看不见。"),
        Readout("tau_m", "电机力矩 τ_m", "N·m", 2),
        Readout("tau_s", "弹簧力矩 τ_s", "N·m", 2,
                help="真正传给连杆的力矩。动起来就和 τ_m 分家。"),
        Readout("resid_theory", "理论残余振动", "", 4, theory=True,
                help="由整形器脉冲序列与当前真实频率独立算出"
                     "（不读整形器内部量）。1 = 完全不抑制，0 = 完全抵消。"),
        Readout("shaper_lag", "整形延迟", "ms", 1,
                help="输入整形的代价。zvd 是 zv 的两倍。"),
        Readout("notch_f", "陷波中心频率", "Hz", 2,
                help="自动跟随理论谐振频率。"),
        Readout("cur_lag", "电流环滞后", "ms", 2, theory=True,
                help="1/(2π f_i)。内环慢下来时它变大。"),
        Readout("tau_max_pct", "最大关节饱和度", "%", 1),
    ]
    traces = [
        Trace("vib_amp", "振动幅值", "#ff5252", unit="mrad/s"),
        Trace("err", "跟踪误差", "#4da3ff", unit="mrad"),
    ]
    bars = [_torque_bars()]

    JOINT = 3               # 演示关节 J4（肘部）：惯量适中、运动看得清
    FFT_N = 4096            # 缓冲长度 2 ms × 4096 ≈ 8.2 s
    FFT_SEG = 1024          # Welch 分段长度 ⇒ 频率分辨率约 0.49 Hz
    SUBSTEPS = 10           # 见 _servo_dt 的说明

    @property
    def _servo_dt(self) -> float:
        """伺服环的控制周期 = 仿真步长 / SUBSTEPS。

        ⭐ 为什么必须比仿真步长细
        ------------------------
        电机侧惯量 J_m 很小（0.05），速度环的离散稳定条件大致是

            K_vp · dt / J_m < 2   ⇒   K_vp < 2 J_m / dt

        用仿真步长 2 ms 直接算，上限只有 K_vp < 50。而机械谐振失稳
        （本场景的教学目标）实测要 K_vp ≈ 80 才发生 —— **数值发散会先到，
        于是永远看不到真正的谐振**，演示就成了假的。

        细分 10 倍之后上限抬到 ~500，谐振失稳落在中间，现象才是物理的。
        这也更贴近真实：工业伺服的速度环是 4~8 kHz，
        远快于机械本身的时间尺度，本来就不该和仿真步长绑在一起。

        负载侧仍由 MuJoCo 以 2 ms 步进 —— 负载动态慢，够用。
        子步内负载角度/速度保持不变（零阶保持）。
        """
        return self.dt / self.SUBSTEPS

    def build(self):
        scene = build_panda_scene([], with_visuals=True)
        self.robot = _make_robot(scene.model)
        self.gripper = ParallelJawGripper(self.robot)
        self.data = self.robot.data
        self.model = scene.model
        self.reg = _make_arm_model(DynamicsRegressor(PANDA_XML))
        self.reg.set_finger_positions(0.5 * (self.gripper_width or 0.0))
        return scene.model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot.qpos_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        j = self.JOINT
        self.joint = TwoMassJoint(j_m=self.get("j_motor"),
                                  k_s=self.get("stiffness"),
                                  d_s=self.get("damping"))
        # ⭐ 初始就把弹簧预扭到重力平衡位：τ_s(0) = τ_gravity。
        # 否则 t=0 时 θ_m=θ_l ⇒ 弹簧力矩为零 ⇒ 重力补偿力矩全部加速电机
        # （J_m 很小，一步就窜到 ~1 rad/s），画面一开场就是一个假瞬态。
        q0 = self.data.qpos[self.robot.qpos_idx][:N_ARM]
        tau_g = float(self.reg.bias(q0, np.zeros(N_ARM))[j])
        self.joint.reset(theta=Q_HOME[j] + tau_g / self.get("stiffness"),
                         omega=0.0)
        self.servo = CascadeServo(kpp=self.get("kpp"), kvp=self.get("kvp"),
                                  kvi=self.get("kvi"),
                                  f_current=self.get("f_current"),
                                  tau_max=float(self.robot.tau_limit[j]))
        self._notch_f = self._resonance_hz()
        self._rebuild_notch()
        self._rebuild_shaper()

        self._vel_buf = np.zeros(self.FFT_N)
        self._vel_i = 0
        self._vel_full = False
        self._svf_lp = 0.0              # 带通滤波器状态：低通支路
        self._svf_bp = 0.0              # 带通滤波器状态：带通支路（就是输出）
        self._hp_fc = float(np.sqrt(self._antiresonance_hz() * self._resonance_hz()))
        self._vib = 0.0
        self._f_meas = 0.0
        self._fft_countdown = 0
        # 编码器差分测速的状态：上一拍角度、滤波后速度、送进控制器的延迟值
        self._theta_m_prev = self.joint.theta_m
        self._omega_meas = 0.0
        self._theta_m_delayed = self.joint.theta_m
        self._omega_delayed = 0.0
        # 固定种子：同样的参数每次跑出同样的结果，方便对照实验
        self._rng = np.random.default_rng(20240717)
        self._tau = np.zeros(N_ARM)
        self._theta_ref = Q_HOME[j]
        self._err = 0.0
        self.trail = TrailBuffer(capacity=280, min_step=0.004)
        self._p = self.robot.fk(Q_HOME)[0].copy()
        self._p_des = self._p.copy()
        self.note = ("⭐ 主实验：把「速度环比例增益 Kvp」往上调，"
                     "看手臂开始抖，再核对<b>实测振动频率</b>"
                     "是否落在<b>理论谐振频率</b>上")

    # ------------------------------------------------------------ 参数联动

    def _j_load(self) -> float:
        """演示关节感受到的负载惯量 = 质量矩阵对角元 M[j,j]。

        ⭐ 为什么是 M[j,j]
        -----------------
        M[j,j] 的物理含义是「其余关节**刚性固定**时，单位角加速度需要多少力矩」。
        本场景正是这么做的：control() 里把其余 6 个关节**运动学硬钳位**在
        Q_HOME（见那里的注释），所以这个前提严格成立。

        对照：如果其余关节完全自由，有效惯量会变成 1/(M⁻¹)[j,j] = 0.628，
        比 M[j,j] = 1.114 小很多。中间任何一种「软锁」都落在两者之间，
        而且会额外引入邻居自己的振动模态 —— 那正是本场景一开始
        用 PD 锁时频谱一团糟的原因。

        ⚠️ 边界：这是同一个模型内部的一致性（M 由 MuJoCo 模型算出），
        不是真机标定值。真实机器人上有效惯量要靠扫频辨识实测。
        """
        q = self.data.qpos[self.robot.qpos_idx][:N_ARM]
        return float(self.reg.mass_matrix(q)[self.JOINT, self.JOINT])

    def _resonance_hz(self) -> float:
        return resonance_frequency(self.get("stiffness"), self.get("j_motor"),
                                   self._j_load()) / (2.0 * np.pi)

    def _antiresonance_hz(self) -> float:
        return antiresonance_frequency(self.get("stiffness"),
                                       self._j_load()) / (2.0 * np.pi)

    def _rebuild_notch(self):
        """陷波中心频率跟随理论谐振频率。

        真实驱动器上这一步是「自动谐振抑制」：扫频找出谐振点再设陷波。
        这里直接用解析值，省掉辨识环节——但也因此**不能**说这套实现
        证明了自动辨识可行。
        """
        f = self._notch_f
        h = self._servo_dt
        nyq = 0.5 / h
        if self.choice("notch") == "on" and 0.5 < f < 0.9 * nyq:
            self.servo.notch = NotchFilter(
                freq_hz=f, depth=self.get("notch_depth"),
                width=self.get("notch_width"), dt=h)
        else:
            self.servo.notch = None

    def _rebuild_shaper(self):
        f = max(self._notch_f, 0.5)
        self.shaper = InputShaper(self.choice("shaper"), f,
                                  self.SHAPER_ZETA, self.dt)
        self.shaper.reset(self._theta_ref if hasattr(self, "_theta_ref")
                          else Q_HOME[self.JOINT])

    SHAPER_ZETA = 0.02      # 整形器假定的阻尼比。真机上要靠辨识，这里给个小值

    def on_change(self, key):
        if not hasattr(self, "servo"):
            return
        if key == "kvp":
            self.servo.kvp = self.get("kvp")
        elif key == "kvi":
            self.servo.kvi = self.get("kvi")
        elif key == "kpp":
            self.servo.kpp = self.get("kpp")
        elif key == "f_current":
            self.servo.f_current = self.get("f_current")
        elif key in ("stiffness", "j_motor", "damping"):
            self.joint.k_s = self.get("stiffness")
            self.joint.j_m = self.get("j_motor")
            self.joint.d_s = self.get("damping")
            self._notch_f = self._resonance_hz()
            self._rebuild_notch()
            self._rebuild_shaper()
        elif key in ("notch", "notch_depth", "notch_width"):
            self._rebuild_notch()
        elif key == "shaper":
            self._rebuild_shaper()

    # ------------------------------------------------------------ 参考轨迹

    def _ref(self, t):
        """演示关节的位置指令（整形之前）。"""
        j = self.JOINT
        a = self.get("move_amp")
        T = self.get("move_period")
        mode = self.choice("move")
        if mode in ("hold", "probe", "probe_cl"):
            return Q_HOME[j]
        if mode == "sine":
            return Q_HOME[j] + a * np.sin(2.0 * np.pi * t / T)
        # point：走过去停一半时间，再走回来停一半时间。
        # 阶跃是**故意**的——输入整形要看的正是「阶跃之后还晃不晃」。
        return Q_HOME[j] + (a if (t % T) < 0.5 * T else 0.0)

    # ------------------------------------------------------------ 控制

    def control(self, t, q, v):
        j = self.JOINT
        tau = np.zeros(N_ARM)

        # 其余关节：⭐ 计算力矩锁定（带宽 50 Hz，远高于谐振段）。
        #
        # 为什么必须锁得这么硬：双质量模型假设「负载是一个转动惯量 J_l」，
        # 只有其余关节刚性固定时 J4 感受到的惯量才等于 M[j,j]。
        #   · 之前用 kp=900 的普通 PD，带宽只有 √900/2π ≈ 4.8 Hz —— 比谐振还低，
        #     邻居跟着一起晃，频谱里冒出一堆与 K_s、J_m 无关的假峰。
        #   · 之后试过直接改写 qpos/qvel 做运动学钳位，那是**逐步冲击式约束**，
        #     会按频率吸放能量，实测把谐振峰推高约 16%。
        # 计算力矩法用完整质量矩阵解耦：给定期望角加速度 a_des，
        # 所需力矩就是 M·a_des + h。演示关节那一行用它**实际的**角加速度
        # （从 MuJoCo 读），于是耦合项被精确抵消。
        lock = np.ones(N_ARM, dtype=bool)
        lock[j] = False
        w_lock = 2.0 * np.pi * 50.0
        a_des = w_lock ** 2 * (Q_HOME - q) - 2.0 * w_lock * v
        a_des[j] = float(self.data.qacc[self.robot.qvel_idx[j]])
        tau_lock = self.reg.mass_matrix(q) @ a_des + self.reg.bias(q, v)
        tau[lock] = tau_lock[lock]

        # 演示关节：指令 → 输入整形 → 三环串级 → 电机侧积分 → 弹簧力矩
        raw_ref = self._ref(t)
        self._theta_ref = self.shaper(raw_ref)

        theta_l = float(q[j])
        omega_l = float(v[j])
        tau_g = float(self.reg.bias(q, v)[j])
        # 伺服环跑 SUBSTEPS 个子步；负载侧在子步内保持不变（零阶保持）
        h = self._servo_dt
        # 速度反馈滤波系数：一阶低通 a = h/(h+T)，T = 1/(2π f_vel)
        a_vel = h / (h + 1.0 / (2.0 * np.pi * max(self.get("f_vel"), 1e-6)))
        # probe 辨识模式：在电机转矩上叠加白噪声。
        # ⭐ 为什么要有这个模式：阶跃指令（point）自己就含全频段谐波，
        #    频谱上分不清「哪根峰是机械谐振、哪根是指令谐波」，f_meas 会读到假值。
        #    白噪声在各频率能量相同，系统只会在自己的谐振点被放大 ⇒ 峰就是谐振。
        #    这正是工业驱动器「自动谐振检测 / 一键整定」的做法。
        # ⭐ 两种辨识模式（第四轮审核 P1-07）：
        #   probe    —— 注入噪声 **且旁路伺服环**：量的是纯机械谐振。
        #   probe_cl —— 注入噪声 **但伺服环留在回路里**：量的是闭环峰。
        # 分开是因为"f_meas 不随 Kvp 变"这句话在 probe 下是**恒真**的
        # （伺服环根本不在回路里），不能拿它去证明闭环性质。
        _mv = self.choice("move")
        inject = _mv in ("probe", "probe_cl")
        probe = _mv == "probe"
        tau_probe = 0.0
        tau_s = 0.0
        tau_m = 0.0
        for _ in range(self.SUBSTEPS):
            # —— 编码器：只测角度，速度靠差分 + 低通（真实伺服就是这样）——
            raw_omega = (self.joint.theta_m - self._theta_m_prev) / h
            self._theta_m_prev = self.joint.theta_m
            self._omega_meas += a_vel * (raw_omega - self._omega_meas)
            # 送进控制器的是「上一拍」的测量值：采样保持 + 计算延迟共一个周期
            tau_m = self.servo.update(self._theta_ref, 0.0,
                                      self._theta_m_delayed, self._omega_delayed,
                                      h)
            self._theta_m_delayed = self.joint.theta_m
            self._omega_delayed = self._omega_meas
            if inject:
                tau_probe = self.get("probe_amp") * self._rng.standard_normal()
            if probe:
                # ⭐ 辨识模式下**旁路伺服环**，只用一个极软的 PD 把关节稳住。
                # 理由：谐振频率是**机械本身**的属性（只由 K_s、J_m、J_l 决定）。
                # 若带着速度环一起测，闭环阻尼会把观测到的峰拉到
                # ω_n·√(1−2ζ²) —— 实测 K_s=400 时峰在 5.86 Hz，正好等于
                # 6.55×0.894，是真实的阻尼峰移，不是测量错误。
                # 但那样测出来的就不再是「机械谐振点」，拿去设陷波会设偏。
                # 真实驱动器的「自动谐振检测」同样是先测被控对象再设陷波。
                # 软 PD 带宽取 0.3 Hz：它自带的阻尼 2·J_m·ω_soft 会给谐振模态
                # 加阻尼，把观测峰按 ω_n·√(1−2ζ²) 往下拉。带宽从 1 Hz 降到 0.3 Hz
                # 后这项阻尼降到 1/3，实测偏差从 −1.8% 收到 1 个谱线以内。
                w_soft = 2.0 * np.pi * 0.3
                tau_m = (self.joint.j_m * w_soft ** 2
                         * (self._theta_ref - self._theta_m_delayed)
                         - 2.0 * self.joint.j_m * w_soft * self._omega_delayed)
            # 重力补偿加在电机侧：真实伺服的前馈也是打在电机指令上的
            tau_s = self.joint.step(tau_m + tau_g + tau_probe, theta_l, omega_l, h)
        tau[j] = tau_s

        self._tau_m = tau_m + tau_g
        self._tau = np.clip(tau, -self.robot.tau_limit, self.robot.tau_limit)
        self._err = raw_ref - theta_l

        # 振动读数 = 电机侧速度里**谐振带内**的能量。
        # ⭐ 为什么必须用带通、不能用一阶高通：
        #    实测发现阶跃指令引起的低频晃动（外环位置环 ≈0.9 Hz）和阶跃本身的
        #    尖角，会大量漏过一阶高通。结果是「陷波打开后带内能量降到 12.5%，
        #    可 vib_amp 反而显示 113%」——读数和真实效果相反。
        #    改成中心在 f_res、Q=2 的二阶带通后，读数才真正代表「谐振振动」。
        # ⭐ 为什么看电机侧不看负载侧：谐振模态下两侧反向摆动，且幅值比是
        #    ω_l/ω_m = J_m/(J_m+J_l)，负载惯量越大负载摆得越小。信号在电机侧
        #    最强，真实伺服调试看的也正是电机编码器。
        # 反谐振与谐振的几何中点，随 K_s / J_m 自动跟着走。
        # ⚠️ 历史遗留：它曾经是 FFT 峰值搜索的下沿，**现在已经不是了**——
        # 峰值搜索改用固定常数 PEAK_SEARCH_FLOOR_HZ（见 _peak_freq 的说明），
        # 本量只作为「反谐振与谐振之间」的参考刻度保留，不参与任何判定。
        self._hp_fc = float(np.sqrt(self._antiresonance_hz() * self._resonance_hz()))
        f0 = self._resonance_hz()
        # ⚠️ 注意：**不要**把带通中心 f0 和上面那个几何中点混为一谈，
        # 两者是两件事：f0 是带通中心（要压在峰上），几何中点是分界参考。
        # Chamberlin 状态变量滤波器：f = 2·sin(π·f0·dt)，q = 1/Q
        kf = 2.0 * np.sin(np.pi * min(f0 * self.dt, 0.24))
        kq = 1.0 / 2.0                                   # Q = 2
        omega_m = self.joint.omega_m
        self._svf_lp += kf * self._svf_bp
        hp = omega_m - self._svf_lp - kq * self._svf_bp
        self._svf_bp += kf * hp
        bp = self._svf_bp
        self._vib = 0.98 * self._vib + 0.02 * bp * bp

        # ⭐ 存进 FFT 缓冲的是**弹簧形变 δ = θ_m − θ_l**，不是速度。
        # 理由是纯数学的：把双质量方程消元可得
        #     δ/τ_m = 1 / ( J_m s² + K_s (J_m+J_l)/J_l )
        # ——**只有极点、没有零点**，极点恰在 ω_res。所以 δ 的频谱上
        # 有且只有一个峰，就是谐振峰。
        # 而电机侧速度 ω_m/τ_m = (J_l s² + K_s)/(s[J_mJ_l s² + K_s(J_m+J_l)])
        # 带一个零点（反谐振），频谱是「先一个凹口再一个峰」，
        # 实测中凹口比峰更显眼，argmax 会挑错。
        self._vel_buf[self._vel_i] = self.joint.theta_m - theta_l
        self._vel_i = (self._vel_i + 1) % self.FFT_N
        if self._vel_i == 0:
            self._vel_full = True
        if self._vel_full and self._fft_countdown <= 0:
            self._f_meas = self._peak_freq()
            self._fft_countdown = 50              # 每 100 ms 更新一次
        self._fft_countdown -= 1

        self._p = self.robot.fk(q)[0].copy()
        qd = q.copy()
        qd[j] = raw_ref
        self._p_des = self.robot.fk(qd)[0].copy()
        self.trail.push(self._p)
        return self._tau

    #: 峰值搜索下沿（Hz）。**固定常数，故意不依赖解析谐振频率**——
    #: 见 :meth:`_peak_freq` 里的说明。参数范围内 f_res ∈ [3.21, 123.8] Hz，
    #: 位置环低频晃动 ≈0.9 Hz，2.0 Hz 把两者分开且留有余量。
    PEAK_SEARCH_FLOOR_HZ = 2.0

    def _peak_freq(self) -> float:
        """用 Welch 平均谱在弹簧形变 δ 上找谐振峰。测不到明显峰就返回 0。

        ⭐ 这是**独立于解析公式**的测量：只用时域 δ 序列做 FFT，
        不碰 resonance_frequency() 的任何结果。所以它能用来验证 f_res。

        为什么要 Welch 平均
        -------------------
        单次 FFT 的每根谱线本身就是随机量（方差等于均值，信噪比恒为 1），
        噪声激励下峰不够尖时，argmax 挑到的往往是噪声。
        把长记录切成多段、各自算功率谱再平均，方差按段数下降，峰才稳。
        真实频谱分析仪按的正是这个原理。

        显著性判据有三条，缺一不可：
        1. 信号本身要够大（δ 的 RMS 超过 1 µrad），否则是数值噪声；
        2. 峰必须是**内部极大值**（比左右两根谱线都高）。谱单调下降时
           argmax 会落在频段最左边那根 —— 那是边界，不是峰；
        3. 峰值要比同频段中位数高 8 倍。

        三条都是为了「测不到就如实说测不到」。高增益把谐振压死时
        本来就没有峰，此时报 0（界面显示 --）比编一个数出来诚实。
        """
        x = np.roll(self._vel_buf, -self._vel_i)
        if float(np.std(x)) < 1e-6:               # 判据 1：太安静
            return 0.0
        n = self.FFT_SEG
        win = np.hanning(n)
        acc = None
        nseg = 0
        for s in range(0, len(x) - n + 1, n // 2):
            seg = x[s:s + n]
            p = np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2
            acc = p if acc is None else acc + p
            nseg += 1
        if acc is None:
            return 0.0
        power = acc / nseg
        freqs = np.fft.rfftfreq(n, self.dt)
        # ⭐ 搜索下沿必须是**固定常数**，不能用解析谐振频率（第四轮审核 P1-07）。
        #
        # 旧代码用 ``freqs >= self._hp_fc``，而 ``_hp_fc`` = √(f_ares·f_res)，
        # 在参数范围内会从 2.30 Hz 一路跟到 38.03 Hz。也就是说
        # **搜索窗口是被待验证的公式自己定位的**——这不是独立测量，
        # 是"我猜峰在这儿，然后只在这儿找，果然找到了"。
        #
        # 换成固定 2.0 Hz：整个参数范围里 f_res 最低是 3.21 Hz
        # （K_s=120, J_m=0.40），仍在窗口内；而位置环低频晃动（≈0.9 Hz）
        # 被挡在外面。窗口一旦固定，"找到的峰 = 理论值"就重新变成
        # 一条**可以证伪**的陈述。
        band = freqs >= self.PEAK_SEARCH_FLOOR_HZ
        if np.count_nonzero(band) < 3:
            return 0.0
        pb = power[band]
        fb = freqs[band]
        # 判据 2：只在内部极大值里挑最大的
        inner = np.where((pb[1:-1] > pb[:-2]) & (pb[1:-1] > pb[2:]))[0] + 1
        if inner.size == 0:
            return 0.0
        i = int(inner[int(np.argmax(pb[inner]))])
        med = float(np.median(pb))
        if med <= 0.0 or pb[i] < 8.0 * med:       # 判据 3
            return 0.0
        # ⚠️ 这里**故意不做**抛物线插值。
        # 试过：对峰值和左右邻居的对数功率拟合抛物线取顶点。结果反而更差
        #   K_s=400：原始 argmax 6.35 Hz（真值 6.55，−3.1%）
        #             插值后 6.19 Hz（−5.4%）
        # 原因是这个峰**不对称**——低频侧压在反谐振（3.02 Hz）的上升裙上，
        # 高频侧则陡降。实测左邻居 1.36e-2 比右邻居 5.95e-3 大一倍多，
        # 抛物线顶点就被裙边拽向低频。
        # 抛物线插值的前提是「峰局部对称」，双质量系统的谐振峰不满足这个前提。
        return float(fb[i])

    # ------------------------------------------------------------ 读数

    def telemetry(self, t, q, v):
        j = self.JOINT
        lim = self.robot.tau_limit
        j_l = self._j_load()
        k_s, j_m = self.get("stiffness"), self.get("j_motor")
        f_res = resonance_frequency(k_s, j_m, j_l) / (2.0 * np.pi)
        f_ares = antiresonance_frequency(k_s, j_l) / (2.0 * np.pi)

        # 理论残余振动：只用整形器的脉冲序列 + 当前真实谐振频率。
        # ⭐ 整形器是按 self._notch_f 设计的；如果之后改了刚度，
        # 真实频率就和设计频率不一致，这个数会自动变大——正是鲁棒性演示。
        amps, times = InputShaper.impulses(
            self.choice("shaper"), max(self._notch_f, 0.5), self.SHAPER_ZETA)
        resid = residual_vibration(amps, times, f_res, self.SHAPER_ZETA)

        twist = self.joint.theta_m - float(q[j])
        return dict(
            f_res=f_res, f_meas=self._f_meas, f_ares=f_ares,
            j_load=j_l, inertia_ratio=j_l / max(j_m, 1e-9),
            vib_amp=float(np.sqrt(max(self._vib, 0.0))) * 1e3,
            err=self._err * 1e3, twist=twist * 1e3,
            theta_m=self.joint.theta_m, theta_l=float(q[j]),
            tau_m=float(self._tau_m), tau_s=float(self._tau[j]),
            resid_theory=resid,
            shaper_lag=self.shaper.lag * 1e3,
            notch_f=self._notch_f if self.servo.notch is not None else 0.0,
            cur_lag=1e3 / (2.0 * np.pi * self.get("f_current")),
            stiffness_now=k_s, j_motor_now=j_m,
            tau=float(np.linalg.norm(self._tau)), tau_vec=self._tau,
            tau_limit=lim,
            tau_max_pct=float(np.max(np.abs(self._tau) / lim)) * 100.0)

    def overlays(self):
        return _pair_markers(self._p_des, self._p) + [
            Trail(self.trail.points, color=C_ERROR, width=0.0045)]


SCENES = [ImpedanceScene, TrackingScene, AdaptiveScene, ObserverScene,
          EstimationScene, AttitudeScene, PlanningScene, ReplanScene,
          GraspScene, TeachingScene, IlcScene, ServoScene]
SCENE_BY_NAME = {c.name: c for c in SCENES}
