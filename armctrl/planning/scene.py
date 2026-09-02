"""在只读的 Menagerie Panda 模型上派生带障碍物的场景。

为什么用 MjSpec 而不是 XML `<include>`
--------------------------------------
`panda.xml` 里写的是 `<compiler meshdir="assets">`（相对路径）。
若在别处新建一个 XML 用 `<include file="/abs/.../panda.xml"/>`，
MuJoCo 解析网格路径时会丢掉这个相对 meshdir，报
`Error opening file '.../franka_emika_panda/link0.stl'`（少了 assets 一层）。
绕开它要么污染 menagerie 目录，要么在临时目录做符号链接。

`mujoco.MjSpec.from_file()` 是按原文件路径加载的，资产解析天然正确；
之后在内存里往 worldbody 里加障碍物再 `compile()`，
**不落地临时文件、不修改 menagerie 中的任何文件**。

场景里的几何体分三类
--------------------
    group 2   默认渲染可见。标记球（contype=conaffinity=0）与本模块加入的
              障碍物（contype=conaffinity=1，真的参与碰撞）都放这里，
              因为 MuJoCo 渲染器默认只显示 group 0-2，放 group 4 会看不见。
              两者靠 SceneInfo 里记录的几何体 id 区分，不靠 group 区分。
    group 3   参与碰撞的机械臂连杆几何（Panda 自带，默认不渲染，只用于碰撞）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from armctrl.assets import panda_dir


MENAGERIE = str(panda_dir())

#: 带标准平行夹爪的整机模型。调参台的全部场景都用它：
#: 末端连杆是 `hand`，TCP 落在两指夹持垫之间的抓取中心（见 core.robot.PANDA_TCP_OFFSET）。
#: 代价是手部质量进入动力学，且多出 finger_joint1/2 两个自由度；
#: 夹爪由腱传动位置伺服驱动，手臂控制律只作用在前 7 个关节上。
PANDA_XML = f"{MENAGERIE}/panda.xml"

#: 不带夹爪的纯 7 自由度模型，`demos/` 下已发布基线仍在用，保留仅为不打断那些数字。
PANDA_NOHAND_XML = f"{MENAGERIE}/panda_nohand.xml"

#: 手臂关节数。夹爪的两个手指关节不在此列。
N_ARM = 7

OBSTACLE_GROUP = 2
MARKER_GROUP = 2
ROBOT_COLLISION_GROUP = 3

#: 调参台所有 Panda 场景共用的眼在手上相机。它是真正挂在 ``hand`` 刚体上的
#: MuJoCo 固定相机，随末端一起运动；但视觉伺服当前仍使用 ObjectPoseSensor
#: 模拟的三维位姿，不从这路 RGB 图像做检测。相机画面只负责观察与教学。
WRIST_CAMERA = "wrist_camera"

_GEOM_TYPES = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}


@dataclass
class Obstacle:
    """一个静态障碍物。

    size 按 MuJoCo 约定：sphere 用 (r,)，box 用半边长 (hx,hy,hz)，
    capsule/cylinder 用 (r, half_length)。不足 3 个分量的补零。
    """

    name: str
    type: str
    size: tuple[float, ...]
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.85, 0.35, 0.25, 1.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def geom_name(self) -> str:
        return f"obs_{self.name}"


@dataclass
class SceneInfo:
    """派生场景的编译结果与索引。"""

    model: mujoco.MjModel
    obstacle_geoms: list[int] = field(default_factory=list)
    robot_geoms: list[int] = field(default_factory=list)
    floor_geom: int = -1


def robot_collision_geoms(model: mujoco.MjModel,
                          group: int = ROBOT_COLLISION_GROUP) -> list[int]:
    """返回机械臂参与碰撞的几何体 id（Panda 把它们放在 group 3）。"""
    return [i for i in range(model.ngeom) if int(model.geom_group[i]) == group]


def _body_chain(model: mujoco.MjModel, body: int) -> list[int]:
    chain, b = [], int(body)
    while b > 0:
        chain.append(b)
        b = int(model.body_parentid[b])
    chain.append(0)
    return chain


def movable_geoms(model: mujoco.MjModel, geoms, joint_ids) -> list[int]:
    """只保留会随被规划关节一起运动的几何体。

    固定底座（Panda 的 link0）上的几何体相对世界完全不动：它本来就压在地面上，
    把它放进"机械臂 vs 环境"的检查里，任何构型都会被判成碰撞。
    判据是从该几何体所在 body 往上走到世界，途中是否经过任何一个被规划的关节。
    """
    joint_bodies = {int(model.jnt_bodyid[j]) for j in joint_ids}
    out = []
    for g in geoms:
        b = int(model.geom_bodyid[g])
        while b > 0:
            if b in joint_bodies:
                out.append(int(g))
                break
            b = int(model.body_parentid[b])
    return out


def _joints_affecting(model: mujoco.MjModel, body: int) -> set[int]:
    """从 body 沿运动链走到世界，沿途遇到的全部关节 id。"""
    out: set[int] = set()
    b = int(body)
    while b > 0:
        adr, num = int(model.body_jntadr[b]), int(model.body_jntnum[b])
        for j in range(adr, adr + num):
            out.add(int(j))
        b = int(model.body_parentid[b])
    return out


def self_collision_pairs(model: mujoco.MjModel, geoms: list[int],
                         joint_ids: list[int] | None = None) -> list[tuple[int, int]]:
    """机械臂自碰撞需要检查的几何体对。

    排除三类：
      1. 同一 body 上的几何体——刚体内部不可能相对运动；
      2. 父子相邻 body——它们在关节处天然接触，任何构型下都"碰撞"，
         检查它们只会让所有构型都非法。
      3. 相对位姿与**被规划关节**无关的对——给定 joint_ids 时才启用。

    非相邻连杆之间即使中间隔了几节也要查：七自由度手臂完全可以自己碰自己。

    第 3 条为什么必要
    ----------------
    两根手指是 `hand` 的兄弟 body，既不同 body 也不是父子，会落进检查集合。
    但它们的相对位姿只由 finger_joint1/2 决定，而手指不在被规划的 7 个关节里；
    夹爪一闭合，规划器就会把"手指互相靠近"判成自碰撞，从而认为所有构型非法。
    手指相互靠近恰恰是抓取要做的事，本来就不该由路径规划器管。

    判据写成通用形式而不是"把手指名字硬编码排除"：
    body A 的位姿由 A 到世界这条链上的全部关节决定，记为 J(A)。
    A 相对 B 的位姿只取决于两条链的**对称差** J(A) △ J(B)（公共祖先段被抵消）。
    若这个对称差与被规划关节集合不相交，则无论规划器怎么动，这一对的相对位姿
    恒定不变，检查它没有任何意义。手指对、手指与手掌对都自动落入此类。
    """
    planned = None if joint_ids is None else {int(j) for j in joint_ids}
    affects = {}
    if planned is not None:
        for g in geoms:
            b = int(model.geom_bodyid[g])
            if b not in affects:
                affects[b] = _joints_affecting(model, b)

    pairs = []
    for a_i, ga in enumerate(geoms):
        for gb in geoms[a_i + 1:]:
            ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
            if ba == bb:
                continue
            if int(model.body_parentid[ba]) == bb or int(model.body_parentid[bb]) == ba:
                continue
            if planned is not None and not (affects[ba] ^ affects[bb]) & planned:
                continue
            pairs.append((ga, gb))
    return pairs



@dataclass
class FreeBody:
    """一个**可自由运动**的工件（带 freejoint），用于抓取场景。

    与 Obstacle 的区别：Obstacle 是焊死在世界上的静态障碍物，
    这个有 6 个自由度，会掉、会滑、会被夹住。

    friction 只给滑动摩擦系数；扭转/滚动用 MuJoCo 缺省值。
    注意实际生效的摩擦是两个几何体**逐元素取最大**的结果，
    指垫的 μ 更大时会主导，算理论夹持力必须从 contact.friction 现场读。
    """

    name: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    mass: float = 0.25
    friction: float = 0.6
    rgba: tuple[float, float, float, float] = (0.35, 0.55, 0.9, 1.0)

    def geom_name(self) -> str:
        return f"obj_{self.name}"

    def body_name(self) -> str:
        return f"objbody_{self.name}"


def _add_free_body(spec: mujoco.MjSpec, obj: FreeBody) -> None:
    b = spec.worldbody.add_body(name=obj.body_name(), pos=list(obj.pos))
    b.add_freejoint()
    g = b.add_geom(
        name=obj.geom_name(),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(obj.size),
        rgba=list(obj.rgba),
        mass=float(obj.mass),
    )
    g.group = OBSTACLE_GROUP
    g.contype = 1
    g.conaffinity = 1
    g.friction = [float(obj.friction), 0.005, 0.0001]


def _add_obstacle(spec: mujoco.MjSpec, obs: Obstacle) -> None:
    if obs.type not in _GEOM_TYPES:
        raise ValueError(f"不支持的障碍物类型: {obs.type}")
    size = np.zeros(3)
    size[: len(obs.size)] = obs.size
    body = spec.worldbody.add_body(name=f"obs_body_{obs.name}", pos=list(obs.pos))
    g = body.add_geom(
        name=obs.geom_name(),
        type=_GEOM_TYPES[obs.type],
        size=size.tolist(),
        quat=list(obs.quat),
        rgba=list(obs.rgba),
    )
    g.group = OBSTACLE_GROUP
    g.contype = 1
    g.conaffinity = 1


def _look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """让相机从 pos 看向 target 的四元数 (w,x,y,z)。

    MuJoCo 相机看向自身坐标系的 −z，x 向右、y 向上。因此
        z = normalize(pos − target)      背离目标
        x = normalize(up × z)            水平向右
        y = z × x                        向上
    比手写 xyaxes 好调：只要改观察点和被看的点。
    四元数转换复用 trajectory.quat_from_matrix（已用一万次随机旋转验证，
    最大往返误差 1.74e-15）。
    """
    from armctrl.planning.trajectory import quat_from_matrix

    z = np.asarray(pos, dtype=float) - np.asarray(target, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return quat_from_matrix(np.column_stack([x, y, z]))


def _add_wrist_camera(spec: mujoco.MjSpec) -> None:
    """在 Panda ``hand`` 上安装眼在手上相机。

    坐标都表达在 hand 局部系：

    * 相机位于夹爪中心线侧上方 65 mm，避免手掌完全挡住视野；
    * 光轴指向两指之间的抓取通道；
    * 70° 视场在 Home 悬停与贴近工件时都能同时看到目标和两根手指。

    MuJoCo 相机看自身 ``-z``，所以不能把 ``quat`` 留成单位四元数（那会看向
    hand 的 ``-z``，正好背对工件）。姿态用 ``_look_at_quat`` 从安装点和
    抓取通道中的一点独立构造，避免手写四元数时把正负方向弄反。
    """
    hand = spec.body("hand")
    if hand is None:
        raise ValueError("Panda 模型缺少 hand，无法安装夹爪相机")

    pos = np.array([0.065, 0.0, 0.015])
    target = np.array([0.0, 0.0, 0.20])
    cam = hand.add_camera()
    cam.name = WRIST_CAMERA
    cam.pos = pos.tolist()
    cam.quat = _look_at_quat(pos, target, up=(1.0, 0.0, 0.0)).tolist()
    cam.fovy = 70.0


def _drop_builtin_lights(spec: mujoco.MjSpec) -> None:
    """删掉模型文件里自带的灯，让布光完全由本模块决定。

    这不是洁癖。menagerie 的 panda.xml 带了一盏 ``mode="trackcom"`` 的聚光灯，
    强度（diffuse 0.7）和这里的主光（0.72）相当，而且**位置跟着机器人质心走**。
    两盏都投影的灯叠在一起，连杆上就会出现随姿态变化的锯齿状自阴影边界。

    只动内存里的 spec，不写回文件——menagerie 是只读依赖。
    """
    for light in list(spec.worldbody.lights):
        spec.delete(light)


def _add_visuals(spec: mujoco.MjSpec) -> None:
    """地面、天空盒、灯光、相机——只影响渲染，不影响规划。

    地面是可碰撞的（机械臂不能戳进地板），天空盒与标记球不可碰撞。

    配色取向
    --------
    机械臂本体是白色的，画面里唯一需要被看清的也是它。因此环境一律压暗、
    压低对比度：地板格子做成低对比的深灰蓝、去掉高亮描边、反射系数归零，
    否则棋盘格和倒影会比机械臂本身更抢眼（这正是改版前的毛病）。

    灯光用三点布光而不是单顶光：
        key    右前上方，带阴影，负责主要明暗与体积感
        fill   左前方，不带阴影，把背光面提亮，避免关节结构糊成一团黑
        rim    后上方，勾出轮廓，把白色机身从灰蓝背景里分离出来
    单顶光时各连杆的圆柱面亮度几乎一致，看不出关节朝向。

    **必须先删掉模型自带的灯。** menagerie 的 panda.xml 里有一盏
    ``<light name="top" pos="0 0 2" mode="trackcom"/>``：默认类型是聚光灯、
    默认 diffuse 0.7、默认投影，而且 ``trackcom`` 让它**跟着机器人质心移动**。
    不删它的话实际是四点布光，其中最强的一盏还在动——
    表现出来就是连杆上出现一条会随姿态变化的锯齿状自阴影边界
    （实测：关掉全部阴影，锯齿消失；只改另外三盏灯，锯齿纹丝不动）。
    删除只发生在内存里的 MjSpec 上，不改动只读的 menagerie 文件。
    """
    _drop_builtin_lights(spec)

    tex_sky = spec.add_texture()
    tex_sky.name = "skybox"
    tex_sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    tex_sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    tex_sky.rgb1 = [0.16, 0.20, 0.26]
    tex_sky.rgb2 = [0.04, 0.05, 0.07]
    tex_sky.width, tex_sky.height = 512, 3072

    tex_gp = spec.add_texture()
    tex_gp.name = "groundplane"
    tex_gp.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex_gp.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex_gp.mark = mujoco.mjtMark.mjMARK_NONE
    tex_gp.rgb1 = [0.115, 0.130, 0.150]
    tex_gp.rgb2 = [0.085, 0.098, 0.115]
    tex_gp.width, tex_gp.height = 512, 512

    mat = spec.add_material()
    mat.name = "groundplane"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    mat.texuniform = True
    mat.texrepeat = [3, 3]
    mat.reflectance = 0.0
    mat.shininess = 0.0
    mat.specular = 0.05

    # 主光必须放在**默认相机的同一侧**（默认方位角 138°，这里取 98°，
    # 即相机左后 40°）。放在对面就是逆光：连杆的自阴影分界线正好落在朝向
    # 相机的那一面，而 MuJoCo 的阴影贴图在这种掠射角下必然走样，
    # 表现为一条随姿态抖动的锯齿带。实测：光源移到相机同侧后锯齿消失，
    # 只调分辨率或裁剪范围都压不掉它。
    _KEY_FROM_AZ, _KEY_ELEV = 98.0, 60.0
    _a, _t = np.deg2rad(_KEY_FROM_AZ), np.deg2rad(_KEY_ELEV)
    _src = np.array([np.cos(_a) * np.cos(_t), np.sin(_a) * np.cos(_t), np.sin(_t)])
    key = spec.worldbody.add_light()
    key.pos = (_src * 3.0).tolist()
    key.dir = (-_src).tolist()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.diffuse = [0.72, 0.72, 0.74]
    key.specular = [0.22, 0.22, 0.24]
    key.castshadow = True

    fill = spec.worldbody.add_light()
    fill.pos = [-1.4, -1.1, 1.6]
    fill.dir = [0.62, 0.49, -0.61]
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.diffuse = [0.26, 0.28, 0.32]
    fill.specular = [0.0, 0.0, 0.0]
    fill.castshadow = False

    rim = spec.worldbody.add_light()
    rim.pos = [-0.4, 1.6, 2.0]
    rim.dir = [0.18, -0.72, -0.67]
    rim.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    rim.diffuse = [0.30, 0.33, 0.40]
    rim.specular = [0.10, 0.10, 0.12]
    rim.castshadow = False

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = "groundplane"

    cam = spec.worldbody.add_camera()
    cam.name = "planning"
    cam_pos = [1.50, -1.38, 1.22]
    cam.pos = cam_pos
    # MjSpec 的相机只接受四元数（没有 XML 里的 xyaxes 简写）
    cam.quat = _look_at_quat(cam_pos, [0.28, 0.0, 0.55]).tolist()

    # 俯视相机：从正上方往下看，绕障碍物的路线在 xy 平面上一目了然。
    # 正上方看时 up 不能取 +z（与视线共线，叉乘为零），改用 +x 作为"上"。
    top = spec.worldbody.add_camera()
    top.name = "topdown"
    top_pos = [0.30, 0.0, 2.35]
    top.pos = top_pos
    top.quat = _look_at_quat(top_pos, [0.30, 0.0, 0.40], up=(1.0, 0.0, 0.0)).tolist()

    # 夹爪相机顾名思义要装在夹爪上；无夹爪模型（nohand）没有 hand，
    # 跳过即可——纯路径规划本来也用不到腕部视角。
    if spec.body("hand") is not None:
        _add_wrist_camera(spec)

    spec.stat.center = [0.28, 0.0, 0.55]
    spec.stat.extent = 1.6
    spec.visual.global_.azimuth = 130.0
    spec.visual.global_.elevation = -22.0
    # 离屏缓冲要覆盖调参台可能请求的最大分辨率，否则 mujoco.Renderer 直接报错。
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440
    # 抗锯齿采样数。Panda 连杆边缘细，4 采样时轮廓有明显阶梯。
    spec.visual.quality.offsamples = 8
    spec.visual.quality.shadowsize = 4096
    # headlight 是跟着相机走的均匀光，调高会把三点布光的明暗全部冲平，
    # 这里只留一点点环境光保底，主光交给上面三盏灯。
    spec.visual.headlight.diffuse = [0.14, 0.14, 0.15]
    spec.visual.headlight.ambient = [0.20, 0.21, 0.23]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    # 自定义 overlay 几何（力箭头、坐标架、轨迹拖尾）用的默认尺度
    spec.visual.scale.forcewidth = 0.035
    spec.visual.scale.contactwidth = 0.06
    spec.visual.scale.contactheight = 0.02
    spec.visual.scale.framelength = 0.22
    spec.visual.scale.framewidth = 0.016
    # 阴影贴图的覆盖范围。对平行光，贴图是一个半边长为
    # ``stat.extent × shadowclip`` 的正方形，用 shadowsize 个像素去铺。
    # 现在 1.6 × 0.8 = 1.28 m 半边、4096 像素 → 每像素 0.625 mm，
    # 比原先 shadowclip=2.0 时的 1.56 mm 细 2.5 倍，阴影边缘的阶梯明显变小。
    # 不再往下调是因为覆盖范围要装得下最远的障碍物阴影（规划场景的桌子），
    # 超出范围的部分会被直接裁掉、投不出影子。
    spec.visual.map.shadowclip = 0.8
    spec.visual.map.shadowscale = 0.8


def _add_marker(spec: mujoco.MjSpec, name: str, rgba) -> None:
    """mocap 标记球：仅用于可视化，不参与碰撞。"""
    b = spec.worldbody.add_body(name=name, pos=[0.0, 0.0, -1.0])
    b.mocap = True
    g = b.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.035, 0.0, 0.0],
        rgba=list(rgba),
    )
    g.group = MARKER_GROUP
    g.contype = 0
    g.conaffinity = 0


def add_imu(spec: mujoco.MjSpec, body: str = "hand",
            site_name: str = "imu_site",
            pos=(0.0, 0.0, 0.0)) -> None:
    """在指定连杆上挂一个 IMU（加速度计 + 陀螺）以及配套的真值传感器。

    传感器语义（实测确认，非查文档）
    ------------------------------
    `accelerometer` 输出的是**比力**，与真实 IMU 一致：

        f_body = R_bodyᵀ · ( a_kin − g )

    验证依据：自由落体的小球读 (0,0,0)；静置地面的小球读 (0,0,+9.81)；
    Panda 在重力补偿 + PD 下做平缓正弦运动时，用 `framelinvel` 的时间中心差分
    独立求出 `a_kin`，`framelinacc == a_kin − g` 偏差 0.0040，
    而 `framelinacc == a_kin` 偏差 9.8113（正好一个 g）。

    因此估计器要还原运动学加速度必须 `a = R·f + g`，**不能**直接积分比力。

    同时挂 `framepos` / `framelinvel`：它们是**仿真真值**通道，
    与估计器完全无关，用作独立对照。真机上没有这一路。
    """
    b = spec.body(body)
    if b is None:
        raise ValueError(f"连杆不存在: {body}")
    site = b.add_site()
    site.name = site_name
    site.pos = list(pos)
    site.group = MARKER_GROUP

    for kind, name in (
        (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "imu_acc"),
        (mujoco.mjtSensor.mjSENS_GYRO, "imu_gyro"),
        (mujoco.mjtSensor.mjSENS_FRAMEPOS, "imu_truth_pos"),
        (mujoco.mjtSensor.mjSENS_FRAMELINVEL, "imu_truth_vel"),
        (mujoco.mjtSensor.mjSENS_FRAMELINACC, "imu_truth_acc"),
    ):
        s = spec.add_sensor()
        s.name = name
        s.type = kind
        s.objtype = mujoco.mjtObj.mjOBJ_SITE
        s.objname = site_name


def build_panda_scene(obstacles: list[Obstacle],
                      with_visuals: bool = True,
                      markers: tuple[str, ...] = (),
                      imu_body: str | None = None,
                      imu_pos=(0.0, 0.0, 0.0),
                      free_bodies: list["FreeBody"] | None = None,
                      nohand: bool = False) -> SceneInfo:
    """派生一个带障碍物的 Panda 场景并编译。

    返回 SceneInfo，其中 obstacle_geoms 已按传入顺序给出几何体 id，
    robot_geoms 是 Panda 自带的碰撞几何。地面若存在也算作障碍物。

    imu_body 非空时在该连杆上挂一套 IMU 与真值传感器（见 add_imu）。
    imu_pos 给出 site 在该连杆坐标系下的位置——**它必须与估计器所用的量测点
    重合**，否则滤波器会看到一个恒定偏差并试图用零偏去解释它。

    nohand=True 载入**不带夹爪**的 7 自由度模型。纯路径规划/时间参数化不关心
    夹爪，用它可以少两个自由度、末端 body 是 ``attachment``；带夹爪时
    ``nq=9``、末端是 ``hand``。两者的末端 body 名字不同，别弄混。
    """
    spec = mujoco.MjSpec.from_file(PANDA_NOHAND_XML if nohand else PANDA_XML)
    if with_visuals:
        _add_visuals(spec)
    for obs in obstacles:
        _add_obstacle(spec, obs)
    for i, name in enumerate(markers):
        _add_marker(spec, name, (0.15, 0.85, 0.35, 0.8) if i else (1.0, 0.2, 0.2, 0.8))
    for fb in (free_bodies or []):
        _add_free_body(spec, fb)
    if imu_body:
        add_imu(spec, imu_body, pos=imu_pos)

    model = spec.compile()

    def gid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise RuntimeError(f"几何体未找到: {name}")
        return i

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obs_ids = [gid(o.geom_name()) for o in obstacles]
    if floor >= 0:
        obs_ids.append(floor)
    return SceneInfo(
        model=model,
        obstacle_geoms=obs_ids,
        robot_geoms=robot_collision_geoms(model),
        floor_geom=int(floor),
    )
