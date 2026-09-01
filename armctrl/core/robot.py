"""机械臂模型封装：运动学、雅可比、动力学量的统一访问接口。

包装 MuJoCo 模型，提供控制算法需要的标准量：
    正运动学     T(q) = [R, p]
    几何雅可比   J(q) ∈ R^{6xn}，末端速度 [v; w] = J(q) q̇
    质量矩阵     M(q) ∈ R^{nxn}
    偏置力       h(q,q̇) = C(q,q̇)q̇ + g(q)
    重力项       g(q)

约定：
    - 所有雅可比、位姿均在世界系表达。
    - 只取手臂的 n 个驱动关节（Panda 为 7）；夹爪是独立的执行器，
      由 armctrl.core.gripper.ParallelJawGripper 管理，不进 arm_joints。

末端点（TCP）约定
----------------
`ee_body` 给出末端**连杆**，`tcp_offset` 给出该连杆坐标系下的工具中心点偏置。
`fk()` / `jacobian()` / `task_space_inertia()` 全部作用在 **TCP** 上，而不是连杆原点。

对装了标准平行夹爪的 Panda，末端连杆是 `hand`（法兰），TCP 是两个指尖夹持垫之间的
抓取中心，二者相差 0.1029 m（见 make_panda 的推导）。把控制点放在法兰上会让
阻抗刚度、IK 目标和可操作度全都算在一个**不接触工件**的点上，是语义错误。
需要法兰位姿时用 `flange_pose()`。

查询与仿真状态的隔离
------------------
`self.data` 是**仿真状态**，由外部循环 `mj_step` 推进，场景直接引用它。
而 `fk(q)` / `jacobian(q)` / `mass_matrix(q)` 这类查询要在**给定构型**上求值，
以前的实现是把 q 写回 `self.data` 再 `mj_forward`。这有两个后果：

1. `gravity(q)` 内部按定义要令 q̇ = 0，于是它会把**仿真的速度清零**。
   动量观测器每个控制周期都要调用 `gravity`，等于每步把被控对象的速度抹掉，
   动力学被彻底改变——而表面上看只是"读了一个重力项"。
2. 在 `q` 与仿真当前状态不同时（例如查询期望构型的正运动学），
   仿真状态会被静默改写成那个查询构型。

因此凡是**显式传入 q** 的查询一律走独立的 `_qdata` 缓冲，绝不碰 `self.data`；
`q=None` 表示"就用当前仿真状态"，此时只读不写。想主动改仿真状态请用 `set_state()`，
那是它唯一的职责。
"""

from __future__ import annotations

import numpy as np
import mujoco

from armctrl.assets import panda_xml


class ArmModel:
    """MuJoCo 机械臂的运动学/动力学查询接口。"""

    def __init__(self, xml_path: str | None = None, ee_body: str = "",
                 arm_joints: list[str] | None = None,
                 torque_control: bool = True,
                 model: "mujoco.MjModel | None" = None,
                 tcp_offset: np.ndarray | None = None):
        # model 直接给已编译模型，用于 MjSpec 派生出来的场景（如带障碍物的规划场景）；
        # 不给时按 xml_path 加载，行为与原来完全一致。
        if model is not None:
            self.model = model
        elif xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            raise ValueError("必须提供 xml_path 或 model 之一")
        self.data = mujoco.MjData(self.model)
        #: 只读查询专用的状态缓冲。所有"在给定构型上求值"的接口都用它，
        #: 保证查询不会改动 self.data 承载的仿真状态。
        self._qdata = mujoco.MjData(self.model)

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_body_id < 0:
            raise ValueError(f"末端 body 不存在: {ee_body}")

        # TCP 相对末端 body 原点的偏置，表达在**末端 body 坐标系**里。
        # 缺省 (0,0,0) 表示 TCP 就是连杆原点（无工具）。
        self.tcp_offset = (
            np.zeros(3) if tcp_offset is None
            else np.asarray(tcp_offset, float).reshape(3).copy()
        )

        self.joint_ids = []
        for name in arm_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"关节不存在: {name}")
            self.joint_ids.append(jid)

        # 关节在 qpos / qvel 中的地址。铰链关节两者一一对应。
        self.qpos_idx = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_idx = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.n = len(self.joint_ids)

        limits = self.model.jnt_range[self.joint_ids]
        self.q_lower = limits[:, 0].copy()
        self.q_upper = limits[:, 1].copy()

        self.tau_limit = np.array(
            [self.model.actuator_forcerange[i, 1] for i in range(self.n)]
        )

        # 驱动这 n 个手臂关节的执行器 id。显式解析 actuator→joint 的传动关系，
        # 而不是假设"执行器下标 == 关节下标"：Panda 的第 8 个执行器驱动的是
        # split 腱（夹爪），必须被排除在力矩化之外，否则夹爪会被改成直通电机
        # 而失去自带的位置伺服。
        self.arm_actuator_ids = self._resolve_arm_actuators()

        if torque_control:
            self._convert_to_torque_actuators()

    def _resolve_arm_actuators(self) -> list[int]:
        """按传动目标把执行器映射到手臂关节，返回与 joint_ids 同序的 id 列表。"""
        joint_to_act = {}
        for a in range(self.model.nu):
            if self.model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue                       # 腱传动（夹爪）等不算手臂执行器
            joint_to_act.setdefault(int(self.model.actuator_trnid[a, 0]), a)
        out = []
        for jid in self.joint_ids:
            a = joint_to_act.get(int(jid))
            if a is None:
                raise ValueError(
                    f"关节 {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)} "
                    "没有对应的关节型执行器"
                )
            out.append(a)
        return out

    def _convert_to_torque_actuators(self) -> None:
        """把**手臂**位置伺服执行器就地改成力矩电机。

        Menagerie 的 Panda 默认是位置伺服（gainprm=kp, biasprm=[0,-kp,-kd]），
        d.ctrl 是位置目标。做力矩级控制（阻抗、计算力矩、力控）必须改成
        直通电机 force = ctrl，否则内置 PD 会和外层控制律打架。

        夹爪执行器**不在此列**：它是腱传动的位置伺服，正是我们想要的行为
        （给宽度指令，内部 PD 去跟），改成直通电机反而没法用。
        """
        for i in self.arm_actuator_ids:
            self.model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            self.model.actuator_gainprm[i, :] = 0.0
            self.model.actuator_gainprm[i, 0] = 1.0
            self.model.actuator_biasprm[i, :] = 0.0
            self.model.actuator_ctrllimited[i] = 0
            self.model.actuator_ctrlrange[i, :] = [-1e6, 1e6]

    # ---------------- 状态读写 ----------------

    def set_state(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        """**改写仿真状态**。这是唯一会写 self.data 的接口。"""
        self.data.qpos[self.qpos_idx] = q
        if qd is not None:
            self.data.qvel[self.qvel_idx] = qd
        mujoco.mj_forward(self.model, self.data)

    def get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def get_qd(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def _at(self, q: np.ndarray | None,
            qd: np.ndarray | None = None) -> "mujoco.MjData":
        """返回一个在指定构型上已 mj_forward 的 MjData。

        q 为 None    直接返回仿真 data，不做任何写入（调用方负责已 forward）。
        q 不为 None  写进独立缓冲 `_qdata` 并 forward，仿真状态完全不受影响。

        非手臂自由度（夹爪的两个手指关节）从仿真状态同步过来，
        这样查询到的手部惯量、重力项与当前夹爪开口宽度一致；
        它们的速度置零，因为这些接口的语义都是"手臂在该构型下的量"。
        """
        if q is None:
            return self.data
        d = self._qdata
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0
        d.qpos[self.qpos_idx] = q
        if qd is not None:
            d.qvel[self.qvel_idx] = qd
        mujoco.mj_forward(self.model, d)
        return d

    # ---------------- 运动学 ----------------

    def fk(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """正运动学，返回 **TCP** 位置 p(3,) 与旋转矩阵 R(3,3)，均在世界系。

            p_tcp = p_body + R_body · tcp_offset
            R_tcp = R_body            （本实现的工具只有平移偏置，无姿态偏置）

        tcp_offset 为零时退化为末端连杆原点，与加 TCP 之前的行为一致。
        """
        d = self._at(q)
        p = d.xpos[self.ee_body_id].copy()
        R = d.xmat[self.ee_body_id].reshape(3, 3).copy()
        if self.tcp_offset.any():
            p = p + R @ self.tcp_offset
        return p, R

    def flange_pose(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """末端**连杆原点**（法兰）的位姿，不含 TCP 偏置。装工具前的口径。"""
        d = self._at(q)
        return (d.xpos[self.ee_body_id].copy(),
                d.xmat[self.ee_body_id].reshape(3, 3).copy())

    def tcp_position(self, q: np.ndarray | None = None) -> np.ndarray:
        return self.fk(q)[0]

    def jacobian(self, q: np.ndarray | None = None) -> np.ndarray:
        """**TCP 点**的几何雅可比 J(q) ∈ R^{6xn}，上 3 行线速度、下 3 行角速度，世界系。

        用 mj_jac(point, body) 而不是 mj_jacBody：后者只给连杆原点的雅可比。
        两者的角速度行相同，线速度行相差 −[R·offset]× · J_rot —— 也就是工具长度
        带来的杠杆项。TCP 偏置不为零时这一项**必须**算进去，否则速度映射、
        DLS 逆解和阻抗刚度都会与实际接触点对不上。
        """
        d = self._at(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self.tcp_offset.any():
            point = (d.xpos[self.ee_body_id]
                     + d.xmat[self.ee_body_id].reshape(3, 3) @ self.tcp_offset)
            mujoco.mj_jac(self.model, d, jacp, jacr, point, self.ee_body_id)
        else:
            mujoco.mj_jacBody(self.model, d, jacp, jacr, self.ee_body_id)
        return np.vstack([jacp[:, self.qvel_idx], jacr[:, self.qvel_idx]])

    def manipulability(self, q: np.ndarray | None = None) -> float:
        """Yoshikawa 可操作度 w = sqrt(det(J Jᵀ))，趋于 0 表示接近奇异。"""
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def condition_number(self, q: np.ndarray | None = None) -> float:
        """雅可比条件数 σ_max/σ_min，衡量各方向可操作性的不均匀程度。"""
        s = np.linalg.svd(self.jacobian(q), compute_uv=False)
        return float(s[0] / max(s[-1], 1e-12))

    # ---------------- 动力学 ----------------

    def mass_matrix(self, q: np.ndarray | None = None) -> np.ndarray:
        """关节空间质量矩阵 M(q) ∈ R^{nxn}，对称正定。

        整机模型含夹爪时取的是**手臂块** M[arm, arm]。夹爪由位置伺服锁住，
        手指的 q̇ 与 q̈ 都为零，代入整机方程后手臂那几行恰好只剩这个块，
        所以它不是"截取"，而是锁定手指条件下的精确降阶。
        """
        d = self._at(q)
        full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, d, full)
        return full[np.ix_(self.qvel_idx, self.qvel_idx)]

    def bias(self, q: np.ndarray | None = None, qd: np.ndarray | None = None) -> np.ndarray:
        """偏置力 h(q,q̇) = C(q,q̇)q̇ + g(q)。"""
        d = self._at(q, qd)
        return d.qfrc_bias[self.qvel_idx].copy()

    def gravity(self, q: np.ndarray | None = None) -> np.ndarray:
        """纯重力项 g(q)：令 q̇ = 0 时的偏置力。

        注意这里必须显式传 q（q 为 None 时取当前仿真构型），否则 `bias` 会走
        `q is None` 的分支去读仿真 data，而那里的 q̇ 通常不为零，
        返回的就不是重力项而是完整偏置力了。
        """
        q_now = self.get_q() if q is None else q
        return self.bias(q_now, np.zeros(self.n))

    def coriolis_times_qd(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """科氏/离心项 C(q,q̇)q̇ = h(q,q̇) - g(q)。"""
        return self.bias(q, qd) - self.gravity(q)

    def joint_anchor(self, index: int, q: np.ndarray | None = None) -> np.ndarray:
        """第 index 个手臂关节的**轴锚点**（世界系）。Panda 第 1 关节为 (0, 0, 0.333)。

        自己保证状态已 forward，不依赖调用方"先调一次 fk 把 data 填好"。
        `xanchor` 只在 mj_forward 之后才有效，未 forward 的 MjData 里全是零；
        取到零就等于把锚点当成世界原点，可达半径之类的量会被系统性算大
        （Panda 会从 0.855 m 变成 1.19 m，正好差一个基座台高度 0.333 m）。
        """
        d = self._at(self.get_q() if q is None else q)
        return np.asarray(d.xanchor[self.joint_ids[index]], float).copy()

    def task_space_inertia(self, q: np.ndarray | None = None) -> np.ndarray:
        """任务空间惯量 Λ(q) = (J M⁻¹ Jᵀ)⁻¹，阻抗控制中用于惯量整形。"""
        J = self.jacobian(q)
        Minv = np.linalg.inv(self.mass_matrix(q))
        return np.linalg.inv(J @ Minv @ J.T + 1e-9 * np.eye(6))

    # ---------------- 工具 ----------------

    def clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_lower, self.q_upper)

    def saturate_torque(self, tau: np.ndarray) -> np.ndarray:
        return np.clip(tau, -self.tau_limit, self.tau_limit)


#: Panda 标准平行夹爪的 TCP：`hand` 法兰坐标系下 z 方向 0.1029 m。
#:
#: 从模型几何**独立推出**，不是抄来的常数：
#:   finger body 原点        hand + 0.0584 m·ẑ      （panda.xml: left_finger pos）
#:   指尖夹持垫 z 向包络     [0.0360, 0.0530] m     （5 个 fingertip_pad 碰撞盒的并集）
#:   夹持垫中心              0.0445 m
#:   TCP = 0.0584 + 0.0445 = 0.1029 m
#: 夹持面在 finger 系 y≈0，两指对称张开，所以抓取中心正好落在 hand 的 z 轴上。
#: Franka 官方 `panda_hand_tcp` 标称 0.1034 m，与此相差 0.5 mm（网格与官方
#: 标称点的差别）。本项目取**模型自洽**的 0.1029，并由 test_endeffector.py
#: 从 mjModel 里重新推导一遍来锁住它。
PANDA_TCP_OFFSET = np.array([0.0, 0.0, 0.1029])


def make_panda(xml_path: str = None, tcp: str = "fingertip") -> ArmModel:
    """构造 Franka Panda（7 自由度 + 标准平行夹爪）模型。

    tcp="fingertip"  控制点在两指夹持垫之间的抓取中心（缺省，语义正确）
    tcp="flange"     控制点在 hand 法兰原点（加夹爪之前的旧口径，仅供对照）
    """
    if xml_path is None:
        xml_path = panda_xml()
    if tcp == "fingertip":
        offset = PANDA_TCP_OFFSET
    elif tcp == "flange":
        offset = np.zeros(3)
    else:
        raise ValueError(f"tcp 只支持 'fingertip' 或 'flange'，收到 {tcp!r}")
    return ArmModel(
        xml_path=xml_path,
        ee_body="hand",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
        tcp_offset=offset,
    )
