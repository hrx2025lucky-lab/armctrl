"""在线调参台的运行时：仿真循环 + 原生 MuJoCo 窗口 + HTTP 服务。

分工
----
    **MuJoCo 原生窗口**   看画面。这是 MuJoCo 自带的 Simulate 界面，
                          比往浏览器里推 JPEG 清楚得多：真正的实时渲染、
                          可以自由转视角、能开关接触点/接触力/惯量椭球等
                          一整套可视化开关，还能直接用鼠标拖拽施加外力。
    **浏览器 WebUI**      读讲解、调参数、看数字与曲线。

早先的版本把画面离屏渲染成 JPEG 再用 MJPEG 推到浏览器里。那样做画质差、
有压缩块、帧率受编码拖累，而且丢掉了原生窗口的全部交互能力。现在两者各司其职。

线程模型
--------
    主线程      仿真循环 + viewer.sync()。MuJoCo 的 GL 上下文只由 viewer
                自己的线程碰，本线程只调 sync()。
    HTTP 线程   ThreadingHTTPServer，只读遥测快照、只往命令队列里塞命令。

**所有会改动 model/data/scene 的操作都走命令队列**，由仿真线程在每轮循环开头
统一执行。这样浏览器点"重置"时不会在物理步中途把状态换掉。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import mujoco
import mujoco.viewer

from armctrl.tuner.base import Arrow, Frame, Ghost, Manual, Sphere, Trail, hex_rgba
from armctrl.tuner.scenes import SCENES, SCENE_BY_NAME
from armctrl.planning.scene import WRIST_CAMERA


HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
TELEMETRY_HZ = 50
HISTORY_SECONDS = 12

#: 三根坐标轴的颜色，x/y/z 对应红/绿/蓝，与绝大多数机器人软件一致。
_AXIS_COLORS = ("#ff5f56", "#66bb6a", "#4da3ff")

_CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
}


class Runner:
    """仿真循环 + 原生窗口。"""

    def __init__(self, scene_name: str, fps: int = 60):
        self.fps = fps
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.speed = 1.0
        self.history: deque = deque(maxlen=TELEMETRY_HZ * HISTORY_SECONDS)
        self.latest: dict = {}
        self.sim_time = 0.0
        self.realtime_factor = 1.0
        #: 遥测条目的自增序号，供前端做增量拉取
        self._seq = 0
        #: 手动干预（夹爪 / 关节扰动 / 摆位）。不属于任何场景，切场景时保留。
        self.manual = Manual()
        #: 原生 MuJoCo 窗口的视角。切场景时保留，避免每次都跳回全景。
        self.camera_mode = "overview"
        #: 接触点 / 接触力标记是否显示。默认关闭，理由见 ``_tune_viewer``。
        #: 与视角一样切场景时保留。
        self.show_contacts = False

        self.viewer = None
        self.scene = None
        self._load(scene_name)

    # ------------------------------------------------------------ 场景管理

    def _load(self, name: str) -> None:
        cls = SCENE_BY_NAME[name]
        scene = cls()
        model = scene.build()
        scene.reset()
        # 换场景就是换 MjModel，viewer 绑定的是旧模型，必须重开一个窗口。
        self._close_viewer()
        self.scene = scene
        self.viewer = mujoco.viewer.launch_passive(
            model, scene.data, key_callback=self._on_key,
            show_left_ui=False, show_right_ui=False)
        self._tune_viewer()
        self._set_camera_mode(self.camera_mode)
        with self.lock:
            self.history.clear()
            self.latest = {}
        self._seq = 0
        self.sim_time = 0.0
        self._step_count = 0
        self._wall0 = time.perf_counter()

    def _close_viewer(self) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.close()
        except Exception:                                     # noqa: BLE001
            pass
        # 给 viewer 线程一点时间把 GL 资源收干净再开下一个窗口，
        # 否则两个 GLFW 上下文会在同一瞬间存在，容易崩。
        time.sleep(0.25)
        self.viewer = None

    def _tune_viewer(self) -> None:
        """原生窗口的默认可视化开关。"""
        v = self.viewer
        # 接触点标记默认直径 60 mm，比抓取场景的 42 mm 工件还大，
        # 一接触就糊成两块橙砖把工件和指尖全挡住。缩到 12 mm 才看得出
        # 「接触点长在哪」——那才是这个标记存在的意义。
        # vis.scale 是 MjModel 上的量，viewer.Handle 不转发，只能改模型本身。
        vis = self.scene.model.vis
        vis.scale.contactwidth = 0.012
        vis.scale.contactheight = 0.004
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        self._apply_contacts()

    def _apply_contacts(self) -> None:
        """把 ``show_contacts`` 同步到原生窗口的两个可视化开关。

        默认**关闭**。这两个标记曾经默认打开，理由是「抓取场景里接触就是
        主角」；实测下来这个理由站不住：夹爪合拢后 ``data.ncon`` 稳定在
        **32**，32 个橙色圆盘（``vis.rgba.contactpoint`` = 0.9/0.6/0.2）
        叠在两个指尖上会连成一片，把指垫、工件棱线和插入间隙全糊掉——
        而这些几何细节才是抓取场景真正要看的东西。缩到 12 mm 也只是让
        单个盘变小，32 个盘重叠的总面积并没有变小。

        离屏渲染做过对照：只开接触力时橙色像素占比 0.000%，只开接触点时
        0.067%（整幅图，全景视角；贴近看时占满指尖）——**橙色 100% 来自
        接触点**，接触力箭头是淡青色（0.7/0.9/0.9），不参与遮挡。

        改成默认关、按需开：网页顶部「接触点」按钮，或聚焦 MuJoCo 窗口后
        按 ``V``。想看接触点长在哪的时候一键打开，平时不挡视线。
        """
        v = self.viewer
        if v is None:
            return
        on = bool(self.show_contacts)
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = on
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = on

    def _set_contacts(self, mode) -> None:
        """``True`` / ``False`` / ``"toggle"``。"""
        if mode == "toggle":
            self.show_contacts = not self.show_contacts
        else:
            self.show_contacts = bool(mode)
        self._apply_contacts()

    def _apply_camera(self, c: dict) -> None:
        v = self.viewer
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.fixedcamid = -1
            v.cam.azimuth = float(c["azimuth"])
            v.cam.elevation = float(c["elevation"])
            v.cam.distance = float(c["distance"])
            v.cam.lookat[:] = np.asarray(c["lookat"], dtype=float)
        self.camera_mode = "overview"

    def _set_camera_mode(self, mode: str) -> None:
        """切换原生窗口的全景 / 眼在手上相机。

        这里只换 ``MjvCamera``，不创建第二个渲染器，也不把画面编码后推到网页。
        因而不会重走早先导致 WebUI 卡顿的 MJPEG 路线。
        """
        if mode == "toggle":
            mode = "overview" if self.camera_mode == "wrist" else "wrist"
        if mode == "overview":
            self._apply_camera(self.scene.camera)
            return
        if mode != "wrist":
            raise ValueError(f"未知视角: {mode}")

        cid = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        if cid < 0:
            raise RuntimeError(f"模型里没有夹爪相机 {WRIST_CAMERA}")
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = cid
        self.camera_mode = "wrist"

    def _on_key(self, key: int) -> None:
        """MuJoCo 原生窗口快捷键：按 C 切换全景 / 夹爪相机，按 V 切换接触点标记。

        回调运行在 viewer 线程，只往命令队列写消息；真正修改相机仍由仿真主线程
        执行，避免两个线程同时碰 ``viewer.cam``。
        """
        if key == ord("C"):
            self.submit("view", "toggle")
        elif key == ord("V"):
            self.submit("contacts", "toggle")

    # ------------------------------------------------------------ 命令

    def submit(self, kind: str, payload=None) -> None:
        self.commands.put((kind, payload))

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(kind, payload)
            except Exception as exc:                          # noqa: BLE001
                self.scene.note = f"操作失败：{exc}"

    def _handle(self, kind: str, payload) -> None:
        if kind == "set":
            need_reset = False
            for k, v in payload.items():
                need_reset |= self.scene.set(k, v)
            if need_reset:
                self._reset()
        elif kind == "reset":
            self._reset()
        elif kind == "scene":
            self._load(payload)
        elif kind == "camera":
            self._apply_camera(dict(self.scene.camera, **payload))
        elif kind == "view":
            self._set_camera_mode(str(payload))
        elif kind == "contacts":
            self._set_contacts(payload)
        elif kind == "pause":
            self.paused = bool(payload)
        elif kind == "speed":
            self.speed = float(np.clip(payload, 0.05, 4.0))
        elif kind == "manual":
            was_pose = self.manual.pose_on
            self.manual.update(payload)
            if self.manual.pose_on and not was_pose and not self.manual.pose_q:
                # 刚打开摆位且还没给过角度：从当前构型起步，
                # 否则机械臂会瞬间跳到一个默认姿势。
                self.manual.pose_q = [
                    float(x) for x in self.scene.data.qpos[self.scene.robot.qpos_idx]]
            if not self.manual.pose_on:
                # 退出摆位模式时清掉残留速度，否则物理会从"上一次摆位的瞬时速度"
                # 接着跑，看起来像机械臂自己弹了一下。
                self.scene.data.qvel[self.scene.robot.qvel_idx] = 0.0

    def _reset(self) -> None:
        self.scene.reset()
        self.sim_time = 0.0
        self._step_count = 0
        self._wall0 = time.perf_counter()
        with self.lock:
            self.history.clear()
        self._seq = 0

    # ------------------------------------------------------------ 主循环

    def loop(self) -> None:
        sc = self.scene
        dt = sc.dt
        model = sc.model
        n_sub = max(int(round(dt / model.opt.timestep)), 1)
        tele_every = max(int(round(1.0 / (TELEMETRY_HZ * dt))), 1)
        sync_every = max(int(round(1.0 / (self.fps * dt))), 1)
        last_probe = time.perf_counter()
        probe_steps = 0

        while self.running:
            self._drain()
            if self.scene is not sc:                          # 场景换了，重取参数
                sc = self.scene
                dt = sc.dt
                model = sc.model
                n_sub = max(int(round(dt / model.opt.timestep)), 1)
                tele_every = max(int(round(1.0 / (TELEMETRY_HZ * dt))), 1)
                sync_every = max(int(round(1.0 / (self.fps * dt))), 1)

            if self.viewer is not None and not self.viewer.is_running():
                # 用户关掉了原生窗口 —— 整个调参台随之退出，
                # 否则会剩一个没有画面的后台进程占着端口。
                self.running = False
                break

            if self.paused or self.manual.pose_on:
                if self.manual.pose_on:
                    self._apply_pose(sc)
                self._sync(sc)
                time.sleep(0.03)
                self._wall0 = time.perf_counter() - self.sim_time / self.speed
                continue

            q = sc.data.qpos[sc.robot.qpos_idx].copy()
            v = sc.data.qvel[sc.robot.qvel_idx].copy()
            tau = sc.control(self.sim_time, q, v)
            # 按传动关系显式寻址，而不是 ctrl[:7]。整机模型有 8 个执行器，
            # 第 8 个是腱传动的夹爪位置伺服；用切片写会把它一起覆盖成 0，
            # 夹爪会毫无征兆地闭合。
            sc.data.ctrl[sc.robot.arm_actuator_ids] = tau
            self._apply_gripper(sc)
            # 手动扰动力矩走 qfrc_applied（广义外力），**不经过执行器限幅**，
            # 所以它是"外部有人在掰关节"，不是"把控制量改掉"。
            # 每步先清零再写：否则松开滑块后力矩会一直留在那里。
            sc.data.qfrc_applied[sc.robot.qvel_idx] = self.manual.joint_torque(
                len(sc.robot.qvel_idx))
            for _ in range(n_sub):
                mujoco.mj_step(model, sc.data)

            self._step_count += 1
            probe_steps += 1
            self.sim_time = self._step_count * dt

            if self._step_count % tele_every == 0:
                # 用**步进之后**的状态做遥测，而不是复用步进之前的 q/v。
                # 两个原因，都是硬伤：
                #   1. 时间基准。self.sim_time 已经是步进后的时刻，配上步进前的
                #      状态就是错拍，画出来的曲线整体平移一个控制周期。
                #   2. ArmModel 的查询接口会在给定构型上求值，传步进前的 q 进去
                #      语义就不对了。传步进后的状态才与当前时刻一致。
                q_now = sc.data.qpos[sc.robot.qpos_idx].copy()
                v_now = sc.data.qvel[sc.robot.qvel_idx].copy()
                tele = sc.telemetry(self.sim_time, q_now, v_now)
                tele["t"] = self.sim_time
                # **在这里清洗一次**，而不是每来一个 HTTP 请求就把整段历史重洗一遍。
                # 历史有 600 条、每条十几个字段，重洗一次约一万五千次数值检查；
                # 按 8 次/秒的轮询算，光这一项就要占掉 20% 以上的 GIL，
                # 直接把仿真线程拖到掉帧。
                clean = _clean(tele)
                self._seq += 1
                clean["seq"] = self._seq
                with self.lock:
                    self.history.append(clean)
                    self.latest = clean
            if self._step_count % sync_every == 0:
                self._sync(sc)

            now = time.perf_counter()
            if now - last_probe > 0.5:
                self.realtime_factor = probe_steps * dt / (now - last_probe)
                probe_steps = 0
                last_probe = now

            lag = self.sim_time / self.speed - (now - self._wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:                                  # 落后太多就重新对齐
                self._wall0 = now - self.sim_time / self.speed

        self._close_viewer()

    # ------------------------------------------------------------ 手动干预

    def _apply_gripper(self, sc) -> None:
        """写夹爪开口指令。手动值优先于场景值。

        场景把 `gripper_width` 设成 None 表示"我自己接管夹爪"（抓取场景），
        这时**手动值也不应该插手**——否则抓取过程中的力控指令会被位置指令顶掉，
        工件会当场掉下去。
        """
        if sc.gripper is None or sc.gripper_width is None:
            return
        w = sc.gripper_width if self.manual.grip is None else self.manual.grip
        sc.gripper.command_width(w)

    def _apply_pose(self, sc) -> None:
        """运动学摆位：直接写构型，只做前向运动学，不积分动力学。

        用 mj_forward 而不是 mj_step：前者只按当前 qpos 更新所有派生量
        （各连杆位姿、雅可比、接触检测），不推进时间、不产生加速度。
        这正是"用手把机械臂搬到某个姿势"的语义。
        """
        idx = sc.robot.qpos_idx
        q = self.manual.pose_q
        if len(q) == len(idx):
            lo = sc.model.jnt_range[sc.robot.joint_ids, 0]
            hi = sc.model.jnt_range[sc.robot.joint_ids, 1]
            sc.data.qpos[idx] = np.clip(np.asarray(q, dtype=float), lo, hi)
        sc.data.qvel[sc.robot.qvel_idx] = 0.0
        sc.data.qfrc_applied[:] = 0.0
        if sc.gripper is not None:
            w = 0.08 if self.manual.grip is None else self.manual.grip
            sc.gripper.set_state(w)                # 直接写手指关节，同样不走动力学
        mujoco.mj_forward(sc.model, sc.data)

    def joint_range(self) -> list[list[float]]:
        """7 个手臂关节的软限位 [lo, hi]（rad），供摆位滑块用。"""
        sc = self.scene
        if sc is None or sc.model is None:
            return []
        r = sc.model.jnt_range[sc.robot.joint_ids]
        return [[float(a), float(b)] for a, b in r]

    def grip_range(self) -> list[float]:
        """夹爪开口宽度的可行区间 [min, max]（m）。"""
        sc = self.scene
        if sc is None or sc.gripper is None:
            return [0.0, 0.08]
        return [float(sc.gripper.width_min), float(sc.gripper.width_max)]

    def manual_info(self) -> dict:
        """手动干预面板要显示的量。

        **由 Runner 自己算，不调场景的 telemetry()**：摆位模式下控制律根本没跑，
        场景遥测里的力矩、跟踪误差、RMS 累积量都会是错的或被污染的。
        这里只报与控制律无关的纯运动学量。
        """
        sc = self.scene
        if sc is None or sc.data is None:
            return {}
        q = sc.data.qpos[sc.robot.qpos_idx]
        info = dict(
            q=[float(x) for x in q],
            tcp=[float(x) for x in sc.robot.tcp_position()],
            grip_actual=None,
            grip_locked=sc.gripper_width is None,
        )
        if sc.gripper is not None:
            info["grip_actual"] = float(sc.gripper.width)
        return info

    # ------------------------------------------------------------ 原生窗口同步

    def _sync(self, sc) -> None:
        v = self.viewer
        if v is None:
            return
        try:
            self._draw_overlays(sc)
        except Exception as exc:                              # noqa: BLE001
            sc.note = f"叠加绘制失败：{exc}"
        v.sync()

    def _next_geom(self):
        """在 viewer 的用户场景里申请一个空 geom 槽位；满了返回 None。"""
        scn = self.viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        return g

    def _connector(self, kind, p0, p1, width, color, alpha) -> None:
        """在两点之间画一根连接体（箭头 / 胶囊 / 线）。

        长度为零时必须跳过：mjv_connector 内部要对方向做归一化，
        两点重合会得到 0/0，画出朝向随机的几何体，看起来像画面在抽搐。
        """
        p0 = np.asarray(p0, dtype=float).reshape(3)
        p1 = np.asarray(p1, dtype=float).reshape(3)
        if np.linalg.norm(p1 - p0) < 1e-9:
            return
        g = self._next_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.zeros(9),
                            hex_rgba(color, alpha))
        mujoco.mjv_connector(g, kind, float(width), p0, p1)

    def _draw_overlays(self, sc) -> None:
        items = sc.overlays()
        scn = self.viewer.user_scn
        scn.ngeom = 0                                          # 每帧重画
        for item in items:
            if isinstance(item, Arrow):
                self._connector(mujoco.mjtGeom.mjGEOM_ARROW, item.start, item.end,
                                item.width, item.color, item.alpha)
            elif isinstance(item, (Sphere, Ghost)):
                g = self._next_geom()
                if g is None:
                    continue
                r = item.radius
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
                    np.asarray(item.pos, dtype=float).reshape(3), np.eye(3).ravel(),
                    hex_rgba(item.color, item.alpha))
            elif isinstance(item, Frame):
                R = np.asarray(item.rot, dtype=float).reshape(3, 3)
                p = np.asarray(item.pos, dtype=float).reshape(3)
                for axis in range(3):
                    self._connector(mujoco.mjtGeom.mjGEOM_ARROW, p,
                                    p + R[:, axis] * item.size, item.width,
                                    _AXIS_COLORS[axis], item.alpha)
            elif isinstance(item, Trail):
                pts = np.asarray(item.points, dtype=float)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                for a, b in zip(pts[:-1], pts[1:]):
                    self._connector(mujoco.mjtGeom.mjGEOM_CAPSULE, a, b,
                                    item.width, item.color, item.alpha)
            else:
                raise TypeError(f"未知的叠加几何类型：{type(item).__name__}")

    # ------------------------------------------------------------ 给 HTTP 用

    def snapshot(self, since: int = -1) -> dict:
        """遥测快照。

        since 给出前端已经拿到的最大序号，只回传比它新的条目。
        全量重传 600 条历史（约 285 KB）× 每秒 8 次 = 2.4 MB/s，
        序列化开销会把 GIL 从仿真线程手里抢走，两边一起卡。
        增量之后每次只有几条，体积降两个数量级。
        since < 0 表示前端刚连上、需要一份全量。
        """
        with self.lock:
            if since < 0:
                hist = list(self.history)
            else:
                # deque 已按序号递增，从尾部往前取到第一个 <= since 的即可
                hist = [h for h in self.history if h.get("seq", 0) > since]
            latest = dict(self.latest)
        cam = dict(azimuth=0.0, elevation=0.0, distance=0.0)
        if self.viewer is not None:
            try:
                cam = dict(azimuth=float(self.viewer.cam.azimuth),
                           elevation=float(self.viewer.cam.elevation),
                           distance=float(self.viewer.cam.distance))
            except Exception:                                  # noqa: BLE001
                pass
        return dict(
            scene=self.scene.name,
            values=dict(self.scene.values),
            note=self.scene.note,
            paused=self.paused,
            speed=self.speed,
            sim_time=self.sim_time,
            realtime=self.realtime_factor,
            alive=bool(self.viewer is not None and self.viewer.is_running()),
            latest=latest,           # 写入时已清洗过，这里不再重复处理
            history=hist,
            seq=self._seq,
            full=(since < 0),
            camera=cam,
            camera_mode=self.camera_mode,
            show_contacts=self.show_contacts,
            manual=self.manual.to_json(),
            manual_info=self.manual_info(),
        )


def _clean(d: dict) -> dict:
    """NaN / inf 不是合法 JSON，转成 None 让前端断线。

    条形图的遥测值是列表（逐关节力矩、逐关节限幅），所以要按元素处理，
    不能只认标量——否则整条列表会原样塞进 JSON，里面的 NaN 会让
    `json.dumps` 产出非法的 `NaN` 字面量，前端解析直接抛错。
    """
    def scrub(v):
        if isinstance(v, (int, float, np.floating, np.integer)):
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else f
        if isinstance(v, np.ndarray):
            return [scrub(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [scrub(x) for x in v]
        return v

    return {k: scrub(v) for k, v in d.items()}


class Handler(BaseHTTPRequestHandler):
    runner: Runner = None                                      # 由 serve() 注入
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):                              # 静音访问日志
        pass

    # ------------------------------------------------------------ 工具

    def _send(self, code: int, body: bytes, ctype: str,
              cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _static(self, root: str, rel: str) -> None:
        """提供静态文件。路径必须落在 root 之内，防目录穿越。"""
        path = os.path.normpath(os.path.join(root, rel.lstrip("/")))
        if not path.startswith(os.path.realpath(root)) and not path.startswith(root):
            self._send(403, b"forbidden", "text/plain")
            return
        if not os.path.isfile(path):
            self._send(404, b"not found", "text/plain")
            return
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            # 字体与 css 不会变，允许缓存，省得每次刷新都重传 600 KB 字体
            self._send(200, f.read(), _CTYPES.get(ext, "application/octet-stream"),
                       cache="max-age=86400")

    # ------------------------------------------------------------ 路由

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path.startswith("/vendor/"):
            self._static(VENDOR, path[len("/vendor/"):])
        elif path == "/meta":
            self._json(dict(
                scenes=[dict(name=c.name, title=c.title) for c in SCENES],
                current=self.runner.scene.meta(),
                manual=self.runner.manual.to_json(),
                joint_range=self.runner.joint_range(),
                grip_range=self.runner.grip_range(),
            ))
        elif path == "/state":
            q = self.path.split("?", 1)
            since = -1
            if len(q) == 2:
                from urllib.parse import parse_qs
                v = parse_qs(q[1]).get("since", ["-1"])[0]
                try:
                    since = int(v)
                except ValueError:
                    since = -1
            self._json(self.runner.snapshot(since))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()
        if path == "/set":
            self.runner.submit("set", body)
        elif path == "/reset":
            self.runner.submit("reset")
        elif path == "/scene":
            self.runner.submit("scene", body["name"])
        elif path == "/camera":
            self.runner.submit("camera", body)
        elif path == "/view":
            self.runner.submit("view", body["mode"])
        elif path == "/contacts":
            self.runner.submit("contacts", body["on"])
        elif path == "/pause":
            self.runner.submit("pause", body["paused"])
        elif path == "/speed":
            self.runner.submit("speed", body["speed"])
        elif path == "/manual":
            self.runner.submit("manual", body)
        else:
            self._send(404, b"not found", "text/plain")
            return
        self._json(dict(ok=True))


def serve(scene: str = "impedance", port: int = 8770, host: str = "127.0.0.1",
          fps: int = 60, open_browser: bool = True) -> None:
    runner = Runner(scene, fps=fps)
    Handler.runner = runner
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    url = f"http://{host}:{port}/"
    print(f"调参台：{url}")
    print("MuJoCo 画面在**另一个原生窗口**里，关掉那个窗口即退出。")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:                                      # noqa: BLE001
            pass

    try:
        runner.loop()
    except KeyboardInterrupt:
        pass
    finally:
        runner.running = False
        httpd.shutdown()
        # GLFW 的清理和解释器销毁抢跑会段错误。这里先把 viewer 关干净、
        # 让它的线程退出，再用 os._exit 跳过 Python 的对象析构。
        time.sleep(0.3)
        os._exit(0)
