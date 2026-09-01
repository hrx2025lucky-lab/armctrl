"""为 `armctrl_讲解/` 生成配图。

**这不是测试代码，是文档的构建脚本。**放进仓库而不是随手跑一遍，
是为了让每一张图都可复现：参数、时长、相机全部写死在下面的 FIGURES 里，
改了控制器代码之后重跑一次就能看出配图该不该更新。

用法::

    cd "motion control"
    PYTHONPATH=. MUJOCO_GL=egl \\
      ../envs/armctrl/bin/python -m armctrl.tools.make_figures        # 全部
    ... -m armctrl.tools.make_figures 01 07                           # 只出这几篇的

输出到 ``armctrl_讲解/图/``。

设计约定
--------
- **每张图都是对照图**：并排的两三格是同一场景在不同参数下的样子。
  单张「好看的截图」讲不清任何事，对照才有信息量。
- 叠加几何（目标球 / 实际球 / 连线 / 力箭头 / 轨迹）直接来自场景自己的
  ``overlays()``，画法与 ``tuner/server.py`` 一致，**不另外编造图示**——
  这样图里的东西和你在原生窗口里看到的是同一套。
- **图上不写数字。** 数字放正文，且必须标来源档位；印进图片就没法标注，
  也没法随实测更新。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from armctrl.tuner.base import Arrow, Frame, Ghost, Sphere, Trail, hex_rgba
from armctrl.tuner.scenes import SCENES

OUT_DIR = Path(__file__).resolve().parents[2] / "armctrl_讲解" / "图"
W, H = 620, 600
#: 分隔条的灰度，和文档背景区分开即可
SEP = 90


def _push_overlays(scn, scene) -> None:
    """把 Scene.overlays() 的几何塞进渲染场景，画法与 tuner/server.py 保持一致。"""
    for item in scene.overlays():
        if isinstance(item, (Sphere, Ghost)):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            r = float(item.radius)
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
                np.asarray(item.pos, float).reshape(3), np.eye(3).ravel(),
                hex_rgba(item.color, item.alpha))
            scn.ngeom += 1
            continue

        if isinstance(item, Arrow):
            segs = [(item.start, item.end, item.width, item.color, item.alpha,
                     mujoco.mjtGeom.mjGEOM_ARROW)]
        elif isinstance(item, Trail):
            pts = np.asarray(item.points, float)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            segs = [(a, b, item.width, item.color, item.alpha,
                     mujoco.mjtGeom.mjGEOM_CAPSULE)
                    for a, b in zip(pts[:-1], pts[1:])]
        elif isinstance(item, Frame):
            p = np.asarray(item.pos, float).reshape(3)
            R = np.asarray(item.rot, float).reshape(3, 3)
            axis_colors = ("#ff5b5b", "#5bff8f", "#5b9bff")
            segs = [(p, p + R[:, k] * item.size, item.width, axis_colors[k],
                     item.alpha, mujoco.mjtGeom.mjGEOM_ARROW) for k in range(3)]
        else:
            continue

        for p0, p1, width, color, alpha, kind in segs:
            p0 = np.asarray(p0, float).reshape(3)
            p1 = np.asarray(p1, float).reshape(3)
            # 长度为零时 mjv_connector 会对零向量做归一化，画出朝向随机的几何
            if np.linalg.norm(p1 - p0) < 1e-9 or scn.ngeom >= scn.maxgeom:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.zeros(9),
                                hex_rgba(color, alpha))
            mujoco.mjv_connector(g, kind, float(width), p0, p1)
            scn.ngeom += 1


def _make_option(spec: dict | None) -> mujoco.MjvOption:
    """按 spec 构造一份可视化选项，用来复刻 MuJoCo 面板上的勾选。

    ⚠️ 这里**只处理 mjvOption**。MuJoCo 有两套互不相干的开关，很容易写混：

      - ``mjvOption.flags``  按 **mjtVisFlag**（mjVIS_*）索引 —— 面板上的
        Visualization 分组，控制「往场景里加哪些辅助几何」（关节轴、惯量盒、
        接触点…）。
      - ``mjvScene.flags``   按 **mjtRndFlag**（mjRND_*）索引 —— 面板上的
        Rendering 分组，控制「怎么画」（线框、阴影、反射、雾…）。

    两个数组长度不同、枚举值也不同。把 mjRND_WIREFRAME(=1) 写进
    ``opt.flags`` 不会报错，而是**悄悄打开了 mjVIS_TEXTURE(=1)**——
    线框没出来，纹理开关却被动了。渲染出来的图看着「差不多」，
    于是这种错误很难发现。渲染标志改由 :func:`_apply_render_flags` 处理。

    spec 支持的键，名字和面板上的分组一一对应：
      vis    —— Visualization 面板要**打开**的 mjVIS_* 名字后缀，如 "JOINT"
      off    —— Visualization 面板要**关掉**的
      rnd    —— Rendering 面板的 mjRND_* 名字后缀，如 "WIREFRAME"
      groups —— Group enable 面板的 geom 组号列表，如 [0, 1, 3]
      frame  —— Frame 下拉框的 mjFRAME_* 名字后缀，如 "SITE"
    """
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    # 与调参台保持一致：接触点/接触力**默认关闭**（见 tuner/server.py 的
    # _apply_contacts）。抓取时 ncon=32，橙色圆盘叠起来会糊住指尖和工件。
    # 需要它们的图（如 08_接触力.png）在 vopt 里显式写 CONTACTPOINT/CONTACTFORCE。
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    if not spec:
        return opt
    for name in spec.get("vis", ()):
        opt.flags[getattr(mujoco.mjtVisFlag, f"mjVIS_{name}")] = True
    for name in spec.get("off", ()):
        opt.flags[getattr(mujoco.mjtVisFlag, f"mjVIS_{name}")] = False
    if "groups" in spec:
        opt.geomgroup[:] = 0
        for g in spec["groups"]:
            opt.geomgroup[g] = 1
    if "frame" in spec:
        opt.frame = getattr(mujoco.mjtFrame, f"mjFRAME_{spec['frame']}")
    return opt


def _apply_render_flags(scn, spec: dict | None) -> None:
    """把 Rendering 面板的开关写进 mjvScene.flags（见 _make_option 的说明）。

    必须在 ``update_scene`` **之后**调用：mjv_updateScene 会重填 scene。
    """
    for name in (spec or {}).get("rnd", ()):
        scn.flags[getattr(mujoco.mjtRndFlag, f"mjRND_{name}")] = True


def render(scene_name: str, values: dict, seconds: float, view: dict,
           vopt: dict | None = None, overlays: bool = True) -> np.ndarray:
    """跑一段仿真再截一帧。

    values 覆盖场景参数，view 覆盖相机，vopt 复刻 MuJoCo 面板的勾选，
    overlays=False 时不画调参台自己的叠加几何（想单看面板效果时用）。
    """
    scene = {s.name: s for s in SCENES}[scene_name]()
    model = scene.build()
    for key, val in values.items():
        param = next(p for p in scene.params if p.key == key)
        scene.values[key] = str(val) if param.choices else float(val)
    scene.reset()
    data = scene.data
    for i in range(int(round(seconds / scene.dt))):
        q = data.qpos[scene.robot.qpos_idx].copy()
        v = data.qvel[scene.robot.qvel_idx].copy()
        data.ctrl[scene.robot.arm_actuator_ids] = scene.control(i * scene.dt, q, v)
        if scene.gripper is not None and scene.gripper_width is not None:
            scene.gripper.command_width(scene.gripper_width)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = view.get("lookat", scene.camera["lookat"])
    cam.distance = view.get("distance", scene.camera["distance"])
    cam.azimuth = view.get("azimuth", scene.camera["azimuth"])
    cam.elevation = view.get("elevation", scene.camera["elevation"])
    renderer.update_scene(data, cam, _make_option(vopt))
    _apply_render_flags(renderer.scene, vopt)
    if overlays:
        _push_overlays(renderer.scene, scene)
    img = renderer.render()
    renderer.close()
    return img


def compose(panels: list[np.ndarray]) -> np.ndarray:
    sep = np.full((panels[0].shape[0], 4, 3), SEP, dtype=np.uint8)
    out = [panels[0]]
    for p in panels[1:]:
        out += [sep, p]
    return np.hstack(out)


#: 文件名 -> (终端回显用的说明, [(场景, 参数覆盖, 仿真秒数, 相机), ...])
FIGURES: dict[str, tuple[str, list]] = {
    "01_标记说明.png": (
        "阻抗：绿球=期望位置，红绿蓝坐标架=TCP 姿态，品红=外力，黄=误差(放大4×)",
        [("impedance", {"k_trans": 1500.0, "force_mode": "constant", "fz": -26.0},
          4.0, dict(distance=0.62, azimuth=126.0, elevation=-8,
                    lookat=(0.42, -0.02, 0.40)))],
    ),
    "01_刚度对照.png": (
        "阻抗：同样 30 N 外力，刚度 2500 / 500 / 150 的下沉量",
        [("impedance", {"k_trans": k, "force_mode": "constant", "fz": -30.0},
          4.0, dict(distance=1.00, elevation=-8, lookat=(0.42, 0.0, 0.40)))
         for k in (2500.0, 500.0, 150.0)],
    ),
    "02_跟不上.png": (
        "跟踪：2 Hz、已对齐带宽，pd_gravity vs ctc 的末端落后量",
        [("tracking", {"controller": c, "freq": 2.0, "align": "on"}, 3.55,
          dict(distance=0.62, elevation=-12, lookat=(0.42, -0.05, 0.45)))
         for c in ("pd_gravity", "ctc")],
    ),
    "02_轨迹对照.png": (
        "跟踪：同一条参考轨迹，两种控制器画出的实际轨迹",
        [("tracking", {"controller": c, "freq": 1.4, "align": "on"}, 8.0,
          dict(distance=1.35, elevation=-14, lookat=(0.40, 0.0, 0.45)))
         for c in ("pd_gravity", "ctc")],
    ),
    # 自适应场景**刻意没有配图**：它要讲的「跟踪误差降下去了、参数偏差没降」
    # 是两条曲线的关系，3D 画面里两种激励强度看起来完全一样。
    # 与其凑一张没有信息的图，不如在正文里让读者去看浏览器里的曲线。
    "04_碰撞退让.png": (
        "观测器：none 硬扛 / compliant 退让中 / compliant 退到底",
        [("observer", {"react": r, "f_amp": -90.0, "k_i": 30.0}, t,
          dict(distance=1.35, elevation=-6, lookat=(0.42, 0.0, 0.30)))
         for r, t in (("none", 3.9), ("compliant", 3.9), ("compliant", 4.4))],
    ),
    "05_关掉修正就漂.png": (
        "状态估计：运动学修正 always vs off（绿=真值，青=估计，橙红线=1:1 间距）",
        [("estimation", {"update_mode": m}, 3.0,
          dict(distance=1.30, elevation=-10, lookat=(0.68, 0.02, 0.42)))
         for m in ("always", "off")],
    ),
    "06_裕度与平滑.png": (
        "规划：裕度 0.01 / 裕度 0.09 / 不平滑",
        [("planning", {"margin": m, "smooth_iters": s}, 0.4,
          dict(distance=2.30, elevation=-24, lookat=(0.30, 0.0, 0.45)))
         for m, s in ((0.01, 300.0), (0.09, 300.0), (0.03, 0.0))],
    ),
    "07_抓取阶段.png": (
        "抓取：对准 / 下降 / 闭合 / 提起",
        [("grasp", {}, t, dict(distance=0.74, elevation=-12,
                               lookat=(0.52, 0.0, 0.20)))
         for t in (1.2, 2.7, 4.0, 5.6)],
    ),
    "09_姿态误差.png": (
        "姿态解算：三路量测全关（漂）/ 只开水平修正（倾角回、偏航不回）/ 开位置量测（都回）",
        [("attitude", v, 8.0, dict(distance=0.42, elevation=-6,
                                   lookat=(0.42, -0.02, 0.42)))
         for v in ({"use_level": "off", "use_pos": "off", "motion": "still",
                    "tilt0": 8.0, "yaw0": 8.0},
                   {"use_level": "on", "use_pos": "off", "motion": "still",
                    "tilt0": 8.0, "yaw0": 8.0},
                   {"use_level": "off", "use_pos": "on", "motion": "both",
                    "tilt0": 8.0, "yaw0": 8.0})],
    ),
    "10_拖动示教.png": (
        "拖动示教：重力补偿准（停在绿球）/ +20% 误差（自己飘走）/ η=1.2（自激爬行）",
        [("teaching", v, 8.0, dict(distance=1.5, elevation=-12,
                                   lookat=(0.36, 0.0, 0.45)))
         for v in ({"assist": "off"},
                   {"assist": "off", "grav_err": 20.0},
                   {"assist": "off", "comp_ratio": 1.2})],
    ),
    "08_碰撞体.png": (
        "MuJoCo 面板：Geom 组 2（外观网格）vs 组 3（Panda 的简化碰撞体）",
        [("planning", {}, 0.4, dict(distance=2.30, elevation=-24,
                                    lookat=(0.30, 0.0, 0.45)), g, False)
         for g in ({"groups": [0, 1, 2]}, {"groups": [0, 1, 3]})],
    ),
    "08_关节与惯量.png": (
        "MuJoCo 面板：默认 / 关节轴+半透明 / 等效惯量 / 线框",
        [("impedance", {}, 2.0, dict(distance=1.55, elevation=-14,
                                     lookat=(0.32, 0.0, 0.45)), v, False)
         for v in (None,
                   {"vis": ["JOINT", "TRANSPARENT"]},
                   {"vis": ["INERTIA"]},
                   {"rnd": ["WIREFRAME"]})],
    ),
    "08_接触力.png": (
        "MuJoCo 面板：抓取瞬间的接触点与接触力（默认关闭，需按 V 或点「接触点」打开）",
        [("grasp", {}, t, dict(distance=0.30, elevation=-8,
                               lookat=(0.52, 0.0, 0.10)),
          {"vis": ["CONTACTPOINT", "CONTACTFORCE"]}, False)
         for t in (4.0, 5.6)],
    ),
    "07_开环扑空.png": (
        "抓取：工件偏移 60 mm，visual 抓到 vs openloop 扑空",
        [("grasp", {"closed_loop": c, "obj_jitter": 0.06}, 6.4,
          dict(distance=0.80, elevation=-14, lookat=(0.52, 0.03, 0.18)))
         for c in ("visual", "openloop")],
    ),
}


def main(argv: list[str]) -> None:
    import imageio.v2 as imageio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [a for a in argv if not a.startswith("-")]
    for fname, (desc, panels) in FIGURES.items():
        if wanted and not any(fname.startswith(w) for w in wanted):
            continue
        print(f"渲染 {fname}  —— {desc}")
        imgs = [render(*spec[:4], *spec[4:]) for spec in panels]
        imageio.imwrite(OUT_DIR / fname, compose(imgs))
    print(f"输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
