# armctrl — 机械臂运动规划与柔顺控制栈

[![tests](https://github.com/hrx2025lucky-lab/armctrl/actions/workflows/tests.yml/badge.svg)](https://github.com/hrx2025lucky-lab/armctrl/actions/workflows/tests.yml)

以 **Franka Panda**（7 自由度冗余臂）为对象，在 MuJoCo 中实现从运动学、轨迹规划
到力矩级柔顺控制的完整链路。每个模块都配可复现的定量验证实验，**723 个测试全部通过**
（其中 98 项校验的是不随本仓库分发的讲义，clone 环境下自动 skip，详见下文）。

```
armctrl_project/
├── armctrl/                      实现 + 27 个测试文件
├── tools/verification/           探针与变异验证脚本
├── requirements.txt              运行依赖
├── armctrl_调参台使用说明.md      12 个场景的动手实验与原理
└── tuner.sh                      交互式调参台
```

---

## 快速开始

模型来自 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0），
是独立的第三方资产仓库，不随本仓库分发：

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
export MUJOCO_MENAGERIE=/path/to/mujoco_menagerie   # 不设也行，见下

pip install -r requirements.txt
```

`armctrl/assets.py` 按 `MUJOCO_MENAGERIE` → 若干常见位置 → 兜底候选的顺序解析，
放在主目录或仓库同级时无需任何配置。

```bash
export PYTHONPATH=.

python armctrl/demos/demo_impedance.py    # 阻抗控制三组实验
python armctrl/demos/demo_collision.py    # 动量观测器碰撞检测
python armctrl/demos/demo_gripper.py      # TCP vs 法兰、夹持力标定、pick & place

./tuner.sh --list                         # 列出全部 12 个场景
./tuner.sh impedance
```

调参台会同时开两样东西：

| | 是什么 | 干什么 |
|---|---|---|
| **MuJoCo 原生窗口** | 3D 画面 | 转视角、**鼠标直接拖拽机械臂施加外力** |
| **浏览器页面** `http://127.0.0.1:8770/` | 讲解 + 滑块 + 读数 + 曲线 | 在线改参数，立刻看到物理量怎么变 |

⚠️ **原生窗口需要显示器**（走 `glfw`）。无显示器的服务器上开不出来——
测试与 demo 不受影响，只有调参台有这个要求。

**想让别的机器打开**（比如演示给别人看）：

```bash
./tuner.sh impedance --host 0.0.0.0     # 然后对方访问 http://<本机IP>:8770/
```

默认只绑 `127.0.0.1`（仅本机）是**有意为之**：这个服务**没有任何认证**，
谁连上都能改参数、重置场景。演示时再开 `--host 0.0.0.0`，不要长期挂着。

#### 原生窗口里几个容易误按的键

MuJoCo 自带一批可视化开关，按错了画面会突然变得很奇怪，**都不是报错**：

| 键 | 作用 | 按下去像什么 |
|---|---|---|
| `I` | 惯量盒 | 机械臂被一堆**红色半透明盒子**罩住，有的比本体还大 |
| `H` | 凸包 | 每个几何体外面套一层多面体 |
| `C` / `F` | 接触点 / 接触力 | 接触处冒出小点和箭头 |
| `T` | 透明 | 整机变半透明 |

再按一次就恢复。其中 `I` 那个红盒子最吓人，解释见
[`armctrl_调参台使用说明.md`](armctrl_调参台使用说明.md) 的「原生窗口按键」一节。

测试：

```bash
PYTHONPATH=. MUJOCO_GL=egl python -m unittest discover -s armctrl/tests -t .
```

⚠️ 没装 menagerie（或没设 `MUJOCO_MENAGERIE`）时，需要模型的用例会集中报
`ParseXML: Error opening file ...`。跑一条自检先确认环境：

```bash
PYTHONPATH=. python -c "from armctrl.assets import require_panda_xml; print(require_panda_xml())"
```

它要么打印模型路径，要么给出带安装指引的报错。

### 怎么确认这些测试不是摆设

```bash
PYTHONPATH=. MUJOCO_GL=egl python tools/verification/mut_e3_creep.py
```

把实现逐条改坏，看测试红不红——**一组永远不会红的测试，和没有测试是一回事**。
本项目用这个办法实际抓到过「名字很对、内容是空的」测试。
详见 [`tools/verification/README.md`](tools/verification/README.md)。

---

## 仿真录像

不想装环境的话，仓库里有 7 段 MuJoCo 录像可以直接看
（`armctrl/videos/`，都由 `armctrl/demos/` 里的脚本生成，可复现）：

| 录像 | 看什么 |
|---|---|
| `impedance.mp4` | 末端被推开后柔顺退让、松手弹回 |
| `stiffness.mp4` | 同样的外力，刚度不同 ⇒ 偏移量不同 |
| `collision.mp4` | 动量观测器检测到碰撞后主动退让 |
| `teaching.mp4` | 拖动示教：人拖着走，松手后复现 |
| `gripper.mp4` | 完整 pick & place |
| `gripper_close.mp4` | 夹持特写：指垫接触与夹持力建立 |
| `planning.mp4` | RRT-Connect 绕开障碍 + 平滑后的执行 |

自己重新生成（`MUJOCO_GL=egl` 用离屏渲染，无显示器也能录）：

```bash
# 前四个走通用录像入口，--scene 可选 impedance / stiffness / collision / teaching
MUJOCO_GL=egl PYTHONPATH=. python armctrl/demos/visualize.py --scene impedance

# 后三个由各自的 demo 带 --video
MUJOCO_GL=egl PYTHONPATH=. python armctrl/demos/demo_gripper.py --video
MUJOCO_GL=egl PYTHONPATH=. python armctrl/demos/demo_gripper.py --video \
    --camera grasp_close --out armctrl/videos/gripper_close.mp4
MUJOCO_GL=egl PYTHONPATH=. python armctrl/demos/demo_planning.py --video
```

---

## 模块

| 模块 | 内容 |
|---|---|
| `core/robot.py` | TCP 正运动学与**点雅可比**、M(q)、C q̇、g(q)、任务空间惯量 Λ、可操作度、条件数 |
| `core/gripper.py` | 平行二指夹爪：伺服增益与夹持力标定、两段式夹持、接触力与抓取检测 |
| `core/kinematics.py` | SO(3) 主值对数与**连续 lift**、数值 IK（伪逆 / DLS / 自适应 DLS）、零空间投影 |
| `planning/trajectory.py` | 五次多项式、七段式 S 曲线（jerk 有界）、四元数 slerp、直线与圆弧插补 |
| `planning/path.py` | RRT-Connect + 轨迹平滑 + TOPP-RA 时间最优重整 |
| `control/impedance.py` | 笛卡尔阻抗（Λ 自适应临界阻尼 + 零空间刚度）、导纳、混合力位（Raibert–Craig） |
| `control/momentum_observer.py` | 广义动量观测器：无传感器外力估计、碰撞检测、拖动示教 |
| `control/` 其余 | 计算力矩、自适应、ILC、MPC、模态振动抑制、视觉伺服 |
| `identification/` | τ = Y(q,q̇,q̈)π 回归矩阵、基参数集、傅里叶激励轨迹条件数优化、WLS |

完整的推导、实测数据与设计取舍见 [`armctrl/README.md`](armctrl/README.md)（341 行）。

---

## 一个设计特点：文档与代码的一致性有测试锁住

本项目另有一套配套技术讲义（14 篇，含推导与实测数据）。其中 8 个测试文件会**先用数值
独立测出源码的真实行为，再要求讲义里的说法与之一致**——纯字符串比对只能锁住「文案没被
改」，锁不住「文案说的对不对」。

这个机制是被迫加的：早期审核里同一类缺陷出现过五次，都是**代码里是对的、
讲义里是错的**（自适应符号、倒角预算、`f_meas` 信号方向、`vib_amp` 信号方向、
动量观测器延迟）。根因是那一层从来没有任何测试覆盖，改了代码没人提醒去改文案。

讲义本身混有个人学习笔记，不随本仓库分发，但**校验它的测试代码保留在仓库里**
（`armctrl/tests/` 中带 `@requires_docs` 的部分），讲义缺失时这些用例自动 skip
而不是失败，机制见 [`armctrl/tests/_docs.py`](armctrl/tests/_docs.py)。

---

## 许可

本仓库代码为 MIT，见 [LICENSE](LICENSE)。

⚠️ 机器人模型**不在本仓库内**：Franka Panda 来自 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Google DeepMind，Apache-2.0），需按上面「快速开始」自行获取。
