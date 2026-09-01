# armctrl — 机械臂运动规划与柔顺控制栈

以 **Franka Panda**（7 自由度冗余臂）为对象，在 MuJoCo 中实现从运动学、轨迹规划
到力矩级柔顺控制的完整链路。每个模块都配可复现的定量验证实验，**683 个测试全部通过**。

```
armctrl_project/
├── armctrl/                      实现 + 26 个测试文件
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

pip install mujoco pin numpy scipy matplotlib toppra osqp
```

`armctrl/assets.py` 按 `MUJOCO_MENAGERIE` → 若干常见位置 → 兜底候选的顺序解析，
放在主目录或仓库同级时无需任何配置。

```bash
export PYTHONPATH=.

python armctrl/demos/demo_impedance.py    # 阻抗控制三组实验
python armctrl/demos/demo_collision.py    # 动量观测器碰撞检测
python armctrl/demos/demo_gripper.py      # TCP vs 法兰、夹持力标定、pick & place

./tuner.sh --list                         # 交互式调参台
./tuner.sh impedance
```

测试：

```bash
PYTHONPATH=. python -m unittest discover -s armctrl/tests -t .
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
