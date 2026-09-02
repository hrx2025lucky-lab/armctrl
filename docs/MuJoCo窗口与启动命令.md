# MuJoCo 原生窗口：看板与启动命令

> 调参台跑起来会开**两个**界面：MuJoCo 原生窗口（3D 画面）和浏览器页面（讲解 + 滑块 + 读数）。
> 这一篇只讲**原生窗口**——它的看板怎么调出来、每个键干什么、以及各种启动命令。
>
> 本文所有快捷键都是从 `mujoco` 安装包的 `_simulate` 二进制里**直接读出来的**，
> 不是凭印象写的。想现场核对：窗口聚焦后按 <kbd>F1</kbd> 会弹出官方帮助。

---

## 一、启动命令速查

```bash
cd <仓库根>

./tuner.sh                       # 默认场景（阻抗控制）
./tuner.sh --list                # 列出全部 12 个场景
./tuner.sh grasp                 # 指定场景
./tuner.sh planning --port 8899  # 换端口
./tuner.sh impedance --host 0.0.0.0   # ⚠️ 允许别的机器访问，见 §五
```

`tuner.sh` 替你做三件事，所以不用手敲一长串环境变量：

1. 找解释器：`$ARMCTRL_PYTHON` → `<仓库>/.venv` → `<仓库同级>/envs/armctrl` → `python3`
2. 设 `PYTHONPATH` 与 `MUJOCO_GL=glfw`
3. 端口被占时**直接说清楚**，而不是让你对着 `Address already in use` 猜

指定解释器：

```bash
ARMCTRL_PYTHON=/path/to/venv/bin/python ./tuner.sh
```

### 12 个场景

| 场景 | 讲什么 |
|---|---|
| `impedance` | 笛卡尔阻抗控制 |
| `tracking` | PD+重力补偿 vs 计算力矩 |
| `adaptive` | 自适应控制：MBAC 与 RBF |
| `observer` | 动量观测器：不用力传感器估外力 |
| `estimation` | 状态估计：IMU 与运动学互补融合 |
| `attitude` | 姿态解算：陀螺、重力修正与偏航不可观 |
| `planning` | RRT-Connect 路径规划与避障 |
| `replan` | 动态避障：执行途中在线重规划 |
| `grasp` | 抓取：夹持力、滑移检测与力自适应 |
| `teaching` | 拖动示教 |
| `ilc` | 轨迹精度补偿：迭代学习控制 |
| `servo` | 伺服环路：谐振、陷波与输入整形 |

### ⚠️ 原生窗口需要显示器

调参台走 `glfw`，**无显示器的服务器上开不出来**。
测试和 demo 不受影响（它们不做渲染），只有调参台有这个要求。

不带界面跑东西：

```bash
export PYTHONPATH=.

python armctrl/demos/demo_impedance.py     # 阻抗控制三组实验
python armctrl/demos/demo_collision.py     # 动量观测器碰撞检测
python armctrl/demos/demo_gripper.py       # TCP vs 法兰、夹持力标定、pick & place

# 录像（离屏渲染，无显示器也能出片）
MUJOCO_GL=egl python armctrl/demos/visualize.py --scene impedance

# 测试：不要设 MUJOCO_GL
python -m unittest discover -s armctrl/tests -t .
```

> ⚠️ 跑测试时**不要**设 `MUJOCO_GL=egl`。测试里没有任何一处做渲染，
> 硬设这个变量，在没有 EGL 驱动的机器上会让**整个套件在 import 期就崩**
> （`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`）。
> 这个坑是 CI 第一次运行时踩到的。

---

## 二、⭐ 看板默认是关的，按 Tab 调出来

调参台启动时用的是：

```python
mujoco.viewer.launch_passive(..., show_left_ui=False, show_right_ui=False)
```

**两块看板都默认隐藏**——因为讲解和读数都在浏览器页面里，原生窗口留给画面。

| 想看什么 | 按键 |
|---|---|
| **左看板**（选项 / 仿真 / Watch / 物理 / 渲染） | <kbd>Tab</kbd> |
| **右看板**（Profiler 性能剖析 / Sensor 传感器曲线） | <kbd>Shift</kbd>+<kbd>Tab</kbd> |

### 左看板里有什么

| 分区 | 内容 |
|---|---|
| **File** | 存 XML / 存 mjb / 打印模型 / 打印数据 / 截图 / 退出 |
| **Option** | 帮助、信息、Profiler、Sensor、全屏、垂直同步、字号、配色 |
| **Simulation** | 暂停、重置、重载、对齐、拷贝状态、噪声、历史，以及 ⭐ **Watch** |
| **Physics** | 积分器（Euler / implicit / implicitfast）、摩擦锥（Pyramidal / Elliptic）、雅可比（Dense / Sparse）、求解器（Newton…）、时间步长 |
| **Rendering** | 相机选择、可视化开关、渲染开关、几何体分组显隐 |

### ⭐ Watch：直接盯住任意一个内部变量

`Simulation` 分区里的 **Watch** 有三栏：**Field / Index / Value**。
填上 `MjData` 的字段名和下标，就能实时看到它的值。

讲义里几处「看 `Watch → ncon`」「看 `qfrc_constraint`」指的就是这里。举例：

| Field | 看的是 |
|---|---|
| `ncon` | 当前接触点个数（悬空是 0，压到地板会变正） |
| `qfrc_constraint` | 约束力矩（`01_阻抗控制.md` §10 用它验证「地面帮忙扛了多少」） |
| `qpos` / `qvel` | 关节位置 / 速度 |
| `sensordata` | 传感器读数 |

> ⭐ Watch 是**只读**的，纯粹用来查看，不会影响仿真。

---

## 三、快捷键全表

以下按键从 `_simulate` 二进制里读出，与官方帮助（<kbd>F1</kbd>）一致。

### 仿真控制

| 键 | 作用 |
|---|---|
| <kbd>Space</kbd> | 播放 / 暂停 |
| <kbd>+</kbd> <kbd>-</kbd> | 加速 / 减速 |
| <kbd>←</kbd> <kbd>→</kbd> | 单步后退 / 前进 |

### 界面

| 键 | 作用 |
|---|---|
| <kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd> | 切换左 / 右看板 |
| <kbd>F1</kbd> | 帮助（**官方按键表，忘了就按它**） |
| <kbd>F2</kbd> | 信息（时间、FPS、接触数等） |
| <kbd>F3</kbd> | Profiler（各阶段耗时） |
| <kbd>F4</kbd> | Sensor（传感器曲线） |
| <kbd>F5</kbd> | 全屏 |

### 相机与鼠标

| 操作 | 作用 |
|---|---|
| 左键拖动 | 旋转视角 |
| <kbd>Shift</kbd>+右键拖动 | 平移视角 |
| 滚轮 / 中键拖动 | 缩放 |
| <kbd>[</kbd> <kbd>]</kbd> | 在模型自带相机之间循环 |
| 双击 | 选中物体 |
| 右键双击 | 把相机对准该点 |
| <kbd>Ctrl</kbd>+右键双击 | 跟踪相机（镜头跟着物体走） |
| <kbd>Page Up</kbd> | 选中父物体 |

### ⭐ 用鼠标直接推机械臂

| 操作 | 作用 |
|---|---|
| <kbd>Ctrl</kbd>+左键拖动 | 旋转被选中的物体（施加力矩） |
| <kbd>Ctrl</kbd>+右键拖动 | 平移被选中的物体（**施加外力**） |

这是阻抗、导纳、拖动示教这几个场景最直观的玩法：
先双击选中机械臂末端，然后 <kbd>Ctrl</kbd>+右键拖它，看它怎么退让、松手怎么弹回。

---

## 四、可视化开关：画面突然变奇怪不是报错

MuJoCo 自带一批可视化开关，**按错了再按一次就恢复**。

### 常用

| 键 | 开关 | 按下去像什么 |
|---|---|---|
| <kbd>I</kbd> | 惯量盒 | 一堆**红色半透明盒子**罩住机械臂，有的比本体还大（见 §六） |
| <kbd>H</kbd> | 凸包 | 每个几何体外套一层多面体 |
| <kbd>T</kbd> | 透明 | 整机变半透明，能看见内部 |
| <kbd>J</kbd> | 关节 | 画出每个关节的轴 |
| <kbd>U</kbd> | 执行器 | 显示驱动状态 |
| <kbd>M</kbd> | 质心 | 标出各连杆质心 |
| <kbd>W</kbd> | 线框 | 只画棱线 |
| <kbd>S</kbd> | 阴影 | 关掉能提帧率 |
| <kbd>E</kbd> | 等式约束 | 画出 equality 约束 |
| <kbd>A</kbd> | 自动连线 | 连杆之间画连线 |

### ⚠️ 两个被调参台**改了含义**的键

调参台注册了自己的按键回调，这两个键会**额外**触发调参台的动作：

| 键 | MuJoCo 原生 | 调参台额外做的事 |
|---|---|---|
| <kbd>C</kbd> | 接触点 | **切换全景 / 夹爪视角** |
| <kbd>V</kbd> | 肌腱 | **切换接触点与接触力标记** |

所以在调参台里：**想看接触点请按 <kbd>V</kbd>**（或点网页上的「接触点」按钮），
而不是 MuJoCo 原生的 <kbd>C</kbd>。

> 为什么接触点标记默认关：抓取场景夹爪合拢后 `ncon` 稳定在 **32**，
> 32 个橙色圆盘叠在两个指尖上会连成一片，把指垫、工件棱线、插入间隙全糊掉——
> 而这些几何细节才是抓取场景真正要看的。离屏渲染做过对照：
> 只开接触力时橙色像素占比 0.000%，只开接触点时 0.067%，
> **橙色 100% 来自接触点**，接触力箭头是淡青色，不参与遮挡。

---

## 五、让别的机器打开

```bash
./tuner.sh impedance --host 0.0.0.0
```

对方浏览器访问 `http://<本机IP>:8770/`。

> ⚠️ 默认只绑 `127.0.0.1`（仅本机）是**有意为之**：
> 这个服务**没有任何认证**，谁连上都能改参数、重置场景。
> 演示时再开 `--host 0.0.0.0`，**不要长期挂着**。

注意 3D 画面**不会**跟着走——原生窗口在跑服务的那台机器上。
远程的人看到的是浏览器页面（讲解 + 滑块 + 读数 + 曲线）。

### 端口被占了

`tuner.sh` 会直接告诉你怎么办：

```
端口 8770 已被占用——很可能上一次的调参台还开着。
先关掉那个 MuJoCo 窗口，或执行：
  kill $(ss -ltnp 2>/dev/null | grep ':8770 ' | grep -oP 'pid=\K[0-9]+' | head -1)
```

---

## 六、⭐ 那个红盒子（按 <kbd>I</kbd> 出来的）到底是什么

它叫**等效惯量盒**：把连杆换成一个**均匀密度的长方体**，
要求这个盒子的**质量**和**三个主惯量**跟真实连杆完全一样。

半边长由 $I_{xx}=\frac{m}{3}(b^2+c^2)$ 等三式反解：

$$
a=\sqrt{\tfrac{3}{2m}\left(I_{yy}+I_{zz}-I_{xx}\right)},\qquad \text{其余轮换}
$$

**为什么盒子比机械臂还大**：真实连杆是空壳，电机和减速器压在两端，
**质量离转轴很远**。实心均匀的盒子要凑出同样大的转动惯量，只能把体积撑大。

本模型（Panda）实测：

| 连杆 | 质量 | 等效盒半边长 (m) | 最长/最短 |
|---|---|---|---|
| link1 | 4.971 kg | [0.038 0.061 **0.650**] | 17 |
| link2 | 0.647 kg | [**0.002** 0.113 0.362] | **182** |
| link5 | 1.226 kg | [0.012 0.140 0.265] | 21 |

link1 的等效盒子有 **1.3 米长**，而连杆本身远没那么大。

**那几片「纸」**：link2 有一个方向只剩 **2 毫米**，被压成薄片——
因为它的惯量正好卡在**薄板极限** $I_z = I_x + I_y$（垂直轴定理）上。

### 顺带一个和参数辨识直接相关的检查

主惯量必须满足**三角不等式** $I_x + I_y \ge I_z$（三个轮换都要）。
违反它意味着**没有任何实体能有这种惯量**——物理上不可实现。

实测本模型 11 根连杆**全部满足**，最紧的是 link2，余量只有 `1.7e-6`（几乎贴边）。

> ⭐ 这条不是摆设：辨识（见 `armctrl/identification/`）辨出来的惯量参数**必须**过这一关。
> 最小二乘只保证「拟合误差小」，**不保证结果是个真实存在的物体**——
> 这正是「物理一致性约束」要解决的问题。

---

## 七、常见问题

| 现象 | 原因与办法 |
|---|---|
| 画面全是红色半透明盒子 | 误按了 <kbd>I</kbd>，再按一次 |
| 看不到任何面板 | 看板默认关，按 <kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd> |
| 忘了快捷键 | 按 <kbd>F1</kbd>，官方帮助就在窗口里 |
| `端口 8770 已被占用` | 上次的调参台还开着，按提示 kill 掉 |
| 服务器上开不出窗口 | 需要显示器（glfw）。测试和 demo 不受影响 |
| 一堆 `ParseXML: Error opening file` | 没装 Panda 模型。先跑：<br>`PYTHONPATH=. python -c "from armctrl.assets import require_panda_xml; print(require_panda_xml())"` |
| 想看接触点 | 按 <kbd>V</kbd>（不是 <kbd>C</kbd>），或点网页上的「接触点」按钮 |
| 关掉窗口后调参台还在跑？ | 不会。关掉原生窗口即退出整个调参台 |
