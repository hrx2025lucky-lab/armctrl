# 模型预测控制MPC

> 一句话：把未来若干步的跟踪目标和执行器/关节约束一起放进在线 QP，提前避开饱和和限位。

## 一、要解决的问题

普通 PD、计算力矩、阻抗和自适应控制都是即时反馈律。

它们根据当前误差直接给出力矩。

问题在于机械臂有硬约束：

- 关节力矩上限；
- 关节位置限位；
- 关节速度上限；
- 接近限位时必须提前减速；
- 负载变化时同一加速度需要不同力矩。

如果控制器先算一个力矩，再在最后简单裁剪，闭环设计就被破坏。

裁剪期间，控制器以为自己发出了大力矩，实际机器人没有执行。

这会导致：

- 轨迹拐弯处滞后；
- 接近限位时仍继续加速；
- 速度过大时制动不及时；
- 饱和后积分或学习项得到错误反馈；
- 约束违反只在发生后才被发现。

MPC 的思想是先预测未来，再决定当前动作。

它每个控制周期求解一次优化问题，只执行第一步，然后下一周期重算。

## 二、原理

完整动力学为

$$
M(q)\ddot q+h(q,\dot q)=\tau
$$

本项目采用反馈线性化：

$$
\tau=M(q)a+h(q,\dot q)
$$

代入后得到

$$
\ddot q=a
$$

于是在线优化只需处理双积分器。

离散化为

$$
q_{k+1}=q_k+\dot q_k\Delta t+\frac12a_k\Delta t^2
$$

$$
\dot q_{k+1}=\dot q_k+a_k\Delta t
$$

状态为

$$
x_k=\begin{bmatrix}q_k\\\dot q_k\end{bmatrix}
$$

预测时域长度为 $N$。

将所有未来状态写成初始状态和输入序列 $U=[a_0,\dots,a_{N-1}]$：

$$
Q=S_qx_0+T_qU
$$

$$
V=S_vx_0+T_vU
$$

代价函数为

$$
J=\sum_{k=1}^N\|q_k-q_{d,k}\|_Q^2+\sum_{k=1}^N\|\dot q_k-\dot q_{d,k}\|_R^2
$$

$$
+\sum_{k=0}^{N-1}\|a_k-a_{d,k}\|_S^2+\sum_{k=0}^{N-1}\|a_k-a_{k-1}\|_{S_r}^2
$$

力矩约束在当前周期冻结 $M,h$：

$$
-\tau_{lim}\le M(q)a_k+h(q,\dot q)\le\tau_{lim}
$$

这是关于 $a_k$ 的线性不等式。

关节位置和速度约束为

$$
q_{min}\le q_k\le q_{max}
$$

$$
-\dot q_{lim}\le\dot q_k\le\dot q_{lim}
$$

若当前状态已经越限，硬状态约束会使 QP 不可行。

代码使用松弛变量 $s\ge0$ 软化状态约束，并加入精确罚函数：

$$
\rho_{lin}\sum s+\rho_{quad}\|s\|^2
$$

但力矩约束仍是硬约束。

求解失败时，控制器用制动兜底：

$$
a_0=-8\dot q
$$

随后仍将最终力矩裁剪到上限内，并反算一致的 $a_0$。

## 三、怎么用在机械臂上

MPC 位于关节空间轨迹跟踪层。

输入：

- 当前 `q`；
- 当前 `v`；
- 未来 `N` 步 `q_ref_seq`；
- 未来 `N` 步 `v_ref_seq`；
- 未来 `N` 步 `a_ref_seq`；
- 当前构型质量矩阵 `M`；
- 当前偏置力 `h`。

输出：

- `tau`：最终执行力矩；
- `a0`：与最终力矩一致的第一步加速度；
- `info`：求解状态、松弛量、是否裁剪、越限量等诊断。

上游通常是轨迹生成器。

下游是力矩接口。

参数整定：

- `horizon` 决定向前看多远；
- `dt` 可以大于仿真步长，以减小 QP 规模；
- `w_pos` 提高位置跟踪；
- `w_vel` 抑制速度误差；
- `w_acc` 降低加速度幅值；
- `w_rate` 抑制控制抖动；
- `tau_margin < 1` 可为真实驱动器保留裕度；
- `soft_state=True` 是接近限位时的安全默认值。

工程陷阱：

- 传入的 `M,h` 必须和当前 `q,v` 对应；
- 预测中冻结 `M,h`，所以时域不能过长；
- 参考序列长度必须正好是 `(N,n)`；
- OSQP 失败时不能退回 `a_ref`，因为参考可能正在把关节推向限位；
- `info["saturated"]` 表示裁剪前越限，`info["tau_clipped"]` 表示实际触发有意义裁剪；
- 成功求解时 `torque_clips` 应保持 0；
- 若松弛量长期非零，说明轨迹或限位约束本身不可达。

MPC 没有专门调参台场景，但可和轨迹跟踪/规划场景一起理解。

## 四、代码在哪

| 文件 | 关键函数/类 | 作用 |
|---|---|---|
| `armctrl/control/mpc.py` | `JointSpaceMPC` | 反馈线性化加线性 MPC 的关节空间控制器 |
| `armctrl/control/mpc.py` | `JointSpaceMPC._build_prediction_matrices` | 构造双积分器预测矩阵 |
| `armctrl/control/mpc.py` | `JointSpaceMPC._build_cost` | 构造二次代价矩阵 |
| `armctrl/control/mpc.py` | `JointSpaceMPC._slack_expand` | 将每关节松弛量展开到预测时域 |
| `armctrl/control/mpc.py` | `JointSpaceMPC._constraint_matrix` | 构造 QP 约束矩阵 |
| `armctrl/control/mpc.py` | `JointSpaceMPC._constraint_bounds` | 构造每周期约束上下界 |
| `armctrl/control/mpc.py` | `JointSpaceMPC.compute` | 求解 QP 并返回力矩、加速度和诊断信息 |
| `armctrl/control/mpc.py` | `JointSpaceMPC.reset` | 清空求解器、前一加速度和诊断计数 |
| `armctrl/demos/demo_mpc.py` | 脚本文件 | MPC 演示入口 |
| `armctrl/demos/demo_mpc_limits.py` | 脚本文件 | 限位相关 MPC 演示入口 |

## 五、怎么验证它是对的

相关测试：

- `armctrl/tests/test_mpc.py`。

可运行命令：

```bash
cd /home/limx/workspace/Roxan_warmup/armctrl_project && PYTHONPATH=. MUJOCO_MENAGERIE=/home/limx/workspace/Roxan_warmup/repos/mujoco_menagerie /home/limx/workspace/Roxan_warmup/envs/armctrl/bin/python -m unittest armctrl.tests.test_mpc 2>&1 | grep -v Polishing
```

本次核对的真实符号：

- `JointSpaceMPC.compute`；
- `JointSpaceMPC.reset`；
- `JointSpaceMPC._constraint_matrix`；
- `JointSpaceMPC._constraint_bounds`。

测试判据包括：

- 无约束 QP 解与解析正规方程解一致；
- 力矩约束违反量小于测试阈值；
- QP 不可行时使用 `a0=-8*v` 制动兜底；
- 兜底后最终 `tau` 仍满足力矩上限；
- 裁剪后返回的 `a0` 与 `M @ a0 + h == tau` 一致；
- 软状态约束可避免已越限状态导致的硬不可行；
- 成功求解时不应出现有意义的最终力矩裁剪。

代码注释中保留了一个回归案例：旧兜底路径曾在 Panda 关节 2 返回 152.7 N·m，而上限是 87 N·m，约为 1.76 倍越限。

本次综合验证实测：按项目指定前缀运行 13 个相关测试模块，`unittest` 输出 `Ran 349 tests in 418.719s`，结果 `OK`。

## 六、边界与局限

这是仿真 MPC，没有真机实时性和驱动器接口验证。

局限包括：

- 不是完整非线性 MPC；
- 预测时域内冻结 `M,h`，构型变化剧烈时会失准；
- 稠密 QP 适合 7 轴和短时域，规模更大时需稀疏或结构化求解；
- 状态约束是软约束，物理安全仍应由上层规划和硬限位保证；
- 力矩上限是模型中的限制，不代表真实驱动器热限制；
- OSQP 可用性依赖环境安装；
- 没有考虑接触约束、视觉延迟和摩擦不确定性；
- 真实系统中还要处理求解超时、热降额和通信丢包。
