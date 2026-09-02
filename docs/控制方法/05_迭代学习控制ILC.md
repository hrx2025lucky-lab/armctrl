# 迭代学习控制ILC

> 一句话：利用同一轨迹每次重复时相似的误差，把上一遍的偏差提前补到下一遍参考中。

## 一、要解决的问题

机械臂重复执行同一条轨迹时，误差往往不是随机的。

例如固定节拍搬运：

- 某个拐角总是慢半拍；
- 某段高速运动总是向同一侧偏；
- 负载力矩导致关节柔性弯曲；
- 齿隙、摩擦和热漂移带来可重复偏差；
- 提高反馈增益会改善误差，但可能激发振动。

这些误差未必容易建模。

但只要轨迹、负载、速度曲线基本相同，它们就可以被“记住”。

ILC 的工程直觉很简单：这一遍哪里偏了，下一遍在对应时间提前补上。

它特别适合：

- 产线固定动作；
- 重复扫描；
- 重复喷涂；
- 重复装配轨迹；
- 仿真中验证补偿算法的收敛性。

它不适合一次性随机任务。

换轨迹后，学到的时间序列通常不能直接复用。

## 二、原理

令第 $k$ 遍的前馈修正为 $u_k(t)$。

关节空间误差为

$$
e_k(t)=q_d(t)-q_k(t)
$$

P 型 ILC 更新律为

$$
u_{k+1}(t)=Q\left[u_k(t)+L e_k(t+\Delta)\right]
$$

其中：

- $L$ 是学习增益；
- $\Delta$ 是时间超前；
- $Q$ 是低通滤波器；
- $u_k$ 会加到下一遍参考轨迹上。

若闭环对象在重复轨迹附近近似为线性算子 $P$，误差递推可写成

$$
e_{k+1}=Q(I-LP)e_k+(I-Q)d
$$

其中 $d$ 是不可学习或被滤掉的误差成分。

收敛核心条件为

$$
\rho(Q(I-LP))<1
$$

标量情形下，理论收缩因子是

$$
|1-LP|
$$

因此：

- $LP=1$ 时最快；
- $LP=2$ 是临界；
- $LP>2$ 会发散；
- 高频相位滞后会让 $I-LP$ 变坏。

这就是 Q-filter 必须存在的原因。

它放弃学习高频误差，换取稳定收敛。

本项目的 `zero_phase_lowpass` 用前向和反向各滤一次。

因为 ILC 是离线逐遍更新，整段误差已经记录下来，可以使用非因果零相位滤波。

这样不会额外引入时间延迟。

关节柔度补偿是可建模部分：

$$
\Delta q=\frac{\tau_{load}}{k_{joint}}
$$

前馈补偿为

$$
q_{cmd}=q_{ref}+\Delta q
$$

符号要和负载弯曲方向一致，否则误差会变大。

## 三、怎么用在机械臂上

ILC 位于参考轨迹修正层。

它不直接输出力矩。

输入：

- 一遍轨迹的采样点数 `n_steps`；
- 关节数 `n_joints`；
- 上一遍的关节空间误差 `error`，形状为 `(n_steps, n_joints)`；
- 可选负载力矩，用于柔度前馈。

输出：

- `correction()` 返回下一遍要加到参考上的关节修正；
- `update(error)` 返回本遍 RMS，并更新内部修正表。

典型流程：

```python
ilc = IterativeLearningController(n_steps, 7)
for k in range(n_trials):
    q_cmd = q_ref + ilc.correction()
    error = run_one_trial(q_cmd)
    rms = ilc.update(error)
```

上游是固定参考轨迹。

下游可以是 PD、计算力矩、MPC 或阻抗控制器。

参数整定：

- `gain` 从小到大调，不要认为越大越好；
- `lead` 用来补偿执行滞后，过小会错位，过大会提前过头；
- `q_alpha` 越小低通越强，越稳但残差越大；
- `limit` 防止错误学习把轨迹推飞；
- `n_steps` 必须和误差记录长度一致；
- 每次换轨迹、换速度、换负载都应重新学习或至少重置验证。

关节柔度模型位于 ILC 之前。

可先用 `JointComplianceModel.compensate` 抵消线性弯曲，再让 ILC 学剩余重复误差。

工程陷阱：

- `error` 的定义是参考减实际，不要反号；
- `lead` 用 `np.roll` 后末段不能绕回，代码用最后误差填充；
- Q-filter 截止频率以上的误差学不掉；
- 反馈控制器不稳定时，ILC 不会神奇修好底层闭环；
- 学习得到的是时间序列，不是物理参数；
- 噪声大时应降低 `gain` 或增强低通。

对应调参台场景：`ilc`。

## 四、代码在哪

| 文件 | 关键函数/类 | 作用 |
|---|---|---|
| `armctrl/control/ilc.py` | `JointComplianceModel` | 线性关节柔度模型 |
| `armctrl/control/ilc.py` | `JointComplianceModel.deflection` | 由负载力矩计算关节弯曲角 |
| `armctrl/control/ilc.py` | `JointComplianceModel.compensate` | 对参考关节角做柔度前馈补偿 |
| `armctrl/control/ilc.py` | `zero_phase_lowpass` | 前向/反向零相位一阶低通 |
| `armctrl/control/ilc.py` | `IterativeLearningController` | P 型 ILC 控制器 |
| `armctrl/control/ilc.py` | `IterativeLearningController.reset` | 清空修正表和历史 RMS |
| `armctrl/control/ilc.py` | `IterativeLearningController.correction` | 返回当前前馈修正表 |
| `armctrl/control/ilc.py` | `IterativeLearningController.update` | 根据上一遍误差学习下一遍修正 |
| `armctrl/control/ilc.py` | `IterativeLearningController.convergence_ratio` | 返回末遍 RMS 相对首遍 RMS 的比例 |
| `armctrl/control/ilc.py` | `IterativeLearningController.contraction_factor` | 计算标量理论收缩因子 |
| `armctrl/tuner/scenes.py` | `IlcScene` | 交互式 ILC 调参场景 |

## 五、怎么验证它是对的

相关测试：

- `armctrl/tests/test_ilc_replan.py`。

可运行命令：

```bash
PYTHONPATH=. python -m unittest armctrl.tests.test_ilc_replan 2>&1 | grep -v Polishing
```


测试判据覆盖：

- 柔度补偿方向正确；
- 补偿后误差小于不补偿；
- 零相位低通不引入整体相位滞后；
- ILC 多遍更新后 RMS 下降；
- 学习增益过大时理论收缩因子提示发散风险；
- 动态重规划相关测试确认轨迹变化时不应复用旧补偿。

代码注释中有一个可复核示例：在标量对象、`q_alpha=0.15` 时，高频 150 周期分量从 `0.7071` 到 `0.702098`，只学掉约 `0.7%`。


## 六、边界与局限

这是仿真中的重复轨迹学习，没有真机长周期漂移验证。

局限包括：

- 必须重复同一条轨迹；
- 轨迹、速度、负载或接触条件改变后需要重新学习；
- 不学习 Q-filter 截止频率以上的误差；
- 学到的修正不可解释为质量或摩擦参数；
- 底层闭环本身要稳定；
- 噪声和非重复扰动会被错误记忆；
- 柔度模型只是一阶线性刚度，不含齿隙、迟滞和温度效应；
- 真机上还要考虑安全限幅、示教误差和生产节拍变化。
