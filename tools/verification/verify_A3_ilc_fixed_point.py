"""A3 / P0-07 独立判据：ILC 的稳态误差到底是不是 0。

判据的独立性
------------
被测对象是 `IterativeLearningController.update()`（真实实现）。
判据**不读它的任何中间量**：只在外面搭一个标量被控对象
    e_k = (r-d) - P·u_k
把 `correction()` 拿到的 u 喂进去，自己算误差，再喂回 `update()`。
解析预测完全由手推递推式给出，与实现无关。

手推（lead=0）
--------------
    u_{k+1} = Q(u_k + L e_k),      e_k = (r-d) - P u_k
    ⇒ e_{k+1} = (I-Q)(r-d) + Q(I-PL) e_k
文档只写了 Q(I-LP)e_k，漏掉仿射项 (I-Q)(r-d)。

⭐ 取 L·P = 1 时 Q(I-PL) = 0，递推**一步到位**：
       e_1 = (I-Q)(r-d)，且 e_2 = e_1 = e_3 = ...
文档说「一遍学完，误差 = 0」；真值是「一遍之后卡在残差地板上，再也不动」。
这给出一个极干净、可判真伪的预测。

判别力
------
同一个判据对**直流** (r-d) 必须给出 ≈0（否则它是"永远报错"的假判据），
对**高频** (r-d) 必须给出显著非零。两种情形都跑。
"""
import numpy as np
from armctrl.control.ilc import IterativeLearningController, zero_phase_lowpass

N, ALPHA = 400, 0.15
np.set_printoptions(precision=6, suppress=True)


def run_ilc(rd, gain, plant, n_trials=40, alpha=ALPHA):
    """把真实 ILC 挂在标量被控对象上跑 n_trials 遍，返回每遍的误差范数。"""
    ilc = IterativeLearningController(N, 1, gain=gain, q_alpha=alpha,
                                      lead=0, limit=1e9)
    errs, last = [], None
    for _ in range(n_trials):
        u = ilc.correction()[:, 0]
        e = rd - plant * u
        errs.append(float(np.sqrt(np.mean(e ** 2))))
        last = e
        ilc.update(e.reshape(N, 1))
    return np.array(errs), last


t = np.arange(N)
cases = {
    "直流   (r-d = 1)":            np.ones(N),
    "低频   (2 周期正弦)":          np.sin(2 * np.pi * 2 * t / N),
    "高频   (150 周期正弦)":        np.sin(2 * np.pi * 150 * t / N),
    "奈奎斯特 (逐点交替 ±1)":       (-1.0) ** t,
}

print("=" * 74)
print("① L·P = 1：文档预测「一遍学完，误差 → 0」")
print("=" * 74)
print(f"{'(r-d) 成分':<26}{'e0':>9}{'e1':>10}{'e10':>10}{'e39':>10}"
      f"{'解析 ‖(I-Q)(r-d)‖':>20}")
for name, rd in cases.items():
    errs, _ = run_ilc(rd, gain=1.0, plant=1.0)
    pred = float(np.sqrt(np.mean((rd - zero_phase_lowpass(
        rd.reshape(N, 1), ALPHA)[:, 0]) ** 2)))
    print(f"{name:<26}{errs[0]:>9.4f}{errs[1]:>10.6f}{errs[10]:>10.6f}"
          f"{errs[39]:>10.6f}{pred:>20.6f}")

print()
print("  判读：")
print("   • e1 == e10 == e39  ⇒ 一遍之后就**不动了**，不是「衰减到 0」")
print("   • e1 与解析 ‖(I-Q)(r-d)‖ 逐例吻合 ⇒ 仿射项 (I-Q)(r-d) 确实存在")
print("   • 直流一栏 ≈ 0 ⇒ 判据有判别力，不是恒报错")

print()
print("=" * 74)
print("② Q 的直流增益——决定「残差地板」落在哪个频段")
print("=" * 74)
for a in (0.05, 0.15, 0.5):
    dc = zero_phase_lowpass(np.ones((N, 1)), a)[:, 0]
    hf = zero_phase_lowpass(((-1.0) ** t).reshape(N, 1), a)[:, 0]
    print(f"  q_alpha={a:<5}  直流增益 = {dc.mean():.12f}（偏离 1 的最大值 "
          f"{np.abs(dc - 1).max():.2e}）   奈奎斯特保留 {np.abs(hf).mean()*100:6.3f}%")
print("  ⇒ Q 的直流增益**恒等于 1** ⇒ (I-Q) 把直流**完全抹掉**")
print("  ⇒ 低频可重复误差是**学得干净的**；残差地板 = Q 滤掉的那段频带，"
      "与 alpha 直接挂钩。")

print()
print("=" * 74)
print("③ 复现审核报告的验收条件：抽象标量 Q=0.5, P=L=1, r-d=1, e0=1 ⇒ e1=0.5")
print("=" * 74)
Q, P, L, rd0 = 0.5, 1.0, 1.0, 1.0
e = 1.0
seq = [e]
for _ in range(6):
    e = (1 - Q) * rd0 + Q * (1 - P * L) * e
    seq.append(e)
print("  e_k =", np.round(seq, 6))
print(f"  e1 = {seq[1]:.4f}  （报告要求 0.5 → "
      f"{'✅ 通过' if abs(seq[1]-0.5) < 1e-12 else '❌'}）"
      f"；文档递推 Q(I-LP)e_k 预测 e1 = 0.0000  ❌")
print(f"  解析不动点 e_∞ = (1-Q)(r-d)/[1-Q(1-PL)] = "
      f"{(1-Q)*rd0/(1-Q*(1-P*L)):.4f}，实测 e6 = {seq[-1]:.4f}")
print()
print("  ⚠️ 但注意：本仓库的 Q 是**零相位低通、直流增益为 1**，")
print("     所以对直流 (r-d) 并不会出现 0.5 这种残差——报告的标量反例")
print("     成立于抽象 Q=0.5，不能直接套到本实现上。见 ① 直流一栏。")

print()
print("=" * 74)
print("④ 收缩因子只管齐次部分：L 变化时 e_∞ 怎么动")
print("=" * 74)
rd = cases["高频   (150 周期正弦)"]
print(f"{'L':>6}{'|1-LP| (文档判据)':>20}{'实测 e39':>12}{'解析不动点':>14}")
for L in (0.2, 0.6, 1.0, 1.6):
    errs, _ = run_ilc(rd, gain=L, plant=1.0, n_trials=60)
    iq = rd - zero_phase_lowpass(rd.reshape(N, 1), ALPHA)[:, 0]
    # 逐点解析不动点：e = (I-Q)(r-d) + Q(I-PL)e，用算子迭代求解
    ef = np.zeros(N)
    for _ in range(3000):
        ef = iq + zero_phase_lowpass(((1 - L) * ef).reshape(N, 1), ALPHA)[:, 0]
    print(f"{L:>6.1f}{abs(1-L):>20.2f}{errs[-1]:>12.6f}"
          f"{float(np.sqrt(np.mean(ef**2))):>14.6f}")
print("  ⇒ |1-LP| 描述的是**收敛速度**；终点由仿射项决定，二者必须分开讲。")
