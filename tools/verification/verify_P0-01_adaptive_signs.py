"""P0-01 独立验收：Slotine-Li 控制律与自适应律的符号到底该取哪个。

独立性说明
----------
本脚本**不导入 armctrl 的任何控制或动力学代码**。
2 自由度平面臂的 M、C 与回归矩阵 Y 全部在本文件内按 Christoffel 公式手推，
并在开跑前用 `Y π = M q̈_r + C q̇_r` 自检通过，否则整个结论无效。

被测的两种符号约定（误差约定同为 e = q_d − q，s = ė + λe）
-----------------------------------------------------------
  CODE 版（armctrl/control/adaptive.py:115,118）：τ = Yπ̂ **+** K_d s，π̂̇ = **+**ΓYᵀs
  DOC  版（armctrl_讲解/03_自适应控制.md:177-179）：τ = Yπ̂ **−** K_d s，π̂̇ = **−**ΓYᵀs

判据（独立于两种实现）：Lyapunov 函数 V = ½sᵀM s + ½π̃ᵀΓ⁻¹π̃ 必须单调不增。
这同时就是**变异证明**：若两种符号都通过，说明本脚本是空测试，结论作废。
"""
import numpy as np

PI_TRUE = np.array([1.0, 0.4, 0.5])       # π = [a, b, c]，ab=0.40 > c²=0.25 ⇒ M 处处正定
LAM, KD, GAM = 5.0, 10.0, 20.0
DT, T_END = 2e-4, 12.0


def mass(q, p=PI_TRUE):
    a, b, c = p
    c2 = np.cos(q[1])
    return np.array([[a + b + 2 * c * c2, b + c * c2], [b + c * c2, b]])


def coriolis(q, v, p=PI_TRUE):
    c = p[2]
    s2 = np.sin(q[1])
    return np.array([[-c * s2 * v[1], -c * s2 * (v[0] + v[1])], [c * s2 * v[0], 0.0]])


def regressor(q, v, vr, ar):
    """Y 满足 Y π = M(q)·ar + C(q,v)·vr，对任意 π 线性。"""
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    return np.array([
        [ar[0], ar[0] + ar[1],
         2 * c2 * ar[0] + c2 * ar[1] - s2 * v[1] * vr[0] - s2 * (v[0] + v[1]) * vr[1]],
        [0.0,   ar[0] + ar[1],
         c2 * ar[0] + s2 * v[0] * vr[0]],
    ])


# ---------------------------------------------------------------- 回归矩阵自检
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(500):
    q, v, vr, ar = (rng.uniform(-3, 3, 2) for _ in range(4))
    lhs = regressor(q, v, vr, ar) @ PI_TRUE
    rhs = mass(q) @ ar + coriolis(q, v) @ vr
    worst = max(worst, np.abs(lhs - rhs).max())
print(f"回归矩阵自检  max|Yπ − (M·a_r + C·v_r)| = {worst:.3e}", "✅" if worst < 1e-12 else "❌ 作废")
assert worst < 1e-12, "回归矩阵不自洽，后续结论无效"


def q_des(t):
    return (np.array([0.5 * np.sin(t), 0.5 * np.cos(1.3 * t)]),
            np.array([0.5 * np.cos(t), -0.65 * np.sin(1.3 * t)]),
            np.array([-0.5 * np.sin(t), -0.845 * np.cos(1.3 * t)]))


def run(sign: int, label: str):
    """sign=+1 → CODE 版；sign=−1 → DOC 版。"""
    q = np.array([0.30, 0.80])
    v = np.zeros(2)
    pi_hat = np.zeros(3)
    Vs, s_norms, e_norms = [], [], []
    n = int(T_END / DT)
    for k in range(n):
        t = k * DT
        qd, vd, ad = q_des(t)
        e, de = qd - q, vd - v
        s = de + LAM * e
        vr, ar = vd + LAM * e, ad + LAM * de

        Y = regressor(q, v, vr, ar)
        tau = Y @ pi_hat + sign * KD * s          # ← 控制律符号
        pi_hat = pi_hat + sign * GAM * (Y.T @ s) * DT   # ← 自适应律符号

        pi_err = pi_hat - PI_TRUE
        Vs.append(0.5 * s @ mass(q) @ s + 0.5 * pi_err @ pi_err / GAM)
        s_norms.append(np.linalg.norm(s))
        e_norms.append(np.linalg.norm(e))

        acc = np.linalg.solve(mass(q), tau - coriolis(q, v) @ v)
        v = v + acc * DT
        q = q + v * DT
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) > 1e6:
            print(f"  [{label}] 第 {t:.3f}s 数值发散，提前终止")
            break

    V = np.array(Vs)
    rise = np.max(np.diff(V)) if len(V) > 1 else np.nan
    print(f"\n[{label}]  τ = Yπ̂ {'+' if sign > 0 else '−'} K_d·s ，"
          f"π̂̇ = {'+' if sign > 0 else '−'}ΓYᵀs")
    print(f"   V(0) = {V[0]:.6f}   V(末) = {V[-1]:.6e}   V 最大单步上升 = {rise:.3e}")
    print(f"   ‖s‖  首 {s_norms[0]:.4f} → 末 {s_norms[-1]:.6e}"
          f"   ；末 1s 跟踪误差 RMS = {np.sqrt(np.mean(np.square(e_norms[-int(1/DT):]))):.6e}")
    mono = rise <= 1e-9 * max(1.0, V[0])
    print(f"   V 单调不增？ {'✅ 是' if mono else '❌ 否'}    "
          f"收敛？ {'✅ 是' if np.isfinite(V[-1]) and V[-1] < 0.01 * V[0] else '❌ 否'}")
    return mono, V


print("\n" + "=" * 70)
ok_code, V_code = run(+1, "CODE 版")
ok_doc, V_doc = run(-1, "DOC 版")
print("\n" + "=" * 70)
print("变异证明（本脚本能否区分对错）：")
print(f"  CODE 版通过 = {ok_code} ，DOC 版通过 = {ok_doc}")
if ok_code and not ok_doc:
    print("  ✅ 判据能区分两种符号 —— 不是空测试。结论：**代码对，文档错**。")
else:
    print("  ❌ 判据无法区分，本脚本作废，结论不可采信。")
