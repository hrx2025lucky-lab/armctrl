"""S 曲线速度规划的数学不变量测试。

覆盖五类工况：
    A 达到 v_max 且达到 a_max（有匀速段 + 匀加速段）        cruise_amax
    B 达到 v_max 但达不到 a_max（有匀速段，加速段纯 jerk）   cruise_jerk
    C 无匀速但达到 a_max（三角形工况 + 匀加速段）            tri_amax
    D 纯 jerk 三角形（无匀速、无匀加速）                     tri_jerk
    E 退化：零距离、极小距离、负距离                        degenerate

oracle 独立性（第三轮审查的整改点）
----------------------------------
上一版的 `classify()` 直接读取被测对象的 `p.Tj / p.Ta / p.Tv` 判断分支归属，
这是**用被测实现给自己判卷**。本版所有 oracle 只用输入 `(d, v_max, a_max, j_max)`：

  1. `predict_plan()`   由输入闭式推出 Tj/Ta/Tv/v_peak 与分支名（独立实现）。
  2. `reference_state()` 由 predict_plan 的时间表**逐段积分 jerk**，得到独立的
     s/v/a 参考轨迹；与被测实现的分段闭式代码是两条不同的计算路径。
  3. `classify_from_signals()` 只看采样到的 v(t)/a(t) 波形判断有没有匀速段、
     有没有匀加速段——完全不碰任何内部量。

连续性判据也改为**只依赖输入**的 Lipschitz 界：
    |Δs| ≤ v_max·h,   |Δv| ≤ a_max·h,   |Δa| ≤ j_max·h
在稠密均匀网格上检查，任何超过该界的相邻样本差就是跳变。
"""

from __future__ import annotations

import unittest

import numpy as np

from armctrl.planning.trajectory import SCurveProfile


TOL_POS = 1e-9
TOL_VEL = 1e-7
TOL_ACC = 1e-5

BRANCHES = {"cruise_amax", "cruise_jerk", "tri_amax", "tri_jerk", "degenerate"}


# --------------------------------------------------------------- 独立 oracle


def predict_plan(d, v, a, j):
    """**只由输入**推出七段时间表与分支名。独立于被测实现。

    加速半程 +J / 0 / −J，时长 Tj / (Ta−2Tj) / Tj。逐段积分：
        a_pk = J·Tj,  v_pk = J·Tj·(Ta−Tj),  s_acc = ½·v_pk·Ta
    全程 d = v_pk·(Ta+Tv), T = 2Ta+Tv。
    存在匀加速段 ⟺ Ta ≥ 2Tj ⟺ v/a ≥ a/j ⟺ v·j ≥ a²。
    """
    D = abs(float(d))
    if D < 1e-12:
        return dict(branch="degenerate", Tj=0.0, Ta=0.0, Tv=0.0, T=0.0,
                    v_peak=0.0, a_peak=0.0, d=0.0)
    if v * j >= a * a:
        Tj = a / j
        Ta = Tj + v / a
    else:
        Tj = np.sqrt(v / j)
        Ta = 2.0 * Tj
    d_acc = v * Ta
    if D >= d_acc:
        Tv = (D - d_acc) / v
        v_peak = v
        branch = "cruise_amax" if v * j >= a * a else "cruise_jerk"
    else:
        Tv = 0.0
        if D * j * j >= 2.0 * a**3:
            Tj = a / j
            Ta = 0.5 * (Tj + np.sqrt(Tj * Tj + 4.0 * D / a))
            v_peak = a * (Ta - Tj)
            branch = "tri_amax"
        else:
            Tj = (D / (2.0 * j)) ** (1.0 / 3.0)
            Ta = 2.0 * Tj
            v_peak = j * Tj * Tj
            branch = "tri_jerk"
    return dict(branch=branch, Tj=Tj, Ta=Ta, Tv=Tv, T=2.0 * Ta + Tv,
                v_peak=v_peak, a_peak=j * Tj, d=D)


def _segments(plan, j):
    """七段 (jerk, 时长)。中间的匀加速/匀减速段时长可以为 0。"""
    Tj, Ta, Tv = plan["Tj"], plan["Ta"], plan["Tv"]
    tc = max(Ta - 2.0 * Tj, 0.0)
    return [(+j, Tj), (0.0, tc), (-j, Tj), (0.0, Tv),
            (-j, Tj), (0.0, tc), (+j, Tj)]


def reference_state(plan, j, sign, t):
    """**独立参考实现**：按 jerk 分段精确积分，返回 (s, v, a)。

    与被测实现的"加速段闭式 + 时间镜像"是完全不同的计算路径。
    """
    if plan["T"] <= 0.0:
        return 0.0, 0.0, 0.0
    t = float(np.clip(t, 0.0, plan["T"]))
    s = vv = aa = 0.0
    for jk, dur in _segments(plan, j):
        if dur <= 0.0:
            continue
        if t <= dur:
            s = s + vv * t + 0.5 * aa * t * t + jk * t**3 / 6.0
            vv = vv + aa * t + 0.5 * jk * t * t
            aa = aa + jk * t
            return sign * s, sign * vv, sign * aa
        s = s + vv * dur + 0.5 * aa * dur * dur + jk * dur**3 / 6.0
        vv = vv + aa * dur + 0.5 * jk * dur * dur
        aa = aa + jk * dur
        t -= dur
    return sign * s, sign * vv, sign * aa


def classify_from_signals(p, d, v, a, j):
    """**只看采样波形**判断分支：有没有匀速段、加速段里有没有匀加速平台。

    采样窗口也用输入闭式预测的总时长，因此本函数除了调用 p(t) 取值以外，
    不读取被测对象的任何属性。
    """
    T = predict_plan(d, v, a, j)["T"]
    if T <= 0.0:
        return "degenerate"
    n = 20001
    ts = np.linspace(0.0, T, n)
    V = np.array([p(t)[1] for t in ts])
    A = np.array([p(t)[2] for t in ts])
    h = float(ts[1] - ts[0])
    vmax = float(np.abs(V).max())
    amax = float(np.abs(A).max())
    # 匀速段：|v| 贴住峰值且 a≈0 的持续时间明显超过一个采样步
    cruise = np.sum((np.abs(V) > vmax * (1 - 1e-9)) & (np.abs(A) < 1e-9 * max(amax, 1.0)))
    has_cruise = cruise * h > 5.0 * h
    # 匀加速平台：只在前半程（加速段）找 |a| 贴住峰值的持续时间
    half = ts <= 0.5 * T
    plateau = np.sum(half & (np.abs(A) > amax * (1 - 1e-9)))
    has_const_acc = plateau * h > 5.0 * h
    if has_cruise:
        return "cruise_amax" if has_const_acc else "cruise_jerk"
    return "tri_amax" if has_const_acc else "tri_jerk"


# --------------------------------------------------------------- 不变量检查


def check_profile(d, v, a, j, name, n=40001):
    """返回 (profile, 失败列表)。所有判据只依赖输入 (d, v, a, j)。"""
    p = SCurveProfile(d, v, a, j)
    plan = predict_plan(d, v, a, j)
    sign = 1.0 if d >= 0 else -1.0
    fails = []

    if plan["branch"] == "degenerate":
        if p.T > 0.0:
            fails.append(f"零距离应给出 T=0，实际 {p.T}")
        for t in (0.0, 1.0):
            s, sv, sa = p(t)
            if abs(s) + abs(sv) + abs(sa) > 0:
                fails.append(f"零距离应恒为 0，实际 {s},{sv},{sa}")
        return p, fails

    T = plan["T"]
    ts = np.linspace(0.0, T, n)
    h = float(ts[1] - ts[0])
    S = np.empty(n)
    V = np.empty(n)
    A = np.empty(n)
    Sr = np.empty(n)
    Vr = np.empty(n)
    Ar = np.empty(n)
    for i, t in enumerate(ts):
        S[i], V[i], A[i] = p(t)
        Sr[i], Vr[i], Ar[i] = reference_state(plan, j, sign, t)

    # --- 起终点（输入口径）---
    if abs(S[0]) > TOL_POS:
        fails.append(f"s(0)={S[0]:.3e} 应为 0")
    if abs(V[0]) > TOL_VEL or abs(A[0]) > TOL_ACC:
        fails.append(f"v(0)={V[0]:.3e}, a(0)={A[0]:.3e} 应为 0")
    if abs(S[-1] - d) > max(TOL_POS, 1e-9 * abs(d)):
        fails.append(f"s(T)={S[-1]:.9f} 应为 {d:.9f}，误差 {abs(S[-1]-d):.3e}")
    if abs(V[-1]) > TOL_VEL or abs(A[-1]) > TOL_ACC:
        fails.append(f"v(T)={V[-1]:.3e}, a(T)={A[-1]:.3e} 应为 0")

    # --- 与独立逐段积分参考一致 ---
    e_s, e_v, e_a = (np.abs(S - Sr).max(), np.abs(V - Vr).max(), np.abs(A - Ar).max())
    if e_s > max(1e-9, 1e-9 * abs(d)):
        fails.append(f"位移与独立积分参考不符，最大 {e_s:.3e}")
    if e_v > 1e-9 * max(v, 1.0):
        fails.append(f"速度与独立积分参考不符，最大 {e_v:.3e}")
    if e_a > 1e-8 * max(a, 1.0):
        fails.append(f"加速度与独立积分参考不符，最大 {e_a:.3e}")

    # --- 连续性：只用输入的 Lipschitz 界（不读任何内部分段时刻）---
    ds = np.abs(np.diff(S)).max()
    dv = np.abs(np.diff(V)).max()
    da = np.abs(np.diff(A)).max()
    if ds > v * h * (1 + 1e-6) + 1e-12:
        fails.append(f"位置跳变 {ds:.3e} > v_max·h = {v*h:.3e}")
    if dv > a * h * (1 + 1e-6) + 1e-12:
        fails.append(f"速度跳变 {dv:.3e} > a_max·h = {a*h:.3e}")
    if da > j * h * (1 + 1e-6) + 1e-9:
        fails.append(f"加速度跳变 {da:.3e} > j_max·h = {j*h:.3e}")

    # --- 约束（输入口径）---
    if np.abs(V).max() > v * (1 + 1e-6) + 1e-9:
        fails.append(f"|v|max={np.abs(V).max():.9f} > v_max={v}")
    if np.abs(A).max() > a * (1 + 1e-6) + 1e-9:
        fails.append(f"|a|max={np.abs(A).max():.9f} > a_max={a}")
    jerk = np.diff(A) / h
    if np.abs(jerk).max() > j * (1 + 1e-3) + 1e-6:
        fails.append(f"|jerk|max={np.abs(jerk).max():.6f} > j_max={j}")

    # --- 峰值与输入预测一致（在**预测的**时刻精确取值，不读 p.v_peak/p.a_peak）---
    # 峰值只在一个瞬时取到，均匀网格可能取不中；用 predict_plan 给出的时刻直接求值。
    v_at_Ta = p(plan["Ta"])[1]
    if abs(abs(v_at_Ta) - plan["v_peak"]) > 1e-7 * max(plan["v_peak"], 1.0) + 1e-12:
        fails.append(f"v(Ta)={v_at_Ta:.9f} != 预测峰值 {plan['v_peak']:.9f}")
    a_at_Tj = p(plan["Tj"])[2]
    a_pred = min(plan["a_peak"], a)
    if abs(abs(a_at_Tj) - a_pred) > 1e-7 * max(a_pred, 1.0) + 1e-12:
        fails.append(f"a(Tj)={a_at_Tj:.9f} != 预测峰值 {a_pred:.9f}")

    # --- 数值微分一致性 ---
    dS = np.diff(S) / h
    mid = (V[:-1] + V[1:]) / 2
    if np.median(np.abs(dS - mid)) > 1e-6:
        fails.append(f"ds/dt 与 v 不一致，中位误差 {np.median(np.abs(dS-mid)):.3e}")

    return p, fails


CASES = [
    # (d, v_max, a_max, j_max, 说明, 期望分支)
    (1.0, 0.5, 2.0, 10.0, "A 有匀速+匀加速 (v*j>=a^2, d>=d_acc)", "cruise_amax"),
    (2.0, 0.3, 1.0, 5.0, "A 有匀速+匀加速", "cruise_amax"),
    (0.167332, 0.15, 0.6, 5.0, "A 阻抗 demo 实际参数（原 bug 现场）", "cruise_amax"),
    (1.0, 0.2, 5.0, 1.0, "B 有匀速，加速段纯 jerk (v*j<a^2)", "cruise_jerk"),
    (0.19, 0.5, 2.0, 10.0, "C 无匀速但达 a_max (d<d_acc, d*j^2>=2a^3)", "tri_amax"),
    (0.05, 0.5, 2.0, 10.0, "D 纯 jerk 三角形", "tri_jerk"),
    (0.001, 0.5, 2.0, 10.0, "D 极小距离", "tri_jerk"),
    (-1.0, 0.5, 2.0, 10.0, "E 负距离", "cruise_amax"),
    (0.0, 0.5, 2.0, 10.0, "E 零距离", "degenerate"),
]


class TestBranchOracleIndependence(unittest.TestCase):
    """两条互相独立的分支判定路径必须一致，且都不读被测对象的内部时间。"""

    def test_predicted_branch_matches_signal_based_branch(self):
        rng = np.random.default_rng(0)
        samples = [(d, v, a, j) for d, v, a, j, _, _ in CASES]
        for _ in range(120):
            samples.append((rng.uniform(-3.0, 3.0), rng.uniform(0.05, 1.5),
                            rng.uniform(0.1, 5.0), rng.uniform(0.5, 40.0)))
        for d, v, a, j in samples:
            with self.subTest(d=d, v=v, a=a, j=j):
                p = SCurveProfile(d, v, a, j)
                want = predict_plan(d, v, a, j)["branch"]
                got = classify_from_signals(p, d, v, a, j)
                self.assertEqual(got, want,
                                 f"输入闭式说 {want}，波形说 {got}")

    def test_oracle_does_not_read_implementation_internals(self):
        """守护性检查：oracle 的**可执行代码**里不得出现 p.Tj/p.Ta/p.Tv/p.T/p.v_peak。

        注释与 docstring 里可以提到它们（用来说明为什么不能读），所以先剥掉。
        """
        import ast
        import inspect
        for fn in (predict_plan, reference_state, classify_from_signals, _segments):
            tree = ast.parse(inspect.getsource(fn).lstrip())
            attrs = {
                f"p.{node.attr}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "p"
            }
            self.assertEqual(
                attrs, set(),
                f"{fn.__name__} 读取了被测实现的内部量 {sorted(attrs)}",
            )


class TestSCurveInvariants(unittest.TestCase):

    def test_named_cases(self):
        for d, v, a, j, name, want_branch in CASES:
            with self.subTest(case=name):
                p, fails = check_profile(d, v, a, j, name)
                self.assertEqual(predict_plan(d, v, a, j)["branch"], want_branch,
                                 f"{name}: 输入闭式给出的分支不是 {want_branch}")
                self.assertEqual(classify_from_signals(p, d, v, a, j), want_branch,
                                 f"{name}: 波形判定的分支不是 {want_branch}")
                self.assertEqual(fails, [], f"{name}: {fails}")

    def test_all_branches_covered(self):
        """分支覆盖必须用**输入闭式** oracle 统计，不能用被测实现的内部时间。"""
        by_prediction = {predict_plan(d, v, a, j)["branch"]
                         for d, v, a, j, _, _ in CASES}
        by_signal = {classify_from_signals(SCurveProfile(d, v, a, j), d, v, a, j)
                     for d, v, a, j, _, _ in CASES}
        self.assertEqual(by_prediction, BRANCHES)
        self.assertEqual(by_signal, BRANCHES)

    def test_random_sweep(self):
        rng = np.random.default_rng(0)
        seen = {}
        for i in range(200):
            d = rng.uniform(-3.0, 3.0)
            v = rng.uniform(0.05, 1.5)
            a = rng.uniform(0.1, 5.0)
            j = rng.uniform(0.5, 40.0)
            _, fails = check_profile(d, v, a, j, "random", n=8001)
            self.assertEqual(fails, [],
                             f"随机样本 {i}: d={d} v={v} a={a} j={j} -> {fails}")
            b = predict_plan(d, v, a, j)["branch"]
            seen[b] = seen.get(b, 0) + 1
        # 随机扫描必须真的走遍非退化分支，否则"200 组全过"没有意义
        for b in ("cruise_amax", "cruise_jerk", "tri_amax", "tri_jerk"):
            self.assertGreater(seen.get(b, 0), 0, f"随机扫描未覆盖 {b}：{seen}")

    def test_boundary_cases_at_the_three_branch_switches(self):
        """三个判据的等号边界两侧都必须满足全部不变量。"""
        eps = 1e-9
        # (1) v·j = a²
        a, j = 2.0, 8.0
        v_eq = a * a / j
        for v in (v_eq * (1 - 1e-6), v_eq, v_eq * (1 + 1e-6)):
            _, fails = check_profile(1.0, v, a, j, "vj=a^2")
            self.assertEqual(fails, [], f"v={v}: {fails}")
        # (2) d = d_acc
        v, a, j = 0.5, 2.0, 10.0
        pl = predict_plan(1.0, v, a, j)
        d_eq = v * pl["Ta"]
        for d in (d_eq * (1 - 1e-6), d_eq, d_eq * (1 + 1e-6)):
            _, fails = check_profile(d, v, a, j, "d=d_acc")
            self.assertEqual(fails, [], f"d={d}: {fails}")
        # (3) d·j² = 2a³
        v, a, j = 0.5, 2.0, 10.0
        d_eq = 2.0 * a**3 / (j * j)
        for d in (d_eq * (1 - 1e-6), d_eq, d_eq * (1 + 1e-6)):
            _, fails = check_profile(d, v, a, j, "dj^2=2a^3")
            self.assertEqual(fails, [], f"d={d}: {fails}")
        self.assertGreater(eps, 0)


class TestMutationOracleHasTeeth(unittest.TestCase):
    """mutation：把参考实现故意改错后，check_profile 必须报错。"""

    def test_wrong_Ta_formula_is_detected(self):
        """用旧 bug 的 Ta = v/a（缺 Tj）构造参考时间表，独立积分必然对不上。"""
        d, v, a, j = 1.0, 0.5, 2.0, 10.0
        good = predict_plan(d, v, a, j)
        bad = dict(good)
        bad["Ta"] = v / a                       # 原 bug
        bad["Tv"] = (d - v * bad["Ta"]) / v
        bad["T"] = 2 * bad["Ta"] + bad["Tv"]
        p = SCurveProfile(d, v, a, j)
        worst = 0.0
        for t in np.linspace(0.0, min(good["T"], bad["T"]), 2001):
            worst = max(worst, abs(p(t)[0] - reference_state(bad, j, 1.0, t)[0]))
        self.assertGreater(worst, 1e-3,
                           "错误的 Ta 参考竟然与实现吻合，说明 oracle 没有约束力")


if __name__ == "__main__":
    unittest.main(verbosity=2)
