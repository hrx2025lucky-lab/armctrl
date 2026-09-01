"""P1-06 独立 oracle：视觉伺服延迟环的离散稳定边界。

完全不依赖 MuJoCo / 场景代码，只复刻 scenes.py:2748-2751 的递推：

    sp(k+1) = sp(k) + clip(gain*dt*(p_obj - sp(k-d)), +-step_max)   , 带 deadband

线性化后（去掉死区/限幅）令 e = sp - p_obj：

    e(k+1) = e(k) - lam*dt*e(k-d)

特征方程 z^(d+1) - z^d + lam*dt = 0。
边界：z=e^{j th} => 2 sin(th/2) = lam*dt 且 th(d+1/2) = pi/2
   => th = pi/(2d+1),  lam*dt = 2 sin(pi/(4d+2)),  lam*tau = 2d sin(pi/(4d+2))
d -> inf 时收敛到 pi/2。
"""
import numpy as np

DT = 0.002


def boundary_lam_tau(d: int) -> float:
    return 2.0 * d * np.sin(np.pi / (4.0 * d + 2.0))


def spectral_radius(lam: float, d: int, dt: float = DT) -> float:
    """e(k+1)=e(k)-lam*dt*e(k-d) 的伴随矩阵谱半径（严格线性）。"""
    n = d + 1
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    A[0, -1] = -lam * dt
    for i in range(1, n):
        A[i, i - 1] = 1.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def simulate(lam, d, dt=DT, T=60.0, e0=0.05, deadband=1e-3, step_max=4e-3):
    """带死区+限幅的真实递推。返回 (最终误差, 最大误差, 末段峰峰值)。"""
    n = int(T / dt)
    hist = np.zeros(n + d + 1)
    hist[: d + 1] = e0
    for k in range(d, n + d):
        e_del = hist[k - d]
        if abs(e_del) < deadband:
            delta = 0.0
        else:
            delta = -lam * dt * e_del
            if abs(delta) > step_max:
                delta = np.sign(delta) * step_max
        hist[k + 1] = hist[k] + delta
    tail = hist[-int(5.0 / dt):]
    return abs(hist[-1]), float(np.max(np.abs(hist))), float(np.ptp(tail))


def growth_rate(lam, d, dt=DT, T=40.0, e0=1e-4):
    """纯线性（无死区无限幅）下的每秒增长倍数——真正的稳定判据。"""
    n = int(T / dt)
    hist = np.zeros(n + d + 1)
    hist[: d + 1] = e0
    for k in range(d, n + d):
        hist[k + 1] = hist[k] - lam * dt * hist[k - d]
    a = float(np.max(np.abs(hist[int(0.5 * n): int(0.6 * n)])))
    b = float(np.max(np.abs(hist[int(0.9 * n): int(1.0 * n)])))
    span = (0.95 - 0.55) * T
    if a <= 0:
        return 0.0
    return float(np.log(b / a) / span)


if __name__ == "__main__":
    print("=" * 74)
    print("① 解析边界 vs 谱半径数值根（两条独立路径必须对上）")
    print("=" * 74)
    print(f"{'tau(ms)':>8} {'d':>4} {'解析 lam*tau':>13} {'解析 lam_crit':>13} "
          f"{'数值 lam_crit':>13} {'相对差':>9}")
    for tau_ms in (10, 20, 30, 50, 80, 120):
        tau = tau_ms * 1e-3
        d = int(round(tau / DT))
        lt = boundary_lam_tau(d)
        lam_a = lt / tau
        lo, hi = 0.1, 2000.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if spectral_radius(mid, d) < 1.0:
                lo = mid
            else:
                hi = mid
        lam_n = 0.5 * (lo + hi)
        print(f"{tau_ms:>8} {d:>4} {lt:>13.6f} {lam_a:>13.4f} {lam_n:>13.4f} "
              f"{abs(lam_n - lam_a) / lam_a:>8.2e}")

    print()
    print("=" * 74)
    print("② d -> inf 时是否趋向 pi/2 = 1.570796")
    print("=" * 74)
    for d in (1, 2, 5, 10, 25, 50, 100, 500, 5000):
        print(f"  d={d:>5}  lam*tau_crit = {boundary_lam_tau(d):.6f}")

    print()
    print("=" * 74)
    print("③ ⭐ 死区/限幅如何摧毁『收没收敛』这个判据")
    print("   d=25 (tau=50ms) -> 线性边界 lam_crit = %.3f" % (boundary_lam_tau(25) / 0.05))
    print("=" * 74)
    d = 25
    lam_c = boundary_lam_tau(d) / (d * DT)
    print(f"{'lam':>7} {'lam*tau':>8} {'谱半径':>9} {'线性增长/s':>11} "
          f"{'带死区末值mm':>13} {'带死区峰值mm':>13} {'判定':>10}")
    for lam in (10, 20, 25, 30, 30.8, 31, 35, 45, 60):
        rho = spectral_radius(lam, d)
        g = growth_rate(lam, d)
        fin, mx, pp = simulate(lam, d)
        verdict = "线性发散" if rho > 1.0 else "线性稳定"
        print(f"{lam:>7.1f} {lam * d * DT:>8.3f} {rho:>9.5f} {g:>11.3f} "
              f"{fin * 1e3:>13.3f} {mx * 1e3:>13.3f} {verdict:>10}")
    print()
    print(f"  解析临界 lam_crit = {lam_c:.4f}")
    print("  ⭐ 注意最后几行：谱半径 > 1（数学上必然发散），")
    print("     但带死区/限幅的『末值』依然是零点几毫米——看起来完全收敛。")
