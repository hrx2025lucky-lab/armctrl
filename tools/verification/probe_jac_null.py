"""独立探针：7 自由度零空间 + 奇异点附近的速度爆炸 + DLS 阻尼。"""
import numpy as np
from armctrl.core.robot import make_panda
np.set_printoptions(precision=4, suppress=True)
r = make_panda()
Q = np.array([0.0,-0.3,0.0,-2.2,0.0,2.0,0.785])

print("=== 1. 零空间：7 个关节做 6 维任务，多出来 1 个自由度 ===")
J = r.jacobian(Q)
U,S,Vt = np.linalg.svd(J)
nullv = Vt[-1]                      # 最后一行 = 零空间基（V 的第 7 列）
print("零空间方向 n =", nullv, " ||n||=%.6f" % np.linalg.norm(nullv))
print("J @ n =", J@nullv, " 范数 = %.3e  (应为 0)" % np.linalg.norm(J@nullv))
p0,_ = r.fk(Q)
for eps in (0.01, 0.05, 0.20):
    p1,_ = r.fk(Q + eps*nullv)
    print(f"  沿零空间走 {eps:.2f} rad → 末端只挪了 {np.linalg.norm(p1-p0)*1e3:8.4f} mm"
          f"   关节最大挪了 {np.max(np.abs(eps*nullv)):.4f} rad")
p1,_ = r.fk(Q + 0.05*Vt[0])
print(f"  对照：沿 σmax 方向走 0.05 rad → 末端挪了 {np.linalg.norm(p1-p0)*1e3:.2f} mm")

print("\n=== 2. 靠近肘奇异时会发生什么 ===")
print(f"{'肘关节 q4':>10}{'w 可操作度':>14}{'kappa':>10}{'sigma_min':>11}{'伪逆最大关节速度':>18}{'DLS最大关节速度':>17}")
vdes = np.array([0.,0.,0.05,0.,0.,0.])          # 末端想以 5 cm/s 竖直上移
for q4 in (-2.2,-1.5,-1.0,-0.5,-0.2,-0.05,-0.01):
    q = Q.copy(); q[3]=q4
    Jq = r.jacobian(q); Sq=np.linalg.svd(Jq,compute_uv=False)
    qd_pinv = np.linalg.pinv(Jq) @ vdes
    lam = 0.05
    qd_dls  = Jq.T @ np.linalg.solve(Jq@Jq.T + lam**2*np.eye(6), vdes)
    print(f"{q4:>10.2f}{r.manipulability(q):>14.6f}{r.condition_number(q):>10.1f}"
          f"{Sq[-1]:>11.5f}{np.max(np.abs(qd_pinv)):>18.4f}{np.max(np.abs(qd_dls)):>17.4f}")

print("\n=== 3. 阻尼 lambda 的代价：跟踪误差 ===")
q = Q.copy(); q[3]=-0.05
Jq = r.jacobian(q)
print(f"{'lambda':>8}{'最大关节速度':>14}{'实际末端速度':>14}{'方向误差(deg)':>15}")
for lam in (0.0, 0.01, 0.05, 0.1, 0.3):
    if lam==0: qd = np.linalg.pinv(Jq)@vdes
    else: qd = Jq.T @ np.linalg.solve(Jq@Jq.T + lam**2*np.eye(6), vdes)
    got = Jq@qd
    cs = np.dot(got[:3],vdes[:3])/(np.linalg.norm(got[:3])*np.linalg.norm(vdes[:3])+1e-12)
    print(f"{lam:>8.2f}{np.max(np.abs(qd)):>14.4f}{np.linalg.norm(got[:3]):>14.5f}"
          f"{np.degrees(np.arccos(np.clip(cs,-1,1))):>15.2f}")
