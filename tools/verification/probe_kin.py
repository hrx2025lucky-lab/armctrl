"""独立探针：为《机器人学基础》讲解实测所有数字。"""
import numpy as np
from armctrl.core.robot import make_panda
np.set_printoptions(precision=4, suppress=True)
r = make_panda()
Q_HOME = np.array([0.0,-0.3,0.0,-2.2,0.0,2.0,0.785])
p,R = r.fk(Q_HOME)
print("=== 正运动学 FK(Q_HOME) ===")
print("末端位置 p =", p, "(m)")
print("末端姿态 R =\n", R)
J = r.jacobian(Q_HOME)
print("\n=== 雅可比 J 形状 =", J.shape, "===")
print(J)
S = np.linalg.svd(J, compute_uv=False)
print("\n奇异值 sigma =", S)
print("可操作度 w =", r.manipulability(Q_HOME), " 连乘核对 =", np.prod(S))
print("条件数 kappa =", r.condition_number(Q_HOME), " 手算 =", S[0]/S[-1])
print("\n=== 几个位形对比 ===")
cfgs = {"HOME": Q_HOME,
 "接近伸直": np.array([0.,0.,0.,-0.05,0.,1.0,0.785]),
 "半屈": np.array([0.,-0.6,0.,-1.5,0.,1.5,0.785]),
 "蜷缩": np.array([0.,-1.2,0.,-2.8,0.,2.5,0.785])}
print(f"{'位形':<12}{'w':>12}{'kappa':>12}{'smin':>10}{'smax':>10}{'伸展m':>9}")
for k,q in cfgs.items():
    S2=np.linalg.svd(r.jacobian(q),compute_uv=False)
    pp,_=r.fk(q)
    print(f"{k:<12}{r.manipulability(q):>12.5f}{r.condition_number(q):>12.2f}{S2[-1]:>10.4f}{S2[0]:>10.4f}{np.linalg.norm(pp[:2]):>9.4f}")
M = r.mass_matrix(Q_HOME)
print("\n=== 质量矩阵 M(q) 对角线 (kg m^2) ===\n", np.diag(M))
ev = np.linalg.eigvalsh(M)
print("M 特征值 min/max =", ev[0], ev[-1], " 条件数 =", ev[-1]/ev[0])
print("\n=== 重力力矩 g(q) (N m) ===\n", r.gravity(Q_HOME))
Lam = r.task_space_inertia(Q_HOME)
print("\n=== 任务空间惯量 Lambda 对角线 ===\n", np.diag(Lam))
print("Lambda 平移块特征值 (kg) =", np.linalg.eigvalsh(Lam[:3,:3]))
