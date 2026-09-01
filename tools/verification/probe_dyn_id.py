"""独立探针：逆动力学恒等式 + 回归矩阵 + 基参数（讲解 14 号的取数脚本）。

跑法：
    cd "motion control"
    PYTHONPATH=. MUJOCO_GL=egl <venv>/bin/python <本文件>
"""
import numpy as np
from armctrl.identification.regressor import DynamicsRegressor, LockedFingerArmView
from armctrl.demos.demo_mpc import PANDA_XML

np.set_printoptions(precision=5, suppress=True)
reg = DynamicsRegressor(PANDA_XML)
arm = LockedFingerArmView(reg, n_arm=7)
q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])
z = np.zeros(7)
rng = np.random.default_rng(0)
v = rng.normal(0, 0.5, 7)
a = rng.normal(0, 1.0, 7)

print("=== 1. 参数向量 pi 的构成 ===")
pi = arm.true_parameters()
print("pi 长度 =", pi.shape[0],
      " = 连杆惯性 10x7 =", 10 * reg.nv,
      " + armature", reg.nv, " + 粘滞摩擦", reg.nv, " + 库仑摩擦", reg.nv)

print("\n=== 2. 恒等式  tau = Y(q,v,a) @ pi  ===")
Y = arm.regressor(q, v, a)
print("Y 形状 =", Y.shape)
print("RNEA   =", arm.rnea(q, v, a))
print("Y @ pi =", Y @ pi)
print("最大绝对差 =", np.max(np.abs(arm.rnea(q, v, a) - Y @ pi)))

print("\n=== 3. 三项分解  tau = M a + C v + g + 摩擦 ===")
M, Cv, g = arm.mass_matrix(q), arm.coriolis_times(q, v, v), arm.gravity(q)
fric = reg.fv_true[:7] * v + reg.fc_true[:7] * np.tanh(200.0 * v)
print("M a  范数 = %.4f" % np.linalg.norm(M @ a))
print("C v  范数 = %.4f" % np.linalg.norm(Cv))
print("g    范数 = %.4f" % np.linalg.norm(g))
print("摩擦 范数 = %.4f" % np.linalg.norm(fric))
print("四项之和 vs RNEA 最大差 =",
      np.max(np.abs(M @ a + Cv + g + fric - arm.rnea(q, v, a))))
print("⚠️ 少了摩擦这一项的话，最大差 =",
      np.max(np.abs(M @ a + Cv + g - arm.rnea(q, v, a))))

print("\n=== 4. 基参数：91 个里只有几个能辨识出来 ===")
P, rank, sv = arm.base_parameter_svd(samples=400, seed=0)
print("投影矩阵 P 形状 =", P.shape, "  可辨识秩 =", rank, " / 91")
print("辨识不了的方向数 =", 91 - rank)
print("信息矩阵奇异值 前5 =", np.round(sv[:5], 3))
print("                末5 =", np.round(sv[-5:], 6))
print("非零部分条件数 = %.3f" % (sv[0] / sv[rank - 1]))
