import numpy as np
from armctrl.estimation.kalman import TcpStateEstimator

dt=0.002
# --- A. 独立可观性判据：只用状态方程结构，不调用被测滤波器 ---
F=np.array([[1,dt,0],[0,1,-dt],[0,0,1.0]])   # [p, v, b], b 通过 -b*dt 进 v
H=np.array([[1.0,0,0]])                       # 只测位置
O=np.vstack([H, H@F, H@F@F])
print("可观性矩阵 O =\n",O)
print("det(O) =",np.linalg.det(O),"   -dt^3 =",-dt**3)
print("rank(O) =",np.linalg.matrix_rank(O),"/3  -> 满秩=可观")

# --- B. 端到端：完全静止(零运动激励)，看零偏能否被估出 ---
est=TcpStateEstimator(acc_noise=2e-3,bias_walk=1e-6,meas_noise=1e-3)
p_true=np.array([0.4,0.0,0.5]); v_true=np.zeros(3)
b_true=np.array([0.05,-0.03,0.02])            # 真实零偏
est.reset(p_true, v_true, b0=np.zeros(3))
R=np.eye(3)
g=np.array([0,0,-9.81])
for k in range(int(20/dt)):
    # 静止：真实比力 = -g，IMU 读数被零偏污染
    f_body = (-g) + b_true
    est.predict(f_body, R, dt)
    est.update_position(p_true)               # 绝对位置量测(前向运动学)
print("\n完全静止 20s 后：")
print("  b_true =",b_true)
print("  b_hat  =",est.bias)
print("  误差范数 =",np.linalg.norm(est.bias-b_true))
