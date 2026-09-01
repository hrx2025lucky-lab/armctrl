"""批次 A2 独立验收：P1-02（白噪声位置漂移阶数）与 P1-03（重力泄漏的方向条件）。

独立性说明
----------
* P1-02 用**蒙特卡洛**直接测量标准差随时间的幂次，再用最小二乘拟合指数，
  不引用任何解析式；随后才与手推的 Var[p]=q t³/3 对照。
* P1-03 的姿态误差传播由本文件按小角度公式独立推导，
  重力方向与 IMU 体系姿态取自 MuJoCo 模型本身（第三方）。
"""
import numpy as np

G = 9.81
GRAV_W = np.array([0.0, 0.0, -G])          # 世界系重力加速度矢量

print("=" * 72)
print("P1-02  连续白加速度噪声两次积分后，位置误差到底按几次方长？")
print("=" * 72)
rng = np.random.default_rng(20260828)
dt, n_steps, n_mc = 1e-3, 20000, 4000
q = 1.0                                     # 噪声密度平方 (m/s²)²/Hz
sigma_d = np.sqrt(q / dt)                   # 离散步的标准差

acc = rng.normal(0.0, sigma_d, size=(n_mc, n_steps))
vel = np.cumsum(acc, axis=1) * dt
pos = np.cumsum(vel, axis=1) * dt
t = np.arange(1, n_steps + 1) * dt

std_v, std_p = vel.std(axis=0), pos.std(axis=0)
sl = slice(n_steps // 20, None)             # 掐掉起始瞬态再拟合
kv = np.polyfit(np.log(t[sl]), np.log(std_v[sl]), 1)[0]
kp = np.polyfit(np.log(t[sl]), np.log(std_p[sl]), 1)[0]

print(f"  速度误差 std ∝ t^{kv:.4f}   （手推 0.5   → √t）")
print(f"  位置误差 std ∝ t^{kp:.4f}   （手推 1.5   → t^(3/2)）")
print(f"  文档写的是位置 ∝ √t（指数 0.5）→ 与实测 {kp:.3f} 相差 {abs(kp-0.5):.3f}  ❌")
print(f"  末点校核 std[p]={std_p[-1]:.4f}  vs  √(q·t³/3)={np.sqrt(q*t[-1]**3/3):.4f}"
      f"   相对偏差 {abs(std_p[-1]-np.sqrt(q*t[-1]**3/3))/np.sqrt(q*t[-1]**3/3)*100:.2f}%")
print("  ⇒ √t 是**速度**误差的标度，被错安到了位置上。")

print()
print("=" * 72)
print("P1-03  陀螺零偏造成的重力泄漏，取决于零偏相对重力的**方向**")
print("=" * 72)


def leak(bg_world, t_):
    """静止时重力补偿后的假加速度 δa = −(R·b_g) × g ·t（小角度）。"""
    return -np.cross(bg_world, GRAV_W) * t_


bg = 0.01                                   # rad/s，场景默认值
for name, vec in [("与重力平行 (b_g ∥ g，即竖直轴)", np.array([0, 0, 1.0])),
                  ("与重力垂直 (b_g ⊥ g，即水平轴)", np.array([1.0, 0, 0]))]:
    d = leak(bg * vec, 1.0)
    p_err = np.linalg.norm(-np.cross(bg * vec, GRAV_W)) / 6.0 * 10 ** 3
    print(f"  {name}")
    print(f"     t=1s 时假加速度 |δa| = {np.linalg.norm(d):.6f} m/s²"
          f"    ；10 s 位置误差 = {p_err:.4f} m")
print(f"  文档的通用式 ⅙·g·b_g·t³ 在 t=10s 给 {G*bg/6*1000:.4f} m —— "
      f"**只在垂直情形成立**，平行情形真值为 0 ❌")

print()
print("-" * 72)
print("场景实际注入的是哪个方向？（scenes.py: gyro_bias0=(0,0,b_g_z)，体系 z 轴）")
print("-" * 72)
try:
    import mujoco
    from armctrl.planning.scene import build_panda_scene
    from armctrl.tuner.scenes import Q_HOME

    scene = build_panda_scene([], with_visuals=True)
    model = scene.model
    data = mujoco.MjData(model)
    # IMU 挂在 hand 连杆上（scene.add_imu 默认 body="hand"，site 无旋转偏置），
    # 所以陀螺体系 = hand 体系。
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    nq = len(Q_HOME)
    data.qpos[:nq] = Q_HOME
    mujoco.mj_forward(model, data)
    R = data.xmat[bid].reshape(3, 3)         # 世界 ← 体
    z_body_in_world = R[:, 2]
    cos_ang = float(np.dot(z_body_in_world, np.array([0, 0, 1.0])))
    ang = np.degrees(np.arccos(np.clip(abs(cos_ang), -1, 1)))
    print(f"  Q_HOME 处 IMU(hand) 体系 z 轴在世界系的方向 = {z_body_in_world.round(4)}")
    print(f"  与竖直方向夹角 = {ang:.2f}°   (0° = 完全平行于重力 → 泄漏为 0)")
    bg_w = R @ np.array([0.0, 0.0, bg])
    da = np.linalg.norm(leak(bg_w, 1.0))
    da_max = bg * G
    print(f"  实际 |δa|(t=1s) = {da:.6f} m/s²  ；同幅值垂直零偏上限 = {da_max:.6f} m/s²")
    print(f"  ⇒ 实际重力泄漏只有理论上限的 {da / da_max * 100:.1f}%"
          f"；UI 称之为「绕竖直轴」{'成立' if ang < 15 else '**并不成立**'}")
    print(f"  ⇒ 文档通用式在此场景会把 10s 位置误差高估到 "
          f"{G * bg / 6 * 1000:.2f} m，实际应为 "
          f"{np.linalg.norm(-np.cross(bg_w, GRAV_W)) / 6 * 10 ** 3:.2f} m")
except Exception as exc:                     # pragma: no cover
    print("  场景检查跳过：", exc)

print()
print("=" * 72)
print("P1-03(代码侧)  scenes.py 用总姿态误差 att 算重力泄漏，把纯偏航也算了进去")
print("=" * 72)
# 独立构造：绕重力轴的纯偏航误差，真值泄漏必须为 0
g_hat = np.array([0.0, 0.0, -1.0])                # 重力方向单位矢量
for name, axis in [("纯偏航误差 (绕重力轴)", g_hat),
                   ("纯倾斜误差 (垂直重力)", np.array([1.0, 0.0, 0.0]))]:
    for deg in (2.0, 5.0):
        dth = np.radians(deg) * axis              # 姿态误差矢量
        # 真值：重力矢量被姿态误差旋转后产生的水平分量
        true_leak = np.linalg.norm(np.cross(dth, GRAV_W))
        code_leak = G * np.sin(np.linalg.norm(dth))   # 现有实现 g·sin(att)
        print(f"  {name}  {deg:.0f}°：真值泄漏 = {true_leak:.6f} m/s²"
              f"   ；现有实现报 {code_leak:.6f} m/s²"
              f"   {'❌ 虚报' if true_leak < 1e-12 < code_leak else '✅ 一致'}")
print("  ⇒ 应改用 tilt_part（垂直于重力的分量），该量在 scenes.py 中已算好。")
