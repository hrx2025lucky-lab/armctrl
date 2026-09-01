"""ImpedanceScene 的两张实测表（`01_阻抗控制.md` §9.2 / §9.3）必须能重跑出来。

为什么需要这一组
----------------
§9.3 那张表里原本写着「ζ=0.3 → 超调 17.2 %，稳态 40.000 mm」。
2026-08-31 复现时发现**这两列来自两次不同的实验**：

    17.2 % 其实是 F_z = −30 N 那次跑出来的（17.27 %）；
    40.000 mm 对应的却是 F_z = −20 N（20/500 = 40 mm）。

单看任何一格都"对"，拼在一起就成了一个**不存在的实验**。
这和之前变异扫描抓到的 M21（表格里的数字在别处也出现，assertIn 蒙混过关）
是同一类病：**表格的行必须整行绑定，不能一格一格地验。**

oracle 独立性
------------
- 稳态判据用**解析式** Δz = F/K 直接算，不读控制器的任何中间量，
  也不读场景的 `dz_theory` 读数（那是被测对象自己算的）。
- 超调判据用**二阶欠阻尼系统的解析超调量** exp(−πζ/√(1−ζ²))
  给出「ζ=0.3 该有超调、ζ≥0.7 不该有」的定性预期，
  再用仿真曲线量实际值。两条路径只共享「二阶系统」这一个前提。
- ⭐ 表格里的期望值**从 Markdown 现场解析**，不写死在测试里。
  第一版把 (0.3, 17.5, −40.000) 硬编码进测试，结果变异扫描里
  「把文档某一格改错」全部活口——测试验的是仿真，根本没验文档。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import mujoco
import numpy as np

from armctrl.tuner.scenes import SCENES
from armctrl.tests._docs import requires_docs

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "armctrl_讲解" / "01_阻抗控制.md"

_IMP = [s for s in SCENES if s.name == "impedance"][0]

_NUM = r"[-−]?\d+(?:\.\d+)?"


def _f(s):
    """把文档里的数字转成 float（注意用的是全角减号 U+2212）。"""
    return float(s.replace("−", "-"))


def doc_section(head):
    text = DOC.read_text(encoding="utf-8")
    return text.split(head, 1)[1].split("\n### ", 1)[0]


def parse_stiffness_rows():
    """从 §9.2 解析 [(K, 下沉 mm), ...]。"""
    rows = re.findall(rf"^\|\s*({_NUM})\s*\|\s*({_NUM})\s*mm\s*\|",
                      doc_section("### 9.2"), re.M)
    assert rows, "§9.2 的刚度表解析不出来——表格格式变了？"
    return [(_f(k), _f(v)) for k, v in rows]


def parse_damping_rows():
    """从 §9.3 解析 [(ζ, 超调 %, 稳态 mm), ...]。"""
    rows = re.findall(
        rf"^\|\s*({_NUM})\s*\|\s*\**\s*({_NUM})\s*%\s*\**\s*\|"
        rf"\s*({_NUM})\s*mm\s*\|",
        doc_section("### 9.3"), re.M)
    assert rows, "§9.3 的阻尼表解析不出来——表格格式变了？"
    return [(_f(z), _f(o), _f(s)) for z, o, s in rows]


def analytic_overshoot(zeta):
    """二阶系统阶跃响应的超调量（独立解析式，占比）。"""
    if zeta >= 1.0:
        return 0.0
    return float(np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta * zeta)))


def roll(seconds, **overrides):
    """跑一段仿真，返回 Δz 的时间序列（mm）。"""
    sc = _IMP()
    sc.build()
    for key, val in overrides.items():
        sc.set(key, val)
    sc.reset()
    d, r = sc.data, sc.robot
    sub = max(1, int(round(sc.dt / sc.model.opt.timestep)))
    out = np.empty(int(seconds / sc.dt))
    for k in range(out.size):
        t = k * sc.dt
        q = d.qpos[r.qpos_idx][:7]
        v = d.qvel[r.qvel_idx][:7]
        tau = sc.control(t, q, v)
        d.ctrl[r.arm_actuator_ids] = tau
        for _ in range(sub):
            mujoco.mj_step(sc.model, d)
        out[k] = sc.telemetry(
            t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])["dz"]
    return out


def step_response(zeta, fz=-20.0, k_trans=500.0, seconds=6.0):
    """一次干净的阶跃：period=24 保证前 12 秒力一直加着。

    返回 (稳态 mm, 超调占比)。
    """
    ser = roll(seconds, k_trans=k_trans, zeta=zeta, fz=fz,
               force_mode="square", period=24.0, damping="critical")
    settle = float(ser[-1])
    peak = float(ser.min())          # 向下为负，最深处
    return settle, max((peak - settle) / settle, 0.0)


@requires_docs
class TestTheStiffnessSweepTableIsReproducible(unittest.TestCase):
    """§9.2：Δz 与 K 严格成反比。判据是解析式 F/K，期望值现场读文档。"""

    FZ = -20.0

    @classmethod
    def setUpClass(cls):
        cls.rows = parse_stiffness_rows()
        cls.measured = {}
        for k, _ in cls.rows:
            ser = roll(6.0, k_trans=k, fz=cls.FZ, force_mode="constant")
            cls.measured[k] = float(ser[-1])

    def test_the_table_still_has_four_rows(self):
        self.assertEqual(len(self.rows), 4,
                         f"§9.2 应有 4 行，解析到 {len(self.rows)}：{self.rows}")

    def test_each_row_matches_the_analytic_deflection(self):
        for k, doc_mm in self.rows:
            with self.subTest(K=k):
                oracle = self.FZ / k * 1000.0        # 解析式 F/K，单位 mm
                self.assertAlmostEqual(
                    self.measured[k], oracle, delta=0.02,
                    msg=f"K={k} 实测 {self.measured[k]:.3f} mm 与解析式 "
                        f"{oracle:.3f} mm 不符")
                self.assertAlmostEqual(
                    abs(self.measured[k]), abs(doc_mm), delta=0.02,
                    msg=f"K={k} 文档写 {doc_mm} mm，实测 "
                        f"{abs(self.measured[k]):.3f} mm")

    def test_tripling_the_stiffness_thirds_the_deflection(self):
        """文档那句"K 乘 3，下沉除以 3"本身也要成立。"""
        for small, big in ((250, 750), (500, 1500)):
            with self.subTest(pair=(small, big)):
                self.assertIn(big, self.measured, "表里缺 K=%d 这一档" % big)
                self.assertAlmostEqual(
                    self.measured[big] / self.measured[small],
                    1.0 / 3.0, places=3)


@requires_docs
class TestTheDampingSweepTableIsReproducible(unittest.TestCase):
    """§9.3：整行绑定——ζ、超调、稳态三个数必须来自同一次实验。"""

    FZ, K = -20.0, 500.0

    @classmethod
    def setUpClass(cls):
        cls.rows = parse_damping_rows()
        cls.runs = {z: step_response(z, fz=cls.FZ, k_trans=cls.K)
                    for z, _, _ in cls.rows}

    def test_the_table_still_has_four_rows(self):
        self.assertEqual(len(self.rows), 4,
                         f"§9.3 应有 4 行，解析到 {len(self.rows)}：{self.rows}")

    def test_every_row_comes_from_one_and_the_same_experiment(self):
        for zeta, doc_over, doc_settle in self.rows:
            settle, over = self.runs[zeta]
            with self.subTest(zeta=zeta):
                self.assertAlmostEqual(
                    settle, doc_settle, delta=0.02,
                    msg=f"ζ={zeta} 稳态实测 {settle:.3f} mm，文档写 {doc_settle}")
                self.assertAlmostEqual(
                    over * 100.0, doc_over, delta=0.3,
                    msg=f"ζ={zeta} 超调实测 {over * 100:.2f} %，"
                        f"文档写 {doc_over} %")

    def test_the_steady_state_ignores_damping_entirely(self):
        """K e = F 里没有 D，所以四档 ζ 的终点必须一模一样。"""
        settles = [self.runs[z][0] for z, _, _ in self.rows]
        self.assertLess(
            max(settles) - min(settles), 1e-3,
            f"四档 ζ 的稳态应当完全相同，实测 {np.round(settles, 6)}")

    def test_the_analytic_oracle_agrees_on_who_overshoots(self):
        """独立解析式只用来定性把关：谁该有超调、谁不该有。"""
        for zeta, _, _ in self.rows:
            _, over = self.runs[zeta]
            with self.subTest(zeta=zeta):
                if analytic_overshoot(zeta) > 0.05:
                    self.assertGreater(
                        over, 0.05,
                        f"ζ={zeta} 解析式预测有明显超调，实测却只有 {over:.4f}")
                else:
                    self.assertLess(
                        over, 0.02,
                        f"ζ={zeta} 解析式预测无超调，实测却有 {over:.4f}")

    def test_the_overshoot_is_quoted_together_with_the_force(self):
        """超调随外力变，所以文档引用它时必须把 F_z 一起写出来。"""
        settle30, over30 = step_response(0.3, fz=-30.0, k_trans=self.K)
        self.assertAlmostEqual(settle30, -60.0, delta=0.02)
        over20 = self.runs[0.3][1]
        # −30 N 那次是 17.27 %，和 −20 N 的 17.53 % 明显不同
        self.assertGreater(abs(over30 - over20) * 100, 0.15,
                           "两种外力下的超调若无差别，本条测试就失去意义了")
        head = doc_section("### 9.3").replace("$", "").replace("\\", "")
        self.assertIn("F_z = -20", head,
                      "§9.3 必须写明这张表用的外力是多少")

    def test_the_retracted_number_is_marked_as_wrong(self):
        """旧的 17.2 % 不许再以"结论"的身份出现。"""
        text = DOC.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "17.2 %" in line and "17.27" not in line:
                self.assertRegex(
                    line, r"错|更正|原来",
                    f"01 篇仍在把已撤回的 17.2 % 当结论用：\n  {line}")


# ============================================== §10 K=50 压到地板的那张实测清单


def k50_state(seconds=8.0, K=50.0, FZ=-30.0):
    """跑到稳态，把 §10.1 那张表需要的量一次全测出来。

    oracle 独立性
    ------------
    - 地面托力**不读** `qfrc_constraint`，而是逐个接触调 `mj_contactForce`
      拿接触力、自己转到世界系取 z 分量求和。
    - 弹簧承担的力**不读**控制器内部的 `f_cmd`，用胡克定律 K·Δz 现算。
    - 于是「地面 + 弹簧 = 外力」才是一句有内容的校验，而不是自己验自己。
    """
    sc = _IMP()
    sc.build()
    for k, v in dict(k_trans=K, fz=FZ, force_mode="constant",
                     damping="critical").items():
        sc.set(k, v)
    sc.reset()
    d, r, m = sc.data, sc.robot, sc.model
    sub = max(1, int(round(sc.dt / m.opt.timestep)))
    tel = {}
    for k in range(int(seconds / sc.dt)):
        t = k * sc.dt
        d.ctrl[r.arm_actuator_ids] = sc.control(
            t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])
        for _ in range(sub):
            mujoco.mj_step(m, d)
        tel = sc.telemetry(t, d.qpos[r.qpos_idx][:7], d.qvel[r.qvel_idx][:7])

    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    tau_c, buf, ground = np.zeros(m.nv), np.zeros(6), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, buf)
        f_world = c.frame.reshape(3, 3).T @ buf[:3]
        ground += float(f_world[2])
        b1 = int(m.geom_bodyid[int(c.geom1)])
        b2 = int(m.geom_bodyid[int(c.geom2)])
        body, sign = (b2, +1.0) if b1 == 0 else (b1, -1.0)
        mujoco.mj_jac(m, d, jacp, jacr, c.pos, body)
        tau_c += sign * (jacp.T @ f_world)

    q = d.qpos[r.qpos_idx][:7]
    p_ee, _R = r.fk(q)
    J = r.jacobian(q)
    zhat = np.zeros(6)
    zhat[2] = 1.0
    dz_mm = abs(float(tel["dz"]))
    return {
        "theory_mm": abs(FZ) / K * 1e3,
        "dz_mm": dz_mm,
        "ee_mm": float(p_ee[2]) * 1e3,
        "ncon": int(d.ncon),
        "ground_N": ground,
        "spring_N": K * dz_mm * 1e-3,
        "qfrc": float(np.linalg.norm(d.qfrc_constraint[r.qvel_idx][:7])),
        "jtf": float(np.linalg.norm(tau_c[r.qvel_idx][:7])),
        "cond": float(np.linalg.cond(J)),
        "jtz": float(np.linalg.norm(J.T @ zhat)),
        "fz": abs(FZ),
    }


def parse_k50_rows():
    """把 §10.1 那张表解析成 {标签: 数值}。期望值一律现场读文档，不写死。"""
    out = {}
    for line in doc_section("### 10.1").splitlines():
        if not line.startswith("|") or line.count("|") < 3:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        hit = re.search(_NUM, cells[1].replace(",", ""))
        if hit:
            out[cells[0]] = _f(hit.group(0))
    assert out, "§10.1 的清单解析不出来——表格格式变了？"
    return out


@requires_docs
class TestTheContactBreakdownAtK50IsReproducible(unittest.TestCase):
    """§10「K=50 那 33% 偏差」里的每一个数都必须能重跑出来。

    为什么需要这一组
    ----------------
    这一节原本写着「4 个接触点 / 地面 9.78 N / `qfrc_constraint` 4.121」，
    **一个都没有测试守着**——是三轮以前手工跑一次贴上去的。
    之后场景改过好几次，第四轮审核复现时发现 `ncon` 其实是 3、
    `qfrc_constraint` 其实是 4.2611。

    ⭐ 教训：**「跑一次贴上来」的数字，寿命只到下一次改代码为止。**
    所以这里照 §9.2 / §9.3 的老规矩办：**期望值从 Markdown 现场解析**，
    写死在测试里就等于把文档漏掉了（旧变异体 M21 就是这么活的）。
    """

    @classmethod
    def setUpClass(cls):
        cls.s = k50_state()
        cls.rows = parse_k50_rows()

    def _doc(self, key):
        for label, val in self.rows.items():
            if key in label:
                return val
        self.fail(f"§10.1 的表里找不到「{key}」这一行")

    def test_the_table_still_has_every_row(self):
        for key in ("理论下沉", "实测下沉", "末端离地", "接触点数",
                    "地面托力", "弹簧承担", "qfrc_constraint", "搬到关节空间"):
            self._doc(key)

    def test_the_shortfall_is_real_and_is_about_a_third(self):
        """先确认现象还在：不到理论值的 70%。不然整节就该删掉而不是改数字。"""
        s = self.s
        self.assertAlmostEqual(s["theory_mm"], self._doc("理论下沉"), delta=0.01)
        self.assertAlmostEqual(s["dz_mm"], self._doc("实测下沉"), delta=0.05,
                               msg="实测下沉变了，§10.1 要跟着改")
        self.assertLess(s["dz_mm"] / s["theory_mm"], 0.70,
                        "偏差不再是三成，这一节的标题就不成立了")

    def test_the_gripper_really_is_on_the_floor(self):
        s = self.s
        self.assertAlmostEqual(s["ee_mm"], self._doc("末端离地"), delta=0.05)
        self.assertEqual(s["ncon"], int(self._doc("接触点数")),
                         f"接触点数实测 {s['ncon']}，文档写的是 "
                         f"{int(self._doc('接触点数'))}")
        self.assertGreater(s["ncon"], 0, "没有接触，那这一节的解释就不成立")

    def test_the_two_load_paths_add_up_to_the_external_force(self):
        """牛顿第三定律：外力被「地面 + 弹簧」两家分完。两条测量路径互不相干。"""
        s = self.s
        self.assertAlmostEqual(s["ground_N"], self._doc("地面托力"), delta=0.02)
        self.assertAlmostEqual(s["spring_N"], self._doc("弹簧承担"), delta=0.02)
        self.assertAlmostEqual(
            s["ground_N"] + s["spring_N"], s["fz"], delta=0.15,
            msg="地面 + 弹簧 ≠ 外力，说明至少有一条路径算错了")

    def test_the_constraint_torque_matches_an_independent_jacobian_route(self):
        """同一件事换两条路算：读 MuJoCo 的结果 vs 自己拿雅可比搬接触力。"""
        s = self.s
        self.assertAlmostEqual(s["qfrc"], self._doc("qfrc_constraint"),
                               delta=0.001)
        self.assertAlmostEqual(s["jtf"], self._doc("搬到关节空间"), delta=0.001)
        self.assertAlmostEqual(
            s["qfrc"], s["jtf"], delta=1e-4,
            msg="两条独立路径对不上，说明接触力的搬运方式有问题")

    def test_the_singularity_hypothesis_stays_ruled_out(self):
        """§10.3 说「不是奇异」。这个否定结论也要能重跑，否则只是句口号。"""
        s = self.s
        text = doc_section("### 10.3")
        cond_doc = _f(re.search(rf"条件数只有 \*\*({_NUM})\*\*", text).group(1))
        jtz_doc = _f(re.search(rf"\\rVert = ({_NUM})", text).group(1))
        self.assertAlmostEqual(s["cond"], cond_doc, delta=0.02)
        self.assertLess(s["cond"], 30.0, "条件数这么大就不能说「远不算病态」了")
        self.assertAlmostEqual(s["jtz"], jtz_doc, delta=0.005)
        self.assertGreater(s["jtz"], 1e-3,
                           "z 方向若真落在 Jᵀ 的零空间里，§10.3 的结论就反了")



if __name__ == "__main__":
    unittest.main()
