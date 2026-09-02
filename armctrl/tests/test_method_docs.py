"""`docs/控制方法/` 那 14 篇公开文档的防漂移守护。

为什么需要这一组
----------------
讲义（`armctrl_讲解/`）早就有 `test_teaching_consistency.py` 守着：数字从
Markdown 现场解析再逐格重跑，改错任何一格直接测试红。

但 `docs/控制方法/` 这批**公开**文档写完时**一条守护都没有**——
它们引用了几十个函数名、文件路径、默认值和可运行命令，其中任何一个
在下次重构后失效，都不会有任何东西报警。

这正是本项目反复吃过亏的模式（元教训 #58：
「跑一次贴上来的数字，寿命只到下一次改代码为止」）。

\u2b50 和讲义那组的关键区别：`docs/控制方法/` **在仓库里**，
所以这组测试**不挂 `@requires_docs`**，CI 上照跑。

守什么
------
1. 六节结构完整——文档骨架不许被悄悄改掉；
2. 引用的 `armctrl/**.py` 路径真实存在；
3. 反引号里的符号（函数/类/常量）在代码里真实存在——防编造 API；
4. \u2b50 写成 `key=value` 的**默认值**必须和代码里的实际默认值一致；
5. `python -m unittest armctrl.tests.X` 里的测试模块真实存在；
6. 不许出现本机绝对路径（别人照抄会失败）；
7. 不许出现写作过程自述（那是过程，不是文档内容）。

第 4 条是这组里最有价值的：它把「文档说默认值是多少」和
「代码里默认值真的是多少」绑在一起，改代码不改文档就会红。
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC_DIR = REPO / "docs" / "控制方法"

#: 六节骨架（用小标题的关键词匹配，不写死全角顿号前的编号）
SECTIONS = ("要解决的问题", "原理", "怎么用在机械臂上",
            "代码在哪", "怎么验证它是对的", "边界与局限")

#: 正文篇（不含索引 README）
def method_docs() -> list[pathlib.Path]:
    return sorted(p for p in DOC_DIR.glob("*.md") if p.name != "README.md")


def all_source() -> str:
    """全部 Python 源码拼成一个字符串，用于「这个符号存不存在」的检查。"""
    out = []
    for f in sorted((REPO / "armctrl").rglob("*.py")):
        out.append(f.read_text(encoding="utf-8"))
    return "\n".join(out)


class TestTheMethodDocsExist(unittest.TestCase):
    """前提：文档目录得在，且篇数别悄悄变少。"""

    def test_the_directory_is_there(self):
        self.assertTrue(DOC_DIR.is_dir(), f"找不到 {DOC_DIR}")

    def test_there_are_still_thirteen_method_docs_plus_an_index(self):
        docs = method_docs()
        self.assertEqual(
            len(docs), 13,
            f"正文应有 13 篇，实际 {len(docs)} 篇："
            f"{[d.name for d in docs]}")
        self.assertTrue((DOC_DIR / "README.md").is_file(), "索引 README 不见了")


class TestEveryDocKeepsTheSixSectionSkeleton(unittest.TestCase):
    """\u2b50 六节结构是这批文档的骨架，缺一节就说明有人改了约定。"""

    def test_every_doc_has_all_six_sections(self):
        for d in method_docs():
            heads = re.findall(r"^## (.+)$", d.read_text(encoding="utf-8"), re.M)
            for want in SECTIONS:
                with self.subTest(doc=d.name, section=want):
                    self.assertTrue(
                        any(want in h for h in heads),
                        f"{d.name} 缺「{want}」这一节")


class TestEveryReferencedPathAndSymbolIsReal(unittest.TestCase):
    """防编造：文档写的文件和符号必须真的存在。"""

    @classmethod
    def setUpClass(cls):
        cls.src = all_source()

    def test_every_referenced_source_file_exists(self):
        missing = []
        for d in method_docs():
            for m in re.finditer(r"`(armctrl/[A-Za-z0-9_/]+\.py)`",
                                 d.read_text(encoding="utf-8")):
                if not (REPO / m.group(1)).is_file():
                    missing.append(f"{d.name} -> {m.group(1)}")
        self.assertEqual(missing, [], "文档引用了不存在的源文件：\n" +
                         "\n".join(missing))

    def test_every_backticked_symbol_exists_in_the_code(self):
        """反引号里的函数/类/常量必须在源码里出现过。

        \u26a0\ufe0f 这是「出现过」而不是「定义于」——文档也会引用参数名、
        局部变量名，要求它们是顶层定义会误报。但只要是编造的名字，
        全仓库搜不到，照样会被抓住。
        """
        missing = set()
        for d in method_docs():
            t = d.read_text(encoding="utf-8")
            syms = {m.group(1) for m in
                    re.finditer(r"`([A-Za-z_][A-Za-z0-9_]*)\(\)?`", t)}
            syms |= {m.group(1) for m in
                     re.finditer(r"`([A-Z][A-Za-z0-9_]{3,})`", t)}
            for s in syms:
                if re.search(rf"\b{re.escape(s)}\b", self.src) is None:
                    missing.add(f"{d.name} -> {s}")
        self.assertEqual(sorted(missing), [],
                         "文档引用了代码里找不到的符号：\n" +
                         "\n".join(sorted(missing)))


class TestDocumentedDefaultsMatchTheCode(unittest.TestCase):
    """\u2b50\u2b50 文档写的默认值，必须等于代码里真实的默认值。

    这条是本组最有价值的一条：它把两边绑在一起，
    改了代码不改文档（或反过来）都会红。

    做法：扫描文档里形如 ``key=1.23`` 的写法，再去所有函数/类的签名里
    找同名参数，逐个比对默认值。找不到同名参数的就跳过——
    文档也会写场景滑块的取值（那不是函数参数）。
    """

    #: 这些模块里找默认值就够了；全仓库 import 会拖慢且引入副作用
    MODULES = (
        "armctrl.core.kinematics", "armctrl.core.robot", "armctrl.core.gripper",
        "armctrl.control.impedance", "armctrl.control.momentum_observer",
        "armctrl.control.adaptive", "armctrl.control.mpc", "armctrl.control.ilc",
        "armctrl.control.grasp", "armctrl.control.visual_servo",
        "armctrl.control.servo", "armctrl.control.modal",
        "armctrl.planning.collision", "armctrl.planning.topp",
        "armctrl.identification.regressor",
    )

    @classmethod
    def setUpClass(cls):
        cls.defaults: dict[str, set[float]] = {}
        for name in cls.MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                                  # noqa: BLE001
                continue
            for _, obj in vars(mod).items():
                targets = []
                if inspect.isfunction(obj):
                    targets = [obj]
                elif inspect.isclass(obj):
                    targets = [v for _, v in vars(obj).items()
                               if inspect.isfunction(v)]
                for fn in targets:
                    try:
                        sig = inspect.signature(fn)
                    except (TypeError, ValueError):
                        continue
                    for pname, p in sig.parameters.items():
                        if isinstance(p.default, (int, float)) and \
                                not isinstance(p.default, bool):
                            cls.defaults.setdefault(pname, set()).add(
                                float(p.default))

    def test_the_defaults_table_is_not_empty(self):
        """前提：真的采集到了默认值，否则下一条会退化成空测试。"""
        self.assertGreater(
            len(self.defaults), 50,
            f"只采到 {len(self.defaults)} 个带默认值的参数，采集逻辑大概率坏了")

    #: ⚠️ 只比对**明确写了「默认」**的那些。
    #: 第一版把所有 `key=value` 都拿来比，立刻误报：
    #: 11 篇里 `null_gain=0.0` / `null_gain=1.0` 是一组**对照实验取值**，
    #: 不是默认值——文档并没有说错。判据必须只认「默认」二字。
    DEFAULT_CLAIM = re.compile(
        r"默认[^\n`]{0,6}`([a-z_][a-z0-9_]*) *= *(-?[0-9]+\.?[0-9]*)`")

    def test_every_documented_default_matches_the_signature(self):
        bad = []
        for d in method_docs():
            for m in self.DEFAULT_CLAIM.finditer(d.read_text(encoding="utf-8")):
                key, val = m.group(1), float(m.group(2))
                if key not in self.defaults:
                    continue                 # 不是函数参数（可能是滑块取值）
                if val not in self.defaults[key]:
                    bad.append(f"{d.name}: 文档写 {key}={m.group(2)}，"
                               f"代码里的默认值是 "
                               f"{sorted(self.defaults[key])}")
        self.assertEqual(bad, [], "文档默认值与代码不一致：\n" + "\n".join(bad))


class TestEveryDocumentedCommandIsRunnable(unittest.TestCase):
    """文档里的 `python -m unittest armctrl.tests.X` 必须指向真实模块。"""

    def test_every_referenced_test_module_exists(self):
        missing = []
        for d in method_docs():
            for m in re.finditer(r"armctrl\.tests\.(test_[a-z0-9_]+)",
                                 d.read_text(encoding="utf-8")):
                if not (REPO / "armctrl" / "tests" /
                        f"{m.group(1)}.py").is_file():
                    missing.append(f"{d.name} -> {m.group(1)}")
        self.assertEqual(missing, [], "文档引用了不存在的测试模块：\n" +
                         "\n".join(missing))

    def test_the_docs_actually_reference_some_tests(self):
        """前提检查：一条都没引用的话，上一条就是空测试。"""
        n = sum(len(re.findall(r"armctrl\.tests\.test_",
                               d.read_text(encoding="utf-8")))
                for d in method_docs())
        self.assertGreater(n, 10, f"只找到 {n} 处测试引用，正则大概率失效了")


class TestTheDocsStayPortableAndFreeOfProcessNoise(unittest.TestCase):
    """\u26a0\ufe0f 这两类内容进过一次公开文档，必须挡住第二次。"""

    def test_no_machine_specific_absolute_paths(self):
        """本机绝对路径写进「可运行命令」，别人照抄必然失败。"""
        bad = []
        for d in method_docs() + [DOC_DIR / "README.md"]:
            for pat in (r"/home/[a-z0-9_]+/", r"C:\\\\Users"):
                if re.search(pat, d.read_text(encoding="utf-8")):
                    bad.append(f"{d.name}（匹配 {pat}）")
        self.assertEqual(bad, [], "文档里出现了本机绝对路径：\n" +
                         "\n".join(bad))

    def test_no_writing_process_narration(self):
        """「本次核对的真实符号」这类是写作过程，不是给读者的内容。

        而且「本次实测 Ran N tests in T」里的耗时是某一次运行的快照，
        必然过期。
        """
        banned = ("本次核对的真实符号", "本次写文档前核对",
                  "本次综合验证实测", "未发现编造")
        bad = []
        for d in method_docs() + [DOC_DIR / "README.md"]:
            t = d.read_text(encoding="utf-8")
            for w in banned:
                if w in t:
                    bad.append(f"{d.name} 含「{w}」")
        self.assertEqual(bad, [], "文档里混进了写作过程自述：\n" +
                         "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
