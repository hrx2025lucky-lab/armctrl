"""讲义文档的定位与可用性判断。

为什么需要这个模块
------------------
`armctrl_讲解/` 是与代码同步维护的技术讲义，其中 8 个测试会**先用数值独立测出
源码的真实行为，再要求讲义里的说法与之一致**——这是本项目防止「代码对了、文档过期」
的机制（见根 README）。

但那批讲义同时混入了大量个人学习笔记，因此不随公开仓库分发。
依赖它的测试用 :data:`requires_docs` 装饰，文档缺失时**自动 skip 而不是失败**：
测试代码本身保留在仓库里（机制可见），但不会因为缺少非公开材料而报红。
"""

from __future__ import annotations

import pathlib
import unittest

#: 讲义根目录（<repo>/armctrl_讲解）
DOCS_ROOT = pathlib.Path(__file__).resolve().parents[2] / "armctrl_讲解"


def has_docs() -> bool:
    """讲义目录是否可用。"""
    return DOCS_ROOT.is_dir()


def doc_path(name: str) -> pathlib.Path:
    """返回讲义文件路径，例如 ``doc_path("01_阻抗控制.md")``。"""
    return DOCS_ROOT / name


def read_doc(name: str) -> str:
    """读取讲义全文。调用方应先用 :data:`requires_docs` 保护。"""
    return doc_path(name).read_text(encoding="utf-8")


#: 用于装饰依赖讲义的 TestCase / 测试方法。
requires_docs = unittest.skipUnless(
    has_docs(), "armctrl_讲解/ 不随公开仓库分发，跳过文档一致性校验"
)
