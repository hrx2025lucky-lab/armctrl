"""第三方模型资产的路径解析。

为什么需要这个模块
------------------
本项目用 Franka Panda 作为被控对象，模型来自 DeepMind 的
`mujoco_menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_
（Apache-2.0）。那是一个独立的资产仓库，不随本仓库分发，所以它在每台机器上的
位置都不一样。原先各模块各自硬编码一条绝对路径，后果是换一台机器就全部失效。

解析策略
--------
1. 环境变量 ``MUJOCO_MENAGERIE``——显式指定，优先级最高；
2. 若干常见位置（用户主目录、仓库同级、仓库内 ``third_party/``）；
3. 都不存在时**不抛异常**，返回最后一个候选。

第 3 条是刻意的：本模块被多个模块在 import 期就求值的常量引用
（例如 ``planning.scene.PANDA_XML``）。如果解析失败即抛异常，没装 menagerie 的人
连 ``import`` 都会炸，连不碰模型的纯算法测试也跑不了。让路径以字符串形式传下去，
真正加载模型时再由 MuJoCo 报「文件不存在」，失败点更靠近原因。

需要提前明确校验时，调用 :func:`require_panda_xml`。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 仓库根目录（本文件位于 <repo>/armctrl/assets.py）
REPO_ROOT = Path(__file__).resolve().parents[1]

_ENV_VAR = "MUJOCO_MENAGERIE"

_CANDIDATES = (
    Path.home() / "mujoco_menagerie",
    REPO_ROOT.parent / "repos" / "mujoco_menagerie",
    REPO_ROOT.parent / "mujoco_menagerie",
    REPO_ROOT / "third_party" / "mujoco_menagerie",
)


def menagerie_root() -> Path:
    """返回 mujoco_menagerie 根目录。找不到时返回最后一个候选，不抛异常。"""
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser()
    for cand in _CANDIDATES:
        if cand.is_dir():
            return cand
    return _CANDIDATES[-1]


def panda_dir() -> Path:
    """Franka Panda 资产目录（含 panda.xml、assets/、scene.xml）。"""
    return menagerie_root() / "franka_emika_panda"


def panda_xml() -> str:
    """带标准平行夹爪的整机模型 XML 路径。"""
    return str(panda_dir() / "panda.xml")


def panda_nohand_xml() -> str:
    """不带夹爪的纯 7 自由度模型 XML 路径。"""
    return str(panda_dir() / "panda_nohand.xml")


def require_panda_xml() -> str:
    """同 :func:`panda_xml`，但文件不存在时立即抛出带安装指引的异常。"""
    path = panda_xml()
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"找不到 Panda 模型：{path}\n"
            f"本项目依赖 mujoco_menagerie（Apache-2.0），需自行获取：\n"
            f"    git clone https://github.com/google-deepmind/mujoco_menagerie.git\n"
            f"    export {_ENV_VAR}=/path/to/mujoco_menagerie\n"
            f"已尝试的位置：{', '.join(str(c) for c in _CANDIDATES)}"
        )
    return path
