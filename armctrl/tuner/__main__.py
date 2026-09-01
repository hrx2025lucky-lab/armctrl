"""在线调参台入口。

    export PYTHONPATH=.
    MUJOCO_GL=glfw python -m armctrl.tuner --scene impedance

会同时出现两样东西：

    **MuJoCo 原生窗口**   画面在这里看。可自由转视角、开关接触点/接触力等
                          可视化选项，还能用鼠标直接拖拽机械臂施加外力。
    **浏览器 WebUI**      http://127.0.0.1:8770/ ，读讲解、调参数、看数字与曲线。

关掉原生窗口即退出整个调参台。
"""

from __future__ import annotations

import argparse
import os

from armctrl.tuner.scenes import SCENES
from armctrl.tuner.server import serve


def main() -> None:
    ap = argparse.ArgumentParser(description="armctrl 在线调参台")
    ap.add_argument("--scene", default="impedance",
                    choices=[c.name for c in SCENES], help="启动时的场景")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1",
                    help="设成 0.0.0.0 可以从别的机器打开")
    ap.add_argument("--fps", type=int, default=60,
                    help="原生窗口的刷新率")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args()

    # 本机 egl/osmesa 不可用，离屏渲染必须走 glfw
    os.environ.setdefault("MUJOCO_GL", "glfw")
    serve(a.scene, port=a.port, host=a.host, fps=a.fps,
          open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
