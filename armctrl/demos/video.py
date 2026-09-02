"""录像输出：同时写 mp4 与 webm。

为什么要写两份
--------------
本机 Ubuntu 22.04 的默认播放器 Totem 走 GStreamer，而这台机器上只装了
`gstreamer1.0-plugins-good`，**没有任何 H.264 解码器**
（缺 `gstreamer1.0-libav` 提供的 avdec_h264），所以 mp4 打开是一个破损图标。

已装的插件里有 `vp8dec` / `vp9dec` / `matroskademux`，因此 **WebM(VP9) 可以直接播**。

    mp4 (H.264)   通用格式，分享、发给别人、传网页都合适，本机播不了
    webm (VP9)    本机 Totem 直接能播，体积相近

彻底修复本机 mp4 的办法是（需要 sudo 密码）：

    sudo apt install gstreamer1.0-libav gstreamer1.0-plugins-bad

在那之前，本机看 webm，或者用已经装好的 ffplay：

    ffplay -autoexit media/videos/planning.mp4
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

#: 仓库根（本文件位于 <repo>/armctrl/demos/video.py）
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: ⭐ 录像统一输出到 <repo>/media/videos/。
#: 为什么不放在 armctrl/ 里：那是**代码包**，二进制产物混进去会让
#: `pip install .` 之类的打包把几十 MB 视频一起装走，也让「代码有多少行」
#: 这类统计失真。产物归产物，代码归代码。
VIDEO_DIR = REPO_ROOT / "media" / "videos"


def video_path(name: str) -> str:
    """录像的标准输出路径；目录不存在就建。"""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return str(VIDEO_DIR / f"{name}.mp4")


def write_video(frames, out_path: str, fps: int = 50,
                also_webm: bool = True) -> list[str]:
    """把帧序列写成 mp4（H.264）以及同名 webm（VP9）。

    frames    RGB 图像数组的序列
    out_path  mp4 输出路径；webm 用同一主干名
    返回实际写出的文件路径列表。
    """
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    for i, fr in enumerate(frames):
        imageio.imwrite(f"{tmpdir}/f{i:05d}.png", fr)

    written = []
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", f"{tmpdir}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", out_path],
        check=True,
    )
    written.append(out_path)

    if also_webm:
        webm = os.path.splitext(out_path)[0] + ".webm"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", f"{tmpdir}/f%05d.png", "-c:v", "libvpx-vp9",
             "-b:v", "0", "-crf", "32", "-deadline", "good", "-cpu-used", "4",
             "-row-mt", "1", "-pix_fmt", "yuv420p", webm],
            check=True,
        )
        written.append(webm)
    return written


def transcode_to_webm(mp4_path: str) -> str:
    """把已有的 mp4 转成同名 webm（本机 Totem 能播的格式）。"""
    webm = os.path.splitext(mp4_path)[0] + ".webm"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path,
         "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
         "-deadline", "good", "-cpu-used", "4", "-row-mt", "1",
         "-pix_fmt", "yuv420p", webm],
        check=True,
    )
    return webm


if __name__ == "__main__":
    import glob
    import sys

    targets = sys.argv[1:] or sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "videos", "*.mp4")))
    for mp4 in targets:
        out = transcode_to_webm(mp4)
        print(f"  {os.path.basename(mp4)}  ->  {os.path.basename(out)} "
              f"({os.path.getsize(out)/1e6:.2f} MB)")
