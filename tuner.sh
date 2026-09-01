#!/usr/bin/env bash
#
# armctrl 在线调参台启动脚本
#
#     ./tuner.sh                  默认场景（阻抗控制）
#     ./tuner.sh observer         指定场景
#     ./tuner.sh --list           列出全部场景
#
# 为什么需要这个脚本
# ----------------
# 直接敲 `python -m armctrl.tuner` 常常会失败，有三个原因，缺一不可：
#   1. 有些机器上没有 `python` 这个命令，只有 `python3`；
#   2. 就算用 python3，系统解释器里通常没有 mujoco / pinocchio / osqp，
#      必须用装好依赖的虚拟环境；
#   3. 还要 PYTHONPATH 指到仓库根目录，以及设置 MUJOCO_GL。
# 把这三件事固定下来，就不会再踩了。
#
# 解释器按以下顺序解析，都可以用环境变量覆盖：
#   $ARMCTRL_PYTHON  →  <仓库>/.venv  →  <仓库同级>/envs/armctrl  →  python3

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${ARMCTRL_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in \
      "$HERE/.venv/bin/python" \
      "$HERE/../envs/armctrl/bin/python" \
      "$(command -v python3 || true)"; do
    if [[ -n "$cand" && -x "$cand" ]]; then PY="$cand"; break; fi
  done
fi

if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "找不到可用的 Python 解释器。" >&2
  echo "请设置 ARMCTRL_PYTHON 指向装有 mujoco / pinocchio / osqp 的解释器，例如：" >&2
  echo "    ARMCTRL_PYTHON=/path/to/venv/bin/python ./tuner.sh" >&2
  exit 1
fi

cd "$HERE"
export PYTHONPATH="$HERE"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
# 没有 DISPLAY 就开不出原生窗口；留出覆盖入口，缺省沿用本机的 :1。
export DISPLAY="${DISPLAY:-:1}"

if [[ "${1:-}" == "--list" ]]; then
  "$PY" - <<'EOF'
from armctrl.tuner.scenes import SCENES
print("可用场景：")
for c in SCENES:
    print("  %-10s %s" % (c.name, c.title))
EOF
  exit 0
fi

SCENE="${1:-impedance}"
shift || true

# 端口被上一次没退干净的进程占着时，先说清楚，别让用户对着
# "Address already in use" 猜。
PORT=8770
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已被占用——很可能上一次的调参台还开着。"
  echo "先关掉那个 MuJoCo 窗口，或执行："
  echo "  kill \$(ss -ltnp 2>/dev/null | grep ':${PORT} ' | grep -oP 'pid=\\K[0-9]+' | head -1)"
  exit 1
fi

echo "启动场景：${SCENE}"
echo "画面在 MuJoCo 原生窗口，讲解与调参在 http://127.0.0.1:${PORT}/"
exec "$PY" -m armctrl.tuner --scene "$SCENE" "$@"
