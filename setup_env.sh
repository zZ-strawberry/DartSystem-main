#!/usr/bin/env bash
# DartSystem 环境部署脚本（Linux）

set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "错误: 未找到 $PYTHON_BIN，请先安装 Python 3.10+" >&2
  exit 1
fi

echo "[1/4] 创建虚拟环境 .venv（若不存在）"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

echo "[2/4] 激活虚拟环境"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/4] 安装 Python 依赖"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] 检查系统依赖状态"
MISSING_PKGS=()

for pkg in libxcb-cursor0 libxkbcommon-x11-0 libgl1 libglib2.0-0; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "- $pkg: 已安装"
  else
    echo "- $pkg: 未安装"
    MISSING_PKGS+=("$pkg")
  fi
done

if command -v ffmpeg >/dev/null 2>&1; then
  echo "- ffmpeg: 已安装"
else
  echo "- ffmpeg: 未安装（如启用录屏，请安装: sudo apt install -y ffmpeg）"
fi

if [[ -d /opt/MVS/lib/64 ]]; then
  echo "- MVS 运行库: 已检测到 /opt/MVS/lib/64"
else
  echo "- MVS 运行库: 未检测到 /opt/MVS/lib/64（海康相机模式需要先安装 MVS SDK）"
fi

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  echo ""
  echo "检测到缺失系统库，执行以下命令可安装："
  echo "sudo apt update && sudo apt install -y ${MISSING_PKGS[*]}"
fi

echo ""
echo "环境部署完成。"
echo "启动命令: ./run.sh"
