#!/bin/bash
# DartSystem 启动脚本

cd "$(dirname "$0")"
export DISPLAY=:1
export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dart
python main.py
