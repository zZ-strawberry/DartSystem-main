#!/bin/bash
# HSV 调试工具启动脚本

cd "$(dirname "$0")"
export DISPLAY=:1
export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dart

if [ "$1" == "--hik" ]; then
    python opencv_green_detection.py --debug --hik
elif [ "$1" == "--video" ] && [ -n "$2" ]; then
    python opencv_green_detection.py --debug --video "$2"
else
    echo "HSV 调试工具使用方法:"
    echo "  视频模式:   ./debug_hsv.sh --video <视频路径>"
    echo "  海康相机:   ./debug_hsv.sh --hik"
    echo ""
    echo "示例:"
    echo "  ./debug_hsv.sh --video test.mp4"
    echo "  ./debug_hsv.sh --hik"
fi
