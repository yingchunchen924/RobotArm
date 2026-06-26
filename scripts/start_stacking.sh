#!/usr/bin/env bash
# 启动色块堆叠（手册 4.5）。在 Atlas 开发板上运行。
set -e

BOARD_HOME="/home/HwHiAiUser"
ARM_DIR="$BOARD_HOME/E2ESamples/src/E2E-Sample/ros2_robot_arm"
SCRIPT="$BOARD_HOME/RobotArm/scripts/board/color_sort_ros.py"

echo "1) 确认运动学服务已启动："
echo "   cd $ARM_DIR && source setenv.sh && ros2 run dofbot_moveit dofbot_server"
echo
echo "2) 只规划不抓取，检查识别点/逆解/堆叠层位："
echo "   python3 $SCRIPT stack --color any --layers 4"
echo
echo "3) 真正执行连续堆叠："
echo "   python3 $SCRIPT stack --color any --layers 4 --execute"
echo
echo "提示：如堆叠区里的已放方块被再次识别，用 --exclude-region x1,y1,x2,y2 排除该画面区域。"
