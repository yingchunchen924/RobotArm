#!/usr/bin/env bash
# 启动色块堆叠（手册 3.5）。在 Atlas 开发板上、两个终端分别运行。
# ⚠️ 仅开发板可运行。
set -e
ARM_DIR="/home/HwHiAiUser/E2ESamples/src/E2E-Sample/ros2_robot_arm"

echo "终端1：启动运动学服务"
echo "  cd $ARM_DIR && source setenv.sh && ros2 run dofbot_moveit dofbot_server"
echo
echo "终端2：启动色块堆叠"
echo "  cd $ARM_DIR && source setenv.sh && ros2 run robot_arm_color_stacking color_stacking"
echo
echo "堆叠层位由 config/stacking.yaml 配置（取代手册硬编码）。"
