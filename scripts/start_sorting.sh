#!/usr/bin/env bash
# 启动色块分拣（手册 3.4）。在 Atlas 开发板上、两个终端分别运行。
# ⚠️ 仅开发板可运行。
set -e
ARM_DIR="/home/HwHiAiUser/E2ESamples/src/E2E-Sample/ros2_robot_arm"

echo "终端1：启动运动学服务"
echo "  cd $ARM_DIR && source setenv.sh && ros2 run dofbot_moveit dofbot_server"
echo
echo "终端2：启动色块分拣"
echo "  cd $ARM_DIR && source setenv.sh && ros2 run dofbot_garbage_yolov5 block_cls"
echo
echo "提示：需使用带标定板框的地图，摄像头对准标定框。"
