#!/usr/bin/env bash
# 真机一键启动（开发计划阶段八）。
# ⚠️ 仅在 Atlas 200I DK A2 开发板上运行。按顺序拉起：ROS2 运动学服务 -> 识别抓取
#    主程序 -> Web 可视化后端。每个组件在独立后台进程，日志输出到 logs/。
#
# 前置：已按 scripts/setup_env.sh 配好环境，相机已标定，power_objects.om 已就位，
#       power_arm_control 已实现 SerialArmDriver / Ros2Kinematics。
#
# 用法：bash scripts/start_all.sh [web_port]
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARM_DIR="/home/HwHiAiUser/E2ESamples/src/E2E-Sample/ros2_robot_arm"
WEB_PORT="${1:-8000}"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

PIDS=()
cleanup() {
  echo; echo "停止所有组件…"
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "已退出。"
}
trap cleanup INT TERM EXIT

wait_port() {  # wait_port <port> <timeout_s>
  local port="$1" timeout="${2:-15}" i=0
  while ! (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    sleep 1; i=$((i+1)); [ "$i" -ge "$timeout" ] && return 1
  done
  return 0
}

echo "============================================================"
echo "  电力库房机械臂系统 —— 真机一键启动"
echo "============================================================"

# 1) ROS2 运动学服务
echo "[1/3] 启动 ROS2 运动学服务 dofbot_server…"
( cd "$ARM_DIR" && source setenv.sh && ros2 run dofbot_moveit dofbot_server ) \
  > "$LOGS/ros2_server.log" 2>&1 &
PIDS+=($!)
sleep 5   # 给服务初始化时间

# 2) 识别抓取主程序（电力物品分类抓取）
echo "[2/3] 启动识别抓取主程序（acl 后端）…"
( cd "$ROOT" && python scripts/run_power_sorting.py \
    --backend acl --model models/power_objects.om ) \
  > "$LOGS/power_sorting.log" 2>&1 &
PIDS+=($!)
sleep 3

# 3) Web 可视化后端
echo "[3/3] 启动 Web 可视化后端（端口 $WEB_PORT）…"
( cd "$ROOT/web/backend" && uvicorn app:app --host 0.0.0.0 --port "$WEB_PORT" ) \
  > "$LOGS/web.log" 2>&1 &
PIDS+=($!)

if wait_port "$WEB_PORT" 15; then
  echo "------------------------------------------------------------"
  echo "  全部启动完成。"
  echo "  Web 界面: http://<开发板IP>:$WEB_PORT/"
  echo "  日志目录: $LOGS/"
  echo "  按 Ctrl+C 停止所有组件。"
  echo "------------------------------------------------------------"
else
  echo "Web 端口未就绪，请查看 $LOGS/web.log" >&2
fi

# 保持前台，等待中断
wait
