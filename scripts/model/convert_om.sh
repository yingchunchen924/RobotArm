#!/usr/bin/env bash
# ONNX -> .om 转换（开发计划阶段五）。
# ⚠️ 仅在 Atlas 200I DK A2 开发板（已装 CANN/ATC）上运行。PC 上没有 atc 命令。
#
# 用法：bash scripts/model/convert_om.sh power_objects.onnx power_objects
set -e

ONNX="${1:-models/power_objects.onnx}"
OUT="${2:-models/power_objects}"          # 不带 .om 后缀，atc 自动追加
INPUT_NAME="${3:-images}"                  # ONNX 输入名，导出时默认 images
SOC="${4:-Ascend310B1}"                    # 用 npu-smi info 确认实际芯片型号

if ! command -v atc >/dev/null 2>&1; then
  echo "未找到 atc 命令。请在已安装 CANN 工具包的 Atlas 开发板上运行。" >&2
  echo "若已安装，请先 source 昇腾环境，如：source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
  exit 1
fi

if [ ! -f "$ONNX" ]; then
  echo "找不到 ONNX 模型：$ONNX" >&2
  exit 2
fi

echo "转换：$ONNX -> ${OUT}.om  (soc=$SOC, input=$INPUT_NAME)"
atc \
  --model="$ONNX" \
  --framework=5 \
  --output="$OUT" \
  --input_format=NCHW \
  --input_shape="${INPUT_NAME}:1,3,640,640" \
  --soc_version="$SOC"

echo "完成：${OUT}.om"
echo "把 ${OUT}.om 用于 AclDetector，并确认 config/categories.yaml 类别顺序与训练一致。"
