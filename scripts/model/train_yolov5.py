"""训练 YOLOv5 电力物品检测模型（开发计划阶段五）的封装脚本。

本脚本**不内置 YOLOv5**，而是封装对官方 yolov5 仓库 train.py 的调用，统一参数与路径，
并自动使用本项目生成的 dataset/yaml/power_objects.yaml（类别来自 categories.yaml）。

前置：
    git clone https://github.com/ultralytics/yolov5
    pip install -r yolov5/requirements.txt
    # 数据集已采集/标注/划分，并生成 power_objects.yaml（见 dataset/README.md）

用法：
    python scripts/model/train_yolov5.py --yolov5 /path/to/yolov5 \
        --data dataset/yaml/power_objects.yaml --epochs 100 --batch 16 --img 640

⚠️ 训练需要 GPU 与已标注数据集；本脚本在有这些条件的机器上运行（通常不是 Atlas 板）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLOv5 电力物品模型")
    ap.add_argument("--yolov5", required=True, help="官方 yolov5 仓库路径")
    ap.add_argument("--data", default=os.path.join(ds.yaml_dir(), "power_objects.yaml"),
                    help="数据集配置 yaml")
    ap.add_argument("--weights", default="yolov5s.pt", help="预训练权重（迁移学习）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--name", default="power_objects", help="run 名称")
    ap.add_argument("--device", default="", help="cuda 设备，如 0；空=自动")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    train_py = os.path.join(args.yolov5, "train.py")
    if not os.path.isfile(train_py):
        print(f"找不到 {train_py}，请先 clone yolov5。", file=sys.stderr)
        return 2
    if not os.path.isfile(args.data):
        print(f"数据集配置不存在：{args.data}\n"
              f"请先运行 scripts/dataset/gen_dataset_yaml.py 生成。", file=sys.stderr)
        return 2

    cmd = [
        sys.executable, train_py,
        "--data", os.path.abspath(args.data),
        "--weights", args.weights,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch),
        "--img", str(args.img),
        "--name", args.name,
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("训练命令：\n  " + " ".join(cmd))
    print(f"\n类别数 nc={ds.num_classes()}：{ds.class_names()}")
    print(f"训练完成后，最佳权重在 {args.yolov5}/runs/train/{args.name}/weights/best.pt")
    print("下一步导出 ONNX：参见 scripts/model/export_onnx.md")

    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=args.yolov5)


if __name__ == "__main__":
    raise SystemExit(main())
