"""静态图片推理测试（开发计划阶段五验收）。

对一张图片跑检测，打印类别/置信度/中心点，可选保存可视化结果。
PC 用 onnx 后端，真机用 acl 后端，后处理逻辑相同。

用法：
    python scripts/model/infer_test.py --backend onnx --model models/power_objects.onnx --image test.jpg
    python scripts/model/infer_test.py --backend acl  --model models/power_objects.om  --image test.jpg --save out.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds            # noqa: E402
from robotarm.detectors import build_detector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv5 静态图片推理测试")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "acl"])
    ap.add_argument("--model", required=True, help=".onnx 或 .om 模型路径")
    ap.add_argument("--image", required=True, help="测试图片")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--save", help="保存可视化结果路径（需要 opencv）")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("需要 opencv-python 读取图片。", file=sys.stderr)
        return 3

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"无法读取图片：{args.image}", file=sys.stderr)
        return 2

    det = build_detector(
        args.backend, args.model,
        img_size=args.img, conf_thres=args.conf, iou_thres=args.iou)
    results = det.detect(frame)

    print(f"检测到 {len(results)} 个目标：")
    for r in results:
        print(f"  {r.name:14s} ({ds.class_label(r.name)})  "
              f"conf={r.conf:.2f}  中心=({r.cx:.0f}, {r.cy:.0f})")

    if args.save:
        for r in results:
            cv2.circle(frame, (int(r.cx), int(r.cy)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{r.name} {r.conf:.2f}",
                        (int(r.cx) - 20, int(r.cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(args.save, frame)
        print(f"已保存可视化：{args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
