"""按类别采集电力物品图片（开发计划阶段四）。

用 USB 摄像头拍照，按类别存到 ``dataset/raw/<类别>/``，统一缩放 640x480。
类别列表来自 categories.yaml（见 robotarm.dataset.class_names）。

用法：
    # 交互选类别后开始采集
    python scripts/dataset/collect_images.py --class resistor
    # 自动每 0.5s 拍一张，共 200 张
    python scripts/dataset/collect_images.py --class wrench --auto --interval 0.5 --count 200
    # 列出所有可用类别
    python scripts/dataset/collect_images.py --list

采集时按键：
    空格 = 拍一张    a = 切换自动拍    q/ESC = 退出

⚠️ 需要摄像头。本脚本在有摄像头的机器（PC 或开发板）上运行。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds  # noqa: E402


def existing_count(class_dir: str) -> int:
    if not os.path.isdir(class_dir):
        return 0
    return sum(1 for f in os.listdir(class_dir) if ds.is_image(f))


def save_frame(frame, class_dir: str, idx: int, cls: str) -> str:
    import cv2
    os.makedirs(class_dir, exist_ok=True)
    path = os.path.join(class_dir, f"{cls}_{idx:04d}.jpg")
    cv2.imwrite(path, frame)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="电力物品图片采集")
    ap.add_argument("--class", dest="cls", help="类别名（见 --list）")
    ap.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    ap.add_argument("--auto", action="store_true", help="自动间隔拍照")
    ap.add_argument("--interval", type=float, default=0.5, help="自动拍间隔秒")
    ap.add_argument("--count", type=int, default=0, help="采集张数上限，0=不限")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--list", action="store_true", help="列出可用类别后退出")
    args = ap.parse_args()

    names = ds.class_names()
    if args.list or not args.cls:
        print("可用类别（共 %d 类）：" % len(names))
        for i, n in enumerate(names):
            print(f"  [{i}] {n} ({ds.class_label(n)})")
        if not args.cls:
            print("\n请用 --class <类别名> 指定要采集的类别。")
        return 0

    if args.cls not in names:
        print(f"未知类别: {args.cls}。用 --list 查看可用类别。", file=sys.stderr)
        return 2

    try:
        import cv2
    except ImportError:
        print("需要 opencv-python：pip install opencv-python", file=sys.stderr)
        return 3

    class_dir = os.path.join(ds.raw_dir(), args.cls)
    start_idx = existing_count(class_dir)
    print(f"采集类别 [{args.cls}] -> {class_dir}（已有 {start_idx} 张）")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}", file=sys.stderr)
        return 4

    auto = args.auto
    idx = start_idx
    saved = 0
    last_shot = 0.0
    print("空格=拍照  a=切换自动  q/ESC=退出")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取帧失败", file=sys.stderr)
                break
            frame = cv2.resize(frame, (args.width, args.height))

            view = frame.copy()
            mode = "AUTO" if auto else "MANUAL"
            cv2.putText(view, f"{args.cls} | {mode} | saved={saved} total={idx}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("collect", view)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()
            shoot = False
            if key == ord(" "):
                shoot = True
            elif key == ord("a"):
                auto = not auto
            elif key in (ord("q"), 27):
                break
            if auto and (now - last_shot) >= args.interval:
                shoot = True

            if shoot:
                save_frame(frame, class_dir, idx, args.cls)
                idx += 1
                saved += 1
                last_shot = now
                if args.count and saved >= args.count:
                    print(f"已达上限 {args.count} 张")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"完成：本次采集 {saved} 张，类别 {args.cls} 现共 {idx} 张。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
