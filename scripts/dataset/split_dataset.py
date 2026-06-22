"""数据集划分（开发计划阶段四：划分训练/验证/测试集）。

把已标注的 images/ + labels/ 按比例随机划分为 train/val/test，复制到 YOLO 约定的
目录结构，并保证：
    - 固定随机种子 -> 可复现；
    - 图片与其标注成对移动，不会拆散；
    - 没有对应标注的图片跳过并提示。

划分核心 split_files 不做 IO，便于单测；CLI 负责实际复制。

用法：
    python scripts/dataset/split_dataset.py \
        --images dataset/labeled/images --labels dataset/labeled/labels \
        --out dataset/yolo --train 0.7 --val 0.2 --test 0.1 --seed 42

输出结构：
    dataset/yolo/
      ├── train/{images,labels}/
      ├── val/{images,labels}/
      └── test/{images,labels}/
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from typing import Dict, List, Sequence

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds  # noqa: E402


def split_files(
    items: Sequence[str],
    train: float,
    val: float,
    test: float,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """把 items 按比例随机划分为 train/val/test，返回 {split: [items]}。

    纯函数：输入文件名列表，输出划分结果。比例之和需接近 1。
    采用固定 seed 的本地 Random 实例，保证可复现且不污染全局随机状态。
    """
    total_ratio = train + val + test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train+val+test 应为 1.0，实得 {total_ratio}")

    ordered = sorted(items)              # 先排序，消除文件系统顺序差异
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(n * train)
    n_val = int(n * val)
    # test 取剩余，避免取整丢样本
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 数据集划分")
    ap.add_argument("--images", required=True, help="已标注图片目录")
    ap.add_argument("--labels", required=True, help="标注目录")
    ap.add_argument("--out", default=os.path.join(ds.dataset_root(), "yolo"),
                    help="输出根目录")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="移动而非复制")
    args = ap.parse_args()

    if not os.path.isdir(args.images):
        print(f"图片目录不存在：{args.images}", file=sys.stderr)
        return 2

    images = [f for f in os.listdir(args.images) if ds.is_image(f)]
    # 只保留有标注的图片
    paired, skipped = [], []
    for img in images:
        stem = os.path.splitext(img)[0]
        if os.path.isfile(os.path.join(args.labels, stem + ".txt")):
            paired.append(img)
        else:
            skipped.append(img)
    if skipped:
        print(f"跳过 {len(skipped)} 张无标注图片（如 {skipped[:3]} …）")
    if not paired:
        print("没有可划分的已标注图片。", file=sys.stderr)
        return 1

    splits = split_files(paired, args.train, args.val, args.test, args.seed)
    op = shutil.move if args.move else shutil.copy2

    for split, files in splits.items():
        img_out = os.path.join(args.out, split, "images")
        lbl_out = os.path.join(args.out, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img in files:
            stem = os.path.splitext(img)[0]
            op(os.path.join(args.images, img), os.path.join(img_out, img))
            op(os.path.join(args.labels, stem + ".txt"),
               os.path.join(lbl_out, stem + ".txt"))
        print(f"  {split}: {len(files)} 张 -> {os.path.join(args.out, split)}")

    print(f"划分完成（seed={args.seed}，可复现）。输出：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
