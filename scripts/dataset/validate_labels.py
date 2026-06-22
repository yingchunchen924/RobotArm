"""校验 YOLO 格式标注（开发计划阶段四验收：「数据集能被 YOLO 训练脚本正常读取」）。

检查项：
    - 标注行格式（5 字段、可解析）
    - class_id 是否越界（>= 类别数 或 < 0）
    - 归一化坐标是否在 [0,1] 且 w/h>0
    - 图片缺少对应标注（漏标）
    - 标注缺少对应图片（孤儿标注）
    - 空标注文件（可能漏标，警告）

可作为脚本运行，也可 import validate_pair / validate_dirs 在测试中调用。

用法：
    # 校验一对 images/ labels/ 目录
    python scripts/dataset/validate_labels.py --images dataset/labeled/images --labels dataset/labeled/labels
    # 校验 YOLO 划分后的整个数据集（train/val/test 各自的 images+labels）
    python scripts/dataset/validate_labels.py --root dataset/yolo
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds  # noqa: E402


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_images: int = 0
    checked_labels: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "Report") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.checked_images += other.checked_images
        self.checked_labels += other.checked_labels

    def summary(self) -> str:
        head = (f"检查图片 {self.checked_images} 张，标注 {self.checked_labels} 个；"
                f"错误 {len(self.errors)}，警告 {len(self.warnings)}")
        lines = [head]
        for e in self.errors:
            lines.append(f"  [错误] {e}")
        for w in self.warnings:
            lines.append(f"  [警告] {w}")
        return "\n".join(lines)


def validate_label_file(label_path: str, num_classes: int) -> List[str]:
    """校验单个标注文件内容，返回错误信息列表（空=通过）。"""
    errs: List[str] = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i, ln in enumerate(lines, 1):
        try:
            box = ds.parse_label_line(ln)
        except ValueError as e:
            errs.append(f"{label_path}:{i} 格式错误：{e}")
            continue
        if box.class_id < 0 or box.class_id >= num_classes:
            errs.append(
                f"{label_path}:{i} class_id={box.class_id} 越界（应 0..{num_classes - 1}）")
        if not ds.bbox_in_range(box):
            errs.append(
                f"{label_path}:{i} 坐标越界或非法："
                f"cx={box.cx} cy={box.cy} w={box.w} h={box.h}")
    return errs


def validate_dirs(images_dir: str, labels_dir: str, num_classes: int) -> Report:
    """校验一对 images/ labels/ 目录。"""
    rep = Report()
    if not os.path.isdir(images_dir):
        rep.errors.append(f"图片目录不存在：{images_dir}")
        return rep
    if not os.path.isdir(labels_dir):
        rep.errors.append(f"标注目录不存在：{labels_dir}")
        return rep

    images = [f for f in sorted(os.listdir(images_dir)) if ds.is_image(f)]
    label_files = set(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    used_labels = set()

    for img in images:
        rep.checked_images += 1
        stem = os.path.splitext(img)[0]
        lbl_name = stem + ".txt"
        lbl_path = os.path.join(labels_dir, lbl_name)
        if not os.path.isfile(lbl_path):
            rep.errors.append(f"图片缺少标注：{img}")
            continue
        used_labels.add(lbl_name)
        rep.checked_labels += 1
        if os.path.getsize(lbl_path) == 0:
            rep.warnings.append(f"空标注文件（可能漏标）：{lbl_name}")
            continue
        rep.errors.extend(validate_label_file(lbl_path, num_classes))

    # 孤儿标注：有 .txt 没对应图
    for orphan in sorted(label_files - used_labels):
        rep.errors.append(f"标注缺少对应图片：{orphan}")

    return rep


def validate_root(root: str, num_classes: int) -> Report:
    """校验 YOLO 划分后的数据集（root 下含 train/val/test，每个含 images+labels）。"""
    rep = Report()
    found_split = False
    for split in ("train", "val", "test"):
        img_d = os.path.join(root, split, "images")
        lbl_d = os.path.join(root, split, "labels")
        if os.path.isdir(img_d):
            found_split = True
            sub = validate_dirs(img_d, lbl_d, num_classes)
            if sub.errors or sub.warnings or sub.checked_images:
                rep.merge(sub)
    if not found_split:
        rep.errors.append(f"{root} 下未找到 train/val/test 划分结构")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 标注格式校验")
    ap.add_argument("--images", help="图片目录")
    ap.add_argument("--labels", help="标注目录")
    ap.add_argument("--root", help="YOLO 划分根目录（含 train/val/test）")
    args = ap.parse_args()

    nc = ds.num_classes()
    print(f"类别数 nc={nc}：{ds.class_names()}")

    if args.root:
        rep = validate_root(args.root, nc)
    elif args.images and args.labels:
        rep = validate_dirs(args.images, args.labels, nc)
    else:
        ap.error("需提供 --root 或 (--images 且 --labels)")
        return 2

    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
