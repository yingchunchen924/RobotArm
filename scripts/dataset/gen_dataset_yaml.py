"""生成 YOLOv5 数据集配置 power_objects.yaml（开发计划阶段四产出）。

类别名与数量从 categories.yaml 派生（robotarm.dataset），保证与系统其余部分一致。
不手写类别，避免不同步。

用法：
    python scripts/dataset/gen_dataset_yaml.py --root dataset/yolo --out dataset/yaml/power_objects.yaml

生成内容（YOLOv5 约定）：
    path:  数据集根
    train/val/test: 各划分的 images 目录（相对 path）
    nc:    类别数
    names: 类别名列表（索引=class id）
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds  # noqa: E402


def build_yaml_text(root: str, names) -> str:
    """构造 power_objects.yaml 文本。手写以保证 names 顺序与注释可控。"""
    lines = [
        "# YOLOv5 电力物品数据集配置",
        "# 由 scripts/dataset/gen_dataset_yaml.py 从 config/categories.yaml 自动生成。",
        "# 请勿手改类别；改类别请改 categories.yaml 后重新生成。",
        "",
        f"path: {root}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}   # {ds.class_label(n)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 power_objects.yaml")
    ap.add_argument("--root", default=os.path.join(ds.dataset_root(), "yolo"),
                    help="数据集根（含 train/val/test）")
    ap.add_argument("--out", default=os.path.join(ds.yaml_dir(), "power_objects.yaml"),
                    help="输出 yaml 路径")
    args = ap.parse_args()

    names = ds.class_names()
    if not names:
        print("categories.yaml 中没有类别，无法生成。", file=sys.stderr)
        return 1

    text = build_yaml_text(os.path.abspath(args.root), names)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"已生成 {args.out}（nc={len(names)}）")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
