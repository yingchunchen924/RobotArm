"""数据集工具模块 —— 类别顺序与路径的单一真相源。

YOLO 训练要求类别有**稳定的整数 id 顺序**（class 0,1,2,…）。本模块从
``config/categories.yaml`` 的 ``category_to_zone`` 派生这个顺序，使得：

- 采集脚本、标注、power_objects.yaml、训练、推理、抓取映射 全部用同一份类别表；
- 改类别只改 categories.yaml 一处，不会出现"数据集类别和系统类别对不上"。

同时提供数据集目录路径常量与 YOLO 标注行的解析/校验工具。无重型依赖（仅 yaml），
可单元测试。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import config_loader as cl


# ---- 类别顺序（单一真相源）------------------------------------------------

def class_names() -> List[str]:
    """返回有序类别名列表，索引即 YOLO class id。

    顺序取自 categories.yaml 的 ``category_to_zone`` 插入顺序（YAML 保序）。
    """
    cfg = cl.get_categories_config()
    mapping = cfg.get("category_to_zone", {})
    return list(mapping.keys())


def num_classes() -> int:
    return len(class_names())


def name_to_id() -> Dict[str, int]:
    return {name: i for i, name in enumerate(class_names())}


def id_to_name() -> Dict[int, str]:
    return {i: name for i, name in enumerate(class_names())}


def class_label(name: str) -> str:
    """类别的中文显示名（category_label），缺失时回退英文名。"""
    cfg = cl.get_categories_config()
    return cfg.get("category_label", {}).get(name, name)


# ---- 数据集路径 -----------------------------------------------------------

def dataset_root() -> str:
    return os.path.join(cl.find_project_root(), "dataset")


def raw_dir() -> str:
    return os.path.join(dataset_root(), "raw")


def labeled_dir() -> str:
    return os.path.join(dataset_root(), "labeled")


def yaml_dir() -> str:
    return os.path.join(dataset_root(), "yaml")


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def is_image(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTS)


def label_path_for_image(image_path: str, images_dir: str, labels_dir: str) -> str:
    """给定图片路径，返回其对应的 YOLO 标注 .txt 路径。

    YOLO 约定：images/xxx.jpg <-> labels/xxx.txt（同名不同目录/后缀）。
    """
    rel = os.path.relpath(image_path, images_dir)
    stem = os.path.splitext(rel)[0]
    return os.path.join(labels_dir, stem + ".txt")


# ---- YOLO 标注行解析 ------------------------------------------------------

@dataclass
class BBox:
    """一条 YOLO 标注：class_id + 归一化中心点和宽高（均在 [0,1]）。"""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float


def parse_label_line(line: str) -> BBox:
    """解析一行 YOLO 标注 ``class_id cx cy w h``。

    :raises ValueError: 字段数不对或无法转换。
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"标注应为 5 个字段，实得 {len(parts)}: {line!r}")
    cid = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:])
    return BBox(cid, cx, cy, w, h)


def bbox_in_range(b: BBox) -> bool:
    """校验归一化坐标是否都在 [0,1]。"""
    return all(0.0 <= v <= 1.0 for v in (b.cx, b.cy, b.w, b.h)) and b.w > 0 and b.h > 0
