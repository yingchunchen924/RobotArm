"""统一配置加载。

集中读取 ``config/*.yaml``，对外提供便捷访问函数，使其余模块不直接碰文件路径，
也不在代码里散落魔数（所有可变量都在 yaml 中）。

设计要点：
- 自动定位项目根目录（向上找含 ``config`` 目录的祖先），不依赖当前工作目录。
- 带内存缓存，重复读取同一配置不重复解析。
- 缺字段时由调用方用 ``cfg.get(..., default)`` 兜底；缺文件时抛出明确错误。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 依赖缺失时给清晰提示
    raise ImportError(
        "缺少 PyYAML，请先安装：pip install pyyaml"
    ) from exc


def find_project_root(start: str | None = None) -> str:
    """从 ``start`` 起向上查找包含 ``config`` 子目录的目录，作为项目根。

    找不到时回退到本文件所在仓库根（src/robotarm 的上两级）。
    """
    if start is None:
        start = os.path.dirname(os.path.abspath(__file__))
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, "config")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # 到达文件系统根
            break
        cur = parent
    # 回退：本文件位于 <root>/src/robotarm/config_loader.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def config_dir() -> str:
    """返回 ``config`` 目录的绝对路径。"""
    return os.path.join(find_project_root(), "config")


@lru_cache(maxsize=None)
def load_config(name: str) -> Dict[str, Any]:
    """加载 ``config/<name>.yaml`` 并返回字典（带缓存）。

    :param name: 配置名，不含扩展名，如 ``"arm"`` / ``"grasp"``。
    :raises FileNotFoundError: 配置文件不存在。
    """
    path = os.path.join(config_dir(), f"{name}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def clear_cache() -> None:
    """清空配置缓存（改了 yaml 后需要重新加载时调用）。"""
    load_config.cache_clear()


# ---- 便捷访问函数 ----------------------------------------------------------

def get_arm_config() -> Dict[str, Any]:
    return load_config("arm")


def get_grasp_config() -> Dict[str, Any]:
    return load_config("grasp")


def get_stacking_config() -> Dict[str, Any]:
    return load_config("stacking")


def get_categories_config() -> Dict[str, Any]:
    return load_config("categories")


def get_web_config() -> Dict[str, Any]:
    return load_config("web")


def category_to_zone(category: str) -> str:
    """把识别类别名映射到库区 key；未知类别归入 ``default_zone``。"""
    cfg = get_categories_config()
    mapping = cfg.get("category_to_zone", {})
    return mapping.get(category, cfg.get("default_zone", "unsorted"))


def gripper_grasp_angle(category: str | None = None) -> float:
    """返回某类别的夹爪夹紧角度；无覆盖时用 grasp.yaml 的 defaults。"""
    cfg = get_grasp_config()
    default = cfg.get("defaults", {}).get("gripper_grasp", 100)
    if category:
        per = cfg.get("per_object", {}).get(category)
        if per and "gripper_grasp" in per:
            return per["gripper_grasp"]
    return default
