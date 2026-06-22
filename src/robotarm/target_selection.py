"""目标筛选策略。

实现开发计划阶段六的「目标筛选策略」：

- 优先抓取置信度最高的目标。
- 如果多个目标重叠/接近，选离机械臂最近（或最容易抓取）的目标。
- 如果目标在不可达区域，跳过并给出提示。

输入为检测结果列表，每项是一个 ``Detection``，纯函数实现，便于单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass
class Detection:
    """单个检测目标。

    :param name: 类别名（模型输出）
    :param conf: 置信度 0~1
    :param cx: 中心点像素 x
    :param cy: 中心点像素 y
    :param arm_x: 机械臂平面 x（坐标转换后，可选）
    :param arm_y: 机械臂平面 y（坐标转换后，可选）
    """

    name: str
    conf: float
    cx: float
    cy: float
    arm_x: Optional[float] = None
    arm_y: Optional[float] = None


@dataclass
class SelectionResult:
    """筛选结果。"""

    target: Optional[Detection]
    reason: str
    skipped: List[Detection]


# 机械臂基座在图像平面中的参考点（默认取画面底部中点，越靠近越易抓取）。
DEFAULT_ARM_ORIGIN = (320.0, 480.0)


def _distance(d: Detection, origin) -> float:
    return math.hypot(d.cx - origin[0], d.cy - origin[1])


def select_target(
    detections: Sequence[Detection],
    *,
    min_conf: float = 0.5,
    overlap_px: float = 60.0,
    arm_origin=DEFAULT_ARM_ORIGIN,
    reachable: Optional[Callable[[Detection], bool]] = None,
) -> SelectionResult:
    """从检测列表中选出本次要抓取的目标。

    策略顺序：
    1. 过滤掉置信度低于 ``min_conf`` 的目标。
    2. 过滤掉不可达目标（``reachable`` 返回 False），记入 skipped。
    3. 在剩余目标中取置信度最高者；若存在与其位置接近（中心距 < ``overlap_px``）
       的其它高置信目标，则在这一簇里选离机械臂最近的，便于抓取。

    :param reachable: 可选的可达性判断函数，对每个 Detection 返回 bool。
                      通常用 coordinate.is_reachable 包一层传入。
    :returns: SelectionResult，target 为 None 表示无可抓取目标。
    """
    if not detections:
        return SelectionResult(None, "未识别到目标", [])

    skipped: List[Detection] = []
    candidates: List[Detection] = []
    for d in detections:
        if d.conf < min_conf:
            skipped.append(d)
            continue
        if reachable is not None and not reachable(d):
            skipped.append(d)
            continue
        candidates.append(d)

    if not candidates:
        return SelectionResult(None, "目标均不满足置信度或不可达", skipped)

    # 以置信度最高者为锚
    anchor = max(candidates, key=lambda d: d.conf)

    # 找出与锚点位置接近的一簇（含锚点自身）
    cluster = [
        d for d in candidates
        if _distance_between(d, anchor) <= overlap_px
    ]
    if len(cluster) > 1:
        # 重叠/接近：选离机械臂最近者
        target = min(cluster, key=lambda d: _distance(d, arm_origin))
        reason = (
            f"多目标接近，选最近: {target.name} "
            f"(conf={target.conf:.2f})"
        )
    else:
        target = anchor
        reason = f"置信度最高: {target.name} (conf={target.conf:.2f})"

    return SelectionResult(target, reason, skipped)


def _distance_between(a: Detection, b: Detection) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)
