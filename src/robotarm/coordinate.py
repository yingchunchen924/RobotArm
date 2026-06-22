"""像素坐标 -> 机械臂平面坐标 的转换。

公式来自手册 ``get_pos`` / ``get_Sqaure``（见 docs/手册提取文本）：

    a = round((point_x - x_center) / x_div, 5)
    b = round((image_h - point_y) / y_div * y_scale + y_bias, 5)

手册中这些魔数（320 / 4000 / 3000 / 0.8 / 0.19）直接写死在代码里，本模块把它们
全部抽到 ``config/arm.yaml`` 的 ``coordinate`` 段，标定时只改 yaml 即可。

本模块为纯函数，无任何硬件依赖，可在 PC 上单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CoordinateParams:
    """坐标转换参数（从 arm.yaml 的 coordinate 段构造）。"""

    image_width: int = 640
    image_height: int = 480
    x_center: float = 320.0
    x_div: float = 4000.0
    y_div: float = 3000.0
    y_scale: float = 0.8
    y_bias: float = 0.19

    @classmethod
    def from_config(cls, cfg: Dict) -> "CoordinateParams":
        c = (cfg or {}).get("coordinate", {})
        return cls(
            image_width=c.get("image_width", 640),
            image_height=c.get("image_height", 480),
            x_center=c.get("x_center", 320.0),
            x_div=c.get("x_div", 4000.0),
            y_div=c.get("y_div", 3000.0),
            y_scale=c.get("y_scale", 0.8),
            y_bias=c.get("y_bias", 0.19),
        )


@dataclass(frozen=True)
class ReachBounds:
    """机械臂可达工作范围（平面坐标，米）。"""

    x_min: float = -0.20
    x_max: float = 0.20
    y_min: float = 0.10
    y_max: float = 0.35

    @classmethod
    def from_config(cls, cfg: Dict) -> "ReachBounds":
        b = (cfg or {}).get("reach_bounds", {})
        return cls(
            x_min=b.get("x_min", -0.20),
            x_max=b.get("x_max", 0.20),
            y_min=b.get("y_min", 0.10),
            y_max=b.get("y_max", 0.35),
        )


def pixel_to_arm(
    point_x: float,
    point_y: float,
    params: CoordinateParams | None = None,
) -> Tuple[float, float]:
    """把图像中的目标中心像素坐标转换为机械臂平面坐标 (x, y)，单位米。

    与手册公式逐项对应，结果保留 5 位小数。
    """
    p = params or CoordinateParams()
    x = round((point_x - p.x_center) / p.x_div, 5)
    y = round((p.image_height - point_y) / p.y_div * p.y_scale + p.y_bias, 5)
    return x, y


def apply_offset(y: float, offset: float) -> float:
    """对 y 坐标施加硬件误差补偿。

    对应手册 ``request.tar_y = posxy[1] + self.offset``。offset 来自标定后手工
    调整的 offset.txt（见 config/calibration.md）。
    """
    return round(y + offset, 5)


def is_reachable(x: float, y: float, bounds: ReachBounds | None = None) -> bool:
    """判断平面坐标是否在机械臂可达范围内。

    用于开发计划阶段六「如果目标在不可达区域，跳过并提示」。
    """
    b = bounds or ReachBounds()
    return (b.x_min <= x <= b.x_max) and (b.y_min <= y <= b.y_max)
