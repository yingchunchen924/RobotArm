"""coordinate 测试。

逐项验证手册坐标公式：
    a = round((point_x - 320) / 4000, 5)
    b = round((480 - point_y) / 3000 * 0.8 + 0.19, 5)
"""

from robotarm.coordinate import (
    CoordinateParams,
    ReachBounds,
    apply_offset,
    is_reachable,
    pixel_to_arm,
)
from robotarm import config_loader as cl


def test_pixel_to_arm_matches_manual_formula():
    # 取画面正中心 (320, 240)，用默认参数
    x, y = pixel_to_arm(320, 240)
    expected_x = round((320 - 320) / 4000, 5)
    expected_y = round((480 - 240) / 3000 * 0.8 + 0.19, 5)
    assert x == expected_x == 0.0
    assert y == expected_y


def test_pixel_to_arm_offcenter():
    x, y = pixel_to_arm(420, 180)
    assert x == round((420 - 320) / 4000, 5)
    assert y == round((480 - 180) / 3000 * 0.8 + 0.19, 5)


def test_params_from_config():
    params = CoordinateParams.from_config(cl.get_arm_config())
    assert params.x_center == 320
    assert params.x_div == 4000.0
    assert params.y_bias == 0.19


def test_apply_offset():
    assert apply_offset(0.2, 0.01) == 0.21
    assert apply_offset(0.2, -0.005) == 0.195


def test_is_reachable_within_and_outside():
    bounds = ReachBounds()  # 默认 x[-0.2,0.2] y[0.1,0.35]
    assert is_reachable(0.0, 0.2, bounds) is True
    assert is_reachable(0.5, 0.2, bounds) is False   # x 越界
    assert is_reachable(0.0, 0.05, bounds) is False  # y 太近
    assert is_reachable(0.0, 0.4, bounds) is False   # y 太远


def test_reach_bounds_from_config():
    bounds = ReachBounds.from_config(cl.get_arm_config())
    assert bounds.x_min == -0.20
    assert bounds.y_max == 0.35
