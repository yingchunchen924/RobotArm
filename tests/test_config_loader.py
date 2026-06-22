"""config_loader 测试。"""

import pytest

from robotarm import config_loader as cl


def test_find_project_root_has_config():
    root = cl.find_project_root()
    import os
    assert os.path.isdir(os.path.join(root, "config"))


def test_load_arm_config_has_coordinate():
    cfg = cl.get_arm_config()
    assert "coordinate" in cfg
    assert cfg["coordinate"]["image_width"] == 640


def test_load_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        cl.load_config("does_not_exist_xyz")


def test_category_to_zone_known():
    assert cl.category_to_zone("resistor") == "electronic"
    assert cl.category_to_zone("wrench") == "tools"
    assert cl.category_to_zone("clamp") == "hardware"


def test_category_to_zone_unknown_falls_back():
    assert cl.category_to_zone("totally_unknown") == "unsorted"


def test_gripper_grasp_angle_default_and_override():
    # 默认值
    assert cl.gripper_grasp_angle() == 100
    # per_object 覆盖（resistor 配的是 130）
    assert cl.gripper_grasp_angle("resistor") == 130
    # 未覆盖的类别回退默认
    assert cl.gripper_grasp_angle("relay") == 100


def test_cache_clear_works():
    cl.get_arm_config()
    cl.clear_cache()  # 不应抛错
