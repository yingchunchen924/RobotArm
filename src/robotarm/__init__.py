"""robotarm —— 电力库房机械臂系统的纯 Python 核心模块。

本包内的模块**不依赖** ROS2 / 昇腾 NPU / 串口硬件，因此可在任意装有 Python 的机器
（包括开发 PC）上运行与单元测试。真机相关能力通过 ``interfaces`` 中的抽象接口注入。
"""

__all__ = [
    "config_loader",
    "coordinate",
    "target_selection",
    "inventory",
    "interfaces",
    "states",
    "grasp_motion",
    "pipeline",
    "dataset",
    "detect_postprocess",
    "detectors",
    "runner",
]
