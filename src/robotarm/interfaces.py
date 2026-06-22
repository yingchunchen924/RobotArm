"""硬件相关抽象接口 + Mock 实现。

真机能力（机械臂串口控制、ROS2 运动学服务、NPU 推理）在开发 PC 上无法运行，
因此这里用抽象基类定义统一接口，并提供 Mock 实现，使整套抓取流程逻辑可以在 PC 上
跑通与测试。真机上分别替换为：

- ArmDriver   -> 串口实现（手册 Arm_serial_servo_write* 系列）
- Kinematics  -> ROS2 Kinemarics 服务客户端（手册 server_joint）
- Detector    -> ais_bench/aclruntime 加载 .om 模型推理（手册 infer_image）

替换时只需实现同名方法，上层抓取流程代码（target_selection + coordinate + 本接口）
无需改动。
"""

from __future__ import annotations

import abc
from typing import List, Sequence

from .target_selection import Detection


class ArmDriver(abc.ABC):
    """机械臂执行接口。关节角单位与手册一致（舵机角度）。"""

    @abc.abstractmethod
    def move_joints(self, joints: Sequence[float], duration_ms: int = 1000) -> None:
        """整臂移动到 6 个关节角，duration_ms 为运动时间。

        对应手册 ``Arm_serial_servo_write6_array``。
        """

    @abc.abstractmethod
    def set_gripper(self, angle: float, duration_ms: int = 500) -> None:
        """设置夹爪角度（舵机6）。对应手册 ``Arm_serial_servo_write(6, angle, t)``。"""

    @abc.abstractmethod
    def home(self) -> None:
        """回到初始姿态。"""


class Kinematics(abc.ABC):
    """逆运动学接口。"""

    @abc.abstractmethod
    def inverse(self, x: float, y: float, z: float = 0.0) -> List[float]:
        """逆解：给定机械臂基坐标系目标点，返回 5~6 个关节角。

        对应手册 ROS2 Kinemarics 服务 ``ik`` 模式。
        """


class Detector(abc.ABC):
    """目标检测接口。"""

    @abc.abstractmethod
    def detect(self, frame) -> List[Detection]:
        """对一帧图像推理，返回检测结果列表。

        对应手册 ``infer_image`` / yolov5 om 推理。
        """


# ---------------------------------------------------------------------------
# Mock 实现：用于 PC 上跑通流程，不接触任何硬件
# ---------------------------------------------------------------------------

class MockArmDriver(ArmDriver):
    """打印动作日志的假机械臂，记录调用历史便于测试。"""

    def __init__(self) -> None:
        self.history: List[str] = []

    def move_joints(self, joints: Sequence[float], duration_ms: int = 1000) -> None:
        msg = f"[MockArm] move_joints {list(joints)} ({duration_ms}ms)"
        self.history.append(msg)
        print(msg)

    def set_gripper(self, angle: float, duration_ms: int = 500) -> None:
        msg = f"[MockArm] set_gripper {angle} ({duration_ms}ms)"
        self.history.append(msg)
        print(msg)

    def home(self) -> None:
        msg = "[MockArm] home"
        self.history.append(msg)
        print(msg)


class MockKinematics(Kinematics):
    """返回固定/可预测关节角的假逆运动学。"""

    def inverse(self, x: float, y: float, z: float = 0.0) -> List[float]:
        # 简单可预测映射，仅用于流程联调（非真实运动学）
        j1 = 90.0 + x * 100.0
        j2 = 80.0
        j3 = 50.0
        j4 = 50.0
        j5 = 265.0
        return [j1, j2, j3, j4, j5]


class MockDetector(Detector):
    """返回预设检测结果的假检测器。"""

    def __init__(self, fake: Sequence[Detection] | None = None) -> None:
        # 注意：用 `is None` 区分「未提供」与「显式传入空列表」，
        # 后者表示「这一帧没识别到目标」，不能回退到默认值。
        if fake is None:
            self._fake = [
                Detection(name="resistor", conf=0.91, cx=300, cy=240),
                Detection(name="wrench", conf=0.85, cx=420, cy=180),
            ]
        else:
            self._fake = list(fake)

    def detect(self, frame) -> List[Detection]:
        return list(self._fake)
