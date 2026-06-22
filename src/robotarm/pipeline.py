"""抓取流程编排（pipeline）。

把已有的纯逻辑模块串成完整闭环，对应开发计划阶段三（分拣/堆叠）与阶段六（分类抓取）的
执行主干。本模块**不直接碰硬件**：摄像头/NPU/串口/运动学都通过 ``interfaces`` 的抽象
注入，因此可在 PC 上用 Mock 端到端跑通并单测。

一次循环（process_once）的数据流：

    frame ──Detector.detect──▶ [Detection]
          ──coordinate.pixel_to_arm + apply_offset──▶ 机械臂平面坐标
          ──coordinate.is_reachable──▶ 可达性
          ──target_selection.select_target──▶ 选中目标
          ──Kinematics.inverse──▶ 关节角
          ──grasp_motion.pick_and_place──▶ 机械臂执行
          ──inventory + states──▶ 库存与状态更新

两种模式：
    SORTING  分拣：按类别映射到库区放置（grasp.yaml place_zones）
    STACKING 堆叠：按当前层号放置（stacking.yaml layers），逐层叠高

异常处理（开发计划阶段六）：
    未识别到目标 / 逆解失败 / 目标不可达 / 抓取动作异常 —— 各自降级为 FAILED 状态并记录。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from . import config_loader as cl
from .coordinate import (
    CoordinateParams,
    ReachBounds,
    apply_offset,
    is_reachable,
    pixel_to_arm,
)
from .grasp_motion import GraspMotion, compose_grasp_joints, fix_ik_overflow
from .interfaces import ArmDriver, Detector, Kinematics
from .inventory import Inventory, OperationRecord
from .states import GraspState, StatusReporter
from .target_selection import Detection, select_target


class Mode(str, Enum):
    SORTING = "sorting"      # 分拣：按类别放库区
    STACKING = "stacking"    # 堆叠：按层放堆叠区


class GraspPipeline:
    """抓取流程编排器。

    :param detector:   目标检测（真机 NPU / Mock）
    :param kinematics: 逆运动学（真机 ROS2 / Mock）
    :param arm:        机械臂执行（真机串口 / Mock）
    :param reporter:   状态广播（默认新建）
    :param inventory:  库存模型（默认新建）
    :param offset:     硬件误差补偿（来自标定 offset.txt）
    :param clock:      返回时间戳字符串的函数，供日志使用（默认空串，便于测试可注入）
    """

    def __init__(
        self,
        detector: Detector,
        kinematics: Kinematics,
        arm: ArmDriver,
        reporter: Optional[StatusReporter] = None,
        inventory: Optional[Inventory] = None,
        offset: float = 0.0,
        clock=None,
    ) -> None:
        self.detector = detector
        self.kinematics = kinematics
        self.arm = arm
        self.reporter = reporter or StatusReporter()
        self.inventory = inventory or Inventory()
        self.offset = offset
        self._clock = clock or (lambda: "")
        self.motion = GraspMotion(arm)

        # 从配置加载（一次）
        arm_cfg = cl.get_arm_config()
        self.coord_params = CoordinateParams.from_config(arm_cfg)
        self.bounds = ReachBounds.from_config(arm_cfg)
        self.grasp_cfg = cl.get_grasp_config()
        self.stacking_cfg = cl.get_stacking_config()

        # 堆叠层计数
        self._stack_level = 0

    # ------------------------------------------------------------------
    # 主入口：处理一帧 / 一次抓取
    # ------------------------------------------------------------------
    def process_once(self, frame, mode: Mode = Mode.SORTING) -> bool:
        """对一帧执行一次「识别 -> 选目标 -> 抓取 -> 放置」。

        :returns: True 表示成功完成一次抓取放置；False 表示本次未执行（无目标/失败）。
        """
        rep = self.reporter

        # 1. 识别 -----------------------------------------------------------
        rep.report(GraspState.WAIT_DETECT)
        try:
            detections = self.detector.detect(frame)
        except Exception as e:  # 摄像头/推理异常
            rep.report(GraspState.FAILED, f"识别异常: {e}")
            return False

        if not detections:
            rep.report(GraspState.FAILED, "未识别到目标")
            return False

        # 2. 坐标转换（给每个检测补上机械臂坐标）---------------------------
        for d in detections:
            ax, ay = pixel_to_arm(d.cx, d.cy, self.coord_params)
            ay = apply_offset(ay, self.offset)
            d.arm_x, d.arm_y = ax, ay

        # 3. 选目标（含可达性过滤）-----------------------------------------
        def reachable(d: Detection) -> bool:
            return (
                d.arm_x is not None
                and is_reachable(d.arm_x, d.arm_y, self.bounds)
            )

        sel = select_target(detections, reachable=reachable)
        if sel.target is None:
            rep.report(GraspState.FAILED, sel.reason)
            return False

        target = sel.target
        rep.report(GraspState.DETECTED, sel.reason, target=target.name)

        # 4. 逆解 -----------------------------------------------------------
        try:
            ik = self.kinematics.inverse(target.arm_x, target.arm_y)
        except Exception as e:
            rep.report(GraspState.FAILED, f"逆解失败: {e}", target=target.name)
            self._log(target, "failed", self._zone_of(target), f"逆解失败: {e}")
            return False

        if not ik:
            rep.report(GraspState.FAILED, "逆解无结果", target=target.name)
            self._log(target, "failed", self._zone_of(target), "逆解无结果")
            return False

        ik = fix_ik_overflow(ik)

        # 5. 执行抓取放置 ---------------------------------------------------
        try:
            if mode == Mode.STACKING:
                ok = self._do_stacking(target, ik)
            else:
                ok = self._do_sorting(target, ik)
        except Exception as e:
            rep.report(GraspState.FAILED, f"抓取动作异常: {e}", target=target.name)
            self._log(target, "failed", self._zone_of(target), f"动作异常: {e}")
            return False

        return ok

    # ------------------------------------------------------------------
    # 分拣模式
    # ------------------------------------------------------------------
    def _do_sorting(self, target: Detection, ik) -> bool:
        zone = self._zone_of(target)
        place = self.grasp_cfg.get("place_zones", {}).get(zone)
        if not place or "place_joints" not in place:
            self.reporter.report(
                GraspState.FAILED, f"库区 {zone} 缺少放置姿态", target=target.name)
            self._log(target, "failed", zone, "缺少放置姿态")
            return False

        gripper = self._gripper_cfg()
        grasp_angle = cl.gripper_grasp_angle(target.name)
        grasp_joints = compose_grasp_joints(ik, gripper_open=gripper["open"])

        self.reporter.report(GraspState.GRASPING, target=target.name)
        self.motion.pick_and_place(
            grasp_joints=grasp_joints,
            place_joints=place["place_joints"],
            ready_pose=self._arm_ready_pose(),
            lift_pose=self._arm_ready_pose(),
            gripper_open=gripper["open"],
            gripper_grasp=grasp_angle,
        )
        self.reporter.report(GraspState.PLACING, f"放入 {zone}", target=target.name)

        # 注意：_log 的 "success" 会自动给 zone 计数（见 inventory.log_operation），
        # 这里不要再额外 add，否则重复计数。
        self._log(target, "success", zone)
        self.reporter.report(GraspState.DONE, f"已放入 {zone}", target=target.name)
        return True

    # ------------------------------------------------------------------
    # 堆叠模式
    # ------------------------------------------------------------------
    def _do_stacking(self, target: Detection, ik) -> bool:
        layers = self.stacking_cfg.get("layers", [])
        max_layers = self.stacking_cfg.get("max_layers", len(layers))
        if self._stack_level >= max_layers:
            self.reporter.report(
                GraspState.FAILED, f"已达最大层数 {max_layers}", target=target.name)
            return False

        layer = layers[self._stack_level]
        place_joints = layer["joints_down"]

        gripper = self.stacking_cfg.get("gripper", {"open": 0, "grasp": 100})
        grasp_joints = compose_grasp_joints(ik, gripper_open=gripper["open"])

        self.reporter.report(
            GraspState.GRASPING, f"第 {layer['level']} 层", target=target.name)
        self.motion.pick_and_place(
            grasp_joints=grasp_joints,
            place_joints=place_joints,
            ready_pose=self.stacking_cfg.get("ready_pose", self._arm_ready_pose()),
            lift_pose=self.stacking_cfg.get("lift_pose", self._arm_ready_pose()),
            gripper_open=gripper["open"],
            gripper_grasp=gripper["grasp"],
        )
        self.reporter.report(
            GraspState.PLACING, f"堆叠第 {layer['level']} 层", target=target.name)

        self._stack_level += 1
        # _log 的 "success" 会自动给 "stacking" 计数，勿重复 add。
        self._log(target, "success", "stacking", f"第 {layer['level']} 层")
        self.reporter.report(
            GraspState.DONE,
            f"已堆叠第 {layer['level']} 层 ({self._stack_level}/{max_layers})",
            target=target.name,
        )
        return True

    def reset_stack(self) -> None:
        """重置堆叠层计数（开始新一轮堆叠时调用）。"""
        self._stack_level = 0

    @property
    def stack_level(self) -> int:
        return self._stack_level

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _zone_of(self, target: Detection) -> str:
        return cl.category_to_zone(target.name)

    def _gripper_cfg(self) -> dict:
        arm_cfg = cl.get_arm_config()
        g = arm_cfg.get("gripper", {})
        return {"open": g.get("open", 0), "grasp": g.get("grasp", 100)}

    def _arm_ready_pose(self):
        return cl.get_arm_config().get("ready_pose", {}).get(
            "joints", [90, 80, 50, 50, 265, 30])

    def _log(self, target: Detection, result: str, zone: str, note: str = "") -> None:
        self.inventory.log_operation(OperationRecord(
            timestamp=self._clock(),
            category=target.name,
            confidence=round(float(target.conf), 4),
            result=result,
            zone=zone,
            note=note,
        ))
