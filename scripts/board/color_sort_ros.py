#!/usr/bin/env python3
"""Color sorting debug script using the official calibration + ROS2 IK path.

This is intended for the Atlas 200I DK A2 board. It keeps color detection
simple, but uses the same calibrated path as the official samples:

    raw frame -> perspective transform(dp.bin) -> pixel(x,y)
    -> arm plane coordinate + offset.txt -> ROS2 trial_service IK
    -> Arm_Lib motion

The script is safe by default. Commands that move the arm only print the
planned motion unless `--execute` is passed.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_VERSION = "stack-debug-2026-06-25-r26"

COLORS = {
    "red": ((0, 70, 60), (12, 255, 255), (165, 70, 60), (180, 255, 255)),
    "yellow": ((22, 100, 100), (35, 255, 255)),
    "green": ((45, 80, 80), (75, 255, 255)),
    "blue": ((105, 100, 80), (125, 255, 255)),
}

DEFAULT_CONFIGS = [
    "/home/HwHiAiUser/RobotArm/config",
    "/home/HwHiAiUser/E2ESamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/install/robot_arm_color_stacking/share/robot_arm_color_stacking/config",
    "/home/HwHiAiUser/E2ESamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/install/dofbot_garbage_yolov5/share/dofbot_garbage_yolov5/config",
]

PLACE_JOINTS = {
    "red": [45, 50, 20, 60, 265, 130],
    "yellow": [133, 50, 20, 60, 265, 130],
    "green": [147, 75, 0, 50, 265, 130],
    "blue": [27, 75, 0, 50, 265, 130],
}

DEFAULT_STACKING = {
    "ready_pose": [90, 80, 50, 50, 265, 100],
    "lift_pose": [135, 80, 50, 50, 265, 30],
    "gripper": {"open": 0, "grasp": 175, "release": 30},
    "pick_order": ["red", "yellow", "blue", "green"],
    "detect_poses": {},
    "post_center_detect_poses": {},
    "place_target_xy": [0.01614, 0.24412],
    "place_nudge": [0.0, 0.0],
    "detect_stack_center": True,
    "stack_center_min_layer": 2,
    "stack_center_roi": [220, 170, 520, 440],
    "stack_center_detect_point": "centroid",
    "stack_center_min_area": 1500,
    "stack_center_nudge": [-0.012, -0.063],
    "stack_center_nudges": {},
    "stack_center_stable_frames": 1,
    "stack_center_max_shift": 20,
    "stack_center_colors": {},
    "use_fixed_pick_fallback": True,
    "force_fixed_pick_colors": [],
    "require_detect_colors": [],
    "fixed_pick_xy": {},
    "fixed_pick_z": {},
    "fixed_pick_base": {},
    "pick_points": {
        "red": [567, 210],
        "yellow": [297, 242],
        "blue": [236, 469],
        "green": [599, 390],
    },
    "color_rois": {
        "yellow": [100, 40, 420, 285],
        "red": [330, 40, 640, 310],
        "blue": [60, 210, 390, 480],
        "green": [390, 210, 640, 480],
    },
    "layers": [
        {"level": 1, "joints_down": [90, 50, 20, 60, 265, 100]},
        {"level": 2, "joints_down": [90, 55, 38, 38, 265, 100]},
        {"level": 3, "joints_down": [90, 60, 45, 30, 265, 100]},
        {"level": 4, "joints_down": [90, 65, 55, 20, 265, 100]},
    ],
    "max_layers": 4,
}


@dataclass
class Target:
    name: str
    area: float
    cx: int
    cy: int
    centroid_cx: int | None = None
    centroid_cy: int | None = None


@dataclass
class DetectionCandidate:
    name: str
    area: float
    cx: int
    cy: int
    centroid_cx: int
    centroid_cy: int
    allowed: bool


@dataclass
class Calibration:
    config_dir: Path
    dp: np.ndarray
    y_offset: float
    x_offset: float
    xy: list[int]
    model: str = "legacy"
    affine: list[list[float]] | None = None


def find_config_dir(explicit: str | None) -> Path:
    candidates = [explicit] if explicit else DEFAULT_CONFIGS
    for item in candidates:
        if not item:
            continue
        path = Path(item)
        if (
            (path / "dp.bin").is_file()
            and (path / "offset.txt").is_file()
            and (path / "XYT_config.txt").is_file()
        ):
            return path
    raise RuntimeError("no config dir with dp.bin, offset.txt and XYT_config.txt")


def load_calibration(config_dir: Path) -> Calibration:
    dp = np.fromfile(str(config_dir / "dp.bin"), dtype=np.int32).reshape(4, 2)
    with open(config_dir / "offset.txt", "r", encoding="utf-8") as f:
        y_offset = float(f.readline().strip())
        x_offset = float(f.readline().strip())
    vals = {}
    with open(config_dir / "XYT_config.txt", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                vals[key] = value
    xy = [int(vals.get("x", 90)), int(vals.get("y", 130))]

    # 本机重新标定的仿射映射（可选）。affine.txt 每行一行 6 个数，共 2 行：
    #   a11 a12 a13
    #   a21 a22 a23
    # 由 scripts/board/calibrate_pixel_map.py 拟合得到。存在则覆盖 legacy 公式。
    model = "legacy"
    affine = None
    affine_path = config_dir / "affine.txt"
    if affine_path.is_file():
        rows = []
        with open(affine_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [float(v) for v in line.replace(",", " ").split()]
                if len(parts) >= 3:
                    rows.append(parts[:3])
        if len(rows) == 2:
            model = "affine"
            affine = rows
    return Calibration(config_dir, dp, y_offset, x_offset, xy, model, affine)


def perspective_transform(dp: np.ndarray, image):
    upper_left = lower_left = lower_right = upper_right = None
    for point in dp:
        x, y = point
        if x < 320 and y < 240:
            upper_left = point
        elif x < 320 and y > 240:
            lower_left = point
        elif x > 320 and y > 240:
            lower_right = point
        elif x > 320 and y < 240:
            upper_right = point
    if any(p is None for p in (upper_left, lower_left, lower_right, upper_right)):
        raise RuntimeError(f"invalid dp points: {dp.tolist()}")
    pts1 = np.float32([upper_left, lower_left, lower_right, upper_right])
    pts2 = np.float32([[0, 0], [0, 480], [640, 480], [640, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (640, 480))


def pixel_to_xy(cx: int, cy: int, calib: Calibration, x_nudge: float = 0.0, y_nudge: float = 0.0):
    """像素中心 -> 机械臂 IK 输入坐标 (x, y)。

    若标定目录里有本机拟合的 affine.txt，用仿射映射 tar = A·[cx,cy,1]ᵀ；
    否则回退到华为手册 legacy 公式 (cx-320)/4000 等。

    x_offset/y_offset(offset.txt) 与 x_nudge/y_nudge 始终作为全局误差补偿叠加。
    """
    if calib.model == "affine" and calib.affine is not None:
        a = calib.affine
        x = a[0][0] * cx + a[0][1] * cy + a[0][2] + calib.x_offset
        y = a[1][0] * cx + a[1][1] * cy + a[1][2] + calib.y_offset
    else:
        x = round((cx - 320) / 4000, 5) + calib.x_offset
        y = round(((480 - cy) / 3000) * 0.8 + 0.19, 5) + calib.y_offset
    x += x_nudge
    y += y_nudge
    return round(x, 5), round(y, 5)


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index} with CAP_V4L2")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def parse_regions(region_specs) -> list[tuple[int, int, int, int]]:
    regions = []
    for spec in region_specs or []:
        parts = [int(round(float(v))) for v in spec.replace(",", " ").split()]
        if len(parts) != 4:
            raise ValueError(f"region must be x1,y1,x2,y2: {spec}")
        x1, y1, x2, y2 = parts
        regions.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return regions


def point_in_regions(cx: int, cy: int, regions) -> bool:
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in regions)


def point_allowed(cx: int, cy: int, include_regions, exclude_regions) -> bool:
    includes = parse_regions(include_regions)
    excludes = parse_regions(exclude_regions)
    if includes and not point_in_regions(cx, cy, includes):
        return False
    if point_in_regions(cx, cy, excludes):
        return False
    return True


def contour_ground_point(frame, contour):
    """Return a cube footprint point instead of the projected top centroid.

    A top-face centroid is visually farther than the physical contact point when
    the camera sees the cube at an angle. The footprint point makes IK less
    likely to flatten the gripper and miss the cube.
    """
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    centroid_cx = int(moments["m10"] / moments["m00"])
    centroid_cy = int(moments["m01"] / moments["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    x0, x1 = max(0, centroid_cx - 12), min(w_img, centroid_cx + 12)
    if x0 >= x1:
        return centroid_cx, centroid_cy, centroid_cx, centroid_cy

    row_min = gray[:, x0:x1].min(axis=1)
    limit = min(h_img - 1, y + int(h * 2.2))
    bottom = min(h_img - 1, y + h)
    for yy in range(y + h, limit):
        if row_min[yy] < 110:
            bottom = yy
        elif yy - bottom > 12:
            break
    ground_cy = max(0, bottom - 8)
    return centroid_cx, ground_cy, centroid_cx, centroid_cy


def color_mask(hsv, name: str):
    ranges = COLORS[name]
    mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
    if name == "red":
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3])))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def detect_candidates(
    frame,
    color: str,
    min_area: int,
    detect_point: str = "ground",
    include_regions=None,
    exclude_regions=None,
) -> list[DetectionCandidate]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    names = [color] if color != "any" else list(COLORS.keys())
    candidates: list[DetectionCandidate] = []
    for name in names:
        mask = color_mask(hsv, name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid_cx = int(moments["m10"] / moments["m00"])
            centroid_cy = int(moments["m01"] / moments["m00"])
            if detect_point == "ground":
                point = contour_ground_point(frame, contour)
                if point is None:
                    continue
                cx, cy, centroid_cx, centroid_cy = point
            else:
                cx, cy = centroid_cx, centroid_cy
            # ROI is used to select the fixed cube area in the camera image.
            # Keep the actual pickup point as `ground`, but filter by the
            # visible color centroid so bottom/ground points do not fall
            # outside a top-face ROI.
            allowed = point_allowed(centroid_cx, centroid_cy, include_regions, exclude_regions)
            candidates.append(
                DetectionCandidate(name, area, cx, cy, centroid_cx, centroid_cy, allowed)
            )
    candidates.sort(key=lambda item: item.area, reverse=True)
    return candidates


def detect_best(
    frame,
    color: str,
    min_area: int,
    detect_point: str = "ground",
    include_regions=None,
    exclude_regions=None,
) -> Target | None:
    for item in detect_candidates(
        frame, color, min_area, detect_point, include_regions, exclude_regions
    ):
        if item.allowed:
            return Target(item.name, item.area, item.cx, item.cy, item.centroid_cx, item.centroid_cy)
    return None


def format_candidates(candidates: list[DetectionCandidate], limit: int = 8) -> str:
    if not candidates:
        return "none"
    parts = []
    for item in candidates[:limit]:
        roi = "yes" if item.allowed else "no"
        parts.append(
            f"{item.name}:area={item.area:.0f},point=({item.cx},{item.cy}),"
            f"centroid=({item.centroid_cx},{item.centroid_cy}),roi={roi}"
        )
    return "; ".join(parts)


def print_detection_debug(frame, args):
    print(
        "DETECT_DEBUG "
        f"color={args.color} min_area={args.min_area} detect_point={args.detect_point} "
        f"include_region={list(args.include_region)} exclude_region={list(args.exclude_region)}",
        flush=True,
    )
    candidates = detect_candidates(
        frame,
        args.color,
        args.min_area,
        args.detect_point,
        args.include_region,
        args.exclude_region,
    )
    print(f"CANDIDATES {format_candidates(candidates)}", flush=True)
    if not candidates:
        loose_min_area = max(100, int(args.min_area / 4))
        loose = detect_candidates(
            frame,
            args.color,
            loose_min_area,
            args.detect_point,
            [],
            [],
        )
        print(
            f"LOOSE_CANDIDATES no_roi_min_area={loose_min_area} {format_candidates(loose)}",
            flush=True,
        )


def capture_target(args, calib: Calibration):
    cap = open_camera(args.camera, args.width, args.height)
    try:
        samples: list[Target] = []
        last_frame = None
        for _ in range(args.warmup):
            ok, raw = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.resize(raw, (args.width, args.height))
            if args.perspective:
                frame = perspective_transform(calib.dp, frame)
            last_frame = frame
            target = detect_best(
                frame,
                args.color,
                args.min_area,
                args.detect_point,
                args.include_region,
                args.exclude_region,
            )
            if target:
                if args.stable_frames <= 1:
                    return target, frame
                if samples and samples[-1].name != target.name:
                    samples = []
                samples.append(target)
                samples = samples[-args.stable_frames :]
                if len(samples) >= args.stable_frames:
                    xs = [sample.cx for sample in samples]
                    ys = [sample.cy for sample in samples]
                    if (
                        max(xs) - min(xs) <= args.max_shift
                        and max(ys) - min(ys) <= args.max_shift
                    ):
                        cxs = [sample.centroid_cx for sample in samples if sample.centroid_cx is not None]
                        cys = [sample.centroid_cy for sample in samples if sample.centroid_cy is not None]
                        avg = Target(
                            name=samples[-1].name,
                            area=sum(sample.area for sample in samples) / len(samples),
                            cx=round(sum(xs) / len(xs)),
                            cy=round(sum(ys) / len(ys)),
                            centroid_cx=round(sum(cxs) / len(cxs)) if cxs else None,
                            centroid_cy=round(sum(cys) / len(cys)) if cys else None,
                        )
                        return avg, frame
            else:
                samples = []
            time.sleep(0.05)
    finally:
        cap.release()
    if last_frame is not None:
        print_detection_debug(last_frame, args)
    raise RuntimeError(
        f"no target detected color={args.color} include_region={list(args.include_region)}"
    )


def fixed_pick_target(stack_cfg: dict, color: str, reason: str):
    point = (stack_cfg.get("pick_points") or {}).get(color)
    if not point or len(point) != 2:
        raise RuntimeError(f"no fixed pick point configured for color={color}")
    cx, cy = int(round(float(point[0]))), int(round(float(point[1])))
    print(f"FIXED_PICK_POINT color={color} cx={cx} cy={cy} reason={reason}", flush=True)
    target = Target(color, 0.0, cx, cy, cx, cy)
    return target, None


def capture_stack_target(args, calib: Calibration, stack_cfg: dict, color: str):
    fixed_colors = set(stack_cfg.get("force_fixed_pick_colors") or [])
    if color in fixed_colors:
        return fixed_pick_target(stack_cfg, color, "force-fixed-layout")
    try:
        return capture_target(args, calib)
    except RuntimeError:
        if color in set(stack_cfg.get("require_detect_colors") or []):
            raise
        if not stack_cfg.get("use_fixed_pick_fallback", False):
            raise
        return fixed_pick_target(stack_cfg, color, "color-detect-failed")


def stack_center_color_for_layer(stack_cfg: dict, pick_order: list[str], idx: int, layer: dict):
    configured = stack_cfg.get("stack_center_colors") or {}
    level = layer.get("level", idx + 1)
    for key in (level, str(level), idx + 1, str(idx + 1)):
        if key in configured:
            return configured[key]
    if idx > 0 and pick_order:
        return pick_order[(idx - 1) % len(pick_order)]
    return None


def stack_center_nudge_for_layer(stack_cfg: dict, layer: dict, idx: int):
    configured = stack_cfg.get("stack_center_nudges") or {}
    level = layer.get("level", idx + 1)
    for key in (level, str(level), idx + 1, str(idx + 1)):
        if key in configured:
            return configured[key] or [0.0, 0.0]
    return stack_cfg.get("stack_center_nudge", [0.0, 0.0]) or [0.0, 0.0]


def color_float_override(stack_cfg: dict, key: str, color: str, default=None):
    configured = stack_cfg.get(key) or {}
    if color in configured and configured[color] is not None:
        return float(configured[color])
    return default


def capture_stack_center(args, calib: Calibration, stack_cfg: dict, color: str, layer: dict, idx: int):
    roi = stack_cfg.get("stack_center_roi")
    include_region = []
    if roi and len(roi) == 4:
        include_region = [",".join(str(v) for v in roi)]

    old_color = args.color
    old_include = args.include_region
    old_detect_point = args.detect_point
    old_min_area = args.min_area
    old_stable_frames = args.stable_frames
    old_max_shift = args.max_shift
    try:
        args.color = color
        args.include_region = include_region
        args.detect_point = stack_cfg.get("stack_center_detect_point", "ground")
        args.min_area = int(stack_cfg.get("stack_center_min_area", old_min_area))
        args.stable_frames = int(stack_cfg.get("stack_center_stable_frames", old_stable_frames))
        args.max_shift = int(stack_cfg.get("stack_center_max_shift", old_max_shift))
        target, frame = capture_target(args, calib)
    finally:
        args.color = old_color
        args.include_region = old_include
        args.detect_point = old_detect_point
        args.min_area = old_min_area
        args.stable_frames = old_stable_frames
        args.max_shift = old_max_shift

    xy = pixel_to_xy(target.cx, target.cy, calib, args.x_nudge, args.y_nudge)
    nudge = stack_center_nudge_for_layer(stack_cfg, layer, idx)
    place_xy = (
        round(xy[0] + float(nudge[0]), 5),
        round(xy[1] + float(nudge[1]), 5),
    )
    print(
        f"STACK_CENTER color={color} area={target.area:.0f} "
        f"point=({target.cx},{target.cy}) centroid=({target.centroid_cx},{target.centroid_cy}) "
        f"x={xy[0]:.5f} y={xy[1]:.5f} "
        f"nudge=({float(nudge[0]):.5f},{float(nudge[1]):.5f}) "
        f"place_x={place_xy[0]:.5f} place_y={place_xy[1]:.5f} roi={include_region}",
        flush=True,
    )
    return target, place_xy


class RosIkClient:
    def __init__(self, timeout_sec: float = 5.0):
        import rclpy
        from dofbot_info.srv import Kinemarics

        self.rclpy = rclpy
        self.Kinemarics = Kinemarics
        rclpy.init(args=None)
        self.node = rclpy.create_node("color_sort_ros_debug")
        self.client = self.node.create_client(Kinemarics, "trial_service")
        if not self.client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("ROS2 service trial_service is not available")

    def inverse(self, x: float, y: float, z: float = 0.0):
        request = self.Kinemarics.Request()
        request.tar_x = float(x)
        request.tar_y = float(y)
        request.tar_z = float(z)
        request.kin_name = "ik"
        future = self.client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if response is None:
            raise RuntimeError("IK returned no response")
        joints = [response.joint1, response.joint2, response.joint3, response.joint4, response.joint5]
        # 关节角补偿，与华为原版 garbage_identify.server_joint 一致：
        #   joints[1] += joints[2] / 2; joints[3] += joints[2] * 3 / 4
        if joints[2] < 0:
            joints[1] += joints[2] / 2
            joints[3] += joints[2] * 3 / 4
            joints[2] = 0
        return [round(v, 2) for v in joints]

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def make_arm():
    import Arm_Lib

    return Arm_Lib.Arm_Device()


def move_joints(arm, joints, ms: int, execute: bool):
    prefix = "MOVE" if execute else "PLAN MOVE"
    print(f"{prefix} {joints} {ms}ms", flush=True)
    if execute:
        arm.Arm_serial_servo_write6_array(list(joints), ms)
        time.sleep(ms / 1000.0 + 0.2)


def set_gripper(arm, angle: float, ms: int, execute: bool):
    prefix = "GRIP" if execute else "PLAN GRIP"
    print(f"{prefix} {angle} {ms}ms", flush=True)
    if execute:
        arm.Arm_serial_servo_write(6, angle, ms)
        time.sleep(ms / 1000.0 + 0.2)


def print_plan(target: Target, xy, joints, calib: Calibration):
    print(f"CONFIG {calib.config_dir}", flush=True)
    print(f"DP {calib.dp.tolist()}", flush=True)
    print(f"OFFSET x={calib.x_offset} y={calib.y_offset} XYT={calib.xy}", flush=True)
    print(f"MODEL {calib.model}", flush=True)
    print(
        f"TARGET {target.name} area={target.area:.0f} cx={target.cx} cy={target.cy}"
        f" centroid=({target.centroid_cx},{target.centroid_cy})",
        flush=True,
    )
    print(f"ARM_XY x={xy[0]:.5f} y={xy[1]:.5f}", flush=True)
    print(f"IK {joints}", flush=True)


def elevated_xy_joints(ik: RosIkClient, xy, z: float):
    """IK at a higher z for safer approach above the target."""
    if z <= 0:
        return None
    return ik.inverse(xy[0], xy[1], z)


def check_reachable(xy, args) -> str | None:
    """检查坐标是否在舒适抓取区。超出则返回警告字符串，否则 None。

    实测本机：y>0.30 时 IK 解出 joint2 接近 0/负，机械臂前倾、夹爪近水平，
    抓取易夹空。把方块放在 y<=0.30、|x|<=0.13 的甜区抓取姿态才正常。
    """
    x, y = xy
    msgs = []
    if not (-args.reach_xmax <= x <= args.reach_xmax):
        msgs.append(f"x={x:.3f} 超出 [±{args.reach_xmax}]")
    if not (args.reach_ymin <= y <= args.reach_ymax):
        msgs.append(f"y={y:.3f} 超出 [{args.reach_ymin},{args.reach_ymax}]")
    if msgs:
        return "目标可能不可达/姿态扭曲：" + "；".join(msgs) + "（把方块放近一点）"
    return None


def load_stacking_config(explicit: str | None, calib: Calibration) -> dict:
    cfg = copy.deepcopy(DEFAULT_STACKING)
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        calib.config_dir / "stacking.yaml",
        Path("/home/HwHiAiUser/RobotArm/config/stacking.yaml"),
    ])

    loaded = None
    for path in candidates:
        if not path or not path.is_file():
            continue
        try:
            import yaml
        except ImportError:
            print("WARN PyYAML not installed; using built-in stacking defaults", flush=True)
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        print(f"STACKING_CONFIG {path}", flush=True)
        break

    if loaded:
        for key in (
            "ready_pose",
            "lift_pose",
            "layers",
            "max_layers",
            "detect_poses",
            "post_center_detect_poses",
            "place_target_xy",
            "place_nudge",
            "detect_stack_center",
            "stack_center_min_layer",
            "stack_center_roi",
            "stack_center_detect_point",
            "stack_center_min_area",
            "stack_center_nudge",
            "stack_center_nudges",
            "stack_center_stable_frames",
            "stack_center_max_shift",
            "stack_center_colors",
            "use_fixed_pick_fallback",
            "force_fixed_pick_colors",
            "require_detect_colors",
            "fixed_pick_xy",
            "fixed_pick_z",
            "fixed_pick_base",
            "pick_points",
        ):
            if key in loaded:
                cfg[key] = loaded[key]
        for key in ("pick_order", "color_rois"):
            if key in loaded:
                cfg[key] = loaded[key]
        if "gripper" in loaded:
            cfg["gripper"].update(loaded["gripper"] or {})
    return cfg


def with_gripper(joints, angle: float):
    out = list(joints)
    while len(out) < 6:
        out.append(0)
    out[5] = angle
    return out


def build_motion_plan(args, calib: Calibration, target: Target, xy):
    ik = RosIkClient(args.service_timeout)
    try:
        joints = ik.inverse(*xy, args.tar_z)
        above_joints = elevated_xy_joints(ik, xy, args.approach_z)
    finally:
        ik.close()

    grasp = [joints[0], joints[1], joints[2], joints[3], args.wrist, args.open]
    grasp_hold = [joints[0], joints[1], joints[2], joints[3], args.wrist, args.close]
    if above_joints is None:
        approach = [calib.xy[0], 80, 50, 50, args.wrist, args.open]
    else:
        approach = [
            above_joints[0],
            above_joints[1],
            above_joints[2],
            above_joints[3],
            args.wrist,
            args.open,
        ]
    approach_hold = list(approach)
    approach_hold[5] = args.close
    ready = [calib.xy[0], 80, 50, 50, args.wrist, args.close]
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, args.release]
    place = PLACE_JOINTS.get(target.name, PLACE_JOINTS["blue"])
    place = [place[0], place[1], place[2], place[3], place[4], args.close]
    lift = [place[0], 80, 50, 50, args.wrist, args.release]
    return {
        "joints": joints,
        "home": home,
        "ready": ready,
        "approach": approach,
        "approach_hold": approach_hold,
        "grasp": grasp,
        "grasp_hold": grasp_hold,
        "place": place,
        "lift": lift,
    }


def build_stack_plan(
    args,
    calib: Calibration,
    target: Target,
    xy,
    layer: dict,
    stack_cfg: dict,
    place_xy_override=None,
):
    gripper = stack_cfg.get("gripper", {})
    open_angle = args.open if args.open is not None else gripper.get("open", 0)
    grasp_angle = args.close if args.close is not None else gripper.get("grasp", 100)
    release_angle = args.release if args.release is not None else gripper.get("release", 30)
    wrist = args.wrist
    pick_z = color_float_override(stack_cfg, "fixed_pick_z", target.name, args.tar_z)
    pick_base = color_float_override(stack_cfg, "fixed_pick_base", target.name, None)
    place_approach_z = float(getattr(args, "place_approach_z", 0.0) or 0.0)

    ik = RosIkClient(args.service_timeout)
    try:
        joints = ik.inverse(*xy, pick_z)
        above_joints = elevated_xy_joints(ik, xy, args.approach_z)
        place_approach_joints = None
        place_target_xy = place_xy_override or stack_cfg.get("place_target_xy")
        if place_target_xy and len(place_target_xy) == 2:
            place_nudge = [0.0, 0.0] if place_xy_override else (stack_cfg.get("place_nudge", [0.0, 0.0]) or [0.0, 0.0])
            place_xy = (
                float(place_target_xy[0]) + float(place_nudge[0]),
                float(place_target_xy[1]) + float(place_nudge[1]),
            )
            if place_approach_z > 0:
                place_approach_joints = ik.inverse(place_xy[0], place_xy[1], place_approach_z)
            place_joints = ik.inverse(place_xy[0], place_xy[1], args.tar_z)
            base_layer = (stack_cfg.get("layers") or [layer])[0]
            # The camera defines the stack XY center. Keep the configured
            # per-layer joint deltas for height so layer 2+ does not descend
            # to the first-layer release height.
            if base_layer is not layer:
                for axis in (1, 2, 3):
                    place_joints[axis] = round(
                        place_joints[axis]
                        + float(layer["joints_down"][axis])
                        - float(base_layer["joints_down"][axis]),
                        2,
                    )
        else:
            place_xy = None
            place_joints = list(layer["joints_down"][:5])
    finally:
        ik.close()

    ready_open = with_gripper(stack_cfg.get("ready_pose", DEFAULT_STACKING["ready_pose"]), open_angle)
    ready_hold = with_gripper(stack_cfg.get("ready_pose", DEFAULT_STACKING["ready_pose"]), grasp_angle)
    if above_joints is None:
        approach = with_gripper([ready_open[0], 80, 50, 50, wrist, open_angle], open_angle)
    else:
        approach = [above_joints[0], above_joints[1], above_joints[2], above_joints[3], wrist, open_angle]
    if pick_base is not None:
        joints[0] = round(pick_base, 2)
        approach[0] = round(pick_base, 2)
    grasp = [joints[0], joints[1], joints[2], joints[3], wrist, open_angle]
    approach_hold = with_gripper(approach, grasp_angle)
    place_base = [place_joints[0], place_joints[1], place_joints[2], place_joints[3], wrist, grasp_angle]
    place_hold = with_gripper(place_base, grasp_angle)
    place_approach = None
    if place_approach_joints is not None:
        place_approach = [
            place_approach_joints[0],
            place_approach_joints[1],
            place_approach_joints[2],
            place_approach_joints[3],
            wrist,
            grasp_angle,
        ]
    lift = with_gripper(stack_cfg.get("lift_pose", DEFAULT_STACKING["lift_pose"]), release_angle)
    turn = list(ready_hold)
    turn[0] = place_hold[0]
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, release_angle]

    return {
        "joints": joints,
        "home": home,
        "ready_open": ready_open,
        "ready_hold": ready_hold,
        "approach": approach,
        "approach_hold": approach_hold,
        "grasp": grasp,
        "turn": turn,
        "place_approach": place_approach,
        "place": place_hold,
        "place_xy": place_xy,
        "place_approach_z": place_approach_z,
        "pick_z": pick_z,
        "pick_base": pick_base,
        "lift": lift,
        "open": open_angle,
        "grasp_angle": grasp_angle,
        "release": release_angle,
    }


def execute_step_plan(args, arm, plan, step: str):
    if step in ("ready", "approach", "grasp", "pick"):
        move_joints(arm, plan["ready"], 1000, args.execute)
        set_gripper(arm, args.open, 500, args.execute)
    if step in ("approach", "grasp", "pick"):
        move_joints(arm, plan["approach"], args.approach_ms, args.execute)
    if step in ("grasp", "pick"):
        move_joints(arm, plan["grasp"], args.grasp_ms, args.execute)
    if step == "pick":
        # 夹紧并多停一会，让夹爪把方块咬实，再慢速抬起搬运，避免惯性晃脱。
        set_gripper(arm, args.close, args.grip_ms, args.execute)
        if args.execute:
            time.sleep(args.grip_settle)
        move_joints(arm, plan["approach_hold"], args.move_ms, args.execute)
        move_joints(arm, plan["ready"], args.move_ms, args.execute)
        move_joints(arm, plan["place"], args.move_ms, args.execute)
        set_gripper(arm, args.release, 500, args.execute)
        move_joints(arm, plan["lift"], args.move_ms, args.execute)
        move_joints(arm, plan["home"], args.move_ms, args.execute)


def execute_stack_plan(args, arm, plan):
    move_joints(arm, plan["ready_open"], args.move_ms, args.execute)
    set_gripper(arm, plan["open"], args.release_ms, args.execute)
    move_joints(arm, plan["approach"], args.approach_ms, args.execute)
    move_joints(arm, plan["grasp"], args.grasp_ms, args.execute)
    set_gripper(arm, plan["grasp_angle"], args.grip_ms, args.execute)
    if args.execute:
        time.sleep(args.grip_settle)
    move_joints(arm, plan["approach_hold"], args.move_ms, args.execute)
    move_joints(arm, plan["ready_hold"], args.move_ms, args.execute)
    move_joints(arm, plan["turn"], args.move_ms, args.execute)
    if plan.get("place_approach") is not None:
        move_joints(arm, plan["place_approach"], args.move_ms, args.execute)
    move_joints(arm, plan["place"], args.place_ms, args.execute)
    if args.execute and args.place_settle > 0:
        time.sleep(args.place_settle)
    set_gripper(arm, plan["release"], args.release_ms, args.execute)
    if args.execute and args.release_settle > 0:
        time.sleep(args.release_settle)
    move_joints(arm, plan["lift"], args.move_ms, args.execute)
    if args.home_each:
        move_joints(arm, plan["home"], args.move_ms, args.execute)


def build_held_place_plan(args, calib: Calibration, layer: dict, stack_cfg: dict, place_xy):
    gripper = stack_cfg.get("gripper", {})
    grasp_angle = args.close if args.close is not None else gripper.get("grasp", 100)
    release_angle = args.release if args.release is not None else gripper.get("release", 30)
    wrist = args.wrist

    ik = RosIkClient(args.service_timeout)
    try:
        place_joints = ik.inverse(place_xy[0], place_xy[1], args.tar_z)
    finally:
        ik.close()

    base_layer = (stack_cfg.get("layers") or [layer])[0]
    if base_layer is not layer:
        for axis in (1, 2, 3):
            place_joints[axis] = round(
                place_joints[axis]
                + float(layer["joints_down"][axis])
                - float(base_layer["joints_down"][axis]),
                2,
            )

    ready_hold = with_gripper(stack_cfg.get("ready_pose", DEFAULT_STACKING["ready_pose"]), grasp_angle)
    place_hold = [place_joints[0], place_joints[1], place_joints[2], place_joints[3], wrist, grasp_angle]
    turn = list(ready_hold)
    turn[0] = place_hold[0]
    lift = with_gripper(stack_cfg.get("lift_pose", DEFAULT_STACKING["lift_pose"]), release_angle)
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, release_angle]
    return {
        "ready_hold": ready_hold,
        "turn": turn,
        "place": place_hold,
        "place_xy": place_xy,
        "lift": lift,
        "home": home,
        "grasp_angle": grasp_angle,
        "release": release_angle,
    }


def execute_held_place_plan(args, arm, plan):
    move_joints(arm, plan["ready_hold"], args.move_ms, args.execute)
    move_joints(arm, plan["turn"], args.move_ms, args.execute)
    move_joints(arm, plan["place"], args.place_ms, args.execute)
    if args.execute and args.place_settle > 0:
        time.sleep(args.place_settle)
    set_gripper(arm, plan["release"], args.release_ms, args.execute)
    if args.execute and args.release_settle > 0:
        time.sleep(args.release_settle)
    move_joints(arm, plan["lift"], args.move_ms, args.execute)
    if args.home:
        move_joints(arm, plan["home"], args.move_ms, args.execute)


def cmd_detect(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    cap = open_camera(args.camera, args.width, args.height)
    print(f"CONFIG {calib.config_dir}", flush=True)
    print("detecting; Ctrl+C to stop", flush=True)
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("camera read failed", flush=True)
                time.sleep(0.2)
                continue
            frame = cv2.resize(raw, (args.width, args.height))
            if args.perspective:
                frame = perspective_transform(calib.dp, frame)
            target = detect_best(
                frame,
                args.color,
                args.min_area,
                args.detect_point,
                args.include_region,
                args.exclude_region,
            )
            if target:
                xy = pixel_to_xy(target.cx, target.cy, calib, args.x_nudge, args.y_nudge)
                print(
                    f"TARGET {target.name} area={target.area:.0f} "
                    f"cx={target.cx} cy={target.cy} "
                    f"centroid=({target.centroid_cx},{target.centroid_cy}) "
                    f"x={xy[0]:.5f} y={xy[1]:.5f}",
                    flush=True,
                )
            else:
                print("NO_TARGET", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        cap.release()


def cmd_ik(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    target, _ = capture_target(args, calib)
    xy = pixel_to_xy(target.cx, target.cy, calib, args.x_nudge, args.y_nudge)
    ik = RosIkClient(args.service_timeout)
    try:
        joints = ik.inverse(*xy, args.tar_z)
    finally:
        ik.close()
    print_plan(target, xy, joints, calib)
    return 0


def cmd_once(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, args.release]
    arm = make_arm() if args.execute else None

    if args.execute:
        # Calibration is valid only from the camera pose saved in XYT_config.
        # Always return there before capturing the target for an executed move.
        move_joints(arm, home, 1000, True)
    else:
        print("DRY_RUN capture uses the current camera pose", flush=True)

    target, _ = capture_target(args, calib)
    xy = pixel_to_xy(target.cx, target.cy, calib, args.x_nudge, args.y_nudge)
    warn = check_reachable(xy, args)
    if warn:
        print(f"WARN {warn}", flush=True)
    plan = build_motion_plan(args, calib, target, xy)
    print_plan(target, xy, plan["joints"], calib)
    print(f"APPROACH {plan['approach']}", flush=True)

    if not args.execute:
        print("DRY_RUN add --execute to move arm", flush=True)
    if args.step == "home":
        return 0
    if warn and args.skip_unreachable:
        print("SKIP target unreachable; not moving (use --no-skip-unreachable to force)", flush=True)
        return 0
    execute_step_plan(args, arm, plan, args.step)
    return 0


def cmd_auto(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    # Even in dry-run mode, return to the calibrated camera pose before
    # capturing; otherwise the camera may still be at the previous test pose.
    arm = make_arm()
    cycles = 0

    if not args.execute:
        print("DRY_RUN only the camera-home move is executed; add --execute to pick", flush=True)

    while args.cycles == 0 or cycles < args.cycles:
        cycles += 1
        print(f"\n=== CYCLE {cycles} ===", flush=True)
        home = [calib.xy[0], calib.xy[1], 0, 0, 90, args.release]
        move_joints(arm, home, 1000, True)

        target, _ = capture_target(args, calib)
        xy = pixel_to_xy(target.cx, target.cy, calib, args.x_nudge, args.y_nudge)
        warn = check_reachable(xy, args)
        if warn:
            print(f"WARN {warn}", flush=True)
        plan = build_motion_plan(args, calib, target, xy)
        print_plan(target, xy, plan["joints"], calib)
        print(f"APPROACH {plan['approach']}", flush=True)
        if warn and args.skip_unreachable:
            print("SKIP target unreachable; not moving this cycle", flush=True)
        else:
            execute_step_plan(args, arm, plan, "pick")

        if args.cycles == 0 or cycles < args.cycles:
            time.sleep(args.cooldown)
    return 0


def cmd_probe_fixed(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    stack_cfg = load_stacking_config(args.stacking_config, calib)
    fixed_xy = (stack_cfg.get("fixed_pick_xy") or {}).get(args.color)
    if args.x is not None and args.y is not None:
        xy = (round(float(args.x), 5), round(float(args.y), 5))
    elif fixed_xy and len(fixed_xy) == 2:
        xy = (round(float(fixed_xy[0]), 5), round(float(fixed_xy[1]), 5))
    else:
        raise RuntimeError(f"no fixed xy for color={args.color}; pass --x and --y")

    arm = make_arm() if args.execute else None
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, args.release]
    ready = [calib.xy[0], 80, 50, 50, args.wrist, args.open]

    ik = RosIkClient(args.service_timeout)
    try:
        joints = ik.inverse(xy[0], xy[1], args.tar_z)
        above_joints = elevated_xy_joints(ik, xy, args.approach_z)
    finally:
        ik.close()

    if above_joints is None:
        approach = [ready[0], 80, 50, 50, args.wrist, args.open]
    else:
        approach = [above_joints[0], above_joints[1], above_joints[2], above_joints[3], args.wrist, args.open]
    down = [joints[0], joints[1], joints[2], joints[3], args.wrist, args.open]
    grasp = [joints[0], joints[1], joints[2], joints[3], args.wrist, args.close]
    if args.base is not None:
        approach[0] = float(args.base)
        down[0] = float(args.base)
        grasp[0] = float(args.base)
    lift = list(approach)
    lift[5] = args.close

    print(f"FIXED_PROBE color={args.color} x={xy[0]:.5f} y={xy[1]:.5f}", flush=True)
    print(f"IK {joints}", flush=True)
    print(f"APPROACH {approach}", flush=True)
    print(f"DOWN {down}", flush=True)
    if not args.execute:
        print("DRY_RUN add --execute to move arm", flush=True)
        return 0

    move_joints(arm, ready, args.move_ms, True)
    set_gripper(arm, args.open, args.release_ms, True)
    if args.step in ("approach", "down", "grasp", "pick"):
        move_joints(arm, approach, args.move_ms, True)
    if args.step in ("down", "grasp", "pick"):
        move_joints(arm, down, args.grasp_ms, True)
    if args.step in ("grasp", "pick"):
        set_gripper(arm, args.close, args.grip_ms, True)
        if args.grip_settle > 0:
            time.sleep(args.grip_settle)
    if args.step == "pick":
        move_joints(arm, lift, args.move_ms, True)
    if args.home:
        move_joints(arm, home, args.move_ms, True)
    return 0


def cmd_place_held(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    stack_cfg = load_stacking_config(args.stacking_config, calib)
    layers = list(stack_cfg.get("layers", []))
    layer_index = args.layer - 1
    if layer_index < 0 or layer_index >= len(layers):
        raise RuntimeError(f"layer must be in 1..{len(layers)}")
    layer = layers[layer_index]

    release_angle = args.release if args.release is not None else stack_cfg["gripper"].get("release", 30)
    grasp_angle = args.close if args.close is not None else stack_cfg["gripper"].get("grasp", 100)
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, grasp_angle]
    arm = make_arm()

    if not args.execute:
        print("DRY_RUN only the camera-home move is executed; add --execute to place held cube", flush=True)
    print(
        f"PLACE_HELD held_color={args.held_color} layer={args.layer} "
        f"center_color={args.center_color}",
        flush=True,
    )
    move_joints(arm, home, 1000, True)

    old_color = args.color
    try:
        args.color = args.center_color
        _, place_xy = capture_stack_center(args, calib, stack_cfg, args.center_color, layer, layer_index)
    finally:
        args.color = old_color

    plan = build_held_place_plan(args, calib, layer, stack_cfg, place_xy)
    print(
        f"PLACE_XY x={plan['place_xy'][0]:.5f} y={plan['place_xy'][1]:.5f}",
        flush=True,
    )
    print(f"STACK_PLACE level={layer.get('level', args.layer)} joints={plan['place']}", flush=True)
    if not args.execute:
        print("DRY_RUN add --execute to move arm and release held cube", flush=True)
        return 0
    execute_held_place_plan(args, arm, plan)
    return 0


def cmd_stack(args) -> int:
    calib = load_calibration(find_config_dir(args.config))
    stack_cfg = load_stacking_config(args.stacking_config, calib)
    layers = list(stack_cfg.get("layers", []))
    if not layers:
        raise RuntimeError("stacking config has no layers")

    max_layers = int(stack_cfg.get("max_layers", len(layers)))
    start_index = args.start_layer - 1
    if start_index < 0 or start_index >= max_layers or start_index >= len(layers):
        raise RuntimeError(f"start-layer must be in 1..{min(max_layers, len(layers))}")
    count = min(args.layers, max_layers - start_index, len(layers) - start_index)
    if count <= 0:
        raise RuntimeError("layers must be > 0")
    pick_order = list(stack_cfg.get("pick_order", []))
    color_rois = stack_cfg.get("color_rois", {}) or {}
    detect_poses = stack_cfg.get("detect_poses", {}) or {}
    pick_points = stack_cfg.get("pick_points", {}) or {}

    arm = make_arm()
    release_angle = args.release if args.release is not None else stack_cfg["gripper"].get("release", 30)
    home = [calib.xy[0], calib.xy[1], 0, 0, 90, release_angle]

    if not args.execute:
        print("DRY_RUN only the camera-home move is executed; add --execute to stack", flush=True)
    print(
        f"STACK start_layer={args.start_layer} layers={count}/{max_layers} "
        f"detect_point={args.detect_point}",
        flush=True,
    )
    if stack_cfg.get("use_fixed_pick_fallback", False):
        print(f"FIXED_PICK_FALLBACK enabled points={pick_points}", flush=True)

    for idx in range(start_index, start_index + count):
        layer = layers[idx]
        layer_color = args.color
        include_region = list(args.include_region)
        if pick_order and args.color == "any":
            layer_color = pick_order[idx % len(pick_order)]
        roi = color_rois.get(layer_color)
        if roi and not include_region:
            include_region = [",".join(str(v) for v in roi)]

        print(
            f"\n=== STACK LAYER {layer.get('level', idx + 1)} color={layer_color} ===",
            flush=True,
        )
        if include_region:
            print(f"PICK_ROI color={layer_color} region={include_region}", flush=True)
        detect_pose = detect_poses.get(layer_color, home)
        print(f"DETECT_POSE color={layer_color} joints={detect_pose}", flush=True)
        move_joints(arm, detect_pose, 1000, True)

        place_xy_override = None
        if (
            stack_cfg.get("detect_stack_center", False)
            and idx + 1 >= int(stack_cfg.get("stack_center_min_layer", 2))
        ):
            center_color = stack_center_color_for_layer(stack_cfg, pick_order, idx, layer)
            if not center_color:
                raise RuntimeError(f"no stack center color for layer={idx + 1}")
            center_target, place_xy_override = capture_stack_center(
                args, calib, stack_cfg, center_color, layer, idx
            )

        post_center_pose = (stack_cfg.get("post_center_detect_poses") or {}).get(layer_color)
        if post_center_pose:
            print(f"POST_CENTER_DETECT_POSE color={layer_color} joints={post_center_pose}", flush=True)
            move_joints(arm, post_center_pose, 1000, True)

        old_color, old_include = args.color, args.include_region
        args.color, args.include_region = layer_color, include_region
        try:
            target, _ = capture_stack_target(args, calib, stack_cfg, layer_color)
        finally:
            args.color, args.include_region = old_color, old_include
        fixed_xy = (stack_cfg.get("fixed_pick_xy") or {}).get(layer_color)
        if fixed_xy and len(fixed_xy) == 2:
            xy = (round(float(fixed_xy[0]), 5), round(float(fixed_xy[1]), 5))
            print(f"FIXED_PICK_XY color={layer_color} x={xy[0]:.5f} y={xy[1]:.5f}", flush=True)
        else:
            xy = pixel_to_xy(target.cx, target.cy, calib, args.x_nudge, args.y_nudge)
        warn = check_reachable(xy, args)
        if warn:
            print(f"WARN {warn}", flush=True)
        plan = build_stack_plan(
            args,
            calib,
            target,
            xy,
            layer,
            stack_cfg,
            place_xy_override=place_xy_override,
        )
        print_plan(target, xy, plan["joints"], calib)
        if plan.get("place_xy") is not None:
            print(
                f"PLACE_XY x={plan['place_xy'][0]:.5f} y={plan['place_xy'][1]:.5f}",
                flush=True,
            )
        if plan.get("place_approach") is not None:
            print(
                f"PLACE_APPROACH z={plan['place_approach_z']:.5f} joints={plan['place_approach']}",
                flush=True,
            )
        print(
            f"PICK_PARAMS color={layer_color} z={plan['pick_z']:.5f} base={plan['pick_base']}",
            flush=True,
        )
        print(f"STACK_PLACE level={layer.get('level', idx + 1)} joints={plan['place']}", flush=True)
        print(f"APPROACH {plan['approach']}", flush=True)

        if warn and args.skip_unreachable:
            print("SKIP target unreachable; not moving this layer", flush=True)
            continue
        execute_stack_plan(args, arm, plan)

        if idx + 1 < start_index + count:
            time.sleep(args.cooldown)
    return 0


def add_common(ap):
    ap.add_argument("--config", default=None, help="directory containing dp.bin/offset.txt/XYT_config.txt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--color", default="any", choices=["any", *COLORS.keys()])
    ap.add_argument("--min-area", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--stable-frames", type=int, default=6)
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--detect-point", choices=["ground", "centroid"], default="ground",
                    help="ground uses the cube footprint point; centroid matches the original sample")
    ap.add_argument("--include-region", action="append", default=[],
                    help="only accept detected points inside x1,y1,x2,y2; repeatable")
    ap.add_argument("--exclude-region", action="append", default=[],
                    help="ignore detected points inside x1,y1,x2,y2; repeatable")
    # 与华为原版 garbage_identify.get_pos 一致：识别在 resize 后的原图上进行，
    # 不做透视变换。透视变换(--perspective) 仅用于标定阶段生成 dp.bin，
    # 这里默认关闭；坐标公式 (cx-320)/4000 等魔数是针对原图标定的。
    ap.add_argument("--perspective", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--x-nudge", type=float, default=0.0, help="global meter correction after realtime detection")
    ap.add_argument("--y-nudge", type=float, default=0.0, help="global meter correction after realtime detection")
    # 舒适抓取区边界（米）。超出则警告，默认跳过不抓，避免在前倾/扭曲姿态夹空。
    ap.add_argument("--reach-xmax", type=float, default=0.13)
    ap.add_argument("--reach-ymin", type=float, default=0.15)
    ap.add_argument("--reach-ymax", type=float, default=0.32)
    ap.add_argument("--skip-unreachable", action=argparse.BooleanOptionalAction, default=True,
                    help="坐标超出舒适区时跳过抓取（--no-skip-unreachable 强制抓）")


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect")
    add_common(p_detect)
    p_detect.add_argument("--interval", type=float, default=0.5)
    p_detect.set_defaults(func=cmd_detect)

    p_ik = sub.add_parser("ik")
    add_common(p_ik)
    p_ik.add_argument("--service-timeout", type=float, default=5.0)
    p_ik.add_argument("--tar-z", type=float, default=0.0)
    p_ik.set_defaults(func=cmd_ik)

    p_once = sub.add_parser("once")
    add_common(p_once)
    p_once.add_argument("--service-timeout", type=float, default=5.0)
    p_once.add_argument("--tar-z", type=float, default=0.0)
    p_once.add_argument("--execute", action="store_true")
    p_once.add_argument("--step", choices=["home", "ready", "approach", "grasp", "pick"], default="pick")
    p_once.add_argument("--open", type=float, default=0)
    p_once.add_argument("--close", type=float, default=175)
    p_once.add_argument("--release", type=float, default=30)
    p_once.add_argument("--wrist", type=float, default=265)
    p_once.add_argument("--approach-z", type=float, default=0.22, help="higher z IK point before descending")
    p_once.add_argument("--approach-ms", type=int, default=1000)
    p_once.add_argument("--grasp-ms", type=int, default=1200)
    p_once.add_argument("--move-ms", type=int, default=1500, help="抬起/搬运段速度(ms)，越大越慢越稳")
    p_once.add_argument("--grip-ms", type=int, default=800, help="夹紧动作时间(ms)")
    p_once.add_argument("--grip-settle", type=float, default=0.6, help="夹紧后停顿(s)，让方块咬实")
    p_once.set_defaults(func=cmd_once)

    p_auto = sub.add_parser("auto")
    add_common(p_auto)
    p_auto.add_argument("--service-timeout", type=float, default=5.0)
    p_auto.add_argument("--tar-z", type=float, default=0.0)
    p_auto.add_argument("--execute", action="store_true")
    p_auto.add_argument("--open", type=float, default=0)
    p_auto.add_argument("--close", type=float, default=175)
    p_auto.add_argument("--release", type=float, default=30)
    p_auto.add_argument("--wrist", type=float, default=265)
    p_auto.add_argument("--approach-z", type=float, default=0.22)
    p_auto.add_argument("--approach-ms", type=int, default=1000)
    p_auto.add_argument("--grasp-ms", type=int, default=1200)
    p_auto.add_argument("--move-ms", type=int, default=1500, help="抬起/搬运段速度(ms)，越大越慢越稳")
    p_auto.add_argument("--grip-ms", type=int, default=800, help="夹紧动作时间(ms)")
    p_auto.add_argument("--grip-settle", type=float, default=0.6, help="夹紧后停顿(s)，让方块咬实")
    p_auto.add_argument("--cycles", type=int, default=1, help="0 means loop forever")
    p_auto.add_argument("--cooldown", type=float, default=1.0)
    p_auto.set_defaults(func=cmd_auto)

    p_probe = sub.add_parser("probe-fixed")
    add_common(p_probe)
    p_probe.add_argument("--service-timeout", type=float, default=5.0)
    p_probe.add_argument("--stacking-config", default=None, help="optional path to stacking.yaml")
    p_probe.add_argument("--tar-z", type=float, default=0.0)
    p_probe.add_argument("--execute", action="store_true")
    p_probe.add_argument("--step", choices=["approach", "down", "grasp", "pick"], default="approach")
    p_probe.add_argument("--x", type=float, default=None)
    p_probe.add_argument("--y", type=float, default=None)
    p_probe.add_argument("--base", type=float, default=None, help="override joint1/base angle for direction probing")
    p_probe.add_argument("--open", type=float, default=0)
    p_probe.add_argument("--close", type=float, default=175)
    p_probe.add_argument("--release", type=float, default=30)
    p_probe.add_argument("--wrist", type=float, default=265)
    p_probe.add_argument("--approach-z", type=float, default=0.22)
    p_probe.add_argument("--move-ms", type=int, default=1500)
    p_probe.add_argument("--grasp-ms", type=int, default=1200)
    p_probe.add_argument("--grip-ms", type=int, default=800)
    p_probe.add_argument("--release-ms", type=int, default=700)
    p_probe.add_argument("--grip-settle", type=float, default=0.6)
    p_probe.add_argument("--home", action=argparse.BooleanOptionalAction, default=False)
    p_probe.set_defaults(func=cmd_probe_fixed)

    p_place = sub.add_parser("place-held")
    add_common(p_place)
    p_place.add_argument("--service-timeout", type=float, default=5.0)
    p_place.add_argument("--tar-z", type=float, default=0.0)
    p_place.add_argument("--execute", action="store_true")
    p_place.add_argument("--stacking-config", default=None, help="optional path to stacking.yaml")
    p_place.add_argument("--layer", type=int, default=3, help="1-based stack layer for the held cube")
    p_place.add_argument("--center-color", default="yellow", choices=list(COLORS.keys()),
                         help="visible top cube color to center on")
    p_place.add_argument("--held-color", default="blue", choices=list(COLORS.keys()),
                         help="cube currently held by the gripper")
    p_place.add_argument("--close", type=float, default=None)
    p_place.add_argument("--release", type=float, default=None)
    p_place.add_argument("--wrist", type=float, default=265)
    p_place.add_argument("--move-ms", type=int, default=1500)
    p_place.add_argument("--place-ms", type=int, default=1500)
    p_place.add_argument("--release-ms", type=int, default=1200)
    p_place.add_argument("--place-settle", type=float, default=0.5)
    p_place.add_argument("--release-settle", type=float, default=0.4)
    p_place.add_argument("--home", action=argparse.BooleanOptionalAction, default=True)
    p_place.set_defaults(func=cmd_place_held)

    p_stack = sub.add_parser("stack")
    add_common(p_stack)
    p_stack.add_argument("--service-timeout", type=float, default=5.0)
    p_stack.add_argument("--tar-z", type=float, default=0.0)
    p_stack.add_argument("--execute", action="store_true")
    p_stack.add_argument("--stacking-config", default=None, help="optional path to stacking.yaml")
    p_stack.add_argument("--start-layer", type=int, default=1,
                         help="1-based layer index to start from; use 2 after layer 1 is already placed")
    p_stack.add_argument("--layers", type=int, default=4, help="number of layers to stack")
    p_stack.add_argument("--open", type=float, default=None)
    p_stack.add_argument("--close", type=float, default=None)
    p_stack.add_argument("--release", type=float, default=None)
    p_stack.add_argument("--wrist", type=float, default=265)
    p_stack.add_argument("--approach-z", type=float, default=0.22)
    p_stack.add_argument("--place-approach-z", type=float, default=0.0,
                         help="optional high z waypoint above the stack before descending to place")
    p_stack.add_argument("--approach-ms", type=int, default=1000)
    p_stack.add_argument("--grasp-ms", type=int, default=1200)
    p_stack.add_argument("--move-ms", type=int, default=1500, help="抬起/搬运段速度(ms)，越大越慢越稳")
    p_stack.add_argument("--place-ms", type=int, default=1500, help="下降到堆叠点的时间(ms)")
    p_stack.add_argument("--grip-ms", type=int, default=800, help="夹紧动作时间(ms)")
    p_stack.add_argument("--release-ms", type=int, default=1200, help="松开动作时间(ms)")
    p_stack.add_argument("--grip-settle", type=float, default=0.6, help="夹紧后停顿(s)")
    p_stack.add_argument("--place-settle", type=float, default=0.5, help="到达堆叠点后停顿(s)")
    p_stack.add_argument("--release-settle", type=float, default=0.4, help="松爪后抬起前停顿(s)")
    p_stack.add_argument("--cooldown", type=float, default=1.0)
    p_stack.add_argument("--home-each", action=argparse.BooleanOptionalAction, default=True,
                         help="return to camera pose after each layer")
    p_stack.set_defaults(func=cmd_stack)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
