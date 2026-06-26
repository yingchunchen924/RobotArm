#!/usr/bin/env python3
"""Single-file color grasp diagnostics for Atlas 200I DK A2 + DOFBOT.

Copy this file to the board and run it there. It intentionally supports
step-by-step modes so a bad calibration does not keep repeating full pick
cycles.

Examples:
    python3 color_grasp_debug.py detect --color blue
    python3 color_grasp_debug.py gripper --open 90 --close 150
    python3 color_grasp_debug.py once --color blue --dry-run
    python3 color_grasp_debug.py once --color blue
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


COLORS = {
    "red": ((0, 120, 80), (8, 255, 255), (170, 120, 80), (180, 255, 255)),
    "yellow": ((22, 100, 100), (35, 255, 255)),
    "green": ((45, 80, 80), (75, 255, 255)),
    "blue": ((105, 100, 80), (125, 255, 255)),
}


@dataclass
class Target:
    name: str
    area: float
    cx: int
    cy: int


def clamp(value: float, low: float = 0.0, high: float = 180.0) -> float:
    return max(low, min(high, value))


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index} with CAP_V4L2")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def detect_best(frame, color: str, min_area: int) -> Target | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    names = [color] if color != "any" else list(COLORS.keys())
    best: Target | None = None

    for name in names:
        ranges = COLORS[name]
        mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
        if name == "red":
            mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
            mask = cv2.bitwise_or(mask, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            if best is None or area > best.area:
                best = Target(name=name, area=area, cx=cx, cy=cy)
    return best


def pixel_to_joints(args, cx: int, cy: int):
    """Temporary linear pixel-to-joint calibration for simple color blocks."""
    base = args.base0 + (cx - args.cx0) * args.k_base
    j2 = args.j2_0 + (cy - args.cy0) * args.k_j2
    base = clamp(base)
    j2 = clamp(j2)

    above = [
        round(base, 2),
        round(j2, 2),
        args.above_j3,
        args.above_j4,
        args.wrist,
        args.open,
    ]
    down = [
        round(base, 2),
        round(j2 + args.down_j2_delta, 2),
        args.down_j3,
        args.down_j4,
        args.wrist,
        args.open,
    ]
    return above, down


def make_arm():
    import Arm_Lib

    return Arm_Lib.Arm_Device()


def move_joints(arm, joints, ms: int, dry_run: bool):
    print(f"MOVE {joints} {ms}ms", flush=True)
    if not dry_run:
        arm.Arm_serial_servo_write6_array(joints, ms)
        time.sleep(ms / 1000.0 + 0.15)


def set_gripper(arm, angle: float, ms: int, dry_run: bool):
    print(f"GRIP {angle} {ms}ms", flush=True)
    if not dry_run:
        arm.Arm_serial_servo_write(6, angle, ms)
        time.sleep(ms / 1000.0 + 0.15)


def capture_target(args) -> tuple[Target, np.ndarray]:
    cap = open_camera(args.camera, args.width, args.height)
    try:
        target = None
        frame = None
        for _ in range(args.warmup):
            ok, raw = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.resize(raw, (args.width, args.height))
            target = detect_best(frame, args.color, args.min_area)
            if target is not None:
                break
            time.sleep(0.05)
        if target is None or frame is None:
            raise RuntimeError("no target detected")
        return target, frame
    finally:
        cap.release()


def cmd_detect(args) -> int:
    cap = open_camera(args.camera, args.width, args.height)
    print("detecting; Ctrl+C to stop", flush=True)
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("camera read failed", flush=True)
                time.sleep(0.2)
                continue
            frame = cv2.resize(raw, (args.width, args.height))
            target = detect_best(frame, args.color, args.min_area)
            if target:
                print(
                    f"TARGET {target.name} area={target.area:.0f} "
                    f"cx={target.cx} cy={target.cy}",
                    flush=True,
                )
            else:
                print("NO_TARGET", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        cap.release()


def cmd_gripper(args) -> int:
    arm = make_arm()
    set_gripper(arm, args.open, 500, args.dry_run)
    time.sleep(0.5)
    set_gripper(arm, args.close, 500, args.dry_run)
    time.sleep(0.5)
    set_gripper(arm, args.open, 500, args.dry_run)
    return 0


def cmd_once(args) -> int:
    target, _ = capture_target(args)
    above, down = pixel_to_joints(args, target.cx, target.cy)
    print(
        f"TARGET {target.name} area={target.area:.0f} "
        f"cx={target.cx} cy={target.cy}",
        flush=True,
    )
    print(f"ABOVE {above}", flush=True)
    print(f"DOWN  {down}", flush=True)

    if args.dry_run:
        print("DRY_RUN no arm movement", flush=True)
        return 0

    arm = make_arm()
    move_joints(arm, args.home, 1000, False)
    set_gripper(arm, args.open, 400, False)

    if args.step in ("above", "down", "pick"):
        move_joints(arm, above, 900, False)
    if args.step in ("down", "pick"):
        move_joints(arm, down, 700, False)
    if args.step == "pick":
        set_gripper(arm, args.close, 500, False)
        move_joints(arm, above, 900, False)
        move_joints(arm, args.place, 1000, False)
        set_gripper(arm, args.open, 400, False)
        move_joints(arm, args.home, 1000, False)
    return 0


def add_common(ap):
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--color", default="any", choices=["any", *COLORS.keys()])
    ap.add_argument("--min-area", type=int, default=2000)
    ap.add_argument("--open", type=float, default=90)
    ap.add_argument("--close", type=float, default=150)
    ap.add_argument("--dry-run", action="store_true")


def add_calibration(ap):
    ap.add_argument("--cx0", type=float, default=320)
    ap.add_argument("--cy0", type=float, default=270)
    ap.add_argument("--base0", type=float, default=90)
    ap.add_argument("--j2-0", type=float, default=75)
    ap.add_argument("--k-base", type=float, default=-0.10)
    ap.add_argument("--k-j2", type=float, default=0.10)
    ap.add_argument("--above-j3", type=float, default=50)
    ap.add_argument("--above-j4", type=float, default=50)
    ap.add_argument("--down-j2-delta", type=float, default=15)
    ap.add_argument("--down-j3", type=float, default=35)
    ap.add_argument("--down-j4", type=float, default=40)
    ap.add_argument("--wrist", type=float, default=90)
    ap.add_argument("--home", nargs=6, type=float, default=[90, 130, 0, 0, 90, 30])
    ap.add_argument("--place", nargs=6, type=float, default=[90, 70, 70, 30, 90, 30])


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="print detected color target")
    add_common(p_detect)
    p_detect.add_argument("--interval", type=float, default=0.5)
    p_detect.set_defaults(func=cmd_detect)

    p_gripper = sub.add_parser("gripper", help="test open/close angles")
    add_common(p_gripper)
    p_gripper.set_defaults(func=cmd_gripper)

    p_once = sub.add_parser("once", help="run one target step")
    add_common(p_once)
    add_calibration(p_once)
    p_once.add_argument("--warmup", type=int, default=30)
    p_once.add_argument("--step", choices=["above", "down", "pick"], default="pick")
    p_once.set_defaults(func=cmd_once)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
