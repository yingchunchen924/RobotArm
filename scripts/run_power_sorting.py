"""电力物品分类抓取主入口（开发计划阶段六）。

组装：检测器（配置选 onnx/acl 后端）+ 摄像头帧源 + 逆运动学 + 机械臂 -> PowerSortingRunner，
连续识别并按类别分拣。

后端由 --backend 选择：
    onnx  -> OnnxDetector（PC 验证用）
    acl   -> AclDetector（Atlas 真机，.om 模型）

⚠️ 真机运行需要：摄像头、ROS2 Kinemarics 服务（dofbot_server）、机械臂串口驱动。
   --mock 可在无硬件的 PC 上用假逆解+假机械臂烟测整条装配（识别仍走真实后端）。

用法：
    # 真机
    python scripts/run_power_sorting.py --backend acl --model models/power_objects.om
    # PC 烟测（识别真实，臂/逆解为 Mock）
    python scripts/run_power_sorting.py --backend onnx --model models/power_objects.onnx --mock
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import config_loader as cl                       # noqa: E402
from robotarm.detectors import build_detector                  # noqa: E402
from robotarm.interfaces import MockArmDriver, MockKinematics  # noqa: E402
from robotarm.runner import PowerSortingRunner                 # noqa: E402
from robotarm.states import StatusEvent                        # noqa: E402


def make_camera_source(camera_index: int, width: int, height: int):
    """返回一个读帧函数：每次调用返回缩放后的一帧 BGR 图，失败返回 None。"""
    import cv2
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}")

    def read():
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.resize(frame, (width, height))

    read.release = cap.release  # type: ignore[attr-defined]
    return read


def load_offset() -> float:
    """从标定 offset.txt 读取硬件误差补偿（真机标定产物）。缺失则 0。"""
    arm_cfg = cl.get_arm_config()
    path = arm_cfg.get("calibration_files", {}).get("offset_txt", "")
    full = os.path.join(_ROOT, path) if path else ""
    if full and os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                return float(f.readline().split()[0])
        except Exception:
            pass
    return 0.0


def build_kinematics(mock: bool):
    if mock:
        return MockKinematics()
    # 真机：ROS2 Kinemarics 服务客户端。
    # 实现放在 ros2_ws/src/power_arm_control（继承 robotarm.interfaces.Kinematics，
    # 在 inverse() 里调用 dofbot_info/srv/Kinemarics 的 ik 模式）。
    try:
        from power_arm_control.ros2_kinematics import Ros2Kinematics  # type: ignore
        return Ros2Kinematics()
    except Exception as e:
        print(f"无法加载 ROS2 逆解客户端（真机功能包未就绪）：{e}", file=sys.stderr)
        print("可加 --mock 在 PC 上用假逆解烟测。", file=sys.stderr)
        raise


def build_arm(mock: bool):
    if mock:
        return MockArmDriver()
    try:
        from power_arm_control.serial_arm import SerialArmDriver  # type: ignore
        return SerialArmDriver()
    except Exception as e:
        print(f"无法加载串口机械臂驱动（真机功能包未就绪）：{e}", file=sys.stderr)
        print("可加 --mock 在 PC 上用假机械臂烟测。", file=sys.stderr)
        raise


def on_status(ev: StatusEvent):
    tgt = f" [{ev.target}]" if ev.target else ""
    extra = f" - {ev.detail}" if ev.detail else ""
    print(f"  状态: {ev.label}{tgt}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(description="电力物品分类抓取主程序")
    ap.add_argument("--backend", default="acl", choices=["onnx", "acl"])
    ap.add_argument("--model", required=True, help=".onnx 或 .om 模型路径")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="逆解+机械臂用 Mock（PC 烟测）")
    ap.add_argument("--max-frames", type=int, default=0, help="处理帧数上限，0=不限")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    web_cfg = cl.get_web_config()
    w = web_cfg.get("video", {}).get("width", 640)
    h = web_cfg.get("video", {}).get("height", 480)

    detector = build_detector(args.backend, args.model, conf_thres=args.conf)
    kinematics = build_kinematics(args.mock)
    arm = build_arm(args.mock)
    offset = load_offset()

    cam = make_camera_source(args.camera, w, h)

    runner = PowerSortingRunner(
        detector=detector, kinematics=kinematics, arm=arm,
        frame_source=cam, offset=offset,
        clock=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    runner.pipeline.reporter.subscribe(on_status)

    print(f"开始连续分拣（backend={args.backend}, mock={args.mock}, offset={offset}）")
    try:
        stats = runner.run(max_frames=args.max_frames or None)
    except KeyboardInterrupt:
        runner.stop()
        stats = runner.stats
    finally:
        if hasattr(cam, "release"):
            cam.release()

    print(f"\n结束。帧={stats.frames} 抓取={stats.grasped} "
          f"空={stats.empty} 失败={stats.failed}")
    print("库房计数:", dict(runner.pipeline.inventory.counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
