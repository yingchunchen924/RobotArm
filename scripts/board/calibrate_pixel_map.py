#!/usr/bin/env python3
"""在本机重新标定 相机像素 -> 机械臂 IK 输入坐标 的映射。

背景见 plans/jazzy-wishing-zebra.md。原仓库的 (cx-320)/4000 等系数是从别的
机器拷来的，对本机相机安装角度无效。本脚本在本机现场采集若干
``(cx, cy) -> (tar_x, tar_y)`` 样本，最小二乘拟合一个仿射 2x3 映射：

    [tar_x]   [a11 a12 a13]   [cx]
    [tar_y] = [a21 a22 a23] . [cy]
                              [ 1]

拟合结果打印成可直接贴进 ``config/arm.yaml`` 的 coordinate.affine。

设计原则与 color_sort_ros.py 一致：
- 纯函数(fit_affine/fit_homography/residuals/apply_*)不依赖硬件，可在 PC 单测。
- 触碰硬件(相机/ROS/机械臂)的部分集中在 client 类与子命令里。
- 默认安全：collect 只在显式 --execute 时移动机械臂。

样本来源（两种结合，见计划）：
1. 格点法：方块放标定板已知格点，操作者直接报该点的 tar_x/tar_y。
2. FK 辅助：让末端对准方块后用 fk 读真实笛卡尔坐标做参考(仅参考，不直接作
   拟合目标，因为 IK 输入与 FK 输出不是同一坐标系)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------- 纯函数（可单测） -----------------------------


@dataclass
class Sample:
    """一条标定样本：像素 (cx,cy) 对应的目标 IK 坐标 (tar_x,tar_y)。"""

    cx: float
    cy: float
    tar_x: float
    tar_y: float

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(float(d["cx"]), float(d["cy"]), float(d["tar_x"]), float(d["tar_y"]))


def fit_affine(samples: Sequence[Sample]) -> np.ndarray:
    """最小二乘拟合 2x3 仿射矩阵 A，使 [tar_x,tar_y]^T ≈ A·[cx,cy,1]^T。

    需要至少 3 个不共线的样本。返回 shape (2,3) 的 numpy 数组。
    """
    if len(samples) < 3:
        raise ValueError(f"need >=3 samples for affine, got {len(samples)}")
    src = np.array([[s.cx, s.cy, 1.0] for s in samples], dtype=np.float64)  # (N,3)
    dst = np.array([[s.tar_x, s.tar_y] for s in samples], dtype=np.float64)  # (N,2)
    # 解 src @ Aᵀ = dst  ->  Aᵀ = lstsq(src, dst)
    sol, *_ = np.linalg.lstsq(src, dst, rcond=None)  # (3,2)
    return sol.T  # (2,3)


def fit_homography(samples: Sequence[Sample]) -> np.ndarray:
    """拟合 3x3 单应矩阵 H（齐次），归一化 H[2,2]=1。需要至少 4 个样本。

    解 DLT：对每个点构造两行约束，对 9 维 h 求最小奇异向量。
    """
    if len(samples) < 4:
        raise ValueError(f"need >=4 samples for homography, got {len(samples)}")
    rows = []
    for s in samples:
        x, y, X, Y = s.cx, s.cy, s.tar_x, s.tar_y
        rows.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        rows.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    A = np.array(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def apply_affine(A: np.ndarray, cx: float, cy: float) -> Tuple[float, float]:
    v = A @ np.array([cx, cy, 1.0])
    return float(v[0]), float(v[1])


def apply_homography(H: np.ndarray, cx: float, cy: float) -> Tuple[float, float]:
    v = H @ np.array([cx, cy, 1.0])
    if abs(v[2]) < 1e-12:
        raise ValueError("homography produced w≈0")
    return float(v[0] / v[2]), float(v[1] / v[2])


def residuals(apply_fn, samples: Sequence[Sample]) -> dict:
    """计算每点误差(米)、RMS、最大误差。apply_fn(cx,cy)->(x,y)。"""
    per = []
    for s in samples:
        px, py = apply_fn(s.cx, s.cy)
        err = ((px - s.tar_x) ** 2 + (py - s.tar_y) ** 2) ** 0.5
        per.append(err)
    per_arr = np.array(per) if per else np.array([0.0])
    return {
        "per_point_m": per,
        "rms_m": float(np.sqrt(np.mean(per_arr ** 2))),
        "max_m": float(np.max(per_arr)),
        "n": len(samples),
    }


def leave_one_out_rms(samples: Sequence[Sample], fit_fn, apply_factory) -> float:
    """留一交叉验证 RMS(米)，评估泛化、防过拟合。

    fit_fn(subset)->model；apply_factory(model)->callable(cx,cy)->(x,y)。
    样本太少(<=最小拟合数)时返回 nan。
    """
    n = len(samples)
    errs = []
    for i in range(n):
        subset = [s for j, s in enumerate(samples) if j != i]
        try:
            model = fit_fn(subset)
        except ValueError:
            return float("nan")
        apply_fn = apply_factory(model)
        px, py = apply_fn(samples[i].cx, samples[i].cy)
        errs.append(((px - samples[i].tar_x) ** 2 + (py - samples[i].tar_y) ** 2) ** 0.5)
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))


def to_yaml_block(A: np.ndarray, stats: dict, calibrated_at: str = "") -> str:
    """生成可直接贴进 config/arm.yaml 的 coordinate 段文本。"""
    a = A.tolist()
    lines = [
        "coordinate:",
        "  model: affine",
        "  image_width: 640",
        "  image_height: 480",
        f"  affine: [[{a[0][0]:.8f}, {a[0][1]:.8f}, {a[0][2]:.8f}], "
        f"[{a[1][0]:.8f}, {a[1][1]:.8f}, {a[1][2]:.8f}]]",
        f"  rms_residual_m: {stats.get('rms_m', 0.0):.5f}",
    ]
    if calibrated_at:
        lines.append(f'  calibrated_at: "{calibrated_at}"')
    # 保留 legacy 字段以便随时回退
    lines += [
        "  # legacy 兼容字段（model=legacy 时使用）",
        "  x_center: 320",
        "  x_div: 4000.0",
        "  y_div: 3000.0",
        "  y_scale: 0.8",
        "  y_bias: 0.19",
    ]
    return "\n".join(lines)


def parse_samples_text(text: str) -> List[Sample]:
    """从文本解析样本。每行可以是 JSON 对象，或 'cx cy tar_x tar_y' 四个数。"""
    out: List[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            out.append(Sample.from_dict(json.loads(line)))
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        out.append(Sample(float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    return out


# ----------------------------- 硬件/ROS 部分 -----------------------------


COLORS = {
    "red": ((0, 120, 80), (8, 255, 255), (170, 120, 80), (180, 255, 255)),
    "yellow": ((22, 100, 100), (35, 255, 255)),
    "green": ((45, 80, 80), (75, 255, 255)),
    "blue": ((105, 100, 80), (125, 255, 255)),
}


def detect_best(frame, color: str, min_area: int):
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    names = [color] if color != "any" else list(COLORS.keys())
    best = None  # (cx, cy, area, name)
    for name in names:
        ranges = COLORS[name]
        mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
        if name == "red":
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3])))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if best is None or area > best[2]:
                best = (cx, cy, area, name)
    return best


def capture_pixel(args):
    """回拍照姿态拍若干帧，返回最稳定的方块像素中心 (cx, cy, area, name)。"""
    import cv2

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try:
        samples = []
        for _ in range(args.warmup):
            ok, raw = cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            frame = cv2.resize(raw, (args.width, args.height))
            best = detect_best(frame, args.color, args.min_area)
            if best:
                samples.append(best)
            time.sleep(0.03)
        if not samples:
            raise RuntimeError("no target detected")
        # 取众数颜色的均值
        last_name = samples[-1][3]
        same = [s for s in samples if s[3] == last_name]
        cx = round(sum(s[0] for s in same) / len(same))
        cy = round(sum(s[1] for s in same) / len(same))
        area = sum(s[2] for s in same) / len(same)
        return cx, cy, area, last_name
    finally:
        cap.release()


class RosKinClient:
    """ROS2 trial_service 客户端，支持 ik 与 fk。"""

    def __init__(self, timeout_sec: float = 5.0):
        import rclpy
        from dofbot_info.srv import Kinemarics

        self.rclpy = rclpy
        self.Kinemarics = Kinemarics
        rclpy.init(args=None)
        self.node = rclpy.create_node("calibrate_pixel_map")
        self.client = self.node.create_client(Kinemarics, "trial_service")
        if not self.client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("ROS2 service trial_service is not available")

    def _call(self, request):
        future = self.client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future)
        return future.result()

    def inverse(self, x: float, y: float, z: float = 0.0):
        req = self.Kinemarics.Request()
        req.tar_x = float(x)
        req.tar_y = float(y)
        req.tar_z = float(z)
        req.kin_name = "ik"
        res = self._call(req)
        if res is None:
            raise RuntimeError("IK no response")
        joints = [res.joint1, res.joint2, res.joint3, res.joint4, res.joint5]
        if joints[2] < 0:
            joints[1] += joints[2] / 2
            joints[3] += joints[2] * 3 / 4
            joints[2] = 0
        return [round(v, 2) for v in joints]

    def forward(self, joints6, unit: str = "deg"):
        """正运动学：传 6 个关节角，返回末端真实笛卡尔 (x,y,z)。"""
        scale = (np.pi / 180.0) if unit == "rad" else 1.0
        req = self.Kinemarics.Request()
        req.kin_name = "fk"
        req.cur_joint1 = float(joints6[0]) * scale
        req.cur_joint2 = float(joints6[1]) * scale
        req.cur_joint3 = float(joints6[2]) * scale
        req.cur_joint4 = float(joints6[3]) * scale
        req.cur_joint5 = float(joints6[4]) * scale
        req.cur_joint6 = float(joints6[5]) * scale
        res = self._call(req)
        if res is None:
            raise RuntimeError("FK no response")
        return round(res.x, 5), round(res.y, 5), round(res.z, 5)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def parse_pose(text: str) -> List[int]:
    parts = [int(round(float(p))) for p in text.replace(",", " ").split()]
    if len(parts) != 6:
        raise ValueError("pose must have 6 numbers, e.g. '91,135,0,0,90,30'")
    return parts


# ----------------------------- 子命令 -----------------------------


def cmd_collect(args) -> int:
    """逐点采集。每点：回拍照姿态拍像素 -> 操作者报该点 tar_x/tar_y -> 打印样本。"""
    pose = parse_pose(args.pose)
    arm = None
    if args.execute:
        import Arm_Lib

        arm = Arm_Lib.Arm_Device()

    print(f"# pose={pose}  color={args.color}", flush=True)
    print("# 每点：把方块放到一个已知位置，回车拍像素，然后输入该点 tar_x tar_y", flush=True)
    print("# 输入 q 结束。样本以 JSON 行打印，请自行复制保存。", flush=True)
    collected: List[Sample] = []
    while True:
        cmd = input("放好方块后按回车拍照（q 退出）> ").strip()
        if cmd.lower() == "q":
            break
        if arm is not None:
            arm.Arm_serial_servo_write6_array(pose, 1000)
            time.sleep(1.4)
        try:
            cx, cy, area, name = capture_pixel(args)
        except RuntimeError as exc:
            print(f"  detect failed: {exc}", flush=True)
            continue
        print(f"  detected {name} cx={cx} cy={cy} area={area:.0f}", flush=True)
        tar = input("  输入该点 tar_x tar_y（空格分隔，s 跳过）> ").strip()
        if tar.lower() == "s" or not tar:
            continue
        try:
            tx, ty = [float(v) for v in tar.replace(",", " ").split()[:2]]
        except ValueError:
            print("  解析失败，跳过", flush=True)
            continue
        s = Sample(cx, cy, tx, ty)
        collected.append(s)
        print(json.dumps({"cx": cx, "cy": cy, "tar_x": tx, "tar_y": ty}), flush=True)

    print("\n# ===== 全部样本（粘贴给 fit 子命令）=====", flush=True)
    for s in collected:
        print(json.dumps({"cx": s.cx, "cy": s.cy, "tar_x": s.tar_x, "tar_y": s.tar_y}), flush=True)
    return 0


def cmd_fit(args) -> int:
    """从文件/stdin 读样本，拟合仿射+单应，打印残差与可贴进 yaml 的文本块。"""
    if args.samples:
        with open(args.samples, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("# 粘贴样本（每行 JSON 或 'cx cy tar_x tar_y'），Ctrl-D 结束：", flush=True)
        text = sys.stdin.read()
    samples = parse_samples_text(text)
    print(f"parsed {len(samples)} samples", flush=True)
    if len(samples) < 3:
        print("ERROR need >=3 samples", file=sys.stderr)
        return 1

    A = fit_affine(samples)
    aff_stats = residuals(lambda cx, cy: apply_affine(A, cx, cy), samples)
    print("\n=== AFFINE 2x3 ===", flush=True)
    print(f"A = {A.tolist()}", flush=True)
    print(
        f"residual: rms={aff_stats['rms_m']*1000:.2f}mm max={aff_stats['max_m']*1000:.2f}mm",
        flush=True,
    )
    loo = leave_one_out_rms(samples, fit_affine, lambda m: (lambda cx, cy: apply_affine(m, cx, cy)))
    print(f"leave-one-out rms={loo*1000:.2f}mm" if loo == loo else "leave-one-out: n/a", flush=True)

    if len(samples) >= 4:
        try:
            H = fit_homography(samples)
            h_stats = residuals(lambda cx, cy: apply_homography(H, cx, cy), samples)
            print("\n=== HOMOGRAPHY 3x3 (对照) ===", flush=True)
            print(
                f"residual: rms={h_stats['rms_m']*1000:.2f}mm max={h_stats['max_m']*1000:.2f}mm",
                flush=True,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"homography failed: {exc}", flush=True)

    print("\n=== 每点误差(mm) ===", flush=True)
    for s, e in zip(samples, aff_stats["per_point_m"]):
        print(f"  cx={s.cx:.0f} cy={s.cy:.0f} -> ({s.tar_x:.4f},{s.tar_y:.4f})  err={e*1000:.2f}mm", flush=True)

    print("\n=== 贴进 config/arm.yaml 的 coordinate 段 ===", flush=True)
    print(to_yaml_block(A, aff_stats, args.calibrated_at), flush=True)
    return 0


def cmd_verify(args) -> int:
    """给一个像素点，用提供的 affine 打印映射坐标，可选调 IK 看可达。"""
    A = np.array(json.loads(args.affine), dtype=np.float64)
    if A.shape != (2, 3):
        print("ERROR affine must be 2x3", file=sys.stderr)
        return 1
    x, y = apply_affine(A, args.cx, args.cy)
    print(f"pixel({args.cx},{args.cy}) -> tar_x={x:.5f} tar_y={y:.5f}", flush=True)
    if args.ik:
        ik = RosKinClient(args.service_timeout)
        try:
            joints = ik.inverse(x, y, args.tar_z)
        finally:
            ik.close()
        print(f"IK -> {joints}", flush=True)
    return 0


def add_camera_args(ap):
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--color", default="blue", choices=["any", *COLORS.keys()])
    ap.add_argument("--min-area", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=20)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="逐点采集像素+人工报坐标")
    add_camera_args(p_collect)
    p_collect.add_argument("--pose", default="91,135,0,0,90,30", help="拍照姿态 6 关节角")
    p_collect.add_argument("--execute", action="store_true", help="真实移动机械臂到拍照姿态")
    p_collect.set_defaults(func=cmd_collect)

    p_fit = sub.add_parser("fit", help="拟合并打印 yaml 块")
    p_fit.add_argument("--samples", default=None, help="样本文件；省略则从 stdin 读")
    p_fit.add_argument("--calibrated-at", default="", help="标定日期，写进 yaml")
    p_fit.set_defaults(func=cmd_fit)

    p_verify = sub.add_parser("verify", help="用 affine 验证单个像素映射")
    p_verify.add_argument("--affine", required=True, help='2x3 JSON, e.g. "[[..],[..]]"')
    p_verify.add_argument("--cx", type=float, required=True)
    p_verify.add_argument("--cy", type=float, required=True)
    p_verify.add_argument("--ik", action="store_true")
    p_verify.add_argument("--tar-z", type=float, default=0.0)
    p_verify.add_argument("--service-timeout", type=float, default=5.0)
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
