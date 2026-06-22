"""阶段六端到端演示：真实 ONNX 检测器 -> pipeline 分拣 -> Mock 机械臂。

证明「识别（真 onnxruntime 推理）」能驱动「分类抓取」整条闭环，全程在 PC 上跑，
不需要真机硬件。检测器是合成 ONNX（输出已知框），机械臂是 Mock。

运行：
    python scripts/demo_power_sorting.py
（需 onnxruntime + onnx；若本机不可用会提示跳过。）
"""

import itertools
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import dataset as ds                              # noqa: E402
from robotarm.interfaces import MockArmDriver, MockKinematics  # noqa: E402
from robotarm.runner import PowerSortingRunner                 # noqa: E402
from robotarm.states import StatusEvent                        # noqa: E402


def make_synthetic_onnx(path, boxes):
    """boxes: [(cx,cy,w,h,cls), ...] 网络坐标系。生成常量输出 ONNX。"""
    import onnx
    from onnx import TensorProto, helper

    nc = ds.num_classes()
    det = np.zeros((1, len(boxes), 5 + nc), dtype=np.float32)
    for i, (cx, cy, w, h, cls) in enumerate(boxes):
        det[0, i, :5] = [cx, cy, w, h, 0.9]
        det[0, i, 5 + cls] = 0.9
    const = helper.make_node(
        "Constant", [], ["raw"],
        value=helper.make_tensor("v", TensorProto.FLOAT, det.shape, det.flatten()))
    ident = helper.make_node("Identity", ["raw"], ["output"])
    g = helper.make_graph(
        [const, ident], "y",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(boxes), 5 + nc])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def on_status(ev: StatusEvent):
    tgt = f" [{ev.target}]" if ev.target else ""
    extra = f" - {ev.detail}" if ev.detail else ""
    print(f"  状态: {ev.label}{tgt}{extra}")


def main():
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception as e:
        print(f"本机 onnxruntime/onnx 不可用，跳过演示：{e}")
        print("可 pip install onnxruntime==1.16.3 onnx 后重试。")
        return 0

    from robotarm.detectors import OnnxDetector

    n2i = ds.name_to_id()
    # 三类电力物品，分别落到 电子区/工具区/金具区
    onnx_path = os.path.join(tempfile.gettempdir(), "demo_power.onnx")
    make_synthetic_onnx(onnx_path, [(320, 320, 40, 40, n2i["resistor"])])

    detector = OnnxDetector(onnx_path, img_size=640, conf_thres=0.25)

    # 帧源：返回纯色图（内容不影响合成模型输出，但走完整预处理）
    frames = iter([np.full((480, 640, 3), 60, np.uint8) for _ in range(3)])
    counter = itertools.count()
    runner = PowerSortingRunner(
        detector=detector,
        kinematics=MockKinematics(),
        arm=MockArmDriver(),
        frame_source=lambda: next(frames, None),
        clock=lambda: f"2026-06-21 11:{next(counter):02d}:00",
        skip_repeat=0,
    )
    runner.pipeline.reporter.subscribe(on_status)

    print("=" * 56)
    print("  阶段六端到端：ONNX 识别 -> 分类抓取（Mock 臂）")
    print("=" * 56)
    stats = runner.run(max_frames=3)

    print("\n" + "=" * 56)
    print(f"  帧={stats.frames} 抓取={stats.grasped} 空={stats.empty} 失败={stats.failed}")
    print("  库房计数:", dict(runner.pipeline.inventory.counts))
    print("  操作日志:")
    for r in runner.pipeline.inventory.logs:
        print(f"    {r.timestamp}  {r.category:10s} {r.result:8s} -> {r.zone}")
    print("\n识别由真实 onnxruntime 推理驱动；真机换 AclDetector + 真臂即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
