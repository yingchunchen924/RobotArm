"""阶段三 端到端演示（全程 Mock，可在 PC 上直接运行）。

运行：
    python scripts/demo_pipeline.py

依次演示：
    1. 分拣模式：识别多个色块/物品并按类别放入对应库区。
    2. 堆叠模式：连续抓取并逐层堆叠。
最后打印库房统计与操作日志。
"""

import itertools
import os
import sys

# 把 src/ 加入路径
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm.interfaces import MockArmDriver, MockDetector, MockKinematics  # noqa: E402
from robotarm.pipeline import GraspPipeline, Mode  # noqa: E402
from robotarm.states import StatusEvent  # noqa: E402
from robotarm.target_selection import Detection  # noqa: E402


def banner(text):
    print("\n" + "=" * 56)
    print(f"  {text}")
    print("=" * 56)


def on_status(ev: StatusEvent):
    tgt = f" [{ev.target}]" if ev.target else ""
    extra = f" - {ev.detail}" if ev.detail else ""
    print(f"  状态: {ev.label}{tgt}{extra}")


def build(detections):
    clock = itertools.count()
    p = GraspPipeline(
        detector=MockDetector(detections),
        kinematics=MockKinematics(),
        arm=MockArmDriver(),
        clock=lambda: f"2026-06-21 10:{next(clock):02d}:00",
    )
    p.reporter.subscribe(on_status)
    return p


def demo_sorting():
    banner("演示 1：分拣模式（按类别放入库区）")
    items = [
        [Detection("resistor", 0.93, 300, 240)],     # -> 电子元器件区
        [Detection("wrench", 0.88, 320, 260)],       # -> 工具区
        [Detection("clamp", 0.81, 310, 250)],        # -> 电力金具区
    ]
    p = build(items[0])
    for batch in items:
        p.detector = MockDetector(batch)
        print(f"\n-- 识别到: {[d.name for d in batch]}")
        p.process_once(frame=None, mode=Mode.SORTING)
    return p


def demo_stacking():
    banner("演示 2：堆叠模式（连续逐层堆叠）")
    p = build([Detection("red_block", 0.90, 300, 240)])
    max_layers = p.stacking_cfg.get("max_layers", 4)
    for i in range(max_layers):
        print(f"\n-- 第 {i + 1} 次抓取")
        p.process_once(frame=None, mode=Mode.STACKING)
    return p


def print_summary(p, title):
    banner(f"统计：{title}")
    print("  库房计数:", dict(p.inventory.counts), "| 总计:", p.inventory.total())
    print("  操作日志:")
    for r in p.inventory.logs:
        print(f"    {r.timestamp}  {r.category:12s} {r.result:8s} "
              f"-> {r.zone:12s} {r.note}")


if __name__ == "__main__":
    p1 = demo_sorting()
    print_summary(p1, "分拣")

    p2 = demo_stacking()
    print_summary(p2, "堆叠")

    print("\n演示完成。以上动作在真机上由 ArmDriver 串口执行，逻辑完全一致。")
