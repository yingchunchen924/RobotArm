"""runner 测试：连续分拣、防抖、start/stop、异常降级（全 Mock，无硬件）。"""

import itertools

from robotarm.interfaces import (
    Detector,
    MockArmDriver,
    MockDetector,
    MockKinematics,
)
from robotarm.runner import PowerSortingRunner
from robotarm.target_selection import Detection


def make_runner(detector, frames, **kw):
    """frames 是一个帧列表；frame_source 依次弹出，弹完返回 None。"""
    seq = iter(frames)

    def src():
        return next(seq, None)

    counter = itertools.count()
    return PowerSortingRunner(
        detector=detector,
        kinematics=MockKinematics(),
        arm=MockArmDriver(),
        frame_source=src,
        clock=lambda: f"t{next(counter)}",
        **kw,
    )


def test_continuous_sorting_grasps_each_frame():
    det = MockDetector([Detection("resistor", 0.92, 300, 240)])
    # 5 个非空帧
    r = make_runner(det, frames=[object()] * 5, skip_repeat=0)
    stats = r.run(max_frames=5)
    assert stats.frames == 5
    assert stats.grasped == 5
    # 全部 resistor -> electronic
    assert r.pipeline.inventory.counts.get("electronic") == 5


def test_empty_frame_counted_as_empty():
    det = MockDetector([Detection("resistor", 0.9, 300, 240)])
    # 混入 None 帧
    r = make_runner(det, frames=[object(), None, object()], skip_repeat=0)
    stats = r.run(max_frames=3)
    assert stats.frames == 3
    assert stats.empty >= 1
    assert stats.grasped == 2


def test_no_detection_counted_empty_not_failed():
    det = MockDetector([])   # 永远识别不到
    r = make_runner(det, frames=[object()] * 3)
    stats = r.run(max_frames=3)
    assert stats.grasped == 0
    assert stats.empty == 3
    assert stats.failed == 0


class _RaisingDetector(Detector):
    def detect(self, frame):
        raise RuntimeError("camera lost")


def test_exception_counted_failed_not_crash():
    r = make_runner(_RaisingDetector(), frames=[object()] * 3)
    stats = r.run(max_frames=3)        # 不应抛出
    assert stats.failed == 3
    assert stats.grasped == 0


def test_stop_halts_loop():
    det = MockDetector([Detection("resistor", 0.9, 300, 240)])
    r = make_runner(det, frames=[object()] * 100, skip_repeat=0)

    # should_continue 在抓到 2 次后停
    def cond():
        return r.stats.grasped < 2

    stats = r.run(should_continue=cond)
    assert stats.grasped == 2


def test_cooldown_set_after_grasp():
    det = MockDetector([Detection("resistor", 0.9, 300, 240)])
    r = make_runner(det, frames=[object()] * 5, skip_repeat=3)
    r.step()                            # 抓一次
    assert r._cooldown == 3             # 抓后进入冷却


def test_max_frames_limit():
    det = MockDetector([Detection("resistor", 0.9, 300, 240)])
    r = make_runner(det, frames=[object()] * 100, skip_repeat=0)
    stats = r.run(max_frames=4)
    assert stats.frames == 4
