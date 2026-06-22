"""pipeline 端到端测试（全程 Mock，无硬件）。

覆盖：分拣成功流转、堆叠多层、各异常分支、状态历史、库存计数正确性。
"""

import itertools

import pytest

from robotarm.interfaces import (
    Detector,
    Kinematics,
    MockArmDriver,
    MockDetector,
    MockKinematics,
)
from robotarm.pipeline import GraspPipeline, Mode
from robotarm.states import GraspState
from robotarm.target_selection import Detection


def make_pipeline(detector=None, kinematics=None):
    counter = itertools.count()
    return GraspPipeline(
        detector=detector or MockDetector([
            Detection("resistor", 0.91, 300, 240),  # 居中、可达
        ]),
        kinematics=kinematics or MockKinematics(),
        arm=MockArmDriver(),
        clock=lambda: f"t{next(counter)}",
    )


def states_of(p):
    return [e.state for e in p.reporter.history]


# ---- 分拣 ----------------------------------------------------------------

def test_sorting_success_flow_and_inventory():
    p = make_pipeline()
    ok = p.process_once(frame=None, mode=Mode.SORTING)
    assert ok is True
    st = states_of(p)
    # 状态依次经过 等待识别->已识别->抓取中->放置中->完成
    assert GraspState.WAIT_DETECT in st
    assert GraspState.DETECTED in st
    assert GraspState.GRASPING in st
    assert GraspState.PLACING in st
    assert st[-1] == GraspState.DONE
    # resistor -> electronic 库区，计数 1，不重复计数
    assert p.inventory.counts.get("electronic") == 1
    assert p.inventory.total() == 1
    assert len(p.inventory.logs) == 1
    assert p.inventory.logs[0].result == "success"


def test_sorting_arm_executed_actions():
    p = make_pipeline()
    p.process_once(frame=None, mode=Mode.SORTING)
    # 机械臂确有动作（夹紧角应为 resistor 的覆盖值 130）
    joined = " | ".join(p.arm.history)
    assert "set_gripper 130" in joined


# ---- 异常分支 ------------------------------------------------------------

def test_no_detection_fails():
    p = make_pipeline(detector=MockDetector([]))
    ok = p.process_once(frame=None)
    assert ok is False
    assert p.reporter.current.state == GraspState.FAILED
    assert "未识别" in p.reporter.current.detail


class _RaisingDetector(Detector):
    def detect(self, frame):
        raise RuntimeError("camera lost")


def test_detector_exception_fails_gracefully():
    p = make_pipeline(detector=_RaisingDetector())
    ok = p.process_once(frame=None)
    assert ok is False
    assert p.reporter.current.state == GraspState.FAILED
    assert "识别异常" in p.reporter.current.detail


class _NoneKinematics(Kinematics):
    def inverse(self, x, y, z=0.0):
        return []


def test_ik_empty_fails_and_logs():
    p = make_pipeline(kinematics=_NoneKinematics())
    ok = p.process_once(frame=None)
    assert ok is False
    assert p.reporter.current.state == GraspState.FAILED
    assert p.inventory.logs[-1].result == "failed"


class _RaisingKinematics(Kinematics):
    def inverse(self, x, y, z=0.0):
        raise ValueError("no solution")


def test_ik_exception_fails():
    p = make_pipeline(kinematics=_RaisingKinematics())
    ok = p.process_once(frame=None)
    assert ok is False
    assert "逆解失败" in p.reporter.current.detail


def test_unreachable_target_fails():
    # 当前坐标公式的输出范围恒落在默认可达框内，因此这里把可达范围收紧到一个
    # 不可能命中的窗口，验证 pipeline 的“不可达 -> 跳过/失败”分支。
    from robotarm.coordinate import ReachBounds

    det = MockDetector([Detection("resistor", 0.95, 300, 240)])
    p = make_pipeline(detector=det)
    p.bounds = ReachBounds(x_min=10.0, x_max=11.0, y_min=10.0, y_max=11.0)
    ok = p.process_once(frame=None)
    assert ok is False
    assert p.reporter.current.state == GraspState.FAILED
    assert "不可达" in p.reporter.current.detail


# ---- 堆叠 ----------------------------------------------------------------

def test_stacking_multiple_layers():
    p = make_pipeline()
    max_layers = p.stacking_cfg.get("max_layers", 4)
    done = 0
    for _ in range(max_layers):
        if p.process_once(frame=None, mode=Mode.STACKING):
            done += 1
    assert done == max_layers
    assert p.stack_level == max_layers
    # 库存里 stacking 计数 == 层数
    assert p.inventory.counts.get("stacking") == max_layers


def test_stacking_exceeds_max_fails():
    p = make_pipeline()
    max_layers = p.stacking_cfg.get("max_layers", 4)
    for _ in range(max_layers):
        p.process_once(frame=None, mode=Mode.STACKING)
    # 再来一次应失败（超过最大层）
    ok = p.process_once(frame=None, mode=Mode.STACKING)
    assert ok is False
    assert "最大层数" in p.reporter.current.detail


def test_reset_stack():
    p = make_pipeline()
    p.process_once(frame=None, mode=Mode.STACKING)
    assert p.stack_level == 1
    p.reset_stack()
    assert p.stack_level == 0
