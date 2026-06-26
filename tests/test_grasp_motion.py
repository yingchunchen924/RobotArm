"""grasp_motion 测试。"""

from robotarm.grasp_motion import (
    GraspMotion,
    compose_grasp_joints,
    fix_ik_overflow,
)
from robotarm.interfaces import MockArmDriver


def test_fix_ik_overflow_negative_j3():
    # joint3 < 0 时按 3/5 分摊给 j2/j4 并夹到 0
    j = fix_ik_overflow([90, 50, -10, 40])
    assert j[2] == 0
    assert j[1] == 50 + (-10) * 3 / 5
    assert j[3] == 40 + (-10) * 3 / 5


def test_fix_ik_overflow_noop_when_positive():
    j = fix_ik_overflow([90, 50, 20, 40])
    assert j == [90, 50, 20, 40]


def test_compose_grasp_joints_pads_and_appends_wrist_gripper():
    j = compose_grasp_joints([10, 20, 30, 40])
    assert j == [10, 20, 30, 40, 265.0, 30.0]
    # 不足 4 个时补零
    j2 = compose_grasp_joints([10, 20])
    assert j2[:4] == [10, 20, 0.0, 0.0]


def test_pick_and_place_action_order():
    arm = MockArmDriver()
    GraspMotion(arm).pick_and_place(
        grasp_joints=[10, 20, 30, 40, 265, 30],
        place_joints=[135, 50, 20, 60, 265, 100],
        ready_pose=[90, 80, 50, 50, 265, 30],
        lift_pose=[90, 80, 50, 50, 265, 30],
        gripper_open=0,
        gripper_grasp=100,
    )
    h = arm.history
    # 至少包含：架起、松爪(open=0)、下降、夹紧(100)、抬起、转j1、放置、松爪、抬起
    joined = " | ".join(h)
    assert "set_gripper 0" in joined          # 松开
    assert "set_gripper 100" in joined        # 夹紧
    # 夹紧必须在松开之后、放置之前
    open_idxs = [i for i, m in enumerate(h) if "set_gripper 0" in m]
    grasp_idx = next(i for i, m in enumerate(h) if "set_gripper 100" in m)
    assert open_idxs[0] < grasp_idx < open_idxs[-1]


def test_pick_and_place_keeps_gripper_closed_while_carrying():
    arm = MockArmDriver()
    GraspMotion(arm).pick_and_place(
        grasp_joints=[10, 20, 30, 40, 265, 0],
        place_joints=[135, 50, 20, 60, 265, 0],
        ready_pose=[90, 80, 50, 50, 265, 0],
        lift_pose=[135, 80, 50, 50, 265, 0],
        gripper_open=0,
        gripper_grasp=100,
    )

    grasp_idx = next(i for i, m in enumerate(arm.history) if "set_gripper 100" in m)
    release_idx = next(
        i for i, m in enumerate(arm.history[grasp_idx + 1 :], start=grasp_idx + 1)
        if "set_gripper 0" in m
    )
    carrying_moves = [
        m for m in arm.history[grasp_idx + 1 : release_idx]
        if "move_joints" in m
    ]
    assert carrying_moves
    assert all(m.rstrip().endswith("100] (1000ms)") for m in carrying_moves)
