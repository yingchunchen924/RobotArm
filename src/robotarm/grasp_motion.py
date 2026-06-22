"""抓取动作序列封装。

把手册 ``garbage_grap.py`` 的 ``move()`` 动作流程抽象为一个与硬件解耦的动作编排器。
真机上由 ``ArmDriver`` 执行具体串口写入；PC 上由 MockArmDriver 打印，逻辑完全一致。

手册动作序列（综合分拣 4.4 与堆叠 4.5 的 move()）：
    1. 移动到「架起/过渡」姿态 ready_pose（物体上方）
    2. 松开夹爪 (open)
    3. 移动到物体位置 grasp_joints（逆解 + 下降）
    4. 夹紧夹爪 (grasp)
    5. 抬回过渡姿态
    6. 先转 joint1 对准放置区，再移动到放置姿态 place_joints
    7. 松开夹爪，释放物体
    8. 抬起 lift_pose

同时实现手册 ``server_joint`` 里的**逆解越界修正**：
    if joints[2] < 0: joints[1] += joints[2]*3/5; joints[3] += joints[2]*3/5; joints[2] = 0
"""

from __future__ import annotations

from typing import List, Sequence

from .interfaces import ArmDriver


def fix_ik_overflow(joints: Sequence[float]) -> List[float]:
    """逆解越界修正（手册 server_joint）。

    当 joint3 (索引2) 为负，把超出的量按 3/5 分摊给 joint2、joint4，并夹到 0。
    """
    j = list(joints)
    if len(j) >= 4 and j[2] < 0:
        j[1] += j[2] * 3 / 5
        j[3] += j[2] * 3 / 5
        j[2] = 0
    return j


def compose_grasp_joints(
    ik_joints: Sequence[float],
    wrist: float = 265.0,
    gripper_open: float = 30.0,
) -> List[float]:
    """把逆解得到的关节角拼成完整 6 关节抓取姿态。

    对应手册 ``joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]``：
    逆解只给前 4 个有效关节，腕部固定 265，夹爪此刻保持张开。
    """
    j = list(ik_joints)
    while len(j) < 4:
        j.append(0.0)
    return [j[0], j[1], j[2], j[3], wrist, gripper_open]


class GraspMotion:
    """抓取动作编排器。一次「抓取并放置」对应 ``pick_and_place``。

    所有姿态、夹爪角、速度均由调用方（pipeline）从配置取出后传入，本类不读配置，
    保持纯动作逻辑。
    """

    def __init__(self, arm: ArmDriver) -> None:
        self.arm = arm

    def pick_and_place(
        self,
        grasp_joints: Sequence[float],
        place_joints: Sequence[float],
        ready_pose: Sequence[float],
        lift_pose: Sequence[float],
        gripper_open: float,
        gripper_grasp: float,
        speed_normal: int = 1000,
        speed_fast: int = 500,
    ) -> None:
        """完整的抓取 -> 放置动作序列（手册 move()）。

        :param grasp_joints: 物体位置的 6 关节角（已含腕部/夹爪占位）
        :param place_joints: 放置区的 6 关节角
        :param ready_pose:   过渡/架起姿态
        :param lift_pose:    结束抬起姿态
        """
        arm = self.arm
        # 1. 架起到过渡姿态
        arm.move_joints(ready_pose, speed_normal)
        # 2. 松开夹爪
        arm.set_gripper(gripper_open, speed_fast)
        # 3. 下降到物体位置
        arm.move_joints(grasp_joints, speed_fast)
        # 4. 夹紧
        arm.set_gripper(gripper_grasp, speed_fast)
        # 5. 抬回过渡姿态
        arm.move_joints(ready_pose, speed_normal)
        # 6a. 先转 joint1 对准放置区（手册：单独先转 joint1，避免扫到其它物体）
        turn = list(ready_pose)
        turn[0] = place_joints[0]
        arm.move_joints(turn, speed_normal)
        # 6b. 移动到放置姿态
        arm.move_joints(place_joints, speed_normal)
        # 7. 松开夹爪，释放
        arm.set_gripper(gripper_open, speed_fast)
        # 8. 抬起
        arm.move_joints(lift_pose, speed_normal)
