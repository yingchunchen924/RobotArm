# power_arm_control —— 机械臂控制与抓取流程功能包（ROS2，真机）

本项目**新增**功能包，封装抓取/分拣/堆叠的执行流程，对应开发计划阶段三/六。
占位结构，真机上实现。

## 抓取流程（手册 4.5 抽象）
1. 移动到目标上方过渡姿态（`config/arm.yaml` ready_pose）。
2. 松开夹爪（gripper.open）。
3. 下降到目标位置（逆解关节角）。
4. 夹紧夹爪（按物体 `grasp.yaml` 的 gripper_grasp）。
5. 抬起。
6. 移动到分类区/堆叠区（`grasp.yaml` place_zones / `stacking.yaml` layers）。
7. 释放夹爪，完成放置。

## 状态输出（开发计划阶段三）
等待识别 / 已识别 / 抓取中 / 放置中 / 完成 / 失败 —— 写入共享状态供 Web 展示。

## 异常处理（开发计划阶段六）
未识别到目标 / 逆解失败 / 抓取后滑落 / 摄像头读取失败 —— 各自降级与提示。

## 串口接口（手册）
机械臂控制基于 `Arm_serial_servo_write6_array(joints, ms)` 与
`Arm_serial_servo_write(6, angle, ms)`，对应本仓库 `interfaces.ArmDriver`。
真机驱动安装见 `scripts/setup_env.sh`（手册 3.2 步骤6：`0.py_install/setup.py`）。

## 真机需实现的两个类（被 scripts/run_power_sorting.py 引用）

| 文件 | 类 | 继承 | 实现内容 |
| --- | --- | --- | --- |
| `power_arm_control/serial_arm.py` | `SerialArmDriver` | `robotarm.interfaces.ArmDriver` | 在 `move_joints`/`set_gripper`/`home` 里调机械臂串口（手册 Arm_serial_servo_write*） |
| `power_arm_control/ros2_kinematics.py` | `Ros2Kinematics` | `robotarm.interfaces.Kinematics` | 在 `inverse(x,y)` 里调 ROS2 `Kinemarics` 服务 ik 模式（手册 server_joint），含逆解越界修正 |

实现后，`run_power_sorting.py --backend acl`（去掉 --mock）即可真机连续分拣。
上层 `runner.py` / `pipeline.py` 已在 PC 上验证，无需改动。
