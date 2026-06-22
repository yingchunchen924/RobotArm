# dofbot_info —— 机械臂接口功能包（ROS2，真机编译）

定义 ROS2 服务消息 `Kinemarics.srv`，供运动学服务器与客户端通信。
本仓库为占位结构，真实内容来自华为 `ros2_robot_arm` 样例，需在 Atlas 开发板上获取并编译。

## Kinemarics.srv 消息格式（手册 4.2）

请求（客户端 -> 服务器）：
```
float64 tar_x      # 机械臂基坐标系下目标坐标
float64 tar_y
float64 tar_z
float64 roll       # 目标姿态角
float64 pitch
float64 yaw
float64 cur_joint1 # 当前六关节（fk 模式用）
float64 cur_joint2
float64 cur_joint3
float64 cur_joint4
float64 cur_joint5
float64 cur_joint6
string  kin_name   # "fk"(正解) 或 "ik"(逆解)
---
# 响应（服务器 -> 客户端）
float64 joint1     # ik 模式：六个关节目标角
float64 joint2
float64 joint3
float64 joint4
float64 joint5
float64 joint6
float64 x          # fk 模式：到达坐标与姿态
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
```

- `ik` 逆解：无需填当前关节，返回各关节目标角。
- `fk` 正解：只需填当前六关节，返回坐标与姿态角。
