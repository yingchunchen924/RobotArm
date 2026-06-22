# dofbot_moveit —— 机械臂描述与运动学服务功能包（ROS2，真机编译）

提供机械臂 URDF 三维模型、正逆运动学解算服务器、RViz 可视化。占位结构，
真实内容来自华为样例，需在 Atlas 开发板上编译。

## 关键内容（手册 4.3）
- `urdf/dofbot.urdf` —— 机械臂三维结构描述。
- `src/dofbot_server.cpp` —— 正逆运动学服务器，回调 `srvicecallback()`：
  - `kin_name == "fk"` → 调 `dofbot.dofbot_getFK`，返回坐标姿态。
  - `kin_name == "ik"` → 逆解，返回关节角。
  - 角度↔弧度换算：`(joint - 90) * DE2RA`。
- `lib/libdofbot_kinemarics.so` —— 运动学求解库（ARM 预编译）。
  部署时需 `cp libdofbot_kinemarics.so /usr/lib`（手册 3.2 步骤4）。

## 启动（开发板）
```bash
source setenv.sh
ros2 run dofbot_moveit dofbot_server
```
