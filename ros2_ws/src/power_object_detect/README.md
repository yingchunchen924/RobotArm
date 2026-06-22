# power_object_detect —— 电力物品识别抓取功能包（ROS2，真机）

本项目**新增**功能包，对应开发计划阶段五/六：电力元器件、金具、工具的识别与分类抓取。
在华为 `dofbot_garbage_yolov5` 基础上改造而来。占位结构，真机上实现。

## 职责
1. 摄像头抓帧 → 缩放 640×480 → `dp.bin` 透视变换。
2. 加载 `power_objects.om` 模型推理（ais_bench/aclruntime），输出类别/置信度/中心点。
3. 调用本仓库 `src/robotarm` 的纯逻辑：
   - `coordinate.pixel_to_arm` 像素→机械臂坐标 + `apply_offset` 补偿。
   - `target_selection.select_target` 选目标。
   - `config_loader.category_to_zone` 决定库区，`grasp.yaml` 取夹爪角与放置姿态。
4. 调 `Kinemarics` 服务逆解 → `ArmDriver` 执行抓取移动。
5. 更新 `inventory`，推送状态给 Web。

## 与纯逻辑层的关系
本包是真机「胶水层」：把硬件 I/O（摄像头/NPU/串口/ROS2）接到 `src/robotarm`
的纯逻辑上。逻辑已在 PC 上单测通过，真机只需保证接口实现正确。

## 模型与类别
- 模型：`models/power_objects.om`（阶段五训练转换）。
- 类别名：`models/power_names.txt`。
- 类别→库区映射：`config/categories.yaml`。
