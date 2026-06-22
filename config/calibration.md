# 相机标定与坐标转换说明

本目录下三个标定文件由**首次使用时在 Atlas 开发板上**通过相机校准生成，
本仓库不包含它们（每台机械臂硬件不同，需各自标定）。来源见手册第 3.3 节
《（可选）校准摄像头》与附录 A。

## 三个文件

| 文件 | 作用 | 生成/修改方式 |
| --- | --- | --- |
| `dp.bin` | 透视变换矩阵，4×2 的 int32，存十字标定框四个角点 | 相机校准.ipynb 自动生成 |
| `XYT_config.txt` | 相机标定配置（标定时机械臂 joint1/joint2 等） | 相机校准.ipynb 自动生成 |
| `offset.txt` | 机械臂硬件误差补偿，两个浮点数 | 相机校准后手工微调（见下） |

## 生成步骤（开发板上）

1. 新终端进入 jupyter notebook：
   ```bash
   jupyter notebook --allow-root <开发板IP>
   ```
2. 进入 `ros2_robot_arm/.../dofbot_garbage_yolov5/tools`，打开 `相机校准.ipynb`。
3. Kernel → Restart & Run All。
4. 点击 `calibration_model`，调节 `joint1`、`joint2`，使**蓝色边框覆盖整个十字标定框**
   （保证光源充足）。
5. 点击 `calibration_ok`，生成/更新标定配置。
6. Ctrl+C 停止 notebook。

## offset.txt 手工校准（附录 A）

相机校准后抓取仍可能有误差，因机械臂硬件规格差异。修改 `offset.txt` 两个浮点参数：

- 抓取**偏后**（落在色块后方）→ **加大**参数（如 0.008 → 0.01）
- 抓取**偏前**（落在色块前方）→ **减小**参数（如 0.008 → 0.006）

> 在本项目代码中，offset 已抽象为坐标转换的入参（见 `src/robotarm/coordinate.py`
> 的 `apply_offset`），对应手册 `request.tar_y = posxy[1] + self.offset`。

## 配置共享

标定完成后，把 `dp.bin`、`XYT_config.txt`、`offset.txt` 复制到**分拣**与**堆叠**
两个功能包各自的 config 目录，使二者共享同一标定（附录 A 步骤2）。
