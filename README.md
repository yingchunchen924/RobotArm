# 电力库房智能识别与机械臂抓取系统

基于华为 **Atlas 200I DK A2** 智能机械臂的电力智能运维 / 库房管理系统
（实践题目 8）。在华为 `ros2_robot_arm` 样例基础上，扩展为面向电力场景的
识别、抓取、分类、堆叠与可视化管理系统。

> 详细设计见 [`docs/开发计划.md`](docs/开发计划.md)，硬件与样例说明见根目录《05机械臂使用手册.pdf》。

## 运行环境分层（重要）

本项目代码分两层，**别在错误的机器上跑**：

| 层 | 运行位置 | 内容 |
| --- | --- | --- |
| **纯逻辑层** `src/robotarm/` | 任意 PC | 坐标转换、目标筛选、配置加载、库房统计、抽象接口。无硬件依赖，**可在本机单测**。 |
| **真机层** `ros2_ws/` | Atlas 开发板 | ROS2 运动学服务、NPU 推理、机械臂串口控制。需 ARM/Ubuntu/ROS2/昇腾环境。 |
| **可视化层** `web/` | PC 或开发板 | FastAPI 后端 + 前端。mock 模式下可在 PC 上完整演示。 |

## 目录结构

```
RobotArm/
├── 05机械臂使用手册.pdf      # 华为官方手册
├── docs/                     # 开发计划、环境部署记录、测试记录
├── config/                   # ★ 配置体系：所有可变量集中于此
│   ├── arm.yaml              #   机械臂姿态/夹爪/速度/坐标转换参数/可达范围
│   ├── grasp.yaml            #   各物体夹爪角 + 库区放置姿态
│   ├── stacking.yaml         #   色块堆叠层位（取代手册硬编码）
│   ├── categories.yaml       #   类别 -> 库区 映射
│   ├── web.yaml              #   Web 端口/视频源/存储/mock 开关
│   └── calibration.md        #   dp.bin/XYT_config/offset 说明
├── src/robotarm/             # ★ 纯 Python 逻辑（本机可单测）
│   ├── config_loader.py      #   统一配置加载
│   ├── coordinate.py         #   像素 -> 机械臂坐标 + offset + 可达性
│   ├── target_selection.py   #   目标筛选策略
│   ├── inventory.py          #   库房统计 + 操作日志
│   ├── states.py             #   抓取状态机（等待识别/抓取中/完成/失败…）
│   ├── grasp_motion.py       #   抓取动作序列（手册 move() + 逆解越界修正）
│   ├── pipeline.py           #   ★ 抓取流程编排（分拣/堆叠两模式）
│   ├── dataset.py            #   数据集类别单一真相源 + 标注解析
│   ├── detect_postprocess.py #   YOLOv5 前后处理(letterbox/NMS/坐标还原) 纯numpy
│   ├── detectors.py          #   OnnxDetector(PC) + AclDetector(真机) 共享前后处理
│   ├── runner.py             #   ★ 连续分拣运行器（识别->抓取闭环，帧源可注入）
│   └── interfaces.py          #   ArmDriver/Kinematics/Detector 抽象 + Mock
├── ros2_ws/src/              # ROS2 功能包（真机编译，当前为占位 README）
│   ├── dofbot_info/          #   Kinemarics.srv 服务消息
│   ├── dofbot_moveit/        #   运动学服务 + URDF
│   ├── power_object_detect/  #   ★ 新增：电力物品识别抓取
│   └── power_arm_control/    #   ★ 新增：抓取/堆叠流程控制
├── web/                      # FastAPI 后端 + 前端页面
├── models/                   # .om 模型 + 类别名（占位）
├── dataset/                  # 数据集（占位）
├── scripts/                  # 环境配置 + 启动脚本
└── tests/                    # 单元测试
```

## 快速开始（本机）

```bash
# 1. 安装本机依赖
pip install -r requirements.txt

# 2. 运行单元测试（坐标转换 / 配置 / 目标筛选）
pytest tests/ -v

# 3. 启动可视化系统（mock 模式，浏览器演示）
bash scripts/start_dashboard.sh        # 或 cd web/backend && uvicorn app:app --reload
# 打开 http://127.0.0.1:8000/

# 4. 端到端演示抓取流程（分拣 + 堆叠，全程 Mock 无需硬件）
python scripts/demo_pipeline.py

# 5. 阶段六端到端：真实 ONNX 识别 -> 分类抓取（Mock 臂，需 onnxruntime）
python scripts/demo_power_sorting.py
```

## 设计约定

1. **配置驱动**：类别映射、抓取参数、堆叠层位、坐标参数等全部在 `config/*.yaml`，
   代码不写魔数。调参只改 yaml。
2. **逻辑与硬件分离**：抓取流程的纯逻辑（坐标/筛选/库存）独立于硬件，先在 PC 上
   验证正确，真机层只做「胶水」实现 `interfaces.py` 的抽象接口。
3. **真机部分留接口**：`interfaces.py` 提供 Mock 实现，使整套流程在无硬件时也能联调。

## 开发阶段（详见开发计划）

| 阶段 | 内容 | 本仓库支撑 |
| --- | --- | --- |
| 一 | 硬件部署 + 样例复现 | `scripts/setup_env.sh`、`docs/环境部署记录.md` |
| 二 | 标定 + 坐标转换 | `coordinate.py`、`config/calibration.md` |
| 三 | 色块识别/抓取/堆叠 | ✅ `pipeline.py`、`states.py`、`grasp_motion.py`、`config/stacking.yaml` |
| 四 | 电力物品数据集 | ✅ `dataset.py` + `scripts/dataset/`（采集/校验/划分/生成 yaml） |
| 五 | 识别模型(.om) | ✅ `detectors.py`+`detect_postprocess.py`+`scripts/model/`（训练/导出/转换/推理测试） |
| 六 | 电力物品分类抓取 | ✅ `runner.py`+`scripts/run_power_sorting.py`（识别->分类抓取闭环，onnx/acl 切换） |
| 七 | 可视化系统 | ✅ `web/`（FastAPI 接后台 runner，实时状态/库存/日志，start/stop 真控制） |
| 八 | 联调演示 | ✅ `scripts/start_all.sh`(真机) / `start_demo.sh`(PC) / `demo_full_system.py` + `docs/联调与演示.md` |

## 当前进度

阶段三~七的「PC 可验证部分」全部落地并通过测试（**65 项单测/集成测试全绿**），
阶段一/二真机脚本与文档就绪，阶段八联调编排完成。仅余需真机/实物的收尾
（硬件部署、相机标定、数据集采集与训练、实物抓取演示），详见
[`docs/联调与演示.md`](docs/联调与演示.md) 的验收清单核对表。

PC 上可直接复现：

```bash
pip install -r requirements.txt
pytest tests/ -q                      # 65 passed
python scripts/demo_full_system.py    # 分拣+堆叠+电力物品 端到端
python scripts/demo_power_sorting.py  # 真实 ONNX 识别 -> 抓取
bash   scripts/start_demo.sh          # Web 实时界面演示
```

## 说明（未用 git）

当前未初始化 git。以下为生成物/大文件，将来纳入版本管理时建议忽略：
`__pycache__/`、`.pytest_cache/`、`web/backend/data/`、`models/*.om`、
`dataset/raw|labeled/`、`config/dp.bin|XYT_config.txt|offset.txt`、
`docs/手册提取文本_manual_text.txt`。
