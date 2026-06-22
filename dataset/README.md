# 数据集目录（开发计划阶段四）

电力物品检测数据集，YOLO 格式。大文件不纳入版本管理。

```
dataset/
├── raw/<类别>/   # 采集脚本按类别存放的原始图片
├── labeled/      # 标注后的 images/ + labels/（YOLO 格式，LabelImg/Roboflow/CVAT 产出）
│   ├── images/
│   └── labels/
├── yolo/         # 划分脚本生成的 train/val/test（训练直接用）
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── test/{images,labels}/
└── yaml/
    └── power_objects.yaml   # 由脚本生成，YOLOv5 训练配置
```

## 类别（单一真相源）
**不在数据集脚本里手写类别**。类别顺序（YOLO class id）由 `config/categories.yaml`
的 `category_to_zone` 派生，统一从 `src/robotarm/dataset.py` 读取。改类别只改
categories.yaml 一处，重新生成 yaml 即可，避免数据集与系统类别对不上。

当前 15 类：电阻/电容/继电器/接线端子/开关模块 · 线夹/螺栓/垫片/挂环/绝缘子 ·
螺丝刀/扳手/钳子/卷尺/电工刀。

## 工具链（scripts/dataset/）

| 脚本 | 作用 |
| --- | --- |
| `collect_images.py` | OpenCV 调摄像头按类别采集图片到 `raw/<类别>/`（需摄像头） |
| `validate_labels.py` | 校验 YOLO 标注：class id/坐标越界、漏标、孤儿标注 |
| `split_dataset.py` | 按比例随机划分 train/val/test（固定种子可复现） |
| `gen_dataset_yaml.py` | 从类别表生成 `power_objects.yaml` |

## 典型流程

```bash
# 1. 采集（每类 100~300 张，不同角度/光照/位置）
python scripts/dataset/collect_images.py --class resistor
python scripts/dataset/collect_images.py --class wrench --auto --interval 0.5 --count 200

# 2. 用 LabelImg / Roboflow / CVAT 标注，导出 YOLO 格式到 dataset/labeled/{images,labels}

# 3. 校验标注
python scripts/dataset/validate_labels.py \
    --images dataset/labeled/images --labels dataset/labeled/labels

# 4. 划分
python scripts/dataset/split_dataset.py \
    --images dataset/labeled/images --labels dataset/labeled/labels \
    --out dataset/yolo --train 0.7 --val 0.2 --test 0.1 --seed 42

# 5. 再次校验划分结果 + 生成训练配置
python scripts/dataset/validate_labels.py --root dataset/yolo
python scripts/dataset/gen_dataset_yaml.py --root dataset/yolo --out dataset/yaml/power_objects.yaml
```

## 验收（开发计划阶段四）
- 每个目标都有准确标注框 → `validate_labels.py` 无错误。
- 数据集能被 YOLO 训练脚本正常读取 → `power_objects.yaml` 的 nc/names 与标注一致。
