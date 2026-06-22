# 模型目录（开发计划阶段五）

存放识别模型与类别名文件。`.onnx` / `.om` / `.pt` 等大文件不纳入版本管理。

| 文件 | 说明 | 来源 |
| --- | --- | --- |
| `yolov5s_bs1.om` | 色块识别模型 | 华为样例自带 |
| `coco_names.txt` | 色块/样例类别名 | 华为样例自带 |
| `power_objects.onnx` | 电力物品识别模型（PC 推理用） | 训练后用 yolov5 export.py 导出 |
| `power_objects.om` | 电力物品识别模型（Atlas 真机用） | ATC 转换 onnx 得到 |
| `power_names.txt` | 电力物品类别名 | 由 categories.yaml 生成（顺序=class id） |

## 阶段五完整流程（脚本在 scripts/model/）

```
采集+标注+划分(阶段四)
   └─ dataset/yaml/power_objects.yaml
        │
        ▼ scripts/model/train_yolov5.py（封装官方 yolov5 train.py，需 GPU）
   best.pt
        │
        ▼ yolov5 export.py（见 scripts/model/export_onnx.md）
   power_objects.onnx ──┐
        │               ├─ PC 验证：scripts/model/infer_test.py --backend onnx
        ▼               │
   scripts/model/convert_om.sh（Atlas 上 ATC，见 export_onnx.md）
   power_objects.om ────┘
        │
        ▼ 真机验证：infer_test.py --backend acl
   接入 pipeline（detectors.AclDetector）
```

## 推理后端（src/robotarm/detectors.py）

| 后端 | 类 | 运行位置 | 依赖 |
| --- | --- | --- | --- |
| ONNX | `OnnxDetector` | PC / 开发板 | onnxruntime |
| Ascend | `AclDetector` | 仅 Atlas NPU | ais_bench |

两者**共享** `detect_postprocess.py` 的 letterbox 预处理 + NMS + 坐标还原，
因此 PC 上验证过的后处理逻辑搬到真机一字不改。类别 id→名 用 `dataset.id_to_name()`，
与 categories.yaml 一致。

## 生成 power_names.txt

```bash
python -c "import sys; sys.path.insert(0,'src'); from robotarm import dataset as ds; \
open('models/power_names.txt','w',encoding='utf-8').write('\n'.join(ds.class_names()))"
```

## 生成 power_objects.om（开发板，简述）
1. YOLOv5 训练（train_yolov5.py）→ best.pt
2. 导出 ONNX（export_onnx.md）
3. ATC 转换（convert_om.sh）→ power_objects.om
4. 上传开发板，`AclDetector` 加载，静态图 + 实时画面测试
