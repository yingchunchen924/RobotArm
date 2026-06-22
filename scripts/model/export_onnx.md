# 导出 ONNX 与转换 .om（开发计划阶段五）

训练得到 `best.pt` 后，导出 ONNX（PC 上做），再在 **Atlas 开发板**上用 ATC 工具
转成 `.om`。

## 1. 导出 ONNX（PC）

用官方 yolov5 的 `export.py`：

```bash
cd /path/to/yolov5
python export.py \
  --weights runs/train/power_objects/weights/best.pt \
  --include onnx \
  --img 640 \
  --opset 11 \
  --simplify
# 产出 best.onnx
```

要点：
- `--img 640` 必须与训练、与本项目 `detect_postprocess.preprocess` 的 img_size 一致。
- `--opset 11` 兼容性较好；ATC 也建议较低 opset。
- 导出后可在 PC 上用 `scripts/model/infer_test.py --backend onnx` 验证后处理正确。

把 `best.onnx` 放到 `models/power_objects.onnx`。

## 2. 转换 .om（Atlas 开发板，需 ATC 工具）

ATC（Ascend Tensor Compiler）随 CANN 工具包，仅在昇腾环境可用。
脚本模板见 `scripts/model/convert_om.sh`。核心命令：

```bash
atc \
  --model=power_objects.onnx \
  --framework=5 \                 # 5 = ONNX
  --output=power_objects \        # 产出 power_objects.om
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --soc_version=Ascend310B1        # 200I DK A2 的 soc，按实际 npu-smi 查询确认
```

要点：
- `--input_shape` 的输入名（`images`）须与 ONNX 实际输入名一致（导出时默认 images）。
- `--soc_version` 用 `npu-smi info` 查询实际芯片型号填写。
- 产出 `power_objects.om` 放到开发板的 `models/`，供 `AclDetector` 加载。

## 3. 类别名文件

生成 `models/power_names.txt`（每行一个类别，顺序=class id），与
`config/categories.yaml` 一致。可直接：

```bash
python -c "import sys; sys.path.insert(0,'src'); from robotarm import dataset as ds; \
open('models/power_names.txt','w',encoding='utf-8').write('\n'.join(ds.class_names()))"
```

> 注意：`detectors.AclDetector` 默认直接用 `dataset.class_names()` 取类别，
> 不强依赖 power_names.txt；该文件主要供其它原生样例脚本使用。

## 4. 验证

- PC：`python scripts/model/infer_test.py --backend onnx --model models/power_objects.onnx --image <图>`
- 真机：`python scripts/model/infer_test.py --backend acl --model models/power_objects.om --image <图>`

两者后处理逻辑相同，输出应一致（数值因精度略有差异属正常）。
