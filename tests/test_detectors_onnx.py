"""OnnxDetector 端到端测试。

构造一个最小 YOLOv5 形状的合成 ONNX（常量输出），验证 OnnxDetector 的
预处理->推理->后处理->类别映射->坐标还原 全链路。

若环境无 onnxruntime / onnx（如本机 DLL 不兼容），自动跳过。
"""

import os
import tempfile

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from robotarm.detectors import OnnxDetector  # noqa: E402


def _make_synthetic_onnx(path: str, nc: int = 15):
    det = np.zeros((1, 2, 5 + nc), dtype=np.float32)
    det[0, 0, :5] = [320, 320, 40, 40, 0.9]   # 高置信 cls0
    det[0, 0, 5 + 0] = 0.9
    det[0, 1, :5] = [100, 100, 20, 20, 0.1]   # 低置信 cls5（应被过滤）
    det[0, 1, 5 + 5] = 0.1

    const = helper.make_node(
        "Constant", [], ["raw"],
        value=helper.make_tensor("v", TensorProto.FLOAT, det.shape, det.flatten()))
    ident = helper.make_node("Identity", ["raw"], ["output"])
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 5 + nc])
    graph = helper.make_graph([const, ident], "synthetic_yolo", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def test_onnx_detector_end_to_end():
    path = os.path.join(tempfile.gettempdir(), "robotarm_test_yolo.onnx")
    _make_synthetic_onnx(path)
    det = OnnxDetector(path, img_size=640, conf_thres=0.25)
    results = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(results) == 1               # 低置信框被过滤
    r = results[0]
    assert r.name == "resistor"            # cls 0 -> categories.yaml 第 0 类
    assert r.conf > 0.8
    assert abs(r.cx - 320) < 2
    assert abs(r.cy - 240) < 2             # letterbox 还原正确


def test_onnx_detector_none_frame():
    path = os.path.join(tempfile.gettempdir(), "robotarm_test_yolo.onnx")
    _make_synthetic_onnx(path)
    det = OnnxDetector(path)
    assert det.detect(None) == []
