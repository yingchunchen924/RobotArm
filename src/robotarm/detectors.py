"""目标检测后端实现：OnnxDetector（PC）与 AclDetector（Atlas 真机）。

两者都实现 ``interfaces.Detector``，返回 ``target_selection.Detection`` 列表，
并**共享** ``detect_postprocess`` 的前后处理。区别仅在于「怎么把输入张量喂给模型、
拿回输出张量」：

    OnnxDetector  -> onnxruntime 加载 .onnx（PC/开发板都可，本机验证后处理用）
    AclDetector   -> ais_bench InferSession 加载 .om（仅 Atlas NPU）

类别 id -> 名称 用 ``dataset.id_to_name()``，与 categories.yaml 单一真相源一致。
重型依赖（onnxruntime / ais_bench）延迟到构造时才 import，使本模块可被无依赖环境导入。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import dataset as ds
from . import detect_postprocess as pp
from .interfaces import Detector
from .target_selection import Detection


class _YoloDetectorBase(Detector):
    """共享前后处理的 YOLOv5 检测器基类。子类只需实现 ``_infer``。"""

    def __init__(
        self,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        names: Optional[List[str]] = None,
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = names if names is not None else ds.class_names()

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        """子类实现：输入 (1,3,H,W) float32，返回 (N,5+nc) 或 (1,N,5+nc)。"""
        raise NotImplementedError

    def detect(self, frame) -> List[Detection]:
        if frame is None:
            return []
        img = np.asarray(frame)
        blob, info = pp.preprocess(img, self.img_size)
        output = self._infer(blob)
        raws = pp.postprocess(output, info, self.conf_thres, self.iou_thres)
        return [self._to_detection(r) for r in raws]

    def _to_detection(self, r: pp.RawDetection) -> Detection:
        name = self.names[r.class_id] if 0 <= r.class_id < len(self.names) \
            else str(r.class_id)
        return Detection(name=name, conf=r.conf, cx=r.cx, cy=r.cy)


class OnnxDetector(_YoloDetectorBase):
    """onnxruntime 后端（PC 可跑，用于本机验证整条识别->后处理链路）。"""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None, **kw):
        super().__init__(**kw)
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要 onnxruntime：pip install onnxruntime") from e
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: blob})
        return out[0]


class AclDetector(_YoloDetectorBase):
    """ais_bench / aclruntime 后端（仅 Atlas NPU，加载 .om 模型）。

    真机部署时使用。接口与 OnnxDetector 完全一致，pipeline 无需改动。
    """

    def __init__(self, model_path: str, device_id: int = 0, **kw):
        super().__init__(**kw)
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as e:  # pragma: no cover - 仅真机有
            raise ImportError(
                "需要 ais_bench（仅 Atlas 开发板）：见 scripts/setup_env.sh") from e
        self.session = InferSession(device_id, model_path)

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        outputs = self.session.infer([blob])
        return outputs[0]


def build_detector(backend: str, model_path: str, **kw) -> Detector:
    """工厂：按后端名构造检测器。

    :param backend: "onnx" | "acl"
    """
    backend = backend.lower()
    if backend == "onnx":
        return OnnxDetector(model_path, **kw)
    if backend in ("acl", "om", "ais_bench"):
        return AclDetector(model_path, **kw)
    raise ValueError(f"未知后端: {backend}（应为 onnx 或 acl）")
