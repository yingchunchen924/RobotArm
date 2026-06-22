"""YOLOv5 推理的预处理 / 后处理纯逻辑（numpy）。

把 letterbox 预处理、解码、NMS、坐标还原从具体推理后端（onnxruntime / ais_bench）
中抽出来，使 PC 上的 OnnxDetector 与真机的 AclDetector **共享同一套前后处理**——
这样在 PC 上验证过的后处理逻辑，搬到真机一字不改。

约定：YOLOv5 输出张量形状 (num_boxes, 5+nc)，每行 = [cx, cy, w, h, obj, cls_0..cls_n]，
坐标是在「letterbox 后的网络输入尺寸」(默认 640) 下的像素值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ---- 预处理：letterbox ----------------------------------------------------

@dataclass
class LetterboxInfo:
    """letterbox 变换参数，用于把检测框还原回原图坐标。"""

    ratio: float          # 缩放比例
    pad_w: float          # 左右填充（单边）
    pad_h: float          # 上下填充（单边）
    new_w: int            # 缩放后未填充的宽
    new_h: int


def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxInfo]:
    """等比缩放并填充到 new_shape×new_shape（YOLOv5 标准预处理）。

    :param img: HxWx3 BGR 图像
    :returns: (填充后图像, letterbox 参数)
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    # 用 numpy 实现缩放以避免硬依赖 cv2（cv2 可用时由调用方替换更快）
    resized = _resize(img, new_w, new_h)

    pad_w = (new_shape - new_w) / 2
    pad_h = (new_shape - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    out = np.full((new_shape, new_shape, 3), color, dtype=img.dtype)
    out[top:top + new_h, left:left + new_w] = resized
    return out, LetterboxInfo(r, left, top, new_w, new_h)


def _resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """最近邻缩放（纯 numpy，无 cv2 依赖）。生产可换 cv2.resize 提速。"""
    try:
        import cv2
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        h, w = img.shape[:2]
        ys = (np.arange(new_h) * (h / new_h)).astype(int).clip(0, h - 1)
        xs = (np.arange(new_w) * (w / new_w)).astype(int).clip(0, w - 1)
        return img[ys][:, xs]


def preprocess(img: np.ndarray, new_shape: int = 640):
    """letterbox + BGR->RGB + HWC->CHW + 归一化，输出 (1,3,H,W) float32。"""
    lb, info = letterbox(img, new_shape)
    x = lb[:, :, ::-1]                       # BGR->RGB
    x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)
    return np.ascontiguousarray(x), info


# ---- 后处理 ---------------------------------------------------------------

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """中心点+宽高 -> 左上右下。boxes: (N,4)。"""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> List[int]:
    """单类 NMS，返回保留框的索引（按分数降序）。boxes 为 xyxy。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """把网络输入尺寸下的 xyxy 框还原到原图坐标。"""
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] -= info.pad_w
    out[:, [1, 3]] -= info.pad_h
    out /= info.ratio
    return out


@dataclass
class RawDetection:
    """后处理输出的一个框（原图像素 xyxy + 类别 id + 置信度）。"""

    class_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def postprocess(
    output: np.ndarray,
    info: LetterboxInfo,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[RawDetection]:
    """YOLOv5 原始输出 -> 还原到原图的检测框列表。

    :param output: (N, 5+nc) 或 (1, N, 5+nc)
    """
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] == 0:
        return []

    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    class_ids = cls_scores.argmax(axis=1)
    class_conf = cls_scores[np.arange(len(cls_scores)), class_ids]
    conf = obj * class_conf

    mask = conf >= conf_thres
    if not mask.any():
        return []
    pred = pred[mask]
    conf = conf[mask]
    class_ids = class_ids[mask]

    xyxy = xywh2xyxy(pred[:, :4])

    # 按类别分别 NMS
    results: List[RawDetection] = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        keep = nms(xyxy[idx], conf[idx], iou_thres)
        for k in keep:
            gi = idx[k]
            results.append((gi, int(c)))

    # 还原坐标
    if not results:
        return []
    gis = [g for g, _ in results]
    scaled = scale_boxes(xyxy[gis], info)
    out: List[RawDetection] = []
    for (gi, c), box in zip(results, scaled):
        out.append(RawDetection(
            class_id=c, conf=float(conf[gi]),
            x1=float(box[0]), y1=float(box[1]),
            x2=float(box[2]), y2=float(box[3]),
        ))
    # 按置信度降序
    out.sort(key=lambda d: d.conf, reverse=True)
    return out
