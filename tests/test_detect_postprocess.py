"""detect_postprocess 测试：letterbox / NMS / 坐标还原 / 完整后处理。

用构造的合成张量，验证几何与过滤逻辑正确（这部分逻辑 PC 与真机共用）。
"""

import numpy as np

from robotarm import detect_postprocess as pp


# ---- letterbox ----

def test_letterbox_shape_and_info():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, info = pp.letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    # 640x480 -> ratio 1.0，上下各填充 (640-480)/2=80
    assert abs(info.ratio - 1.0) < 1e-6
    assert info.new_w == 640 and info.new_h == 480
    assert abs(info.pad_h - 80) <= 1
    assert info.pad_w == 0


def test_letterbox_roundtrip_coordinates():
    # 原图正中心一点，经 letterbox 再 scale_boxes 应还原回原坐标
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    _, info = pp.letterbox(img, 640)
    # 网络输入坐标系下：原图中心 (300,200) 映射后的位置
    net_x = 300 * info.ratio + info.pad_w
    net_y = 200 * info.ratio + info.pad_h
    box = np.array([[net_x - 5, net_y - 5, net_x + 5, net_y + 5]], dtype=np.float32)
    back = pp.scale_boxes(box, info)
    cx = (back[0, 0] + back[0, 2]) / 2
    cy = (back[0, 1] + back[0, 3]) / 2
    assert abs(cx - 300) < 1.0
    assert abs(cy - 200) < 1.0


# ---- xywh2xyxy ----

def test_xywh2xyxy():
    boxes = np.array([[100, 100, 40, 20]], dtype=np.float32)
    out = pp.xywh2xyxy(boxes)
    assert list(out[0]) == [80, 90, 120, 110]


# ---- NMS ----

def test_nms_suppresses_overlap():
    # 两个几乎重合的框 + 一个远处框 -> 保留 2 个
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],     # 与第一个高度重叠
        [200, 200, 240, 240],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    keep = pp.nms(boxes, scores, iou_thres=0.45)
    assert 0 in keep           # 最高分保留
    assert 1 not in keep       # 重叠被抑制
    assert 2 in keep           # 远处框保留
    assert len(keep) == 2


def test_nms_empty():
    assert pp.nms(np.zeros((0, 4)), np.zeros((0,))) == []


# ---- postprocess 完整链路 ----

def _make_pred(cx, cy, w, h, nc, cls, obj=0.9, cls_conf=0.9):
    row = np.zeros(5 + nc, dtype=np.float32)
    row[:4] = [cx, cy, w, h]
    row[4] = obj
    row[5 + cls] = cls_conf
    return row


def test_postprocess_filters_and_decodes():
    nc = 15
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, info = pp.letterbox(img, 640)
    # 两个目标：一个高置信 cls=0，一个低置信(应被过滤) cls=5
    high = _make_pred(320, 320, 40, 40, nc, cls=0, obj=0.9, cls_conf=0.9)   # conf 0.81
    low = _make_pred(100, 100, 20, 20, nc, cls=5, obj=0.2, cls_conf=0.2)    # conf 0.04
    output = np.stack([high, low])[None, ...]   # (1,N,5+nc)

    dets = pp.postprocess(output, info, conf_thres=0.25, iou_thres=0.45)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 0
    assert d.conf > 0.8
    # 网络坐标 (320,320) 在 640 画布上，pad_h=80 -> 原图 y≈240，x≈320
    assert abs(d.cx - 320) < 2
    assert abs(d.cy - 240) < 2


def test_postprocess_empty_output():
    nc = 15
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, info = pp.letterbox(img, 640)
    empty = np.zeros((0, 5 + nc), dtype=np.float32)
    assert pp.postprocess(empty, info) == []


def test_postprocess_all_below_threshold():
    nc = 15
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, info = pp.letterbox(img, 640)
    weak = _make_pred(320, 320, 40, 40, nc, cls=1, obj=0.1, cls_conf=0.1)
    out = weak[None, None, :]
    assert pp.postprocess(out, info, conf_thres=0.25) == []
