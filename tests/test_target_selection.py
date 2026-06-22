"""target_selection 测试。"""

from robotarm.target_selection import Detection, select_target


def test_empty_returns_none():
    res = select_target([])
    assert res.target is None
    assert "未识别" in res.reason


def test_picks_highest_confidence():
    dets = [
        Detection("resistor", 0.6, 100, 100),
        Detection("wrench", 0.95, 500, 400),
    ]
    res = select_target(dets, overlap_px=10)  # 二者相距远，不成簇
    assert res.target.name == "wrench"
    assert "置信度最高" in res.reason


def test_low_confidence_filtered():
    dets = [
        Detection("resistor", 0.3, 100, 100),
        Detection("wrench", 0.4, 500, 400),
    ]
    res = select_target(dets, min_conf=0.5)
    assert res.target is None
    assert len(res.skipped) == 2


def test_overlap_picks_nearest_to_arm():
    # 两个高置信目标位置接近(中心距<60)，应选离机械臂(320,480)更近者
    dets = [
        Detection("a", 0.90, 300, 200),   # 离 (320,480) 远
        Detection("b", 0.92, 320, 450),   # 离 (320,480) 近
    ]
    # 让两者成簇：把 overlap_px 调大
    res = select_target(dets, overlap_px=300)
    assert res.target.name == "b"
    assert "最近" in res.reason


def test_unreachable_skipped():
    dets = [
        Detection("far", 0.9, 100, 100),
        Detection("ok", 0.8, 320, 450),
    ]

    def reachable(d: Detection) -> bool:
        return d.name != "far"

    res = select_target(dets, reachable=reachable, overlap_px=10)
    assert res.target.name == "ok"
    assert any(d.name == "far" for d in res.skipped)


def test_all_unreachable():
    dets = [Detection("x", 0.9, 100, 100)]
    res = select_target(dets, reachable=lambda d: False)
    assert res.target is None
    assert "不可达" in res.reason
