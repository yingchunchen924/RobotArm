"""Web 后端集成测试（开发计划阶段七）。

用 FastAPI TestClient 验证后台 runner 与 Web 共享状态：start -> 库存真实增长 ->
状态接口反映运行 -> stop -> 停止。AppState 用很短的帧间隔以便测试快速看到效果。

把 web/backend 加入 import 路径以加载 app。
"""

import os
import sys
import time

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "web", "backend"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as webapp           # noqa: E402
from app_state import STATE    # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    STATE.stop()
    STATE.reset()
    yield
    STATE.stop()
    STATE.reset()


def _client():
    return TestClient(webapp.app)


def test_status_endpoint_shape():
    c = _client()
    r = c.get("/api/status")
    assert r.status_code == 200
    j = r.json()
    for k in ("running", "arm_status", "state", "state_label", "mock_mode"):
        assert k in j
    assert j["running"] is False


def test_start_runs_and_inventory_grows():
    # 用很短的帧间隔，加速测试
    started = STATE.start(mock=True, interval=0.02)
    assert started is True
    assert STATE.is_running()

    # 等后台线程抓几次
    deadline = time.time() + 3.0
    while STATE.inventory.total() < 3 and time.time() < deadline:
        time.sleep(0.05)

    assert STATE.inventory.total() >= 3
    # 轮换检测器会铺到多个库区
    assert len(STATE.inventory.counts) >= 2

    STATE.stop()
    assert not STATE.is_running()


def test_start_stop_via_api():
    c = _client()
    r = c.post("/api/tasks/start")
    assert r.json()["running"] is True
    time.sleep(0.1)
    r2 = c.post("/api/tasks/stop")
    assert r2.json()["running"] is False


def test_double_start_is_noop():
    assert STATE.start(mock=True, interval=0.05) is True
    assert STATE.start(mock=True, interval=0.05) is False   # 已在运行
    STATE.stop()


def test_detections_reflect_runner():
    STATE.start(mock=True, interval=0.02)
    deadline = time.time() + 2.0
    while not STATE.detections_snapshot() and time.time() < deadline:
        time.sleep(0.05)
    dets = STATE.detections_snapshot()
    STATE.stop()
    assert len(dets) >= 1
    assert "zone" in dets[0] and "label" in dets[0]


def test_inventory_endpoint_has_labels_and_logs():
    STATE.start(mock=True, interval=0.02)
    time.sleep(0.5)
    STATE.stop()
    c = _client()
    inv = c.get("/api/inventory").json()
    assert "zone_labels" in inv
    assert "recent_logs" in inv
    assert inv["total"] >= 1


def test_reset_clears():
    STATE.start(mock=True, interval=0.02)
    time.sleep(0.4)
    STATE.stop()
    assert STATE.inventory.total() >= 1
    c = _client()
    c.post("/api/inventory/reset")
    assert STATE.inventory.total() == 0
