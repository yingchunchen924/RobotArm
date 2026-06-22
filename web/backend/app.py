"""可视化电力库房管理系统 —— FastAPI 后端（开发计划第7节）。

接入真实运行状态：后台线程跑 PowerSortingRunner，本服务与之共享 AppState
（StatusReporter + Inventory + 最近检测）。start/stop 控制后台 runner，各接口返回
其真实状态。mock 模式下 runner 用 Mock 检测器/机械臂，前端能看到动态的状态流转与
库存增长；真机模式由开发板入口注入真实 detector/逆解/串口。

接口（开发计划第7节）：
    GET  /api/status        系统 + 抓取状态
    GET  /api/detections    最近识别结果
    POST /api/tasks/start   启动后台连续分拣
    POST /api/tasks/stop    停止
    POST /api/arm/home      机械臂回初始姿态
    GET  /api/inventory     库房统计 + 日志
    POST /api/inventory/reset 清空统计
    GET  /video             实时视频流（占位）

本机运行：
    cd web/backend && uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from contextlib import asynccontextmanager                       # noqa: E402

from fastapi import FastAPI                                        # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse          # noqa: E402

from robotarm import config_loader as cl                          # noqa: E402

from app_state import STATE                                       # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    STATE.stop()   # 服务关闭时停止后台 runner


app = FastAPI(title="电力库房机械臂管理系统", version="0.2.0", lifespan=lifespan)


def _mock_mode() -> bool:
    return bool(cl.get_web_config().get("mock_mode", True))


@app.get("/api/status")
def get_status():
    snap = STATE.status_snapshot()
    snap["mock_mode"] = _mock_mode()
    return snap


@app.get("/api/detections")
def get_detections():
    return {"detections": STATE.detections_snapshot()}


@app.post("/api/tasks/start")
def start_task():
    started = STATE.start(mock=_mock_mode(), interval=1.0)
    return {"ok": True, "running": STATE.is_running(), "started": started}


@app.post("/api/tasks/stop")
def stop_task():
    stopped = STATE.stop()
    return {"ok": True, "running": STATE.is_running(), "stopped": stopped}


@app.post("/api/arm/home")
def arm_home():
    STATE.home()
    return {"ok": True}


@app.get("/api/inventory")
def get_inventory():
    return JSONResponse(STATE.inventory_snapshot())


@app.post("/api/inventory/reset")
def reset_inventory():
    STATE.reset()
    return {"ok": True}


@app.get("/video")
def video():
    """实时视频流占位。真机：cv2 抓帧 + 推理叠加框 -> MJPEG。"""
    return JSONResponse({
        "stream": "mock",
        "note": "真机上此处返回 MJPEG 视频流；当前为占位。",
    })


@app.get("/", response_class=HTMLResponse)
def index():
    fe = os.path.join(_ROOT, "web", "frontend", "index.html")
    if os.path.isfile(fe):
        with open(fe, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>电力库房机械臂管理系统</h1><p>前端页面缺失。</p>")
