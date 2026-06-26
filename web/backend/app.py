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

import time                                                       # noqa: E402
from contextlib import asynccontextmanager                       # noqa: E402

from fastapi import FastAPI, Request                              # noqa: E402
from fastapi.responses import (                                    # noqa: E402
    HTMLResponse, JSONResponse, Response, StreamingResponse,
)

from robotarm import config_loader as cl                          # noqa: E402

from app_state import STATE, CAMERA, _REAL_OBJECTS                 # noqa: E402

try:
    import cv2                                                     # noqa: E402
except Exception:                                                  # pragma: no cover
    cv2 = None   # PC/无 OpenCV 环境下降级为占位


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
    # 全抓：固定顺序 red→yellow→blue→green（colors=None 即全抓）
    started = STATE.start(mock=_mock_mode(), interval=1.0)
    return {"ok": True, "running": STATE.is_running(), "started": started}


_VALID_COLORS = {"red", "yellow", "blue", "green"}


@app.post("/api/tasks/start_color")
def start_task_color(color: str):
    """只抓指定颜色一次。color ∈ {red,yellow,blue,green}。"""
    if color not in _VALID_COLORS:
        return JSONResponse({"ok": False, "error": f"无效颜色 {color}"},
                            status_code=400)
    started = STATE.start(mock=_mock_mode(), interval=1.0, colors=[color])
    return {"ok": True, "running": STATE.is_running(),
            "started": started, "color": color}


@app.post("/api/tasks/start_seq")
def start_task_seq(colors: str):
    """按自定义顺序连贯抓取。colors 为逗号分隔的颜色串，如 blue,red,green。"""
    seq = [c.strip() for c in colors.split(",") if c.strip()]
    bad = [c for c in seq if c not in _VALID_COLORS]
    if not seq or bad:
        return JSONResponse(
            {"ok": False, "error": f"无效颜色序列 {colors}"}, status_code=400)
    started = STATE.start(mock=_mock_mode(), interval=1.0, colors=seq)
    return {"ok": True, "running": STATE.is_running(),
            "started": started, "colors": seq}


@app.post("/api/tasks/stop")
def stop_task():
    stopped = STATE.stop()
    return {"ok": True, "running": STATE.is_running(), "stopped": stopped}


@app.post("/api/tasks/stack")
def stack_task():
    """颜色方块堆叠：红→黄→蓝→绿依次抓取码到中心十字（最多 4 层）。"""
    started = STATE.start_real_stack()
    return {"ok": True, "running": STATE.is_running(), "started": started}


@app.post("/api/voice")
async def voice_command(request: Request):
    """语音指令：前端把录音二进制 POST 到这里（Content-Type 任意音频类型）。
    后端转 wav -> mimo-v2.5-asr 识别 -> 关键词匹配 -> 触发机械臂动作。
    返回 {ok, text(识别文字), action, action_label}。
    """
    audio = await request.body()
    if not audio:
        return JSONResponse({"ok": False, "error": "空音频"}, status_code=400)
    mime = request.headers.get("content-type", "")
    result = STATE.voice_command(audio, mime)
    return JSONResponse(result)


# ---- 实物分类（点照片抓取）----
_VALID_OBJECTS = set(_REAL_OBJECTS)


@app.get("/api/photo")
def get_photo():
    """拍一张工作区静态照片（640x480 JPEG）供前端点击选位。忙时返回 409。"""
    data = STATE.capture_photo()
    if not data:
        return JSONResponse({"ok": False, "error": "忙或拍照失败"},
                            status_code=409)
    return Response(content=data, media_type="image/jpeg")


def _parse_pick_item(token: str):
    """'obj:cx,cy' -> (obj, cx, cy) 或抛 ValueError。"""
    obj, _, coords = token.partition(":")
    obj = obj.strip()
    cx_s, _, cy_s = coords.partition(",")
    cx, cy = int(cx_s), int(cy_s)
    if obj not in _VALID_OBJECTS:
        raise ValueError(f"无效物体 {obj}")
    if not (0 <= cx <= 639 and 0 <= cy <= 479):
        raise ValueError(f"坐标越界 {cx},{cy}")
    return obj, cx, cy


@app.post("/api/pick")
def pick(object: str, cx: int, cy: int):
    """单次抓取一个实物。object ∈ 四物品；cx/cy 为 640x480 画面像素。"""
    if object not in _VALID_OBJECTS:
        return JSONResponse({"ok": False, "error": f"无效物体 {object}"},
                            status_code=400)
    if not (0 <= cx <= 639 and 0 <= cy <= 479):
        return JSONResponse({"ok": False, "error": f"坐标越界 {cx},{cy}"},
                            status_code=400)
    started = STATE.start_real_pick(object, cx, cy)
    return {"ok": True, "running": STATE.is_running(),
            "started": started, "object": object, "cx": cx, "cy": cy}


@app.post("/api/pick_seq")
def pick_seq(items: str):
    """连贯抓取。items 形如 'copper_terminal:300,210;battery:480,170'。"""
    try:
        parsed = [_parse_pick_item(t) for t in items.split(";") if t.strip()]
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    if not parsed:
        return JSONResponse({"ok": False, "error": "空队列"}, status_code=400)
    started = STATE.start_real_pick_seq(parsed)
    return {"ok": True, "running": STATE.is_running(),
            "started": started, "count": len(parsed)}


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


def _video_cfg() -> dict:
    cfg = cl.get_web_config().get("video", {}) or {}
    return {
        "source": cfg.get("source", 0),
        "width": int(cfg.get("width", 640)),
        "height": int(cfg.get("height", 480)),
        "fps": int(cfg.get("fps", 15)),
    }


def _mjpeg_frames():
    """MJPEG 推流：只读 CAMERA 单例的「最新帧」，不自己开摄像头。

    V4L2 单设备独占——若每个 /video 连接各开一次摄像头，第二个连接起就黑屏。改由
    进程级 CameraHub 单线程独占摄像头读帧，这里所有连接共享同一份最新帧，任意多个
    浏览器/标签同时看都正常。抓取/拍照时 CameraHub 暂停读帧、画面停在最后一帧。
    """
    CAMERA.ensure_started()
    interval = 1.0 / max(1, _video_cfg()["fps"])
    blank_secs = 0.0
    while True:
        jpeg = CAMERA.latest_jpeg()
        if jpeg is None:
            blank_secs += 0.2
            if blank_secs > 20:   # 长时间无帧，结束本流让前端重连
                return
            time.sleep(0.2)
            continue
        blank_secs = 0.0
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg + b"\r\n")
        time.sleep(interval)


@app.get("/video")
def video():
    """实时视频流。开发板有摄像头则推 MJPEG；否则回退占位 JSON。"""
    if cv2 is None or _mock_mode():
        return JSONResponse({
            "stream": "mock",
            "note": "mock 模式或无 OpenCV：当前为占位，真机非 mock 时返回 MJPEG 流。",
        })
    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    fe = os.path.join(_ROOT, "web", "frontend", "index.html")
    if os.path.isfile(fe):
        with open(fe, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>电力库房机械臂管理系统</h1><p>前端页面缺失。</p>")
