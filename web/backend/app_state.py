"""Web 与 runner 之间的共享状态层（开发计划阶段七）。

FastAPI（主线程，异步）与 PowerSortingRunner（后台线程，阻塞循环）共享一份状态：
StatusReporter + Inventory + 最近检测结果，全部加锁访问。start/stop 控制后台线程。

支持两种数据源：
- mock（默认，无硬件）：MockDetector + MockKinematics + MockArmDriver，runner 真跑，
  前端能看到真实动态的状态流转与库存增长。
- real：通过 build_detector 接真实模型 + 真机逆解/串口（在开发板上由入口配置注入）。

帧源带节流（每帧间隔 interval 秒），避免后台线程空转打满 CPU、也让前端轮询能看清状态。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from robotarm import config_loader as cl                       # noqa: E402
from robotarm.interfaces import (                              # noqa: E402
    Detector, Kinematics, ArmDriver,
    MockArmDriver, MockDetector, MockKinematics,
)
from robotarm.inventory import Inventory                       # noqa: E402
from robotarm.runner import PowerSortingRunner                 # noqa: E402
from robotarm.states import GraspState, StatusEvent, StatusReporter  # noqa: E402
from robotarm.target_selection import Detection                # noqa: E402


class ThrottledFrameSource:
    """节流帧源：每 interval 秒返回一帧。

    mock 模式下返回一个占位对象（内容不重要，MockDetector 不看帧）；
    真机模式下可换成 OpenCV 读帧。可被 stop() 中断 sleep。
    """

    def __init__(self, interval: float = 1.0, real_reader=None) -> None:
        self.interval = interval
        self.real_reader = real_reader
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def __call__(self):
        # 可中断的等待
        if self._stop.wait(self.interval):
            return None
        if self.real_reader is not None:
            return self.real_reader()
        return object()  # mock 占位帧


class AppState:
    """Web 全局共享状态。线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reporter = StatusReporter()
        self.inventory = Inventory()
        self.runner: Optional[PowerSortingRunner] = None
        self._thread: Optional[threading.Thread] = None
        self._frame_source: Optional[ThrottledFrameSource] = None
        # 缓存：最近一次识别结果（供 /api/detections）
        self._last_detections: List[Dict] = []
        self.arm_status: str = "idle"
        # 订阅状态事件，更新 arm_status
        self.reporter.subscribe(self._on_status)

    # ---- 状态回调 ----
    def _on_status(self, ev: StatusEvent) -> None:
        with self._lock:
            mapping = {
                GraspState.WAIT_DETECT: "moving",
                GraspState.DETECTED: "moving",
                GraspState.GRASPING: "grasping",
                GraspState.PLACING: "placing",
                GraspState.DONE: "idle",
                GraspState.FAILED: "idle",
                GraspState.IDLE: "idle",
            }
            self.arm_status = mapping.get(ev.state, "idle")

    # ---- 检测器构造 ----
    def _make_detector(self, mock: bool) -> Detector:
        if mock:
            base: Detector = _RotatingMockDetector()
        else:
            # 真机：由部署时通过环境/配置接入，此处占位
            raise RuntimeError("real 模式需在开发板入口注入真实 detector")
        # 包一层：每次 detect 的结果缓存到共享状态，供 /api/detections 展示
        return _CachingDetector(base, self)

    def is_running(self) -> bool:
        with self._lock:
            return self.runner is not None and self.runner.running

    # ---- 控制 ----
    def start(self, mock: bool = True, interval: float = 1.0) -> bool:
        with self._lock:
            if self.is_running():
                return False
            detector = self._make_detector(mock)
            kinematics: Kinematics = MockKinematics()
            arm: ArmDriver = MockArmDriver()
            self._frame_source = ThrottledFrameSource(interval=interval)
            self.runner = PowerSortingRunner(
                detector=detector, kinematics=kinematics, arm=arm,
                frame_source=self._frame_source,
                reporter=self.reporter, inventory=self.inventory,
                clock=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
                skip_repeat=0,
            )
            t = threading.Thread(target=self._run_loop, daemon=True)
            self._thread = t
            t.start()
            return True

    def _run_loop(self) -> None:
        assert self.runner is not None
        self.runner.run(should_continue=lambda: self.runner.running)

    def stop(self) -> bool:
        with self._lock:
            if not self.is_running():
                return False
            self.runner.stop()
            if self._frame_source:
                self._frame_source.stop()
        # 等线程退出（不持锁等待，避免死锁）
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._lock:
            self.arm_status = "idle"
        return True

    def home(self) -> None:
        with self._lock:
            self.arm_status = "homing"
            if self.runner is not None:
                try:
                    self.runner.pipeline.arm.home()
                except Exception:
                    pass
            self.arm_status = "idle"

    def reset(self) -> None:
        with self._lock:
            self.inventory.reset()
            self.reporter.reset()
            self._last_detections = []

    # ---- 只读快照（供接口）----
    def status_snapshot(self) -> Dict:
        with self._lock:
            cur = self.reporter.current
            return {
                "running": self.is_running(),
                "arm_status": self.arm_status,
                "state": cur.state.value,
                "state_label": cur.label,
                "detail": cur.detail,
                "target": cur.target,
            }

    def detections_snapshot(self) -> List[Dict]:
        with self._lock:
            return list(self._last_detections)

    def set_detections(self, dets: List[Detection]) -> None:
        with self._lock:
            self._last_detections = [
                {"name": d.name, "label": _label(d.name),
                 "conf": round(float(d.conf), 3),
                 "cx": d.cx, "cy": d.cy,
                 "zone": cl.category_to_zone(d.name)}
                for d in dets
            ]

    def inventory_snapshot(self) -> Dict:
        with self._lock:
            cats = cl.get_categories_config()
            zones = cats.get("zones", {})
            data = self.inventory.to_dict()
            data["zone_labels"] = {k: v.get("label", k) for k, v in zones.items()}
            data["recent_logs"] = [
                {"timestamp": r.timestamp, "category": r.category,
                 "confidence": r.confidence, "result": r.result,
                 "zone": r.zone, "note": r.note}
                for r in self.inventory.recent_logs(20)
            ]
            return data


def _label(name: str) -> str:
    return cl.get_categories_config().get("category_label", {}).get(name, name)


class _RotatingMockDetector(Detector):
    """mock 模式用：每次返回不同类别的单目标，制造多样的库区增长。"""

    _CYCLE = ["resistor", "wrench", "clamp", "capacitor", "screwdriver"]

    def __init__(self) -> None:
        self._i = 0

    def detect(self, frame) -> List[Detection]:
        name = self._CYCLE[self._i % len(self._CYCLE)]
        self._i += 1
        # 居中、可达
        return [Detection(name=name, conf=0.90, cx=320, cy=240)]


class _CachingDetector(Detector):
    """包装任意 Detector：把每次检测结果写入 AppState，供 Web 展示。"""

    def __init__(self, base: Detector, state: "AppState") -> None:
        self._base = base
        self._state = state

    def detect(self, frame) -> List[Detection]:
        dets = self._base.detect(frame)
        self._state.set_detections(dets)
        return dets


# 进程级单例
STATE = AppState()
