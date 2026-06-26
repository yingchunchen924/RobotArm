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
import re
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# 真机抓取：复用板上 run_robotarm_env.sh（已 source ROS humble + 工作区，再调
# scripts/board/color_sort_ros.py）。Web 点「开始」即以子进程跑 `auto --execute`，
# 动作与手动 `bash run_robotarm_env.sh auto --execute` 完全一致。
_REAL_GRAB_ENV = os.path.join(_ROOT, "run_robotarm_env.sh")

# 实物（非颜色方块）抓取：board 上 scripts/board/pick_xy.py CX CY ZONE --execute。
# 实物没有颜色识别，坐标由前端「在工作区照片上点击」给出。pick_xy 复用 color_sort_ros
# 的标定/IK/抓放序列，但入口不同于 color_sort_ros，所以不能用 run_robotarm_env.sh，
# 需自己 source ROS 再 python3 pick_xy.py。capture_shot.py 回拍照位拍一张 640x480 静态图。
_PICK_XY = "scripts/board/pick_xy.py"
_CAPTURE_SHOT = "scripts/board/capture_shot.py"
_STACK_ENV = os.path.join(_ROOT, "run_stacking_env.sh")
_PHOTO_PATH = "/tmp/workspace_shot.jpg"
_ROS_SOURCE = (
    "source /opt/ros/humble/setup.bash && "
    "source /home/HwHiAiUser/E2ESamples/src/E2E-Sample/"
    "ros2_robot_arm/ros2_ws/install/setup.bash"
)

# 物体 -> 放置区 + 抓取参数。前端四个物品按钮选「物体」即定了区和参数。
# None = 不传该 flag，用 pick_xy 默认。电池矮需 grip-down 下扎（已真机验证）。
_REAL_OBJECTS = {
    "copper_terminal": {"label": "铜端子", "zone": "blue",
                        "close": 180, "wrist": None, "grip_down": None},
    "copper_tube":     {"label": "铜管", "zone": "blue",
                        "close": 180, "wrist": None, "grip_down": None},
    "udisk":           {"label": "U盘", "zone": "yellow",
                        "close": 180, "wrist": None, "grip_down": None},
    "battery":         {"label": "蓝电池", "zone": "red",
                        "close": 180, "wrist": None, "grip_down": -14},
}

# 小米 MiMo API（语音识别 mimo-v2.5-asr，OpenAI 兼容）。
# API key 优先级：环境变量 MIMO_API_KEY > web.yaml 的 mimo.api_key。
# 不在代码里硬编码 key（避免上传仓库泄露）。本地运行前先配置其中之一。
def _mimo_cfg():
    c = (cl.get_web_config().get("mimo") or {})
    return {
        "api_key": os.environ.get("MIMO_API_KEY") or c.get("api_key", ""),
        "base_url": c.get("base_url", "https://api.xiaomimimo.com/v1"),
        "asr_model": c.get("asr_model", "mimo-v2.5-asr"),
    }

# 语音指令关键词 -> 动作。识别文字命中关键词即触发对应动作（稳、可控、不再调大模型）。
# 顺序有讲究：先匹配「停止/回位」这类控制词，再匹配颜色/堆叠/全抓。
_VOICE_INTENTS = [
    # (关键词列表, 动作类型, 参数, 中文说明)
    (["停", "停止", "暂停", "别动"], "stop", None, "停止"),
    (["初始", "复位", "回位", "归位", "回去"], "home", None, "回初始姿态"),
    (["堆叠", "堆", "码放", "叠起来", "叠"], "stack", None, "堆叠"),
    (["全抓", "全部", "都抓", "依次", "连贯"], "all", None, "连贯全抓"),
    (["红"], "color", "red", "抓红色"),
    (["黄"], "color", "yellow", "抓黄色"),
    (["蓝"], "color", "blue", "抓蓝色"),
    (["绿"], "color", "green", "抓绿色"),
]

# color_sort_ros 是颜色方块四区分拣，categories.yaml 里没有颜色映射，这里内置兜底。
# 与 config/stacking.yaml + 记忆“垃圾分类放置”的四区一致。
_COLOR_ZONE_LABEL = {
    "red": "有害垃圾",
    "yellow": "其他垃圾",
    "green": "厨余垃圾",
    "blue": "可回收物",
}

# auto 子命令参数见 _run_one_grab：每色跑
#   auto --execute --no-skip-unreachable --reach-ymax 0.42 --color <c> --cycles 1
# 这些都是 color_sort_ros 已有的命令行参数，不改脚本本身。

from robotarm import config_loader as cl                       # noqa: E402
from robotarm.interfaces import (                              # noqa: E402
    Detector, Kinematics, ArmDriver,
    MockArmDriver, MockDetector, MockKinematics,
)
from robotarm.inventory import Inventory, OperationRecord     # noqa: E402
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
        # 真机抓取子进程（color_sort_ros auto）相关
        self._grab_proc: Optional[subprocess.Popen] = None
        self._grab_stop: bool = False    # stop() 置 True，调度线程据此中断
        self.grab_active: bool = False   # True 时 /video 退避，避免抢摄像头
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
            if self._grab_proc is not None and self._grab_proc.poll() is None:
                return True
            return self.runner is not None and self.runner.running

    # ---- 控制 ----
    def start(self, mock: bool = True, interval: float = 1.0,
              colors: Optional[List[str]] = None) -> bool:
        """启动抓取。

        :param colors: 真机模式下要抓的颜色序列（按序串行）。
            None/[] -> 全抓固定顺序 red→yellow→blue→green；
            ["red"] -> 只抓红色一次。mock 模式忽略此参数。
        """
        with self._lock:
            if self.is_running():
                return False
            if not mock:
                return self._start_real_grab(colors)
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

    # ---- 真机抓取：子进程串行跑 color_sort_ros auto --color X ----
    _ALL_COLORS = ["red", "yellow", "blue", "green"]   # 全抓固定顺序

    def _start_real_grab(self, colors: Optional[List[str]] = None) -> bool:
        """启动调度线程，按 colors 顺序逐色跑子进程。需在持锁下调用。"""
        if not os.path.isfile(_REAL_GRAB_ENV):
            self.reporter.report(
                GraspState.FAILED, f"找不到启动脚本 {_REAL_GRAB_ENV}")
            return False
        seq = [c for c in (colors or self._ALL_COLORS)
               if c in _COLOR_ZONE_LABEL]
        if not seq:
            self.reporter.report(GraspState.FAILED, "无有效颜色")
            return False
        self._grab_stop = False
        self.grab_active = True   # /video 退避，让出摄像头
        self.reporter.report(GraspState.WAIT_DETECT, "启动真机抓取…")
        t = threading.Thread(target=self._real_grab_schedule, args=(seq,),
                             daemon=True)
        self._thread = t
        t.start()
        return True

    def _real_grab_schedule(self, colors: List[str]) -> None:
        """按序对每个颜色跑一个子进程，串行执行，可被 stop() 打断。"""
        # grab_active 已置 True；先等一拍让 /video 释放摄像头，避免第一色 open 失败
        time.sleep(1.0)
        try:
            for color in colors:
                if self._grab_stop:
                    break
                self._run_one_grab(color)
        finally:
            with self._lock:
                self.grab_active = False
                self.arm_status = "idle"
                self._grab_proc = None
            if self.reporter.current.state != GraspState.FAILED:
                self.reporter.report(GraspState.IDLE, "抓取结束")

    def _run_one_grab(self, color: str) -> None:
        """跑一次 `auto --color <color> --cycles 1`，阻塞到子进程结束。"""
        try:
            with open("/tmp/grab.log", "a", encoding="utf-8") as _lf:
                _lf.write(f"\n######## START color={color} {time.strftime('%H:%M:%S')} ########\n")
        except Exception:
            pass
        args = ["bash", _REAL_GRAB_ENV, "auto", "--execute",
                "--no-skip-unreachable", "--reach-ymax", "0.42",
                "--color", color, "--cycles", "1"]
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, cwd=_ROOT,
            )
        except Exception as e:
            self.reporter.report(GraspState.FAILED, f"启动抓取失败: {e}")
            return
        with self._lock:
            self._grab_proc = proc
        self._real_grab_loop(proc, default_color=color)

    def _real_grab_loop(self, proc: subprocess.Popen,
                        default_color: Optional[str] = None) -> None:
        """逐行读取 color_sort_ros stdout，映射为状态/检测/日志。

        被 _run_one_grab 每色调用一次；只负责解析“一个”子进程的输出，
        不动 grab_active / 不报 IDLE（那是调度线程 _real_grab_schedule 的事）。

        关键输出标记（board 实测）：
          === CYCLE n ===            一轮开始 -> WAIT_DETECT
          TARGET <color> ... cx=.. cy=..   识别到 -> DETECTED + 回写检测
          GRIP <a> <ms>ms            夹爪动作；夹紧(角度大)视为 GRASPING
          MOVE [..] ..               移动；抓取后段视为 PLACING
          WARN ..                    透传 detail
          SKIP ..                    跳过 -> 记 skipped 日志
          ERROR ..                   失败 -> FAILED
        """
        clock = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
        # 单色模式下 default_color 给定，SKIP/ERROR 即使没 TARGET 也能归到该色
        state = {"color": default_color, "grasped": False}

        def finish_cycle(result: str, note: str = "") -> None:
            color = state["color"]
            if not color:
                return
            zone = color  # 用颜色英文做 zone key，前端 zone_labels 补中文
            self.inventory.log_operation(OperationRecord(
                timestamp=clock(), category=_COLOR_ZONE_LABEL.get(color, color),
                confidence=1.0, result=result, zone=zone, note=note,
            ))
            state["color"] = None
            state["grasped"] = False

        try:
            for raw in proc.stdout:            # 阻塞按行读，进程结束自然退出
                line = raw.rstrip()
                if not line:
                    continue
                # 落盘便于排查（每色子进程的完整 stdout）
                try:
                    with open("/tmp/grab.log", "a", encoding="utf-8") as _lf:
                        _lf.write(line + "\n")
                except Exception:
                    pass

                if line.startswith("=== CYCLE"):
                    # 新一轮开始前，若上一轮已抓取但没显式结束，补记成功
                    if state["color"] and state["grasped"]:
                        finish_cycle("success")
                    m = re.search(r"CYCLE\s+(\d+)", line)
                    n = m.group(1) if m else "?"
                    self.reporter.report(GraspState.WAIT_DETECT, f"第 {n} 轮 识别中")

                elif line.startswith("TARGET"):
                    m = re.search(
                        r"TARGET\s+(\w+).*?cx=(\d+)\s+cy=(\d+)", line)
                    if m:
                        color, cx, cy = m.group(1), int(m.group(2)), int(m.group(3))
                        state["color"] = color
                        self.set_detections([Detection(
                            name=color, conf=1.0, cx=cx, cy=cy)])
                        self.reporter.report(
                            GraspState.DETECTED,
                            f"{_COLOR_ZONE_LABEL.get(color, color)}",
                            target=color)

                elif line.startswith("GRIP"):
                    # GRIP <angle> <ms>ms：角度大=夹紧=抓取中
                    m = re.search(r"GRIP\s+([\d.]+)", line)
                    if m and float(m.group(1)) >= 100:
                        state["grasped"] = True
                        self.reporter.report(
                            GraspState.GRASPING, target=state["color"])

                elif line.startswith("MOVE") and state["grasped"]:
                    # 夹紧后的移动属于搬运/放置段
                    self.reporter.report(GraspState.PLACING, target=state["color"])

                elif line.startswith("WARN"):
                    self.reporter.report(
                        self.reporter.current.state,
                        line[4:].strip(), target=state["color"])

                elif line.startswith("SKIP"):
                    finish_cycle("skipped", line[4:].strip())
                    self.reporter.report(GraspState.WAIT_DETECT, "跳过本轮")

                elif line.startswith("ERROR"):
                    finish_cycle("failed", line[5:].strip())
                    self.reporter.report(GraspState.FAILED, line[5:].strip())
        except Exception:
            pass
        finally:
            # 进程收尾：本色若抓取成功但没被新 CYCLE 收尾，补记
            if state["color"] and state["grasped"]:
                finish_cycle("success")
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            proc.wait()
            # 注意：不在此处复位 grab_active / 报 IDLE —— 多色串行时由
            # 调度线程 _real_grab_schedule 在全部颜色跑完后统一收尾。

    def _run_loop(self) -> None:
        assert self.runner is not None
        self.runner.run(should_continue=lambda: self.runner.running)

    # ---- 实物抓取：拍静态照片供前端点击 ----
    def capture_photo(self) -> Optional[bytes]:
        """回拍照位拍一张 640x480 工作区静态图，返回 JPEG 字节。

        与 /video、抓取共争摄像头（V4L2 独占）：拍照期间置 grab_active=True 让 /video
        退避。忙（正在抓取）则返回 None。mock 模式返回 photos/ 下一张静态图供 PC 演示。
        """
        with self._lock:
            if self.is_running():
                return None
            self.grab_active = True
        try:
            time.sleep(1.5)  # 让 /video 退避并彻底 release 摄像头（V4L2 独占）
            cmd = (f"{_ROS_SOURCE} && cd {_ROOT} && "
                   f"python3 {_CAPTURE_SHOT} {_PHOTO_PATH}")
            subprocess.run(["bash", "-lc", cmd], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(_PHOTO_PATH, "rb") as f:
                return f.read()
        except Exception:
            # PC/mock：回退一张仓库内静态图，方便前端联调
            for name in ("20.jpg", "run0.jpg"):
                p = os.path.join(_ROOT, "photos", name)
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        return f.read()
            return None
        finally:
            # 子进程已退出释放摄像头，再稍等让 V4L2 设备就绪，避免 /video 重开拿到坏句柄
            time.sleep(0.5)
            with self._lock:
                self.grab_active = False

    # ---- 实物抓取：子进程串行跑 pick_xy.py CX CY ZONE ----
    def start_real_pick(self, obj: str, cx: int, cy: int) -> bool:
        """单次：抓一个实物。obj ∈ _REAL_OBJECTS。"""
        return self.start_real_pick_seq([(obj, cx, cy)])

    def start_real_pick_seq(self, items: List) -> bool:
        """连贯：按 items=[(obj,cx,cy),...] 顺序逐个抓。"""
        with self._lock:
            if self.is_running():
                return False
            valid = []
            for obj, cx, cy in items:
                if obj not in _REAL_OBJECTS:
                    continue
                if not (0 <= cx <= 639 and 0 <= cy <= 479):
                    continue
                valid.append((obj, int(cx), int(cy)))
            if not valid:
                self.reporter.report(GraspState.FAILED, "无有效抓取项")
                return False
            self._grab_stop = False
            self.grab_active = True
            self.reporter.report(GraspState.WAIT_DETECT, "启动实物抓取…")
            t = threading.Thread(target=self._real_pick_schedule,
                                 args=(valid,), daemon=True)
            self._thread = t
            t.start()
            return True

    def _real_pick_schedule(self, items: List) -> None:
        """逐项跑 pick_xy 子进程，串行执行，可被 stop() 打断。"""
        time.sleep(1.0)  # 等 /video 释放摄像头，避免第一个 open 失败
        try:
            for obj, cx, cy in items:
                if self._grab_stop:
                    break
                self._run_one_pickxy(cx, cy, obj)
        finally:
            with self._lock:
                self.grab_active = False
                self.arm_status = "idle"
                self._grab_proc = None
            if self.reporter.current.state != GraspState.FAILED:
                self.reporter.report(GraspState.IDLE, "抓取结束")

    def _run_one_pickxy(self, cx: int, cy: int, obj: str) -> None:
        """跑一次 pick_xy.py CX CY ZONE --execute，阻塞到子进程结束。"""
        spec = _REAL_OBJECTS[obj]
        try:
            with open("/tmp/grab.log", "a", encoding="utf-8") as _lf:
                _lf.write(f"\n######## PICK {obj} ({cx},{cy}) "
                          f"{time.strftime('%H:%M:%S')} ########\n")
        except Exception:
            pass
        pyargs = (f"python3 {_PICK_XY} {cx} {cy} {spec['zone']} "
                  f"--execute --close {spec['close']}")
        if spec["wrist"] is not None:
            pyargs += f" --wrist {spec['wrist']}"
        if spec["grip_down"] is not None:
            pyargs += f" --grip-down {spec['grip_down']}"
        cmd = f"{_ROS_SOURCE} && cd {_ROOT} && {pyargs}"
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, cwd=_ROOT,
            )
        except Exception as e:
            self.reporter.report(GraspState.FAILED, f"启动抓取失败: {e}")
            return
        with self._lock:
            self._grab_proc = proc
        self._pickxy_loop(proc, obj)

    def _pickxy_loop(self, proc: subprocess.Popen, obj: str) -> None:
        """逐行解析 pick_xy.py stdout，映射为状态/检测/库存。

        标记（board 实测）：PIXEL (cx,cy) -> ARM_XY .. / IK [..] / APPROACH [..] /
        GRASP [..] / PLACE(zone) [..] / DONE。无显式 ERROR：失败为 Python Traceback。
        """
        clock = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
        spec = _REAL_OBJECTS[obj]
        state = {"done": False, "failed": False}

        def log_result(result: str, note: str = "") -> None:
            self.inventory.log_operation(OperationRecord(
                timestamp=clock(), category=spec["label"], confidence=1.0,
                result=result, zone=spec["zone"], note=note,
            ))

        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                try:
                    with open("/tmp/grab.log", "a", encoding="utf-8") as _lf:
                        _lf.write(line + "\n")
                except Exception:
                    pass

                if line.startswith("PIXEL"):
                    m = re.search(r"PIXEL\s+\((\d+),(\d+)\)", line)
                    if m:
                        cx, cy = int(m.group(1)), int(m.group(2))
                        self.set_detections([Detection(
                            name=obj, conf=1.0, cx=cx, cy=cy)])
                    self.reporter.report(
                        GraspState.DETECTED, spec["label"], target=obj)
                elif (line.startswith("IK") or line.startswith("APPROACH")
                      or line.startswith("GRASP")):
                    self.reporter.report(GraspState.GRASPING, target=obj)
                elif line.startswith("PLACE"):
                    self.reporter.report(GraspState.PLACING, target=obj)
                elif line.startswith("DONE"):
                    state["done"] = True
                    log_result("success")
                    self.reporter.report(GraspState.DONE,
                                         f"{spec['label']} 已放置", target=obj)
                elif "Traceback" in line or line.startswith("Error") \
                        or "Error:" in line:
                    if not state["failed"]:
                        state["failed"] = True
                        log_result("failed", line[:120])
                        self.reporter.report(GraspState.FAILED, line[:120])
        except Exception:
            pass
        finally:
            if not state["done"] and not state["failed"]:
                # 进程结束却没见 DONE 也没明确报错 -> 判失败
                log_result("failed", "未完成（无 DONE）")
                self.reporter.report(GraspState.FAILED, "抓取未完成")
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            proc.wait()
            # 不在此复位 grab_active：多项串行时由 _real_pick_schedule 统一收尾。

    # ---- 颜色方块堆叠：子进程跑 run_stacking_env.sh（color_sort_ros stack）----
    def start_real_stack(self) -> bool:
        """启动堆叠：按 red→yellow→blue→green 依次抓方块码到中心十字，最多 4 层。"""
        with self._lock:
            if self.is_running():
                return False
            if not os.path.isfile(_STACK_ENV):
                self.reporter.report(
                    GraspState.FAILED, f"找不到堆叠脚本 {_STACK_ENV}")
                return False
            self._grab_stop = False
            self.grab_active = True
            self.reporter.report(GraspState.WAIT_DETECT, "启动堆叠…")
            t = threading.Thread(target=self._real_stack_schedule, daemon=True)
            self._thread = t
            t.start()
            return True

    def _real_stack_schedule(self) -> None:
        time.sleep(1.0)  # 等 /video 释放摄像头
        try:
            if not self._grab_stop:
                self._run_stack()
        finally:
            with self._lock:
                self.grab_active = False
                self.arm_status = "idle"
                self._grab_proc = None
            if self.reporter.current.state != GraspState.FAILED:
                self.reporter.report(GraspState.IDLE, "堆叠结束")

    def _run_stack(self) -> None:
        try:
            with open("/tmp/grab.log", "a", encoding="utf-8") as _lf:
                _lf.write(f"\n######## STACK {time.strftime('%H:%M:%S')} ########\n")
        except Exception:
            pass
        try:
            proc = subprocess.Popen(
                ["bash", _STACK_ENV, "--execute"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, cwd=_ROOT,
            )
        except Exception as e:
            self.reporter.report(GraspState.FAILED, f"启动堆叠失败: {e}")
            return
        with self._lock:
            self._grab_proc = proc
        self._stack_loop(proc)

    def _stack_loop(self, proc: subprocess.Popen) -> None:
        """逐行解析 color_sort_ros stack stdout，映射为状态/库存。

        标记（board 实测）：STACK start_layer.. / TARGET <color> ..cx=..cy=.. /
        STACK_PLACE level=N .. / GRIP <角度> / MOVE .. / DONE / ERROR。
        （--execute 时 PLAN 前缀的行是计划预览，真实动作是无 PLAN 前缀的 GRIP/MOVE。）
        """
        clock = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
        state = {"color": None, "level": 0, "grasped": False, "placed": 0,
                 "failed": False}

        def finish_layer(result: str, note: str = "") -> None:
            color = state["color"] or "block"
            self.inventory.log_operation(OperationRecord(
                timestamp=clock(),
                category=f"堆叠第{state['level']}层 {_COLOR_ZONE_LABEL.get(color, color)}",
                confidence=1.0, result=result, zone="stack", note=note,
            ))

        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                try:
                    with open("/tmp/grab.log", "a", encoding="utf-8") as _lf:
                        _lf.write(line + "\n")
                except Exception:
                    pass

                if line.startswith("STACK ") or line.startswith("STACK\t"):
                    self.reporter.report(GraspState.WAIT_DETECT, "堆叠开始")
                elif line.startswith("TARGET"):
                    m = re.search(r"TARGET\s+(\w+).*?cx=(\d+)\s+cy=(\d+)", line)
                    if m:
                        color, cx, cy = m.group(1), int(m.group(2)), int(m.group(3))
                        state["color"] = color
                        state["grasped"] = False
                        self.set_detections([Detection(
                            name=color, conf=1.0, cx=cx, cy=cy)])
                        self.reporter.report(
                            GraspState.DETECTED,
                            f"堆叠 {_COLOR_ZONE_LABEL.get(color, color)}",
                            target=color)
                elif line.startswith("STACK_PLACE"):
                    m = re.search(r"level=(\d+)", line)
                    if m:
                        state["level"] = int(m.group(1))
                    self.reporter.report(
                        GraspState.PLACING,
                        f"放置第 {state['level']} 层", target=state["color"])
                elif line.startswith("GRIP"):
                    m = re.search(r"GRIP\s+([\d.]+)", line)
                    if m and float(m.group(1)) >= 100:
                        state["grasped"] = True
                        self.reporter.report(
                            GraspState.GRASPING, target=state["color"])
                elif line.startswith("MOVE") and state["grasped"]:
                    self.reporter.report(GraspState.PLACING, target=state["color"])
                elif line.startswith("WARN"):
                    self.reporter.report(self.reporter.current.state,
                                         line[4:].strip(), target=state["color"])
                elif line.startswith("SKIP"):
                    finish_layer("skipped", line[4:].strip())
                elif line.startswith("ERROR"):
                    state["failed"] = True
                    finish_layer("failed", line[5:].strip())
                    self.reporter.report(GraspState.FAILED, line[5:].strip())
                elif line.startswith("DONE"):
                    # 一层完成
                    if state["color"]:
                        finish_layer("success")
                        state["placed"] += 1
                    self.reporter.report(
                        GraspState.DONE,
                        f"已堆叠 {state['placed']} 层", target=state["color"])
                    state["color"] = None
                    state["grasped"] = False
        except Exception:
            pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            proc.wait()
            # grab_active 由 _real_stack_schedule 统一收尾。

    # ---- 语音指令：音频 -> ASR 文字 -> 关键词匹配 -> 触发动作 ----
    def voice_command(self, audio_bytes: bytes, mime: str = "") -> Dict:
        """处理一段录音：转 wav -> 调 mimo-v2.5-asr 得文字 -> 匹配意图 -> 触发动作。

        返回 {ok, text, action, action_label, error}。text=识别文字，action=触发的动作。
        浏览器录音多为 webm/ogg，这里用 ffmpeg 统一转 16k 单声道 wav 再送 ASR。
        """
        import base64
        import tempfile
        try:
            import requests
        except Exception:
            return {"ok": False, "text": "", "action": None,
                    "error": "板子缺 requests 库"}

        # 1) 落盘原始音频，ffmpeg 转 wav
        src = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        src.write(audio_bytes)
        src.close()
        wav = src.name + ".wav"
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", src.name, "-ar", "16000",
                 "-ac", "1", wav],
                capture_output=True, timeout=30)
            if not os.path.isfile(wav) or os.path.getsize(wav) == 0:
                return {"ok": False, "text": "", "action": None,
                        "error": "音频转换失败（ffmpeg）"}
            # 2) base64 -> 调 ASR
            b64 = base64.b64encode(open(wav, "rb").read()).decode()
            cfg = _mimo_cfg()
            if not cfg["api_key"]:
                return {"ok": False, "text": "", "action": None,
                        "error": "未配置 MiMo API key（设环境变量 MIMO_API_KEY "
                                 "或 config/web.yaml 的 mimo.api_key）"}
            resp = requests.post(
                f"{cfg['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {cfg['api_key']}",
                         "Content-Type": "application/json"},
                json={"model": cfg["asr_model"],
                      "messages": [{"role": "user", "content": [
                          {"type": "input_audio", "input_audio": {
                              "data": f"data:audio/wav;base64,{b64}"}}]}],
                      "asr_options": {"language": "zh"}},
                timeout=60)
            if resp.status_code != 200:
                return {"ok": False, "text": "", "action": None,
                        "error": f"ASR 失败 {resp.status_code}: {resp.text[:120]}"}
            text = (resp.json().get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or "").strip()
        except Exception as e:
            return {"ok": False, "text": "", "action": None,
                    "error": f"语音识别异常: {e}"}
        finally:
            for p in (src.name, wav):
                try:
                    os.remove(p)
                except Exception:
                    pass

        if not text:
            return {"ok": True, "text": "", "action": None,
                    "action_label": "没听清", "error": ""}

        # 3) 关键词匹配意图
        action, param, label = self._match_voice_intent(text)
        if action is None:
            return {"ok": True, "text": text, "action": None,
                    "action_label": "未匹配到指令", "error": ""}

        # 4) 触发动作（复用现有真机抓取/堆叠/停止/home）
        started = self._dispatch_voice_action(action, param)
        return {"ok": True, "text": text, "action": action,
                "action_label": label,
                "started": started, "error": ""}

    def _match_voice_intent(self, text: str):
        """匹配意图。先匹配控制词（停止/回位/堆叠/全抓），
        再按说话顺序提取所有颜色：单个=抓该色，多个=按顺序连贯抓。"""
        # 控制词优先（这些不看颜色）
        for kws, action, param, label in _VOICE_INTENTS:
            if action in ("stop", "home", "stack", "all"):
                if any(k in text for k in kws):
                    return action, param, label
        # 按出现顺序提取颜色
        color_map = [("红", "red", "红"), ("黄", "yellow", "黄"),
                     ("蓝", "blue", "蓝"), ("绿", "green", "绿")]
        seq = []
        for ch in text:
            for kw, color, cn in color_map:
                if ch == kw:
                    seq.append((color, cn))
        if not seq:
            return None, None, None
        if len(seq) == 1:
            return "color", seq[0][0], f"抓{seq[0][1]}色"
        # 多颜色 -> 按顺序连贯抓
        colors = [c for c, _ in seq]
        label = "连贯抓 " + "→".join(cn for _, cn in seq)
        return "seq", colors, label

    def _dispatch_voice_action(self, action, param) -> bool:
        mock = bool(cl.get_web_config().get("mock_mode", True))
        if action == "stop":
            return self.stop()
        if action == "home":
            self.home()
            return True
        if action == "stack":
            return self.start_real_stack()
        if action == "all":
            return self.start(mock=mock, interval=1.0)   # 全抓固定顺序
        if action == "color":
            return self.start(mock=mock, interval=1.0, colors=[param])
        if action == "seq":
            return self.start(mock=mock, interval=1.0, colors=param)
        return False

    def stop(self) -> bool:
        with self._lock:
            if not self.is_running():
                return False
            # 真机：通知调度线程别再起下一个颜色，并终止当前子进程
            self._grab_stop = True
            if self._grab_proc is not None and self._grab_proc.poll() is None:
                try:
                    self._grab_proc.terminate()
                except Exception:
                    pass
            # mock：停 runner
            if self.runner is not None:
                self.runner.stop()
            if self._frame_source:
                self._frame_source.stop()
        # 等线程退出（不持锁等待，避免死锁）
        if self._thread:
            self._thread.join(timeout=5.0)
        with self._lock:
            self.grab_active = False
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
            def _det_label(name: str) -> str:
                if name in _REAL_OBJECTS:
                    return _REAL_OBJECTS[name]["label"]
                return _COLOR_ZONE_LABEL.get(name) or _label(name)

            def _det_zone(name: str) -> str:
                if name in _REAL_OBJECTS:
                    return _REAL_OBJECTS[name]["zone"]
                if name in _COLOR_ZONE_LABEL:
                    return name
                return cl.category_to_zone(name)

            self._last_detections = [
                {"name": d.name,
                 "label": _det_label(d.name),
                 "conf": round(float(d.conf), 3),
                 "cx": d.cx, "cy": d.cy,
                 "zone": _det_zone(d.name)}
                for d in dets
            ]

    def inventory_snapshot(self) -> Dict:
        with self._lock:
            cats = cl.get_categories_config()
            zones = cats.get("zones", {})
            data = self.inventory.to_dict()
            zone_labels = {k: v.get("label", k) for k, v in zones.items()}
            # 真机颜色分拣的四区不在 categories.yaml，补进去前端才能显示中文区名
            zone_labels.update(_COLOR_ZONE_LABEL)
            data["zone_labels"] = zone_labels
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


class CameraHub:
    """进程级单例摄像头读帧器。

    V4L2 是单设备独占：同一时刻只能有一个 VideoCapture 句柄。若每个 /video 连接各开
    一次摄像头，第二个连接起就拿不到帧（黑屏）——这正是浏览器多连接/刷新残留导致的 bug。
    本类用**一个**后台线程独占摄像头、持续读「最新帧」，所有 /video 连接只读这份共享帧，
    底层永远只有一个 capture。抓取/拍照活跃时（STATE.grab_active）暂停读帧并 release，
    让位给子进程，结束后自动重开。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._cv2 = None

    def ensure_started(self) -> None:
        with self._lock:
            if self._started:
                return
            try:
                import cv2  # 延迟导入：PC 无 cv2 也不影响其它接口
                self._cv2 = cv2
            except Exception:
                return
            self._started = True
            t = threading.Thread(target=self._run, daemon=True)
            self._thread = t
            t.start()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def _run(self) -> None:
        cv2 = self._cv2
        backend = getattr(cv2, "CAP_V4L2", 0) if sys.platform.startswith("linux") else 0
        cfg = cl.get_web_config().get("video", {}) or {}
        w, h = int(cfg.get("width", 640)), int(cfg.get("height", 480))
        fps = int(cfg.get("fps", 15))
        interval = 1.0 / max(1, fps)
        cap = None
        read_fails = 0

        def _open():
            c = cv2.VideoCapture(int(cfg.get("source", 0)), backend)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            return c if c.isOpened() else (c.release() or None)

        while True:
            # 抓取/拍照活跃：彻底让出摄像头，期间不读帧（画面停在最后一帧）。
            if STATE.grab_active:
                if cap is not None:
                    cap.release()
                    cap = None
                read_fails = 0
                time.sleep(0.3)
                continue
            if cap is None:
                cap = _open()
                if cap is None:
                    time.sleep(0.5)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                read_fails += 1
                time.sleep(0.2 if read_fails < 5 else 1.0)
                continue
            read_fails = 0
            # 叠加最近一次识别框
            for d in STATE.detections_snapshot():
                cx, cy = int(d.get("cx", 0)), int(d.get("cy", 0))
                label = f"{d.get('label') or d.get('name')} {int(d.get('conf', 0) * 100)}%"
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, label, (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
            time.sleep(interval)


# 进程级单例摄像头读帧器
CAMERA = CameraHub()
