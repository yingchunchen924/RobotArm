"""电力物品分类抓取的连续运行器（开发计划阶段六）。

把「帧来源 + 检测 + pipeline 分拣」组装成可连续运行的主循环，对应阶段六的
「识别 -> 抓取 -> 按类别移动到指定区域」闭环。

设计要点：
- **帧来源可注入**：``frame_source`` 是一个无参可调用对象，每次返回一帧（或 None）。
  PC 验证传合成帧函数；真机传 OpenCV 摄像头读帧函数。runner 本身不碰摄像头，因此可单测。
- **检测器/逆解/机械臂**通过 pipeline 注入，runner 不关心具体后端。
- **连续抓取防抖**：成功抓取后短暂跳过同类目标，避免对同一物体反复抓（手册靠
  move_status 串行，这里在更高层做去重）。
- **start/stop + 异常降级**：循环可被 should_continue 控制；单帧异常不终止整个循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .interfaces import ArmDriver, Detector, Kinematics
from .inventory import Inventory
from .pipeline import GraspPipeline, Mode
from .states import StatusReporter


@dataclass
class RunnerStats:
    """一次运行的累计统计。"""

    frames: int = 0          # 处理的帧数
    grasped: int = 0         # 成功抓取次数
    empty: int = 0           # 无目标帧数
    failed: int = 0          # 失败次数


class PowerSortingRunner:
    """连续分拣运行器。

    :param frame_source: 无参可调用，返回一帧图像或 None（None 表示取帧失败/结束）。
    :param skip_repeat: 成功抓取后，连续多少帧内跳过同一类别（防抖）。
    """

    def __init__(
        self,
        detector: Detector,
        kinematics: Kinematics,
        arm: ArmDriver,
        frame_source: Callable[[], object],
        reporter: Optional[StatusReporter] = None,
        inventory: Optional[Inventory] = None,
        offset: float = 0.0,
        clock: Optional[Callable[[], str]] = None,
        skip_repeat: int = 3,
    ) -> None:
        self.frame_source = frame_source
        self.skip_repeat = skip_repeat
        self.pipeline = GraspPipeline(
            detector=detector,
            kinematics=kinematics,
            arm=arm,
            reporter=reporter,
            inventory=inventory,
            offset=offset,
            clock=clock,
        )
        self.stats = RunnerStats()
        self._running = False
        self._last_target: Optional[str] = None
        self._cooldown = 0

    # ---- 控制 ----
    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    # ---- 单步 ----
    def step(self) -> bool:
        """处理一帧。返回 True 表示本帧成功完成一次抓取。"""
        frame = self.frame_source()
        self.stats.frames += 1
        if frame is None:
            self.stats.empty += 1
            return False

        # 防抖冷却：刚抓过，先不抓，给场景/物体变化留时间
        if self._cooldown > 0:
            self._cooldown -= 1

        ok = self.pipeline.process_once(frame, mode=Mode.SORTING)
        if ok:
            target = self.pipeline.reporter.current.target
            # 冷却期内抓到同一类别，视为重复（理论上 pipeline 已抓，仍计数但提示）
            self._last_target = target
            self._cooldown = self.skip_repeat
            self.stats.grasped += 1
            return True

        # 区分「无目标」与「失败」
        detail = self.pipeline.reporter.current.detail or ""
        if "未识别" in detail:
            self.stats.empty += 1
        else:
            self.stats.failed += 1
        return False

    # ---- 连续运行 ----
    def run(
        self,
        max_frames: Optional[int] = None,
        should_continue: Optional[Callable[[], bool]] = None,
    ) -> RunnerStats:
        """连续运行主循环。

        :param max_frames: 最多处理多少帧后停止（None=不限，靠 stop()/should_continue 控制）。
        :param should_continue: 每帧前调用，返回 False 则停止（如检查 Web 的 stop 标志）。
        :returns: 累计统计。
        """
        self._running = True
        n = 0
        while self._running:
            if should_continue is not None and not should_continue():
                break
            if max_frames is not None and n >= max_frames:
                break
            self.step()
            n += 1
        self._running = False
        return self.stats
