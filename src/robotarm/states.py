"""抓取流程状态机。

对应开发计划阶段三「增加执行状态输出：等待识别、已识别、抓取中、放置中、完成、失败」。

设计为轻量状态枚举 + 一个状态广播器（StatusReporter），让 pipeline 在每一步推进状态，
Web/日志/控制台都可以订阅同一份状态，不必各自维护。无硬件依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional


class GraspState(str, Enum):
    """抓取流程的状态。继承 str 便于直接 JSON 序列化与前端展示。"""

    IDLE = "idle"                  # 空闲
    WAIT_DETECT = "wait_detect"    # 等待识别
    DETECTED = "detected"          # 已识别
    GRASPING = "grasping"          # 抓取中
    PLACING = "placing"            # 放置中
    DONE = "done"                  # 完成
    FAILED = "failed"              # 失败


# 状态的中文显示名（前端/日志用）
STATE_LABELS = {
    GraspState.IDLE: "空闲",
    GraspState.WAIT_DETECT: "等待识别",
    GraspState.DETECTED: "已识别",
    GraspState.GRASPING: "抓取中",
    GraspState.PLACING: "放置中",
    GraspState.DONE: "完成",
    GraspState.FAILED: "失败",
}


@dataclass
class StatusEvent:
    """一次状态变更事件。"""

    state: GraspState
    detail: str = ""               # 附加信息，如类别名、失败原因
    target: Optional[str] = None   # 当前目标类别（可选）

    @property
    def label(self) -> str:
        return STATE_LABELS.get(self.state, self.state.value)


class StatusReporter:
    """状态广播器：记录当前状态 + 历史，并通知订阅者。

    pipeline 调用 ``report(state, detail)`` 推进状态；外部通过 ``subscribe(cb)``
    注册回调（如推送到 Web 或写日志）。
    """

    def __init__(self) -> None:
        self.current: StatusEvent = StatusEvent(GraspState.IDLE)
        self.history: List[StatusEvent] = []
        self._subscribers: List[Callable[[StatusEvent], None]] = []

    def subscribe(self, callback: Callable[[StatusEvent], None]) -> None:
        self._subscribers.append(callback)

    def report(
        self,
        state: GraspState,
        detail: str = "",
        target: Optional[str] = None,
    ) -> StatusEvent:
        event = StatusEvent(state=state, detail=detail, target=target)
        self.current = event
        self.history.append(event)
        for cb in self._subscribers:
            cb(event)
        return event

    def reset(self) -> None:
        self.current = StatusEvent(GraspState.IDLE)
        self.history.clear()
