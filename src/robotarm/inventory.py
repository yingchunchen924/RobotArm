"""库房统计与操作日志数据结构。

对应开发计划第7节：库房区域统计（电子元器件/电力金具/工具/未分类数量）与
操作日志（时间、类别、置信度、抓取结果、放置区域）。

纯内存 + 可选 JSON/CSV 落盘，无硬件依赖，供 Web 后端与抓取主程序共享状态。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class OperationRecord:
    """一次抓取操作的日志记录。"""

    timestamp: str          # 由调用方传入（避免本模块依赖时钟，便于测试）
    category: str           # 识别类别
    confidence: float       # 置信度
    result: str             # "success" / "failed" / "skipped"
    zone: str               # 放置库区 key
    note: str = ""          # 备注（失败原因等）


@dataclass
class Inventory:
    """库房库存与操作日志的内存模型。

    counts: 各库区计数，例如 {"electronic": 3, "tools": 1, ...}
    logs:   操作日志列表。
    """

    counts: Dict[str, int] = field(default_factory=dict)
    logs: List[OperationRecord] = field(default_factory=list)

    # ---- 计数 ----
    def add(self, zone: str, n: int = 1) -> None:
        """入库：某库区计数 +n。"""
        self.counts[zone] = self.counts.get(zone, 0) + n

    def remove(self, zone: str, n: int = 1) -> None:
        """出库：某库区计数 -n，不低于 0。"""
        self.counts[zone] = max(0, self.counts.get(zone, 0) - n)

    def total(self) -> int:
        return sum(self.counts.values())

    def reset(self) -> None:
        """清空统计与日志（对应前端「清空统计」按钮）。"""
        self.counts.clear()
        self.logs.clear()

    # ---- 日志 ----
    def log_operation(self, record: OperationRecord) -> None:
        """记录一次操作；若为成功抓取则同时更新对应库区计数。"""
        self.logs.append(record)
        if record.result == "success":
            self.add(record.zone)

    def recent_logs(self, limit: int = 20) -> List[OperationRecord]:
        return self.logs[-limit:][::-1]  # 最近的在前

    # ---- 序列化 ----
    def to_dict(self) -> Dict:
        return {
            "counts": dict(self.counts),
            "total": self.total(),
            "logs": [asdict(r) for r in self.logs],
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def save_logs_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = ["timestamp", "category", "confidence", "result", "zone", "note"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.logs:
                w.writerow(asdict(r))

    @classmethod
    def load_json(cls, path: str) -> "Inventory":
        if not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inv = cls(counts=dict(data.get("counts", {})))
        for r in data.get("logs", []):
            inv.logs.append(OperationRecord(**r))
        return inv
