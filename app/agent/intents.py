"""
意图定义 —— 路由系统的统一数据结构

三种意图：
  FAQ       → 静态知识库问答（Phase 1 已实现）
  TICKET    → 动态业务工单（查订单/物流/退款，需要调用工具）
  CHITCHAT  → 闲聊/兜底（无法识别意图时的友好回复）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Intent(str, Enum):
    FAQ = "faq"            # 静态 FAQ 问答
    TICKET = "ticket"      # 动态业务工单
    CHITCHAT = "chitchat"  # 闲聊/兜底


@dataclass
class RouteResult:
    """路由结果：意图 + 抽取的槽位 + 路由来源（LLM or 规则降级）"""
    intent: Intent
    confidence: float = 1.0           # 置信度（0-1，规则降级固定为 0.6）
    slots: dict[str, Any] = field(default_factory=dict)  # 抽取的参数（如 order_id）
    source: str = "rule"              # "llm" or "rule"
    raw_reason: str = ""              # 路由理由（调试用）

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "slots": self.slots,
            "source": self.source,
            "reason": self.raw_reason,
        }
