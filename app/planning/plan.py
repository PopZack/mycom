"""
Plan / PlanStep 数据结构 —— Phase 3 Agent 自主规划

设计要点（来自经验教训）:
  - step_id 必须稳定: 重规划时可复用原 ID，避免 retry 计数丢失
  - 状态机清晰: pending → running → success/failed/skipped
  - 结果归一化: executor 边界把异构 ToolResult 统一为 {status, data, error}
  - while+游标执行: 非 for-each，运行时重规划可真正生效
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---- 状态枚举 ----

class StepStatus(str, Enum):
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    SUCCESS = "success"       # 执行成功
    FAILED = "failed"         # 执行失败
    SKIPPED = "skipped"       # 条件未满足，跳过


class PlanStatus(str, Enum):
    PLANNING = "planning"     # 生成中
    RUNNING = "running"       # 执行中
    SUCCESS = "success"       # 全部成功
    FAILED = "failed"         # 有步骤失败且无法重规划
    PAUSED = "paused"         # 暂停（等待人工/澄清）


# ---- 数据结构 ----

@dataclass
class PlanStep:
    """单个计划步骤"""
    step_id: str                              # 稳定 ID，重规划时复用
    tool_name: str                            # 工具名
    params: Dict[str, Any] = field(default_factory=dict)  # 工具参数
    description: str = ""                     # 给 LLM/用户看的人类可读描述
    condition: str = ""                       # 条件表达式（规则路径用，如 "context.logistics.has_problem"）
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2                      # 单步最多重试次数
    result: Optional[Dict[str, Any]] = None   # 执行结果（归一化后）
    error: str = ""                           # 失败原因

    def can_run(self) -> bool:
        """判断是否可以开始执行"""
        return self.status == StepStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "params": self.params,
            "description": self.description,
            "condition": self.condition,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error": self.error,
        }


@dataclass
class Plan:
    """完整执行计划"""
    plan_id: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PLANNING
    original_query: str = ""
    session_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)  # 工具产出，后续步骤可引用
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_steps: int = 0

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"plan-{uuid.uuid4().hex[:10]}"
        self.total_steps = len(self.steps)

    def touch(self) -> None:
        self.updated_at = time.time()

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def current_running_index(self) -> int:
        """返回当前执行到的游标位置（第一个非 success 的步骤索引）"""
        for i, s in enumerate(self.steps):
            if s.status != StepStatus.SUCCESS:
                return i
        return len(self.steps)  # 全部成功

    def all_success(self) -> bool:
        return all(s.status == StepStatus.SUCCESS for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def mark_running(self) -> None:
        self.status = PlanStatus.RUNNING
        self.touch()

    def mark_success(self) -> None:
        self.status = PlanStatus.SUCCESS
        self.touch()

    def mark_failed(self) -> None:
        self.status = PlanStatus.FAILED
        self.touch()

    def mark_paused(self) -> None:
        self.status = PlanStatus.PAUSED
        self.touch()

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "original_query": self.original_query,
            "session_id": self.session_id,
            "total_steps": self.total_steps,
            "steps": [s.to_dict() for s in self.steps],
            "context_keys": list(self.context.keys()),
            "created_at": self.created_at,
        }


# ---- 工具函数 ----

def make_step(
    tool_name: str,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    condition: str = "",
    step_id: Optional[str] = None,
) -> PlanStep:
    """快速构造步骤（自动生成 step_id）"""
    return PlanStep(
        step_id=step_id or f"s{uuid.uuid4().hex[:6]}",
        tool_name=tool_name,
        params=params or {},
        description=description,
        condition=condition,
    )
