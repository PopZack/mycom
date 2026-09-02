"""
Phase 3 — Agent 自主规划模块

子模块:
  plan.py      Plan/PlanStep 数据结构 + 状态机
  planner.py   计划生成器（LLM + 规则模板双路径）
  executor.py  执行引擎（while+游标 + 结果归一化 + 重规划）
  clarifier.py 意图澄清（主动追问模糊请求）
"""
from app.planning.plan import Plan, PlanStep, PlanStatus, StepStatus, make_step
from app.planning.planner import Planner
from app.planning.executor import Executor

__all__ = [
    "Plan", "PlanStep", "PlanStatus", "StepStatus", "make_step",
    "Planner", "Executor",
]
