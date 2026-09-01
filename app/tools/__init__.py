"""
工具模块 —— Phase 2.2-A

将工单链路的业务操作抽象为「工具」，供 LLM Function Calling 或规则路径调用。

设计要点：
  1. 每个 Tool 有统一的 name / description / parameters schema / execute()
  2. parameters 用 OpenAI Function Calling 的 JSON Schema 格式
  3. 无 LLM 时，规则路径也能直接 execute() 工具，保证降级可用
"""
from app.tools.base import Tool, ToolResult
from app.tools.order_tools import QueryOrderTool, QueryLogisticsTool, ApplyRefundTool
from app.tools.registry import ToolRegistry, get_tool_registry

__all__ = [
    "Tool", "ToolResult",
    "QueryOrderTool", "QueryLogisticsTool", "ApplyRefundTool",
    "ToolRegistry", "get_tool_registry",
]
