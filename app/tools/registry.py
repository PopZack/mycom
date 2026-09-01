"""
工具注册表 —— 统一管理所有工具

职责:
  1. 注册并持有全部 Tool 实例
  2. 提供 OpenAI function calling schema 列表
  3. 按 name 调用对应工具

LLM 路径:  tools_schema() → LLM → call(name, args)
规则路径:  直接 call(name, args) 跳过 LLM
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from app.tools.base import Tool, ToolResult
from app.tools.order_tools import QueryOrderTool, QueryLogisticsTool, ApplyRefundTool


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        # 注册内置工具
        self._register(QueryOrderTool())
        self._register(QueryLogisticsTool())
        self._register(ApplyRefundTool())
        logger.info(f"[ToolRegistry] 已注册 {len(self._tools)} 个工具: {list(self._tools.keys())}")

    def _register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def tools_schema(self) -> List[dict]:
        """返回 OpenAI function calling 格式的 tools 列表"""
        return [t.to_openai_function() for t in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """按 name 调用工具

        参数:
            name: 工具名
            arguments: 参数 dict（如 {"order_id": "123456789"}）
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning(f"[ToolRegistry] 未知工具: {name}")
            return ToolResult(
                success=False,
                tool_name=name,
                error="unknown_tool",
                message=f"未知工具: {name}",
            )
        try:
            return tool.execute(**arguments)
        except Exception as e:
            logger.error(f"[ToolRegistry] 工具 {name} 执行异常: {e}")
            return ToolResult(
                success=False,
                tool_name=name,
                error="execution_error",
                message=f"工具执行出错: {e}",
            )

    def call_from_json(self, name: str, arguments_json: str) -> ToolResult:
        """从 LLM 返回的 JSON 字符串解析参数并调用"""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as e:
            logger.warning(f"[ToolRegistry] 参数 JSON 解析失败: {arguments_json!r}, err={e}")
            return ToolResult(
                success=False,
                tool_name=name,
                error="bad_arguments",
                message="工具参数格式错误",
            )
        return self.call(name, args)


# ---- 单例 ----
_registry_singleton: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = ToolRegistry()
    return _registry_singleton
