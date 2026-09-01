"""
工具基类 —— 统一工具接口与返回结构

Tool 子类需实现:
  - name: 工具名（LLM function name，需唯一）
  - description: 工具描述（给 LLM 看，决定是否调用）
  - parameters: JSON Schema，描述参数
  - execute(**kwargs) -> ToolResult: 实际执行逻辑
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool                       # 是否执行成功
    tool_name: str                      # 执行的工具名
    data: Dict[str, Any] = field(default_factory=dict)   # 结构化返回数据
    message: str = ""                   # 给用户/LLM 的自然语言说明
    error: str = ""                     # 失败时的错误信息

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "data": self.data,
            "message": self.message,
            "error": self.error,
        }


class Tool(ABC):
    """工具抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一标识（如 query_order）"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，供 LLM 决定是否调用"""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数 JSON Schema（OpenAI function calling 格式）"""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具，返回 ToolResult"""

    def to_openai_function(self) -> dict:
        """转换为 OpenAI Function Calling 的 tools schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
