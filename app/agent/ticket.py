"""
工单链路 —— 处理动态业务操作（查订单、查物流、申请退款）

Phase 2.2-A 升级为工具化架构：
  - 业务操作封装为 Tool（app/tools/）
  - LLM 路径：Function Calling，让 LLM 自动选工具 + 抽参数
  - 规则路径：_detect_subintent 选工具 → ToolRegistry.call() 执行
  - 无 key 时降级为规则路径，保证可用

数据层（app/services/）通过 USE_MOCK_DATA 切换 mock / real。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from loguru import logger
from openai import OpenAI

from app.config import settings
from app.tools.registry import get_tool_registry
from app.tools.base import ToolResult


# 子意图 → 工具名映射
SUBINTENT_TO_TOOL = {
    "query_order": "query_order",
    "query_logistics": "query_logistics",
    "refund": "apply_refund",
}

# 业务子意图关键词（规则路径用）
SUBINTENT_KEYWORDS = {
    "query_order": ["订单状态", "订单", "查订单", "订单情况"],
    "query_logistics": ["物流", "快递", "到哪", "什么时候到", "送达", "签收"],
    "refund": ["退款", "退货", "退钱", "取消订单"],
}

# 无订单号时的引导话术（LLM 路径也会用到）
ASK_ORDER_ID_MSG = (
    "请问您要查询的订单号是多少呢？\n"
    "您可以在「我的订单」中找到订单号（通常为 9-12 位数字），"
    "直接告诉我订单号即可帮您查询。"
)


class TicketChain:
    """
    工单链路（工具化版）

    用法:
        chain = TicketChain()
        result = chain.run("帮我查一下订单 123456789 的物流", slots={"order_id": "123456789"})
    """

    def __init__(self):
        self._registry = get_tool_registry()
        self._client: Optional[OpenAI] = None
        if settings.LLM_API_KEY:
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
            logger.success("[Ticket] LLM Function Calling 已启用")
        else:
            logger.info("[Ticket] 使用规则工单链路（无 LLM，走 tools 规则路径）")

    def run(self, query: str, slots: Optional[dict] = None) -> Dict[str, Any]:
        """
        处理工单意图

        参数:
            query: 用户原始问题
            slots: 路由器抽取的槽位（如 order_id）
        """
        slots = slots or {}
        order_id = slots.get("order_id")

        # 没有订单号 → 引导用户提供
        if not order_id:
            return self._ask_for_order_id(query)

        # 有订单号 → 选路径执行
        if self._client is not None:
            return self._reply_with_function_calling(query, order_id)
        return self._reply_with_rules(query, order_id)

    # ---- 规则路径：子意图匹配 → 工具调用 ----
    def _reply_with_rules(self, query: str, order_id: str) -> Dict[str, Any]:
        """规则路径：用 _detect_subintent 选工具，调用 ToolRegistry"""
        subintent = self._detect_subintent(query)
        tool_name = SUBINTENT_TO_TOOL.get(subintent, "query_order")
        logger.info(f"[Ticket:rule] subintent={subintent} → tool={tool_name}")

        result = self._registry.call(tool_name, {"order_id": order_id})
        return self._format_result(result, subintent)

    def _detect_subintent(self, query: str) -> str:
        """识别业务子意图：query_order / query_logistics / refund

        优先匹配更具体的意图（物流/退款），最后才兜底为查订单。
        避免"查订单物流"被"订单"关键词截胡返回订单详情。
        """
        # 1. 先匹配物流（最具体）
        if any(kw in query for kw in SUBINTENT_KEYWORDS["query_logistics"]):
            return "query_logistics"
        # 2. 再匹配退款
        if any(kw in query for kw in SUBINTENT_KEYWORDS["refund"]):
            return "refund"
        # 3. 最后兜底查订单
        return "query_order"

    # ---- LLM 路径：Function Calling ----
    def _reply_with_function_calling(self, query: str, order_id: str) -> Dict[str, Any]:
        """LLM Function Calling：让 LLM 选工具 + 抽参数"""
        tools_schema = self._registry.tools_schema()
        system_prompt = (
            "你是电商客服助手。根据用户问题调用合适的工具查询业务信息，"
            "然后用简洁自然的中文回答用户。\n"
            f"已知订单号：{order_id}\n"
            "注意：必须通过调用工具获取数据，不要编造订单信息。"
        )

        try:
            # 第一轮：让 LLM 决定调用哪个工具
            resp = self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=512,
            )
            msg = resp.choices[0].message

            # LLM 没调工具 → 直接返回文本回复
            if not msg.tool_calls:
                logger.info("[Ticket:llm] LLM 未调用工具，直接回复")
                return {
                    "answer": (msg.content or "").strip() or ASK_ORDER_ID_MSG,
                    "sources": [{"order_id": order_id}],
                    "fallback": False,
                    "intent": "ticket",
                    "tool_used": None,
                }

            # 执行工具调用（支持多个 tool_calls，逐个执行）
            tool_results: list[ToolResult] = []
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                msg,
            ]
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                logger.info(f"[Ticket:llm] LLM 调用工具: {tool_name}, args={tc.function.arguments}")
                tr = self._registry.call_from_json(tool_name, tc.function.arguments)
                tool_results.append(tr)
                # 把工具结果回填给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tr.message if tr.success else f"查询失败：{tr.message}",
                })

            # 第二轮：把工具结果交给 LLM 生成自然语言回复
            final_resp = self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            answer = final_resp.choices[0].message.content.strip()

            # 取第一个工具结果作为 sources/元信息
            primary = tool_results[0]
            subintent = self._tool_name_to_subintent(primary.tool_name)
            return {
                "answer": answer,
                "sources": [{"order_id": order_id}],
                "fallback": False,
                "intent": "ticket",
                "subintent": subintent,
                "tool_used": primary.tool_name,
                "tool_success": primary.success,
            }
        except Exception as e:
            logger.warning(f"[Ticket:llm] Function Calling 失败，降级规则: {e}")
            return self._reply_with_rules(query, order_id)

    # ---- 辅助方法 ----
    def _format_result(self, result: ToolResult, subintent: str) -> Dict[str, Any]:
        """把 ToolResult 格式化为链路统一输出"""
        if not result.success:
            # 订单未找到等失败场景
            return {
                "answer": result.message,
                "sources": [],
                "fallback": False,
                "intent": "ticket",
                "subintent": subintent,
                "tool_used": result.tool_name,
                "tool_success": False,
            }
        order_id = result.data.get("order_id", "")
        product = result.data.get("product", "")
        return {
            "answer": result.message,
            "sources": [{"order_id": order_id, "product": product}],
            "fallback": False,
            "intent": "ticket",
            "subintent": subintent,
            "tool_used": result.tool_name,
            "tool_success": True,
        }

    def _ask_for_order_id(self, query: str) -> Dict[str, Any]:
        """引导用户提供订单号"""
        return {
            "answer": ASK_ORDER_ID_MSG,
            "sources": [],
            "fallback": False,
            "intent": "ticket",
            "need_slot": "order_id",
        }

    @staticmethod
    def _tool_name_to_subintent(tool_name: str) -> str:
        """工具名反查子意图（调试/日志用）"""
        for sub, tool in SUBINTENT_TO_TOOL.items():
            if tool == tool_name:
                return sub
        return "unknown"
