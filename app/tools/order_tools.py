"""
订单业务工具集 —— 三个工具对应三种子意图

  QueryOrderTool      → 查订单详情
  QueryLogisticsTool  → 查物流状态
  ApplyRefundTool     → 申请退款

每个工具调用 OrderService，对上层屏蔽 mock/real 差异。
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from app.tools.base import Tool, ToolResult
from app.services.order_service import get_order_service


class QueryOrderTool(Tool):
    """查订单详情"""

    @property
    def name(self) -> str:
        return "query_order"

    @property
    def description(self) -> str:
        return "查询订单详情，包括商品、金额、状态、下单时间。当用户想了解订单整体情况时调用。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，通常为 6-20 位数字",
                },
            },
            "required": ["order_id"],
        }

    def execute(self, order_id: str, **_) -> ToolResult:
        logger.info(f"[Tool:query_order] order_id={order_id}")
        order = get_order_service().query_order(order_id)
        if order is None:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="order_not_found",
                message=f"未找到订单号为 {order_id} 的订单",
            )
        return ToolResult(
            success=True,
            tool_name=self.name,
            data=order,
            message=(
                f"订单 {order['order_id']} 详情：\n"
                f"  商品：{order['product']}\n"
                f"  金额：¥{order['amount']:.2f}\n"
                f"  状态：{order['status']}\n"
                f"  下单时间：{order['created_at']}"
            ),
        )


class QueryLogisticsTool(Tool):
    """查物流状态"""

    @property
    def name(self) -> str:
        return "query_logistics"

    @property
    def description(self) -> str:
        return "查询订单的物流状态，包括快递公司和配送进度。当用户问「到哪了」「什么时候到」「快递」时调用。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，通常为 6-20 位数字",
                },
            },
            "required": ["order_id"],
        }

    def execute(self, order_id: str, **_) -> ToolResult:
        logger.info(f"[Tool:query_logistics] order_id={order_id}")
        logistics = get_order_service().query_logistics(order_id)
        if logistics is None:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="order_not_found",
                message=f"未找到订单号为 {order_id} 的订单",
            )
        return ToolResult(
            success=True,
            tool_name=self.name,
            data=logistics,
            message=(
                f"订单 {logistics['order_id']}（{logistics['product']}）物流信息：\n"
                f"  快递：{logistics['logistics']}\n"
                f"  状态：{logistics['logistics_status']}"
            ),
        )


class ApplyRefundTool(Tool):
    """申请退款"""

    @property
    def name(self) -> str:
        return "apply_refund"

    @property
    def description(self) -> str:
        return "为订单申请退款。当用户明确要求退款/退货/取消订单时调用。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，通常为 6-20 位数字",
                },
                "reason": {
                    "type": "string",
                    "description": "退款原因（可选）",
                },
            },
            "required": ["order_id"],
        }

    def execute(self, order_id: str, reason: str = "", **_) -> ToolResult:
        logger.info(f"[Tool:apply_refund] order_id={order_id}, reason={reason!r}")
        result = get_order_service().apply_refund(order_id, reason=reason)
        return ToolResult(
            success=result.get("success", False),
            tool_name=self.name,
            data=result,
            message=result.get("message", ""),
        )
