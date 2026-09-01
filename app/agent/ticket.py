"""
工单链路 —— 处理动态业务操作（查订单、查物流、申请退款）

Phase 2.1 阶段使用 mock 数据模拟业务接口：
  - 3 条 mock 订单（覆盖不同状态）
  - 规则匹配业务子意图（查订单/查物流/退款）
  - 有 LLM 时用 LLM 理解更复杂的表达

Phase 2.2 会升级为真正的 Function Calling + 外部 API 调用
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from loguru import logger
from openai import OpenAI

from app.config import settings

# ---- Mock 业务数据（模拟数据库/API 返回）----
MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "123456789": {
        "order_id": "123456789",
        "status": "已发货",
        "product": "无线蓝牙耳机",
        "amount": 299.00,
        "logistics": "顺丰快递 SF1234567890",
        "logistics_status": "运输中，预计明天送达",
        "created_at": "2026-08-28 14:30",
    },
    "987654321": {
        "order_id": "987654321",
        "status": "待发货",
        "product": "智能手表",
        "amount": 899.00,
        "logistics": None,
        "logistics_status": "尚未发货，预计 1-2 天内发出",
        "created_at": "2026-08-30 09:15",
    },
    "555666777": {
        "order_id": "555666777",
        "status": "已完成",
        "product": "充电宝 20000mAh",
        "amount": 159.00,
        "logistics": "京东快递 JD55566677701",
        "logistics_status": "已签收",
        "created_at": "2026-08-20 16:45",
    },
}

# 业务子意图关键词
SUBINTENT_KEYWORDS = {
    "query_order": ["订单状态", "订单", "查订单", "订单情况"],
    "query_logistics": ["物流", "快递", "到哪", "什么时候到", "送达", "签收"],
    "refund": ["退款", "退货", "退钱", "取消订单"],
}


class TicketChain:
    """
    工单链路

    用法:
        chain = TicketChain()
        result = chain.run("帮我查一下订单 123456789 的物流", slots={"order_id": "123456789"})
    """

    def __init__(self):
        self._client: Optional[OpenAI] = None
        if settings.LLM_API_KEY:
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
            logger.success("[Ticket] LLM 工单链路已启用")
        else:
            logger.info("[Ticket] 使用规则工单链路（无 LLM）")

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

        # 有订单号 → 查询 mock 数据
        order = MOCK_ORDERS.get(order_id)
        if not order:
            return self._order_not_found(order_id)

        # 识别子意图
        subintent = self._detect_subintent(query)

        # 生成回复
        if self._client is not None:
            return self._reply_with_llm(query, order, subintent)
        return self._reply_with_rules(query, order, subintent)

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

    def _ask_for_order_id(self, query: str) -> Dict[str, Any]:
        """引导用户提供订单号"""
        return {
            "answer": (
                "请问您要查询的订单号是多少呢？\n"
                "您可以在「我的订单」中找到订单号（通常为 9-12 位数字），"
                "直接告诉我订单号即可帮您查询。"
            ),
            "sources": [],
            "fallback": False,
            "intent": "ticket",
            "need_slot": "order_id",
        }

    def _order_not_found(self, order_id: str) -> Dict[str, Any]:
        """订单未找到"""
        return {
            "answer": (
                f"抱歉，未找到订单号为 {order_id} 的订单。\n"
                "请确认订单号是否正确，或联系人工客服协助查询。"
            ),
            "sources": [],
            "fallback": False,
            "intent": "ticket",
        }

    def _reply_with_rules(self, query: str, order: dict, subintent: str) -> Dict[str, Any]:
        """规则生成回复（无 LLM 时）"""
        oid = order["order_id"]
        product = order["product"]
        status = order["status"]
        amount = order["amount"]

        if subintent == "query_logistics":
            logistics = order.get("logistics", "无")
            logistics_status = order.get("logistics_status", "未知")
            answer = (
                f"订单 {oid}（{product}）物流信息：\n"
                f"  快递：{logistics}\n"
                f"  状态：{logistics_status}"
            )
        elif subintent == "refund":
            answer = (
                f"订单 {oid}（{product}）当前状态：{status}\n"
                f"如需退款，请在「我的订单」中点击「申请退款」。\n"
                f"已发货订单需先拒收，已签收订单需走退货流程。"
            )
        else:  # query_order
            answer = (
                f"订单 {oid} 详情：\n"
                f"  商品：{product}\n"
                f"  金额：¥{amount:.2f}\n"
                f"  状态：{status}\n"
                f"  下单时间：{order['created_at']}"
            )

        return {
            "answer": answer,
            "sources": [{"order_id": oid, "product": product}],
            "fallback": False,
            "intent": "ticket",
            "subintent": subintent,
            "mock_data": True,
        }

    def _reply_with_llm(self, query: str, order: dict, subintent: str) -> Dict[str, Any]:
        """用 LLM 生成更自然的回复"""
        order_info = (
            f"订单号：{order['order_id']}\n"
            f"商品：{order['product']}\n"
            f"金额：¥{order['amount']:.2f}\n"
            f"状态：{order['status']}\n"
            f"快递：{order.get('logistics', '无')}\n"
            f"物流状态：{order.get('logistics_status', '未知')}\n"
            f"下单时间：{order['created_at']}"
        )
        prompt = (
            f"用户问题：{query}\n\n"
            f"订单信息：\n{order_info}\n\n"
            f"识别到的子意图：{subintent}\n"
            f"请基于订单信息，用简洁自然的语气回答用户问题。使用中文。"
        )
        try:
            resp = self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            answer = resp.choices[0].message.content.strip()
            return {
                "answer": answer,
                "sources": [{"order_id": order["order_id"], "product": order["product"]}],
                "fallback": False,
                "intent": "ticket",
                "subintent": subintent,
                "mock_data": True,
            }
        except Exception as e:
            logger.warning(f"[Ticket] LLM 调用失败，降级规则: {e}")
            return self._reply_with_rules(query, order, subintent)