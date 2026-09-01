"""
Agent 统一调度入口 —— Phase 2 核心

职责：
  1. 调用 Router 判断意图
  2. 分发到对应链路（FAQ / Ticket / Chitchat）
  3. 统一返回格式 + 调试信息

数据流：
  用户问题 → Router → [FAQChain | TicketChain | ChitchatChain] → 统一输出
"""
from __future__ import annotations

import time
from typing import Dict, Any

from loguru import logger

from app.agent.router import Router
from app.agent.intents import Intent
from app.agent.ticket import TicketChain
from app.agent.chitchat import ChitchatChain
from app.chains.faq_chain import FAQChain


class AgentChain:
    """
    Phase 2 Agent 统一调度器

    用法:
        agent = AgentChain()
        result = agent.run("帮我查一下订单 123456789 的物流")
    """

    def __init__(self, faq_data_path: str = "data/faq_sample.json"):
        logger.info("[Agent] 初始化路由器...")
        self._router = Router()

        logger.info("[Agent] 初始化工单链路...")
        self._ticket = TicketChain()

        logger.info("[Agent] 初始化闲聊链路...")
        self._chitchat = ChitchatChain()

        logger.info("[Agent] 初始化 FAQ 链路...")
        self._faq = FAQChain(faq_data_path=faq_data_path)

        logger.success("[Agent] 全部链路就绪 ✓")

    def run(self, query: str, include_debug: bool = True) -> Dict[str, Any]:
        """
        执行完整 Agent 流程

        流程:
          1. Router 判断意图
          2. 分发到对应链路
          3. 统一返回格式
        """
        t0 = time.time()

        # Step 1: 路由
        route = self._router.route(query)
        t_route = int((time.time() - t0) * 1000)
        logger.info(f"[Agent] 路由: '{query}' → {route.intent.value} ({route.source}, {t_route}ms)")

        # Step 2: 分发到对应链路
        t1 = time.time()
        if route.intent == Intent.FAQ:
            result = self._faq.run(query, include_debug=False)
        elif route.intent == Intent.TICKET:
            result = self._ticket.run(query, slots=route.slots)
        else:  # CHITCHAT
            result = self._chitchat.run(query)
        t_chain = int((time.time() - t1) * 1000)

        # Step 3: 统一输出
        output: Dict[str, Any] = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "fallback": result.get("fallback", False),
            "intent": route.intent.value,
        }
        if result.get("note"):
            output["note"] = result["note"]
        if result.get("need_slot"):
            output["need_slot"] = result["need_slot"]

        if include_debug:
            output["debug"] = {
                "query": query,
                "route": route.to_dict(),
                "chain_ms": t_chain,
                "route_ms": t_route,
                "total_ms": t_route + t_chain,
            }
            # FAQ 链路的候选信息（如果有）
            if route.intent == Intent.FAQ and result.get("debug"):
                output["debug"]["candidates"] = result["debug"].get("candidates", [])

        logger.info(
            f"[Agent] '{query}' → {route.intent.value} "
            f"({t_route}ms路由 + {t_chain}ms链路)"
        )
        return output