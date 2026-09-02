"""
Agent 统一调度入口 —— Phase 2 + Phase 3 混合架构

Phase 2 能力（基础）:
  Router → [FAQChain | TicketChain | ChitchatChain] 单工具处理

Phase 3 叠加（Agent 自主规划）:
  当 TICKET 意图包含复合连接词（「然后」「接着」）或 LLM 识别为多步骤时:
    → Planner 生成多步 Plan → Executor 逐步执行
  当检测到模糊请求时:
    → 主动追问（Clarifier 集成在 Planner 里）

数据流:
  Phase 2:  用户 → Router → [FAQ | Ticket | Chitchat] → 单工具
  Phase 3:  用户 → Router → Planner → Plan(多步) → Executor → 多步聚合结果
"""
from __future__ import annotations

import re
import time
from typing import Dict, Any, List

from loguru import logger

from app.agent.router import Router
from app.agent.intents import Intent
from app.agent.session_store import SessionContext, get_session_store
from app.agent.ticket import TicketChain
from app.agent.chitchat import ChitchatChain
from app.chains.faq_chain import FAQChain
from app.planning import Planner, Executor, PlanStatus
from app.agents import AnswerAgent


# 复合意图连接词（和 Planner 里的规则保持一致）
_COMPOUND_CONNECTORS = ["然后", "接着", "再", "之后", "同时", "顺便", "帮我也", "并且"]


class AgentChain:
    """
    Agent 统一调度器（Phase 2 + Phase 3）

    Phase 3 自动触发条件:
      - TICKET 意图 + 含复合连接词 → Planner + Executor
      - TICKET 意图 + 模糊请求 → 澄清追问
      - 单意图 TICKET → 走 Phase 2 TicketChain（更快更直接）

    用法:
        agent = AgentChain()
        # 单意图（Phase 2 路径）
        result = agent.run("帮我查订单123456789")
        # 复合意图（Phase 3 路径）
        result = agent.run("帮我查订单123的物流，然后退款", session_id="s1")
        # 模糊请求 → 追问
        result = agent.run("帮我查一下")
    """

    def __init__(self, faq_data_path: str = "data/faq_sample.json"):
        logger.info("[Agent] 初始化路由器...")
        self._router = Router()

        logger.info("[Agent] 初始化会话存储...")
        self._sessions = get_session_store()

        logger.info("[Agent] 初始化工单链路 (Phase 2)...")
        self._ticket = TicketChain()

        logger.info("[Agent] 初始化闲聊链路...")
        self._chitchat = ChitchatChain()

        logger.info("[Agent] 初始化 FAQ 链路...")
        self._faq = FAQChain(faq_data_path=faq_data_path)

        # Phase 3 新增
        logger.info("[Agent] 初始化 Planner + Executor (Phase 3)...")
        self._planner = Planner()
        self._executor = Executor()

        # Phase 3 答案生成 Agent
        logger.info("[Agent] 初始化 AnswerAgent...")
        self._answer_agent = AnswerAgent()

        logger.success("[Agent] Phase 2+3 全部组件就绪 ✓")

    def run(
        self,
        query: str,
        include_debug: bool = True,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """执行完整 Agent 流程（Phase 2 基础 + Phase 3 叠加）"""
        t0 = time.time()

        # Step 1: 加载会话
        session: SessionContext | None = None
        if session_id:
            session = self._sessions.load(session_id)
            if session is None:
                session = SessionContext(session_id=session_id)

        # Step 2: 路由（Phase 2 基础）
        route = self._router.route(query, session=session)
        t_route = int((time.time() - t0) * 1000)
        logger.info(f"[Agent] 路由: '{query}' → {route.intent.value} ({route.source}, {t_route}ms)")

        # Step 3: 槽位继承
        slot_inherited = False
        if (
            session is not None
            and route.intent == Intent.TICKET
            and not route.slots.get("order_id")
            and session.slots.get("order_id")
        ):
            route.slots["order_id"] = session.slots["order_id"]
            slot_inherited = True
            logger.info(f"[Agent] 继承会话订单号: {route.slots['order_id']}")

        # Step 4: 判断走 Phase 3 还是 Phase 2
        use_phase3 = False
        clarify_result = None

        if route.intent == Intent.TICKET:
            # 复合意图检测 → 走 Planner + Executor（Phase 3）
            if self._is_compound(query):
                use_phase3 = True
                logger.info("[Agent] 检测到复合意图，走 Phase 3 Planner")

            # 模糊请求检测（Phase 3 Clarifier）
            # 只拦截真正模糊到 TicketChain 也处理不了的请求
            # （TicketChain 本身能处理「帮我查订单」→ need_slot=order_id）
            clarify_result = self._detect_vague(query, session, route)
            if clarify_result is not None:
                use_phase3 = False

        # Step 5: 执行分发
        t1 = time.time()

        if clarify_result is not None:
            # 模糊请求 → 生成追问
            result = self._make_clarify_result(query, clarify_result, session)
        elif use_phase3:
            # Phase 3: Planner → Executor
            result = self._run_phase3(query, session_id or "", session)
        elif route.intent == Intent.FAQ:
            result = self._faq.run(query, include_debug=False)
        elif route.intent == Intent.TICKET:
            result = self._ticket.run(query, slots=route.slots)
            # Phase 2 TICKET: AnswerAgent 润色单工具结果
            if result.get("tool_used") and not result.get("need_slot"):
                result["answer"] = self._answer_agent.compose_single(
                    tool_name=result["tool_used"],
                    data=result.get("tool_data", {}),
                    success=result.get("tool_success", True),
                    message=result.get("answer", ""),
                    error="" if result.get("tool_success") else result.get("answer", ""),
                )
        else:  # CHITCHAT
            result = self._chitchat.run(query)

        t_chain = int((time.time() - t1) * 1000)

        # Step 6: 会话回写
        if session is not None:
            if route.intent == Intent.TICKET:
                if route.slots.get("order_id"):
                    session.slots["order_id"] = route.slots["order_id"]
                # Phase 3 结果也有 need_slot
                need_slot = result.get("need_slot") if isinstance(result, dict) else None
                session.pending_slot = (
                    "order_id" if need_slot == "order_id" else None
                )
            elif route.intent == Intent.CHITCHAT:
                session.clear()
            else:
                session.pending_slot = None
            session.last_intent = route.intent.value
            self._sessions.save(session)

        # Step 7: 统一输出
        output = self._build_output(
            result, route, include_debug, session, session_id,
            slot_inherited, t_route, t_chain, query,
            is_phase3=use_phase3, is_clarify=clarify_result is not None,
        )

        logger.info(
            f"[Agent] '{query}' → {route.intent.value} "
            f"({'P3' if use_phase3 else 'P2'}"
            f"{'澄清' if clarify_result else ''})"
            f" ({t_route}ms路由 + {t_chain}ms链路)"
        )
        return output

    # ---- Phase 3 核心流程 ----
    def _run_phase3(
        self,
        query: str,
        session_id: str,
        session: SessionContext | None,
    ) -> Dict[str, Any]:
        """Phase 3: Planner → Executor 完整链路"""
        # 生成计划
        plan = self._planner.plan(query, session_id=session_id, session=session)

        # 模糊请求（Planner 内部也做了检测）
        if plan.status == PlanStatus.PAUSED and plan.context.get("need_clarify"):
            options = plan.context.get("clarify_options", [])
            return self._make_clarify_result(query, options, session)

        # 空计划 → 降级 FAQ
        if not plan.steps:
            logger.info("[Agent:P3] Planner 返回空计划，降级 FAQ")
            return self._faq.run(query, include_debug=False)

        # 执行计划
        exec_result = self._executor.execute(plan)

        # 答案生成 Agent 润色（替换 Executor 简单拼接）
        step_results = exec_result.get("step_results", [])
        polished_answer = self._answer_agent.compose_multi(
            step_results=step_results,
            original_query=query,
        ) or exec_result.get("answer", "")  # 兜底

        return {
            "answer": polished_answer,
            "sources": [],
            "fallback": exec_result.get("final_status") != "success",
            "intent": "ticket",
            "phase3": True,
            "plan": exec_result,
            "plan_id": exec_result.get("plan_id"),
        }

    # ---- 模糊检测 ----
    def _detect_vague(
        self,
        query: str,
        session: SessionContext | None,
        route=None,
    ) -> List[str] | None:
        """检测模糊请求，返回澄清选项或 None

        核心原则: 只拦截 **TicketChain 也处理不了** 的请求
        TicketChain 能处理的（如「帮我查订单」→ need_slot=order_id）一律放行

        拦截场景（真正模糊）:
          1. 只有代词/模糊动词，路由可能无法正确分类（「那」「那个」「帮我」）
          2. 只有业务名词无动词（「那个订单」→ 不知道做什么）
        不拦截:
          - 「帮我查订单」「帮我退款」→ TicketChain 返回 need_slot
          - 「查订单」→ TicketChain 返回 need_slot
          - 路由已正确识别为 TICKET 且含 order_id → 直接放行
        """
        q = query.strip()
        has_order_id = bool(re.search(r"\d{6,20}", q))
        session_order_id = session.slots.get("order_id") if session else None

        # 前置: 会话已有订单号 → 上下文已消除歧义，不做模糊检测
        # （「那订单详情呢」「物流呢」都有明确指代）
        if session_order_id:
            return None

        biz_kw = any(kw in q for kw in ["订单", "物流", "快递", "退款", "退货"])
        action_kw = any(kw in q for kw in ["查", "看", "查询", "看看", "跟踪", "退", "申请", "取消", "帮我"])

        # 场景 1: 纯代词/模糊动词 → 连意图都不确定
        if q in ("那", "那个", "它", "这个", "帮我", "请帮我", "查一下", "看一下"):
            return ["查订单", "查物流", "申请退款"]

        # 场景 2: 只有业务名词无动词（「那个订单」「物流呢」）
        # → 不知道用户想做什么（查？退？）
        if biz_kw and not action_kw and not has_order_id:
            return ["查订单详情", "查物流状态", "申请退款"]

        # 场景 3: 会话有 pending_slot（上轮在等订单号），但本轮不是纯数字也没新订单号
        # → 用户没给订单号，反而说了别的，但也没说清要做什么
        if session and session.pending_slot and not has_order_id and not re.fullmatch(r"\d{6,20}", q):
            # 检查当前 query 是否明确是业务动词（如"帮我退款"）
            # 如果是，说明用户改变了意图，让 TicketChain 处理
            pass  # 不拦截，让下游处理

        return None

    # ---- 追问话术 ----
    @staticmethod
    def _make_clarify_result(
        query: str,
        options: List[str],
        session: SessionContext | None,
    ) -> Dict[str, Any]:
        """生成澄清追问的结果"""
        if "需要订单号" in options:
            answer = (
                "请问您要查询的订单号是多少呢？\n"
                "您可以在「我的订单」中找到订单号（通常为 9-12 位数字），"
                "直接告诉我订单号即可帮您查询。"
            )
        else:
            answer = (
                "您想做什么操作呢？可以告诉我：\n"
                f"  • {options[0]}\n"
                f"  • {options[1]}\n"
                f"  • {options[2] if len(options) > 2 else '其它需求'}"
            )

        return {
            "answer": answer,
            "sources": [],
            "fallback": False,
            "intent": "clarify",
            "clarify_options": options,
        }

    # ---- 复合意图检测 ----
    @staticmethod
    def _is_compound(query: str) -> bool:
        """检测是否为复合意图（含连接词）"""
        return any(connector in query for connector in _COMPOUND_CONNECTORS)

    # ---- 输出构建 ----
    @staticmethod
    def _build_output(
        result: Dict[str, Any],
        route,
        include_debug: bool,
        session: SessionContext | None,
        session_id: str | None,
        slot_inherited: bool,
        t_route: int,
        t_chain: int,
        query: str,
        is_phase3: bool,
        is_clarify: bool,
    ) -> Dict[str, Any]:
        """统一构建输出字典"""
        output: Dict[str, Any] = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "fallback": result.get("fallback", False),
            "intent": result.get("intent", route.intent.value),
        }
        if result.get("note"):
            output["note"] = result["note"]
        if result.get("need_slot"):
            output["need_slot"] = result["need_slot"]
        if result.get("clarify_options"):
            output["clarify_options"] = result["clarify_options"]
        if result.get("phase3"):
            output["phase3"] = True
            output["plan_id"] = result.get("plan_id")
        if session_id:
            output["session_id"] = session_id

        if include_debug:
            output["debug"] = {
                "query": query,
                "route": route.to_dict(),
                "chain_ms": t_chain,
                "route_ms": t_route,
                "total_ms": t_route + t_chain,
                "phase3": is_phase3,
                "clarify": is_clarify,
            }
            if session is not None:
                output["debug"]["session"] = {
                    "slot_inherited": slot_inherited,
                    "memory_slots": dict(session.slots),
                    "pending_slot": session.pending_slot,
                }
            if is_phase3 and result.get("plan"):
                output["debug"]["plan"] = result["plan"]

        return output
