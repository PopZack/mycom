"""
Planner —— 计划生成器（Phase 3 Agent 自主规划）

双路径设计：
  1. LLM 路径（配置 LLM_API_KEY 时）：
     用 Function Calling + 规划 Prompt，让 LLM 直接输出结构化 Plan JSON
     准确率高，能理解复杂条件和上下文依赖
  2. 规则降级路径（无 Key 时）：
     关键词匹配复合意图连接词 → 模板化分解
     覆盖常见的「查→退」「查物流→查订单」等组合

规划能力边界：
  - 复合意图分解：「查订单123物流，然后退款」→ [query_logistics, apply_refund]
  - 条件步骤：「如果有问题就退款」→ step.condition = "context.logistics.has_problem"
  - 模糊检测：「帮我查一下」→ 返回 NEED_CLARIFY 标记，触发 Clarifier
  - 单意图降级：简单查询 → 单步 Plan（兼容 Phase 2）
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from loguru import logger
from openai import OpenAI

from app.config import settings
from app.agent.intents import Intent
from app.agent.router import Router
from app.agent.session_store import SessionContext
from app.planning.plan import Plan, PlanStep, make_step
from app.tools.registry import get_tool_registry


# ---- 规则路径：复合意图连接词 ----
COMPOUND_CONNECTORS = ["然后", "接着", "再", "之后", "同时", "顺便", "帮我也", "并且"]
CONDITION_PATTERN = re.compile(r"如果(.+?)(?:就|那么|的话)(.+)", re.DOTALL)

# ---- 规则路径：动词→工具映射（复用 router 的关键词词典 + 扩展）----
VERB_TO_TOOL = {
    # 查订单
    "查订单": "query_order", "查一下订单": "query_order",
    "订单状态": "query_order", "订单详情": "query_order",
    "看看订单": "query_order", "订单情况": "query_order",
    # 查物流
    "查物流": "query_logistics", "查快递": "query_logistics",
    "快递到哪": "query_logistics", "物流状态": "query_logistics",
    "什么时候到": "query_logistics", "送达": "query_logistics",
    "跟踪": "query_logistics", "追踪": "query_logistics",
    # 退款/退货
    "退款": "apply_refund", "退货": "apply_refund",
    "申请退款": "apply_refund", "取消订单": "apply_refund",
    "退钱": "apply_refund", "拒收": "apply_refund",
}

# ---- 模糊请求检测：只有动词没有具体宾语 ----
VAGUE_QUERIES = {
    "帮我查一下": "query_order", "帮我看看": "query_order",
    "查一下": "query_order", "看一下": "query_order",
    "帮我查询": "query_order", "查询一下": "query_order",
    "帮我退": "apply_refund", "帮我申请": "apply_refund",
}


class Planner:
    """
    计划生成器

    用法:
        planner = Planner()
        plan = planner.plan("帮我查订单123的物流，然后退款", session_id="s1")
        # plan.steps → [query_logistics(s1), apply_refund(条件)]
    """

    # 特殊返回标记：表示需要澄清
    NEED_CLARIFY = "__NEED_CLARIFY__"

    def __init__(self):
        self._client: Optional[OpenAI] = None
        if settings.LLM_API_KEY:
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
            logger.success("[Planner] LLM 规划已启用")
        else:
            logger.info("[Planner] LLM_API_KEY 未配置，使用规则模板规划")

        self._registry = get_tool_registry()
        self._router = Router()

    def plan(
        self,
        query: str,
        session_id: str = "",
        session: Optional[SessionContext] = None,
    ) -> Plan:
        """生成执行计划（入口）

        返回:
            Plan 对象；如果需要澄清，Plan.status = PAUSED 且 context["need_clarify"] = True
        """
        # 先做模糊检测
        clarify_result = self._try_clarify(query)
        if clarify_result is not None:
            plan = Plan(
                plan_id="",
                original_query=query,
                session_id=session_id,
                status=PlanStatus.PAUSED,
            )
            plan.context["need_clarify"] = True
            plan.context["clarify_options"] = clarify_result
            plan.touch()
            logger.info(f"[Planner] 模糊请求，需澄清: {clarify_result}")
            return plan

        # LLM 路径（优先）
        if self._client is not None:
            try:
                return self._plan_with_llm(query, session_id, session)
            except Exception as e:
                logger.warning(f"[Planner] LLM 规划失败，降级规则: {e}")

        # 规则路径
        return self._plan_with_rules(query, session_id, session)

    # ---- 模糊检测 ----
    def _try_clarify(self, query: str) -> Optional[List[str]]:
        """检测模糊请求，返回澄清选项（具体意图列表）或 None

        模糊判断标准：
          - 只有动词没有具体宾语（「帮我查一下」→ 不知道查什么）
          - 只提到业务名词但无动作（「那个订单」→ 不知道做什么）
          - 代词指代不明确（「它」「这个」→ 不知道指什么）
        """
        q = query.strip()

        # 精确匹配模糊短语
        if q in VAGUE_QUERIES:
            tool_hint = VAGUE_QUERIES[q]
            return self._tool_to_clarify_options(tool_hint)

        # 只有业务名词无动词：「那个订单」「物流呢」
        biz_kw = any(kw in q for kw in ["订单", "物流", "快递", "退款", "退货"])
        action_kw = any(kw in q for kw in ["查", "看", "查询", "看看", "跟踪", "退", "申请", "取消"])
        if biz_kw and not action_kw and not re.search(r"\d{6,20}", q):
            return ["查订单详情", "查物流状态", "申请退款", "取消订单"]

        # 代词指代
        if q in ("那", "那个", "它", "这个", "帮我", "请帮我"):
            return ["查订单", "查物流", "申请退款"]

        return None

    def _tool_to_clarify_options(self, tool_hint: str) -> List[str]:
        """根据 hint 工具生成澄清选项"""
        if tool_hint == "query_order":
            return ["查订单详情", "查物流状态", "申请退款"]
        elif tool_hint == "apply_refund":
            return ["申请退款", "取消订单", "查订单"]
        else:
            return ["查订单", "查物流", "申请退款"]

    # ---- LLM 路径 ----
    def _plan_with_llm(
        self,
        query: str,
        session_id: str,
        session: Optional[SessionContext],
    ) -> Plan:
        """用 LLM Function Calling 规划多步任务"""
        tools_schema = self._registry.tools_schema()
        order_id_hint = ""
        if session and session.slots.get("order_id"):
            order_id_hint = f"\n会话历史订单号：{session.slots['order_id']}（若用户未提及新订单号可继承）"

        prompt = PLAN_PROMPT.format(query=query, order_id_hint=order_id_hint)
        resp = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=512,
        )
        msg = resp.choices[0].message

        # LLM 输出了 tool_calls → 拆成 PlanStep
        if msg.tool_calls:
            steps: List[PlanStep] = []
            for i, tc in enumerate(msg.tool_calls):
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                steps.append(make_step(
                    tool_name=tc.function.name,
                    params=args,
                    description=f"LLM 规划第 {i+1} 步: {tc.function.name}({args})",
                ))
            plan = Plan(
                plan_id="",
                steps=steps,
                original_query=query,
                session_id=session_id,
            )
            logger.success(f"[Planner:llm] 生成 {len(steps)} 步计划")
            return plan

        # LLM 没调工具 → 可能是单意图或闲聊
        logger.info("[Planner:llm] LLM 未调用工具，走意图路由器")
        return self._plan_with_rules(query, session_id, session)

    # ---- 规则路径 ----
    def _plan_with_rules(
        self,
        query: str,
        session_id: str,
        session: Optional[SessionContext],
    ) -> Plan:
        """规则模板分解复合意图"""
        q = query.strip()
        steps: List[PlanStep] = []

        # 0. 会话继承的订单号
        session_order_id = session.slots.get("order_id") if session else None
        order_id_match = re.search(r"(?:订单号|订单|单号|order)[号:：\s]*(\d{6,20})|(\d{10,20})", q, re.IGNORECASE)
        query_order_id = (order_id_match.group(1) or order_id_match.group(2)) if order_id_match else None
        order_id = query_order_id or session_order_id

        # 1. 检测复合意图（含连接词）
        has_compound = any(connector in q for connector in COMPOUND_CONNECTORS)

        if has_compound:
            # 按连接词拆分，逐个匹配工具
            sub_queries = re.split(r"然后|接着|再|之后|同时|顺便|帮我也|并且", q)
            steps = []
            for sub in sub_queries:
                sub = sub.strip()
                if not sub:
                    continue
                tool_name, params = self._match_tool(sub, order_id, session_order_id)
                if tool_name:
                    # 后续步骤如果没订单号，尝试从前面步骤继承
                    if tool_name in ("query_logistics", "apply_refund") and "order_id" not in params:
                        # 看是否前面有 query_order 步骤
                        if any(s.tool_name == "query_order" for s in steps):
                            params["order_id"] = "{{prev.order_id}}"  # 占位，executor 会替换
                        elif order_id:
                            params["order_id"] = order_id
                    steps.append(make_step(
                        tool_name=tool_name,
                        params=params,
                        description=f"规则模板: {tool_name}({params})",
                    ))

            # 2. 检测条件步骤（「如果...就...」）
            cond_match = CONDITION_PATTERN.search(q)
            if cond_match and len(steps) >= 2:
                # 把最后一个步骤标记为条件执行
                condition_text = cond_match.group(1).strip()
                steps[-1].condition = self._build_condition(condition_text)
                steps[-1].description += f" [条件: {condition_text}]"

        else:
            # 单意图：匹配一个工具
            tool_name, params = self._match_tool(q, order_id, session_order_id)
            if tool_name:
                steps.append(make_step(
                    tool_name=tool_name,
                    params=params,
                    description=f"规则模板: {tool_name}({params})",
                ))
            else:
                # 没匹配到业务工具 → 返回空计划，让上层走 FAQ/闲聊
                logger.info("[Planner:rule] 未匹配业务工具，返回空计划")

        plan = Plan(
            plan_id="",
            steps=steps,
            original_query=query,
            session_id=session_id,
        )
        plan.touch()
        logger.success(f"[Planner:rule] 生成 {len(steps)} 步计划: {[s.tool_name for s in steps]}")
        return plan

    def _match_tool(
        self,
        sub_query: str,
        order_id: Optional[str],
        session_order_id: Optional[str],
    ) -> tuple[Optional[str], Dict]:
        """子查询 → (tool_name, params)"""
        for verb, tool in VERB_TO_TOOL.items():
            if verb in sub_query:
                params: Dict = {}
                if order_id:
                    params["order_id"] = order_id
                elif session_order_id and tool != "query_order":
                    params["order_id"] = session_order_id
                return tool, params
        return None, {}

    def _build_condition(self, condition_text: str) -> str:
        """把自然语言条件转为可执行的条件表达式"""
        # 规则路径只支持简单条件：物流/订单状态类
        if any(kw in condition_text for kw in ["有问题", "异常", "坏了", "破损", "拒收"]):
            return "context.has_problem"
        if any(kw in condition_text for kw in ["已签收", "收到", "完成"]):
            return "context.is_completed"
        return "context.default"  # 默认条件：总是执行


# ---- LLM Prompt ----
PLAN_PROMPT = """你是电商客服 Agent 的规划器。请分析用户请求，决定需要调用哪些工具、按什么顺序执行。

可用工具:
{tools_desc}

规划规则:
1. 每个步骤必须调用上面列出的工具之一
2. 步骤之间有顺序依赖（先查再退）
3. 如果用户提到条件（"如果有问题就...""），把条件写在 description 里
4. 如果会话历史有订单号且用户没提新订单号，可直接用会话订单号

用户请求: {query}
{order_id_hint}

请直接输出 JSON 数组，每个元素是一个步骤:
[{{"tool": "工具名", "args": {{}}, "description": "步骤说明"}}]

如果是闲聊或问政策（不用工具），输出空数组 []
"""
