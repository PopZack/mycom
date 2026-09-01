"""
路由分发器 —— 判断用户意图，分发到对应链路

双路径设计：
  1. LLM 路径（配置 LLM_API_KEY 时）：
     用 LLM 理解语义，输出结构化意图 + 槽位
     准确率高，能处理复杂表达
  2. 规则降级路径（未配置 key 时）：
     关键词匹配 + 正则抽取订单号
     覆盖常见表达，保证无 key 也能跑通

路由策略：
  - 含订单号/物流单号 + 业务动词（查/看/跟踪）→ TICKET
  - 业务动词 + 业务名词（订单/物流/退款）但无单号 → TICKET（引导补全）
  - 问候/感谢/闲聊词 → CHITCHAT
  - 其它 → FAQ（Phase 1 检索兜底）
"""
from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger
from openai import OpenAI

from app.config import settings
from app.agent.intents import Intent, RouteResult


# ---- 规则降级：关键词词典 ----
TICKET_KEYWORDS = {
    "订单", "物流", "快递", "退货", "退款", "换货",
    "发货", "签收", "运费", "取件", "寄件",
    "跟踪", "催单", "取消订单",
}

TICKET_ACTION_VERBS = {
    "查", "查询", "看", "查看", "跟踪", "追踪",
    "取消", "申请", "提交", "催", "帮我", "帮忙",
}

# 疑问词：含这些词的 query 是"问政策"而非"操作业务"，应走 FAQ
FAQ_QUESTION_WORDS = {
    "如何", "怎么", "怎样", "什么是", "什么是", "能不能", "可以吗",
    "是否有", "支持吗", "有哪些", "多久", "几点", "在哪",
}

CHITCHAT_KEYWORDS = {
    "你好", "您好", "hi", "hello", "嗨",
    "谢谢", "感谢", "thanks",
    "再见", "bye", "拜拜",
    "你是谁", "你是机器人", "你是ai",
}

# 订单号正则：支持「订单号/order」后跟数字，或纯数字（6-20位）
ORDER_ID_PATTERN = re.compile(
    r"(?:订单号|订单|单号|order)[号:：\s]*(\d{6,20})|(\d{10,20})",
    re.IGNORECASE,
)


class Router:
    """
    意图路由器

    用法:
        router = Router()
        route = router.route("帮我查一下订单号 123456789 的物流")
        # → RouteResult(intent=Intent.TICKET, slots={"order_id": "123456789"})
    """

    def __init__(self):
        self._client: Optional[OpenAI] = None
        if settings.LLM_API_KEY:
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
            logger.success("[Router] LLM 路由已启用")
        else:
            logger.warning("[Router] LLM_API_KEY 未配置，使用规则降级路由")

    def route(self, query: str) -> RouteResult:
        """主入口：判断意图 + 抽取槽位"""
        if self._client is not None:
            try:
                return self._route_with_llm(query)
            except Exception as e:
                logger.warning(f"[Router] LLM 路由失败，降级为规则: {e}")
        return self._route_with_rules(query)

    # ---- LLM 路径 ----
    def _route_with_llm(self, query: str) -> RouteResult:
        """用 LLM 理解语义，输出结构化意图"""
        prompt = ROUTER_PROMPT.format(query=query)
        resp = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 路由任务用低温度，保证稳定
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        return self._parse_llm_response(raw, query)

    def _parse_llm_response(self, raw: str, query: str) -> RouteResult:
        """解析 LLM 输出的 JSON：{"intent": "ticket", "slots": {"order_id": "..."}}"""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)

        intent_str = data.get("intent", "faq").lower()
        intent = Intent(intent_str) if intent_str in ("faq", "ticket", "chitchat") else Intent.FAQ
        slots = data.get("slots", {}) or {}
        reason = data.get("reason", "")

        logger.info(f"[Router] LLM 路由: '{query}' → {intent.value} (slots={slots})")
        return RouteResult(
            intent=intent,
            confidence=0.9,
            slots=slots,
            source="llm",
            raw_reason=reason,
        )

    # ---- 规则降级路径 ----
    def _route_with_rules(self, query: str) -> RouteResult:
        """关键词 + 正则的规则路由（无 LLM 时降级）"""
        q_lower = query.lower().strip()

        # 1. 闲聊检测
        for kw in CHITCHAT_KEYWORDS:
            if kw in q_lower:
                logger.debug(f"[Router] 规则匹配闲聊: '{kw}'")
                return RouteResult(
                    intent=Intent.CHITCHAT,
                    confidence=0.6,
                    source="rule",
                    raw_reason=f"命中闲聊关键词: {kw}",
                )

        # 2. 工单检测：业务关键词 + 动作动词
        has_ticket_kw = any(kw in query for kw in TICKET_KEYWORDS)
        has_action_verb = any(v in query for v in TICKET_ACTION_VERBS)
        has_question_word = any(w in query for w in FAQ_QUESTION_WORDS)
        order_id = self._extract_order_id(query)

        # 含疑问词（如何/怎么）→ 问政策，走 FAQ（即使含业务关键词）
        if has_question_word:
            logger.debug("[Router] 规则匹配疑问词 → FAQ（问政策）")
            return RouteResult(
                intent=Intent.FAQ,
                confidence=0.7,
                source="rule",
                raw_reason="命中疑问词，判定为问政策",
            )

        if has_ticket_kw and (has_action_verb or order_id):
            slots = {}
            if order_id:
                slots["order_id"] = order_id
            logger.debug(f"[Router] 规则匹配工单 (order_id={order_id})")
            return RouteResult(
                intent=Intent.TICKET,
                confidence=0.6,
                slots=slots,
                source="rule",
                raw_reason="命中业务关键词+动作动词",
            )

        # 3. 默认走 FAQ
        logger.debug("[Router] 规则默认 → FAQ")
        return RouteResult(
            intent=Intent.FAQ,
            confidence=0.5,
            source="rule",
            raw_reason="未命中其它意图，走FAQ检索兜底",
        )

    @staticmethod
    def _extract_order_id(query: str) -> Optional[str]:
        """从用户输入抽取订单号"""
        m = ORDER_ID_PATTERN.search(query)
        if m:
            return m.group(1) or m.group(2)
        return None


# ---- LLM Prompt ----
ROUTER_PROMPT = """你是一个智能客服路由器。请判断用户问题的意图，输出 JSON。

三种意图：
1. "faq"      —— 咨询通用问题（退款政策、营业时间、支付方式等静态知识）
2. "ticket"   —— 查询/操作具体业务（查订单、查物流、申请退款等，通常含订单号或业务动词）
3. "chitchat" —— 闲聊、问候、感谢

输出格式（只输出 JSON，不要其它内容）:
{{"intent": "faq|ticket|chitchat", "slots": {{}}, "reason": "简短理由"}}

slots 抽取规则：
- 如果是 ticket 且用户提到了订单号，抽取 {{"order_id": "订单号"}}
- 其它情况 slots 为空对象 {{}}

用户问题：{query}
"""