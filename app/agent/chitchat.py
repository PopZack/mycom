"""
闲聊链路 —— 处理问候、感谢、告别等非业务意图

无 LLM 时也能工作：内置常见回复模板
有 LLM 时：用 LLM 生成更自然的回复
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from loguru import logger
from openai import OpenAI

from app.config import settings

# 规则降级：关键词 → 回复模板
CHITCHAT_TEMPLATES: Dict[str, str] = {
    "你好": "您好！我是智能客服助手，有什么可以帮您的吗？😊",
    "您好": "您好！我是智能客服助手，有什么可以帮您的吗？😊",
    "hi": "Hi！我是智能客服助手，请问有什么可以帮您？",
    "hello": "Hello！我是智能客服助手，请问有什么可以帮您？",
    "嗨": "嗨！我是智能客服助手，有什么可以帮您的吗？",
    "谢谢": "不客气！很高兴能帮到您，还有其它问题吗？",
    "感谢": "不客气！很高兴能帮到您，还有其它问题吗？",
    "thanks": "You're welcome! 还有其它可以帮您的吗？",
    "再见": "再见！祝您生活愉快，有问题随时找我～",
    "bye": "Bye！有问题随时回来找我～",
    "拜拜": "拜拜！祝您一切顺利～",
    "你是谁": "我是智能客服助手，可以帮您查询订单、物流，回答常见问题。",
    "你是机器人": "是的，我是智能客服助手，7x24 小时为您服务。",
    "你是ai": "是的，我是 AI 智能客服助手，有什么可以帮您？",
}

CHITCHAT_SYSTEM_PROMPT = """你是一个友好的智能客服助手。用户正在和你闲聊（问候、感谢、告别等）。
请用简短、自然、友好的语气回复（1-2句话）。使用中文。不要编造业务信息。"""


class ChitchatChain:
    """闲聊链路"""

    def __init__(self):
        self._client: Optional[OpenAI] = None
        if settings.LLM_API_KEY:
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )

    def run(self, query: str) -> Dict[str, Any]:
        """处理闲聊意图"""
        # 优先匹配模板（即使有 LLM 也先用模板，更快更稳定）
        q_lower = query.lower().strip()
        for kw, reply in CHITCHAT_TEMPLATES.items():
            if kw in q_lower:
                logger.debug(f"[Chitchat] 模板匹配: '{kw}'")
                return {
                    "answer": reply,
                    "sources": [],
                    "fallback": False,
                    "intent": "chitchat",
                }

        # 有 LLM → 生成自然回复
        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.7,
                    max_tokens=128,
                )
                answer = resp.choices[0].message.content.strip()
                return {
                    "answer": answer,
                    "sources": [],
                    "fallback": False,
                    "intent": "chitchat",
                }
            except Exception as e:
                logger.warning(f"[Chitchat] LLM 调用失败: {e}")

        # 兜底
        return {
            "answer": "您好！我是智能客服助手。您可以问我订单、物流、退款等问题，也可以直接描述您的需求。",
            "sources": [],
            "fallback": True,
            "intent": "chitchat",
        }