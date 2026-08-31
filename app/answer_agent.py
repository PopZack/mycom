"""
答案生成 Agent —— 用 LLM 把检索到的 FAQ 候选组织成最终答案

策略：
  - 把检索到的 FAQ 候选作为 context
  - 让 LLM 基于 context 回答用户问题
  - 如果候选里没有相关的，让 LLM 诚实说"我暂时没找到"
  - 要求 LLM 引用来源（FAQ 的 id 或 question），方便追溯
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from loguru import logger
from openai import OpenAI

from app.config import settings

SYSTEM_PROMPT = """你是一个专业的智能客服助手。请严格基于提供的【FAQ 知识库】内容回答用户问题。

规则：
1. 优先使用 FAQ 知识库中的条目回答，回答要简洁准确
2. 回答时请引用来源（格式：[来源：FAQ #id]）
3. 如果 FAQ 知识库中没有相关内容，请诚实说："抱歉，我暂时没找到相关信息，请您换个方式描述或稍后再试。" 不要编造答案
4. 如果用户问题模糊，可主动建议他们尝试的关键词
5. 使用中文回答
"""


class AnswerAgent:
    """
    LLM 答案生成器

    用法:
        agent = AnswerAgent()
        answer = agent.generate(user_query, faq_candidates)
    """

    def __init__(self):
        if not settings.LLM_API_KEY:
            logger.warning("[AnswerAgent] LLM_API_KEY 未配置，将返回降级答案")
            self._client = None
        else:
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
            logger.success(f"[AnswerAgent] LLM 客户端初始化完成 → {settings.LLM_MODEL}")

    def _format_context(self, candidates: List[Dict[str, Any]]) -> str:
        """把 FAQ 候选格式化成 LLM 容易理解的 context"""
        lines = []
        for item in candidates:
            faq_id = item.get("id", "?")
            q = item.get("question", "")
            a = item.get("answer", "")
            cat = item.get("category", "")
            lines.append(f"[FAQ #{faq_id}] [{cat}] Q: {q}\nA: {a}")
        return "\n\n".join(lines)

    def generate(
        self,
        user_query: str,
        faq_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        基于 FAQ 候选 + 用户问题，生成最终答案

        返回:
            {"answer": "...", "sources": [...], "fallback": false}
        """
        context = self._format_context(faq_candidates)

        # 降级：没有 LLM 时，直接把 top1 FAQ 的 answer 返回
        if self._client is None:
            logger.warning("[AnswerAgent] LLM 不可用，返回降级答案")
            if faq_candidates:
                top = faq_candidates[0]
                return {
                    "answer": top.get("answer", ""),
                    "sources": [{"id": top.get("id"), "question": top.get("question")}],
                    "fallback": True,
                    "note": "LLM 未配置，直接返回最高相关 FAQ。配置 LLM_API_KEY 后可获得更好效果。",
                }
            return {
                "answer": "抱歉，我暂时没找到相关信息。",
                "sources": [],
                "fallback": True,
            }

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"【用户问题】\n{user_query}\n\n【FAQ 知识库】\n{context}",
                },
            ]

            response = self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content.strip()

            # 收集来源
            sources = [
                {"id": c.get("id"), "question": c.get("question")}
                for c in faq_candidates
            ]

            logger.info(f"[AnswerAgent] 生成答案 ({len(answer)} chars)")
            return {
                "answer": answer,
                "sources": sources,
                "fallback": False,
            }

        except Exception as e:
            logger.error(f"[AnswerAgent] LLM 调用失败: {e}")
            # LLM 也挂了 → 最终降级
            if faq_candidates:
                top = faq_candidates[0]
                return {
                    "answer": top.get("answer", ""),
                    "sources": [{"id": top.get("id"), "question": top.get("question")}],
                    "fallback": True,
                    "note": f"LLM 暂不可用({e})，直接返回最高相关 FAQ。",
                }
            return {
                "answer": "抱歉，我暂时没找到相关信息。",
                "sources": [],
                "fallback": True,
            }
