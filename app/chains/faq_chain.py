"""
FAQ 完整链路 —— 把检索 + 答案生成串起来

输入: 用户自然语言问题
输出: 最终答案 + 候选来源 + 调试信息
"""
from __future__ import annotations

import time
from typing import List, Dict, Any

from loguru import logger

from app.retriever.faq_retriever import FAQRetriever
from app.answer_agent import AnswerAgent


class FAQChain:
    """
    FAQ 端到端链路

    用法:
        chain = FAQChain()
        result = chain.run("如何申请退款?")
    """

    def __init__(self, faq_data_path: str = "data/faq_sample.json"):
        logger.info("[FAQChain] 初始化 FAQ 检索器...")
        self._retriever = FAQRetriever(data_path=faq_data_path)
        logger.info("[FAQChain] 初始化 LLM 答案生成器...")
        self._agent = AnswerAgent()
        logger.success("[FAQChain] 就绪 ✓")

    def run(
        self,
        query: str,
        include_debug: bool = True,
    ) -> Dict[str, Any]:
        """
        执行完整的 FAQ 链路

        流程：
          1. 混合检索（Milvus + BM25 + RRF）
          2. LLM 答案生成

        返回:
            {
                "answer": "最终答案",
                "sources": [...],
                "fallback": false,
                "debug": {                     # include_debug=True 时才有
                    "query": "...",
                    "candidates": [...],
                    "retrieval_ms": 123,
                    "answer_ms": 456,
                }
            }
        """
        t0 = time.time()

        # Step 1: 混合检索
        candidates = self._retriever.retrieve(query)
        t_retrieval = int((time.time() - t0) * 1000)

        # Step 2: LLM 答案生成
        t1 = time.time()
        result = self._agent.generate(query, candidates)
        t_answer = int((time.time() - t1) * 1000)

        output = {
            "answer": result["answer"],
            "sources": result["sources"],
            "fallback": result.get("fallback", False),
        }
        if result.get("note"):
            output["note"] = result["note"]

        if include_debug:
            output["debug"] = {
                "query": query,
                "candidates": [
                    {
                        "id": c.get("id"),
                        "question": c.get("question"),
                        "category": c.get("category"),
                        "rrf_score": c.get("rrf_score"),
                    }
                    for c in candidates
                ],
                "retrieval_ms": t_retrieval,
                "answer_ms": t_answer,
                "total_ms": t_retrieval + t_answer,
            }

        logger.info(
            f"[FAQChain] '{query}' → {t_retrieval}ms检索 + {t_answer}ms生成"
        )
        return output
