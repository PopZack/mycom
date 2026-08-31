"""
BM25 关键词检索 —— 纯 Python 实现，无需 ES

使用 rank-bm25 库，在 FAQ 量级（<1 万条）下足够用。
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple

from loguru import logger
from rank_bm25 import BM25Okapi
import jieba  # 中文分词


def _tokenize(text: str) -> List[str]:
    """中文分词：按词切分 + 转小写"""
    if not text:
        return []
    tokens = list(jieba.cut(text))
    return [t.strip().lower() for t in tokens if t.strip()]


class BM25Retriever:
    """
    轻量 BM25 检索器

    用法：
        retriever = BM25Retriever()
        retriever.build(indexed_docs)          # 用 FAQ 数据构建索引
        results = retriever.search(query, k=5)  # 检索
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._docs: List[Dict[str, Any]] = []  # 原始 FAQ 条目

    def build(self, faq_list: List[Dict[str, Any]]) -> None:
        """
        用 FAQ 列表构建 BM25 索引

        faq_list 格式:
            [
                {"id": 1, "question": "如何退款?", "answer": "...", "category": "订单"},
                ...
            ]
        """
        self._docs = faq_list
        # 对每条 FAQ 的 question + variations + answer 拼接后分词
        # （纳入变体文本，让不同问法的关键词都能命中同一条 FAQ）
        corpus = []
        for item in faq_list:
            parts = [item.get("question", "")]
            parts.extend(item.get("variations", []))
            parts.append(item.get("answer", ""))
            text = " ".join(p for p in parts if p)
            corpus.append(_tokenize(text))

        self._bm25 = BM25Okapi(corpus)
        logger.success(f"[BM25] 索引构建完成，共 {len(faq_list)} 条")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索，返回 top_k 条

        返回格式（Milvus 风格，方便统一处理）:
            [{"id": ..., "question": ..., "answer": ..., "score": ...}, ...]
        """
        if self._bm25 is None:
            logger.warning("[BM25] 索引未构建，返回空结果")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # 取 top_k 索引
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc = self._docs[idx]
            results.append({
                "id": doc.get("id"),
                "question": doc.get("question", ""),
                "answer": doc.get("answer", ""),
                "category": doc.get("category", ""),
                "score": float(score),
            })

        logger.debug(f"[BM25] query='{query}' → {len(results)} 条")
        return results
