"""
混合检索融合 —— RRF (Reciprocal Rank Fusion)

RRF 公式: score(d) = Σ 1/(k + rank_i(d))
  - rank_i(d): 文档 d 在第 i 个检索器中的排名（从 1 开始）
  - k: 平滑参数，推荐 60（Milvus 官方默认值）

优点：不依赖检索器的绝对分数，只看相对排名，天然适合异构检索融合。
"""
from __future__ import annotations

from typing import List, Dict, Any

from loguru import logger

from app.config import settings


def rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int | None = None,
    final_top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    RRF 融合多路检索结果

    参数:
        ranked_lists: 多路检索结果，每路都是按分数降序排列的列表
        k: RRF 平滑参数，默认 settings.RRF_K (60)
        final_top_k: 融合后最终保留条数，默认 settings.FINAL_TOP_K

    返回:
        融合后的列表，按 RRF 分数降序排列。每条多一个 "rrf_score" 字段。

    用法示例:
        vector_results = milvus.search(query_vec, top_k=8)
        bm25_results   = bm25.search(query, top_k=8)
        fused = rrf_fuse([vector_results, bm25_results], final_top_k=5)
    """
    k = k or settings.RRF_K
    final_top_k = final_top_k or settings.FINAL_TOP_K

    # 用 id 作为融合键（如果没有 id 则用 question）
    def _key(doc: Dict[str, Any]) -> str:
        doc_id = doc.get("id")
        if doc_id is not None:
            return str(doc_id)
        return doc.get("question", "")

    # 累加 RRF 分数
    fusion_scores: Dict[str, float] = {}
    doc_by_key: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _key(doc)
            rrf = 1.0 / (k + rank)
            fusion_scores[key] = fusion_scores.get(key, 0.0) + rrf
            if key not in doc_by_key:
                doc_by_key[key] = doc

    # 排序取 top_k
    sorted_keys = sorted(fusion_scores.keys(), key=lambda k: fusion_scores[k], reverse=True)
    results = []
    for key in sorted_keys[:final_top_k]:
        doc = dict(doc_by_key[key])  # 拷贝
        doc["rrf_score"] = round(fusion_scores[key], 6)
        results.append(doc)

    logger.debug(
        f"[RRF] {len(ranked_lists)} 路融合 → {len(results)} 条 (k={k})"
    )
    return results
