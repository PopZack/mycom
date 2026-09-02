"""
FAQ 检索器 —— 整合 Milvus 向量检索 + BM25 关键词检索 + RRF 融合

这是 Phase 1 的核心检索组件。
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from loguru import logger

# 必须先 import config（它会设置 HF_ENDPOINT 等环境变量），再 import sentence_transformers
from app.config import settings
from sentence_transformers import SentenceTransformer

from app.retriever.milvus_client import MilvusClientManager, MilvusNotReadyError
from app.retriever.bm25_retriever import BM25Retriever
from app.retriever.hybrid_search import rrf_fuse


class FAQRetriever:
    """
    FAQ 混合检索器

    初始化时会：
      1. 加载本地 Embedding 模型（首次运行会下载 ~40MB）
      2. 尝试连接 Milvus（失败不阻塞，降级为仅 BM25）
      3. 加载 FAQ 数据，构建 BM25 索引

    用法:
        retriever = FAQRetriever(data_path="data/faq_sample.json")
        results = retriever.retrieve("如何退款?")
    """

    def __init__(
        self,
        data_path: str = "data/faq_sample.json",
        milvus_collection: Optional[str] = None,
    ):
        self.data_path = data_path
        self._milvus_ready = False
        self._milvus = MilvusClientManager()
        self._bm25 = BM25Retriever()

        # 1. 加载 Embedding 模型（强制 CPU，避免 CUDA 初始化占用额外内存触发 os error 1455）
        logger.info(f"[FAQ] 加载 Embedding 模型: {settings.EMBEDDING_MODEL}")
        self._embed_model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu",
        )
        logger.success("[FAQ] Embedding 模型加载完成 ✓")

        # 2. 加载 FAQ 数据
        faq_data = self._load_faq_data(data_path)

        # 3. 构建 BM25 索引（内存中，随时可用）
        self._bm25.build(faq_data)

        # 4. 尝试连接 Milvus（降级友好）
        try:
            self._milvus.ensure_ready()
            self._milvus_ready = True
            logger.success("[FAQ] Milvus 可用，将执行混合检索 ✓")
        except MilvusNotReadyError as e:
            logger.warning(f"[FAQ] Milvus 不可用: {e}")
            logger.warning("[FAQ] → 降级为仅 BM25 关键词检索")
            self._milvus_ready = False

    def _load_faq_data(self, path: str) -> List[Dict[str, Any]]:
        """从 JSON 加载 FAQ 数据"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[FAQ] 加载 {len(data)} 条 FAQ 数据 ← {path}")
            return data
        except FileNotFoundError:
            logger.error(f"[FAQ] FAQ 数据文件不存在: {path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"[FAQ] FAQ 数据文件 JSON 解析失败: {e}")
            return []

    def _embed(self, text: str) -> List[float]:
        """文本 → 向量"""
        vec = self._embed_model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def retrieve(
        self,
        query: str,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        FAQ 混合检索入口

        流程：
          1. BM25 关键词检索
          2. [可选] Milvus 向量检索（不可用时跳过）
          3. RRF 融合 → 返回最终 top_k

        返回格式:
            [{"id": ..., "question": ..., "answer": ..., "category": ..., "rrf_score": ...}, ...]
        """
        vector_top_k = vector_top_k or settings.VECTOR_TOP_K
        bm25_top_k = bm25_top_k or settings.BM25_TOP_K
        final_top_k = final_top_k or settings.FINAL_TOP_K

        bm25_results = self._bm25.search(query, top_k=bm25_top_k)

        ranked_lists = [bm25_results]

        if self._milvus_ready:
            try:
                query_vec = self._embed(query)
                vector_results = self._milvus.search(
                    query_vec, top_k=vector_top_k
                )
                ranked_lists.append(vector_results)
            except Exception as e:
                logger.warning(f"[FAQ] Milvus 检索异常，跳过: {e}")

        # RRF 融合
        fused = rrf_fuse(ranked_lists, final_top_k=final_top_k)
        logger.info(f"[FAQ] query='{query}' → 返回 {len(fused)} 条候选")
        return fused
