"""
Milvus 客户端封装 —— 核心容错层

设计要点：
  1. 连接超时：10s（防 Proxy 未就绪时无限阻塞）
  2. 重试次数：3 次，指数退避（1s → 2s → 4s）
  3. 失败后抛出明确的 MilvusNotReadyError，由上层决定降级或快速返回
"""
from __future__ import annotations

import time
import asyncio
from typing import Optional, List, Dict, Any

from loguru import logger
from pymilvus import (
    MilvusClient as PyMilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

from app.config import settings


class MilvusNotReadyError(Exception):
    """Milvus 未就绪/连接失败的明确异常"""


class MilvusClientManager:
    """
    Milvus 连接管理器 —— 单例，带超时 + 有限重试

    用法：
        mgr = MilvusClientManager()
        mgr.ensure_ready()           # 启动时或首次调用前检查
        mgr.insert(...)              # 写数据
        results = mgr.search(...)    # 搜数据
    """

    _instance: Optional["MilvusClientManager"] = None
    _client: Optional[PyMilvusClient] = None
    _ready: bool = False

    def __new__(cls) -> "MilvusClientManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _connect_with_retry(self) -> PyMilvusClient:
        """连接 Milvus，最多重试 3 次（指数退避）"""
        uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        last_err: Optional[Exception] = None

        for attempt in range(3):
            backoff = 2 ** attempt  # 1s, 2s, 4s
            try:
                logger.info(f"[Milvus] 连接尝试 {attempt+1}/3 → {uri}")
                client = PyMilvusClient(uri=uri, timeout=settings.MILVUS_TIMEOUT)
                # 真正触发 Proxy 检查
                client.list_collections(timeout=settings.MILVUS_TIMEOUT)
                self._client = client
                self._ready = True
                logger.success("[Milvus] 连接成功 ✓")
                return client
            except Exception as e:
                last_err = e
                logger.warning(f"[Milvus] 尝试 {attempt+1} 失败: {e}")
                if attempt < 2:
                    logger.info(f"[Milvus] {backoff}s 后重试...")
                    time.sleep(backoff)

        raise MilvusNotReadyError(
            f"Milvus 连接失败（3 次重试耗尽）: {last_err}"
        )

    def ensure_ready(self) -> PyMilvusClient:
        if self._ready and self._client is not None:
            return self._client
        return self._connect_with_retry()

    async def aensure_ready(self) -> PyMilvusClient:
        """异步包装：同步连接逻辑放到线程池"""
        return await asyncio.to_thread(self.ensure_ready)

    # ---- Collection 管理 ----
    def create_faq_collection(self, dim: Optional[int] = None) -> None:
        """创建 FAQ collection（幂等：已存在则跳过）"""
        dim = dim or settings.EMBEDDING_DIM
        client = self.ensure_ready()
        col_name = settings.MILVUS_COLLECTION

        if client.has_collection(col_name):
            logger.info(f"[Milvus] collection '{col_name}' 已存在，跳过创建")
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200, default_value="general"),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ],
            description="FAQ 向量知识库",
        )

        client.create_collection(
            collection_name=col_name,
            schema=schema,
        )

        # IVF_FLAT 索引（FAQ 量级足够用）
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            index_name="vector_index",
            params={"nlist": 128},
        )
        client.create_index(
            collection_name=col_name,
            index_params=index_params,
        )
        logger.success(f"[Milvus] collection '{col_name}' 创建完成 (dim={dim}) ✓")

    # ---- 写入 ----
    def insert(self, data: List[Dict[str, Any]]) -> int:
        """批量插入 FAQ 数据"""
        client = self.ensure_ready()
        result = client.insert(collection_name=settings.MILVUS_COLLECTION, data=data)
        logger.info(f"[Milvus] 插入 {len(data)} 条 → {result}")
        return len(data)

    def flush(self) -> None:
        client = self.ensure_ready()
        client.flush(settings.MILVUS_COLLECTION)

    # ---- 检索 ----
    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """向量相似度检索"""
        client = self.ensure_ready()
        top_k = top_k or settings.VECTOR_TOP_K

        filter_expr = f'category == "{category}"' if category else None

        results = client.search(
            collection_name=settings.MILVUS_COLLECTION,
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            filter=filter_expr,
            output_fields=["question", "answer", "category"],
            timeout=settings.MILVUS_TIMEOUT,
        )

        hits = results[0] if results else []
        out = []
        for hit in hits:
            entity = hit.get("entity", {})
            out.append({
                "id": hit["id"],
                "question": entity.get("question", ""),
                "answer": entity.get("answer", ""),
                "category": entity.get("category", ""),
                "score": hit.get("distance", hit.get("score", 0.0)),
            })
        logger.debug(f"[Milvus] search 返回 {len(out)} 条")
        return out

    def drop_collection(self) -> None:
        client = self.ensure_ready()
        col_name = settings.MILVUS_COLLECTION
        if client.has_collection(col_name):
            client.drop_collection(col_name)
            logger.warning(f"[Milvus] collection '{col_name}' 已删除")


def get_milvus() -> MilvusClientManager:
    return MilvusClientManager()
