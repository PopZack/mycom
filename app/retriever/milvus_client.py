"""
Milvus 客户端封装 —— 核心容错层（Schema v2：问题变体多向量）

设计要点：
  1. 连接超时：10s（防 Proxy 未就绪时无限阻塞）
  2. 重试次数：3 次，指数退避（1s → 2s → 4s）
  3. 失败后抛出明确的 MilvusNotReadyError，由上层决定降级或快速返回

Schema v2（对比 v1 的关键变化）：
  - 每条 FAQ 的 原始问题 + N 个问题变体 各占一行，共享同一个 faq_id
    → 变体各自算向量，显著提升向量召回率
  - 行主键 id = faq_id * 100 + variant_idx（确定性 ID，支持增量同步）
  - 新增标量字段: faq_id / variant_idx / status / content_hash
  - content_hash 用于增量同步 diff：内容变没变，对比哈希即可
  - 向量索引从 IVF_FLAT 升级为 HNSW（FAQ 场景读多写少，HNSW 查询更快更准）
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
        mgr.insert_rows(...)         # 批量写入
        results = mgr.search(...)    # 向量检索
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
    def create_faq_collection(
        self,
        dim: Optional[int] = None,
        force_recreate: bool = False,
    ) -> None:
        """
        创建 FAQ collection v2（幂等）

        - 已存在但缺少 faq_id 字段（v1 旧 schema）→ 自动删除重建
        - force_recreate=True → 强制删除重建
        """
        dim = dim or settings.EMBEDDING_DIM
        client = self.ensure_ready()
        col_name = settings.MILVUS_COLLECTION

        if client.has_collection(col_name):
            need_rebuild = force_recreate
            if not need_rebuild:
                # 检测旧 schema：v1 没有 faq_id 字段
                info = client.describe_collection(col_name)
                field_names = {f["name"] for f in info.get("fields", [])}
                if "faq_id" not in field_names:
                    logger.warning("[Milvus] 检测到 v1 旧 schema（缺 faq_id 字段），自动重建为 v2")
                    need_rebuild = True
            if need_rebuild:
                client.drop_collection(col_name)
                logger.warning(f"[Milvus] collection '{col_name}' 已删除（重建）")

        if client.has_collection(col_name):
            logger.info(f"[Milvus] collection '{col_name}' 已存在，跳过创建")
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="faq_id", dtype=DataType.INT64),
                FieldSchema(name="variant_idx", dtype=DataType.INT16),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200, default_value="general"),
                FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32, default_value="published"),
                FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ],
            description="FAQ 向量知识库 v2（问题变体多向量）",
        )

        client.create_collection(
            collection_name=col_name,
            schema=schema,
        )

        # HNSW 索引：查询性能与召回率优于 IVF_FLAT，适合读多写少的 FAQ 场景
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            index_name="vector_index",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_index(
            collection_name=col_name,
            index_params=index_params,
        )
        logger.success(f"[Milvus] collection '{col_name}' 创建完成 (dim={dim}, HNSW) ✓")

    # ---- 增量同步支撑 ----
    def get_current_hashes(self) -> Dict[int, str]:
        """
        读取 Milvus 中现有全部 FAQ 的 {faq_id: content_hash}

        同一 faq_id 的所有变体行哈希相同，取任意一行即可。
        """
        client = self.ensure_ready()
        col_name = settings.MILVUS_COLLECTION
        if not client.has_collection(col_name):
            return {}

        rows = client.query(
            collection_name=col_name,
            filter="faq_id >= 0",
            output_fields=["faq_id", "content_hash"],
            limit=16384,
        )
        out: Dict[int, str] = {}
        for r in rows:
            out[int(r["faq_id"])] = r.get("content_hash", "")
        logger.debug(f"[Milvus] 当前库内 {len(out)} 条 FAQ")
        return out

    def delete_faq_rows(self, faq_ids: List[int]) -> None:
        """删除一组 faq_id 的所有变体行（变更时先删后插，防止旧变体残留）"""
        if not faq_ids:
            return
        client = self.ensure_ready()
        client.delete(
            collection_name=settings.MILVUS_COLLECTION,
            filter=f"faq_id in {faq_ids}",
        )
        logger.info(f"[Milvus] 删除 {len(faq_ids)} 个 FAQ 的全部变体行")

    def insert_rows(
        self,
        rows: List[Dict[str, Any]],
        batch_size: int = 200,
    ) -> int:
        """批量写入（自动分批，避免单次 payload 过大）"""
        if not rows:
            return 0
        client = self.ensure_ready()
        total = 0
        batches = (len(rows) + batch_size - 1) // batch_size
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            client.insert(collection_name=settings.MILVUS_COLLECTION, data=batch)
            total += len(batch)
            logger.debug(f"[Milvus] 批次 {i // batch_size + 1}/{batches} 写入 {len(batch)} 行")
        logger.info(f"[Milvus] 共插入 {total} 行（{batches} 批）")
        return total

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
        """
        向量相似度检索（只返回 status=published 的行）

        注意：一条 FAQ 的多个变体行可能同时命中。
        返回的 "id" 已映射回 faq_id（同一 FAQ 的变体共享 id），
        上游 RRF 按 id 融合时自然去重。
        """
        client = self.ensure_ready()
        top_k = top_k or settings.VECTOR_TOP_K

        conditions = ['status == "published"']
        if category:
            conditions.append(f'category == "{category}"')
        filter_expr = " and ".join(conditions)

        results = client.search(
            collection_name=settings.MILVUS_COLLECTION,
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            filter=filter_expr,
            output_fields=["faq_id", "question", "answer", "category"],
            timeout=settings.MILVUS_TIMEOUT,
        )

        hits = results[0] if results else []
        out = []
        for hit in hits:
            entity = hit.get("entity", {})
            out.append({
                "id": entity.get("faq_id", hit["id"]),  # 映射回 faq_id
                "row_id": hit["id"],                     # 实际命中的变体行 id
                "question": entity.get("question", ""),  # 命中的变体/原始问题文本
                "answer": entity.get("answer", ""),
                "category": entity.get("category", ""),
                "score": hit.get("distance", hit.get("score", 0.0)),
            })
        logger.debug(f"[Milvus] search 返回 {len(out)} 行")
        return out

    def drop_collection(self) -> None:
        client = self.ensure_ready()
        col_name = settings.MILVUS_COLLECTION
        if client.has_collection(col_name):
            client.drop_collection(col_name)
            logger.warning(f"[Milvus] collection '{col_name}' 已删除")


def get_milvus() -> MilvusClientManager:
    return MilvusClientManager()
