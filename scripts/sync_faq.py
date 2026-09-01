"""
FAQ 增量同步脚本（替代旧的全量 build_milvus.py）

职责：
  1. 读取 FAQ 数据文件（source of truth）
  2. 与 Milvus 现有内容做 diff（按 content_hash 对比）
  3. 只同步 新增/变更/删除 的部分，秒级生效，检索服务不中断

数据格式（variations 可选，兼容旧格式）:
    [
        {
            "id": 1,
            "question": "退款多久能到账？",
            "variations": ["钱什么时候回来", "退款几天到账"],   # 可选
            "answer": "1-3 个工作日...",
            "category": "退款售后",
            "status": "published"                              # 可选，非 published 不参与检索
        }
    ]

用法:
    python scripts/sync_faq.py             # 增量同步（只处理 diff）
    python scripts/sync_faq.py --full      # 全量重建（drop 后全部写入）
    python scripts/sync_faq.py --data x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# 必须在 import torch/sentence_transformers 之前设置，限制 PyTorch 虚拟内存预留
# 修复 Windows os error 1455（页面文件太小）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 避免多进程 fork 内存翻倍
os.environ.setdefault("OMP_NUM_THREADS", "2")              # 限制 OpenMP 线程数，减少内存占用
# 使用国内 HuggingFace 镜像，修复连接超时（WinError 10060）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from loguru import logger
from sentence_transformers import SentenceTransformer

# 确保能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retriever.milvus_client import MilvusClientManager, MilvusNotReadyError


def content_hash(item: dict) -> str:
    """对 FAQ 内容算 md5 —— 参与哈希的字段决定'变更'的判定范围"""
    raw = json.dumps(
        {
            "q": item.get("question", ""),
            "v": item.get("variations", []),
            "a": item.get("answer", ""),
            "c": item.get("category", "general"),
            "s": item.get("status", "published"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_target_rows(faq_list: list[dict], model: SentenceTransformer) -> list[dict]:
    """
    把 FAQ 列表展开成 Milvus 行：
      variant_idx=0   → 原始问题
      variant_idx=1..N → 问题变体（每个变体独立向量，指向同一条 answer）
    行 id = faq_id * 100 + variant_idx（确定性 ID，重复同步幂等）
    """
    texts: list[str] = []
    metas: list[dict] = []
    for item in faq_list:
        faq_id = int(item["id"])
        variant_texts = [item.get("question", "")] + list(item.get("variations", []))
        h = content_hash(item)
        for idx, text in enumerate(variant_texts):
            if not text.strip():
                continue
            texts.append(text)
            metas.append({
                "id": faq_id * 100 + idx,
                "faq_id": faq_id,
                "variant_idx": idx,
                "question": text,
                "answer": item.get("answer", ""),
                "category": item.get("category", "general"),
                "status": item.get("status", "published"),
                "content_hash": h,
            })

    logger.info(f"📚 目标状态: {len(faq_list)} 条 FAQ → {len(texts)} 行（含变体）")
    if not texts:
        return []

    # 分批编码，避免一次性占用过多内存（修复 Windows 页面文件不足）
    BATCH_SIZE = 16
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors.extend(vecs)
    if len(texts) > 50:
        logger.info(f"   向量化完成: {len(vectors)} 条")
    for meta, vec in zip(metas, vectors):
        meta["vector"] = vec.tolist()
    return metas


def main():
    parser = argparse.ArgumentParser(description="FAQ 增量同步到 Milvus")
    parser.add_argument("--data", default="data/faq_sample.json", help="FAQ 数据路径")
    parser.add_argument("--full", action="store_true", help="全量重建（drop 后全部写入）")
    args = parser.parse_args()

    # 1. 加载 FAQ 数据
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"FAQ 数据不存在: {data_path}")
        sys.exit(1)
    with open(data_path, "r", encoding="utf-8") as f:
        faq_list = json.load(f)
    logger.info(f"📖 加载 {len(faq_list)} 条 FAQ ← {data_path}")

    # 2. 加载 Embedding 模型
    logger.info(f"🧠 加载 Embedding 模型: {settings.EMBEDDING_MODEL}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # 3. 连接 Milvus
    logger.info("🔌 连接 Milvus...")
    try:
        mgr = MilvusClientManager()
        mgr.ensure_ready()
    except MilvusNotReadyError as e:
        logger.error(f"❌ Milvus 未就绪: {e}")
        logger.error("请先启动 Milvus: docker compose up -d")
        sys.exit(1)

    # 4. 全量模式：直接 drop
    if args.full:
        logger.warning("🧹 --full 已指定，删除旧 collection 全量重建...")
        mgr.drop_collection()

    # 5. 创建 collection（幂等；旧 v1 schema 会自动重建）
    mgr.create_faq_collection(dim=settings.EMBEDDING_DIM)

    # 6. 计算目标状态 + diff
    target_rows = build_target_rows(faq_list, model)
    target_hashes = {r["faq_id"]: r["content_hash"] for r in target_rows}
    current_hashes = {} if args.full else mgr.get_current_hashes()

    to_upsert = [fid for fid, h in target_hashes.items() if current_hashes.get(fid) != h]
    to_delete = [fid for fid in current_hashes if fid not in target_hashes]

    if not to_upsert and not to_delete:
        logger.success("✅ 已是最新，无需同步")
        return

    logger.info(f"🔁 增量同步: 新增/变更 {len(to_upsert)} 条, 删除 {len(to_delete)} 条")

    # 7. 先删后插（变更的 FAQ 变体数量可能变化，先删保证不留脏行）
    mgr.delete_faq_rows(to_delete + to_upsert)

    upsert_set = set(to_upsert)
    rows_to_insert = [r for r in target_rows if r["faq_id"] in upsert_set]
    mgr.insert_rows(rows_to_insert)
    mgr.flush()

    logger.success(
        f"✅ 同步完成: 写入 {len(rows_to_insert)} 行, "
        f"清理 {len(to_delete) + len(to_upsert)} 个 FAQ 的旧行"
    )

    # 8. 简单验证
    logger.info("🔍 验证写入...")
    test_vec = model.encode("如何退款", normalize_embeddings=True).tolist()
    hits = mgr.search(test_vec, top_k=3)
    logger.info(f"   测试查询 '如何退款' → {len(hits)} 条")
    for h in hits:
        logger.info(f"     #{h['id']} {h['question']} (score={h['score']:.4f})")


if __name__ == "__main__":
    main()
