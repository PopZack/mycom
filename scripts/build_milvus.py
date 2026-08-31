"""
Milvus 建索引脚本

把 data/faq_sample.json 里的 FAQ 写入 Milvus 向量库。

用法:
    python scripts/build_milvus.py           # 正常执行
    python scripts/build_milvus.py --drop     # 先清空 collection 再重建
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from sentence_transformers import SentenceTransformer

# 确保能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retriever.milvus_client import MilvusClientManager, MilvusNotReadyError


def main():
    parser = argparse.ArgumentParser(description="构建 Milvus FAQ 向量索引")
    parser.add_argument("--data", default="data/faq_sample.json", help="FAQ 数据路径")
    parser.add_argument("--drop", action="store_true", help="先删除已有 collection 再重建")
    args = parser.parse_args()

    # 1. 加载 FAQ 数据
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"FAQ 数据不存在: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        faq_list = json.load(f)
    logger.info(f"📚 加载 {len(faq_list)} 条 FAQ")

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

    # 4. 可选：清空重建
    if args.drop:
        logger.warning("🧹 --drop 已指定，删除旧 collection...")
        mgr.drop_collection()

    # 5. 创建 collection（幂等）
    mgr.create_faq_collection(dim=settings.EMBEDDING_DIM)

    # 6. 向量化 + 写入
    logger.info("📝 向量化并写入 Milvus...")
    rows = []
    for item in faq_list:
        text = item.get("question", "") + " " + item.get("answer", "")
        vec = model.encode(text, normalize_embeddings=True).tolist()
        rows.append({
            "id": item["id"],
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "category": item.get("category", "general"),
            "vector": vec,
        })

    mgr.insert(rows)
    mgr.flush()
    logger.success(f"✅ 完成！{len(rows)} 条 FAQ 已写入 Milvus")

    # 7. 简单验证
    logger.info("🔍 验证写入...")
    test_vec = model.encode("如何退款", normalize_embeddings=True).tolist()
    hits = mgr.search(test_vec, top_k=3)
    logger.info(f"   测试查询 '如何退款' → {len(hits)} 条")
    for h in hits:
        logger.info(f"     #{h['id']} {h['question']} (score={h['score']:.4f})")


if __name__ == "__main__":
    main()
