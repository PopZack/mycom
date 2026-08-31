"""
FAQ 问题变体生成脚本 —— 用 LLM 为每条 FAQ 生成 N 个同义问法

作用：
  一条 FAQ 多个问法 → 多个独立向量 → 显著提升向量召回率
  （业界性价比最高的 FAQ 优化手段，召回率通常提升 30%+）

生成结果直接写回数据文件（写入 item["variations"] 字段），
之后运行 python scripts/sync_faq.py 同步到 Milvus。

用法:
    python scripts/gen_variations.py --dry-run       # 预览生成结果，不写回文件
    python scripts/gen_variations.py --limit 5       # 只处理前 5 条（试跑）
    python scripts/gen_variations.py --force         # 已有变体的也重新生成
    python scripts/gen_variations.py --count 5       # 每条生成 5 个变体（默认 4）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger
from openai import OpenAI

# 确保能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

PROMPT_TEMPLATE = """你是 FAQ 数据增强助手。请为下面的 FAQ 标准问题生成 {n} 个用户可能的其它问法。

要求：
1. 意思完全相同，只是措辞/口语化程度/侧重点不同
2. 不要输出与标准问题完全一样的问法
3. 模拟真实用户的多样化表达（口语、简称、省略主语等自然表达）
4. 只输出 JSON 字符串数组，不要输出任何其它内容
   示例格式: ["问法1", "问法2", "问法3"]

【分类】{category}
【标准问题】{question}
【答案】{answer}
"""


def parse_variations(raw_text: str) -> list[str]:
    """从 LLM 输出解析 JSON 数组（容忍 ```json 代码块包裹）"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    arr = json.loads(text)
    if not isinstance(arr, list):
        raise ValueError(f"期望 JSON 数组，实际: {type(arr)}")
    return [v.strip() for v in arr if isinstance(v, str) and v.strip()]


def gen_one(client: OpenAI, item: dict, n: int) -> list[str]:
    """为一条 FAQ 生成 n 个变体"""
    prompt = PROMPT_TEMPLATE.format(
        n=n,
        category=item.get("category", "general"),
        question=item.get("question", ""),
        answer=item.get("answer", "")[:300],  # 答案太长截断，省 token
    )
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,  # 生成类任务用高温度，问法更多样
        max_tokens=512,
    )
    raw = resp.choices[0].message.content
    variations = parse_variations(raw)
    # 过滤与原问题重复的
    original = item.get("question", "").strip()
    return [v for v in variations if v != original][:n]


def main():
    parser = argparse.ArgumentParser(description="LLM 批量生成 FAQ 问题变体")
    parser.add_argument("--data", default="data/faq_sample.json", help="FAQ 数据路径")
    parser.add_argument("--count", type=int, default=4, help="每条 FAQ 生成几个变体")
    parser.add_argument("--limit", type=int, default=0, help="最多处理几条（0=全部）")
    parser.add_argument("--force", action="store_true", help="已有变体的也重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只预览不写回文件")
    args = parser.parse_args()

    if not settings.LLM_API_KEY:
        logger.error("❌ LLM_API_KEY 未配置，无法生成变体。请在 .env 中配置。")
        sys.exit(1)

    client = OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"FAQ 数据不存在: {data_path}")
        sys.exit(1)
    with open(data_path, "r", encoding="utf-8") as f:
        faq_list = json.load(f)
    logger.info(f"📖 加载 {len(faq_list)} 条 FAQ ← {data_path}")

    changed = 0
    failed = 0
    for item in faq_list:
        if args.limit and changed >= args.limit:
            break
        if item.get("variations") and not args.force:
            logger.debug(f"#{item['id']} 已有变体，跳过（--force 可强制重新生成）")
            continue

        try:
            variations = gen_one(client, item, args.count)
        except Exception as e:
            failed += 1
            logger.warning(f"#{item['id']} 生成失败，跳过: {e}")
            continue

        item["variations"] = variations
        changed += 1
        logger.info(f"#{item['id']} 「{item.get('question', '')[:24]}」→ {len(variations)} 个变体")
        for v in variations:
            logger.info(f"      - {v}")

        time.sleep(0.2)  # 轻微限速，避免触发 API 限流

    logger.info(f"\n📊 汇总: 成功 {changed} 条, 失败 {failed} 条")

    if args.dry_run:
        logger.info("🧪 dry-run 模式，未写回文件")
        return

    if changed == 0:
        logger.info("无变更，不写回文件")
        return

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(faq_list, f, ensure_ascii=False, indent=2)
    logger.success(f"✅ 已写回 {data_path}")
    logger.info("👉 下一步: 运行 python scripts/sync_faq.py 将变体同步到 Milvus")


if __name__ == "__main__":
    main()
