"""
统一配置管理 —— 从 .env 读取，带类型校验
"""
import os

# 必须在 import 任何第三方 ML 库之前设置环境变量
# 使用国内 HuggingFace 镜像，修复连接超时（WinError 10060）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Milvus
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "faq_collection"
    MILVUS_TIMEOUT: int = 10

    # Embedding
    EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"
    EMBEDDING_DIM: int = 768

    # LLM
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.3

    # 检索参数
    VECTOR_TOP_K: int = 8
    BM25_TOP_K: int = 8
    FINAL_TOP_K: int = 5
    RRF_K: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
