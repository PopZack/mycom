"""
FastAPI 应用入口

端点:
  GET  /health       —— 健康检查（Milvus 连通性）
  POST /chat         —— FAQ 问答主入口
  GET  /candidates   —— 仅返回检索候选（调试用，不调 LLM）

启动:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from app.config import settings
from app.retriever.faq_retriever import FAQRetriever
from app.retriever.milvus_client import MilvusClientManager, MilvusNotReadyError
from app.chains.faq_chain import FAQChain
from app.agent.agent_chain import AgentChain
from app.agent.router import Router


# ---- 全局单例（启动时初始化一次）----
_retriever: FAQRetriever | None = None
_faq_chain: FAQChain | None = None
_agent_chain: AgentChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时懒加载核心组件"""
    global _retriever, _faq_chain, _agent_chain
    logger.info("=" * 50)
    logger.info("🚀 mycom FAQ 智能问答服务启动中...")
    logger.info(f"   Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    logger.info(f"   LLM:    {settings.LLM_MODEL}")
    logger.info(f"   Embedding: {settings.EMBEDDING_MODEL}")
    logger.info("=" * 50)

    try:
        _agent_chain = AgentChain()
        logger.success("✅ 服务就绪 ✓")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        # 不阻塞启动，/chat 会返回明确错误

    yield

    logger.info("👋 服务关闭")


app = FastAPI(
    title="mycom FAQ 智能问答",
    description="Phase 1 — 静态 FAQ 知识库问答服务（向量 + BM25 混合检索）",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS（方便前端调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Schema ----
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1, max_length=500)
    include_debug: bool = Field(True, description="是否返回调试信息")


class SourceItem(BaseModel):
    id: Optional[int] = None
    question: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    fallback: bool = False
    intent: str = "faq"
    note: Optional[str] = None
    need_slot: Optional[str] = None
    debug: Optional[dict] = None


# ---- Endpoints ----
@app.get("/health", tags=["系统"])
def health():
    """健康检查"""
    milvus_ok = False
    milvus_msg = "未检查"
    try:
        MilvusClientManager().ensure_ready()
        milvus_ok = True
        milvus_msg = "OK"
    except MilvusNotReadyError as e:
        milvus_msg = str(e)

    return {
        "status": "ok" if _agent_chain is not None else "starting",
        "milvus": milvus_ok,
        "milvus_detail": milvus_msg,
        "llm_configured": bool(settings.LLM_API_KEY),
        "phase": "Phase 2 - Agent Router",
        "embedding_model": settings.EMBEDDING_MODEL,
        "collection": settings.MILVUS_COLLECTION,
    }


@app.post("/chat", response_model=ChatResponse, tags=["问答"])
def chat(req: ChatRequest):
    """
    FAQ 问答主入口（Phase 2: Agent 路由 → FAQ/工单/闲聊）

    示例:
        curl -X POST http://localhost:8000/chat \
          -H "Content-Type: application/json" \
          -d '{"query": "如何退款?", "include_debug": true}'
    """
    if _agent_chain is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪，请稍后再试")

    result = _agent_chain.run(req.query, include_debug=req.include_debug)

    response = ChatResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) if isinstance(s, dict) and "question" in s
                 else SourceItem(question=str(s.get("product", s.get("order_id", ""))))
                 for s in result.get("sources", [])],
        fallback=result.get("fallback", False),
        intent=result.get("intent", "faq"),
        note=result.get("note"),
        need_slot=result.get("need_slot"),
        debug=result.get("debug"),
    )
    return response


@app.get("/candidates", tags=["调试"])
def candidates(query: str, top_k: int = 5):
    """
    仅返回检索候选（不调 LLM），用于调试检索效果

    示例:
        curl "http://localhost:8000/candidates?query=如何退款&top_k=3"
    """
    if _retriever is None:
        # AgentChain 初始化时会同时初始化 FAQ 链路的 retriever
        if _agent_chain is None:
            raise HTTPException(status_code=503, detail="服务尚未就绪")
        retriever = _agent_chain._faq._retriever
    else:
        retriever = _retriever

    results = retriever.retrieve(query, final_top_k=top_k)
    return {"query": query, "candidates": results}


@app.get("/", tags=["系统"])
def root():
    return {
        "name": "mycom 智能问答",
        "version": "0.2.0",
        "phase": "Phase 2 - Agent Router",
        "docs": "/docs",
        "health": "/health",
        "routes": ["/chat", "/candidates", "/route"],
    }


@app.get("/route", tags=["调试"])
def route_query(query: str):
    """
    调试端点：查看路由结果（不执行链路，只看意图判断）

    示例:
        curl "http://localhost:8000/route?query=帮我查一下订单123456789"
    """
    if _agent_chain is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    route = _agent_chain._router.route(query)
    return {"query": query, "route": route.to_dict()}
