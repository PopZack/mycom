"""
会话记忆存储 —— 多轮对话的槽位继承（Phase 2.2-C）

职责：
  1. 按 session_id 保存对话上下文（槽位、上一轮意图、待补槽位）
  2. TTL 过期自动清理（默认 30 分钟无活动即失效）
  3. 线程安全（FastAPI 同步端点跑在线程池里，可能并发访问）

企业级演进路径：内存 dict → Redis（接口保持一致，换实现即可）
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger

SESSION_TTL_SECONDS = 30 * 60  # 30 分钟无活动过期


@dataclass
class SessionContext:
    """单个会话的上下文"""
    session_id: str
    slots: Dict[str, Any] = field(default_factory=dict)  # 已确认的槽位（如 order_id）
    last_intent: str = ""                                # 上一轮意图（faq/ticket/chitchat）
    pending_slot: Optional[str] = None                   # 等待用户补全的槽位（如 order_id）
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.updated_at > SESSION_TTL_SECONDS

    def clear(self) -> None:
        """清空全部记忆（话题切换/闲聊时）"""
        self.slots.clear()
        self.last_intent = ""
        self.pending_slot = None


class SessionStore:
    """会话存储（内存版：TTL + 线程安全）"""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> Optional[SessionContext]:
        """读取会话，过期则删除并返回 None"""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is None:
                return None
            if ctx.is_expired():
                del self._sessions[session_id]
                logger.debug(f"[SessionStore] 会话 {session_id} 已过期，清除")
                return None
            return ctx

    def save(self, ctx: SessionContext) -> None:
        """保存会话，顺带清理其它过期会话"""
        ctx.touch()
        with self._lock:
            self._sessions[ctx.session_id] = ctx
            self._purge_expired_locked()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _purge_expired_locked(self) -> None:
        expired = [sid for sid, c in self._sessions.items() if c.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"[SessionStore] 清理 {len(expired)} 个过期会话")


# ---- 全局单例 ----
_store_singleton: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = SessionStore()
    return _store_singleton
