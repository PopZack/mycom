"""
订单业务服务 —— 查订单 / 查物流 / 申请退款

通过 settings.USE_MOCK_DATA 切换 mock 与真实实现：
  - True  → MOCK_ORDERS 内存数据（开发期）
  - False → 调用真实业务 API（生产期）

对上层 tools 暴露统一接口，切换实现不影响调用方。

真实 API 契约（与 scripts/mock_business_api.py 桩服务一致）:
  GET  {base}/orders/{order_id}              → 200 订单JSON | 404 不存在
  GET  {base}/orders/{order_id}/logistics    → 200 物流JSON | 404 不存在
  POST {base}/refunds  {"order_id","reason"} → 200 {"success": bool, "message": str}
  认证: Authorization: Bearer {ORDER_API_KEY}（未配置 key 时省略）

容错（与 milvus_client 约定一致）:
  - 超时 10s，重试 3 次，指数退避 1s→2s→4s
  - 404 → 返回 None（订单不存在，上层生成友好提示）
  - 重试耗尽仍失败 → 抛 BusinessAPIError，由 ToolRegistry 捕获转为失败 ToolResult
"""
from __future__ import annotations

import time
from typing import Dict, Any, Optional

import httpx
from loguru import logger

from app.config import settings


class BusinessAPIError(Exception):
    """业务 API 调用失败（网络错误/超时/5xx，重试后仍失败）"""


# ---- Mock 业务数据（开发期使用，与 ticket.py Phase 2.1 保持一致）----
MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "123456789": {
        "order_id": "123456789",
        "status": "已发货",
        "product": "无线蓝牙耳机",
        "amount": 299.00,
        "logistics": "顺丰快递 SF1234567890",
        "logistics_status": "运输中，预计明天送达",
        "created_at": "2026-08-28 14:30",
    },
    "987654321": {
        "order_id": "987654321",
        "status": "待发货",
        "product": "智能手表",
        "amount": 899.00,
        "logistics": None,
        "logistics_status": "尚未发货，预计 1-2 天内发出",
        "created_at": "2026-08-30 09:15",
    },
    "555666777": {
        "order_id": "555666777",
        "status": "已完成",
        "product": "充电宝 20000mAh",
        "amount": 159.00,
        "logistics": "京东快递 JD55566677701",
        "logistics_status": "已签收",
        "created_at": "2026-08-20 16:45",
    },
}


class OrderService:
    """
    订单业务服务

    用法:
        svc = get_order_service()
        order = svc.query_order("123456789")
    """

    def __init__(self, use_mock: Optional[bool] = None):
        self._use_mock = settings.USE_MOCK_DATA if use_mock is None else use_mock
        if self._use_mock:
            self._http: Optional[httpx.Client] = None
            logger.info("[OrderService] 使用 mock 数据（开发模式）")
        else:
            headers = {}
            if settings.ORDER_API_KEY:
                headers["Authorization"] = f"Bearer {settings.ORDER_API_KEY}"
            self._http = httpx.Client(
                base_url=settings.ORDER_API_BASE_URL,
                timeout=settings.ORDER_API_TIMEOUT,
                headers=headers,
            )
            logger.info(f"[OrderService] 使用真实业务 API: {settings.ORDER_API_BASE_URL}")

    # ---- 统一对外接口 ----
    def query_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """查订单详情（含基础信息 + 物流字段）"""
        if self._use_mock:
            return self._mock_query_order(order_id)
        return self._call_real_query_order(order_id)

    def query_logistics(self, order_id: str) -> Optional[Dict[str, Any]]:
        """查物流状态"""
        if self._use_mock:
            return self._mock_query_logistics(order_id)
        return self._call_real_query_logistics(order_id)

    def apply_refund(self, order_id: str, reason: str = "") -> Dict[str, Any]:
        """申请退款（返回操作结果）"""
        if self._use_mock:
            return self._mock_apply_refund(order_id, reason)
        return self._call_real_apply_refund(order_id, reason)

    # ---- Mock 实现 ----
    def _mock_query_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            logger.debug(f"[OrderService:mock] 订单未找到: {order_id}")
        return order

    def _mock_query_logistics(self, order_id: str) -> Optional[Dict[str, Any]]:
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return None
        return {
            "order_id": order_id,
            "product": order["product"],
            "status": order["status"],
            "logistics": order.get("logistics", "无"),
            "logistics_status": order.get("logistics_status", "未知"),
        }

    def _mock_apply_refund(self, order_id: str, reason: str) -> Dict[str, Any]:
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return {"success": False, "message": f"订单 {order_id} 不存在"}
        status = order["status"]
        # 简单业务规则：已发货需先拒收，已签收走退货，待发货可直接退
        if status == "待发货":
            return {"success": True, "message": f"订单 {order_id} 已发起退款，1-3 工作日原路退回"}
        if status == "已发货":
            return {
                "success": False,
                "message": f"订单 {order_id} 已发货，需先拒收快递后才能退款",
            }
        if status == "已完成":
            return {
                "success": False,
                "message": f"订单 {order_id} 已签收，请走「7天无理由退货」流程",
            }
        return {"success": False, "message": "当前订单状态不支持退款"}

    # ---- 真实 API 实现 ----
    def _request_with_retry(
        self, method: str, path: str, *, json_body: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        """带重试的 HTTP 请求（指数退避 1s→2s→4s）

        返回:
            2xx → 响应 JSON dict
            404 → None（资源不存在）
        异常:
            BusinessAPIError: 重试耗尽仍失败
        """
        assert self._http is not None, "真实 API 模式未初始化 http 客户端"
        max_retries = settings.ORDER_API_MAX_RETRIES
        last_err: str = ""

        for attempt in range(max_retries):
            backoff = 2 ** attempt  # 1s, 2s, 4s
            try:
                logger.info(
                    f"[OrderService] {method} {path} 尝试 {attempt+1}/{max_retries}"
                )
                resp = self._http.request(method, path, json=json_body)
                if resp.status_code == 404:
                    logger.debug(f"[OrderService] {path} → 404 资源不存在")
                    return None
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning(f"[OrderService] 尝试 {attempt+1} 失败: {last_err}")
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(f"[OrderService] 尝试 {attempt+1} 失败: {last_err}")

            if attempt < max_retries - 1:
                logger.info(f"[OrderService] {backoff}s 后重试...")
                time.sleep(backoff)

        raise BusinessAPIError(
            f"业务 API 调用失败（已重试 {max_retries} 次）: {last_err}"
        )

    def _call_real_query_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """GET /orders/{order_id}"""
        return self._request_with_retry("GET", f"/orders/{order_id}")

    def _call_real_query_logistics(self, order_id: str) -> Optional[Dict[str, Any]]:
        """GET /orders/{order_id}/logistics"""
        return self._request_with_retry("GET", f"/orders/{order_id}/logistics")

    def _call_real_apply_refund(self, order_id: str, reason: str) -> Dict[str, Any]:
        """POST /refunds {"order_id","reason"}"""
        data = self._request_with_retry(
            "POST", "/refunds", json_body={"order_id": order_id, "reason": reason}
        )
        return data if data else {"success": False, "message": "退款服务返回空响应"}


# ---- 单例 ----
_service_singleton: Optional[OrderService] = None


def get_order_service() -> OrderService:
    """获取订单服务单例（全局共享一个 client）"""
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = OrderService()
    return _service_singleton
