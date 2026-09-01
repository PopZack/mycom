"""
订单业务服务 —— 查订单 / 查物流 / 申请退款

通过 settings.USE_MOCK_DATA 切换 mock 与真实实现：
  - True  → MOCK_ORDERS 内存数据（开发期）
  - False → 调用真实业务 API（生产期，需实现 _call_real_*）

对上层 tools 暴露统一接口，切换实现不影响调用方。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from loguru import logger

from app.config import settings


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
            logger.info("[OrderService] 使用 mock 数据（开发模式）")
        else:
            logger.info("[OrderService] 使用真实业务 API（生产模式）")

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

    # ---- 真实 API 实现（生产环境接入点，当前抛 NotImplementedError）----
    def _call_real_query_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """TODO: 接入真实订单系统 API（如内部 ERP / 订单中心）"""
        raise NotImplementedError("真实订单 API 尚未接入，请在 .env 设置 USE_MOCK_DATA=true")

    def _call_real_query_logistics(self, order_id: str) -> Optional[Dict[str, Any]]:
        """TODO: 接入快递鸟 / 顺丰等物流 API"""
        raise NotImplementedError("真实物流 API 尚未接入，请在 .env 设置 USE_MOCK_DATA=true")

    def _call_real_apply_refund(self, order_id: str, reason: str) -> Dict[str, Any]:
        """TODO: 接入售后工单系统"""
        raise NotImplementedError("真实退款 API 尚未接入，请在 .env 设置 USE_MOCK_DATA=true")


# ---- 单例 ----
_service_singleton: Optional[OrderService] = None


def get_order_service() -> OrderService:
    """获取订单服务单例（全局共享一个 client）"""
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = OrderService()
    return _service_singleton
