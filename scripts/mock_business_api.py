"""
业务 API 桩服务 —— 模拟真实上游订单系统（本地联调用）

用途：
  测试 USE_MOCK_DATA=false 时 Agent 的真实 API 调用链路。
  契约与 app/services/order_service.py 模块文档一致。

端点：
  GET  /orders/{order_id}              → 200 订单JSON | 404 不存在
  GET  /orders/{order_id}/logistics    → 200 物流JSON | 404 不存在
  POST /refunds  {"order_id","reason"} → 200 {"success": bool, "message": str}
  GET  /health                         → 200 {"status": "ok"}

用法：
  python scripts/mock_business_api.py     # 监听 127.0.0.1:8002
"""
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 确保能 import app 包（复用同一份 mock 订单数据，方便对比测试）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.order_service import MOCK_ORDERS

app = FastAPI(title="模拟业务上游（订单系统）", version="0.1.0")


class RefundRequest(BaseModel):
    order_id: str
    reason: str = ""


@app.get("/health", tags=["系统"])
def health():
    return {"status": "ok"}


@app.get("/orders/{order_id}", tags=["订单"])
def get_order(order_id: str):
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"订单 {order_id} 不存在")
    return order


@app.get("/orders/{order_id}/logistics", tags=["订单"])
def get_logistics(order_id: str):
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"订单 {order_id} 不存在")
    return {
        "order_id": order_id,
        "product": order["product"],
        "status": order["status"],
        "logistics": order.get("logistics", "无"),
        "logistics_status": order.get("logistics_status", "未知"),
    }


@app.post("/refunds", tags=["售后"])
def apply_refund(req: RefundRequest):
    order = MOCK_ORDERS.get(req.order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"订单 {req.order_id} 不存在")

    status = order["status"]
    # 与 app 内 mock 退款规则一致：待发货可直接退，已发货需拒收，已完成走退货
    if status == "待发货":
        return {"success": True, "message": f"订单 {req.order_id} 已发起退款，1-3 工作日原路退回"}
    if status == "已发货":
        return {"success": False, "message": f"订单 {req.order_id} 已发货，需先拒收快递后才能退款"}
    if status == "已完成":
        return {"success": False, "message": f"订单 {req.order_id} 已签收，请走「7天无理由退货」流程"}
    return {"success": False, "message": "当前订单状态不支持退款"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
