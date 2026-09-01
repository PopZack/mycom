"""
业务服务层 —— 封装对外部业务系统的调用

通过 settings.USE_MOCK_DATA 切换：
  - True  → 返回 mock 数据（开发期不依赖真实后端）
  - False → 调用真实业务 API（生产环境）

每个 service 暴露统一接口，对上层 tools 透明。
"""
from app.services.order_service import OrderService, get_order_service

__all__ = ["OrderService", "get_order_service"]
