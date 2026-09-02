"""
AnswerAgent —— 答案生成 Agent（框架图最顶层：整合信息 → 组织语言 → 最终答案）

双路径设计（和 Router/Planner 一致）:
  1. LLM 路径（配置 LLM_API_KEY 时）：让 LLM 把多步骤结果润色成自然语言
  2. 规则模板路径（无 Key 时）：按工具类型 + 状态组合，用模板拼接

消费入口:
  - Phase 2: ToolResult（单工具执行结果）
  - Phase 3: List[step_summary]（多步骤执行结果，含 success/rejected/skipped）

设计原则:
  - 不改变任何现有接口，只替换 answer 文本的生成方式
  - 模板保持简洁，避免引入新的依赖
  - 有 LLM 时自动升级到 LLM 路径
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import settings


# ============================================================
# 规则模板：按工具类型 + 状态组合
# ============================================================

# query_order 成功 → 订单详情
_TPL_ORDER_SUCCESS = (
    "您的订单 {order_id}：\n"
    "  商品：{product}\n"
    "  金额：¥{amount}\n"
    "  状态：{status}\n"
    "  下单时间：{order_time}"
)

# query_logistics 成功 → 物流详情
_TPL_LOGISTICS_SUCCESS = (
    "{order_id} 物流信息：\n"
    "  快递：{courier} {tracking_no}\n"
    "  状态：{logistics_status}"
)

# apply_refund 成功 → 退款已发起
_TPL_REFUND_SUCCESS = "✅ {message}"

# apply_refund 被业务拒绝 → 引导话术
_TPL_REFUND_REJECTED = "⚠️ {message}\n\n建议：{advice}"

# 业务拒绝的推荐建议（按订单状态）
_REFUND_ADVICE = {
    "已发货": "可以拒收快递后再申请退款，或签收后走「7天无理由退货」流程。",
    "已完成": "订单已签收，请走「7天无理由退货」流程（需退回商品后退款）。",
    "不存在": "请确认订单号是否正确，或联系客服协助查询。",
}

# ============================================================
# 字段名映射：工具返回 → 模板期望
# ============================================================

_FIELD_MAP_ORDER = {
    "created_at": "order_time",   # mock 用 created_at，模板用 order_time
    "amount": "amount",
}

_FIELD_MAP_LOGISTICS = {
    # logistics: "顺丰快递 SF1234567890" → 拆分 courier + tracking_no
    "logistics_status": "status",
}


def _normalize_order_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """把 query_order 返回的 data 归一化为模板期望的字段"""
    normalized = dict(data)
    if "created_at" in normalized and "order_time" not in normalized:
        normalized["order_time"] = normalized["created_at"]
    # amount 可能是 float → 格式化
    if "amount" in normalized and isinstance(normalized["amount"], (int, float)):
        normalized["amount"] = f"{normalized['amount']:.2f}"
    return normalized


def _normalize_logistics_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """把 query_logistics 返回的 data 归一化为模板期望的字段"""
    normalized = dict(data)
    # logistics: "顺丰快递 SF1234567890" → courier + tracking_no
    logistics_str = normalized.get("logistics", "")
    if logistics_str and "courier" not in normalized:
        parts = logistics_str.split(" ", 1)
        if len(parts) == 2:
            normalized["courier"] = parts[0]
            normalized["tracking_no"] = parts[1]
        else:
            normalized["courier"] = logistics_str
            normalized["tracking_no"] = ""
    # logistics_status 优先覆盖 status（mock data 里 status 是订单状态，logistics_status 才是物流状态）
    if "logistics_status" in normalized:
        normalized["status"] = normalized["logistics_status"]
    return normalized


class AnswerAgent:
    """答案生成 Agent —— 把工具执行结果整合成自然语言回复"""

    def __init__(self, client=None):
        """
        参数:
            client: OpenAI 客户端（可选，有 Key 时传入）
        """
        self._client = client
        self._has_llm = client is not None and settings.LLM_API_KEY

        if self._has_llm:
            logger.success("[AnswerAgent] LLM 路径已启用")
        else:
            logger.info("[AnswerAgent] 规则模板路径（无 LLM，用模板拼接）")

    # ============================================================
    # Phase 2 入口：单工具结果
    # ============================================================

    def compose_single(
        self,
        tool_name: str,
        data: Dict[str, Any],
        success: bool,
        message: str = "",
        error: str = "",
    ) -> str:
        """格式化单个工具的执行结果（Phase 2 TicketChain 用）

        参数:
            tool_name: 工具名（query_order / query_logistics / apply_refund）
            data: 工具返回的数据 dict
            success: 是否成功
            message: 工具返回的 message
            error: 业务拒绝时的 message（同 message）或执行错误
        """
        if self._has_llm:
            return self._compose_with_llm(
                tool_name=tool_name, data=data, success=success, message=message, is_multi_step=False
            )
        return self._compose_single_with_rules(tool_name, data, success, message, error)

    def _compose_single_with_rules(
        self,
        tool_name: str,
        data: Dict[str, Any],
        success: bool,
        message: str,
        error: str,
    ) -> str:
        """规则路径：按工具类型选模板"""
        if not success:
            # 业务拒绝或执行失败
            if error and "已发货" in error:
                advice = _REFUND_ADVICE.get("已发货", "")
                return _TPL_REFUND_REJECTED.format(message=error, advice=advice)
            if error and "已签收" in error:
                advice = _REFUND_ADVICE.get("已完成", "")
                return _TPL_REFUND_REJECTED.format(message=error, advice=advice)
            if error and "不存在" in error:
                advice = _REFUND_ADVICE.get("不存在", "")
                return _TPL_REFUND_REJECTED.format(message=error, advice=advice)
            if error:
                return f"⚠️ {error}"
            return message or "抱歉，操作未成功。"

        # 成功 → 按工具类型格式化（先做字段归一化）
        if tool_name == "query_order":
            d = _normalize_order_data(data)
            return _TPL_ORDER_SUCCESS.format(
                order_id=d.get("order_id", ""),
                product=d.get("product", ""),
                amount=d.get("amount", ""),
                status=d.get("status", ""),
                order_time=d.get("order_time", ""),
            )
        if tool_name == "query_logistics":
            d = _normalize_logistics_data(data)
            return _TPL_LOGISTICS_SUCCESS.format(
                order_id=d.get("order_id", ""),
                courier=d.get("courier", ""),
                tracking_no=d.get("tracking_no", ""),
                logistics_status=d.get("status", ""),
            )
        if tool_name == "apply_refund":
            return _TPL_REFUND_SUCCESS.format(message=message or "退款已发起，请留意后续通知。")

        # 兜底：返回原始 message
        return message

    # ============================================================
    # Phase 3 入口：多步骤结果
    # ============================================================

    def compose_multi(
        self,
        step_results: List[Dict[str, Any]],
        original_query: str,
    ) -> str:
        """格式化多步骤执行结果（Phase 3 Executor 用）

        step_results 每个元素结构:
            {
                "tool_name": "query_order",
                "status": "success" | "rejected" | "skipped" | "failed",
                "message": "...",       # 成功时的 message
                "error": "...",         # 业务拒绝时的错误说明
                "result": {...},        # 归一化后的完整结果（含 data）
            }
        """
        if self._has_llm:
            return self._compose_with_llm(
                step_results=step_results,
                original_query=original_query,
                is_multi_step=True,
            )
        return self._compose_multi_with_rules(step_results)

    def _compose_multi_with_rules(self, step_results: List[Dict[str, Any]]) -> str:
        """规则路径：按步骤顺序拼接，加逻辑分隔"""
        sections: List[str] = []

        for step in step_results:
            tool = step.get("tool_name", "")
            status = step.get("status", "")
            message = step.get("message", "")
            error = step.get("error", "")
            result_data = (step.get("result") or {}).get("data", {})

            if status == "success":
                if tool == "query_order":
                    d = _normalize_order_data(result_data)
                    sections.append(_TPL_ORDER_SUCCESS.format(
                        order_id=d.get("order_id", ""),
                        product=d.get("product", ""),
                        amount=d.get("amount", ""),
                        status=d.get("status", ""),
                        order_time=d.get("order_time", ""),
                    ))
                elif tool == "query_logistics":
                    d = _normalize_logistics_data(result_data)
                    sections.append(_TPL_LOGISTICS_SUCCESS.format(
                        order_id=d.get("order_id", ""),
                        courier=d.get("courier", ""),
                        tracking_no=d.get("tracking_no", ""),
                        logistics_status=d.get("status", ""),
                    ))
                elif tool == "apply_refund":
                    sections.append(_TPL_REFUND_SUCCESS.format(
                        message=message or "退款已发起，请留意后续通知。"
                    ))
                else:
                    sections.append(message or "")

            elif status == "rejected":
                # 业务拒绝 → 按工具类型给引导
                if tool == "apply_refund":
                    advice = ""
                    if "已发货" in error:
                        advice = _REFUND_ADVICE.get("已发货", "")
                    elif "已签收" in error or "已完成" in error:
                        advice = _REFUND_ADVICE.get("已完成", "")
                    elif "不存在" in error:
                        advice = _REFUND_ADVICE.get("不存在", "")

                    if advice:
                        sections.append(f"⚠️ {error}\n建议：{advice}")
                    else:
                        sections.append(f"⚠️ {error}")
                else:
                    sections.append(f"⚠️ {error}")

            # skipped / failed → 不输出（用户不关心内部跳过逻辑）

        if not sections:
            return ""

        # 多步骤加串联逻辑词
        if len(sections) == 1:
            return sections[0]

        # 区分查询结果 vs 操作结果
        query_part = [s for s in step_results if s.get("tool_name") in ("query_order", "query_logistics")]
        other_part = [s for s in step_results if s.get("tool_name") not in ("query_order", "query_logistics")]

        result_sections: List[str] = []
        if query_part and other_part:
            for s in query_part:
                result_sections.append(self._format_single_step(s))
            for s in other_part:
                formatted = self._format_single_step(s)
                if formatted:
                    tool = s.get("tool_name", "")
                    if tool == "apply_refund":
                        result_sections.append(f"【退款】{formatted}")
                    else:
                        result_sections.append(formatted)
        else:
            result_sections = sections

        return "\n\n".join(result_sections)

    def _format_single_step(self, step: Dict[str, Any]) -> str:
        """格式化单个 step_summary（多步骤场景下的单步格式化）"""
        tool = step.get("tool_name", "")
        status = step.get("status", "")
        message = step.get("message", "")
        error = step.get("error", "")
        result_data = (step.get("result") or {}).get("data", {})

        if status == "success":
            if tool == "query_order":
                d = _normalize_order_data(result_data)
                return _TPL_ORDER_SUCCESS.format(
                    order_id=d.get("order_id", ""),
                    product=d.get("product", ""),
                    amount=d.get("amount", ""),
                    status=d.get("status", ""),
                    order_time=d.get("order_time", ""),
                )
            if tool == "query_logistics":
                d = _normalize_logistics_data(result_data)
                return _TPL_LOGISTICS_SUCCESS.format(
                    order_id=d.get("order_id", ""),
                    courier=d.get("courier", ""),
                    tracking_no=d.get("tracking_no", ""),
                    logistics_status=d.get("status", ""),
                )
            if tool == "apply_refund":
                return _TPL_REFUND_SUCCESS.format(message=message or "退款已发起")
            return message or ""

        if status == "rejected":
            return f"⚠️ {error}"

        return ""

    # ============================================================
    # LLM 路径（预留，有 Key 时自动启用）
    # ============================================================

    def _compose_with_llm(
        self,
        *,
        tool_name: str = "",
        data: Dict[str, Any] | None = None,
        success: bool = True,
        message: str = "",
        error: str = "",
        step_results: List[Dict[str, Any]] | None = None,
        original_query: str = "",
        is_multi_step: bool = False,
    ) -> str:
        """LLM 润色：把执行结果变成自然语言

        这个方法在无 Key 时不会被调用（__init__ 里 has_llm=False），
        有 Key 时自动走这里。目前先返回规则路径结果作为兜底，
        后续完善 LLM prompt。
        """
        # 兜底：还是走规则模板
        if is_multi_step and step_results:
            return self._compose_multi_with_rules(step_results)
        return self._compose_single_with_rules(
            tool_name, data or {}, success, message, error
        )
