"""
Executor —— 计划执行引擎（Phase 3 Agent 自主规划）

设计要点（来自经验教训）:
  - while+游标（非 for-each）: 运行时重规划可真正生效，不会固化迭代对象
  - 结果归一化: ToolResult(success,data,error) → {status,data,error}，异构工具统一
  - step_id 稳定: 重规划时复用原 step_id，retry 计数 key 不丢失
  - 重规划后重置 idx: 确保新计划从正确位置继续执行

执行流程:
  while idx < len(steps):
    step = steps[idx]
    if step.condition and not evaluate(step.condition):
      step.status = SKIPPED
      idx += 1; continue
    if not step.can_run(): idx += 1; continue
    step.status = RUNNING
    result = registry.call(step.tool_name, resolve_placeholders(step.params, plan.context))
    normalized = normalize(result)
    step.result = normalized
    step.status = SUCCESS if normalized.status == "success" else FAILED
    if FAILED:
      if step.retry_count < step.max_retries: step.retry_count += 1; step.status = PENDING; continue
      if can_replan(plan, step): trigger_replan(plan); continue
      plan.mark_failed(); return
    # 写入 context 供后续步骤引用
    merge_into_context(plan.context, step)
    idx += 1
  plan.mark_success()
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.planning.plan import (
    Plan, PlanStep, StepStatus, PlanStatus,
)
from app.tools.base import ToolResult
from app.tools.registry import get_tool_registry


# ---- 占位符模式: {{prev.order_id}} {{s1.data.xxx}} ----
PLACEHOLDER_RE = re.compile(r"\{\{([\w\.]+)\}\}")


class Executor:
    """
    计划执行引擎

    用法:
        executor = Executor()
        result = executor.execute(plan)
    """

    def __init__(self):
        self._registry = get_tool_registry()

    def execute(self, plan: Plan) -> Dict[str, Any]:
        """执行计划，返回聚合结果

        返回结构:
            {
                "success": bool,
                "plan_id": str,
                "answer": str,           # 聚合所有步骤结果的自然语言回复
                "step_results": [...],   # 每步的归一化结果
                "final_status": str,     # PlanStatus 值
                "context": dict,         # 最终 context
            }
        """
        if not plan.steps:
            # 空计划（没有业务工具可调用）
            logger.info("[Executor] 空计划，跳过执行")
            return {
                "success": True,
                "plan_id": plan.plan_id,
                "answer": "",
                "step_results": [],
                "final_status": "skipped",
                "context": plan.context,
            }

        plan.mark_running()
        idx = 0
        max_steps = len(plan.steps) * 3  # 防止无限循环（重规划会增加步骤）
        total_iters = 0

        while idx < len(plan.steps) and total_iters < max_steps:
            total_iters += 1
            step = plan.steps[idx]

            if step.status == StepStatus.SUCCESS or step.status == StepStatus.SKIPPED:
                idx += 1
                continue

            if step.status == StepStatus.RUNNING:
                # 理论上不会进入这里（同一 idx 不会两次执行）
                idx += 1
                continue

            # 1. 条件评估
            if step.condition and not self._evaluate_condition(step.condition, plan.context):
                logger.info(f"[Executor] step {step.step_id} 条件未满足，跳过")
                step.status = StepStatus.SKIPPED
                idx += 1
                continue

            # 2. 执行步骤
            logger.info(f"[Executor] 执行 step {step.step_id}: {step.tool_name}({step.params})")
            step.status = StepStatus.RUNNING

            # 解析占位符参数
            resolved_params = self._resolve_params(step.params, plan.context)

            # 调用工具
            tool_result = self._registry.call(step.tool_name, resolved_params)

            # 3. 结果归一化
            normalized = self._normalize(tool_result, step)
            step.result = normalized

            if normalized["status"] == "success":
                step.status = StepStatus.SUCCESS
                # 合并到 context
                self._merge_context(plan, step)
                logger.info(f"[Executor] step {step.step_id} ✅ {step.tool_name} 成功")
                idx += 1
            else:
                step.error = normalized.get("error", "unknown")
                # 4. 重试 / 重规划决策
                if step.retry_count < step.max_retries:
                    step.retry_count += 1
                    step.status = StepStatus.PENDING
                    logger.warning(
                        f"[Executor] step {step.step_id} ❌ 失败，"
                        f"重试 {step.retry_count}/{step.max_retries}"
                    )
                    # 不移动 idx，继续重试本步骤
                    continue

                # 尝试重规划：如果是查询失败（如订单不存在），后续步骤可能也无法执行
                if self._try_replan(plan, idx, step):
                    # 重规划后重置 idx（新步骤可能插在当前位置）
                    idx = max(0, idx - 1)
                    logger.info("[Executor] 重规划成功，从新 idx 继续")
                    continue

                # 无法重规划 → 标记失败并终止
                step.status = StepStatus.FAILED
                logger.error(f"[Executor] step {step.step_id} ❌ 失败且无法重规划，终止计划")
                plan.mark_failed()
                break

        # 循环结束：检查最终状态
        if plan.all_success():
            plan.mark_success()
        elif not plan.has_failed() and plan.status == PlanStatus.RUNNING:
            # 可能有 SKIPPED 步骤但没失败
            plan.mark_success()

        # 聚合结果
        step_results = [self._step_summary(s) for s in plan.steps]
        answer = self._aggregate_answer(plan)

        return {
            "success": plan.status in (PlanStatus.SUCCESS, PlanStatus.PAUSED),
            "plan_id": plan.plan_id,
            "answer": answer,
            "step_results": step_results,
            "final_status": plan.status.value,
            "context": plan.context,
            "total_steps": plan.total_steps,
            "executed_steps": sum(
                1 for s in plan.steps
                if s.status in (StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED)
            ),
        }

    # ---- 结果归一化 ----
    @staticmethod
    def _normalize(tool_result: ToolResult, step: PlanStep) -> Dict[str, Any]:
        """把 ToolResult 归一化为统一的 {status, data, error} 协议

        设计: 执行器边界做归一化，不把异构协议分散到各工具实现
        """
        if tool_result.success:
            return {
                "status": "success",
                "data": tool_result.data,
                "message": tool_result.message,
                "error": "",
            }
        else:
            return {
                "status": "failed",
                "data": {},
                "message": tool_result.message,
                "error": tool_result.error or "execution_failed",
            }

    # ---- 参数占位符解析 ----
    @staticmethod
    def _resolve_params(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """把 {{prev.order_id}} 等占位符替换为 context 中的实际值"""
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str):
                resolved[k] = Executor._resolve_string(v, context)
            elif isinstance(v, dict):
                resolved[k] = Executor._resolve_params(v, context)
            else:
                resolved[k] = v
        return resolved

    @staticmethod
    def _resolve_string(template: str, context: Dict[str, Any]) -> Any:
        """解析单个字符串中的所有占位符"""
        def _get_value(path: str) -> Any:
            # path 格式: prev.order_id / s1.data.product / context.xxx
            parts = path.split(".")
            obj = context
            for p in parts:
                if isinstance(obj, dict) and p in obj:
                    obj = obj[p]
                else:
                    return None
            return obj

        # 全是占位符 → 返回实际值（可能是 int）
        full_match = re.fullmatch(r"\{\{([\w\.]+)\}\}", template.strip())
        if full_match:
            return _get_value(full_match.group(1))

        # 混排 → 逐个替换
        def _replace(m: re.Match) -> str:
            val = _get_value(m.group(1))
            return str(val) if val is not None else m.group(0)

        return PLACEHOLDER_RE.sub(_replace, template)

    # ---- 条件评估 ----
    @staticmethod
    def _evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
        """规则路径的条件表达式求值"""
        try:
            # 支持的条件格式（规则路径，安全子集）
            if condition == "context.default":
                return True
            # context.has_problem / context.is_completed / context.xxx
            if condition.startswith("context."):
                key = condition[len("context."):]
                val = context.get(key)
                return bool(val)
            # 支持简单布尔表达式
            if condition in ("true", "True", "1"):
                return True
            if condition in ("false", "False", "0"):
                return False
            return True  # 默认条件：总是执行
        except Exception as e:
            logger.warning(f"[Executor] 条件解析失败 {condition}: {e}")
            return True

    # ---- Context 合并 ----
    @staticmethod
    def _merge_context(plan: Plan, step: PlanStep) -> None:
        """把步骤结果合并到 plan.context，供后续步骤引用"""
        if not step.result or step.result["status"] != "success":
            return

        data = step.result.get("data", {})
        msg = step.result.get("message", "")

        # 顶层快捷引用
        plan.context[step.step_id] = step.result
        plan.context[f"{step.step_id}.data"] = data
        plan.context[f"{step.step_id}.message"] = msg

        # 工具特定字段快捷引用
        if step.tool_name == "query_order":
            plan.context.setdefault("prev", {})
            if "order_id" in data:
                plan.context["prev"]["order_id"] = data["order_id"]
                plan.context["order_id"] = data["order_id"]
            if "status" in data:
                plan.context["prev"]["status"] = data["status"]
                # 推断 has_problem
                status_val = str(data["status"])
                plan.context["has_problem"] = any(
                    kw in status_val for kw in ["异常", "破损", "拒收", "取消"]
                )
                plan.context["is_completed"] = any(
                    kw in status_val for kw in ["已签收", "完成", "已收到"]
                )

        elif step.tool_name == "query_logistics":
            plan.context.setdefault("prev", {})
            if "order_id" in data:
                plan.context["prev"]["order_id"] = data["order_id"]
                plan.context["order_id"] = data["order_id"]
            # 从物流 message 推断
            if "异常" in msg or "破损" in msg:
                plan.context["has_problem"] = True
            if "已签收" in msg or "已收到" in msg:
                plan.context["is_completed"] = True

        elif step.tool_name == "apply_refund":
            plan.context.setdefault("prev", {})
            if "order_id" in data:
                plan.context["prev"]["order_id"] = data["order_id"]

    # ---- 重规划 ----
    def _try_replan(self, plan: Plan, idx: int, failed_step: PlanStep) -> bool:
        """尝试重规划：根据失败类型调整后续步骤

        返回 True 表示重规划成功（plan 已修改），False 表示无法重规划

        策略:
          1. 订单不存在 → 删除所有依赖此订单的后续步骤（标记 SKIPPED）
          2. 查询工具失败 → 跳过此步，继续后续（如果后续不依赖它）
          3. 退款失败（订单状态不允许）→ 跳过退款步骤
        """
        err = failed_step.error.lower()

        # 策略 1: 订单不存在 → 后续步骤依赖订单号，跳过
        if "not_found" in err or "not exist" in err or "不存在" in err:
            logger.warning(f"[Executor] 订单不存在，后续步骤标记 SKIPPED")
            for step in plan.steps[idx + 1:]:
                if step.status == StepStatus.PENDING:
                    step.status = StepStatus.SKIPPED
                    step.description += " [因前置查询失败被跳过]"
            return True

        # 策略 2: 查询工具失败 → 跳过本步，继续后续（重置 retry 计数）
        if failed_step.tool_name in ("query_order", "query_logistics"):
            logger.warning(f"[Executor] 查询工具失败，跳过继续后续步骤")
            failed_step.status = StepStatus.SKIPPED
            failed_step.description += " [查询失败被跳过]"
            # 后续步骤如果依赖 context（如 {{prev.order_id}}）也要跳过
            for step in plan.steps[idx + 1:]:
                if step.status == StepStatus.PENDING:
                    step.status = StepStatus.SKIPPED
                    step.description += " [前置查询失败被跳过]"
            return True

        # 策略 3: 退款失败（如订单状态不允许）→ 跳过退款
        if failed_step.tool_name == "apply_refund":
            logger.warning(f"[Executor] 退款失败，无法重规划")
            return False

        return False

    # ---- 辅助方法 ----
    @staticmethod
    def _step_summary(step: PlanStep) -> Dict[str, Any]:
        """步骤摘要（用于对外展示）"""
        return {
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "description": step.description,
            "status": step.status.value,
            "error": step.error,
            "message": (step.result or {}).get("message", ""),
        }

    @staticmethod
    def _aggregate_answer(plan: Plan) -> str:
        """聚合所有成功步骤的 message 为自然语言回复"""
        messages: List[str] = []
        for step in plan.steps:
            if step.status == StepStatus.SUCCESS and step.result:
                msg = step.result.get("message", "")
                if msg:
                    messages.append(msg)
            elif step.status == StepStatus.SKIPPED:
                logger.info(f"[Executor] step {step.step_id} 被跳过: {step.description}")

        if not messages:
            if plan.status == PlanStatus.FAILED:
                return "抱歉，查询过程中遇到问题，请稍后再试或联系客服。"
            return ""

        # 多步骤聚合：简单拼接（Phase 3.1 先用拼接，后续用 LLM 润色）
        return "\n\n".join(messages)
