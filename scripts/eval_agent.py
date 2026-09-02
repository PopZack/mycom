"""
Agent 评测脚本（Phase 2.2-D）

评测维度：
  1. 意图准确率：FAQ / Ticket / Chitchat 分类是否正确
  2. 工具选择准确率：Ticket 子意图 → 工具映射是否正确
  3. 槽位抽取准确率：订单号是否正确抽取
  4. 多轮会话场景：槽位补全 / 继承 / 清空 / 隔离

用法:
    # 先启动服务
    python -m uvicorn app.main:app --port 8001

    # 跑评测
    python scripts/eval_agent.py

    # 指定地址
    python scripts/eval_agent.py --base-url http://127.0.0.1:8001
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ---- 评测集 ----

@dataclass
class EvalCase:
    """单条评测用例"""
    id: str
    query: str
    expect_intent: str                          # faq / ticket / chitchat
    expect_tool: Optional[str] = None           # query_order / query_logistics / apply_refund / None
    expect_order_id: Optional[str] = None       # 期望抽取的订单号
    expect_need_slot: Optional[str] = None      # 期望引导补全的槽位
    note: str = ""


# 单轮评测集（30 条，覆盖三种意图 + 边界 case）
SINGLE_TURN_CASES: list[EvalCase] = [
    # ---- Chitchat（6 条）----
    EvalCase("c1",  "你好",           "chitchat", note="基础问候"),
    EvalCase("c2",  "您好啊",         "chitchat", note="礼貌问候"),
    EvalCase("c3",  "hi",             "chitchat", note="英文问候"),
    EvalCase("c4",  "谢谢你的帮助",   "chitchat", note="感谢"),
    EvalCase("c5",  "拜拜",           "chitchat", note="告别"),
    EvalCase("c6",  "你是谁呀",       "chitchat", note="身份询问"),

    # ---- FAQ（12 条，含疑问词优先 + 边界）----
    EvalCase("f1",  "如何申请退款？",            "faq", note="疑问词+退款关键词→FAQ"),
    EvalCase("f2",  "退款多久能到账？",          "faq", note="疑问词+退款关键词→FAQ"),
    EvalCase("f3",  "怎么查快递到哪了",          "faq", note="疑问词+物流关键词→FAQ"),
    EvalCase("f4",  "忘记密码怎么办",            "faq", note="账户类FAQ"),
    EvalCase("f5",  "支持哪些支付方式",          "faq", note="支付类FAQ"),
    EvalCase("f6",  "可以用信用卡支付吗",        "faq", note="疑问词→FAQ"),
    EvalCase("f7",  "会员等级有哪些",            "faq", note="会员类FAQ"),
    EvalCase("f8",  "七天无理由退换吗",          "faq", note="售后FAQ"),
    EvalCase("f9",  "营业时间是什么时候",        "faq", note="通用FAQ"),
    EvalCase("f10", "积分有什么用",              "faq", note="会员FAQ"),
    EvalCase("f11", "发票怎么开",                "faq", note="疑问词→FAQ"),
    EvalCase("f12", "快递丢了怎么办",            "faq", note="疑问词+物流→FAQ（问政策）"),

    # ---- Ticket - 查订单（4 条）----
    EvalCase("t1",  "帮我查一下订单123456789",              "ticket", "query_order",     "123456789", note="带订单号查订单"),
    EvalCase("t2",  "查看订单987654321的情况",               "ticket", "query_order",     "987654321", note="订单情况→查订单"),
    EvalCase("t3",  "订单555666777状态",                     "ticket", "query_order",     "555666777", note="订单状态→查订单"),
    EvalCase("t4",  "帮我查订单",                            "ticket", expect_need_slot="order_id", note="无订单号→引导"),

    # ---- Ticket - 查物流（4 条）----
    EvalCase("t5",  "帮我查一下订单123456789的物流",         "ticket", "query_logistics", "123456789", note="带单号查物流"),
    EvalCase("t6",  "订单987654321快递到哪了",               "ticket", "query_logistics", "987654321", note="快递→查物流"),
    EvalCase("t7",  "订单555666777什么时候能送达",           "ticket", "query_logistics", "555666777", note="送达→查物流"),
    EvalCase("t8",  "帮我查物流",                            "ticket", expect_need_slot="order_id", note="无单号查物流→引导"),

    # ---- Ticket - 退款（4 条）----
    EvalCase("t9",  "帮我申请退款订单987654321",             "ticket", "apply_refund",    "987654321", note="待发货→退款成功"),
    EvalCase("t10", "我要退货订单555666777",                 "ticket", "apply_refund",    "555666777", note="已完成→走退货流程"),
    EvalCase("t11", "取消订单123456789",                     "ticket", "apply_refund",    "123456789", note="取消订单→退款工具"),
    EvalCase("t12", "帮我退款",                              "ticket", expect_need_slot="order_id", note="无单号退款→引导"),
]


# ---- 多轮评测场景 ----
@dataclass
class MultiTurnScenario:
    """多轮评测场景"""
    id: str
    name: str
    session_id: str
    turns: list[dict]  # [{"query": "...", "expect_intent": "...", "expect_tool": ..., "expect_order_id": ..., "note": ...}]


MULTI_TURN_SCENARIOS: list[MultiTurnScenario] = [
    MultiTurnScenario(
        "m1", "槽位补全", "eval-m1",
        [
            {"query": "帮我查订单", "expect_intent": "ticket", "expect_need_slot": "order_id", "note": "首轮→引导"},
            {"query": "123456789", "expect_intent": "ticket", "expect_tool": "query_order", "expect_order_id": "123456789", "note": "纯数字→补全"},
        ],
    ),
    MultiTurnScenario(
        "m2", "槽位继承", "eval-m2",
        [
            {"query": "帮我查一下订单123456789的物流", "expect_intent": "ticket", "expect_tool": "query_logistics", "expect_order_id": "123456789", "note": "首轮带单号"},
            {"query": "那订单详情呢", "expect_intent": "ticket", "expect_tool": "query_order", "note": "继承单号→查订单"},
            {"query": "帮我退款", "expect_intent": "ticket", "expect_tool": "apply_refund", "note": "继承单号→退款"},
        ],
    ),
    MultiTurnScenario(
        "m3", "闲聊清空记忆", "eval-m3",
        [
            {"query": "帮我查一下订单123456789", "expect_intent": "ticket", "expect_tool": "query_order", "expect_order_id": "123456789", "note": "首轮建立记忆"},
            {"query": "你好", "expect_intent": "chitchat", "note": "闲聊→清空"},
            {"query": "帮我退款", "expect_intent": "ticket", "expect_need_slot": "order_id", "note": "记忆已清→重新引导"},
        ],
    ),
    MultiTurnScenario(
        "m4", "跨会话隔离", "eval-m4",
        [
            {"query": "帮我查一下订单987654321", "expect_intent": "ticket", "expect_tool": "query_order", "expect_order_id": "987654321", "note": "m4 建立 987654321"},
            {"query": "帮我退款", "expect_intent": "ticket", "expect_tool": "apply_refund", "note": "m4 用 987654321 退款"},
        ],
    ),
    # m5 用不同 session_id，验证与 m4 隔离
    MultiTurnScenario(
        "m5", "跨会话隔离-2", "eval-m5",
        [
            {"query": "帮我查一下订单555666777", "expect_intent": "ticket", "expect_tool": "query_order", "expect_order_id": "555666777", "note": "m5 建立 555666777"},
            {"query": "帮我退款", "expect_intent": "ticket", "expect_tool": "apply_refund", "note": "m5 用 555666777 退款"},
        ],
    ),
]


# ---- 评测执行 ----

@dataclass
class EvalReport:
    """评测报告"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    failures: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total * 100 if self.total > 0 else 0.0

    def add(self, ok: bool, case_id: str, detail: str) -> None:
        self.total += 1
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append({"case": case_id, "detail": detail})


def chat(base_url: str, query: str, session_id: str | None = None) -> dict:
    """调用 /chat 端点"""
    payload = {"query": query, "include_debug": True}
    if session_id:
        payload["session_id"] = session_id
    r = httpx.post(f"{base_url}/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def run_single_turn(base_url: str, log=print) -> EvalReport:
    """跑单轮评测"""
    report = EvalReport()
    log("\n" + "=" * 60)
    log("单轮评测（意图 + 工具 + 槽位）")
    log("=" * 60)

    for case in SINGLE_TURN_CASES:
        try:
            resp = chat(base_url, case.query)
        except Exception as e:
            report.add(False, case.id, f"请求失败: {e}")
            log(f"  ❌ {case.id} 请求失败: {e}")
            continue

        intent = resp.get("intent", "")
        debug = resp.get("debug", {})
        route = debug.get("route", {})
        actual_order_id = route.get("slots", {}).get("order_id")
        actual_tool = resp.get("debug", {}).get("route", {}).get("reason", "")

        # 从 answer 反推工具（规则路径 answer 包含工具输出特征）
        # 更可靠的方式：用 answer 内容判断
        answer = resp.get("answer", "")
        need_slot = resp.get("need_slot")

        checks = []
        # 意图检查
        checks.append(("intent", intent == case.expect_intent, f"期望={case.expect_intent} 实际={intent}"))

        # 工具检查（仅 ticket 有期望工具时）
        if case.expect_tool:
            # 通过 debug.route.reason 或 answer 内容推断
            tool_ok = False
            if case.expect_tool == "query_order" and ("详情" in answer or "商品" in answer):
                tool_ok = True
            elif case.expect_tool == "query_logistics" and ("物流" in answer or "快递" in answer):
                tool_ok = True
            elif case.expect_tool == "apply_refund" and ("退款" in answer or "退货" in answer or "拒收" in answer):
                tool_ok = True
            checks.append(("tool", tool_ok, f"期望工具={case.expect_tool} answer={answer[:30]}"))

        # 订单号检查
        if case.expect_order_id:
            checks.append(("order_id", actual_order_id == case.expect_order_id,
                           f"期望={case.expect_order_id} 实际={actual_order_id}"))

        # 引导补全检查
        if case.expect_need_slot:
            checks.append(("need_slot", need_slot == case.expect_need_slot,
                           f"期望need_slot={case.expect_need_slot} 实际={need_slot}"))

        all_ok = all(c[1] for c in checks)
        fail_details = " | ".join(f"{c[0]}: {c[2]}" for c in checks if not c[1])
        report.add(all_ok, case.id, fail_details)

        status = "✅" if all_ok else "❌"
        log(f"  {status} {case.id} [{case.expect_intent}] {case.query[:25]:<25} {case.note}")

    log(f"\n  单轮准确率: {report.passed}/{report.total} = {report.accuracy:.1f}%")
    return report


def run_multi_turn(base_url: str, run_id: str, log=print) -> EvalReport:
    """跑多轮评测

    参数:
        run_id: 本次运行唯一标识，拼接到 session_id 前面，避免复用上次遗留的 session
    """
    report = EvalReport()
    log("\n" + "=" * 60)
    log("多轮评测（槽位补全/继承/清空/隔离）")
    log("=" * 60)

    for scenario in MULTI_TURN_SCENARIOS:
        log(f"\n  --- {scenario.id} {scenario.name} ---")
        # 用 run_id 前缀确保每次评测用干净 session，避免上次遗留的槽位污染首轮
        real_session_id = f"{run_id}-{scenario.session_id}"
        for i, turn in enumerate(scenario.turns):
            case_id = f"{scenario.id}-t{i+1}"
            try:
                resp = chat(base_url, turn["query"], session_id=real_session_id)
            except Exception as e:
                report.add(False, case_id, f"请求失败: {e}")
                log(f"    ❌ {case_id} 请求失败: {e}")
                continue

            intent = resp.get("intent", "")
            answer = resp.get("answer", "")
            need_slot = resp.get("need_slot")
            debug = resp.get("debug", {})
            session_info = debug.get("session", {})
            slot_inherited = session_info.get("slot_inherited", False)
            route_slots = debug.get("route", {}).get("slots", {})
            actual_order_id = route_slots.get("order_id")

            checks = []
            checks.append(("intent", intent == turn["expect_intent"], f"期望={turn['expect_intent']} 实际={intent}"))

            if "expect_tool" in turn:
                tool_ok = False
                if turn["expect_tool"] == "query_order" and ("详情" in answer or "商品" in answer):
                    tool_ok = True
                elif turn["expect_tool"] == "query_logistics" and ("物流" in answer or "快递" in answer):
                    tool_ok = True
                elif turn["expect_tool"] == "apply_refund" and ("退款" in answer or "退货" in answer or "拒收" in answer):
                    tool_ok = True
                checks.append(("tool", tool_ok, f"期望工具={turn['expect_tool']} answer={answer[:30]}"))

            if "expect_order_id" in turn:
                checks.append(("order_id", actual_order_id == turn["expect_order_id"],
                               f"期望={turn['expect_order_id']} 实际={actual_order_id}"))

            if "expect_need_slot" in turn:
                checks.append(("need_slot", need_slot == turn["expect_need_slot"],
                               f"期望={turn['expect_need_slot']} 实际={need_slot}"))

            all_ok = all(c[1] for c in checks)
            fail_details = " | ".join(f"{c[0]}: {c[2]}" for c in checks if not c[1])
            report.add(all_ok, case_id, fail_details)

            inherit_tag = " [继承]" if slot_inherited else ""
            status = "✅" if all_ok else "❌"
            log(f"    {status} {case_id} {turn['query'][:25]:<25}{inherit_tag} {turn['note']}")

    log(f"\n  多轮准确率: {report.passed}/{report.total} = {report.accuracy:.1f}%")
    return report


def main():
    parser = argparse.ArgumentParser(description="Agent 评测脚本")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="服务地址")
    args = parser.parse_args()

    # 等服务就绪
    print(f"等待服务就绪: {args.base_url}")
    for i in range(30):
        try:
            if httpx.get(f"{args.base_url}/health", timeout=5).status_code == 200:
                print("服务已就绪 ✓")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("❌ 服务未就绪，请先启动: uvicorn app.main:app --port 8001")
        sys.exit(1)

    # 本次运行唯一标识（拼到 session_id 前面，避免复用上次遗留的 session）
    run_id = time.strftime("%Y%m%d%H%M%S")

    # 收集输出到 buffer，同时打印 + 落盘
    lines: list[str] = []

    def log(msg: str = ""):
        lines.append(msg)
        print(msg, flush=True)

    single = run_single_turn(args.base_url, log)
    multi = run_multi_turn(args.base_url, run_id, log)

    # 汇总
    total = single.total + multi.total
    passed = single.passed + multi.passed
    failed = single.failed + multi.failed
    accuracy = passed / total * 100 if total > 0 else 0

    log("\n" + "=" * 60)
    log("评测汇总")
    log("=" * 60)
    log(f"  单轮: {single.passed}/{single.total} = {single.accuracy:.1f}%")
    log(f"  多轮: {multi.passed}/{multi.total} = {multi.accuracy:.1f}%")
    log(f"  总计: {passed}/{total} = {accuracy:.1f}%")

    if failed > 0:
        log(f"\n失败用例 ({failed}):")
        all_failures = single.failures + multi.failures
        for f in all_failures:
            log(f"  ❌ {f['case']}: {f['detail']}")

    # 落盘（utf-8-sig 带 BOM，让 Windows type 命令正确识别 UTF-8，避免 GBK 乱码）
    with open("eval_result.txt", "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    log(f"\n报告已保存: eval_result.txt")
    log()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"ERROR: {e}")
        with open("eval_result.txt", "w", encoding="utf-8-sig") as f:
            f.write(f"脚本异常:\n{err}")
        sys.exit(2)
