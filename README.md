# MyCom —— 企业级智能客服 Agent

> 一个能**自动回答 FAQ**、**查询订单/物流**、**处理多步骤复合意图**的智能客服系统。
> 没有 LLM Key 也能跑起来，配了 Key 自动解锁全部能力。

---

## 🤔 这东西解决什么问题？

传统客服系统要么只能答静态 FAQ，要么每条都要人处理。**MyCom** 想做的是：

```
用户: "我订单 123456789 的快递到哪了？"     → 自动查物流返回
用户: "帮我查物流然后退款"                   → 先查物流，发现已发货，引导拒收后退款
用户: "怎么申请7天无理由退货？"              → 在 FAQ 里搜出答案
用户: "你好呀"                              → 闲聊接住
```

一句话：**把客服能自动搞定的事全自动化，搞不定的才扔给人。**

---

## 🏗️ 架构长什么样？

三层流水线，从用户提问到最终回答一路传下去：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户提问（自然语言）                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🔀 Agent 路由器（Router）                          │
│            判断用户想干嘛：FAQ？查业务？闲聊？多步骤？                 │
│           （有 LLM 用 LLM 判断，没 LLM 用关键词 + 规则兜底）           │
└──────┬──────────────────────┬──────────────────────┬────────────────┘
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  📚 FAQ 链路  │    │  🔧 业务工具链路  │    │ 🎯 Planner +       │
│  向量 + BM25  │    │  订单/物流/退款   │    │    Executor        │
│  混合检索     │    │  （真实 API 或   │    │  多步骤任务编排     │
│              │    │   Mock 数据）     │    │  （查物流→退款）    │
└──────┬───────┘    └────────┬─────────┘    └────────┬───────────┘
       └─────────────────────┼───────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ✍️ Answer Agent（答案生成）                       │
│         把零散的工具执行结果，整合成人说的自然语言                    │
│       （有 LLM 让 LLM 润色，没 LLM 用模板拼接）                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                        最终回答用户
```

**核心设计思路：每条链路都有「LLM 路径 + 规则兜底」，LLM Key 不是刚需。**

---

## ⭐ 亮点

### 1. 双路径架构：LLM 不是必须的

| 组件 | 有 LLM Key | 没 LLM Key |
|------|-----------|-----------|
| Router | LLM 判断意图 + 抽槽位 | 关键词 + 正则 + 规则 |
| Planner | LLM Function Calling 生成计划 | 复合连接词（"然后/接着"）规则分解 |
| Answer Agent | LLM 润色成自然语言 | 模板拼接 + 字段归一化 |

**好处：** 开发、测试、演示随时能跑，不用担心 API Key 欠费。真要上线再配 Key 就行。

### 2. FAQ 混合检索：向量 + BM25 = 又准又全

纯向量检索对"怎么退款"和"如何退款"这类语义接近但用词不同的 query 可能撞车；
纯 BM25 关键词检索对"那个东西怎么弄"这种口语化表达又抓瞎。

**我们把俩都用：**
- 向量模型（`text2vec-base-chinese`）抓语义
- BM25（`rank-bm25`）抓关键词
- 用 RRF 算法（K=60）把两边结果融合排序

效果：**查得全（不遗漏）+ 查得准（排前面的真有用）。**

### 3. 多步骤编排：不只是"调用工具"，是"理解意图后规划"

用户说"查物流然后退款"，系统不会同时扔两个工具出去——它会：

1. **Planner** 先分解：Step 1 查物流 → Step 2 退款
2. **Executor** 按顺序执行：先查到"已发货"，再调用退款
3. **业务拒绝处理**：退款工具返回"已发货不能退"，Executor 识别这是**业务规则拒绝**（不是技术失败），不会傻重试 3 次，直接把"不能退"的原因 + 建议传给用户
4. **Answer Agent** 整合输出：
   ```
   物流信息：顺丰 SF1234567890，运输中，预计明天送达
   
   【退款】⚠️ 订单已发货，需先拒收快递后才能退款
   建议：可以拒收快递后再申请退款，或签收后走「7天无理由退货」流程。
   ```

### 4. 会话记忆：多轮对话无缝衔接

用户第一轮说"帮我查订单"（没给订单号），系统追问"请告诉我订单号"；
第二轮用户只说 `123456789`，系统自动补上订单号去查——不用用户重复说"帮我查订单 123456789"。

### 5. 规则层 vs 执行层 vs 答案层：各司其职

```
Router（判断）  →  Planner（规划）  →  Executor（执行）  →  Answer Agent（组织语言）
   只管"用户想干嘛"   只管"分几步做"     只管"把每步跑通"      只管"怎么说好听"
```

每一层独立可替换——以后想换 LLM、想加新工具、想换 Milvus 为 ES，改一层就行。

---

## 🛠️ 技术栈

| 分类 | 选型 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | 异步、类型友好、自动生成 API 文档 |
| 向量数据库 | Milvus | 开源、可扩展、生产级 |
| Embedding | text2vec-base-chinese | 轻量中文模型，40MB，CPU 能跑 |
| 关键词检索 | rank-bm25 + jieba | 纯 Python，不用运维 ES |
| LLM | 兼容 OpenAI API 的任意模型 | DeepSeek 便宜好用；不配也行 |
| 日志 | loguru | 比原生 logging 好用 10 倍 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Docker（跑 Milvus）
- 可选：LLM API Key（不配也能跑）

### 1. 拉代码 & 装依赖

```bash
git clone https://github.com/PopZack/mycom.git
cd mycom

python -m venv .venv
# Windows 激活:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 启动 Milvus

```bash
docker compose up -d
# 等 30-60 秒 → 确认三个容器都 Healthy
docker ps
```

### 3. 配环境变量

```bash
cp .env.example .env
# Windows: copy .env.example .env

# 不配 LLM Key 也能跑，全走规则路径
# 要配的话填 LLM_API_KEY 就行
```

### 4. 启动服务

```bash
python -m uvicorn app.main:app --port 8001
```

看到 `✅ 服务就绪 ✓` 就启动成功了。

### 5. 跑评测

```bash
python scripts/eval_agent.py
# 当前: 42/42 = 100% 通过率（30 单轮 + 12 多轮）
```

### 6. 手动测试

```bash
# 查订单
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我查订单123456789"}'

# 查物流 + 退款（多步骤）
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我查订单123456789的物流，然后退款", "session_id": "test-1"}'
```

---

## 📂 目录结构

```
mycom/
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置加载（.env）
│   │
│   ├── agent/                  # Phase 2: 路由 + 会话 + 单工具链路
│   │   ├── agent_chain.py      # 统一调度器（Phase 2 + Phase 3 混合）
│   │   ├── router.py           # 意图路由器（LLM 路径 + 规则降级）
│   │   ├── session_store.py    # 会话存储（槽位记忆、TTL 过期）
│   │   ├── ticket.py           # 工单链路（订单/物流/退款）
│   │   ├── chitchat.py         # 闲聊链路
│   │   └── intents.py          # 意图枚举
│   │
│   ├── agents/                 # Phase 3.1: 答案生成 Agent
│   │   └── answer_agent.py     # 规则模板版 + LLM 路径预留
│   │
│   ├── planning/               # Phase 3: 多步骤规划 + 执行
│   │   ├── planner.py          # 计划生成（LLM Function Calling + 规则分解）
│   │   ├── executor.py         # 计划执行（重试 + 业务拒绝 + 结果聚合）
│   │   └── plan.py             # Plan / PlanStep / 状态枚举
│   │
│   ├── chains/                 # Phase 1: FAQ 链路
│   │   └── faq_chain.py        # FAQ 统一入口
│   │
│   ├── retriever/              # FAQ 检索底层
│   │   ├── milvus_client.py    # Milvus 客户端（带重试）
│   │   ├── bm25_retriever.py   # BM25 关键词检索
│   │   ├── hybrid_search.py    # RRF 混合排序
│   │   └── faq_retriever.py    # FAQ 检索器（向量 + BM25）
│   │
│   ├── tools/                  # 业务工具层
│   │   ├── registry.py         # 工具注册表
│   │   ├── base.py             # Tool / ToolResult 基类
│   │   └── order_tools.py      # 订单/物流/退款工具
│   │
│   └── services/               # 业务数据层
│       └── order_service.py    # Mock + 真实 API 切换
│
├── data/
│   └── faq_sample.json         # FAQ 语料（20 条 + 手写变体）
│
├── scripts/
│   ├── eval_agent.py           # 自动化评测（42 条用例）
│   ├── gen_variations.py       # FAQ 变体生成脚本
│   ├── sync_faq.py             # FAQ 同步到 Milvus
│   └── mock_business_api.py    # Mock 业务 API 服务
│
├── docker-compose.yml          # Milvus 单节点（开发用）
├── requirements.txt
└── .env.example
```

---

## 🧪 评测

42 条用例，覆盖：

| 类型 | 数量 | 覆盖场景 |
|------|------|---------|
| 单轮 | 30 | FAQ 检索、订单/物流/退款、闲聊、意图模糊 |
| 多轮 | 12 | 槽位补全（"帮我查订单" → 用户给号）、跨轮上下文继承 |

```
============================================================
评测汇总
============================================================
  单轮: 30/30 = 100.0%
  多轮: 12/12 = 100.0%
  总计: 42/42 = 100.0%
```

跑一遍：

```bash
python scripts/eval_agent.py
```

---

## 🗺️ 项目路线图

```
Phase 1 ✅  FAQ 静态知识链路（Milvus + BM25 混合检索）
Phase 2 ✅  Agent 路由 + 会话记忆 + 单工具业务链路
Phase 3 ✅  多步骤规划 + Executor + Answer Agent（规则版）
Phase 4 🔜  Milvus 意图库（语义匹配路由，提升边界 case 准确率）
Phase 4 🔜  Answer Agent LLM 路径（配 Key 自动升级答案质量）
Phase 5 🔜  API 网关（认证/限流/监控）+ 真实业务 API 对接
```

---

## ❓ 常见问题

**Q: 必须配 LLM API Key 吗？**
A: 不用。不配 Key 全走规则路径，42 条评测全过。配了 Key 自动解锁 LLM 路径。

**Q: Milvus 必须装吗？**
A: 是的，FAQ 检索依赖 Milvus 的向量搜索。不过只需要 Docker 跑三个容器，不用装 Milvus Cluster。

**Q: Mock 数据怎么换真实 API？**
A: `.env` 里把 `USE_MOCK_DATA=true` 改成 `false`，然后填 `ORDER_API_BASE_URL` 和 `ORDER_API_KEY`。

**Q: 加新工具（比如查优惠券）要改哪？**
A: 1. 在 `app/tools/` 下新增工具类继承 `Tool`；2. 在 `registry.py` 注册；3. 在 `Planner` 的规则里加上新工具的触发关键词。搞定。

---

## 📜 License

MIT
