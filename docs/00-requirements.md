# 多 Agent 量化交易决策系统 — 需求文档 v1.1

> 文档编号：00-requirements
> 更新日期：2026-04-25
> 状态：Aligned，进入设计阶段

---

## 1. 项目定位

### 1.1 一句话描述
面向美股单票决策的多 Agent 编排系统：三面分析 → 多空辩论 → 风控审核 → 执行下单，串联成可解释、可复盘的 LLM 工程化流水线。

### 1.2 核心展示目标（作品集导向）

| 维度 | 展示点 |
|------|--------|
| 多 Agent 编排 | LangGraph 状态机、并行 Fan-out / Fan-in、循环辩论、条件分支（风控否决） |
| LLM 工程化 | Prompt 模板、Pydantic 结构化输出、JSONL Trace、重试与降级 |
| 金融领域理解 | 基于 `docs/analysismetrics.md` 的三面评分体系、硬规则 + VaR + 软规则的风控分层 |

### 1.3 非目标（Non-goals）

- 不托管真实资金
- 不做策略 Alpha 验证 / 大规模回测 / 因子挖掘
- 决策粒度日频，不做高频 / 日内
- 仅美股，不多市场、不多用户
- 不做 Web UI / Dashboard、不做实时流式行情

---

## 2. 系统架构与流水线

### 2.1 主流水线

```
   ┌────────────────────────────────────┐
   │  CLI 入口 (ticker, mode, params)   │
   └────────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────────────────┐
   │      Research Layer (并行 fan-out)  │
   │  Fundamental │ Technical │ Sentimental│
   └────────────────────────────────────┘
                   │ fan-in
                   ▼
   ┌────────────────────────────────────┐
   │   Debate Agent (Bull ↔ Bear → Judge)│
   │   Output: SELL / HOLD / BUY         │
   └────────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────────────────┐
   │  Risk Agent (硬规则 → VaR → 软规则)  │
   └────────────────────────────────────┘
        │ reject              │ approve
        ▼                     ▼
   ┌──────────┐    ┌──────────────────────┐
   │ Halt+Log │    │ HITL 确认 (可选开关) │
   └──────────┘    └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Execution (DryRun /  │
                   │  Paper / Backtest)   │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Portfolio Update +   │
                   │ 双语 Markdown 报告   │
                   └──────────────────────┘
```

### 2.2 共享 State（LangGraph StateGraph）

```
GraphState:
  input:        { ticker, capital, mode, debate_rounds, as_of_date, hitl_enabled }
  market_data:  { info, history, financials, options_chain, ... }
  research:     { fundamental: Report, technical: Report, sentimental: Report }
  debate:       { turns: [DebateTurn], verdict, rationale, proposed_position_pct }
  risk:         { hard_checks, var_result, soft_judgment, decision }
  execution:    { order, status, fill_info }
  portfolio:    { positions_before, positions_after, nav }
  meta:         { trace_id, started_at, tokens_total, latency_ms, warnings: [] }
```

### 2.3 CLI 入口

```bash
python -m fishtrade run \
  --ticker AAPL \
  --capital 100000 \
  --mode dryrun \           # dryrun | paper | backtest
  --debate-rounds 1 \       # 默认 1 轮，可设 0~3
  --hitl off \              # off (默认) | on
  --as-of 2026-04-24        # 仅 backtest 模式有效
```

### 2.4 运行模式

| 模式 | 行为 |
|------|------|
| `dryrun`（默认） | 全流程跑，订单只写日志不发送 |
| `paper` | 订单提交 Alpaca Paper Trading API |
| `backtest` | 指定 `--as-of` 历史日期单点回放，所有数据按当日截断；不写真实订单，只产报告 |

---

## 3. 通用基础设施

所有 Agent 共享，独立模块化。

### 3.1 LLM 客户端（Ark + LangChain）

- **接入方式**：火山引擎 Ark（OpenAI 兼容），全程复用默认接入点 **`ep-20260417212516-rpphh`**
- **Base URL**：`https://ark.cn-beijing.volces.com/api/v3`
- **认证**：`ARK_API_KEY` 环境变量，启动时前置校验，未配置则直接终止
- **客户端封装**：使用 `langchain_openai.ChatOpenAI` 指向 Ark base_url，消息构造统一用 `langchain_core.messages.HumanMessage` / `SystemMessage`
- **统一入口**：`python/llm/` 暴露一个工厂函数返回配置好的 `ChatOpenAI` 实例，所有 Agent 通过它调用，便于注入 trace / 重试 / token 统计
- **结构化输出**：所有 Agent 输出走 Pydantic v2，调用 `ChatOpenAI.with_structured_output(Schema)` 做 schema 校验

### 3.2 Trace（本地 JSONL）

- 每次流水线运行分配 `trace_id`，所有 LLM 调用追加写入 `traces/{trace_id}.jsonl`
- 每条记录字段：`timestamp, agent, role, prompt, response, tokens_in, tokens_out, latency_ms, status`
- 不引入 LangSmith / 火山观测面板
- 提供简易 CLI viewer：`python -m fishtrade trace show <trace_id>`

### 3.3 配置管理

- `pydantic-settings` + `.env`
- 关键配置项：`ARK_API_KEY` / `ALPACA_API_KEY` / `ALPACA_API_SECRET` / 风控阈值 / 报告输出路径

### 3.4 Portfolio 模块

- 文件存储：`data/portfolio.json`
- **首次运行自动初始化**：`{ cash: 100000, positions: [], nav_history: [{date, nav: 100000}] }`
- 每次 Execution 成交后写回（持仓、现金、新增 NAV 节点）
- 提供查询接口：`get_total_nav()` / `get_drawdown()` / `get_position(ticker)` / `get_sector_concentration()`

### 3.5 数据缓存（yfinance）

- 缓存键：`(ticker, date, data_type)`，TTL = 24 小时
- 实现：SQLite 或 parquet 文件，落地 `data/cache/`
- 所有 yfinance 调用必经此层；指数退避重试 3 次（间隔 2s/5s/10s）

### 3.6 双语 Markdown 报告

- 每次运行生成 `reports/{ticker}-{date}-{lang}.md`，输出 `zh` 和 `en` 两份
- 内容：输入参数、三面打分明细、辩论全文、风控判定、执行结果、警告列表
- 报告生成器读取最终 `GraphState`，模板化渲染

### 3.7 配套测试

- `pytest` + VCR.py 录制 yfinance 与 Ark 响应，保证测试不消耗真实 token
- 覆盖各 Agent 的正常路径与降级路径

---

## 4. Agent 模块详细需求

### 4.1 研究层（Research Agents，3 路并行）

#### 4.1.1 输入 / 输出

- **输入**：`ticker`、`as_of_date`
- **输出**（Pydantic）：
  ```
  ResearchReport:
    indicator_scores: [{name, raw_value, score(-1|0|+1), reasoning}]   # 10 项
    total_score: int (-10 ~ +10)
    verdict: BUY | HOLD | SELL
    key_highlights: [str]   # 3-5 条
    confidence: float (0.0-1.0)
    degraded_indicators: [str]   # 因数据缺失降级为 0 的指标名
  ```

#### 4.1.2 评分规范

严格遵循 [`docs/analysismetrics.md`](analysismetrics.md)，每面 10 个指标，单指标 +1/0/-1。

#### 4.1.3 三个 Agent 的特殊点

| Agent | 数据源 | 关键说明 |
|-------|--------|----------|
| Fundamental | `Ticker.info`、`financials` | 先按 `info.sector` / `industry` 映射到 analysismetrics 的行业分类（价值股/成长股/金融/消费/制造等）再套评分区间；指标用代码算，LLM 只做解读与定性 |
| Technical | `Ticker.history()` 近 1 年日线 | 用 `pandas-ta` 计算 MACD / RSI / 均线 / 布林带 / ATR / 斐波那契等；形态识别用简化规则 + LLM 交叉验证 |
| Sentimental | `info`、`institutional_holders`、`insider_transactions`、`upgrades_downgrades`、`option_chain` | 社交媒体指标 yfinance 无数据 → MVP 直接计 0 分降级；财报超预期数据不足同样降级 |

#### 4.1.4 性能

- LangGraph 并行分支同时执行
- 总超时 90s，单 Agent 硬超时 60s

#### 4.1.5 边界 / 异常

| 场景 | 行为 |
|------|------|
| Ticker 不存在 / 已退市 | 流水线前置校验失败，返回 `INVALID_TICKER` 终止 |
| yfinance 限流 / 超时 | 走缓存层重试 3 次，仍失败该指标计 0 分并入 `degraded_indicators` |
| 财报数据不足（< 4 季度） | 相关指标降级为 0，`confidence` 同步下调 |
| 期权链为空 | Put/Call Ratio 指标计 0 分 |
| 非交易日 / 停牌 | 使用最近交易日数据，报告标注实际 `as_of_date` |

---

### 4.2 辩论层（Debate Agent）

#### 4.2.1 角色

| 角色 | 职责 | Temperature |
|------|------|-------------|
| Bull | 提取看多证据，论证买入 | 0.7 |
| Bear | 提取看空证据，论证卖出 | 0.7 |
| Judge | 综合全部辩论，输出最终判定 | 0.2 |

#### 4.2.2 流程

1. **首轮**：Bull / Bear 各自接收完整 research → 独立给出首轮结论
2. **辩论轮**（默认 1 轮，可配 0~3）：每轮双方接收对方上一轮结论 → 反驳 → 给出本轮结论
3. **终审**：Judge 接收全部辩论历史（含 research 摘要），输出 BUY/HOLD/SELL + 仓位建议

#### 4.2.3 输出（Pydantic）

```
DebateTurn:
  round: int (0=opening, 1+=rebuttal)
  role: bull | bear
  argument: str
  cited_indicators: [str]
  conclusion: BUY | HOLD | SELL

DebateResult:
  turns: [DebateTurn]
  final_verdict: BUY | HOLD | SELL
  final_rationale: str
  confidence: float (0.0-1.0)
  proposed_position_pct: float (0-10)   # 传给 Risk Agent
```

#### 4.2.4 约束

- Bull / Bear Prompt 强制"必须引用 research 中具体指标"，禁止凭空论证
- Research 中已降级的面，Judge 收到降级标记并降低其权重

#### 4.2.5 边界 / 异常

| 场景 | 行为 |
|------|------|
| 三面 research 全降级 | 跳过 Bull / Bear，Judge 直接输出 HOLD + 警告 |
| Bull / Bear 某轮拒答或非合规输出 | 用上一轮结论占位，写入 `warnings` |
| 辩论历史 token 超限 | 仅保留最近 2 轮 + research 摘要传给 Judge |
| Ark 调用超时 | 重试 2 次（2s / 5s），仍失败则该角色用规则化降级模板输出 |
| 返回非合法 JSON | 触发"仅返回 JSON"的强约束重试 1 次，再失败进降级 |

---

### 4.3 风控层（Risk Agent）

#### 4.3.1 流程（短路失败）

```
硬规则 ──失败──→ REJECT
   │ 通过
   ▼
VaR 计算 ──超限──→ REJECT
   │ 通过
   ▼
LLM 软规则 ──高风险──→ REJECT
   │ 通过
   ▼
APPROVE → (HITL 可选) → Execution
```

#### 4.3.2 硬规则

| 规则 | 阈值 | 判定 |
|------|------|------|
| R1 单票仓位 | ≤ 10% 总资产 | `proposed_position_pct ≤ 10` |
| R2 组合最大回撤 | ≤ 8% | 基于 `portfolio.nav_history` 滚动峰谷计算 |
| R3 VaR(95%) | 组合单日损失 ≤ 2% | `stock_var_95 × proposed_position_pct ≤ 2%` |
| R4 止损线 | 5% | 下单时附带 `stop_price = entry × 0.95` |

#### 4.3.3 VaR 计算

- 历史模拟法：取近 252 日收益率序列，5% 分位数
- 历史 < 60 交易日 → REJECT 并标注原因
- 组合层面用"单票 VaR × 持仓比例"近似（已知简化，不考虑相关性）

#### 4.3.4 软规则（LLM 判断）

LLM 输入：debate 结论 + 当前 portfolio + 市场波动数据（VIX、大盘 5 日涨跌）

| 触发条件 | 处理 |
|----------|------|
| VIX > 30（市场极端波动） | 建议仓位降至原提议的 50% |
| 同行业持仓累计 > 20% | 警告 + 仓位下调 |
| 日均成交额 < $10M（流动性不足） | 警告 + 限制最大下单量 |

输出：
```
SoftJudgment:
  flags: [MARKET_VOLATILE | CORRELATION_HIGH | LIQUIDITY_LOW]
  adjustment: keep | reduce_to_<pct> | reject
  reasoning: str
```

#### 4.3.5 最终输出

```
RiskDecision:
  decision: approve | reject
  adjusted_position_pct: float
  hard_checks: [{rule, passed, detail}]
  var_result: {var_95, portfolio_impact, passed}
  soft_judgment: SoftJudgment
  reject_reason: str | null
```

#### 4.3.6 边界 / 异常

| 场景 | 行为 |
|------|------|
| Debate 结论为 HOLD | Risk 跳过仓位计算，记录"观望"，Execution 不动作 |
| 历史数据不足 60 日 | 直接 REJECT |
| 组合回撤已达上限 | 一票否决所有 BUY，允许 SELL |
| 软规则 LLM 调用失败 | 降级为保守判定（建议仓位下调 50%）+ 警告 |
| Portfolio 文件首次运行 | 自动初始化（cash=100000, positions=[]，见 §3.4） |

---

### 4.4 执行层（Execution Agent）

#### 4.4.1 前置条件

- `RiskDecision.decision == approve` 才触发
- 若 `hitl_enabled=true`，先在 CLI 打印订单摘要，等待人工 `[y/N]` 确认；超时 60s 默认拒绝

#### 4.4.2 执行模式

| 模式 | 行为 |
|------|------|
| `dryrun` | 生成订单对象写 `logs/orders/*.json`，不发送 |
| `paper` | 调 Alpaca `submit_order`，轮询确认 fill（最多 30s） |
| `backtest` | 按 `as_of_date` 当日收盘价虚拟成交，写回 backtest portfolio |

#### 4.4.3 流程

1. 计算下单量：`shares = floor(capital × adjusted_position_pct / current_price)`
2. 限价单：`limit_price = current × 1.002`（buy）/ `× 0.998`（sell），TIF=DAY
3. 止损单：`stop_price = entry × 0.95`（仅 buy）
4. 提交 / 模拟成交
5. 写回 Portfolio：更新持仓、现金、追加 NAV 节点

#### 4.4.4 输出

```
ExecutionResult:
  mode: dryrun | paper | backtest
  order: { symbol, side, qty, type, limit_price, stop_price, tif }
  status: generated | submitted | filled | partial | failed | rejected_by_user
  fill_info: { avg_price, filled_qty, fill_time } | null
  error: str | null
```

#### 4.4.5 边界 / 异常

| 场景 | 行为 |
|------|------|
| dryrun | 永远成功，仅写日志 |
| Alpaca API 鉴权失败 / 不可达 | 状态 `failed`，portfolio 不更新，错误写入报告 |
| Paper 部分成交 | 按实际成交量更新 portfolio，未成交部分标注 |
| 市场休市 | 订单按 Alpaca 行为排队到下一交易日，报告标注 |
| HITL 用户拒绝 / 超时 | 状态 `rejected_by_user`，portfolio 不更新 |
| Backtest 当日数据缺失 | 终止本次回放，报告说明 |

---

## 5. 后处理

### 5.1 Portfolio 更新

仅 Execution 成功成交（`filled` / `partial`）时写回 `data/portfolio.json`，并在 `nav_history` 追加新节点。

### 5.2 双语 Markdown 报告

- 路径：`reports/{ticker}-{as_of_date}-{zh|en}.md`
- 段落：输入参数 / 三面打分（含降级标记） / 辩论全文 / 风控判定 / 执行结果 / 警告列表 / Trace ID

---

## 6. 范围外（Out of Scope）

- 多股票组合优化（马科维茨、Black-Litterman）
- 实时流式行情、Level 2、逐笔
- 期货 / 期权 / 加密 / 港股 / A 股
- 日内 / 高频
- 真实资金账户对接
- Web / 移动端
- 多用户、权限、计费
- 因子挖掘、强化学习
- 大规模滚动回测
- 合规审计

---

## 7. 验收标准（MVP Done）

1. 单条命令跑通：`python -m fishtrade run --ticker AAPL --capital 100000`
2. 三面 Agent 并行执行，输出符合 analysismetrics.md 规范
3. Debate 默认 1 轮，可通过 `--debate-rounds N` 调整
4. Risk Agent 4 条硬规则全生效，有覆盖拒绝路径的测试用例
5. dryrun / paper / backtest 三种模式可切换
6. HITL 开关可用，拒绝路径正确处理
7. 每次运行生成 `reports/{ticker}-{date}-{zh|en}.md` 双语报告
8. 本地 JSONL trace 可回看每次 Ark 调用（含 messages、token、耗时）
9. Portfolio 首次运行自动初始化（cash=100000，positions=[]）
10. README 含架构图、跑通步骤、示例报告、已知简化项清单

---

*v1.1 — 进入 `docs/01-architecture.md` 与 `docs/02-agents-spec.md`*
