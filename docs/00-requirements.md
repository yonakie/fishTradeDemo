# 多 Agent 量化交易决策系统 — 需求文档 v1.0

> 文档编号：00-requirements
> 作者：Product (AI-assisted)
> 更新日期：2026-04-24
> 状态：Draft for alignment

---

## 0. 阅读导航

| 章节 | 内容 | 目标读者 |
|------|------|----------|
| 1 | 项目定位 & 目标 / 非目标 | 所有人 |
| 2 | 需求完整性评估与补充 | 作者自检 |
| 3 | 系统总体架构与流水线 | 工程实施 |
| 4 | 各 Agent 模块详细需求 | 工程实施 |
| 5 | 技术选型评估 | 工程实施 |
| 6 | 边界条件与异常处理 | 工程实施 + QA |
| 7 | 范围外（Out of Scope） | 对齐预期 |

---

## 1. 项目定位

### 1.1 一句话描述
一个面向美股单票决策的多 Agent 编排系统：从三面分析 → 多空辩论 → 风控审核 → 执行下单，串联成一条可解释、可复盘的 LLM 工程化流水线。

### 1.2 核心展示目标（作品集导向）

| 维度 | 展示点 |
|------|--------|
| **多 Agent 编排** | LangGraph 状态机、并行 Fan-out / Fan-in、带轮次的循环辩论、条件分支（风控否决） |
| **LLM 工程化** | Prompt 模板管理、结构化输出（Pydantic / JSON Schema）、Token 统计与 Trace、重试降级 |
| **金融领域理解** | 基于 `docs/analysismetrics.md` 的三面评分体系、硬规则 + VaR + 软规则的风控分层 |

### 1.3 明确的非目标（Non-goals）

- 不托管真实用户资金，不提供生产级账户系统
- 不保证策略 Alpha，不做大规模回测与因子挖掘
- 不支持高频 / 日内交易，决策粒度为日频（每日盘后或盘前跑一次）
- 不做多市场适配，仅美股
- 不做多用户、多租户、权限体系

---

## 2. 需求完整性评估

你给出的需求已覆盖主干（研究 → 辩论 → 风控 → 执行），**可支撑一个最小闭环**，但作为作品集项目仍有以下缺口需要补齐：

### 2.1 必须补充（否则流水线跑不通）

| 缺口 | 说明 | 补充建议 |
|------|------|----------|
| **输入接口** | 没定义用户如何触发 | CLI 入口：`python -m fishtrade run --ticker AAPL --capital 100000 --mode dryrun`，参数含股票代码、总资产、模式（dryrun / paper）、辩论轮数 |
| **Portfolio State** | Risk Agent 依赖"总资产、现有持仓、历史回撤"，但这些状态来源未定义 | 引入一个简单的 `Portfolio` 模块：初始持仓 JSON 文件 + 每次执行后写回；最大回撤基于历史净值序列滚动计算 |
| **LLM 选型与抽象** | 未指定 | 使用火山引擎 Ark（OpenAI 兼容接口），通过 `openai` SDK 设置 `base_url=https://ark.cn-beijing.volces.com/api/v3`，API Key 从 `ARK_API_KEY` 读取；默认模型接入点 `ep-20260417212516-rpphh`，可用 `ARK_MODEL_ID` 或参数覆盖。后续如需 LangChain 生态工具，统一用 `langchain-openai` 的 `ChatOpenAI` 指向同一 base_url 即可 |
| **结构化输出** | Agent 之间传递需要 schema | 每个 Agent 的输出用 Pydantic 模型定义（如 `FundamentalReport`、`DebateTurn`、`RiskDecision`）。通过 Ark 的 `response_format={"type": "json_object"}` + Prompt 约束 schema 的方式落地；若走 LangChain，则用 `ChatOpenAI.with_structured_output()` |
| **可观测性** | LLM 工程化核心 | 本地 JSONL trace 为主（自建轻量 trace writer，记录每次 LLM 调用的 prompt / response / token / 耗时 / endpoint），可选接入火山引擎自带的观测面板；不绑定 LangSmith（与 Ark 非原生对接，接入成本高） |

### 2.2 强烈建议补充（展示价值高）

| 项 | 价值 |
|----|------|
| **决策报告导出** | 每次跑完输出一份 Markdown 报告（含三面打分、辩论全文、风控判定、最终决策），便于在 README / 作品集中展示 |
| **简单回测模式** | 对历史日期做单点回放（非滚动回测），验证决策流水线的稳定性 |
| **数据缓存** | yfinance 有限流，同一 ticker 当日重复调用走本地缓存（SQLite / parquet） |
| **Human-in-the-loop 开关** | 可选在风控通过后暂停等待用户确认，体现"人机协同"设计感 |

### 2.3 可延后（不影响 MVP）

- Web UI / Dashboard（先 CLI + Markdown 报告）
- 多股票组合级决策（当前按单票跑，portfolio 只做聚合校验）
- 实时行情接入（yfinance 日频数据足够）

---

## 3. 系统总体架构与流水线

### 3.1 主流水线（LangGraph StateGraph）

```
           ┌──────────────────────────────────────┐
           │   Input: ticker, capital, params     │
           └──────────────────────────────────────┘
                           │
                           ▼
           ┌──────────────────────────────────────┐
           │         Research Layer (并行)         │
           │  ┌──────────┬──────────┬──────────┐  │
           │  │Fundamental│Technical│Sentimental│  │
           │  └──────────┴──────────┴──────────┘  │
           └──────────────────────────────────────┘
                           │  (fan-in)
                           ▼
           ┌──────────────────────────────────────┐
           │        Debate Agent (N 轮循环)        │
           │    多方 ↔ 空方  →  裁判方 最终判定     │
           │         Output: SELL / HOLD / BUY     │
           └──────────────────────────────────────┘
                           │
                           ▼
           ┌──────────────────────────────────────┐
           │            Risk Agent                 │
           │  硬规则 → VaR → 软规则(LLM) → 裁决    │
           └──────────────────────────────────────┘
                  │ reject              │ approve
                  ▼                     ▼
          ┌─────────────┐      ┌────────────────┐
          │ Halt + Log  │      │ Execution Agent│
          └─────────────┘      │ DryRun / Paper │
                               └────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────┐
                          │  Portfolio Update   │
                          │  + Markdown Report  │
                          └─────────────────────┘
```

### 3.2 共享 State（LangGraph State Schema）

State 是整条链路的"账本"，各 Agent 读写同一份 state，便于追溯和复盘：

```
GraphState:
  - input:         { ticker, capital, mode, debate_rounds, as_of_date }
  - market_data:   { info, history, financials, options_chain, ... }   # yfinance 原始数据缓存
  - research:      { fundamental: Report, technical: Report, sentimental: Report }
  - debate:        { turns: [DebateTurn...], verdict: SELL|HOLD|BUY, rationale }
  - risk:          { hard_checks: [...], var_result, soft_judgment, decision: approve|reject }
  - execution:     { order: Order|null, status, fill_info }
  - portfolio:     { positions_before, positions_after, nav }
  - meta:          { trace_id, started_at, tokens_total, latency_ms }
```

---

## 4. Agent 模块详细需求

### 4.1 研究层（Research Agents, 3 路并行）

#### 4.1.1 通用要求

- **输入**：`ticker`、`as_of_date`（默认今天）
- **数据源**：yfinance，按需拉取 `Ticker.info` / `history` / `financials` / `option_chain` / `institutional_holders` / `insider_transactions` / `upgrades_downgrades` 等
- **评分体系**：严格遵循 [`docs/analysismetrics.md`](analysismetrics.md) —— 每面 10 个指标，每指标 +1 / 0 / -1，汇总得分映射 BUY / HOLD / SELL
- **输出结构**（Pydantic）：
  - `indicator_scores`: `[{name, raw_value, score, reasoning}]`（10 项）
  - `total_score`: int（-10 ~ +10）
  - `verdict`: `BUY | HOLD | SELL`
  - `key_highlights`: 3-5 条核心亮点 / 风险
  - `confidence`: 0.0-1.0（基于数据完整度与指标一致性）

#### 4.1.2 Fundamental Agent

- 覆盖 analysismetrics.md 第一章的 10 个指标（PE、PB、PS、Revenue Growth、Gross Margin、Net Margin、ROE、D/E、FCF、Analyst Upside）
- **行业分档**：需先从 `info.sector` / `info.industry` 映射到 analysismetrics 中的行业分类（价值股 / 成长股 / 金融 / 消费等）再套评分区间
- **LLM 角色**：不自己"算"指标（用代码算），只做指标解读与综合定性

#### 4.1.3 Technical Agent

- 覆盖第二章 10 个指标（MACD、RSI、均线系统、布林带、Volume、ATR、斐波那契、RS、形态、支撑压力）
- 用 `pandas-ta`（或 `ta-lib`）基于 yfinance `history()` 自行计算衍生指标
- 默认使用近 1 年日线数据
- 形态识别（双底 / 头肩顶等）的判定用简化规则 + LLM 交叉验证，不追求复杂模式识别算法

#### 4.1.4 Sentimental Agent

- 覆盖第三章 10 个指标（Short Float、Insider Tx、机构持股、评级变化、P/C Ratio、回购、股息、社交情绪、52W 位置、Earnings Beat）
- **降级策略**：社交媒体指标（第 8 项）yfinance 无数据，**MVP 阶段直接标记为"数据不可用，计 0 分"**，不接入 Reddit / Stocktwits（写在文档里作为已知简化）
- 财报超预期历史若数据不足，同样降级为中性

#### 4.1.5 性能与并行

- 三个 Agent 通过 LangGraph 的并行分支同时执行
- 总体超时预算：90 秒（单个 Agent 硬超时 60 秒）

---

### 4.2 辩论层（Debate Agent）

#### 4.2.1 参与角色

| 角色 | 职责 | 温度参数 |
|------|------|----------|
| **Bull (多方)** | 从研究报告中提取看多证据，给出买入论证 | 0.7（鼓励发散） |
| **Bear (空方)** | 从研究报告中提取看空证据，给出卖出论证 | 0.7 |
| **Judge (裁判方)** | 综合所有辩论轮次，输出最终 BUY / HOLD / SELL + 理由 | 0.2（收敛、严谨） |

#### 4.2.2 流程

1. **首轮（Opening Round）**：Bull / Bear 各自接受完整 research 结果 → 独立给出首轮结论
2. **辩论轮（Rebuttal Rounds，默认 1 轮，可配置 0~3 轮）**：每轮中 Bull / Bear 各自接受对方上一轮结论 → 反驳 → 给出本轮结论
3. **终审（Closing）**：Judge 接收全部辩论历史（含 research 摘要），输出终局裁决

#### 4.2.3 输出结构

```
DebateTurn:
  - round: int (0=opening, 1+=rebuttal)
  - role: bull | bear
  - argument: str
  - cited_indicators: [str]      # 引用了哪些指标（便于追溯）
  - conclusion: BUY | HOLD | SELL

DebateResult:
  - turns: [DebateTurn]
  - final_verdict: BUY | HOLD | SELL
  - final_rationale: str
  - confidence: 0.0-1.0
  - proposed_position_pct: float  # 建议仓位占总资产百分比（0-10），传给 Risk Agent
```

#### 4.2.4 约束

- Bull / Bear 被 Prompt 强制"必须引用 research 中具体指标"，禁止凭空论证
- 若某面 research 因数据缺失降级，Judge 会收到降级标记，降级面不参与加权

---

### 4.3 风控层（Risk Agent）

#### 4.3.1 流程（严格顺序，短路失败）

```
[1] 硬规则检查 ──失败──→ REJECT (end)
      │ 通过
      ▼
[2] VaR 计算 ──超限──→ REJECT (end)
      │ 通过
      ▼
[3] LLM 软判断 ──高风险──→ REJECT (end)
      │ 通过
      ▼
    APPROVE → Execution Agent
```

#### 4.3.2 硬规则（确定性代码实现，不可绕过）

| 规则 | 阈值 | 判定逻辑 |
|------|------|----------|
| R1 单票仓位上限 | ≤ 10% 总资产 | `proposed_position_pct ≤ 10` |
| R2 组合最大回撤 | ≤ 8% | 基于 portfolio 历史 NAV 滚动计算 peak-to-trough |
| R3 VaR(95%) 上限 | 组合单日损失 ≤ 2% | `stock_var_95 × proposed_position_pct ≤ 2%` |
| R4 止损线 | 5% | 下单时附带 stop-loss 设置 = 入场价 × 0.95 |

#### 4.3.3 VaR 计算方法

- 使用历史模拟法：取近 252 个交易日日收益率序列，取 5% 分位数
- 若历史数据不足 60 个交易日 → **降级为 REJECT** 并标注原因
- 组合层面 VaR 暂按"单票 VaR × 持仓比例"近似，不考虑相关性（MVP 简化，写入 known limitation）

#### 4.3.4 软规则（LLM 判断，可否决）

LLM 输入：debate 结论 + 当前 portfolio 状态 + 最近市场波动数据（VIX、大盘最近 5 日涨跌等）

三个判断点：
- **市场极端波动**：如 VIX > 30，建议降低仓位至原提议的 50%
- **关联集中度**：当前持仓若有同行业股票累计 > 20% → 警告
- **流动性不足**：日均成交额 < $10M → 警告并限制最大下单量

LLM 输出结构化结果：
```
SoftJudgment:
  - flags: [MARKET_VOLATILE | CORRELATION_HIGH | LIQUIDITY_LOW]
  - adjustment: keep | reduce_to_X | reject
  - reasoning: str
```

#### 4.3.5 最终输出

```
RiskDecision:
  - decision: approve | reject
  - adjusted_position_pct: float   # 可能被软规则下调
  - hard_checks: [{rule, passed, detail}]
  - var_result: {var_95, portfolio_impact, passed}
  - soft_judgment: SoftJudgment
  - reject_reason: str | null
```

---

### 4.4 执行层（Execution Agent）

#### 4.4.1 前置条件

只有 `RiskDecision.decision == approve` 才会触发，否则节点直接跳过。

#### 4.4.2 执行模式

| 模式 | 行为 | 用途 |
|------|------|------|
| `dryrun`（默认） | 仅生成订单对象，写日志，不发 API | 演示 / 作品集 |
| `paper` | 提交 Alpaca Paper Trading API | 真实环境模拟 |

#### 4.4.3 流程

1. **计算下单量**：`shares = floor((capital × adjusted_position_pct) / current_price)`
2. **生成限价单**：limit_price = 当前价 × 1.002（buy）/ × 0.998（sell），TIF = DAY
3. **附加止损**：stop_price = 入场价 × (1 - 5%)（仅 buy 单）
4. **提交 / 记录**：
   - dryrun：写入 `logs/orders/*.json`
   - paper：调 Alpaca `submit_order`，轮询确认 fill（最多 30 秒）
5. **更新 Portfolio**：成交后写回 `portfolio.json`

#### 4.4.4 输出

```
ExecutionResult:
  - mode: dryrun | paper
  - order: { symbol, side, qty, type, limit_price, stop_price, tif }
  - status: generated | submitted | filled | partial | failed
  - fill_info: { avg_price, filled_qty, fill_time } | null
  - error: str | null
```

---

## 5. 技术选型评估

### 5.1 用户给定的选型

| 选型 | 评估 | 建议 |
|------|------|------|
| **LangChain + LangGraph** | ✅ 合理 | LangGraph 是多 Agent 编排的一等公民，状态机 + 条件分支 + 循环辩论天然契合。LangChain 部分只用其 Prompt 模板与（可选的）`ChatOpenAI` 指向 Ark base_url，避免过度依赖其工具生态。**也允许不引入 LangChain，仅用原生 `openai` SDK + LangGraph，以降低依赖面积** |
| **yfinance** | ✅ MVP 够用 | 免费、字段全；但稳定性一般、有限流。**必须**加本地缓存层（按 ticker+date 键缓存 24h）+ 指数退避重试 |
| **Alpaca Paper Trading** | ✅ 合理 | 美股 paper 首选，API 稳定、免费；需在 `.env` 管理 API Key，默认模式仍为 dryrun |

### 5.2 需补充的选型

| 位置 | 建议 | 理由 |
|------|------|------|
| **LLM Provider** | **火山引擎 Ark（OpenAI 兼容接口）** | Base URL: `https://ark.cn-beijing.volces.com/api/v3`；认证：`ARK_API_KEY` 环境变量；默认模型接入点 ID：`ep-20260417212516-rpphh`（可通过 `ARK_MODEL_ID` 或函数参数覆盖）；直接使用官方 `openai` SDK 或 `langchain-openai.ChatOpenAI` 指向该 base_url 即可 |
| **LLM 客户端封装** | 统一由 `python/llm/` 下的 `create_ark_client()` 与 `generate_ark_response(messages, model_id=None)` 提供 | 所有 Agent 只依赖这一层，不直接 import `openai`；便于统一加 trace、重试、token 统计 |
| **模型分层策略** | 单一默认接入点起步；若需分层，通过不同 `ARK_MODEL_ID_*` 环境变量（如 `ARK_MODEL_ID_ANALYSIS` / `ARK_MODEL_ID_JUDGE`）指向不同接入点 | Ark 的 "接入点 ID" 已绑定具体模型 + 参数，无需在调用侧区分模型名 |
| **结构化输出** | Pydantic v2 + Prompt 约束 + `response_format={"type": "json_object"}` | Ark OpenAI 兼容接口支持 JSON 模式；Pydantic 做 schema 校验，解析失败走降级重试 |
| **指标计算** | `pandas-ta`（优先）或 `ta-lib` | `pandas-ta` 纯 Python 无编译依赖，对作品集更友好 |
| **数据缓存** | SQLite + `diskcache`，或简单 parquet 文件 | 轻量、无需额外服务 |
| **可观测性** | 自建本地 JSONL trace（prompt / response / tokens / latency / endpoint_id / trace_id） | 不依赖 LangSmith；Ark 侧可在控制台查看调用明细作为补充 |
| **配置管理** | `pydantic-settings` + `.env` | `ARK_API_KEY` / `ARK_MODEL_ID` / Alpaca Key / 阈值统一管理 |
| **日志** | `structlog` + JSON 格式 | 便于生成最终 Markdown 报告 |
| **测试** | `pytest` + VCR.py（录制 yfinance 与 Ark 响应） | 确定性测试，避免每次跑测试都真实消耗 token |

### 5.3 目录结构建议（已存在雏形）

```
python/
  agents/          # 各 Agent 的 LLM 封装与 Prompt
    research/
    debate/
    risk/
    execution/
  graph/           # LangGraph 组装（节点、边、state）
  tools/           # yfinance 封装、指标计算、Alpaca 客户端
  llm/             # LLM 抽象、模型切换
  config/          # Settings、阈值配置
  logging_utils/   # trace、报告生成
  backtest/        # 单日回放（可选）
  tests/
  requirements.txt
```

---

## 6. 边界条件与异常处理

原则：**能降级不失败，降级必留痕**。所有异常与降级都写入最终报告的 `warnings` 字段。

### 6.1 数据层异常

| 场景 | 预期行为 |
|------|----------|
| Ticker 不存在 / 已退市 | 流水线在 research 层前置校验失败 → 直接终止，返回错误码 `INVALID_TICKER` |
| yfinance 请求限流 / 超时 | 指数退避重试 3 次，仍失败则 Agent 标记该指标为"数据不可用（0 分）" |
| 财报数据缺失（如新上市公司 < 4 季度） | 相关指标降级为 0 分，confidence 同步下调 |
| 期权链数据为空 | Put/Call Ratio 指标降级为 0 分 |
| 历史数据 < 60 交易日 | Risk Agent 拒绝（R3 VaR 无法可靠计算） |
| 非美股交易日 / 盘中停牌 | 使用最近一个交易日数据，报告里标注 `as_of_date` |

### 6.2 LLM 层异常

| 场景 | 预期行为 |
|------|----------|
| Ark 调用超时 | 重试 2 次（间隔 2s / 5s），仍失败则该 Agent 用降级模板（规则化汇总）输出 |
| `ARK_API_KEY` 未配置 / 鉴权失败 | 启动时前置校验，直接终止并提示 `export ARK_API_KEY=...`，不进入流水线 |
| 接入点 ID 无效（404 / 模型下线） | 明确错误提示，建议检查 `ARK_MODEL_ID` 或使用默认 `ep-20260417212516-rpphh` |
| 返回非合法 JSON | 先走一次"请仅返回 JSON，不要 markdown 代码块"的重试；仍失败则 Agent 进入降级模板 |
| Token 超限（上下文 / 输出） | Research 报告传递给 Debate 时做摘要压缩（只保留 verdict + 3 个关键指标）；辩论历史超长则只保留最近 2 轮 |
| Rate limit（429） | 指数退避；若 > 60s 仍未恢复，流水线中断并保存当前 state 供后续恢复 |

### 6.3 业务逻辑异常

| 场景 | 预期行为 |
|------|----------|
| 三面 research 全部降级（数据质量太差） | Debate 跳过，Judge 直接输出 HOLD + 警告 |
| Bull / Bear 某轮拒答或输出不合规 | 用上一轮结论占位，标注异常 |
| Debate 结论为 HOLD | Risk Agent 跳过仓位计算，Execution 层不动作，记录为"观望" |
| Risk Agent 软规则判 `reject` | Execution 跳过，报告写入 reject 理由 |
| Portfolio 文件不存在（首次运行） | 自动初始化为 `{capital, positions: [], nav_history: [capital]}` |
| Portfolio 回撤已达上限 | Risk Agent 一票否决所有 BUY，允许 SELL |

### 6.4 执行层异常

| 场景 | 预期行为 |
|------|----------|
| dryrun 模式 | 永远"成功"，只写日志 |
| Alpaca API 不可达 / 鉴权失败 | 订单状态标记为 `failed`，portfolio 不更新，报告附错误详情 |
| Paper 模式部分成交（partial fill） | 按实际成交量更新 portfolio，报告标注未成交部分 |
| 市场休市 | 订单排队到下一交易日（Alpaca 行为），报告标注 |

### 6.5 边界条件清单（用于测试用例）

1. ✅ 正常单票决策（AAPL，三面数据齐全）
2. ✅ 小盘股流动性警告（软规则触发）
3. ✅ VaR 超限的高波动股（R3 拒绝）
4. ✅ 辩论双方都给 HOLD → Judge 也 HOLD
5. ✅ 辩论 0 轮（只有首轮） / 3 轮
6. ✅ 新上市股票（基本面数据不足 → 降级）
7. ✅ Ticker 拼错 → 前置校验失败
8. ✅ Portfolio 首次运行（无历史 NAV）
9. ✅ 回撤已达上限时尝试 BUY → 拒绝
10. ✅ LLM 超时 → 降级模板

---

## 7. 范围外（Out of Scope）

以下明确不做，避免范围蔓延：

- ❌ 多股票组合优化（马科维茨、Black-Litterman 等）
- ❌ 实时流式行情（Level 2、逐笔）
- ❌ 期货 / 期权 / 加密货币 / 港股 / A 股
- ❌ 日内 / 高频策略
- ❌ 真实资金账户对接
- ❌ Web 前端 / 移动端
- ❌ 多用户 / 权限 / 计费系统
- ❌ 策略因子自动挖掘 / 强化学习
- ❌ 大规模历史回测框架（只做单点回放验证）
- ❌ 合规审计与监管报送

---

## 8. 验收标准（Acceptance Criteria）

项目视为 "MVP 完成" 需满足：

1. ✅ 一条命令跑通全流程：`python -m fishtrade run --ticker AAPL --capital 100000`
2. ✅ 三面 Agent 并行执行，输出均符合 analysismetrics.md 的评分规范
3. ✅ Debate 默认 1 轮辩论，可通过 `--debate-rounds N` 调整
4. ✅ Risk Agent 四条硬规则全部生效，有测试用例覆盖拒绝路径
5. ✅ dryrun / paper 两种模式可切换
6. ✅ 每次运行生成一份 `reports/{ticker}-{date}.md` 报告（含三面打分、辩论全文、风控判定、执行记录）
7. ✅ 本地 JSONL trace 可回看每次 Ark 调用（含 endpoint_id、messages、token、耗时）
8. ✅ README 包含架构图、跑通步骤、示例报告、已知简化项清单

---

## 9. 开放问题（待对齐）

以下事项需要在进入设计 / 编码阶段前明确：

1. **Ark 模型分层**：MVP 是否全程复用默认接入点 `ep-20260417212516-rpphh`？还是为 Judge / Risk 软规则单独开一个更强模型的接入点（通过 `ARK_MODEL_ID_JUDGE` 区分）？建议 MVP 先全程复用默认接入点，跑通后再按需拆分。
2. **是否引入 LangChain**：LangGraph 必用；LangChain（仅用其 `ChatOpenAI` + Prompt 模板）可选。不用则减少依赖层，自己封装调 `openai` SDK。倾向？
3. **Portfolio 初始状态**：首次运行是 0 持仓现金 10 万，还是允许从 JSON 文件导入既有持仓？
4. **回测是否纳入 MVP**：若纳入，简化为"指定历史日期单点回放"即可；若暂不纳入则归入 v2。
5. **报告输出语言**：中文 / 英文 / 双语？作品集场景下英文受众更广。

---

*文档版本：v1.0 — 对齐后进入 `docs/01-architecture.md` 与 `docs/02-agents-spec.md`*
