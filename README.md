# fishtrade — 多 Agent 量化交易决策系统

面向美股单票决策的多 Agent 编排示例：**三面分析 → 多空辩论 → 风控审核 → 执行下单**，
串联成一条可解释、可复盘、可暂停 / 恢复的 LLM 工程化流水线。基于 LangGraph + 火山引擎 Ark
（OpenAI 兼容接口），数据源使用 yfinance，执行层支持 dryrun / Alpaca Paper / backtest 三种模式。

> 这是一个作品集级 Demo，**不托管真实资金**，也不做策略 Alpha 验证。

---

## 1. 快速启动

### 1.1 环境要求

- Python `>= 3.11`
- 一个火山引擎 Ark 的 API Key（必填）
- 可选：Alpaca Paper Trading 的 API Key / Secret（仅 `--mode paper` 需要）

### 1.2 安装

```bash
# 克隆并进入项目
git clone <this-repo>
cd fishTradeDemo/python

# 安装依赖（建议先建虚拟环境）
pip install -r requirements.txt
pip install -e .
```

### 1.3 配置 .env

在项目根目录（与 `python/` 同级）新建一个 `.env` 文件，最少需要：

```bash
# === LLM（火山引擎 Ark，必填）===
ARK_API_KEY=your-ark-api-key-here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL_ID=ep-xxxxxxxxxxxxx-xxxxx     # 默认接入点（兜底）
# 可选：分别指定 research / debate / judge 接入点
# ARK_MODEL_ID_RESEARCH=
# ARK_MODEL_ID_DEBATE=
# ARK_MODEL_ID_JUDGE=
ARK_TIMEOUT_SECONDS=60
ARK_MAX_RETRIES=2

# === Alpaca Paper Trading（仅 paper 模式必需）===
# ALPACA_API_KEY=
# ALPACA_SECRET_KEY=
# ALPACA_BASE_URL=https://paper-api.alpaca.markets

# === 运行参数（默认值即可）===
FISHTRADE_DATA_DIR=./data
FISHTRADE_LOG_DIR=./logs
FISHTRADE_REPORT_DIR=./reports
FISHTRADE_LOG_LEVEL=INFO

# === 风控阈值（按需调整）===
RISK_MAX_POSITION_PCT=10.0
RISK_MAX_DRAWDOWN_PCT=8.0
RISK_VAR95_PORTFOLIO_LIMIT_PCT=2.0
RISK_STOPLOSS_PCT=5.0
```

> `data/` `logs/` `reports/` 目录程序会自动创建，已被 `.gitignore` 忽略。

### 1.4 自检环境

```bash
cd python
python -m fishtrade doctor
```

会逐项检查 ARK Key、Alpaca Key（仅警告）、运行目录可写、yfinance 网络可达。
如果 yfinance 探测较慢可加 `--skip-yfinance`。

### 1.5 跑通第一个用例

```bash
# dryrun 模式：全流程跑，订单只落日志不真实下单
python -m fishtrade run --ticker AAPL --capital 100000 --mode dryrun
```

成功后控制台会打印决策摘要表，并在 `reports/AAPL-<日期>.md` 生成一份双语 Markdown 报告。

### 1.6 其他常用命令

```bash
# HITL（人工确认）模式：风控通过后会暂停，等你输入 y/N
python -m fishtrade run --ticker AAPL --capital 100000 --mode dryrun --hitl

# 提交 paper 订单（需要先配 Alpaca key）
python -m fishtrade run --ticker AAPL --capital 100000 --mode paper

# 历史回放（按 as-of 截断数据）
python -m fishtrade run --ticker AAPL --mode backtest --as-of-date 2026-04-24

# 从中断的 run 恢复（需要 run-id，见上一次输出）
python -m fishtrade resume --run-id run-AAPL-2026-04-28-xxxxxxxx --decision approved

# 不重新调用 LLM，仅根据 trace 重渲染报告
python -m fishtrade replay --run-id run-AAPL-2026-04-28-xxxxxxxx
```

### 1.7 跑测试

```bash
cd python
pytest                       # 默认跳过需要真实网络的用例
pytest -m network            # 跑接入 yfinance / Ark 的用例（需要网络与 key）
```

---

## 2. 功能概览

| 模块 | 能力 |
|------|------|
| **三面研究** | 基本面 / 技术面 / 情绪面 各 10 个指标并行打分（详见 `docs/analysismetrics.md`） |
| **多空辩论** | Bull ↔ Bear 多轮对线（`--debate-rounds 0~3`），Judge 输出 `SELL / HOLD / BUY` 与建议仓位 |
| **风控分层** | 硬规则（仓位 / 回撤 / 黑名单）→ 历史模拟法 VaR → LLM 软规则三道闸 |
| **执行模式** | `dryrun`（仅日志）/ `paper`（Alpaca Paper API）/ `backtest`（按日期截断回放） |
| **HITL 暂停** | LangGraph SQLite Checkpoint 支持任意节点暂停 / 恢复，配合 `resume` 子命令复用 run-id |
| **持久化产物** | `data/portfolio.json` 组合状态、`logs/trace/{run_id}.jsonl` 全链路 trace、`reports/*.md` 双语报告 |
| **可观测性** | structlog 结构化日志、节点级 start / complete / failed 事件、tokens / latency 聚合 |
| **健壮性** | yfinance 24h diskcache + 限流退避、Ark 重试、msgpack 安全的 checkpoint 序列化、缺数据降级评分 |
| **CLI** | Typer + Rich，自带 `doctor` 自检、`run` / `resume` / `replay` 三个子命令 |

---

## 3. 架构

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

### 3.1 编排骨架

整张图由 [`python/fishtrade/graph/builder.py`](python/fishtrade/graph/builder.py) 用 **LangGraph `StateGraph`** 装配：

- `validate_input → fetch_market` 入口与数据拉取（yfinance + 缓存）
- `research_fund / research_tech / research_sent` **并行 fan-out**，三路 LLM 评分
- `debate_open_bull / debate_open_bear → debate_rebuttal → debate_judge` 多轮辩论后由 Judge 给最终判定
- `risk_hard → risk_var → risk_soft` 串行风控链，任一拒绝直接跳到报告渲染
- `hitl_gate` 通过 `interrupt_before` 实现可选人工确认（CLI `resume` 可恢复）
- `execute_router_dispatch` 按 `mode` 路由到 `dryrun / paper / backtest / skip_execution`
- `update_portfolio → render_report → END` 写组合状态、产出双语 Markdown

### 3.2 模块分层

```
python/fishtrade/
├── cli.py / __main__.py        # Typer CLI：doctor / run / resume / replay
├── config/                      # Pydantic Settings（.env） + 风控阈值常量
├── models/                      # 全局 Pydantic schema：state / research / debate / risk / execution / portfolio
├── llm/                         # Ark 客户端、retry 装饰器、token 计数、prompt 模板
├── tools/                       # 纯函数工具：yfinance 包装与缓存、三面指标计算、VaR、行业分档
├── agents/                      # LangGraph 节点函数（输入 state → 输出 state patch）
│   ├── research/                # fundamental / technical / sentimental
│   ├── debate/                  # bull / bear / judge
│   ├── risk/                    # hard_rules / var_check / soft_judge
│   └── execution/               # router / dryrun / paper / backtest / update_portfolio
├── graph/                       # LangGraph 装配：builder / routes / SQLite checkpoint
├── portfolio/                   # 组合持久化（portfolio.json） + NAV 序列与回撤
├── observability/               # structlog 配置、JSONL trace writer、节点级日志包装
└── reporting/                   # GraphState → 双语 Markdown 渲染（Jinja2 模板）
```

### 3.3 关键设计点

- **State 单向追加**：节点只往自己负责的字段写 patch；list 字段用 `Annotated[list, add]` reducer 让并行节点输出自动合并，避免写冲突。
- **Checkpoint 友好的序列化**：所有 DataFrame 经 `_df_to_payload` 序列化，`info` 经 JSON round-trip 剥离 `pandas.Timestamp` 等非原语类型，保证 LangGraph msgpack writer 不会爆。
- **降级而非崩溃**：yfinance 限流 / 字段缺失只追加 `warnings`，节点继续推进，最终在报告里如实标注。
- **风控前置 + 短路**：硬规则在 LLM 之前就能挡住明显违规，省 token 也省时间。
- **HITL 通过 LangGraph 原生实现**：`interrupt_before=["hitl_gate"]` + `SqliteSaver`，`resume` 子命令复用同一个 `run_id` 注入决策即可继续。

---

## 4. 进一步阅读

- 需求文档：[docs/00-requirements.md](docs/00-requirements.md)
- 技术设计：[docs/01-tech-design.md](docs/01-tech-design.md)
- 三面指标定义：[docs/analysismetrics.md](docs/analysismetrics.md)
