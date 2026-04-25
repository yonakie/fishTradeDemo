# 多 Agent 量化交易决策系统 — 技术设计文档 v1.0

> 文档编号：01-tech-design
> 上游依赖：[`docs/00-requirements.md`](00-requirements.md) v1.1、[`docs/analysismetrics.md`](analysismetrics.md)
> 更新日期：2026-04-25
> 状态：Draft for engineering kickoff
> 目标读者：后端开发、QA、Code Reviewer

---

## 0. 阅读导航

| 章节 | 内容 | 预计阅读时间 |
|------|------|--------------|
| 一 | 项目全局视图与工程结构 | 8 min |
| 二 | 全局状态与数据模型 (Pydantic) | 15 min |
| 三 | 基础设施层（LLM、Trace、缓存、Portfolio） | 12 min |
| 四 | Agent 模块详细设计 | 20 min |
| 五 | LangGraph 编排与工作流装配 | 10 min |
| 六 | CLI 入口与后处理 | 5 min |
| 七 | 第一阶段开发 Checklist | 5 min |

> ⚠️ **本设计文档是开发蓝图**：所有数据结构、节点签名、异常码均为契约级，调整必须同步修改本文。

---

# 一、项目全局视图与工程结构

## 1.1 核心技术栈与依赖清单

| 类别 | 库 | 版本（与 `requirements.txt` 对齐） | 用途 |
|------|----|-----------------------------------|------|
| **编排框架** | `langgraph` | 1.1.9 | StateGraph、并行节点、条件边、Checkpoint |
| | `langgraph-checkpoint` | 4.0.2 | HITL 暂停 / 恢复（SQLite saver） |
| | `langchain-openai` | 1.2.1 | `ChatOpenAI` 指向 Ark base_url（仅 Prompt 模板用） |
| **LLM SDK** | `openai` | 2.32.0 | 直连 Ark OpenAI 兼容接口 |
| **数据建模** | `pydantic` | 2.13.3 | v2 schema、`response_format` 校验 |
| | `pydantic-settings` | 需补充安装 | `.env` → `Settings` 注入 |
| **行情数据** | `yfinance` | 1.3.0 | 单票全字段拉取 |
| | `pandas` | 3.0.2 | OHLCV 处理 |
| | `pandas-ta` | 0.4.71b0 | MACD / RSI / 布林带 / ATR 等技术指标 |
| **持久化与缓存** | `diskcache`（需补充） | 最新 | yfinance 24h 本地缓存 |
| | `peewee` | 4.0.5 | trace / portfolio JSONL 索引（轻量） |
| **执行层** | `alpaca-py`（需补充） | 0.30+ | Paper Trading API |
| **观测与日志** | `structlog` | 25.5.0 | 结构化 JSON 日志 |
| | `tenacity` | 9.1.4 | 退避重试装饰器 |
| | `tiktoken` | 0.12.0 | Token 计数（Ark 兼容 cl100k 计算） |
| **CLI** | `typer`（需补充） 或 `argparse` | — | 命令行 |
| | `rich` | 15.0.0 | 彩色终端、进度条 |
| **测试** | `pytest` / `pytest-mock` / `responses` | 9.x / 3.x / 0.26 | mock yfinance & Ark |

**待新增 `requirements.txt` 条目**：`pydantic-settings`、`diskcache`、`alpaca-py`、`typer`、`vcrpy`（用于 Ark 录制回放）。

## 1.2 项目目录结构

```
fishtrade/
├── .env.example                  # 环境变量模板
├── .gitignore
├── README.md                     # 跑通步骤、架构图、示例报告链接
├── pyproject.toml                # 包元数据 + tool.pytest 配置
├── docs/
│   ├── 00-requirements.md
│   ├── 01-tech-design.md        ← 本文
│   └── analysismetrics.md
├── python/
│   ├── requirements.txt
│   ├── fishtrade/                # ← 主包（PEP 517 安装目标）
│   │   ├── __init__.py
│   │   ├── __main__.py           # 支持 `python -m fishtrade`
│   │   ├── cli.py                # CLI 参数解析与入口
│   │   │
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py       # Pydantic Settings：API Key、阈值、超时
│   │   │   └── thresholds.py     # 风控阈值常量（R1/R2/R3/R4 默认值）
│   │   │
│   │   ├── models/               # ← 全局 Pydantic schema（无业务逻辑）
│   │   │   ├── __init__.py
│   │   │   ├── state.py          # GraphState TypedDict
│   │   │   ├── research.py       # IndicatorScore / ResearchReport
│   │   │   ├── debate.py         # DebateTurn / DebateResult
│   │   │   ├── risk.py           # RiskDecision / SoftJudgment
│   │   │   ├── execution.py      # Order / ExecutionResult
│   │   │   └── portfolio.py      # Position / Portfolio / NavSnapshot
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── client.py         # create_ark_client / generate_ark_response
│   │   │   ├── factory.py        # 按 role 取不同 ARK_MODEL_ID 接入点
│   │   │   ├── retry.py          # tenacity 装饰器：超时/429/JSON 解析失败
│   │   │   ├── token_counter.py  # tiktoken 包装
│   │   │   └── prompts/          # Jinja2 / 文本模板
│   │   │       ├── fundamental.md
│   │   │       ├── technical.md
│   │   │       ├── sentimental.md
│   │   │       ├── debate_bull.md
│   │   │       ├── debate_bear.md
│   │   │       ├── debate_judge.md
│   │   │       └── risk_soft.md
│   │   │
│   │   ├── tools/                # ← 纯函数 / 数据工具（不调 LLM）
│   │   │   ├── __init__.py
│   │   │   ├── yf_client.py      # yfinance 缓存包装
│   │   │   ├── yf_cache.py       # diskcache 实现，键：ticker+endpoint+date
│   │   │   ├── indicators_fund.py    # 基本面 10 指标计算与评分
│   │   │   ├── indicators_tech.py    # 技术面 10 指标（pandas-ta）
│   │   │   ├── indicators_sent.py    # 情绪面 10 指标
│   │   │   ├── industry_classifier.py# sector → 行业分档（价值/成长/金融…）
│   │   │   ├── var_calculator.py     # 历史模拟法 VaR
│   │   │   ├── alpaca_client.py      # Paper Trading 包装
│   │   │   └── feature_flags.py      # 财报数据是否充足、停牌等判定
│   │   │
│   │   ├── agents/               # ← LangGraph 节点函数（输入 state → 输出 state patch）
│   │   │   ├── __init__.py
│   │   │   ├── research/
│   │   │   │   ├── fundamental.py
│   │   │   │   ├── technical.py
│   │   │   │   └── sentimental.py
│   │   │   ├── debate/
│   │   │   │   ├── bull.py
│   │   │   │   ├── bear.py
│   │   │   │   └── judge.py
│   │   │   ├── risk/
│   │   │   │   ├── hard_rules.py
│   │   │   │   ├── var_check.py
│   │   │   │   └── soft_judge.py
│   │   │   └── execution/
│   │   │       ├── router.py     # 按 mode 分流 dryrun / paper / backtest
│   │   │       ├── dryrun.py
│   │   │       ├── paper.py
│   │   │       └── backtest.py
│   │   │
│   │   ├── graph/                # ← LangGraph 装配
│   │   │   ├── __init__.py
│   │   │   ├── builder.py        # build_graph()：节点注册、边、条件路由
│   │   │   ├── routes.py         # 条件边逻辑函数（route_after_risk 等）
│   │   │   └── checkpoint.py     # SqliteSaver 实例（HITL）
│   │   │
│   │   ├── portfolio/
│   │   │   ├── __init__.py
│   │   │   ├── store.py          # 读写 data/portfolio.json
│   │   │   └── nav.py            # 滚动 NAV 序列与最大回撤
│   │   │
│   │   ├── observability/
│   │   │   ├── __init__.py
│   │   │   ├── trace.py          # JSONL trace writer（每次 LLM 调用一行）
│   │   │   ├── logger.py         # structlog 配置
│   │   │   └── metrics.py        # 简单聚合（tokens_total、latency_ms）
│   │   │
│   │   └── reporting/
│   │       ├── __init__.py
│   │       ├── render.py         # GraphState → Markdown
│   │       └── templates/
│   │           ├── report_zh.md.j2
│   │           └── report_en.md.j2
│   │
│   └── tests/
│       ├── conftest.py
│       ├── fixtures/             # VCR cassettes、yfinance 样例 JSON
│       ├── unit/
│       │   ├── test_indicators_fund.py
│       │   ├── test_indicators_tech.py
│       │   ├── test_var_calculator.py
│       │   ├── test_yf_cache.py
│       │   └── test_models_schemas.py
│       ├── integration/
│       │   ├── test_research_node.py
│       │   ├── test_debate_loop.py
│       │   ├── test_risk_pipeline.py
│       │   └── test_graph_e2e.py
│       └── boundary/             # 对应需求 6.5 节 10 个 case
│           ├── test_invalid_ticker.py
│           ├── test_drawdown_limit.py
│           └── test_llm_timeout_fallback.py
│
├── data/                         # ← 运行时本地状态（gitignore）
│   ├── portfolio.json            # 当前组合（首次自动初始化）
│   ├── nav_history.jsonl         # 每日净值序列
│   └── cache/                    # diskcache 目录
│
├── logs/                         # ← 运行时日志与 trace（gitignore）
│   ├── trace/{run_id}.jsonl
│   └── orders/{run_id}.json
│
└── reports/                      # ← Markdown 决策报告（gitignore）
    └── {ticker}-{as_of_date}.md
```

## 1.3 环境变量与配置文件

### 1.3.1 `.env.example`（提交版本控制；真实 `.env` 不入库）

```bash
# === LLM（火山引擎 Ark）===
ARK_API_KEY=your-ark-api-key-here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL_ID=ep-20260417212516-rpphh           # 默认接入点（兜底）
ARK_MODEL_ID_RESEARCH=                         # 可选：研究层
ARK_MODEL_ID_DEBATE=                           # 可选：辩论层
ARK_MODEL_ID_JUDGE=                            # 可选：裁判 / 风控软规则
ARK_TIMEOUT_SECONDS=60
ARK_MAX_RETRIES=2

# === Alpaca Paper Trading（仅 paper 模式必需）===
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# === 运行参数 ===
FISHTRADE_DATA_DIR=./data
FISHTRADE_LOG_DIR=./logs
FISHTRADE_REPORT_DIR=./reports
FISHTRADE_LOG_LEVEL=INFO

# === 缓存 ===
YF_CACHE_TTL_SECONDS=86400                     # 24h
YF_RATE_LIMIT_BACKOFF_BASE=2

# === 业务阈值（可被 thresholds.py 覆盖；这里仅暴露常调项）===
RISK_MAX_POSITION_PCT=10.0
RISK_MAX_DRAWDOWN_PCT=8.0
RISK_VAR95_PORTFOLIO_LIMIT_PCT=2.0
RISK_STOPLOSS_PCT=5.0
```

### 1.3.2 首次运行初始化命令

```bash
# 1. 安装依赖
cd python && pip install -r requirements.txt -e .

# 2. 准备 .env
cp ../.env.example ../.env && vim ../.env

# 3. 创建运行时目录（程序也会自动建）
mkdir -p data logs/trace logs/orders reports data/cache

# 4. 验证 Ark 连通性
python -m fishtrade doctor          # 自检：env、ark ping、yfinance 拉一次 AAPL.info

# 5. 跑通最小用例（dryrun，不消耗真实订单）
python -m fishtrade run --ticker AAPL --capital 100000 --mode dryrun
```

`doctor` 子命令是显式的前置校验：检查 `ARK_API_KEY` 是否存在、调用一次最小消息测连通、拉一次 `yf.Ticker("AAPL").info`，三项任一失败即终止并给出修复指引（对应需求 6.2 "鉴权失败前置校验"）。

---

# 二、全局状态与数据模型定义

## 2.1 GraphState：贯穿全流程的"账本"

> 设计原则：
> 1. **单向追加优先**：节点只往 state 写自己负责的字段，不修改其他字段（避免并行写冲突）。
> 2. **LangGraph reducer 兼容**：所有 list 字段使用 `Annotated[list, operator.add]`，让并行节点的输出自动合并。
> 3. **结构化但允许中间为 None**：尚未执行的阶段对应字段为 `None`，便于条件路由判断"是否已完成"。

文件：[`python/fishtrade/models/state.py`](python/fishtrade/models/state.py)

```python
from __future__ import annotations
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from operator import add

from .research import ResearchReport
from .debate import DebateResult, DebateTurn
from .risk import RiskDecision
from .execution import ExecutionResult
from .portfolio import PortfolioSnapshot


class RunInput(TypedDict):
    ticker: str
    capital: float
    mode: Literal["dryrun", "paper", "backtest"]
    debate_rounds: int                # 0~3
    as_of_date: str                   # ISO date "YYYY-MM-DD"
    language: Literal["zh", "en", "bilingual"]
    hitl: bool                        # True 时风控通过后挂起等用户确认


class MarketDataBundle(TypedDict, total=False):
    """yfinance 原始数据快照；按需懒加载，节点只读自己用得到的子集。"""
    info: dict
    history: dict                     # serialized DataFrame (orient='split')
    financials: dict
    cashflow: dict
    balance_sheet: dict
    options_chain: dict | None
    institutional_holders: dict | None
    insider_transactions: dict | None
    upgrades_downgrades: dict | None
    earnings_dates: dict | None
    benchmark_history: dict           # SPY 同期数据，用于 RS 计算
    vix_recent: dict                  # 最近 5 日 VIX，用于风控软规则
    fetch_warnings: list[str]         # 拉取过程中的降级标记


class ResearchSection(TypedDict, total=False):
    fundamental: ResearchReport | None
    technical:   ResearchReport | None
    sentimental: ResearchReport | None


class GraphState(TypedDict, total=False):
    # —— 入口 ——
    input: RunInput

    # —— 数据层 ——
    market_data: MarketDataBundle

    # —— 研究层（三路并行写入）——
    research: ResearchSection

    # —— 辩论层 ——
    debate_turns: Annotated[list[DebateTurn], add]   # 跨节点累加
    debate: DebateResult | None

    # —— 风控层 ——
    risk: RiskDecision | None

    # —— 执行层 ——
    execution: ExecutionResult | None

    # —— 组合 ——
    portfolio_before: PortfolioSnapshot | None
    portfolio_after:  PortfolioSnapshot | None

    # —— 元数据 ——
    run_id: str
    started_at: str
    warnings: Annotated[list[str], add]              # 全流程降级标记
    tokens_total: int
    latency_ms_total: int
    halt_reason: Optional[str]                       # 非空表示流水线被中断
```

> **为什么用 TypedDict 而非 BaseModel 作 GraphState**：LangGraph 的 channel reducer 对 `Annotated[list, operator.add]` 原生支持 TypedDict；用 BaseModel 反而要额外写 `__merge__`。而**节点产出**仍然用 Pydantic v2 BaseModel（更严格的校验），写入 state 时再 `.model_dump()`。

## 2.2 Pydantic Schemas

### 2.2.1 Research 层

文件：[`python/fishtrade/models/research.py`](python/fishtrade/models/research.py)

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

Verdict = Literal["BUY", "HOLD", "SELL"]


class IndicatorScore(BaseModel):
    """单个指标评分。10 项指标的契约，与 docs/analysismetrics.md 严格对齐。"""
    name: str = Field(..., description="指标 ID，如 'PE_RATIO'、'MACD'、'SHORT_FLOAT'")
    display_name_zh: str
    display_name_en: str
    raw_value: float | str | None = Field(
        None, description="原始数值；缺失时为 None 并 score=0、is_degraded=True"
    )
    score: Literal[-1, 0, 1]
    reasoning: str = Field(..., max_length=400)
    is_degraded: bool = False
    degrade_reason: str | None = None


class ResearchReport(BaseModel):
    """单面研究产出（基本面 / 技术面 / 情绪面共用）。"""
    facet: Literal["fundamental", "technical", "sentimental"]
    ticker: str
    as_of_date: str

    indicator_scores: list[IndicatorScore] = Field(..., min_length=10, max_length=10)
    total_score: int = Field(..., ge=-10, le=10)
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)

    key_highlights: list[str] = Field(..., min_length=3, max_length=5)
    industry_class: str | None = None    # 仅 fundamental 用：'value' / 'growth' / 'financial' / ...

    is_facet_degraded: bool = False      # 整面降级（>= 5 项 is_degraded 时触发）
    degrade_summary: str | None = None

    @model_validator(mode="after")
    def _check_total_matches_scores(self):
        s = sum(i.score for i in self.indicator_scores)
        if s != self.total_score:
            raise ValueError(f"total_score {self.total_score} ≠ Σscores {s}")
        return self

    @model_validator(mode="after")
    def _verdict_matches_score(self):
        # 与 analysismetrics.md "汇总评分" 对齐
        v_expected = "BUY" if self.total_score >= 5 else (
                     "HOLD" if self.total_score >= 1 else "SELL")
        if self.verdict != v_expected:
            raise ValueError(f"verdict {self.verdict} 与 total_score {self.total_score} 不一致")
        return self
```

**指标命名表**（与 `analysismetrics.md` 对齐，配套放在 [`indicators_fund.py`](python/fishtrade/tools/indicators_fund.py) 等文件顶部 `INDICATOR_REGISTRY`）：

| 面 | 指标 ID（10 项） |
|----|------------------|
| Fundamental | `PE_RATIO`, `PB_RATIO`, `PS_RATIO`, `REVENUE_GROWTH`, `GROSS_MARGIN`, `NET_MARGIN`, `ROE`, `DEBT_TO_EQUITY`, `FREE_CASHFLOW`, `ANALYST_UPSIDE` |
| Technical | `MACD`, `RSI_14`, `MOVING_AVERAGES`, `BOLLINGER`, `VOLUME_TREND`, `ATR_14`, `FIBONACCI`, `RELATIVE_STRENGTH`, `PRICE_PATTERN`, `SUPPORT_RESISTANCE` |
| Sentimental | `SHORT_FLOAT`, `INSIDER_TX`, `INSTITUTIONAL_HOLD`, `ANALYST_RATING`, `OPTIONS_PCR`, `BUYBACK`, `DIVIDEND`, `RETAIL_SOCIAL`, `WEEK52_POSITION`, `EARNINGS_BEAT` |

### 2.2.2 Debate 层

文件：[`python/fishtrade/models/debate.py`](python/fishtrade/models/debate.py)

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

Verdict = Literal["BUY", "HOLD", "SELL"]


class DebateTurn(BaseModel):
    round: int = Field(..., ge=0, le=3)            # 0=opening
    role: Literal["bull", "bear"]
    argument: str = Field(..., max_length=2000)
    cited_indicators: list[str] = Field(
        ..., min_length=1,
        description="必须从 ResearchReport.indicator_scores.name 中选取"
    )
    conclusion: Verdict
    is_fallback: bool = False                       # LLM 拒答 / 解析失败时占位


class DebateResult(BaseModel):
    turns: list[DebateTurn]
    final_verdict: Verdict
    final_rationale: str = Field(..., max_length=2000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    proposed_position_pct: float = Field(
        ..., ge=0.0, le=10.0,
        description="建议仓位占总资产百分比，HOLD/SELL 时为 0"
    )
    degraded_facets: list[Literal["fundamental", "technical", "sentimental"]] = []
```

### 2.2.3 Risk 层

文件：[`python/fishtrade/models/risk.py`](python/fishtrade/models/risk.py)

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class HardCheckResult(BaseModel):
    rule: Literal["R1_POSITION_LIMIT", "R2_MAX_DRAWDOWN", "R3_VAR95", "R4_STOPLOSS"]
    passed: bool
    actual: float | None = None
    threshold: float | None = None
    detail: str


class VarResult(BaseModel):
    var_95: float = Field(..., description="单票日 VaR(95)，如 0.034 = 3.4%")
    portfolio_impact: float = Field(..., description="加权后组合影响，如 0.0034")
    passed: bool
    sample_size: int                                # 用了多少日收益率
    method: Literal["historical_simulation"] = "historical_simulation"
    fallback_reason: str | None = None              # 数据不足时


class SoftJudgment(BaseModel):
    flags: list[Literal[
        "MARKET_VOLATILE", "CORRELATION_HIGH", "LIQUIDITY_LOW", "NONE"
    ]]
    adjustment: Literal["keep", "reduce", "reject"]
    adjusted_position_pct: float = Field(..., ge=0.0, le=10.0)
    reasoning: str = Field(..., max_length=1000)


class RiskDecision(BaseModel):
    decision: Literal["approve", "reject"]
    adjusted_position_pct: float
    hard_checks: list[HardCheckResult]
    var_result: VarResult
    soft_judgment: SoftJudgment
    reject_reason: str | None = None
```

### 2.2.4 Execution 层

文件：[`python/fishtrade/models/execution.py`](python/fishtrade/models/execution.py)

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Order(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: int = Field(..., gt=0)
    order_type: Literal["limit"] = "limit"
    limit_price: float
    stop_price: float | None = None                  # 仅 buy 单
    tif: Literal["day"] = "day"


class FillInfo(BaseModel):
    avg_price: float
    filled_qty: int
    fill_time: str                                   # ISO


class ExecutionResult(BaseModel):
    mode: Literal["dryrun", "paper", "backtest"]
    order: Order | None
    status: Literal["generated", "submitted", "filled", "partial", "failed", "skipped"]
    fill_info: FillInfo | None = None
    error: str | None = None
    broker_order_id: str | None = None
```

### 2.2.5 Portfolio

文件：[`python/fishtrade/models/portfolio.py`](python/fishtrade/models/portfolio.py)

```python
from __future__ import annotations
from pydantic import BaseModel, Field


class Position(BaseModel):
    symbol: str
    qty: int
    avg_cost: float
    sector: str | None = None


class NavSnapshot(BaseModel):
    date: str
    nav: float


class PortfolioSnapshot(BaseModel):
    cash: float
    positions: list[Position] = Field(default_factory=list)
    nav: float                                       # cash + Σ qty × last_price
    nav_history: list[NavSnapshot] = Field(default_factory=list)
    max_drawdown_pct: float = 0.0
```

### 2.2.6 与 `analysismetrics.md` 的对齐保证

每个指标在工具层有一个**纯函数**实现 + **行业分档表**：

```python
# python/fishtrade/tools/indicators_fund.py
INDICATOR_REGISTRY: dict[str, IndicatorSpec] = {
    "PE_RATIO": IndicatorSpec(
        zh="市盈率", en="PE Ratio",
        fields=["trailingPE", "forwardPE"],
        scorer=score_pe_ratio,                       # (raw, industry_class) -> (score, reasoning)
    ),
    ...
}
```

`scorer` 严格按 `analysismetrics.md` 的"分类阈值表"实现 if/else。新增指标或调整阈值，只改 `INDICATOR_REGISTRY`，不动 Agent 代码。

---

# 三、基础设施层设计

## 3.1 LLM 客户端工厂

文件：[`python/fishtrade/llm/client.py`](python/fishtrade/llm/client.py)、[`python/fishtrade/llm/factory.py`](python/fishtrade/llm/factory.py)

```python
# client.py — 唯一的 Ark 出口
from openai import OpenAI
from pydantic import BaseModel
from typing import TypeVar, Type
from .retry import ark_retry
from ..observability.trace import write_llm_trace
from ..config.settings import settings

T = TypeVar("T", bound=BaseModel)


def create_ark_client() -> OpenAI:
    """单例 OpenAI 客户端，base_url 指向 Ark。"""
    return OpenAI(
        api_key=settings.ark_api_key,
        base_url=settings.ark_base_url,
        timeout=settings.ark_timeout_seconds,
        max_retries=0,                # 重试由 tenacity 统一管，避免双层
    )


@ark_retry                            # 见 retry.py：超时/429/JSON 解析失败重试
def generate_ark_response(
    messages: list[dict],
    *,
    role: str = "default",            # 'research' / 'debate' / 'judge'
    temperature: float = 0.2,
    response_schema: Type[T] | None = None,
    run_id: str,
    node_name: str,
) -> T | str:
    """
    统一封装：
      1. 按 role 解析 model_id（factory.resolve_model_id）
      2. 若有 response_schema，注入 JSON Schema 到 system prompt + 设置 response_format
      3. 调用后写 trace
      4. 解析为 Pydantic 实例（失败抛 JSONParseError → 装饰器重试一次）
    """
    client = create_ark_client()
    model_id = resolve_model_id(role)

    kwargs = {"model": model_id, "messages": messages, "temperature": temperature}
    if response_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}
        messages = _inject_schema_hint(messages, response_schema)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    write_llm_trace(
        run_id=run_id, node=node_name, model_id=model_id,
        prompt=messages, response=resp.model_dump(),
        usage=resp.usage.model_dump() if resp.usage else None,
        latency_ms=latency_ms,
    )

    raw = resp.choices[0].message.content
    if response_schema is None:
        return raw
    try:
        return response_schema.model_validate_json(raw)
    except ValidationError as e:
        raise JSONParseError(raw=raw, schema=response_schema, errors=e.errors())
```

```python
# factory.py — role 到 ARK_MODEL_ID_* 的解析
def resolve_model_id(role: str) -> str:
    mapping = {
        "research": settings.ark_model_id_research,
        "debate":   settings.ark_model_id_debate,
        "judge":    settings.ark_model_id_judge,
    }
    return mapping.get(role) or settings.ark_model_id     # 兜底默认接入点
```

**核心约定**：
- 所有 Agent **只 import** `generate_ark_response`，不 import `openai`。
- 重试策略放在 `retry.py`：`tenacity.retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type((APITimeoutError, RateLimitError, JSONParseError)))`。
- JSON 解析失败时，重试装饰器会**追加一条 system message**："上次返回不是合法 JSON，请仅返回 JSON 对象，不要 markdown 代码块"。

## 3.2 Trace 与日志持久化

文件：[`python/fishtrade/observability/trace.py`](python/fishtrade/observability/trace.py)

每次 LLM 调用追加一行 JSONL 到 `logs/trace/{run_id}.jsonl`：

```json
{"ts": "2026-04-25T10:12:33Z", "run_id": "...", "node": "fundamental_agent",
 "model_id": "ep-...", "prompt_tokens": 1820, "completion_tokens": 412,
 "latency_ms": 4321, "endpoint": "chat.completions",
 "messages_hash": "sha256:...", "ok": true}
```

完整 prompt / response 单独存 `logs/trace/{run_id}_messages.jsonl`（按 `messages_hash` 去重）。`structlog` 全局配置见 [`logger.py`](python/fishtrade/observability/logger.py)。

## 3.3 yfinance 数据缓存层

文件：[`python/fishtrade/tools/yf_client.py`](python/fishtrade/tools/yf_client.py)、[`yf_cache.py`](python/fishtrade/tools/yf_cache.py)

**职责**：
1. 同一 `(ticker, endpoint, as_of_date)` 24h 内只调用一次。
2. 限流（`yfinance` 不抛 429，但会返回空 DataFrame / 抛 `requests.exceptions.HTTPError`）触发指数退避。
3. 暴露**字段级降级**：拿不到 `option_chain` 不要让整面 research 失败。

核心接口（伪代码）：

```python
from diskcache import Cache

class YFinanceClient:
    def __init__(self, cache_dir: str, ttl: int = 86_400):
        self.cache = Cache(cache_dir)
        self.ttl = ttl

    @cached(key_fn=lambda t, d: f"info::{t}::{d}")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_info(self, ticker: str, as_of: str) -> dict: ...

    @cached(key_fn=lambda t, p, d: f"hist::{t}::{p}::{d}")
    def get_history(self, ticker: str, period: str = "1y", as_of: str = ...) -> pd.DataFrame: ...

    def get_option_chain_safe(self, ticker: str) -> dict | None:
        """期权链可能为空 → 返回 None，调用方记 fetch_warnings。"""
        try:
            return yf.Ticker(ticker).option_chain()._asdict()
        except (IndexError, AttributeError):
            return None

    def fetch_bundle(self, ticker: str, as_of: str) -> MarketDataBundle:
        """一次性拉齐 research 三面所需数据，并把降级标记写入 fetch_warnings。"""
        ...
```

**前置校验**：`fetch_bundle` 第一步是 `Ticker.info`，若 `regularMarketPrice` 为 None 或 `quoteType` 为空，立即抛 `InvalidTickerError` → CLI 层捕获并以错误码 `INVALID_TICKER` 退出（对应需求 6.1）。

## 3.4 Portfolio 状态管理

文件：[`python/fishtrade/portfolio/store.py`](python/fishtrade/portfolio/store.py)、[`nav.py`](python/fishtrade/portfolio/nav.py)

```python
class PortfolioStore:
    """文件后端 + 写时复制，避免半写状态。"""
    PATH = Path("data/portfolio.json")
    NAV_PATH = Path("data/nav_history.jsonl")

    def load(self, capital_default: float) -> PortfolioSnapshot:
        if not self.PATH.exists():
            snap = PortfolioSnapshot(cash=capital_default, positions=[], nav=capital_default)
            self.save_atomic(snap)
            return snap
        return PortfolioSnapshot.model_validate_json(self.PATH.read_text())

    def save_atomic(self, snap: PortfolioSnapshot) -> None:
        tmp = self.PATH.with_suffix(".tmp")
        tmp.write_text(snap.model_dump_json(indent=2))
        tmp.replace(self.PATH)                       # POSIX 原子；Windows 也能用

    def append_nav(self, date: str, nav: float) -> None: ...

def compute_max_drawdown(nav_series: list[NavSnapshot]) -> float:
    """peak-to-trough，返回正百分比（如 0.087 = 8.7%）。"""
```

R2（最大回撤 ≤ 8%）由 `compute_max_drawdown` 直接消费 `nav_history.jsonl`。首次运行 `nav_history` 只有一行 → drawdown = 0 → 自动通过 R2，符合需求 6.3 "首次运行" 行为。

---

# 四、Agent 模块详细设计

## 4.1 Research 层（3 路并行）

### 4.1.1 节点函数签名（统一规范）

```python
# python/fishtrade/agents/research/fundamental.py
def fundamental_node(state: GraphState) -> dict:
    """
    Returns: {'research': {'fundamental': ResearchReport.model_dump()},
              'warnings': [...], 'tokens_total': int}
    LangGraph reducer 会把 research 字典合并、warnings 追加。
    """
```

> **关键**：返回**部分 state**而非整个 state。LangGraph 用 channel 合并，三路并行不会互踩。

### 4.1.2 三步法（每个 Agent 共用）

1. **数据获取**（纯函数）：`market_data = state['market_data']` 已被 `data_fetch_node` 预先填充。
2. **指标计算与计分**（纯函数）：`indicators_fund.py::compute_all_scores(info, financials) → list[IndicatorScore]`。**LLM 不参与算指标**，只给 reasoning 文本。
3. **LLM 综合定性**（结构化输出）：把 10 条 IndicatorScore 喂给 LLM，让其产出 `key_highlights` 与 `confidence`，并校验最终 `verdict` 与 `total_score` 一致。

### 4.1.3 行业分档（仅 Fundamental）

文件：[`python/fishtrade/tools/industry_classifier.py`](python/fishtrade/tools/industry_classifier.py)

```python
SECTOR_TO_CLASS: dict[str, Literal["value", "growth", "financial", "consumer", "energy"]] = {
    "Technology": "growth", "Communication Services": "growth",
    "Healthcare": "growth", "Financial Services": "financial",
    "Energy": "energy", "Consumer Defensive": "consumer",
    "Consumer Cyclical": "consumer", "Industrials": "value",
    "Utilities": "value", "Real Estate": "value", "Basic Materials": "value",
}

def classify_industry(info: dict) -> str:
    sector = info.get("sector")
    return SECTOR_TO_CLASS.get(sector, "value")      # 默认按价值股（保守阈值）
```

### 4.1.4 辅助工具函数清单

| 文件 | 函数 | 职责 |
|------|------|------|
| `tools/indicators_fund.py` | `score_pe_ratio(raw, industry_class) -> tuple[int, str]` | 按 analysismetrics 表给 PE 打分 |
| | `score_*` × 10 | 每指标一个；签名一致 |
| | `compute_all_fundamental(market_data) -> list[IndicatorScore]` | 编排器 |
| `tools/indicators_tech.py` | `compute_macd(history) -> dict` | 返回 `{macd, signal, hist, golden_cross, divergence}` |
| | `compute_rsi(history, n=14) -> float` | |
| | `compute_moving_averages(history) -> dict` | SMA20/50/200 + 排列 |
| | `compute_bollinger(history) -> dict` | 上中下轨 + 带宽 |
| | `compute_volume_profile(history) -> dict` | 量价配合 + OBV |
| | `compute_atr(history, n=14) -> float` | |
| | `compute_fibonacci(history) -> dict` | 近期高低 + 4 个回撤位 |
| | `compute_relative_strength(history, benchmark_history) -> float` | vs SPY |
| | `detect_price_pattern(history) -> str \| None` | 简化规则识别（双底/双顶） |
| | `compute_support_resistance(history) -> dict` | 近 60 日高低 + 整数关口 |
| | `compute_all_technical(market_data) -> list[IndicatorScore]` | 编排器 |
| `tools/indicators_sent.py` | `compute_short_float(info) -> ...` 等 10 个 | 同上 |
| | `compute_put_call_ratio(option_chain) -> float \| None` | 期权链空 → None |
| | `compute_52week_position(info) -> float` | |
| `tools/feature_flags.py` | `is_financial_data_sufficient(financials) -> bool` | 财报 ≥ 4 季度 |
| | `is_history_sufficient(history, min_days=60) -> bool` | VaR 前置 |
| | `is_in_earnings_window(earnings_dates, today, window_days=3) -> bool` | 财报窗口标记 |
| `llm/prompt_utils.py` | `build_research_prompt(facet, indicator_scores, industry_class) -> list[dict]` | 模板加载 + 变量注入 |
| | `summarize_research_for_debate(report) -> str` | 仅保留 verdict + 3 个核心指标，控 token |
| | `truncate_debate_history(turns, keep_last_n=2) -> list[DebateTurn]` | 防上下文超限 |

### 4.1.5 降级策略

- **单指标降级**：拿不到原始值（如 `freeCashflow` 为 None）→ `IndicatorScore(score=0, is_degraded=True, degrade_reason="yfinance 未返回该字段")`。
- **整面降级**：`sum(is_degraded) >= 5` → `is_facet_degraded=True`，`confidence` 强制 ≤ 0.4。
- **LLM 解析失败 2 次后**：调用 `_fallback_research_template(scores)`——纯规则汇总，不依赖 LLM，`confidence=0.3`，并在 `warnings` 追加 `"FUNDAMENTAL_LLM_FALLBACK"`。

### 4.1.6 超时

| 范围 | 超时 |
|------|------|
| 单 Agent 总超时（数据 + LLM） | 60s |
| LLM 单次调用 | `ARK_TIMEOUT_SECONDS=60`（节点级用 `asyncio.wait_for(60)` 兜底） |
| 三路并行总超时 | 90s（LangGraph `recursion_limit` + 外层 `asyncio.timeout(90)`） |

## 4.2 Debate 层

### 4.2.1 节点拆分

```
debate_opening_bull   ─┐
debate_opening_bear   ─┴─→  rebuttal_loop_n  ─→  debate_judge
```

**Opening 并行，Rebuttal 串行**（Bear 必须读到 Bull 的上一轮才能反驳）。

文件：[`python/fishtrade/agents/debate/bull.py`](python/fishtrade/agents/debate/bull.py) 等。

### 4.2.2 Prompt 策略

**Bull**（[`llm/prompts/debate_bull.md`](python/fishtrade/llm/prompts/debate_bull.md)）：
- System：投资分析师，立场看多，**禁止凭空论证**，每条论点必须引用至少 1 个 `cited_indicators`。
- User：注入 research 摘要（`summarize_research_for_debate`）+ 上一轮 Bear 论点（如有）。
- 强约束：`response_format=DebateTurn`，`temperature=0.7`。

**Bear** 对称。

**Judge**（`temperature=0.2`）：接收完整 `debate_turns` + 三面 verdict，返回 `DebateResult`。Prompt 中明确：
- 若 `degraded_facets` 非空，对应面权重置 0。
- `proposed_position_pct` 计算指引：`BUY` → 5–10、`HOLD/SELL` → 0；具体值由 confidence 微调。

### 4.2.3 历史轮次截断

每次喂给 Bull/Bear 的 `previous_turns` 经过 `truncate_debate_history(turns, keep_last_n=2)`：仅保留最近 2 轮（即 4 条 DebateTurn），更早的论点用一段摘要替代。Judge 阶段用全量。

### 4.2.4 异常处理

| 场景 | 行为 |
|------|------|
| Bull / Bear 拒答（content 空） | 装饰器追加 "请你必须给出立场并引用指标" 重试 1 次 |
| 重试仍失败 | 用上一轮该角色的结论占位，`is_fallback=True`，`warnings` 追加标记 |
| `cited_indicators` 引用了不存在的指标名 | Pydantic validator 拒绝 → 触发 JSON 重试装饰器 |
| `debate_rounds=0` | 跳过 rebuttal，opening 后直接 judge |
| 三面全部 `is_facet_degraded` | 跳过 debate，直接 `DebateResult(final_verdict="HOLD", confidence=0.2, proposed_position_pct=0, degraded_facets=[...])` |

## 4.3 Risk 层

### 4.3.1 三步顺序节点（短路失败）

```
hard_rules_node ──fail──→ END (decision=reject)
       │ pass
var_check_node ──fail──→ END
       │ pass
soft_judge_node ──fail──→ END
       │ pass
       └──→ APPROVE → execution_router
```

### 4.3.2 硬规则节点

文件：[`python/fishtrade/agents/risk/hard_rules.py`](python/fishtrade/agents/risk/hard_rules.py)

```python
def hard_rules_node(state: GraphState) -> dict:
    debate = state["debate"]
    portfolio = state["portfolio_before"]
    proposed = debate["proposed_position_pct"]

    checks = [
        check_r1_position_limit(proposed),
        check_r2_max_drawdown(portfolio["nav_history"]),
        # R3 在 var_check_node 内做完整计算
        check_r4_stoploss_definable(state["market_data"]["info"]["regularMarketPrice"]),
    ]
    failed = [c for c in checks if not c.passed]
    if failed:
        return {"risk": RiskDecision(
            decision="reject",
            adjusted_position_pct=0,
            hard_checks=checks,
            var_result=VarResult(var_95=0, portfolio_impact=0, passed=False, sample_size=0),
            soft_judgment=SoftJudgment(flags=["NONE"], adjustment="reject",
                                       adjusted_position_pct=0, reasoning="hard_rule_failed"),
            reject_reason=f"硬规则失败：{', '.join(c.rule for c in failed)}",
        ).model_dump()}
    return {"risk_partial_hard": [c.model_dump() for c in checks]}   # 中间产物，等 var/soft 合并
```

特例：**`debate.final_verdict == "HOLD"`** → 整个 Risk 层跳过，`RiskDecision.decision="approve"` 但 `adjusted_position_pct=0`，Execution 层会识别 0 仓位 → `status="skipped"`（对应需求 6.3 "Debate HOLD"）。

### 4.3.3 VaR 计算

文件：[`python/fishtrade/tools/var_calculator.py`](python/fishtrade/tools/var_calculator.py)

```python
def compute_var_historical(
    history: pd.DataFrame, lookback_days: int = 252, confidence: float = 0.95
) -> VarResult:
    if len(history) < 60:
        return VarResult(var_95=0, portfolio_impact=0, passed=False,
                         sample_size=len(history),
                         fallback_reason="历史数据 <60 个交易日，VaR 不可靠")
    returns = history["Close"].pct_change().dropna().tail(lookback_days)
    var_95 = -returns.quantile(1 - confidence)        # 正数表示损失
    return VarResult(var_95=float(var_95), passed=True, sample_size=len(returns), ...)
```

`var_check_node` 把 `var_95 × proposed_position_pct/100` 与 `RISK_VAR95_PORTFOLIO_LIMIT_PCT/100` 比较（R3）。

### 4.3.4 软规则节点

文件：[`python/fishtrade/agents/risk/soft_judge.py`](python/fishtrade/agents/risk/soft_judge.py)

LLM 输入打包：
```python
prompt_vars = {
    "debate_verdict": debate["final_verdict"],
    "proposed_pct": debate["proposed_position_pct"],
    "vix_recent": market_data["vix_recent"],         # 最近 5 日 VIX
    "current_holdings_summary": summarize_holdings(portfolio),
    "stock_avg_volume_usd": info["averageDailyVolume10Day"] * info["regularMarketPrice"],
    "stock_sector": info["sector"],
}
```

返回 `SoftJudgment`：
- `flags`：`MARKET_VOLATILE`（VIX>30）、`CORRELATION_HIGH`（同 sector 持仓 >20%）、`LIQUIDITY_LOW`（日均成交额 <$10M）。
- `adjustment`：`keep` / `reduce` / `reject`；`reduce` 时 `adjusted_position_pct = proposed × 0.5`（按需求 4.3.4）。

**降级**：LLM 调用失败 → 退化为纯规则：任一 flag 触发 → `adjustment=reduce`、`adjusted_position_pct=proposed×0.5`，`reasoning="LLM_FALLBACK_TO_RULES"`。

## 4.4 Execution 层

### 4.4.1 路由器

文件：[`python/fishtrade/agents/execution/router.py`](python/fishtrade/agents/execution/router.py)

```python
def execution_router(state: GraphState) -> str:
    """LangGraph 条件边：返回下一节点名。"""
    risk = state["risk"]
    if risk["decision"] == "reject" or risk["adjusted_position_pct"] <= 0:
        return "skip_execution"
    return {"dryrun": "execute_dryrun",
            "paper":  "execute_paper",
            "backtest": "execute_backtest"}[state["input"]["mode"]]
```

### 4.4.2 三个执行实现

| 文件 | 行为 |
|------|------|
| [`dryrun.py`](python/fishtrade/agents/execution/dryrun.py) | 计算订单 → 写 `logs/orders/{run_id}.json` → `status="generated"` |
| [`paper.py`](python/fishtrade/agents/execution/paper.py) | 调 Alpaca `submit_order` → 轮询 fill 30s → 写 portfolio |
| [`backtest.py`](python/fishtrade/agents/execution/backtest.py) | 用 `as_of_date` 收盘价模拟成交，更新 portfolio 但不发 API |

### 4.4.3 计算下单量

```python
def build_order(price: float, capital: float, pct: float, side: str) -> Order:
    target_value = capital * pct / 100
    qty = max(1, int(target_value // price))
    if side == "buy":
        return Order(side="buy", qty=qty,
                     limit_price=round(price * 1.002, 2),
                     stop_price=round(price * (1 - settings.risk_stoploss_pct/100), 2),
                     ...)
    else:
        return Order(side="sell", qty=qty,
                     limit_price=round(price * 0.998, 2), stop_price=None, ...)
```

### 4.4.4 Alpaca 异常

| 场景 | 行为 |
|------|------|
| 鉴权失败 / 不可达 | `status="failed"`、`error=str(e)`，**portfolio 不更新** |
| Partial fill（轮询 30s 后） | `status="partial"`，按 `filled_qty` 更新 portfolio |
| 市场休市（API 返回 `extended_hours=False`） | 订单进队列，`status="submitted"`，报告标注 |

---

# 五、LangGraph 编排与工作流装配

## 5.1 节点注册与边

文件：[`python/fishtrade/graph/builder.py`](python/fishtrade/graph/builder.py)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # —— 节点 ——
    g.add_node("validate_input",   validate_input_node)
    g.add_node("fetch_market",     fetch_market_node)        # yfinance bundle
    g.add_node("research_fund",    fundamental_node)
    g.add_node("research_tech",    technical_node)
    g.add_node("research_sent",    sentimental_node)
    g.add_node("debate_open_bull", debate_opening_bull_node)
    g.add_node("debate_open_bear", debate_opening_bear_node)
    g.add_node("debate_rebuttal",  debate_rebuttal_loop_node)
    g.add_node("debate_judge",     debate_judge_node)
    g.add_node("risk_hard",        hard_rules_node)
    g.add_node("risk_var",         var_check_node)
    g.add_node("risk_soft",        soft_judge_node)
    g.add_node("hitl_gate",        hitl_gate_node)           # 仅 input.hitl=True 时挂起
    g.add_node("execute_dryrun",   dryrun_node)
    g.add_node("execute_paper",    paper_node)
    g.add_node("execute_backtest", backtest_node)
    g.add_node("skip_execution",   skip_node)
    g.add_node("update_portfolio", update_portfolio_node)
    g.add_node("render_report",    render_report_node)

    # —— 入口 & 数据 ——
    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "fetch_market")

    # —— Fan-out: 三路并行研究 ——
    g.add_edge("fetch_market", "research_fund")
    g.add_edge("fetch_market", "research_tech")
    g.add_edge("fetch_market", "research_sent")

    # —— Fan-in: 用一个 barrier 节点收敛 ——
    # LangGraph 1.x 中：把后续节点的入边都写成来自三路即可
    g.add_edge("research_fund", "debate_open_bull")
    g.add_edge("research_tech", "debate_open_bull")
    g.add_edge("research_sent", "debate_open_bull")
    g.add_edge("research_fund", "debate_open_bear")
    g.add_edge("research_tech", "debate_open_bear")
    g.add_edge("research_sent", "debate_open_bear")

    # —— Debate 循环：rebuttal_loop_node 内部 for round in range(N) ——
    g.add_edge("debate_open_bull", "debate_rebuttal")
    g.add_edge("debate_open_bear", "debate_rebuttal")
    g.add_edge("debate_rebuttal", "debate_judge")

    # —— Risk 串行 + 短路 ——
    g.add_edge("debate_judge", "risk_hard")
    g.add_conditional_edges("risk_hard", route_after_hard,
        {"continue": "risk_var", "reject": "render_report"})
    g.add_conditional_edges("risk_var", route_after_var,
        {"continue": "risk_soft", "reject": "render_report"})
    g.add_conditional_edges("risk_soft", route_after_soft,
        {"continue": "hitl_gate", "reject": "render_report"})

    # —— HITL ——
    g.add_conditional_edges("hitl_gate", route_after_hitl,
        {"approved": "execute_router_dispatch", "rejected": "render_report"})

    # —— Execution 路由 ——
    g.add_conditional_edges("execute_router_dispatch", execution_router,
        {"execute_dryrun": "execute_dryrun",
         "execute_paper":  "execute_paper",
         "execute_backtest": "execute_backtest",
         "skip_execution": "skip_execution"})

    for n in ["execute_dryrun", "execute_paper", "execute_backtest", "skip_execution"]:
        g.add_edge(n, "update_portfolio")
    g.add_edge("update_portfolio", "render_report")
    g.add_edge("render_report", END)

    saver = SqliteSaver.from_conn_string("data/checkpoints.sqlite")
    return g.compile(checkpointer=saver, interrupt_before=["hitl_gate"])
```

### 5.1.1 路由函数

文件：[`python/fishtrade/graph/routes.py`](python/fishtrade/graph/routes.py)

```python
def route_after_hard(state: GraphState) -> str:
    return "reject" if state["risk"] and state["risk"]["decision"] == "reject" else "continue"

def route_after_var(state: GraphState) -> str: ...
def route_after_soft(state: GraphState) -> str: ...

def route_after_hitl(state: GraphState) -> str:
    """从 checkpoint 恢复后读取 state['hitl_decision']（CLI 写入）。"""
    return state.get("hitl_decision", "approved")
```

### 5.1.2 Fan-in 的并发安全

- 三路 research 节点的输出都用 `Annotated[list[str], add]` 合并 `warnings`、`research` 字段则是 `dict update`（LangGraph 默认 dict reducer 是覆盖；为避免互踩，每路只写自己的子键 `research.fundamental` 等）。
- `tokens_total` 用 `Annotated[int, operator.add]` 在 state 定义；并行写入会自动求和。

## 5.2 HITL 挂起与恢复

**触发条件**：`input.hitl=True`。
**实现**：编译时 `interrupt_before=["hitl_gate"]`，graph 在该节点前自动暂停，CheckpointSaver 保存完整 state。

CLI 流程：
```bash
python -m fishtrade run --ticker AAPL --capital 100000 --hitl
# 输出：
# Risk 已通过，建议仓位 7%。是否执行？[y/N/reduce 5%]
# 用户输入后，CLI 调用 graph.update_state({'hitl_decision': 'approved'}) 并 graph.invoke(None, ...) 续跑
```

恢复入口：
```bash
python -m fishtrade resume --run-id <uuid> --decision approved
# 或 --decision rejected
```

---

# 六、CLI 入口与后处理

## 6.1 CLI 参数解析

文件：[`python/fishtrade/cli.py`](python/fishtrade/cli.py)（`typer`）

| 子命令 | 参数 | 说明 |
|--------|------|------|
| `doctor` | — | 自检环境 |
| `run` | `--ticker`（必填）<br>`--capital`（默认 100000）<br>`--mode {dryrun,paper,backtest}`（默认 dryrun）<br>`--debate-rounds`（0–3，默认 1）<br>`--as-of-date`（默认今天）<br>`--language {zh,en,bilingual}`（默认 bilingual）<br>`--hitl`（开关） | 单票决策 |
| `resume` | `--run-id`、`--decision {approved,rejected}` | HITL 恢复 |
| `replay` | `--run-id` | 从 trace 重建报告（不重新调用 LLM） |

参数校验：
- `--mode paper` 时若 `ALPACA_API_KEY` 为空 → 启动时报错 `MISSING_ALPACA_KEY`。
- `--ticker` 立即过 `re.match(r"^[A-Z.\-]{1,6}$")`，再交给 `validate_input_node` 做 yfinance 真实校验。

## 6.2 报告生成器

文件：[`python/fishtrade/reporting/render.py`](python/fishtrade/reporting/render.py)、模板 [`templates/report_*.md.j2`](python/fishtrade/reporting/templates/)

```python
def render_report(state: GraphState, language: str = "bilingual") -> str:
    env = Environment(loader=FileSystemLoader("templates"), autoescape=False)
    if language == "bilingual":
        zh = env.get_template("report_zh.md.j2").render(s=state)
        en = env.get_template("report_en.md.j2").render(s=state)
        return f"# 中文版\n\n{zh}\n\n---\n\n# English\n\n{en}"
    return env.get_template(f"report_{language}.md.j2").render(s=state)
```

模板 8 个 section：
1. **Header**：ticker、as_of_date、run_id、mode、final_verdict
2. **三面打分**：3 张表（每张 10 行：indicator / raw / score / reasoning）
3. **辩论实录**：按轮次列出 Bull / Bear，最后 Judge 收尾
4. **风控判定**：4 条硬规则表 + VaR 表 + Soft flags
5. **执行记录**：order JSON、fill_info、status
6. **Portfolio 变更**：before/after 快照
7. **Warnings**：所有降级标记
8. **Trace 链接**：`logs/trace/{run_id}.jsonl` 路径

写盘路径：`reports/{ticker}-{as_of_date}.md`；同名文件存在 → 追加 `-{run_id短}` 后缀。

---

# 七、第一阶段开发任务清单

> 拆分依据：依赖关系（下游依赖上游）+ 可测试边界。每个 Task 包含**交付物**与**验收**。

### 阶段 A — 工程骨架 & 配置（半天）

- [ ] **A1** 初始化 `python/fishtrade/` 包结构（按 1.2 目录树创建空文件 + `__init__.py`）
- [ ] **A2** 写 `pyproject.toml` + 更新 `requirements.txt`（补 pydantic-settings / diskcache / alpaca-py / typer / vcrpy）
- [ ] **A3** 实现 `config/settings.py`（Pydantic Settings + `.env.example`）
- [ ] **A4** 实现 `cli.py` 的 `doctor` 子命令（仅打印配置 + ping Ark）
- [ ] **A5** 配置 `structlog` + 跑通 `python -m fishtrade doctor`

### 阶段 B — 数据模型（1 天）

- [ ] **B1** 实现 `models/research.py`（含两个 model_validator）+ 单元测试
- [ ] **B2** 实现 `models/debate.py`、`models/risk.py`、`models/execution.py`、`models/portfolio.py`
- [ ] **B3** 实现 `models/state.py`（GraphState TypedDict + reducer 标注）
- [ ] **B4** 写 `tests/unit/test_models_schemas.py`：覆盖每个 model 的边界（total_score 不一致拒绝、verdict 不一致拒绝、cited_indicators 空列表拒绝）

### 阶段 C — 基础设施（2 天）

- [ ] **C1** `llm/client.py` + `llm/factory.py`（含 `_inject_schema_hint` 单测）
- [ ] **C2** `llm/retry.py`（tenacity 装饰器 + `JSONParseError` 自定义异常）
- [ ] **C3** `observability/trace.py` JSONL writer + `metrics.py` 聚合
- [ ] **C4** `tools/yf_client.py` + `yf_cache.py`（diskcache 集成 + 字段级降级）
- [ ] **C5** `portfolio/store.py` + `nav.py` + 首次运行自动初始化测试
- [ ] **C6** `tools/var_calculator.py` + 单元测试（含 <60 日数据降级）
- [ ] **C7** `tools/industry_classifier.py` + 单元测试

### 阶段 D — 指标计算工具（1.5 天，可与 C 并行）

- [ ] **D1** `indicators_fund.py`：10 个 scorer + INDICATOR_REGISTRY + `compute_all_fundamental`
- [ ] **D2** `indicators_tech.py`：10 个计算 + 评分函数（pandas-ta 集成）
- [ ] **D3** `indicators_sent.py`：10 个，含 RETAIL_SOCIAL 永远 score=0、is_degraded=True 的占位
- [ ] **D4** `feature_flags.py` 工具函数集
- [ ] **D5** 用真实 AAPL 数据跑一遍 `compute_all_*`，校验 IndicatorScore 总和 ∈ [-10, 10]

### 阶段 E — Agent 实现（3 天）

- [ ] **E1** Research 三个 agent + Prompt 模板 + 降级回退模板
- [ ] **E2** Debate Bull / Bear / Judge + `truncate_debate_history` + opening 并行 / rebuttal 串行
- [ ] **E3** Risk hard / VaR / soft + 短路逻辑 + LLM 失败规则降级
- [ ] **E4** Execution dryrun / paper / backtest + `execution_router`
- [ ] **E5** `update_portfolio_node`（成交后写回，failed 不写）

### 阶段 F — Graph 装配（1 天）

- [ ] **F1** `graph/builder.py` 完整组装（按 5.1 节）
- [ ] **F2** `graph/routes.py` 全部条件路由函数
- [ ] **F3** `graph/checkpoint.py` SqliteSaver
- [ ] **F4** `tests/integration/test_graph_e2e.py`：mock Ark + mock yfinance，跑通端到端

### 阶段 G — CLI & 报告（1 天）

- [ ] **G1** `cli.py` 完整 typer 实现（run / resume / replay / doctor）
- [ ] **G2** `reporting/render.py` + 双语 Jinja2 模板
- [ ] **G3** HITL 流程手动验证：`--hitl` → ctrl+c → resume

### 阶段 H — 边界 case 测试（1 天）

对应需求 6.5 的 10 个 case，每个一个 pytest 文件：

- [ ] **H1** test_invalid_ticker.py（INVALID_TICKER 错误码）
- [ ] **H2** test_drawdown_limit.py（R2 拒绝 BUY 但允许 SELL）
- [ ] **H3** test_var_exceeded.py（R3 拒绝）
- [ ] **H4** test_debate_all_hold.py（Judge 输出 HOLD）
- [ ] **H5** test_debate_rounds_zero_three.py（参数化 0/3 轮）
- [ ] **H6** test_new_listing_degraded.py（财报数据不足降级）
- [ ] **H7** test_portfolio_first_run.py（首次自动初始化）
- [ ] **H8** test_llm_timeout_fallback.py（mock Ark 超时 → 降级模板）
- [ ] **H9** test_low_liquidity_warning.py（软规则 LIQUIDITY_LOW 触发）
- [ ] **H10** test_paper_partial_fill.py（mock Alpaca 部分成交）

### 阶段 I — README & 作品集打磨（半天）

- [ ] **I1** README.md：架构图（Mermaid）、5 步快速开始、示例报告链接、已知简化项
- [ ] **I2** 跑一次真实 AAPL，把生成的 `reports/AAPL-2026-04-25.md` 作为示例提交
- [ ] **I3** 录一段终端 GIF（asciinema → svg-term）放 README

---

## 附录 A：异常码总表

| 错误码 | 含义 | 终止流水线？ |
|--------|------|--------------|
| `INVALID_TICKER` | yfinance 无法识别 | 是 |
| `MISSING_ARK_KEY` | `ARK_API_KEY` 未设置 | 是（启动期） |
| `MISSING_ALPACA_KEY` | paper 模式但 Alpaca key 缺失 | 是（启动期） |
| `ARK_TIMEOUT` | Ark 调用超过 60s × 3 次 | 否（节点级降级） |
| `ARK_RATE_LIMIT_EXHAUSTED` | 退避 60s+ 仍 429 | 是（保存 state） |
| `JSON_PARSE_FAILED` | LLM 输出 2 次解析失败 | 否（降级模板） |
| `YF_RATE_LIMIT` | yfinance 退避 3 次失败 | 否（指标级降级） |
| `INSUFFICIENT_HISTORY` | 历史 <60 日 | R3 拒绝（业务级） |
| `HARD_RULE_FAILED` | R1/R2/R4 任一失败 | 否（risk reject） |
| `EXECUTION_FAILED` | Alpaca 提交失败 | 否（写报告） |

## 附录 B：超时配置一览

| 范围 | 默认值 | 配置项 |
|------|--------|--------|
| Ark 单次调用 | 60s | `ARK_TIMEOUT_SECONDS` |
| Ark 重试次数 | 2 | `ARK_MAX_RETRIES`（tenacity 内置 +1 = 3 次） |
| Ark 重试间隔 | 2s, 5s 指数 | hardcoded in `retry.py` |
| 单 Research Agent 总超时 | 60s | `asyncio.wait_for` |
| 三路 Research 总超时 | 90s | graph-level |
| Debate 单角色超时 | 45s | LLM 调用 +5s 解析 |
| Risk 软规则超时 | 30s | |
| Alpaca 提交后轮询 fill | 30s | hardcoded |
| yfinance 单次请求 | 15s | `requests.get(timeout=15)` |
| yfinance 退避基数 | 2s | `YF_RATE_LIMIT_BACKOFF_BASE` |

## 附录 C：与 `analysismetrics.md` 对齐 Checklist（实施时勾选）

- [ ] 基本面 10 指标 ID 与 `INDICATOR_REGISTRY["fundamental"]` 一一对应
- [ ] 行业分档（价值 / 成长 / 金融 / 消费 / 能源）的阈值表与 `analysismetrics.md` 第一章逐项对照
- [ ] 技术面 10 指标计算公式与第二章定义一致（特别是 RSI 14、MACD 12-26-9、布林带 20-2σ）
- [ ] 情绪面第 8 项 RETAIL_SOCIAL 显式标记 `is_degraded=True`
- [ ] 财报超预期（EARNINGS_BEAT）数据源不足时降级到 score=0，并在 `key_highlights` 标注
- [ ] 评分汇总边界与 "汇总评分与最终判断框架"（≥+5=BUY、+1~+4=HOLD、≤0=SELL）一致

---

*文档版本：v1.0 — 落入 Task A1 后启动开发；任何数据结构调整需同步本文 §2 与 §4。*
