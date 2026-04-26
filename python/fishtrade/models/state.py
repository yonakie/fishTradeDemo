"""LangGraph GraphState TypedDict — the cross-pipeline ledger."""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from .debate import DebateResult, DebateTurn
from .execution import ExecutionResult
from .portfolio import PortfolioSnapshot
from .research import ResearchReport
from .risk import RiskDecision


class RunInput(TypedDict):
    ticker: str
    capital: float
    mode: Literal["dryrun", "paper", "backtest"]
    debate_rounds: int
    as_of_date: str
    language: Literal["zh", "en", "bilingual"]
    hitl: bool


class MarketDataBundle(TypedDict, total=False):
    """Raw yfinance snapshot. Lazily populated; nodes read only what they need."""

    info: dict
    history: dict
    financials: dict
    cashflow: dict
    balance_sheet: dict
    options_chain: dict | None
    institutional_holders: dict | None
    insider_transactions: dict | None
    upgrades_downgrades: dict | None
    earnings_dates: dict | None
    benchmark_history: dict
    vix_recent: dict
    fetch_warnings: list[str]


class ResearchSection(TypedDict, total=False):
    fundamental: ResearchReport | None
    technical: ResearchReport | None
    sentimental: ResearchReport | None


def _merge_research(
    left: ResearchSection | None, right: ResearchSection | None
) -> ResearchSection:
    """Sub-key merge for the parallel research channel.

    LangGraph reducers operate at top-level field granularity; without this
    merge, three parallel research nodes that each return
    ``{"research": {<facet>: ...}}`` would clobber each other (or trigger
    ``InvalidUpdateError`` on stricter LangGraph versions). Each node owns a
    distinct facet key, so a shallow ``{**left, **right}`` is order-
    independent in practice; ``right`` wins on the (defensive) overlap case.
    """
    if not left and not right:
        return {}
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    return {**left, **right}


class GraphState(TypedDict, total=False):
    # —— entry ——
    input: RunInput

    # —— data layer ——
    market_data: MarketDataBundle

    # —— research layer (3-way parallel writes) ——
    research: Annotated[ResearchSection, _merge_research]

    # —— debate layer ——
    debate_turns: Annotated[list[DebateTurn], add]
    debate: DebateResult | None

    # —— risk layer ——
    risk: RiskDecision | None

    # —— execution layer ——
    execution: ExecutionResult | None

    # —— portfolio ——
    portfolio_before: PortfolioSnapshot | None
    portfolio_after: PortfolioSnapshot | None

    # —— metadata ——
    run_id: str
    started_at: str
    warnings: Annotated[list[str], add]
    tokens_total: Annotated[int, add]
    latency_ms_total: Annotated[int, add]
    halt_reason: Optional[str]
