"""Reusable test fixtures: deterministic mock state + indicator scores.

These fixtures are *not* a pytest plugin — modules import ``make_*`` helpers
directly. Keeps the surface explicit and avoids accidental fixture leaks.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from fishtrade.models.debate import DebateResult, DebateTurn
from fishtrade.models.portfolio import NavSnapshot, PortfolioSnapshot, Position
from fishtrade.models.research import IndicatorScore, ResearchReport


# ---------- IndicatorScore helpers ----------------------------------------


def make_score(
    name: str = "PE_RATIO",
    score: int = 1,
    raw_value=12.5,
    is_degraded: bool = False,
) -> IndicatorScore:
    return IndicatorScore(
        name=name,
        display_name_zh=name,
        display_name_en=name,
        raw_value=raw_value,
        score=score,  # type: ignore[arg-type]
        reasoning=f"reason for {name}",
        is_degraded=is_degraded,
        degrade_reason="缺数据" if is_degraded else None,
    )


def make_indicator_set(scores: Iterable[int], names: Iterable[str] | None = None):
    scores = list(scores)
    if names is None:
        names = [f"IND_{i}" for i in range(len(scores))]
    return [make_score(name=n, score=s) for n, s in zip(names, scores)]


# ---------- ResearchReport helpers ----------------------------------------


def make_report(
    facet: str = "fundamental",
    *,
    ticker: str = "AAPL",
    as_of_date: str = "2026-04-25",
    total_score: int = 6,
    confidence: float = 0.7,
    is_facet_degraded: bool = False,
) -> ResearchReport:
    """Build a 10-indicator report whose verdict matches total_score."""
    # Distribute the requested total over 10 ±1 scores.
    scores: list[int] = []
    remaining = total_score
    for i in range(10):
        if remaining > 0:
            scores.append(1)
            remaining -= 1
        elif remaining < 0:
            scores.append(-1)
            remaining += 1
        else:
            scores.append(0)
    indicators = make_indicator_set(scores)
    if total_score >= 5:
        verdict = "BUY"
    elif total_score >= 1:
        verdict = "HOLD"
    else:
        verdict = "SELL"
    if is_facet_degraded:
        confidence = min(confidence, 0.4)
    return ResearchReport(
        facet=facet,  # type: ignore[arg-type]
        ticker=ticker,
        as_of_date=as_of_date,
        indicator_scores=indicators,
        total_score=total_score,
        verdict=verdict,
        confidence=confidence,
        key_highlights=[
            f"{facet} highlight 1",
            f"{facet} highlight 2",
            f"{facet} highlight 3",
        ],
        industry_class="growth" if facet == "fundamental" else None,
        is_facet_degraded=is_facet_degraded,
        degrade_summary=("simulated degrade" if is_facet_degraded else None),
    )


# ---------- DebateTurn / DebateResult helpers -----------------------------


def make_turn(
    *,
    role: str = "bull",
    round_idx: int = 0,
    conclusion: str = "BUY",
    cited: list[str] | None = None,
) -> DebateTurn:
    return DebateTurn(
        round=round_idx,
        role=role,  # type: ignore[arg-type]
        argument=f"{role} argument round {round_idx}",
        cited_indicators=cited or ["IND_0", "IND_1"],
        conclusion=conclusion,  # type: ignore[arg-type]
        is_fallback=False,
    )


def make_debate_result(
    *,
    final_verdict: str = "BUY",
    proposed_position_pct: float = 7.0,
    confidence: float = 0.6,
    degraded_facets: list[str] | None = None,
    turns: list[DebateTurn] | None = None,
) -> DebateResult:
    if final_verdict in ("HOLD", "SELL"):
        proposed_position_pct = 0.0
    return DebateResult(
        turns=turns or [make_turn(role="bull"), make_turn(role="bear", conclusion="HOLD")],
        final_verdict=final_verdict,  # type: ignore[arg-type]
        final_rationale=f"rationale for {final_verdict}",
        confidence=confidence,
        proposed_position_pct=proposed_position_pct,
        degraded_facets=degraded_facets or [],  # type: ignore[arg-type]
    )


# ---------- Portfolio helpers ---------------------------------------------


def make_portfolio(
    *,
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
    nav: float | None = None,
    nav_history: list[NavSnapshot] | None = None,
    max_drawdown_pct: float = 0.0,
) -> PortfolioSnapshot:
    positions = positions or []
    if nav is None:
        nav = cash + sum(p.qty * p.avg_cost for p in positions)
    return PortfolioSnapshot(
        cash=cash,
        positions=positions,
        nav=nav,
        nav_history=nav_history or [NavSnapshot(date="2026-04-24", nav=nav)],
        max_drawdown_pct=max_drawdown_pct,
    )


# ---------- Market data helpers -------------------------------------------


def make_history(
    n_days: int = 250, start_price: float = 100.0, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Ask for plenty of business days then trim — bfreq + periods can drift on
    # some pandas versions, so we let the index decide the final length.
    dates = pd.date_range(end="2026-04-25", periods=n_days + 10, freq="B")[-n_days:]
    n = len(dates)
    rets = rng.normal(loc=0.0005, scale=0.012, size=n)
    closes = start_price * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": closes * 0.998,
            "High": closes * 1.005,
            "Low": closes * 0.993,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n),
        },
        index=dates,
    )


def make_market_data(
    *,
    ticker: str = "AAPL",
    price: float = 200.0,
    sector: str = "Technology",
    avg_volume: int = 60_000_000,
    n_history_days: int = 250,
    vix_avg: float | None = None,
) -> dict:
    info = {
        "symbol": ticker,
        "regularMarketPrice": price,
        "quoteType": "EQUITY",
        "sector": sector,
        "trailingPE": 22.0,
        "forwardPE": 20.0,
        "priceToBook": 4.5,
        "priceToSalesTrailing12Months": 5.5,
        "revenueGrowth": 0.12,
        "grossMargins": 0.45,
        "profitMargins": 0.22,
        "returnOnEquity": 0.30,
        "debtToEquity": 60.0,
        "freeCashflow": 1.2e10,
        "marketCap": 3.0e12,
        "targetMeanPrice": price * 1.10,
        "averageDailyVolume10Day": avg_volume,
        "shortPercentOfFloat": 0.012,
        "heldPercentInstitutions": 0.62,
        "fiftyTwoWeekHigh": price * 1.2,
        "fiftyTwoWeekLow": price * 0.8,
        "dividendYield": 0.005,
        "payoutRatio": 0.18,
        "recommendationMean": 2.0,
    }
    history = make_history(n_days=n_history_days, start_price=price * 0.85)
    bench = make_history(n_days=n_history_days, start_price=400.0, seed=7)

    md: dict = {
        "info": info,
        "history": history,
        "benchmark_history": bench,
        "fetch_warnings": [],
    }
    if vix_avg is not None:
        # tiny serialised payload understood by soft_judge._vix_avg
        md["vix_recent"] = {
            "columns": ["Close"],
            "index": ["d1", "d2", "d3", "d4", "d5"],
            "data": [[vix_avg], [vix_avg], [vix_avg], [vix_avg], [vix_avg]],
        }
    return md


__all__ = [
    "make_debate_result",
    "make_history",
    "make_indicator_set",
    "make_market_data",
    "make_portfolio",
    "make_report",
    "make_score",
    "make_turn",
]
