"""Shared fixtures for boundary tests (H1–H10).

Mirrors the integration-tier ``patch_ark`` fixture so each boundary
file can ask for it directly. Keeps the boundary suite self-contained
without re-importing from ``tests/integration/_fixtures.py`` (we still
reuse the data factories from there because they're battle-tested).
"""

from __future__ import annotations

import pandas as pd
import pytest

from fishtrade.models.debate import DebateResult, DebateTurn
from fishtrade.models.research import ResearchReport
from fishtrade.models.risk import SoftJudgment
from fishtrade.tools.yf_client import _df_to_payload  # type: ignore

from tests.integration._fixtures import (
    make_debate_result,
    make_market_data,
    make_portfolio,
    make_report,
    make_turn,
)


def _serialize_market(md: dict) -> dict:
    out = dict(md)
    for key in ("history", "benchmark_history", "vix_recent"):
        v = out.get(key)
        if isinstance(v, pd.DataFrame):
            out[key] = _df_to_payload(v)
    return out


def _make_mock_ark(
    *,
    judge_verdict: str = "BUY",
    judge_pct: float = 7.0,
    soft_flags: list[str] | None = None,
    soft_adjustment: str = "keep",
    soft_pct: float | None = None,
    research_total: int = 6,
    raise_for_schema: type | None = None,
):
    """Build a deterministic LLM stand-in routed by ``response_schema``."""

    def _fn(messages, *, response_schema=None, role=None, **kwargs):
        if raise_for_schema is not None and response_schema is raise_for_schema:
            raise TimeoutError("simulated Ark timeout")
        if response_schema is ResearchReport:
            return make_report(facet="fundamental", total_score=research_total, confidence=0.8)
        if response_schema is DebateTurn:
            return DebateTurn(
                round=0,
                role="bull",
                argument="mock argument",
                cited_indicators=["IND_0", "IND_1"],
                conclusion="BUY",
                is_fallback=False,
            )
        if response_schema is DebateResult:
            return make_debate_result(
                final_verdict=judge_verdict,
                proposed_position_pct=judge_pct,
                turns=[make_turn(role="bull"), make_turn(role="bear")],
            )
        if response_schema is SoftJudgment:
            flags = soft_flags or ["NONE"]
            adj = soft_adjustment
            pct = soft_pct if soft_pct is not None else (judge_pct or 7.0)
            return SoftJudgment(
                flags=flags,  # type: ignore[arg-type]
                adjustment=adj,  # type: ignore[arg-type]
                adjusted_position_pct=pct,
                reasoning=f"mock soft ({adj})",
            )
        raise RuntimeError(f"unexpected schema: {response_schema}")

    return _fn


@pytest.fixture
def patch_ark(monkeypatch):
    def _apply(**kwargs):
        fn = _make_mock_ark(**kwargs)
        for target in (
            "fishtrade.agents.research._common.generate_ark_response",
            "fishtrade.agents.debate._common.generate_ark_response",
            "fishtrade.agents.risk.soft_judge.generate_ark_response",
        ):
            monkeypatch.setattr(target, fn)
        return fn

    return _apply


@pytest.fixture
def initial_state():
    """Factory: returns a fresh seed state dict on every call."""

    def _make(
        *,
        ticker: str = "AAPL",
        capital: float = 100_000.0,
        mode: str = "dryrun",
        debate_rounds: int = 0,
        as_of_date: str = "2026-04-25",
        language: str = "zh",
        hitl: bool = False,
        market_data: dict | None = None,
        portfolio_before: dict | None = None,
        run_id: str = "boundary-test",
    ) -> dict:
        return {
            "input": {
                "ticker": ticker,
                "capital": capital,
                "mode": mode,
                "debate_rounds": debate_rounds,
                "as_of_date": as_of_date,
                "language": language,
                "hitl": hitl,
            },
            "market_data": _serialize_market(market_data or make_market_data()),
            "portfolio_before": (portfolio_before or make_portfolio().model_dump()),
            "research": {},
            "debate_turns": [],
            "warnings": [],
            "tokens_total": 0,
            "latency_ms_total": 0,
            "run_id": run_id,
        }

    return _make


__all__ = ["patch_ark", "initial_state"]
