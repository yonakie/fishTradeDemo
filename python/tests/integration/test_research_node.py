"""Isolation tests for the three research nodes.

We mock ``generate_ark_response`` so the tests stay offline while
exercising:
- happy path → research patch with the right facet key
- LLM failure (raises) → fallback report with warning + confidence ≤ 0.4 (degraded)
"""

from __future__ import annotations

import pytest

from fishtrade.agents.research.fundamental import fundamental_node
from fishtrade.agents.research.sentimental import sentimental_node
from fishtrade.agents.research.technical import technical_node
from fishtrade.models.research import ResearchReport

from ._fixtures import make_market_data


# ---------- helpers --------------------------------------------------------


def _state(market_data: dict | None = None) -> dict:
    return {
        "input": {
            "ticker": "AAPL",
            "capital": 100_000.0,
            "mode": "dryrun",
            "debate_rounds": 1,
            "as_of_date": "2026-04-25",
            "language": "zh",
            "hitl": False,
        },
        "market_data": market_data or make_market_data(),
        "run_id": "test-run-research",
    }


def _mock_llm_response(facet: str):
    """Return a function that builds a ResearchReport echoing input scores."""
    def _fn(messages, **kwargs):
        # The user message contains the indicator_scores JSON we sent.
        # Easiest: rebuild a valid report from the schema using fake highlights.
        # We rely on _coerce_llm_report to pin scores deterministically.
        # Provide a minimal valid ResearchReport whose totals will be overwritten.
        from fishtrade.models.research import IndicatorScore

        scores = [
            IndicatorScore(
                name=f"X_{i}",
                display_name_zh=f"X_{i}",
                display_name_en=f"X_{i}",
                raw_value=None,
                score=0,
                reasoning="placeholder",
                is_degraded=True,
                degrade_reason="placeholder",
            )
            for i in range(10)
        ]
        return ResearchReport(
            facet=facet,  # type: ignore[arg-type]
            ticker="AAPL",
            as_of_date="2026-04-25",
            indicator_scores=scores,
            total_score=0,
            verdict="SELL",
            confidence=0.9,  # will be clipped if facet degraded
            key_highlights=["a", "b", "c"],
            industry_class="growth" if facet == "fundamental" else None,
            is_facet_degraded=True,
            degrade_summary="placeholder",
        )

    return _fn


# ---------- happy paths ----------------------------------------------------


@pytest.mark.parametrize(
    "facet, node",
    [
        ("fundamental", fundamental_node),
        ("technical", technical_node),
        ("sentimental", sentimental_node),
    ],
)
def test_research_node_happy_path(monkeypatch, facet, node):
    monkeypatch.setattr(
        f"fishtrade.agents.research._common.generate_ark_response",
        _mock_llm_response(facet),
    )
    patch = node(_state())
    assert "research" in patch
    assert facet in patch["research"]
    body = patch["research"][facet]
    # Reconstruct the report and double-check invariants.
    report = ResearchReport.model_validate(body)
    assert report.facet == facet
    assert report.ticker == "AAPL"
    assert len(report.indicator_scores) == 10
    assert -10 <= report.total_score <= 10
    # The deterministic engine produced the scores; verdict ↔ total_score check
    # happens inside the model validator, so this is a tautology — but an
    # important regression catch.
    if report.is_facet_degraded:
        assert report.confidence <= 0.4


# ---------- fallback paths -------------------------------------------------


def _raise(*_args, **_kwargs):
    raise RuntimeError("simulated Ark outage")


@pytest.mark.parametrize(
    "facet, node, warning",
    [
        ("fundamental", fundamental_node, "FUNDAMENTAL_LLM_FALLBACK"),
        ("technical", technical_node, "TECHNICAL_LLM_FALLBACK"),
        ("sentimental", sentimental_node, "SENTIMENTAL_LLM_FALLBACK"),
    ],
)
def test_research_node_llm_failure_falls_back(monkeypatch, facet, node, warning):
    monkeypatch.setattr(
        "fishtrade.agents.research._common.generate_ark_response", _raise
    )
    patch = node(_state())
    assert warning in patch.get("warnings", [])
    body = patch["research"][facet]
    report = ResearchReport.model_validate(body)
    # Fallback should leave confidence at the rule-template level (≤0.3).
    assert report.confidence <= 0.4


def test_research_node_returns_only_local_keys(monkeypatch):
    """The patch must NOT contain other top-level keys (state-patch contract)."""
    monkeypatch.setattr(
        "fishtrade.agents.research._common.generate_ark_response",
        _mock_llm_response("fundamental"),
    )
    patch = fundamental_node(_state())
    allowed = {"research", "warnings"}
    assert set(patch.keys()).issubset(allowed)
