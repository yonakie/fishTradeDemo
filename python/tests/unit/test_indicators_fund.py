"""Unit tests for fundamental indicators (10 scorers + orchestrator)."""

from __future__ import annotations

import pytest

from fishtrade.models.research import IndicatorScore
from fishtrade.tools.indicators_fund import (
    INDICATOR_REGISTRY,
    compute_all_fundamental,
    score_analyst_upside,
    score_debt_to_equity,
    score_free_cashflow,
    score_gross_margin,
    score_net_margin,
    score_pb_ratio,
    score_pe_ratio,
    score_ps_ratio,
    score_revenue_growth,
    score_roe,
)


# ---------- scorer-level tests --------------------------------------------


def test_pe_growth_low_is_positive():
    out = score_pe_ratio(20.0, "growth", {})
    assert out.score == 1


def test_pe_growth_high_is_negative():
    out = score_pe_ratio(80.0, "growth", {})
    assert out.score == -1


def test_pe_negative_is_negative():
    out = score_pe_ratio(-3.5, "growth", {})
    assert out.score == -1


def test_pe_value_thresholds():
    assert score_pe_ratio(8, "value", {}).score == 1
    assert score_pe_ratio(15, "value", {}).score == 0
    assert score_pe_ratio(40, "value", {}).score == -1


def test_pe_missing_uses_forward_fallback():
    out = score_pe_ratio(None, "growth", {"forwardPE": 18.0})
    assert out.score == 1
    assert out.is_degraded is False


def test_pe_fully_missing_degrades():
    out = score_pe_ratio(None, "growth", {})
    assert out.score == 0
    assert out.is_degraded is True
    assert out.degrade_reason


def test_pb_branches():
    assert score_pb_ratio(0.7, "financial", {}).score == 1
    assert score_pb_ratio(1.2, "financial", {}).score == 0
    assert score_pb_ratio(2.5, "financial", {}).score == -1
    assert score_pb_ratio(None, "value", {}).is_degraded


def test_ps_branches():
    assert score_ps_ratio(2.0, "growth", {}).score == 1
    assert score_ps_ratio(8.0, "growth", {}).score == 0
    assert score_ps_ratio(25.0, "growth", {}).score == -1


def test_revenue_growth_branches():
    assert score_revenue_growth(0.30, "growth", {}).score == 1
    assert score_revenue_growth(0.15, "growth", {}).score == 0
    assert score_revenue_growth(-0.05, "growth", {}).score == -1


def test_gross_margin_growth():
    assert score_gross_margin(0.75, "growth", {}).score == 1
    assert score_gross_margin(0.60, "growth", {}).score == 0
    assert score_gross_margin(0.40, "growth", {}).score == -1


def test_net_margin_growth():
    assert score_net_margin(0.25, "growth", {}).score == 1
    assert score_net_margin(0.05, "growth", {}).score == -1


def test_roe_branches():
    assert score_roe(0.25, "growth", {}).score == 1
    assert score_roe(0.05, "growth", {}).score == -1
    assert score_roe(0.13, "financial", {}).score == 1


def test_debt_to_equity_growth():
    assert score_debt_to_equity(20.0, "growth", {}).score == 1
    assert score_debt_to_equity(50.0, "growth", {}).score == 0
    assert score_debt_to_equity(120.0, "growth", {}).score == -1


def test_free_cashflow_yield_positive():
    out = score_free_cashflow(1_000_000_000, "growth", {"marketCap": 10_000_000_000})
    assert out.score == 1


def test_free_cashflow_negative_is_negative():
    out = score_free_cashflow(-1_000, "growth", {"marketCap": 100_000_000})
    assert out.score == -1


def test_free_cashflow_missing_market_cap_uses_sign():
    out = score_free_cashflow(5_000_000, "growth", {})
    assert out.score == 1


def test_analyst_upside_branches():
    info = {"regularMarketPrice": 100.0}
    assert score_analyst_upside(130.0, "growth", info).score == 1
    assert score_analyst_upside(110.0, "growth", info).score == 0
    assert score_analyst_upside(95.0, "growth", info).score == -1


def test_analyst_upside_missing_price_degrades():
    out = score_analyst_upside(150.0, "growth", {})
    assert out.is_degraded is True


# ---------- orchestrator tests --------------------------------------------


def _full_info() -> dict:
    """A clean info dict that should yield a score for every indicator."""
    return {
        "sector": "Technology",
        "trailingPE": 22.0,
        "forwardPE": 18.0,
        "priceToBook": 4.0,
        "priceToSalesTrailing12Months": 6.0,
        "revenueGrowth": 0.18,
        "grossMargins": 0.65,
        "profitMargins": 0.18,
        "returnOnEquity": 0.22,
        "debtToEquity": 40.0,
        "freeCashflow": 5_000_000_000,
        "marketCap": 1_000_000_000_000,
        "targetMeanPrice": 250.0,
        "regularMarketPrice": 200.0,
    }


def test_compute_all_fundamental_returns_exactly_ten():
    md = {"info": _full_info()}
    scores = compute_all_fundamental(md)
    assert len(scores) == 10
    assert all(isinstance(s, IndicatorScore) for s in scores)


def test_compute_all_fundamental_total_in_bounds():
    md = {"info": _full_info()}
    scores = compute_all_fundamental(md)
    total = sum(s.score for s in scores)
    assert -10 <= total <= 10


def test_compute_all_fundamental_with_empty_info_degrades_all():
    scores = compute_all_fundamental({"info": {}})
    assert len(scores) == 10
    assert all(s.is_degraded for s in scores)
    assert all(s.score == 0 for s in scores)


def test_compute_all_fundamental_with_none_info_does_not_raise():
    scores = compute_all_fundamental({})
    assert len(scores) == 10


def test_indicator_registry_has_canonical_ten_names():
    expected = {
        "PE_RATIO", "PB_RATIO", "PS_RATIO", "REVENUE_GROWTH", "GROSS_MARGIN",
        "NET_MARGIN", "ROE", "DEBT_TO_EQUITY", "FREE_CASHFLOW", "ANALYST_UPSIDE",
    }
    assert set(INDICATOR_REGISTRY.keys()) == expected


def test_total_matches_sum():
    scores = compute_all_fundamental({"info": _full_info()})
    s = sum(i.score for i in scores)
    # sanity: building a ResearchReport with these scores should pass
    from fishtrade.models.research import ResearchReport

    rep = ResearchReport(
        facet="fundamental",
        ticker="TEST",
        as_of_date="2026-04-25",
        indicator_scores=scores,
        total_score=s,
        verdict=("BUY" if s >= 5 else ("HOLD" if s >= 1 else "SELL")),
        confidence=0.6,
        key_highlights=["a", "b", "c"],
        industry_class="growth",
    )
    assert rep.total_score == s
