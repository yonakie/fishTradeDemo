"""Fundamental research node — computes 10 indicators, narrates via LLM."""

from __future__ import annotations

from ...models.state import GraphState
from ...tools.indicators_fund import compute_all_fundamental
from ...tools.industry_classifier import classify_industry
from ._common import _fallback_research_template, run_research_facet


def fundamental_node(state: GraphState) -> dict:
    """LangGraph node — produces a state patch for ``research.fundamental``.

    The deterministic indicator engine always succeeds (it degrades on
    missing fields). The LLM call is allowed to fail; on failure we fall
    back to a pure-rule report and emit a ``FUNDAMENTAL_LLM_FALLBACK``
    warning into the (additive) ``warnings`` channel.
    """
    run_input = state.get("input") or {}
    market_data = state.get("market_data") or {}
    ticker = run_input.get("ticker", "UNKNOWN")
    as_of_date = run_input.get("as_of_date", "")
    run_id = state.get("run_id", "ad-hoc")

    info = market_data.get("info") or {}
    industry_class = classify_industry(info)

    indicator_scores = compute_all_fundamental(market_data)

    try:
        report, warnings = run_research_facet(
            facet="fundamental",
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=industry_class,
            indicator_scores=indicator_scores,
            run_id=run_id,
            node_name="research_fund",
        )
    except Exception:
        report = _fallback_research_template(
            facet="fundamental",
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=industry_class,
            indicator_scores=indicator_scores,
        )
        warnings = ["FUNDAMENTAL_LLM_FALLBACK"]

    patch: dict = {"research": {"fundamental": report.model_dump()}}
    if warnings:
        patch["warnings"] = warnings
    return patch
