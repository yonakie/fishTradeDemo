"""Sentimental research node — drives 10 sentiment indicators + LLM narration."""

from __future__ import annotations

from ...models.state import GraphState
from ...tools.indicators_sent import compute_all_sentimental
from ._common import _fallback_research_template, run_research_facet


def sentimental_node(state: GraphState) -> dict:
    run_input = state.get("input") or {}
    market_data = state.get("market_data") or {}
    ticker = run_input.get("ticker", "UNKNOWN")
    as_of_date = run_input.get("as_of_date", "")
    run_id = state.get("run_id", "ad-hoc")

    indicator_scores = compute_all_sentimental(market_data)

    try:
        report, warnings = run_research_facet(
            facet="sentimental",
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=None,
            indicator_scores=indicator_scores,
            run_id=run_id,
            node_name="research_sent",
        )
    except Exception:
        report = _fallback_research_template(
            facet="sentimental",
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=None,
            indicator_scores=indicator_scores,
        )
        warnings = ["SENTIMENTAL_LLM_FALLBACK"]

    patch: dict = {"research": {"sentimental": report.model_dump()}}
    if warnings:
        patch["warnings"] = warnings
    return patch
