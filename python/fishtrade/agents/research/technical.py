"""Technical research node — drives indicator engine + LLM narration."""

from __future__ import annotations

from ...models.state import GraphState
from ...tools.indicators_tech import compute_all_technical
from ._common import _fallback_research_template, run_research_facet


def technical_node(state: GraphState) -> dict:
    run_input = state.get("input") or {}
    market_data = state.get("market_data") or {}
    ticker = run_input.get("ticker", "UNKNOWN")
    as_of_date = run_input.get("as_of_date", "")
    run_id = state.get("run_id", "ad-hoc")

    indicator_scores = compute_all_technical(market_data)

    try:
        report, warnings = run_research_facet(
            facet="technical",
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=None,
            indicator_scores=indicator_scores,
            run_id=run_id,
            node_name="research_tech",
        )
    except Exception:
        report = _fallback_research_template(
            facet="technical",
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=None,
            indicator_scores=indicator_scores,
        )
        warnings = ["TECHNICAL_LLM_FALLBACK"]

    patch: dict = {"research": {"technical": report.model_dump()}}
    if warnings:
        patch["warnings"] = warnings
    return patch
