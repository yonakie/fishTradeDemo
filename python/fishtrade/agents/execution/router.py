"""Conditional router + skip-execution node.

LangGraph wires this as ``add_conditional_edges("execute_router_dispatch",
execution_router, {...})``. The router is a pure function over state.
"""

from __future__ import annotations

from ...models.execution import ExecutionResult
from ...models.state import GraphState


def execution_router(state: GraphState) -> str:
    """Return the next node name based on risk decision + run mode."""
    risk = state.get("risk") or {}
    if not risk:
        return "skip_execution"
    if risk.get("decision") == "reject":
        return "skip_execution"
    if (risk.get("adjusted_position_pct") or 0) <= 0:
        return "skip_execution"

    mode = (state.get("input") or {}).get("mode", "dryrun")
    return {
        "dryrun": "execute_dryrun",
        "paper": "execute_paper",
        "backtest": "execute_backtest",
    }.get(mode, "execute_dryrun")


def skip_execution_node(state: GraphState) -> dict:
    """Stamp a ``skipped`` ExecutionResult; portfolio_update treats this as no-op."""
    risk = state.get("risk") or {}
    mode = (state.get("input") or {}).get("mode", "dryrun")
    error: str | None = None
    if risk.get("decision") == "reject":
        error = risk.get("reject_reason") or "risk_rejected"
    elif (risk.get("adjusted_position_pct") or 0) <= 0:
        error = "approved_at_zero_position"
    result = ExecutionResult(
        mode=mode,  # type: ignore[arg-type]
        order=None,
        status="skipped",
        fill_info=None,
        error=error,
    )
    return {"execution": result.model_dump()}


__all__ = ["execution_router", "skip_execution_node"]
