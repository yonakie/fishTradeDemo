"""Conditional-edge router functions for the LangGraph builder.

All routers are pure functions over ``GraphState`` (or its private
``BuilderState`` superset) — they read state and return a string label
that maps to a destination node in ``add_conditional_edges``. They never
call the LLM, never mutate state, and never raise.
"""

from __future__ import annotations

from typing import Any

from ..models.state import GraphState


def _decision_of(state: GraphState) -> str | None:
    """Return ``state['risk']['decision']`` if present, else ``None``."""
    risk = state.get("risk") or {}
    if not isinstance(risk, dict):
        return None
    decision = risk.get("decision")
    return decision if isinstance(decision, str) else None


def route_after_hard(state: GraphState) -> str:
    """After ``risk_hard``: jump to ``render_report`` on reject, else ``risk_var``.

    ``hard_rules_node`` writes a terminal ``RiskDecision`` (decision=reject)
    into state on R1/R2/R4 failure. The HOLD/SELL fast-path also stamps an
    ``approve@0%`` decision; we keep that flowing through the chain so VaR
    and soft_judge can short-circuit consistently.
    """
    return "reject" if _decision_of(state) == "reject" else "continue"


def route_after_var(state: GraphState) -> str:
    """After ``risk_var``: reject → render_report, else risk_soft."""
    return "reject" if _decision_of(state) == "reject" else "continue"


def route_after_soft(state: GraphState) -> str:
    """After ``risk_soft``: reject → render_report, else hitl_gate."""
    return "reject" if _decision_of(state) == "reject" else "continue"


def route_after_hitl(state: GraphState) -> str:
    """Read ``state['hitl_decision']`` (CLI writes this on resume).

    Defaults to ``approved`` so non-HITL runs flow straight through.
    """
    decision: Any = state.get("hitl_decision", "approved")
    if decision == "rejected":
        return "rejected"
    return "approved"


__all__ = [
    "route_after_hard",
    "route_after_hitl",
    "route_after_soft",
    "route_after_var",
]
