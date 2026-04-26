"""Conditional-edge router functions for the LangGraph builder.

All routers are pure functions over the runtime state — they read it and
return a string label that maps to a destination node in
``add_conditional_edges``. They never call the LLM, never mutate state,
and never raise.

Type-hint note: routers are deliberately typed as ``Mapping[str, Any]``
(not ``GraphState``). LangGraph 1.x inspects the annotation and filters
the state passed to the router to only the keys declared in that
TypedDict — annotating with ``GraphState`` would silently strip
``hitl_decision`` (which lives on the builder-private ``BuilderState``
superset), causing ``route_after_hitl`` to always read the default
``"approved"`` even after the CLI writes ``"rejected"`` via
``graph.update_state``. See Session-5 brief, "HITL rejected path".
"""

from __future__ import annotations

from typing import Any, Mapping


def route_after_validate(state: Mapping[str, Any]) -> str:
    """After ``validate_input``: if ``halt_reason`` is set, jump to render_report.

    This catches ``INVALID_TICKER`` (and any future startup-time halt
    code) without touching the agent layer's own contracts.
    """
    return "halt" if state.get("halt_reason") else "continue"


def _decision_of(state: Mapping[str, Any]) -> str | None:
    """Return ``state['risk']['decision']`` if present, else ``None``."""
    risk = state.get("risk") or {}
    if not isinstance(risk, dict):
        return None
    decision = risk.get("decision")
    return decision if isinstance(decision, str) else None


def route_after_hard(state: Mapping[str, Any]) -> str:
    """After ``risk_hard``: jump to ``render_report`` on reject, else ``risk_var``.

    ``hard_rules_node`` writes a terminal ``RiskDecision`` (decision=reject)
    into state on R1/R2/R4 failure. The HOLD/SELL fast-path also stamps an
    ``approve@0%`` decision; we keep that flowing through the chain so VaR
    and soft_judge can short-circuit consistently.
    """
    return "reject" if _decision_of(state) == "reject" else "continue"


def route_after_var(state: Mapping[str, Any]) -> str:
    """After ``risk_var``: reject → render_report, else risk_soft."""
    return "reject" if _decision_of(state) == "reject" else "continue"


def route_after_soft(state: Mapping[str, Any]) -> str:
    """After ``risk_soft``: reject → render_report, else hitl_gate."""
    return "reject" if _decision_of(state) == "reject" else "continue"


def route_after_hitl(state: Mapping[str, Any]) -> str:
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
    "route_after_validate",
    "route_after_var",
]
