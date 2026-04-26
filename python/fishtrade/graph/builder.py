"""LangGraph workflow assembly — single entry point :func:`build_graph`.

The graph layer is responsible for *composition only*:
- registering all Session-3 agent nodes under the names from §5.1;
- wiring stub placeholders for layer-G/H nodes that don't exist yet
  (``validate_input`` / ``fetch_market`` / ``hitl_gate`` /
  ``render_report``) so the topology compiles end-to-end;
- declaring a *private* :class:`BuilderState` that carries the
  ``risk_partial`` scratchpad shared across the three risk nodes,
  *without* polluting the public :class:`GraphState` contract.

Per Session 4 contract:
- public ``GraphState`` is **not** modified — its reducer for ``research``
  is already wired in :mod:`fishtrade.models.state`.
- agent nodes from Session 3 are imported as-is and registered verbatim.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ..agents.debate import (
    debate_judge_node,
    debate_opening_bear_node,
    debate_opening_bull_node,
    debate_rebuttal_bear_node,
    debate_rebuttal_bull_node,
)
from ..agents.execution import (
    backtest_node,
    dryrun_node,
    execution_router,
    paper_node,
    skip_execution_node,
    update_portfolio_node,
)
from ..agents.research import fundamental_node, sentimental_node, technical_node
from ..agents.risk import hard_rules_node, soft_judge_node, var_check_node
from ..models.state import GraphState
from .routes import (
    route_after_hard,
    route_after_hitl,
    route_after_soft,
    route_after_var,
)


# ---------------------------------------------------------------------------
# Private graph-implementation state types
# ---------------------------------------------------------------------------


class _RiskPartial(TypedDict, total=False):
    """Scratchpad shared by ``hard_rules → var_check → soft_judge``.

    Holds serialised ``HardCheckResult`` / ``VarResult`` payloads. This
    type is private to the builder — callers should never import it; the
    risk nodes interact with it via ``state.get("risk_partial")``.
    """

    hard_checks: list[dict]
    var_result: dict


class BuilderState(GraphState, total=False):
    """Implementation-only superset of :class:`GraphState`.

    Extends the public contract with two graph-internal channels:

    * ``risk_partial`` — see :class:`_RiskPartial`.
    * ``hitl_decision`` — written by the CLI on resume, read by
      :func:`route_after_hitl`.
    """

    risk_partial: _RiskPartial
    hitl_decision: str


# ---------------------------------------------------------------------------
# Stub nodes for Session-G/H layers (TODO: replace with real impls)
# ---------------------------------------------------------------------------


def _validate_input_stub(state: BuilderState) -> dict:
    """TODO(session 5): replace with real impl in cli/reporting layer.

    Real version will validate the ticker against yfinance and raise
    ``INVALID_TICKER``. The stub trusts whatever the CLI put in
    ``state['input']`` and is a no-op.
    """
    return {}


def _fetch_market_stub(state: BuilderState) -> dict:
    """TODO(session 5): replace with real impl in cli/reporting layer.

    Real version will pull yfinance bundle via ``tools.yf_client``. The
    stub assumes ``state['market_data']`` was pre-seeded by the caller
    (CLI or test fixture).
    """
    return {}


def _hitl_gate_stub(state: BuilderState) -> dict:
    """TODO(session 5): replace with real impl in cli/reporting layer.

    Real version will be reached only when ``input.hitl=True``; the
    graph's ``interrupt_before=["hitl_gate"]`` causes a checkpoint
    pause *before* this node runs. After the CLI calls
    ``graph.update_state({'hitl_decision': ...})`` and ``invoke(None)``,
    this node runs with the decision already written and is itself a
    no-op (the routing happens on its outgoing edges).
    """
    return {}


def _render_report_stub(state: BuilderState) -> dict:
    """TODO(session 5): replace with real impl in cli/reporting layer.

    Real version will render the bilingual markdown report to disk
    via ``reporting/render.py`` and append to ``state['warnings']`` on
    template failure.
    """
    return {}


def _execute_router_dispatch_stub(state: BuilderState) -> dict:
    """Pass-through node so ``execution_router`` can hang off a real node.

    Choice: rather than fuse HITL gating + execution mode dispatch into
    one router (which would obscure trace boundaries), we keep
    ``hitl_gate`` strictly approval/reject and add this no-op as the
    anchor for the mode-based fan-out. Picks one of the two options the
    Session-4 brief offered: *insert a pass-through node*.
    """
    return {}


def _debate_rebuttal_node(state: BuilderState) -> dict:
    """Composite rebuttal node — runs bull then bear sequentially.

    The design doc registers a single ``debate_rebuttal`` node, but
    Session-3 split bull/bear for unit-test isolation. We compose them
    here without modifying agent code. Skips work entirely when
    ``input.debate_rounds == 0`` (per design 4.2.1: rounds=0 means
    opening-only debate).
    """
    rounds = int((state.get("input") or {}).get("debate_rounds", 0) or 0)
    if rounds <= 0:
        return {}

    # Bull writes a turn; we then run bear over the *same* state slice so
    # both compute next_round consistently from the existing turns.
    bull_patch = debate_rebuttal_bull_node(state)
    bear_patch = debate_rebuttal_bear_node(state)

    merged_turns = list(bull_patch.get("debate_turns") or []) + list(
        bear_patch.get("debate_turns") or []
    )
    merged_warnings = list(bull_patch.get("warnings") or []) + list(
        bear_patch.get("warnings") or []
    )
    patch: dict[str, Any] = {}
    if merged_turns:
        patch["debate_turns"] = merged_turns
    if merged_warnings:
        patch["warnings"] = merged_warnings
    return patch


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_graph(
    *,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
):
    """Compile and return the full multi-agent decision graph.

    Parameters
    ----------
    checkpointer:
        Any LangGraph ``BaseCheckpointSaver`` (e.g. ``SqliteSaver``,
        ``MemorySaver``). If omitted, the compiled graph has no
        checkpointer and HITL pause/resume cannot work — caller's
        responsibility (CLI passes a SqliteSaver; tests pass MemorySaver).
    interrupt_before:
        Override the default ``["hitl_gate"]``. Tests that want to skip
        the HITL pause entirely pass ``[]``.
    """
    g: StateGraph = StateGraph(BuilderState)

    # —— nodes (names exactly per design §5.1) ——
    g.add_node("validate_input", _validate_input_stub)
    g.add_node("fetch_market", _fetch_market_stub)
    g.add_node("research_fund", fundamental_node)
    g.add_node("research_tech", technical_node)
    g.add_node("research_sent", sentimental_node)
    g.add_node("debate_open_bull", debate_opening_bull_node)
    g.add_node("debate_open_bear", debate_opening_bear_node)
    g.add_node("debate_rebuttal", _debate_rebuttal_node)
    g.add_node("debate_judge", debate_judge_node)
    g.add_node("risk_hard", hard_rules_node)
    g.add_node("risk_var", var_check_node)
    g.add_node("risk_soft", soft_judge_node)
    g.add_node("hitl_gate", _hitl_gate_stub)
    g.add_node("execute_router_dispatch", _execute_router_dispatch_stub)
    g.add_node("execute_dryrun", dryrun_node)
    g.add_node("execute_paper", paper_node)
    g.add_node("execute_backtest", backtest_node)
    g.add_node("skip_execution", skip_execution_node)
    g.add_node("update_portfolio", update_portfolio_node)
    g.add_node("render_report", _render_report_stub)

    # —— entry & data ——
    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "fetch_market")

    # —— Fan-out: 3-way parallel research ——
    g.add_edge("fetch_market", "research_fund")
    g.add_edge("fetch_market", "research_tech")
    g.add_edge("fetch_market", "research_sent")

    # —— Fan-in to debate openings (bull & bear run in parallel) ——
    for src in ("research_fund", "research_tech", "research_sent"):
        g.add_edge(src, "debate_open_bull")
        g.add_edge(src, "debate_open_bear")

    # —— Debate flow ——
    g.add_edge("debate_open_bull", "debate_rebuttal")
    g.add_edge("debate_open_bear", "debate_rebuttal")
    g.add_edge("debate_rebuttal", "debate_judge")

    # —— Risk: serial chain with short-circuit on reject ——
    g.add_edge("debate_judge", "risk_hard")
    g.add_conditional_edges(
        "risk_hard",
        route_after_hard,
        {"continue": "risk_var", "reject": "render_report"},
    )
    g.add_conditional_edges(
        "risk_var",
        route_after_var,
        {"continue": "risk_soft", "reject": "render_report"},
    )
    g.add_conditional_edges(
        "risk_soft",
        route_after_soft,
        {"continue": "hitl_gate", "reject": "render_report"},
    )

    # —— HITL approval gate ——
    g.add_conditional_edges(
        "hitl_gate",
        route_after_hitl,
        {"approved": "execute_router_dispatch", "rejected": "render_report"},
    )

    # —— Execution mode dispatch ——
    g.add_conditional_edges(
        "execute_router_dispatch",
        execution_router,
        {
            "execute_dryrun": "execute_dryrun",
            "execute_paper": "execute_paper",
            "execute_backtest": "execute_backtest",
            "skip_execution": "skip_execution",
        },
    )

    for n in ("execute_dryrun", "execute_paper", "execute_backtest", "skip_execution"):
        g.add_edge(n, "update_portfolio")
    g.add_edge("update_portfolio", "render_report")
    g.add_edge("render_report", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=(
            interrupt_before if interrupt_before is not None else ["hitl_gate"]
        ),
    )


__all__ = ["build_graph"]
