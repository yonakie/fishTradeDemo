"""LangGraph workflow assembly — single entry point :func:`build_graph`.

The graph layer is responsible for *composition only*:
- registering all Session-3 agent nodes under the names from §5.1;
- providing the small "plumbing" nodes (``validate_input`` /
  ``fetch_market`` / ``hitl_gate`` / ``render_report``) that bridge the
  CLI boundary with the agent layer;
- declaring a *private* :class:`BuilderState` that carries the
  ``risk_partial`` scratchpad shared across the three risk nodes and the
  ``hitl_decision`` channel written by the CLI on resume — *without*
  polluting the public :class:`GraphState` contract.

Per Session 4 contract:
- public ``GraphState`` is **not** modified — its reducer for ``research``
  is already wired in :mod:`fishtrade.models.state`.
- agent nodes from Session 3 are imported as-is and registered verbatim.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
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
from ..observability.logger import get_logger
from ..observability.node_log import wrap_node
from ..tools.yf_client import (
    InvalidTickerError,
    YFinanceClient,
    YFRateLimitError,
    _df_to_payload,
)
from .routes import (
    route_after_hard,
    route_after_hitl,
    route_after_soft,
    route_after_validate,
    route_after_var,
)

logger = get_logger(__name__)
_TICKER_RE = re.compile(r"^[A-Z.\-]{1,6}$")


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


def validate_input_node(state: BuilderState) -> dict:
    """Validate the ticker via regex + a yfinance ``get_info`` round-trip.

    On any validation failure we set ``halt_reason`` to ``INVALID_TICKER``
    and *do not* raise — the routing logic in the rest of the graph
    treats ``halt_reason`` as a terminal signal (every downstream
    business node short-circuits when ``halt_reason`` is set).

    The CLI does its own regex pre-flight; this node defends against
    callers that bypass the CLI (tests, ad-hoc scripting).
    """
    run_input = state.get("input") or {}
    ticker = str(run_input.get("ticker") or "").strip().upper()

    if not ticker or not _TICKER_RE.match(ticker):
        return {
            "halt_reason": "INVALID_TICKER",
            "warnings": [f"INVALID_TICKER:{ticker or '<empty>'}"],
        }

    # If the caller pre-seeded market_data (test fixture), skip the live
    # yfinance probe — they own the data contract.
    if state.get("market_data"):
        return {}

    client = YFinanceClient()
    try:
        info = client.get_info(ticker, as_of=run_input.get("as_of_date"))
    except InvalidTickerError as exc:
        logger.warning("validate_input_invalid_ticker", ticker=ticker, error=str(exc))
        return {
            "halt_reason": "INVALID_TICKER",
            "warnings": [f"INVALID_TICKER:{ticker}"],
        }
    except YFRateLimitError as exc:
        logger.warning("validate_input_yf_rate_limit", ticker=ticker, error=str(exc))
        # Degrade rather than halt: we may still have cached data.
        return {"warnings": [f"YF_RATE_LIMIT:get_info:{ticker}"]}

    # Sanitise info through JSON round-trip to strip any non-primitive types
    # (e.g. pandas.Timestamp) that would be blocked by msgpack on checkpoint read.
    safe_info = json.loads(json.dumps(info, default=str))
    md_seed: dict = {"info": safe_info, "fetch_warnings": []}
    return {"market_data": md_seed}


def fetch_market_node(state: BuilderState) -> dict:
    """Pull yfinance bundle (history / financials / cashflow / balance / VIX).

    Already-populated keys (e.g. seeded by ``validate_input_node`` or a
    test fixture) are preserved; we only fill in what's missing. All
    DataFrames are serialised via :func:`_df_to_payload` so the
    checkpoint msgpack writer can round-trip them.
    """
    if state.get("halt_reason"):
        return {}

    run_input = state.get("input") or {}
    ticker = str(run_input.get("ticker") or "").strip().upper()
    as_of = run_input.get("as_of_date")
    md = dict(state.get("market_data") or {})
    fetch_warnings: list[str] = list(md.get("fetch_warnings") or [])

    client = YFinanceClient()

    def _safe(key: str, fn, *, payload: bool = True) -> Any:
        if key in md and md[key] is not None:
            return md[key]
        try:
            value = fn()
        except YFRateLimitError as exc:
            fetch_warnings.append(f"YF_RATE_LIMIT:{key}")
            logger.warning("fetch_market_rate_limit", key=key, error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 — yfinance leaks many error types
            fetch_warnings.append(f"YF_FETCH_FAILED:{key}")
            logger.warning("fetch_market_failed", key=key, error=str(exc))
            return None
        if payload and isinstance(value, pd.DataFrame):
            return _df_to_payload(value)
        return value

    if "info" not in md or md.get("info") is None:
        info = _safe("info", lambda: client.get_info(ticker, as_of=as_of), payload=False)
        # Strip non-primitive types (e.g. pandas.Timestamp on dividendDate) so
        # the langgraph msgpack writer never sees them.
        md["info"] = json.loads(json.dumps(info, default=str)) if info else info
    md["history"] = _safe("history", lambda: client.get_history(ticker, period="1y", as_of=as_of))
    md["financials"] = _safe("financials", lambda: client.get_financials(ticker, as_of=as_of), payload=False)
    md["cashflow"] = _safe("cashflow", lambda: client.get_cashflow(ticker, as_of=as_of), payload=False)
    md["balance_sheet"] = _safe("balance_sheet", lambda: client.get_balance_sheet(ticker, as_of=as_of), payload=False)
    if "benchmark_history" not in md or md.get("benchmark_history") is None:
        md["benchmark_history"] = _safe(
            "benchmark_history", lambda: client.get_history("SPY", period="1y", as_of=as_of)
        )
    if "vix_recent" not in md or md.get("vix_recent") is None:
        md["vix_recent"] = _safe(
            "vix_recent", lambda: client.get_history("^VIX", period="1mo", as_of=as_of)
        )

    # Optional fields — failure is non-fatal but recorded.
    if "options_chain" not in md:
        try:
            md["options_chain"] = client.get_option_chain_safe(ticker)
        except Exception:  # noqa: BLE001
            md["options_chain"] = None
    md["fetch_warnings"] = fetch_warnings

    patch: dict = {"market_data": md}
    if fetch_warnings:
        patch["warnings"] = list(fetch_warnings)
    return patch


def hitl_gate_node(state: BuilderState) -> dict:
    """No-op: ``interrupt_before=["hitl_gate"]`` pauses *before* this node.

    The CLI writes ``hitl_decision`` via ``graph.update_state`` during
    the pause. The decision is consumed by :func:`route_after_hitl` on
    this node's outgoing conditional edges.
    """
    return {}


def render_report_node(state: BuilderState) -> dict:
    """Render the bilingual markdown report and write it to disk.

    Failure is *never* terminal — we append ``RENDER_FAILED`` to
    ``warnings`` and return so the run can still close cleanly.
    """
    # Local import to avoid a circular import on package init.
    from ..reporting.render import write_report

    run_input = state.get("input") or {}
    language = str(run_input.get("language") or "bilingual")
    try:
        path = write_report(state, language=language)
        logger.info("report_written", path=str(path))
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("render_report_failed", error=str(exc))
        return {"warnings": [f"RENDER_FAILED:{type(exc).__name__}"]}


def _execute_router_dispatch_node(state: BuilderState) -> dict:
    """Pass-through anchor for ``execution_router``'s mode-based fan-out.

    Keeping this as a separate node (rather than collapsing routing into
    ``hitl_gate``) preserves a clean trace boundary.
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
    # Each node is wrapped with `wrap_node` so the console/frontend gets
    # a uniform start / complete / content / failed event stream.
    def _add(name: str, fn) -> None:
        g.add_node(name, wrap_node(name, fn))

    _add("validate_input", validate_input_node)
    _add("fetch_market", fetch_market_node)
    _add("research_fund", fundamental_node)
    _add("research_tech", technical_node)
    _add("research_sent", sentimental_node)
    _add("debate_open_bull", debate_opening_bull_node)
    _add("debate_open_bear", debate_opening_bear_node)
    _add("debate_rebuttal", _debate_rebuttal_node)
    _add("debate_judge", debate_judge_node)
    _add("risk_hard", hard_rules_node)
    _add("risk_var", var_check_node)
    _add("risk_soft", soft_judge_node)
    _add("hitl_gate", hitl_gate_node)
    _add("execute_router_dispatch", _execute_router_dispatch_node)
    _add("execute_dryrun", dryrun_node)
    _add("execute_paper", paper_node)
    _add("execute_backtest", backtest_node)
    _add("skip_execution", skip_execution_node)
    _add("update_portfolio", update_portfolio_node)
    _add("render_report", render_report_node)

    # —— entry & data ——
    g.add_edge(START, "validate_input")
    # validate_input may halt (INVALID_TICKER) — skip straight to render_report.
    g.add_conditional_edges(
        "validate_input",
        route_after_validate,
        {"continue": "fetch_market", "halt": "render_report"},
    )

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
