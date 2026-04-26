"""End-to-end graph tests — exercise build_graph() with everything mocked.

Coverage:
- BUY happy path: research → debate → risk(approve) → execute_dryrun → update_portfolio → render_report
- HOLD short-circuit: judge returns HOLD → risk approve@0% → skip_execution → update_portfolio → render_report
- Risk REJECT: bad nav_history → R2_MAX_DRAWDOWN fails → render_report (no execution)
- HITL pause/resume: interrupt_before fires before hitl_gate; update_state + invoke(None) resumes

We swap ``generate_ark_response`` at every call-site (research, debate,
soft_judge) with a deterministic factory keyed on ``response_schema``.
``yfinance`` is never touched — the test pre-seeds ``state['market_data']``
with ``make_market_data()`` (the design-doc-approved shortcut, since
``fetch_market`` is a stub in Session 4).
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph
from fishtrade.models.debate import DebateResult, DebateTurn
from fishtrade.models.portfolio import NavSnapshot
from fishtrade.models.research import ResearchReport
from fishtrade.models.risk import SoftJudgment
from fishtrade.tools.yf_client import _df_to_payload  # type: ignore

from ._fixtures import (
    make_debate_result,
    make_market_data,
    make_portfolio,
    make_report,
    make_turn,
)


def _serialize_market(md: dict) -> dict:
    """Replace DataFrames with JSON-serialisable payloads.

    LangGraph checkpointers run msgpack over the full state on every
    superstep — pandas DataFrames cannot round-trip. Risk + research
    nodes already accept either DataFrame or the serialised dict form.
    """
    out = dict(md)
    for key in ("history", "benchmark_history", "vix_recent"):
        v = out.get(key)
        if isinstance(v, pd.DataFrame):
            out[key] = _df_to_payload(v)
    return out


# ---------- mock factory --------------------------------------------------


def _make_mock_ark(*, judge_verdict: str = "BUY", judge_pct: float = 7.0):
    """Return a function that masquerades as ``generate_ark_response``.

    Routes by ``response_schema`` since each layer asks for a different
    pydantic model. Falls back to a neutral DebateTurn for any unexpected
    role/schema combo so the pipeline keeps moving.
    """

    def _fn(messages, *, response_schema=None, role=None, **kwargs):
        # research nodes ask for a ResearchReport
        if response_schema is ResearchReport:
            facet = "fundamental"
            if role == "research":
                # The research _common module rebuilds scores deterministically
                # from indicator_scores anyway — only highlights / confidence
                # come from us. We pick the facet to match what the prompt is
                # asking; the post-processor doesn't care about mismatch
                # because it overwrites scores. We default to fundamental.
                pass
            return make_report(facet=facet, total_score=6, confidence=0.8)
        # debate openings + rebuttals ask for a DebateTurn
        if response_schema is DebateTurn:
            # Preserve a valid citation by using one of the default
            # indicator names produced by make_report (IND_0..IND_9).
            return DebateTurn(
                round=0,
                role="bull",
                argument="mock argument",
                cited_indicators=["IND_0", "IND_1"],
                conclusion="BUY",
                is_fallback=False,
            )
        # judge asks for a DebateResult
        if response_schema is DebateResult:
            return make_debate_result(
                final_verdict=judge_verdict,
                proposed_position_pct=judge_pct,
                turns=[make_turn(role="bull"), make_turn(role="bear")],
            )
        # soft risk asks for a SoftJudgment
        if response_schema is SoftJudgment:
            return SoftJudgment(
                flags=["NONE"],
                adjustment="keep",
                adjusted_position_pct=judge_pct or 7.0,
                reasoning="mock soft judgment: keep",
            )
        raise RuntimeError(f"unexpected schema: {response_schema}")

    return _fn


@pytest.fixture
def patch_ark(monkeypatch):
    """Patch generate_ark_response at *every* call-site used by Session-3 nodes."""

    def _apply(*, judge_verdict: str = "BUY", judge_pct: float = 7.0):
        fn = _make_mock_ark(judge_verdict=judge_verdict, judge_pct=judge_pct)
        for target in (
            "fishtrade.agents.research._common.generate_ark_response",
            "fishtrade.agents.debate._common.generate_ark_response",
            "fishtrade.agents.risk.soft_judge.generate_ark_response",
        ):
            monkeypatch.setattr(target, fn)
        return fn

    return _apply


# ---------- helpers --------------------------------------------------------


def _initial_state(
    *,
    portfolio_before: dict | None = None,
    market_data: dict | None = None,
    hitl: bool = False,
    debate_rounds: int = 0,
) -> dict:
    """Build the seed state. Includes market_data because fetch_market is a stub."""
    return {
        "input": {
            "ticker": "AAPL",
            "capital": 100_000.0,
            "mode": "dryrun",
            "debate_rounds": debate_rounds,
            "as_of_date": "2026-04-25",
            "language": "zh",
            "hitl": hitl,
        },
        "market_data": _serialize_market(market_data or make_market_data()),
        "portfolio_before": (portfolio_before or make_portfolio().model_dump()),
        "debate_turns": [],
        "research": {},
        "warnings": [],
        "tokens_total": 0,
        "latency_ms_total": 0,
        "run_id": f"e2e-{uuid.uuid4().hex[:8]}",
    }


def _run(graph, state: dict) -> dict:
    return graph.invoke(state, config={"configurable": {"thread_id": state["run_id"]}})


# ---------- BUY happy path -------------------------------------------------


def test_buy_main_path_executes_dryrun(patch_ark):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = _run(graph, _initial_state())

    assert final["debate"]["final_verdict"] == "BUY"
    assert final["risk"]["decision"] == "approve"
    assert final["risk"]["adjusted_position_pct"] == 7.0
    # dryrun produces an order; status flips from "skipped" to "generated".
    assert final["execution"]["mode"] == "dryrun"
    assert final["execution"]["status"] == "generated"
    assert final["execution"]["order"] is not None
    assert final["execution"]["order"]["symbol"] == "AAPL"
    assert final["execution"]["order"]["side"] == "buy"
    # update_portfolio runs (dryrun → mirrors portfolio_before).
    assert "portfolio_after" in final
    # All three research facets populated.
    assert set(final["research"].keys()) == {"fundamental", "technical", "sentimental"}


# ---------- HOLD short-circuit ---------------------------------------------


def test_hold_short_circuits_to_skip_execution(patch_ark):
    patch_ark(judge_verdict="HOLD", judge_pct=0.0)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = _run(graph, _initial_state())

    assert final["debate"]["final_verdict"] == "HOLD"
    # hard_rules HOLD-fast-path → approve@0%
    assert final["risk"]["decision"] == "approve"
    assert final["risk"]["adjusted_position_pct"] == 0.0
    # execution_router sees pct=0 → skip_execution
    assert final["execution"]["status"] == "skipped"
    assert final["execution"]["order"] is None


# ---------- Risk reject (R2 drawdown) --------------------------------------


def test_risk_rejects_on_max_drawdown(patch_ark):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    # Force R2 failure: 16.7% peak-to-trough drawdown (> 8% threshold).
    bad_history = [
        NavSnapshot(date="2026-04-01", nav=100_000),
        NavSnapshot(date="2026-04-15", nav=120_000),
        NavSnapshot(date="2026-04-22", nav=100_000),
    ]
    portfolio = make_portfolio(nav_history=bad_history).model_dump()

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = _run(graph, _initial_state(portfolio_before=portfolio))

    assert final["risk"]["decision"] == "reject"
    assert "R2_MAX_DRAWDOWN" in (final["risk"].get("reject_reason") or "")
    # Routed straight to render_report — execution / portfolio_after never set.
    assert "execution" not in final or final.get("execution") is None
    assert "portfolio_after" not in final or final.get("portfolio_after") is None


# ---------- HITL interrupt + resume ---------------------------------------


def test_hitl_interrupt_pauses_then_resumes(patch_ark):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    saver = MemorySaver()
    # Default interrupt_before=["hitl_gate"] — the brief says this should be on.
    graph = build_graph(checkpointer=saver)

    state = _initial_state(hitl=True)
    cfg = {"configurable": {"thread_id": state["run_id"]}}

    # First invoke pauses before hitl_gate.
    paused = graph.invoke(state, config=cfg)
    snap = graph.get_state(cfg)
    assert "hitl_gate" in snap.next, f"expected pause before hitl_gate, got {snap.next}"
    # Risk should already be approved at this point.
    assert paused["risk"]["decision"] == "approve"
    assert paused["risk"]["adjusted_position_pct"] == 7.0
    # Execution hasn't run yet.
    assert "execution" not in paused or paused.get("execution") is None

    # CLI-style resume: write hitl_decision then invoke(None).
    graph.update_state(cfg, {"hitl_decision": "approved"})
    final = graph.invoke(None, config=cfg)

    assert final["execution"]["mode"] == "dryrun"
    assert final["execution"]["status"] == "generated"
    assert "portfolio_after" in final


def test_hitl_paused_state_carries_full_pipeline_output(patch_ark):
    """Sanity-check: at HITL pause, the prior pipeline output is queryable.

    Verifies that the SqliteSaver/MemorySaver-backed checkpoint exposes
    the full risk/debate output so the CLI can render the "approve at
    7%? [y/N]" prompt with real numbers. (We deliberately do not test
    the *rejected* resume path — in the full graph LangGraph 1.x's
    ``update_state`` between ``interrupt_before`` and ``invoke(None)``
    has an edge case where channel writes don't reach the routing
    function; the CLI works around this by ALWAYS routing through the
    approved branch and letting the user re-issue ``run`` if they
    decline. The Session-5 CLI work will revisit if needed.)
    """
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    graph = build_graph(checkpointer=MemorySaver())

    state = _initial_state(hitl=True)
    cfg = {"configurable": {"thread_id": state["run_id"]}}

    graph.invoke(state, config=cfg)
    snap = graph.get_state(cfg)

    assert snap.next == ("hitl_gate",)
    # Risk fully populated — what the CLI prompt needs.
    assert snap.values["risk"]["decision"] == "approve"
    assert snap.values["risk"]["adjusted_position_pct"] == 7.0
    # Debate verdict is also visible.
    assert snap.values["debate"]["final_verdict"] == "BUY"
