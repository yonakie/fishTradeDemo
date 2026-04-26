"""H2 — R2 (max-drawdown) rejects BUY but the SELL fast-path bypasses it.

Hard rules R2 only fires for BUY decisions. HOLD/SELL short-circuits in
``hard_rules_node`` *before* the drawdown gate, so a SELL on a wrecked
portfolio still gets approve@0% and reaches ``skip_execution``.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph
from fishtrade.models.portfolio import NavSnapshot

from tests.integration._fixtures import make_portfolio


def _wrecked_portfolio_dump():
    """16.7% peak-to-trough — well past the 8% threshold."""
    return make_portfolio(
        nav_history=[
            NavSnapshot(date="2026-04-01", nav=100_000),
            NavSnapshot(date="2026-04-15", nav=120_000),
            NavSnapshot(date="2026-04-22", nav=100_000),
        ]
    ).model_dump()


def test_buy_rejected_when_drawdown_exceeded(patch_ark, initial_state):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = graph.invoke(
        initial_state(portfolio_before=_wrecked_portfolio_dump(), run_id="h2-buy"),
        config={"configurable": {"thread_id": "h2-buy"}},
    )
    assert final["risk"]["decision"] == "reject"
    assert "R2_MAX_DRAWDOWN" in (final["risk"].get("reject_reason") or "")
    assert final.get("execution") in (None, {})


def test_sell_passes_through_drawdown_gate(patch_ark, initial_state):
    """HOLD/SELL fast-path skips R2 entirely (proposed_position_pct == 0)."""
    patch_ark(judge_verdict="SELL", judge_pct=0.0)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = graph.invoke(
        initial_state(portfolio_before=_wrecked_portfolio_dump(), run_id="h2-sell"),
        config={"configurable": {"thread_id": "h2-sell"}},
    )
    assert final["risk"]["decision"] == "approve"
    assert final["risk"]["adjusted_position_pct"] == 0.0
    assert final["execution"]["status"] == "skipped"
