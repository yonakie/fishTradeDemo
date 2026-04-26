"""H9 — Soft rule LIQUIDITY_LOW triggers when avg dollar volume < $10M.

We feed market data with a tiny ``averageDailyVolume10Day`` and let the
soft-judge node see the rule flag. We set the LLM mock to return
adjustment="reduce" so the final position is halved.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph
from tests.integration._fixtures import make_market_data


def test_low_liquidity_triggers_soft_reduction(patch_ark, initial_state):
    # Mock soft judge returns a LIQUIDITY_LOW + reduce decision (50% of proposed).
    patch_ark(
        judge_verdict="BUY",
        judge_pct=8.0,
        soft_flags=["LIQUIDITY_LOW"],
        soft_adjustment="reduce",
        soft_pct=4.0,
    )

    md = make_market_data()
    md["info"] = {**md["info"], "averageDailyVolume10Day": 1_000}

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = graph.invoke(
        initial_state(market_data=md, run_id="h9-liquidity"),
        config={"configurable": {"thread_id": "h9-liquidity"}},
    )
    assert final["risk"]["decision"] == "approve"
    assert final["risk"]["adjusted_position_pct"] == 4.0
    assert "LIQUIDITY_LOW" in final["risk"]["soft_judgment"]["flags"]
    assert final["risk"]["soft_judgment"]["adjustment"] == "reduce"
