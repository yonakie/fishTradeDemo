"""H-extra — HITL rejected path: ``hitl_decision='rejected'`` skips execution.

Mirrors the test in tests/integration/test_graph_e2e.py but kept here so
the boundary suite carries the complete error-code coverage matrix.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph


def test_hitl_rejected_skips_execution(patch_ark, initial_state):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    saver = MemorySaver()
    graph = build_graph(checkpointer=saver)
    state = initial_state(hitl=True, run_id="h-hitl-rej")
    cfg = {"configurable": {"thread_id": state["run_id"]}}

    paused = graph.invoke(state, config=cfg)
    snap = graph.get_state(cfg)
    assert snap.next == ("hitl_gate",)
    assert paused["risk"]["decision"] == "approve"

    graph.update_state(cfg, {"hitl_decision": "rejected"})
    final = graph.invoke(None, config=cfg)

    assert "execution" not in final or final.get("execution") is None
    assert "portfolio_after" not in final or final.get("portfolio_after") is None
    assert final.get("hitl_decision") == "rejected"
