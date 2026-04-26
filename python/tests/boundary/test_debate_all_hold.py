"""H4 — Judge returns HOLD → risk approve@0% → skip_execution."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph


def test_judge_hold_short_circuits_to_skip_execution(patch_ark, initial_state):
    patch_ark(judge_verdict="HOLD", judge_pct=0.0)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = graph.invoke(
        initial_state(run_id="h4-hold"),
        config={"configurable": {"thread_id": "h4-hold"}},
    )
    assert final["debate"]["final_verdict"] == "HOLD"
    assert final["risk"]["decision"] == "approve"
    assert final["risk"]["adjusted_position_pct"] == 0.0
    assert final["execution"]["status"] == "skipped"
    # The dryrun side-effect file must not be written for skip_execution.
    assert final["execution"]["order"] is None
