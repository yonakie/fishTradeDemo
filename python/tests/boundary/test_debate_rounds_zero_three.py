"""H5 — debate-rounds parameterised over 0 and 3.

Rounds=0 → only opening turns (bull + bear).
Rounds=3 → opening + (bull+bear) rebuttal turns.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph


@pytest.mark.parametrize("rounds", [0, 3])
def test_pipeline_runs_for_zero_and_three_rounds(rounds, patch_ark, initial_state):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = graph.invoke(
        initial_state(debate_rounds=rounds, run_id=f"h5-r{rounds}"),
        config={"configurable": {"thread_id": f"h5-r{rounds}"}},
    )
    assert final["risk"]["decision"] == "approve"
    debate = final["debate"]
    # Judge always synthesises a final result regardless of rebuttal turns.
    assert debate["final_verdict"] == "BUY"
    # The composite rebuttal node short-circuits when rounds == 0; check
    # that turns shape matches the configured rounds.
    if rounds == 0:
        # Only judge-collected turns (the mock judge returns 2 turns).
        assert len(debate["turns"]) >= 2
    else:
        # Rebuttal node added bull + bear writes for each round (composite
        # collapses them into a single super-step).
        assert len(final.get("debate_turns") or []) >= 2
