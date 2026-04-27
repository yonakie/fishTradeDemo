"""Bull-side debate nodes (opening + rebuttal)."""

from __future__ import annotations

from ...models.state import GraphState
from ._common import coerce_turns, run_debate_turn


def _bull_node(state: GraphState, *, round_idx: int, node_name: str) -> dict:
    run_id = state.get("run_id", "ad-hoc")
    turn, warnings = run_debate_turn(
        role="bull",
        round_idx=round_idx,
        state=dict(state),
        run_id=run_id,
        node_name=node_name,
    )
    patch: dict = {"debate_turns": [turn.model_dump()]}
    if warnings:
        patch["warnings"] = warnings
    return patch


def debate_opening_bull_node(state: GraphState) -> dict:
    """Round 0 (opening) bull turn."""
    return _bull_node(state, round_idx=0, node_name="debate_open_bull")


def debate_rebuttal_bull_node(state: GraphState) -> dict:
    """Subsequent rebuttal — uses round inferred from existing turns + 1."""
    prior = coerce_turns(state.get("debate_turns") or [])
    next_round = max((t.round for t in prior), default=0) + 1
    next_round = min(next_round, 3)
    return _bull_node(
        state, round_idx=next_round, node_name="debate_rebuttal_bull"
    )


__all__ = ["debate_opening_bull_node", "debate_rebuttal_bull_node"]
