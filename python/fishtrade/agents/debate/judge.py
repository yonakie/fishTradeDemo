"""Judge node — finalises the DebateResult given research + accumulated turns."""

from __future__ import annotations

from ...models.state import GraphState
from ._common import run_debate_judge


def debate_judge_node(state: GraphState) -> dict:
    run_id = state.get("run_id", "ad-hoc")
    result, warnings = run_debate_judge(
        state=dict(state),
        run_id=run_id,
        node_name="debate_judge",
    )
    patch: dict = {"debate": result.model_dump()}
    if warnings:
        patch["warnings"] = warnings
    return patch


__all__ = ["debate_judge_node"]
