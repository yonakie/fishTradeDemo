"""H-extra — RENDER_FAILED warning when the Markdown template fails.

We monkeypatch ``write_report`` inside the builder's import to raise.
The pipeline must still close cleanly with a ``RENDER_FAILED:*``
entry in ``warnings``.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph


def test_render_failed_appends_warning(patch_ark, initial_state, monkeypatch):
    patch_ark(judge_verdict="HOLD", judge_pct=0.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("template explosion")

    # builder.render_report_node imports write_report locally. Patch the
    # source-of-truth module so the local import inside the node sees it.
    monkeypatch.setattr("fishtrade.reporting.render.write_report", _boom)

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = graph.invoke(
        initial_state(run_id="h-render-fail"),
        config={"configurable": {"thread_id": "h-render-fail"}},
    )
    assert any(w.startswith("RENDER_FAILED:") for w in final.get("warnings") or [])
    # Pipeline still produced its decision.
    assert final["risk"]["decision"] == "approve"
