"""H8 — LLM timeout in research → fallback template + warning.

We force ``generate_ark_response`` (used in the research call-site) to
raise a timeout-style exception and assert that:

1. The pipeline does not crash.
2. The research report ends up with the rule-derived fallback verdict.
3. ``warnings`` contains ``*_LLM_FALLBACK``.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph


def test_research_llm_timeout_falls_back_to_template(monkeypatch, patch_ark, initial_state):
    """Research module hits a timeout; debate / risk still get usable input."""
    # First wire the regular mocks for debate + risk soft.
    patch_ark(judge_verdict="HOLD", judge_pct=0.0)

    # Then override the research call-site to always raise.
    def _boom(*args, **kwargs):
        raise TimeoutError("ARK_TIMEOUT (simulated)")

    monkeypatch.setattr(
        "fishtrade.agents.research._common.generate_ark_response", _boom
    )

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = graph.invoke(
        initial_state(run_id="h8-timeout"),
        config={"configurable": {"thread_id": "h8-timeout"}},
    )

    # Pipeline reached debate/risk (HOLD).
    assert final["risk"]["decision"] == "approve"
    # All three facets fell back to the rule template.
    fallback_warnings = [w for w in final.get("warnings", []) if "LLM_FALLBACK" in w]
    assert len(fallback_warnings) >= 3
