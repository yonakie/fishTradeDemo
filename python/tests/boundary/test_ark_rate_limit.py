"""H-extra — Ark rate-limit / unrecoverable LLM error in research layer.

When the LLM raises persistently, every research facet falls back to
its rule-only template and emits ``*_LLM_FALLBACK``. The debate /
risk layer also uses fallbacks so the pipeline can still reach a
verdict (likely HOLD on neutral fallback scores).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph


def test_ark_persistent_failure_falls_back_to_rule_templates(monkeypatch, initial_state):
    """Patch *every* generate_ark_response call-site to raise."""

    def _boom(*args, **kwargs):
        raise RuntimeError("ARK_RATE_LIMIT_EXHAUSTED (simulated)")

    for target in (
        "fishtrade.agents.research._common.generate_ark_response",
        "fishtrade.agents.debate._common.generate_ark_response",
        "fishtrade.agents.risk.soft_judge.generate_ark_response",
    ):
        monkeypatch.setattr(target, _boom)

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = graph.invoke(
        initial_state(run_id="h-ark-rl"),
        config={"configurable": {"thread_id": "h-ark-rl"}},
    )

    # Research layer warnings.
    fallbacks = [w for w in final.get("warnings") or [] if "LLM_FALLBACK" in w]
    assert any("FUNDAMENTAL_LLM_FALLBACK" in w for w in fallbacks)
    assert any("TECHNICAL_LLM_FALLBACK" in w for w in fallbacks)
    assert any("SENTIMENTAL_LLM_FALLBACK" in w for w in fallbacks)
    # The risk-soft fallback fires only when the pipeline reaches that
    # node (i.e. when judge produced BUY); on a fallback HOLD path it's
    # short-circuited by hard_rules. We only assert research-tier
    # fallbacks here to stay deterministic across both outcomes.
