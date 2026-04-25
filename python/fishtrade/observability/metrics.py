"""Aggregate metrics over a run's trace JSONL."""

from __future__ import annotations

from dataclasses import dataclass, field

from .trace import iter_trace


@dataclass
class RunMetrics:
    run_id: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_total: int = 0
    latency_ms_total: int = 0
    errors: int = 0
    by_node: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_total": self.tokens_total,
            "latency_ms_total": self.latency_ms_total,
            "errors": self.errors,
            "by_node": self.by_node,
        }


def aggregate_run(run_id: str) -> RunMetrics:
    """Read the run's index JSONL and return summed counters."""
    m = RunMetrics(run_id=run_id)
    for rec in iter_trace(run_id):
        m.calls += 1
        pt = rec.get("prompt_tokens") or 0
        ct = rec.get("completion_tokens") or 0
        lat = rec.get("latency_ms") or 0
        m.prompt_tokens += pt
        m.completion_tokens += ct
        m.tokens_total += pt + ct
        m.latency_ms_total += lat
        if not rec.get("ok", True):
            m.errors += 1

        node = rec.get("node", "unknown")
        slot = m.by_node.setdefault(
            node, {"calls": 0, "tokens_total": 0, "latency_ms_total": 0, "errors": 0}
        )
        slot["calls"] += 1
        slot["tokens_total"] += pt + ct
        slot["latency_ms_total"] += lat
        if not rec.get("ok", True):
            slot["errors"] += 1
    return m
