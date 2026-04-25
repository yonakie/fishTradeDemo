"""trace.py + metrics.py: JSONL writer and aggregator."""

from __future__ import annotations

import json
from pathlib import Path

from fishtrade.observability.metrics import aggregate_run
from fishtrade.observability.trace import iter_trace, write_llm_trace


def test_write_llm_trace_appends_jsonl(tmp_path: Path, monkeypatch):
    from fishtrade.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "log_dir", tmp_path)

    write_llm_trace(
        run_id="abc",
        node="n1",
        model_id="ep-1",
        prompt=[{"role": "user", "content": "x"}],
        response={"foo": "bar"},
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        latency_ms=42,
    )
    write_llm_trace(
        run_id="abc",
        node="n1",
        model_id="ep-1",
        prompt=[{"role": "user", "content": "x2"}],
        response={"foo": "baz"},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        latency_ms=10,
    )

    p = tmp_path / "trace" / "abc.jsonl"
    assert p.exists()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    msgs_p = tmp_path / "trace" / "abc_messages.jsonl"
    assert msgs_p.exists()
    body_lines = msgs_p.read_text(encoding="utf-8").strip().splitlines()
    assert len(body_lines) == 2
    parsed = json.loads(body_lines[0])
    assert "messages_hash" in parsed
    assert parsed["prompt"][0]["content"] == "x"


def test_aggregate_run_sums_counters(tmp_path: Path, monkeypatch):
    from fishtrade.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "log_dir", tmp_path)

    for i in range(3):
        write_llm_trace(
            run_id="r",
            node="n",
            model_id="ep",
            prompt=[{"role": "user", "content": f"x{i}"}],
            response={"i": i},
            usage={"prompt_tokens": 10, "completion_tokens": 4},
            latency_ms=100,
        )
    write_llm_trace(
        run_id="r",
        node="n",
        model_id="ep",
        prompt=[{"role": "user", "content": "err"}],
        response=None,
        usage=None,
        latency_ms=20,
        ok=False,
        error="boom",
    )

    m = aggregate_run("r")
    assert m.calls == 4
    assert m.prompt_tokens == 30
    assert m.completion_tokens == 12
    assert m.tokens_total == 42
    assert m.latency_ms_total == 320
    assert m.errors == 1


def test_iter_trace_missing_returns_nothing(tmp_path: Path, monkeypatch):
    from fishtrade.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "log_dir", tmp_path)
    assert list(iter_trace("nonexistent")) == []
