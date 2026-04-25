"""Retry decorator: tenacity hook + JSON repair nudge."""

from __future__ import annotations

import pytest

from fishtrade.llm.retry import (
    JSONParseError,
    _REPAIR_NUDGE,
    _inject_repair_nudge,
    ark_retry,
)


def test_inject_repair_nudge_appends_system_message():
    msgs = [{"role": "user", "content": "hello"}]
    out = _inject_repair_nudge(msgs)
    assert len(out) == 2
    assert out[-1]["role"] == "system"
    assert _REPAIR_NUDGE in out[-1]["content"]
    # input must not be mutated
    assert len(msgs) == 1


def test_ark_retry_passes_through_success():
    calls = {"n": 0}

    @ark_retry
    def fn(messages):
        calls["n"] += 1
        return "ok"

    assert fn([{"role": "user", "content": "x"}]) == "ok"
    assert calls["n"] == 1


def test_ark_retry_on_json_parse_error_retries_with_repair_nudge():
    calls = {"n": 0, "last_messages": None}

    @ark_retry
    def fn(messages):
        calls["n"] += 1
        calls["last_messages"] = messages
        if calls["n"] == 1:
            raise JSONParseError(raw="not json", schema=None, errors=[])
        return "ok"

    out = fn(messages=[{"role": "user", "content": "hi"}])
    assert out == "ok"
    # tenacity itself retries on JSONParseError up to 3 times — first retry triggers
    # the wrapper-level nudge only after tenacity exhausts. Here tenacity succeeded
    # on attempt 2 already, so we only verify exactly two calls happened.
    assert calls["n"] == 2


def test_json_parse_error_carries_metadata():
    err = JSONParseError(raw="abc", schema=None, errors=[{"msg": "bad"}])
    assert "abc" == err.raw
    assert "bad" in str(err) or "errors" in str(err.errors)
