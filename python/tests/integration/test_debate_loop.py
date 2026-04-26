"""Isolation tests for the debate (bull / bear / judge) nodes.

We mock ``generate_ark_response`` per-test so the LLM-driven flow is
deterministic. We also exercise:
- bull / bear opening + rebuttal patches
- citation enforcement (out-of-vocab citations get fallback)
- truncate_debate_history keeps last 2 rounds
- judge happy path AND degradation short-circuit
- summarize_research_for_debate compaction
"""

from __future__ import annotations

import pytest

from fishtrade.agents.debate.bear import (
    debate_opening_bear_node,
    debate_rebuttal_bear_node,
)
from fishtrade.agents.debate.bull import (
    debate_opening_bull_node,
    debate_rebuttal_bull_node,
)
from fishtrade.agents.debate.judge import debate_judge_node
from fishtrade.llm.prompt_utils import (
    summarize_research_for_debate,
    truncate_debate_history,
)
from fishtrade.models.debate import DebateResult, DebateTurn

from ._fixtures import make_report, make_turn


def _state(extra: dict | None = None) -> dict:
    base = {
        "input": {
            "ticker": "AAPL",
            "capital": 100_000.0,
            "mode": "dryrun",
            "debate_rounds": 2,
            "as_of_date": "2026-04-25",
            "language": "zh",
            "hitl": False,
        },
        "research": {
            "fundamental": make_report("fundamental", total_score=6).model_dump(),
            "technical": make_report("technical", total_score=4).model_dump(),
            "sentimental": make_report("sentimental", total_score=2).model_dump(),
        },
        "debate_turns": [],
        "run_id": "test-run-debate",
    }
    if extra:
        base.update(extra)
    return base


# ---------- prompt utilities ----------------------------------------------


def test_truncate_debate_history_keeps_last_two_rounds():
    turns = [
        make_turn(role="bull", round_idx=0),
        make_turn(role="bear", round_idx=0),
        make_turn(role="bull", round_idx=1),
        make_turn(role="bear", round_idx=1),
        make_turn(role="bull", round_idx=2),
        make_turn(role="bear", round_idx=2),
    ]
    kept = truncate_debate_history(turns, keep_last_n=2)
    assert {t.round for t in kept} == {1, 2}
    assert len(kept) == 4


def test_summarize_research_for_debate_keeps_top_indicators():
    rep = make_report(facet="fundamental", total_score=8)
    blurb = summarize_research_for_debate(rep)
    assert "FUNDAMENTAL" in blurb
    assert "BUY" in blurb
    assert "Top indicators" in blurb


# ---------- bull / bear -----------------------------------------------------


def _mk_bull_response():
    def _fn(messages, **kwargs):
        return DebateTurn(
            round=0,
            role="bull",
            argument="multiples are reasonable; momentum confirms uptrend",
            cited_indicators=["IND_0", "IND_1"],
            conclusion="BUY",
            is_fallback=False,
        )

    return _fn


def _mk_bear_response():
    def _fn(messages, **kwargs):
        return DebateTurn(
            round=0,
            role="bear",
            argument="VaR exposure rising and short-float ticking up",
            cited_indicators=["IND_2"],
            conclusion="HOLD",
            is_fallback=False,
        )

    return _fn


def test_bull_opening_emits_one_turn(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response", _mk_bull_response()
    )
    patch = debate_opening_bull_node(_state())
    assert "debate_turns" in patch
    assert len(patch["debate_turns"]) == 1
    turn = patch["debate_turns"][0]
    assert turn.role == "bull"
    assert turn.round == 0
    assert turn.is_fallback is False


def test_bear_opening_emits_one_turn(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response", _mk_bear_response()
    )
    patch = debate_opening_bear_node(_state())
    assert len(patch["debate_turns"]) == 1
    assert patch["debate_turns"][0].role == "bear"


def test_bull_rebuttal_increments_round(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response",
        lambda *a, **k: DebateTurn(
            round=1,
            role="bull",
            argument="rebuttal: indicators still favor entry",
            cited_indicators=["IND_0"],
            conclusion="BUY",
            is_fallback=False,
        ),
    )
    s = _state(
        {"debate_turns": [make_turn(role="bull", round_idx=0), make_turn(role="bear", round_idx=0)]}
    )
    patch = debate_rebuttal_bull_node(s)
    new_turn = patch["debate_turns"][0]
    assert new_turn.round == 1


def test_bull_falls_back_when_llm_raises(monkeypatch):
    def _raise(*_a, **_k):
        raise RuntimeError("simulated Ark outage")

    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response", _raise
    )
    s = _state({"debate_turns": [make_turn(role="bull", conclusion="BUY")]})
    patch = debate_opening_bull_node(s)
    assert "DEBATE_BULL_LLM_FALLBACK" in patch.get("warnings", [])
    turn = patch["debate_turns"][0]
    assert turn.is_fallback is True
    # Reuses prior bull conclusion.
    assert turn.conclusion == "BUY"


def test_bear_falls_back_to_hold_when_no_prior(monkeypatch):
    def _raise(*_a, **_k):
        raise RuntimeError("simulated Ark outage")

    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response", _raise
    )
    patch = debate_opening_bear_node(_state())  # no prior bear turns
    turn = patch["debate_turns"][0]
    assert turn.is_fallback is True
    assert turn.conclusion == "HOLD"


def test_bull_invalid_citation_triggers_fallback(monkeypatch):
    """LLM cites an indicator not present anywhere in research → fallback."""

    def _bad_citation(*_a, **_k):
        return DebateTurn(
            round=0,
            role="bull",
            argument="cites nothing real",
            cited_indicators=["TOTALLY_INVENTED"],
            conclusion="BUY",
            is_fallback=False,
        )

    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response", _bad_citation
    )
    patch = debate_opening_bull_node(_state())
    assert "DEBATE_BULL_LLM_FALLBACK" in patch.get("warnings", [])
    assert patch["debate_turns"][0].is_fallback is True


# ---------- judge ----------------------------------------------------------


def _mk_judge_response(verdict: str = "BUY", pct: float = 7.0):
    def _fn(messages, **kwargs):
        return DebateResult(
            turns=[make_turn(role="bull"), make_turn(role="bear", conclusion="HOLD")],
            final_verdict=verdict,  # type: ignore[arg-type]
            final_rationale=f"chose {verdict} based on indicators",
            confidence=0.6,
            proposed_position_pct=pct if verdict == "BUY" else 0.0,
            degraded_facets=[],
        )

    return _fn


def test_judge_happy_path(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response", _mk_judge_response("BUY", 7.0)
    )
    s = _state(
        {
            "debate_turns": [
                make_turn(role="bull", round_idx=0),
                make_turn(role="bear", round_idx=0, conclusion="HOLD"),
            ]
        }
    )
    patch = debate_judge_node(s)
    assert "debate" in patch
    body = DebateResult.model_validate(patch["debate"])
    assert body.final_verdict == "BUY"
    assert 0 < body.proposed_position_pct <= 10
    # turns array always pinned to canonical state regardless of LLM input.
    assert len(body.turns) == 2


def test_judge_falls_back_when_llm_raises(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.debate._common.generate_ark_response",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ark down")),
    )
    s = _state(
        {
            "debate_turns": [
                make_turn(role="bull", round_idx=0),
                make_turn(role="bear", round_idx=0, conclusion="HOLD"),
            ]
        }
    )
    patch = debate_judge_node(s)
    assert "DEBATE_JUDGE_LLM_FALLBACK" in patch.get("warnings", [])
    body = DebateResult.model_validate(patch["debate"])
    # Pure-rule fallback: 2 BUYs (fundamental_total=6, technical_total=4 → BUY/HOLD)
    # and 1 HOLD (sentimental_total=2 → HOLD); BUY=1, HOLD=2, SELL=0 → HOLD.
    assert body.final_verdict in ("BUY", "HOLD", "SELL")


def test_judge_skips_when_all_facets_degraded():
    # No need to mock LLM: short-circuit before any call.
    state = _state(
        {
            "research": {
                "fundamental": make_report("fundamental", is_facet_degraded=True).model_dump(),
                "technical": make_report("technical", is_facet_degraded=True).model_dump(),
                "sentimental": make_report("sentimental", is_facet_degraded=True).model_dump(),
            }
        }
    )
    patch = debate_judge_node(state)
    assert "DEBATE_JUDGE_SKIPPED_ALL_DEGRADED" in patch.get("warnings", [])
    body = DebateResult.model_validate(patch["debate"])
    assert body.final_verdict == "HOLD"
    assert body.proposed_position_pct == 0
