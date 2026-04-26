"""Isolation tests for the three risk nodes.

We mock ``generate_ark_response`` only where needed (soft_judge); hard
rules and VaR are pure functions over state.
"""

from __future__ import annotations

import pandas as pd

from fishtrade.agents.risk.hard_rules import hard_rules_node
from fishtrade.agents.risk.soft_judge import soft_judge_node
from fishtrade.agents.risk.var_check import var_check_node
from fishtrade.models.portfolio import NavSnapshot
from fishtrade.models.risk import RiskDecision, SoftJudgment
from fishtrade.tools.yf_client import _df_to_payload  # type: ignore

from ._fixtures import (
    make_debate_result,
    make_history,
    make_market_data,
    make_portfolio,
)


def _state(
    *,
    debate_kwargs: dict | None = None,
    portfolio: dict | None = None,
    market_data: dict | None = None,
    risk: dict | None = None,
    risk_partial: dict | None = None,
) -> dict:
    md = market_data or make_market_data()
    if isinstance(md.get("history"), pd.DataFrame):
        md = {**md, "history": _df_to_payload(md["history"])}
    return {
        "input": {
            "ticker": "AAPL",
            "capital": 100_000.0,
            "mode": "dryrun",
            "debate_rounds": 1,
            "as_of_date": "2026-04-25",
            "language": "zh",
            "hitl": False,
        },
        "debate": make_debate_result(**(debate_kwargs or {})).model_dump(),
        "portfolio_before": (portfolio or make_portfolio().model_dump()),
        "market_data": md,
        "risk": risk,
        "risk_partial": risk_partial,
        "run_id": "test-run-risk",
    }


# ---------- hard rules -----------------------------------------------------


def test_hard_rules_pass_emits_partial():
    patch = hard_rules_node(_state())
    assert "risk_partial" in patch
    assert "risk" not in patch
    checks = patch["risk_partial"]["hard_checks"]
    rules = {c["rule"] for c in checks}
    assert {"R1_POSITION_LIMIT", "R2_MAX_DRAWDOWN", "R4_STOPLOSS"}.issubset(rules)


def test_hard_rules_hold_short_circuits_to_approve_zero():
    """HOLD verdict → approve at 0% (Execution will then skip)."""
    patch = hard_rules_node(_state(debate_kwargs={"final_verdict": "HOLD"}))
    risk = RiskDecision.model_validate(patch["risk"])
    assert risk.decision == "approve"
    assert risk.adjusted_position_pct == 0


def test_hard_rules_drawdown_violation_rejects():
    history = [
        NavSnapshot(date="2026-04-01", nav=100_000),
        NavSnapshot(date="2026-04-15", nav=120_000),
        NavSnapshot(date="2026-04-22", nav=100_000),  # ~16.7% peak-to-trough
    ]
    portfolio = make_portfolio(nav_history=history).model_dump()
    patch = hard_rules_node(_state(portfolio=portfolio))
    risk = RiskDecision.model_validate(patch["risk"])
    assert risk.decision == "reject"
    assert "R2_MAX_DRAWDOWN" in (risk.reject_reason or "")


def test_hard_rules_position_over_limit_rejects():
    """A misbehaving upstream that proposes >10% must be rejected by R1.

    DebateResult validates proposed_position_pct ≤ 10, so we drop a raw
    dict (bypassing the Pydantic model) into the state to exercise R1.
    """
    state = _state()
    state["debate"] = {
        "turns": [],
        "final_verdict": "BUY",
        "final_rationale": "raw payload",
        "confidence": 0.6,
        "proposed_position_pct": 12.0,
        "degraded_facets": [],
    }
    patch = hard_rules_node(state)
    risk = RiskDecision.model_validate(patch["risk"])
    assert risk.decision == "reject"
    assert "R1_POSITION_LIMIT" in (risk.reject_reason or "")


# ---------- VaR check ------------------------------------------------------


def test_var_check_pass_extends_partial():
    """Big history + 5% position → impact ≈ 0.1% — well under 2%."""
    state = _state()
    hard_patch = hard_rules_node(state)
    state["risk_partial"] = hard_patch.get("risk_partial")

    var_patch = var_check_node(state)
    assert "risk" not in var_patch
    assert "risk_partial" in var_patch
    rp = var_patch["risk_partial"]
    rules = {c["rule"] for c in rp["hard_checks"]}
    assert "R3_VAR95" in rules
    assert rp["var_result"]["passed"] is True


def test_var_check_short_history_rejects():
    md = make_market_data(n_history_days=20)  # < min_samples
    state = _state(market_data=md)
    hard_patch = hard_rules_node(state)
    state["risk_partial"] = hard_patch.get("risk_partial")

    var_patch = var_check_node(state)
    risk = RiskDecision.model_validate(var_patch["risk"])
    assert risk.decision == "reject"
    assert "VaR" in (risk.reject_reason or "")


def test_var_check_skip_when_already_rejected():
    state = _state(
        risk=RiskDecision(
            decision="reject",
            adjusted_position_pct=0.0,
            hard_checks=[
                {"rule": "R1_POSITION_LIMIT", "passed": False, "actual": 12.0,
                 "threshold": 10.0, "detail": "x"}  # type: ignore[arg-type]
            ],
            var_result={"var_95": 0.0, "portfolio_impact": 0.0,  # type: ignore[arg-type]
                        "passed": False, "sample_size": 0, "method": "historical_simulation",
                        "fallback_reason": "skipped"},
            soft_judgment={"flags": ["NONE"], "adjustment": "reject",  # type: ignore[arg-type]
                           "adjusted_position_pct": 0, "reasoning": "x"},
            reject_reason="hard rule failed",
        ).model_dump()
    )
    patch = var_check_node(state)
    assert patch == {}


# ---------- soft judge -----------------------------------------------------


def _mk_soft_response(adjustment: str = "keep", flags: list[str] | None = None, pct: float = 7.0):
    def _fn(messages, **kwargs):
        return SoftJudgment(
            flags=(flags or ["NONE"]),  # type: ignore[arg-type]
            adjustment=adjustment,  # type: ignore[arg-type]
            adjusted_position_pct=pct,
            reasoning=f"LLM said {adjustment}",
        )

    return _fn


def _full_state_with_partial(market_kwargs: dict | None = None, **overrides):
    state = _state(market_data=make_market_data(**(market_kwargs or {})), **overrides)
    hard_patch = hard_rules_node(state)
    state["risk_partial"] = hard_patch.get("risk_partial")
    var_patch = var_check_node(state)
    if "risk_partial" in var_patch:
        state["risk_partial"] = var_patch["risk_partial"]
    return state


def test_soft_judge_keep_path(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.risk.soft_judge.generate_ark_response",
        _mk_soft_response("keep", ["NONE"], 7.0),
    )
    state = _full_state_with_partial()
    patch = soft_judge_node(state)
    risk = RiskDecision.model_validate(patch["risk"])
    assert risk.decision == "approve"
    assert risk.adjusted_position_pct == 7.0
    assert risk.soft_judgment.adjustment == "keep"


def test_soft_judge_reduce_clamps_to_50pct(monkeypatch):
    """Even if LLM gave a wonky number, the 50% rule is enforced."""
    monkeypatch.setattr(
        "fishtrade.agents.risk.soft_judge.generate_ark_response",
        _mk_soft_response("reduce", ["MARKET_VOLATILE"], 9.9),
    )
    state = _full_state_with_partial()
    patch = soft_judge_node(state)
    risk = RiskDecision.model_validate(patch["risk"])
    # proposed_position_pct=7 in fixture → clamp to 3.5
    assert risk.adjusted_position_pct == 3.5
    assert risk.soft_judgment.adjustment == "reduce"


def test_soft_judge_falls_back_to_rules(monkeypatch):
    """LLM raises + market_signals show high VIX → rule fallback reduces 50%."""
    def _raise(*_a, **_k):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(
        "fishtrade.agents.risk.soft_judge.generate_ark_response", _raise
    )
    state = _full_state_with_partial(market_kwargs={"vix_avg": 35.0})
    patch = soft_judge_node(state)
    assert "RISK_SOFT_LLM_FALLBACK" in patch.get("warnings", [])
    risk = RiskDecision.model_validate(patch["risk"])
    assert risk.adjusted_position_pct == 3.5
    assert "MARKET_VOLATILE" in risk.soft_judgment.flags


def test_soft_judge_skips_when_already_rejected(monkeypatch):
    monkeypatch.setattr(
        "fishtrade.agents.risk.soft_judge.generate_ark_response",
        _mk_soft_response("reject"),
    )
    state = _state(
        risk=RiskDecision(
            decision="reject",
            adjusted_position_pct=0.0,
            hard_checks=[{"rule": "R3_VAR95", "passed": False, "actual": 5.0,  # type: ignore[arg-type]
                          "threshold": 2.0, "detail": "x"}],
            var_result={"var_95": 0.0, "portfolio_impact": 0.0, "passed": False,  # type: ignore[arg-type]
                        "sample_size": 0, "method": "historical_simulation",
                        "fallback_reason": "fail"},
            soft_judgment={"flags": ["NONE"], "adjustment": "reject",  # type: ignore[arg-type]
                           "adjusted_position_pct": 0, "reasoning": "x"},
            reject_reason="VaR fail",
        ).model_dump()
    )
    patch = soft_judge_node(state)
    assert patch == {}
