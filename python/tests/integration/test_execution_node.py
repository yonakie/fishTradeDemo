"""Isolation tests for execution router + dryrun / paper / backtest + portfolio update."""

from __future__ import annotations

import pandas as pd

from fishtrade.agents.execution.backtest import backtest_node
from fishtrade.agents.execution.dryrun import dryrun_node
from fishtrade.agents.execution.paper import paper_node
from fishtrade.agents.execution.portfolio_update import update_portfolio_node
from fishtrade.agents.execution.router import execution_router, skip_execution_node
from fishtrade.models.execution import ExecutionResult, FillInfo, Order
from fishtrade.models.portfolio import PortfolioSnapshot, Position
from fishtrade.models.risk import RiskDecision
from fishtrade.tools.yf_client import _df_to_payload  # type: ignore

from ._fixtures import (
    make_debate_result,
    make_history,
    make_market_data,
    make_portfolio,
)


def _approve_risk(pct: float = 7.0) -> dict:
    return RiskDecision(
        decision="approve",
        adjusted_position_pct=pct,
        hard_checks=[{"rule": "R1_POSITION_LIMIT", "passed": True, "actual": pct,  # type: ignore[arg-type]
                      "threshold": 10.0, "detail": "ok"}],
        var_result={"var_95": 0.02, "portfolio_impact": 0.001,  # type: ignore[arg-type]
                    "passed": True, "sample_size": 250, "method": "historical_simulation",
                    "fallback_reason": None},
        soft_judgment={"flags": ["NONE"], "adjustment": "keep",  # type: ignore[arg-type]
                       "adjusted_position_pct": pct, "reasoning": "ok"},
    ).model_dump()


def _state(*, mode: str = "dryrun", risk: dict | None = None, debate_kwargs: dict | None = None) -> dict:
    md = make_market_data()
    if isinstance(md.get("history"), pd.DataFrame):
        md = {**md, "history": _df_to_payload(md["history"])}
    return {
        "input": {
            "ticker": "AAPL",
            "capital": 100_000.0,
            "mode": mode,
            "debate_rounds": 1,
            "as_of_date": "2026-04-25",
            "language": "zh",
            "hitl": False,
        },
        "debate": make_debate_result(**(debate_kwargs or {})).model_dump(),
        "risk": risk if risk is not None else _approve_risk(),
        "portfolio_before": make_portfolio().model_dump(),
        "market_data": md,
        "run_id": "test-run-exec",
    }


# ---------- router --------------------------------------------------------


def test_router_skips_when_risk_rejects():
    state = _state(risk=RiskDecision(
        decision="reject",
        adjusted_position_pct=0.0,
        hard_checks=[{"rule": "R1_POSITION_LIMIT", "passed": False, "actual": 12.0,  # type: ignore[arg-type]
                      "threshold": 10.0, "detail": "x"}],
        var_result={"var_95": 0, "portfolio_impact": 0, "passed": False,  # type: ignore[arg-type]
                    "sample_size": 0, "method": "historical_simulation", "fallback_reason": "x"},
        soft_judgment={"flags": ["NONE"], "adjustment": "reject",  # type: ignore[arg-type]
                       "adjusted_position_pct": 0, "reasoning": "x"},
        reject_reason="hard fail",
    ).model_dump())
    assert execution_router(state) == "skip_execution"


def test_router_skips_on_zero_position():
    state = _state(risk=_approve_risk(pct=0))
    # adjusted_position_pct=0 fails the >0 model_validator guard for Soft, so build manually.
    state["risk"]["adjusted_position_pct"] = 0
    assert execution_router(state) == "skip_execution"


def test_router_picks_mode():
    assert execution_router(_state(mode="dryrun")) == "execute_dryrun"
    assert execution_router(_state(mode="paper")) == "execute_paper"
    assert execution_router(_state(mode="backtest")) == "execute_backtest"


def test_skip_execution_node_emits_skipped_result():
    patch = skip_execution_node(_state(risk=_approve_risk(pct=0)))
    res = ExecutionResult.model_validate(patch["execution"])
    assert res.status == "skipped"
    assert res.order is None


# ---------- dryrun --------------------------------------------------------


def test_dryrun_generates_order(tmp_path, monkeypatch):
    # The dryrun module imports `settings` by reference at import time, so
    # mutating env vars + clearing the cache won't affect it. Instead we
    # patch the attribute on the live singleton.
    from fishtrade.agents.execution import dryrun as dryrun_mod

    monkeypatch.setattr(dryrun_mod.settings, "log_dir", tmp_path)

    patch = dryrun_node(_state())
    res = ExecutionResult.model_validate(patch["execution"])
    assert res.status == "generated"
    assert res.order is not None
    assert res.order.qty > 0
    assert (tmp_path / "orders" / "test-run-exec.json").exists()


# ---------- paper (uses mock fill since no Alpaca creds) ------------------


def test_paper_node_mock_fill_when_no_creds():
    patch = paper_node(_state(mode="paper"))
    res = ExecutionResult.model_validate(patch["execution"])
    assert res.status == "filled"
    assert res.fill_info is not None
    assert "PAPER_NODE_MOCK_FILL" in patch.get("warnings", [])


# ---------- backtest ------------------------------------------------------


def test_backtest_simulates_fill_at_close():
    patch = backtest_node(_state(mode="backtest"))
    res = ExecutionResult.model_validate(patch["execution"])
    assert res.status == "filled"
    assert res.fill_info is not None
    assert res.broker_order_id and res.broker_order_id.startswith("BT-")


# ---------- portfolio update ----------------------------------------------


def _exec_filled(symbol: str = "AAPL", qty: int = 10, price: float = 200.0) -> dict:
    return ExecutionResult(
        mode="backtest",
        order=Order(
            symbol=symbol,
            side="buy",
            qty=qty,
            limit_price=price * 1.002,
            stop_price=price * 0.95,
        ),
        status="filled",
        fill_info=FillInfo(avg_price=price, filled_qty=qty, fill_time="2026-04-25T16:00:00Z"),
        broker_order_id="BT-test",
    ).model_dump()


def test_update_portfolio_filled_mutates_cash_and_positions(tmp_path, monkeypatch):
    from fishtrade.config.settings import settings as live_settings

    monkeypatch.setattr(live_settings, "data_dir", tmp_path)

    state = _state(mode="backtest")
    state["execution"] = _exec_filled(qty=10, price=200.0)
    patch = update_portfolio_node(state)
    snap = PortfolioSnapshot.model_validate(patch["portfolio_after"])
    assert any(p.symbol == "AAPL" and p.qty == 10 for p in snap.positions)
    assert snap.cash == 100_000.0 - 10 * 200.0


def test_update_portfolio_failed_keeps_state():
    state = _state(mode="paper")
    state["execution"] = ExecutionResult(
        mode="paper",
        order=Order(symbol="AAPL", side="buy", qty=10, limit_price=200.0, stop_price=190.0),
        status="failed",
        error="alpaca down",
    ).model_dump()
    patch = update_portfolio_node(state)
    before = state["portfolio_before"]
    after = patch["portfolio_after"]
    assert before == after


def test_update_portfolio_dryrun_does_not_persist(tmp_path, monkeypatch):
    from fishtrade.config.settings import settings as live_settings

    monkeypatch.setattr(live_settings, "data_dir", tmp_path)

    state = _state(mode="dryrun")
    state["execution"] = ExecutionResult(
        mode="dryrun",
        order=Order(symbol="AAPL", side="buy", qty=10, limit_price=200.0, stop_price=190.0),
        status="generated",
    ).model_dump()
    patch = update_portfolio_node(state)
    snap = PortfolioSnapshot.model_validate(patch["portfolio_after"])
    # dryrun mirrors before, no positions added.
    assert len(snap.positions) == 0
    assert not (tmp_path / "portfolio.json").exists()


def test_update_portfolio_skipped_keeps_state():
    state = _state(mode="dryrun", risk=_approve_risk(pct=0))
    state["risk"]["adjusted_position_pct"] = 0
    state["execution"] = ExecutionResult(
        mode="dryrun", order=None, status="skipped", error="approved_at_zero_position"
    ).model_dump()
    patch = update_portfolio_node(state)
    before = state["portfolio_before"]
    after = patch["portfolio_after"]
    assert before == after
