"""H10 — paper mode with simulated partial fill.

We bypass the real Alpaca SDK by stubbing ``_try_submit_alpaca`` and
``_poll_for_fill`` inside the paper module. The poll-fn returns a
``partial`` status with a partial fill_info; we assert the
``ExecutionResult`` round-trips through Pydantic with status=partial.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fishtrade.agents.execution import paper as paper_mod
from fishtrade.models.execution import ExecutionResult, FillInfo
from tests.integration._fixtures import make_market_data, make_portfolio


def test_paper_node_records_partial_fill(monkeypatch):
    fill = FillInfo(
        avg_price=200.0,
        filled_qty=5,
        fill_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    monkeypatch.setattr(
        paper_mod,
        "_try_submit_alpaca",
        lambda order: ("client-mock", "BROKER-ORDER-1"),
    )
    monkeypatch.setattr(
        paper_mod, "_poll_for_fill", lambda client, oid: ("partial", fill)
    )

    state = {
        "input": {
            "ticker": "AAPL",
            "capital": 100_000.0,
            "mode": "paper",
            "debate_rounds": 0,
            "as_of_date": "2026-04-25",
            "language": "zh",
            "hitl": False,
        },
        "market_data": {"info": make_market_data()["info"]},
        "debate": {"final_verdict": "BUY", "proposed_position_pct": 7.0},
        "risk": {"decision": "approve", "adjusted_position_pct": 7.0},
        "portfolio_before": make_portfolio().model_dump(),
        "run_id": "h10-paper",
    }

    patch = paper_mod.paper_node(state)
    res = ExecutionResult.model_validate(patch["execution"])
    assert res.status == "partial"
    assert res.fill_info is not None
    assert res.fill_info.filled_qty == 5
    assert res.broker_order_id == "BROKER-ORDER-1"
