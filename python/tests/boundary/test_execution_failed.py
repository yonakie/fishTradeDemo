"""H-extra — execution submission failure in paper mode.

We stub ``_try_submit_alpaca`` to raise (mimicking auth failure or a
network error past the SDK's own retries). The mock-fill path takes
over and the run finishes with a recorded warning, never crashing.
"""

from __future__ import annotations

from fishtrade.agents.execution import paper as paper_mod
from fishtrade.models.execution import ExecutionResult
from tests.integration._fixtures import make_market_data, make_portfolio


def test_paper_node_falls_back_to_mock_when_submit_fails(monkeypatch):
    monkeypatch.setattr(paper_mod, "_try_submit_alpaca", lambda order: None)

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
        "run_id": "h-exec-fail",
    }
    patch = paper_mod.paper_node(state)
    assert "PAPER_NODE_MOCK_FILL" in (patch.get("warnings") or [])
    res = ExecutionResult.model_validate(patch["execution"])
    assert res.status == "filled"  # mock fill is the documented degraded path
    assert res.broker_order_id == "MOCK-PAPER-FILL"
