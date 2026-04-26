"""H1 — INVALID_TICKER.

The CLI's ``run`` validates the ticker pattern up-front (``^[A-Z.\\-]{1,6}$``)
and exits with code 2 / message ``INVALID_TICKER`` before the graph is
even built. The ``validate_input_node`` provides the same defence at the
graph layer for callers that bypass the CLI.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from typer.testing import CliRunner

from fishtrade.cli import app
from fishtrade.graph import build_graph


runner = CliRunner()


@pytest.mark.parametrize("bad", ["aapl", "TOOLONGTICKER", "1234", "AA*", ""])
def test_cli_run_rejects_invalid_ticker(bad):
    result = runner.invoke(
        app,
        ["run", "--ticker", bad, "--mode", "dryrun"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "INVALID_TICKER" in result.output


def test_validate_input_node_halts_on_invalid_ticker(initial_state, monkeypatch):
    # patch yfinance probe so we don't hit the network if the regex passes.
    from fishtrade.graph import builder as bld_mod

    class _StubClient:
        def get_info(self, ticker, as_of=None):
            from fishtrade.tools.yf_client import InvalidTickerError

            raise InvalidTickerError(f"unknown {ticker}")

    monkeypatch.setattr(bld_mod, "YFinanceClient", lambda: _StubClient())

    state = initial_state(ticker="AAPL", run_id="h1-direct")
    # Force the node to attempt the live check by clearing market_data
    state.pop("market_data", None)

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = graph.invoke(state, config={"configurable": {"thread_id": "h1"}})
    assert final.get("halt_reason") == "INVALID_TICKER"
    assert "execution" not in final or final.get("execution") is None
    assert any("INVALID_TICKER" in w for w in final.get("warnings") or [])
