"""H-extra — yfinance rate-limit (3 retries exhausted) → field-level degrade.

We patch ``YFinanceClient`` inside the builder so that ``get_history``
raises ``YFRateLimitError``. ``fetch_market_node`` must capture the
failure in ``fetch_warnings`` / ``warnings`` rather than blow up the
pipeline.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph
from fishtrade.tools.yf_client import YFRateLimitError


class _FlakyClient:
    def get_info(self, ticker, as_of=None):
        return {"symbol": ticker, "regularMarketPrice": 100.0, "quoteType": "EQUITY"}

    def get_history(self, ticker, period="1y", as_of=None):
        raise YFRateLimitError(f"history::{ticker} rate limited")

    # Other endpoints mimic the no-data path — return None so fetch_market
    # records them and moves on.
    def get_financials(self, *a, **k): return None
    def get_cashflow(self, *a, **k): return None
    def get_balance_sheet(self, *a, **k): return None
    def get_option_chain_safe(self, *a, **k): return None


def test_yfinance_rate_limit_records_warning(patch_ark, initial_state, monkeypatch):
    patch_ark(judge_verdict="HOLD", judge_pct=0.0)

    from fishtrade.graph import builder as bld_mod

    monkeypatch.setattr(bld_mod, "YFinanceClient", lambda: _FlakyClient())

    state = initial_state(run_id="h-yf-rl")
    # Clear pre-seeded market_data to force fetch_market to call the client.
    state.pop("market_data", None)

    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    final = graph.invoke(state, config={"configurable": {"thread_id": "h-yf-rl"}})
    warnings = final.get("warnings") or []
    assert any("YF_RATE_LIMIT:history" in w for w in warnings)
    # Pipeline did not halt — research / debate / risk still ran.
    assert final.get("debate", {}).get("final_verdict") in {"BUY", "HOLD", "SELL"}
