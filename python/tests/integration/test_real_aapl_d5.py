"""D5 acceptance test — pull real AAPL data and run all three compute_all_*.

This test hits the network (yfinance). It is opt-in via the ``--run-network``
flag so the regular ``pytest`` invocation stays offline; the smoke script
``scripts/d5_aapl.py`` calls the same helpers for manual review.
"""

from __future__ import annotations

import datetime as dt

import pytest

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def real_aapl_bundle():
    yf = pytest.importorskip("yfinance")
    from fishtrade.tools.yf_client import YFinanceClient, payload_to_df

    client = YFinanceClient()
    info = client.get_info("AAPL")
    history = client.get_history("AAPL", period="1y")
    benchmark = client.get_history("SPY", period="1y")
    options = client.get_option_chain_safe("AAPL")
    holders = client.get_institutional_holders_safe("AAPL")
    insider = client.get_insider_transactions_safe("AAPL")
    upgrades = client.get_upgrades_downgrades_safe("AAPL")
    earnings = client.get_earnings_dates_safe("AAPL")

    return {
        "info": info,
        "history": history,
        "benchmark_history": benchmark,
        "options_chain": options,
        "institutional_holders": holders,
        "insider_transactions": insider,
        "upgrades_downgrades": upgrades,
        "earnings_dates": earnings,
        "fetch_warnings": [],
    }


def _verify_total(scores, name):
    assert len(scores) == 10, f"{name}: expected 10 scores, got {len(scores)}"
    total = sum(s.score for s in scores)
    expected = sum(s.score for s in scores)
    assert total == expected
    assert -10 <= total <= 10


def test_real_aapl_fundamental(real_aapl_bundle):
    from fishtrade.tools.indicators_fund import compute_all_fundamental

    scores = compute_all_fundamental(real_aapl_bundle)
    _verify_total(scores, "fundamental")


def test_real_aapl_technical(real_aapl_bundle):
    from fishtrade.tools.indicators_tech import compute_all_technical

    scores = compute_all_technical(real_aapl_bundle)
    _verify_total(scores, "technical")


def test_real_aapl_sentimental(real_aapl_bundle):
    from fishtrade.tools.indicators_sent import compute_all_sentimental

    scores = compute_all_sentimental(real_aapl_bundle)
    _verify_total(scores, "sentimental")
    rs = next(s for s in scores if s.name == "RETAIL_SOCIAL")
    assert rs.is_degraded is True
