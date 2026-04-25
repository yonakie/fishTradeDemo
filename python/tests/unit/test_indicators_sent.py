"""Unit tests for sentiment indicators."""

from __future__ import annotations

import pandas as pd

from fishtrade.tools.indicators_sent import (
    compute_52week_position,
    compute_all_sentimental,
    compute_analyst_rating,
    compute_buyback,
    compute_dividend,
    compute_earnings_beat,
    compute_insider_tx,
    compute_institutional_hold,
    compute_options_pcr,
    compute_retail_social,
    compute_short_float,
)


def test_compute_short_float_low_is_positive():
    out = compute_short_float({"shortPercentOfFloat": 0.02, "shortRatio": 1.5})
    assert out.score == 1


def test_compute_short_float_high_is_negative():
    out = compute_short_float({"shortPercentOfFloat": 0.30, "shortRatio": 12})
    assert out.score == -1


def test_compute_short_float_missing_degrades():
    assert compute_short_float({}).is_degraded is True


def test_compute_institutional_hold_branches():
    assert compute_institutional_hold({"heldPercentInstitutions": 0.85}, None).score == 1
    assert compute_institutional_hold({"heldPercentInstitutions": 0.50}, None).score == 0
    assert compute_institutional_hold({"heldPercentInstitutions": 0.10}, None).score == -1
    assert compute_institutional_hold({}, None).is_degraded is True


def test_compute_analyst_rating_buy():
    assert compute_analyst_rating({"recommendationMean": 1.6}, None).score == 1


def test_compute_analyst_rating_missing_degrades():
    assert compute_analyst_rating({}, None).is_degraded is True


def test_compute_options_pcr_neutral_when_chain_missing():
    out = compute_options_pcr(None)
    assert out.is_degraded is True


def test_compute_options_pcr_positive_when_calls_dominate():
    chain = {
        "calls": pd.DataFrame({"volume": [100, 200, 300]}),
        "puts": pd.DataFrame({"volume": [50, 50, 50]}),
    }
    out = compute_options_pcr(chain)
    assert out.score == 1


def test_compute_options_pcr_negative_when_puts_dominate():
    chain = {
        "calls": pd.DataFrame({"volume": [100, 100]}),
        "puts": pd.DataFrame({"volume": [200, 200]}),
    }
    out = compute_options_pcr(chain)
    assert out.score == -1


def test_compute_buyback_positive_repurchase():
    out = compute_buyback({"netSharesRepurchased": 10_000_000})
    assert out.score == 1


def test_compute_buyback_missing_degrades():
    assert compute_buyback({}).is_degraded is True


def test_compute_dividend_growth_stock_neutral():
    out = compute_dividend({})
    assert out.score == 0


def test_compute_dividend_healthy_yield():
    out = compute_dividend({"dividendYield": 0.04, "payoutRatio": 0.4})
    assert out.score == 1


def test_compute_retail_social_always_degraded():
    out = compute_retail_social()
    assert out.is_degraded is True
    assert out.score == 0


def test_compute_52week_position_high():
    info = {
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 100.0,
        "regularMarketPrice": 180.0,
    }
    assert compute_52week_position(info).score == 1


def test_compute_52week_position_low():
    info = {
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 100.0,
        "regularMarketPrice": 110.0,
    }
    assert compute_52week_position(info).score == -1


def test_compute_52week_position_missing_degrades():
    assert compute_52week_position({}).is_degraded is True


def test_compute_earnings_beat_positive():
    df = pd.DataFrame({"Surprise(%)": [12, 8, 6, 9]})
    out = compute_earnings_beat(df)
    assert out.score == 1


def test_compute_earnings_beat_missing_degrades():
    assert compute_earnings_beat(None).is_degraded is True


def test_compute_insider_tx_missing_degrades():
    assert compute_insider_tx(None).is_degraded is True


def test_compute_all_sentimental_returns_exactly_ten():
    md = {
        "info": {
            "shortPercentOfFloat": 0.03,
            "shortRatio": 2,
            "heldPercentInstitutions": 0.65,
            "recommendationMean": 1.8,
            "netSharesRepurchased": 5_000_000,
            "dividendYield": 0.04,
            "payoutRatio": 0.5,
            "fiftyTwoWeekHigh": 200.0,
            "fiftyTwoWeekLow": 100.0,
            "regularMarketPrice": 180.0,
        }
    }
    scores = compute_all_sentimental(md)
    assert len(scores) == 10
    total = sum(s.score for s in scores)
    assert -10 <= total <= 10
    # RETAIL_SOCIAL must always be degraded
    rs = next(s for s in scores if s.name == "RETAIL_SOCIAL")
    assert rs.is_degraded is True


def test_compute_all_sentimental_empty_market_data():
    scores = compute_all_sentimental({})
    assert len(scores) == 10
