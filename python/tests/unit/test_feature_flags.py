from __future__ import annotations

from datetime import date

import pandas as pd

from fishtrade.tools.feature_flags import (
    has_field,
    is_financial_data_sufficient,
    is_history_sufficient,
    is_in_earnings_window,
)


def test_is_financial_data_sufficient_with_dataframe():
    df = pd.DataFrame({"Q1": [1], "Q2": [2], "Q3": [3], "Q4": [4]})
    assert is_financial_data_sufficient(df) is True


def test_is_financial_data_sufficient_with_payload():
    payload = {"columns": ["Q1", "Q2", "Q3", "Q4"], "index": ["x"], "data": [[1, 2, 3, 4]]}
    assert is_financial_data_sufficient(payload) is True


def test_is_financial_data_sufficient_too_few():
    df = pd.DataFrame({"Q1": [1], "Q2": [2]})
    assert is_financial_data_sufficient(df) is False


def test_is_financial_data_sufficient_none():
    assert is_financial_data_sufficient(None) is False


def test_is_history_sufficient_dataframe():
    df = pd.DataFrame({"Close": list(range(80))})
    assert is_history_sufficient(df, min_days=60) is True
    assert is_history_sufficient(df, min_days=100) is False


def test_is_in_earnings_window_inside():
    dates = {"index": ["2026-04-26"], "columns": [], "data": []}
    assert is_in_earnings_window(dates, date(2026, 4, 25), window_days=3) is True


def test_is_in_earnings_window_outside():
    dates = {"index": ["2026-01-01"], "columns": [], "data": []}
    assert is_in_earnings_window(dates, date(2026, 4, 25), window_days=3) is False


def test_is_in_earnings_window_missing():
    assert is_in_earnings_window(None, date(2026, 4, 25)) is False


def test_has_field_true_for_present():
    assert has_field({"foo": 1}, "foo") is True


def test_has_field_false_for_none_or_missing():
    assert has_field({"foo": None}, "foo") is False
    assert has_field({}, "foo") is False
    assert has_field(None, "foo") is False
