from __future__ import annotations

import pytest

from fishtrade.tools.industry_classifier import classify_industry


@pytest.mark.parametrize(
    "sector,expected",
    [
        ("Technology", "growth"),
        ("Communication Services", "growth"),
        ("Healthcare", "growth"),
        ("Financial Services", "financial"),
        ("Energy", "energy"),
        ("Consumer Defensive", "consumer"),
        ("Consumer Cyclical", "consumer"),
        ("Industrials", "value"),
        ("Utilities", "value"),
        ("Real Estate", "value"),
        ("Basic Materials", "value"),
    ],
)
def test_known_sectors_map_correctly(sector, expected):
    assert classify_industry({"sector": sector}) == expected


def test_unknown_sector_falls_back_to_value():
    assert classify_industry({"sector": "Aerospace & Defense"}) == "value"


def test_missing_sector_falls_back_to_value():
    assert classify_industry({}) == "value"


def test_none_input_falls_back_to_value():
    assert classify_industry(None) == "value"
