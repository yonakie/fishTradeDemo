"""Map yfinance ``info['sector']`` to one of five threshold buckets.

Buckets are referenced by ``score_*`` functions in ``indicators_fund.py`` to
pick the correct row of the analysismetrics threshold tables.
"""

from __future__ import annotations

from typing import Literal

IndustryClass = Literal["value", "growth", "financial", "consumer", "energy"]


SECTOR_TO_CLASS: dict[str, IndustryClass] = {
    "Technology": "growth",
    "Communication Services": "growth",
    "Healthcare": "growth",
    "Financial Services": "financial",
    "Financial": "financial",
    "Energy": "energy",
    "Consumer Defensive": "consumer",
    "Consumer Cyclical": "consumer",
    "Industrials": "value",
    "Utilities": "value",
    "Real Estate": "value",
    "Basic Materials": "value",
}


def classify_industry(info: dict | None) -> IndustryClass:
    """Return the industry bucket for an info dict.

    Falls back to ``"value"`` when ``info`` is missing or sector is unknown —
    the most conservative thresholds.
    """
    if not info:
        return "value"
    sector = info.get("sector") or info.get("Sector")
    if not sector:
        return "value"
    return SECTOR_TO_CLASS.get(str(sector), "value")
