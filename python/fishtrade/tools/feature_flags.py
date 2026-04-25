"""Pure boolean gates used by indicators / risk / agents.

These never call out to the network and never raise on missing data —
callers can rely on a ``False`` answer whenever a field is absent.
"""

from __future__ import annotations

from datetime import date as _date_cls, datetime
from typing import Any

import pandas as pd


def is_financial_data_sufficient(financials: Any) -> bool:
    """True when the financials DataFrame has ≥ 4 reporting periods."""
    if financials is None:
        return False
    if isinstance(financials, dict) and "columns" in financials:
        return len(financials.get("columns", [])) >= 4
    if isinstance(financials, pd.DataFrame):
        return financials.shape[1] >= 4
    return False


def is_history_sufficient(history: Any, min_days: int = 60) -> bool:
    """True when the OHLCV history has at least ``min_days`` rows."""
    if history is None:
        return False
    if isinstance(history, pd.DataFrame):
        return len(history) >= min_days
    if isinstance(history, dict) and "data" in history:
        return len(history.get("data") or []) >= min_days
    return False


def _as_date(x: Any) -> _date_cls | None:
    if x is None:
        return None
    if isinstance(x, _date_cls) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x[:10]).date()
        except ValueError:
            return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    return None


def is_in_earnings_window(
    earnings_dates: Any, today: Any, window_days: int = 3
) -> bool:
    """True when ``today`` is within ``window_days`` of any earnings date."""
    today_d = _as_date(today)
    if today_d is None or earnings_dates is None:
        return False

    if isinstance(earnings_dates, dict) and "index" in earnings_dates:
        candidates = earnings_dates.get("index", [])
    elif isinstance(earnings_dates, pd.DataFrame):
        candidates = list(earnings_dates.index)
    elif isinstance(earnings_dates, (list, tuple)):
        candidates = list(earnings_dates)
    else:
        return False

    for c in candidates:
        d = _as_date(c)
        if d is None:
            continue
        if abs((d - today_d).days) <= window_days:
            return True
    return False


def has_field(info: dict | None, field: str) -> bool:
    """``True`` only when ``info[field]`` is present and not None/NaN."""
    if not info or field not in info:
        return False
    val = info[field]
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return True
