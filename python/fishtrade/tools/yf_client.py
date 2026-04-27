"""yfinance wrapper with disk cache, exponential back-off and field-level degradation.

The downstream agents only call this module — they never import yfinance
directly, which keeps rate-limit handling and cache logic centralised.
"""

from __future__ import annotations

import math
import time
from datetime import date as date_cls, datetime as _dt
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import yfinance as yf

from ..config.settings import settings
from ..observability.logger import get_logger
from .yf_cache import YFCache, make_cache_key

logger = get_logger(__name__)

T = TypeVar("T")


class InvalidTickerError(ValueError):
    """yfinance returned no quote / unknown symbol — pipeline must terminate."""


class YFRateLimitError(RuntimeError):
    """Back-off attempts exhausted — caller should record a degradation."""


def _today_iso() -> str:
    return date_cls.today().isoformat()


def _to_primitive(value: Any) -> Any:
    """Coerce a single cell to a JSON/msgpack-safe primitive.

    Critical for langgraph's checkpoint serializer, which refuses to
    deserialize ``pandas.Timestamp.fromisoformat`` (and similar non-stdlib
    constructors) and emits a noisy warning per blocked call.
    """
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, _dt, date_cls)):
        return str(value)
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    # Last resort — stringify so msgpack does not see an opaque object.
    return str(value)


def _df_to_payload(df: pd.DataFrame | None) -> dict | None:
    """Serialise a DataFrame so it survives diskcache + langgraph checkpoints.

    Cell values are coerced to JSON-safe primitives so that
    ``pandas.Timestamp`` (e.g. yfinance options chain ``lastTradeDate``)
    never reaches msgpack — otherwise langgraph emits one
    ``Blocked deserialization`` warning per cell on checkpoint read.
    """
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        return None
    if df.empty:
        return {"columns": [str(c) for c in df.columns], "index": [], "data": []}
    safe = df.copy()
    safe.index = [str(x) for x in safe.index]
    safe.columns = [str(c) for c in safe.columns]
    rows = safe.where(pd.notna(safe), None).values.tolist()
    sanitized: list[list[Any]] = [[_to_primitive(v) for v in row] for row in rows]
    return {
        "columns": list(safe.columns),
        "index": list(safe.index),
        "data": sanitized,
    }


def payload_to_df(payload: dict | None) -> pd.DataFrame:
    """Inverse of :func:`_df_to_payload`. Empty payload → empty DataFrame."""
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(
        payload.get("data", []),
        columns=payload.get("columns", []),
        index=payload.get("index", []) or None,
    )
    return df


class YFinanceClient:
    """Cache-aware facade over :mod:`yfinance` with field-level degradation."""

    def __init__(
        self,
        cache: YFCache | None = None,
        backoff_base: int | None = None,
        max_attempts: int = 3,
        request_timeout: float = 15.0,
    ) -> None:
        self.cache = cache or YFCache()
        self.backoff_base = (
            backoff_base
            if backoff_base is not None
            else settings.yf_rate_limit_backoff_base
        )
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout

    # ---------- internal helpers ----------------------------------------

    def _retry(self, fn: Callable[[], T], desc: str) -> T:
        """Call ``fn`` with exponential back-off on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # yfinance/requests raise a wide variety
                last_exc = exc
                wait_s = self.backoff_base ** attempt
                logger.warning(
                    "yfinance_call_failed",
                    desc=desc,
                    attempt=attempt,
                    wait_s=wait_s,
                    error=str(exc),
                )
                time.sleep(wait_s)
        raise YFRateLimitError(f"{desc} 退避 {self.max_attempts} 次仍失败: {last_exc}")

    def _cached(
        self,
        endpoint: str,
        ticker: str,
        as_of: str,
        producer: Callable[[], Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        key = make_cache_key(ticker, endpoint, as_of, **(extra or {}))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        value = producer()
        if value is not None:
            self.cache.set(key, value)
        return value

    # ---------- public endpoints ---------------------------------------

    def get_info(self, ticker: str, as_of: str | None = None) -> dict:
        as_of = as_of or _today_iso()

        def producer() -> dict:
            return self._retry(lambda: dict(yf.Ticker(ticker).info or {}), f"info::{ticker}")

        info = self._cached("info", ticker, as_of, producer)
        if not info or info.get("regularMarketPrice") is None or not info.get("quoteType"):
            raise InvalidTickerError(f"yfinance 无法识别 ticker: {ticker}")
        return info

    def get_history(
        self,
        ticker: str,
        period: str = "1y",
        as_of: str | None = None,
    ) -> pd.DataFrame:
        as_of = as_of or _today_iso()

        def producer() -> dict | None:
            df = self._retry(
                lambda: yf.Ticker(ticker).history(period=period, auto_adjust=False),
                f"history::{ticker}",
            )
            return _df_to_payload(df)

        payload = self._cached(
            "hist", ticker, as_of, producer, extra={"period": period}
        )
        return payload_to_df(payload)

    def get_financials(self, ticker: str, as_of: str | None = None) -> dict | None:
        as_of = as_of or _today_iso()

        def producer() -> dict | None:
            try:
                df = yf.Ticker(ticker).financials
            except Exception as exc:
                logger.warning("yf_financials_failed", ticker=ticker, error=str(exc))
                return None
            return _df_to_payload(df)

        return self._cached("financials", ticker, as_of, producer)

    def get_cashflow(self, ticker: str, as_of: str | None = None) -> dict | None:
        as_of = as_of or _today_iso()

        def producer() -> dict | None:
            try:
                df = yf.Ticker(ticker).cashflow
            except Exception as exc:
                logger.warning("yf_cashflow_failed", ticker=ticker, error=str(exc))
                return None
            return _df_to_payload(df)

        return self._cached("cashflow", ticker, as_of, producer)

    def get_balance_sheet(self, ticker: str, as_of: str | None = None) -> dict | None:
        as_of = as_of or _today_iso()

        def producer() -> dict | None:
            try:
                df = yf.Ticker(ticker).balance_sheet
            except Exception as exc:
                logger.warning("yf_balance_sheet_failed", ticker=ticker, error=str(exc))
                return None
            return _df_to_payload(df)

        return self._cached("balance_sheet", ticker, as_of, producer)

    def get_option_chain_safe(self, ticker: str) -> dict | None:
        """Option chain may be empty for many tickers — return None gracefully."""
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return None
            chain = t.option_chain(expirations[0])
            return {
                "calls": _df_to_payload(getattr(chain, "calls", None)),
                "puts": _df_to_payload(getattr(chain, "puts", None)),
                "expiration": expirations[0],
            }
        except (IndexError, AttributeError, KeyError, ValueError) as exc:
            logger.info("yf_option_chain_empty", ticker=ticker, reason=str(exc))
            return None
        except Exception as exc:  # network / parser
            logger.warning("yf_option_chain_failed", ticker=ticker, error=str(exc))
            return None

    def get_institutional_holders_safe(self, ticker: str) -> dict | None:
        try:
            df = yf.Ticker(ticker).institutional_holders
            return _df_to_payload(df)
        except Exception as exc:
            logger.warning("yf_inst_holders_failed", ticker=ticker, error=str(exc))
            return None

    def get_insider_transactions_safe(self, ticker: str) -> dict | None:
        try:
            df = yf.Ticker(ticker).insider_transactions
            return _df_to_payload(df)
        except Exception as exc:
            logger.warning("yf_insider_failed", ticker=ticker, error=str(exc))
            return None

    def get_upgrades_downgrades_safe(self, ticker: str) -> dict | None:
        try:
            df = yf.Ticker(ticker).upgrades_downgrades
            return _df_to_payload(df)
        except Exception as exc:
            logger.warning("yf_upgrades_failed", ticker=ticker, error=str(exc))
            return None

    def get_earnings_dates_safe(self, ticker: str) -> dict | None:
        try:
            df = yf.Ticker(ticker).earnings_dates
            return _df_to_payload(df)
        except Exception as exc:
            logger.warning("yf_earnings_dates_failed", ticker=ticker, error=str(exc))
            return None
