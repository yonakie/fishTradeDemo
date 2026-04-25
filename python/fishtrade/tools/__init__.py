"""Pure data / math utilities (no LLM calls).

Importing modules from here is safe in any context — agents, tests, or
notebooks. None of these functions make network calls beyond the cached
yfinance client in :mod:`fishtrade.tools.yf_client`.
"""

from .feature_flags import (
    has_field,
    is_financial_data_sufficient,
    is_history_sufficient,
    is_in_earnings_window,
)
from .indicators_fund import INDICATOR_REGISTRY, compute_all_fundamental
from .indicators_sent import compute_all_sentimental
from .indicators_tech import compute_all_technical
from .industry_classifier import SECTOR_TO_CLASS, classify_industry
from .var_calculator import compute_var_historical
from .yf_cache import YFCache, make_cache_key
from .yf_client import (
    InvalidTickerError,
    YFinanceClient,
    YFRateLimitError,
    payload_to_df,
)

__all__ = [
    "INDICATOR_REGISTRY",
    "InvalidTickerError",
    "SECTOR_TO_CLASS",
    "YFCache",
    "YFRateLimitError",
    "YFinanceClient",
    "classify_industry",
    "compute_all_fundamental",
    "compute_all_sentimental",
    "compute_all_technical",
    "compute_var_historical",
    "has_field",
    "is_financial_data_sufficient",
    "is_history_sufficient",
    "is_in_earnings_window",
    "make_cache_key",
    "payload_to_df",
]
