"""Pydantic-Settings backed runtime configuration.

Loads `.env` (working dir) on import. Field aliases mirror the variable
names declared in `.env.example` so that operators only have to learn one
naming convention.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Ark (LLM) ===
    ark_api_key: str = Field(default="", validation_alias="ARK_API_KEY")
    ark_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        validation_alias="ARK_BASE_URL",
    )
    ark_model_id: str = Field(default="", validation_alias="ARK_MODEL_ID")
    ark_model_id_research: str | None = Field(
        default=None, validation_alias="ARK_MODEL_ID_RESEARCH"
    )
    ark_model_id_debate: str | None = Field(
        default=None, validation_alias="ARK_MODEL_ID_DEBATE"
    )
    ark_model_id_judge: str | None = Field(
        default=None, validation_alias="ARK_MODEL_ID_JUDGE"
    )
    ark_timeout_seconds: int = Field(default=60, validation_alias="ARK_TIMEOUT_SECONDS")
    ark_max_retries: int = Field(default=2, validation_alias="ARK_MAX_RETRIES")

    # === Alpaca paper trading ===
    alpaca_api_key: str = Field(default="", validation_alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", validation_alias="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        validation_alias="ALPACA_BASE_URL",
    )

    # === Runtime paths ===
    data_dir: Path = Field(default=Path("./data"), validation_alias="FISHTRADE_DATA_DIR")
    log_dir: Path = Field(default=Path("./logs"), validation_alias="FISHTRADE_LOG_DIR")
    report_dir: Path = Field(default=Path("./reports"), validation_alias="FISHTRADE_REPORT_DIR")
    log_level: str = Field(default="INFO", validation_alias="FISHTRADE_LOG_LEVEL")

    # === yfinance cache ===
    yf_cache_ttl_seconds: int = Field(default=86_400, validation_alias="YF_CACHE_TTL_SECONDS")
    yf_rate_limit_backoff_base: int = Field(
        default=2, validation_alias="YF_RATE_LIMIT_BACKOFF_BASE"
    )

    # === Risk thresholds ===
    risk_max_position_pct: float = Field(default=10.0, validation_alias="RISK_MAX_POSITION_PCT")
    risk_max_drawdown_pct: float = Field(default=8.0, validation_alias="RISK_MAX_DRAWDOWN_PCT")
    risk_var95_portfolio_limit_pct: float = Field(
        default=2.0, validation_alias="RISK_VAR95_PORTFOLIO_LIMIT_PCT"
    )
    risk_stoploss_pct: float = Field(default=5.0, validation_alias="RISK_STOPLOSS_PCT")

    # ----- helpers -----------------------------------------------------------

    def has_ark_key(self) -> bool:
        return bool(self.ark_api_key) and self.ark_api_key != "your-ark-api-key-here"

    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key) and bool(self.alpaca_secret_key)

    def resolve_model_id(self, role: str) -> str:
        """Map role -> Ark model id, falling back to the default endpoint."""
        mapping = {
            "research": self.ark_model_id_research,
            "debate": self.ark_model_id_debate,
            "judge": self.ark_model_id_judge,
        }
        return mapping.get(role) or self.ark_model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Memoized accessor — call this from CLI / agents instead of constructing directly."""
    return Settings()


settings = get_settings()
