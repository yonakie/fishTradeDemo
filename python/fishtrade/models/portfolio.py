"""Portfolio-layer Pydantic schemas (positions + NAV history)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=10)
    qty: int = Field(..., gt=0)
    avg_cost: float = Field(..., gt=0)
    sector: str | None = None


class NavSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(..., min_length=1)
    nav: float = Field(..., ge=0.0)


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cash: float = Field(..., ge=0.0)
    positions: list[Position] = Field(default_factory=list)
    nav: float = Field(..., ge=0.0)
    nav_history: list[NavSnapshot] = Field(default_factory=list)
    max_drawdown_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _no_duplicate_symbols(self) -> "PortfolioSnapshot":
        seen: set[str] = set()
        for p in self.positions:
            if p.symbol in seen:
                raise ValueError(f"positions 中存在重复 symbol: {p.symbol}")
            seen.add(p.symbol)
        return self
