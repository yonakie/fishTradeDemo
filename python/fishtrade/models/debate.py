"""Debate-layer Pydantic schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["BUY", "HOLD", "SELL"]
Facet = Literal["fundamental", "technical", "sentimental"]
Role = Literal["bull", "bear"]


class DebateTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round: int = Field(..., ge=0, le=3, description="0 = opening, 1..3 = rebuttal rounds")
    role: Role
    argument: str = Field(..., min_length=1, max_length=2000)
    cited_indicators: list[str] = Field(
        ...,
        min_length=1,
        description="Must reference ResearchReport.indicator_scores.name values.",
    )
    conclusion: Verdict
    is_fallback: bool = False

    @model_validator(mode="after")
    def _cited_indicators_non_empty_strings(self) -> "DebateTurn":
        if any(not s or not s.strip() for s in self.cited_indicators):
            raise ValueError("cited_indicators 中存在空字符串")
        return self


class DebateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turns: list[DebateTurn]
    final_verdict: Verdict
    final_rationale: str = Field(..., min_length=1, max_length=2000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    proposed_position_pct: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Proposed position as percent of total capital. HOLD/SELL ⇒ 0.",
    )
    degraded_facets: list[Facet] = Field(default_factory=list)

    @model_validator(mode="after")
    def _hold_sell_must_be_zero_position(self) -> "DebateResult":
        if self.final_verdict in ("HOLD", "SELL") and self.proposed_position_pct != 0:
            raise ValueError(
                f"final_verdict={self.final_verdict} 时 proposed_position_pct 必须为 0，"
                f"当前 {self.proposed_position_pct}"
            )
        return self

    @model_validator(mode="after")
    def _buy_must_have_position(self) -> "DebateResult":
        if self.final_verdict == "BUY" and self.proposed_position_pct <= 0:
            raise ValueError(
                "final_verdict=BUY 时 proposed_position_pct 必须 > 0"
            )
        return self
