"""Research-layer Pydantic schemas (fundamental / technical / sentimental)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["BUY", "HOLD", "SELL"]
Facet = Literal["fundamental", "technical", "sentimental"]


class IndicatorScore(BaseModel):
    """A single indicator score, aligned 1:1 with docs/analysismetrics.md."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Indicator id, e.g. 'PE_RATIO', 'MACD', 'SHORT_FLOAT'")
    display_name_zh: str
    display_name_en: str
    raw_value: float | str | None = Field(
        default=None,
        description="Original value; None when missing -> score must be 0 and is_degraded True.",
    )
    score: Literal[-1, 0, 1]
    reasoning: str = Field(..., max_length=400)
    is_degraded: bool = False
    degrade_reason: str | None = None

    @model_validator(mode="after")
    def _degraded_must_be_zero(self) -> "IndicatorScore":
        if self.is_degraded and self.score != 0:
            raise ValueError(
                f"is_degraded=True 时 score 必须为 0，当前 score={self.score}"
            )
        if self.is_degraded and not self.degrade_reason:
            raise ValueError("is_degraded=True 时必须提供 degrade_reason")
        return self


class ResearchReport(BaseModel):
    """One facet of research output (fundamental | technical | sentimental)."""

    model_config = ConfigDict(extra="forbid")

    facet: Facet
    ticker: str
    as_of_date: str

    indicator_scores: list[IndicatorScore] = Field(..., min_length=10, max_length=10)
    total_score: int = Field(..., ge=-10, le=10)
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)

    key_highlights: list[str] = Field(..., min_length=3, max_length=5)
    industry_class: str | None = None

    is_facet_degraded: bool = False
    degrade_summary: str | None = None

    @model_validator(mode="after")
    def _check_total_matches_scores(self) -> "ResearchReport":
        s = sum(i.score for i in self.indicator_scores)
        if s != self.total_score:
            raise ValueError(f"total_score {self.total_score} ≠ Σscores {s}")
        return self

    @model_validator(mode="after")
    def _verdict_matches_score(self) -> "ResearchReport":
        if self.total_score >= 5:
            v_expected: Verdict = "BUY"
        elif self.total_score >= 1:
            v_expected = "HOLD"
        else:
            v_expected = "SELL"
        if self.verdict != v_expected:
            raise ValueError(
                f"verdict {self.verdict} 与 total_score {self.total_score} 不一致"
                f"（期望 {v_expected}）"
            )
        return self

    @model_validator(mode="after")
    def _facet_degraded_implies_low_confidence(self) -> "ResearchReport":
        if self.is_facet_degraded and self.confidence > 0.4:
            raise ValueError(
                f"is_facet_degraded=True 时 confidence 必须 ≤ 0.4，当前 {self.confidence}"
            )
        return self
