"""Risk-layer Pydantic schemas (hard rules + VaR + soft judgment)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HardRule = Literal["R1_POSITION_LIMIT", "R2_MAX_DRAWDOWN", "R3_VAR95", "R4_STOPLOSS"]
SoftFlag = Literal["MARKET_VOLATILE", "CORRELATION_HIGH", "LIQUIDITY_LOW", "NONE"]
Adjustment = Literal["keep", "reduce", "reject"]
Decision = Literal["approve", "reject"]


class HardCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: HardRule
    passed: bool
    actual: float | None = None
    threshold: float | None = None
    detail: str = Field(..., min_length=1)


class VarResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    var_95: float = Field(..., ge=0.0, description="Single-name daily VaR(95), e.g. 0.034 = 3.4%")
    portfolio_impact: float = Field(
        ..., ge=0.0, description="Position-weighted portfolio impact, e.g. 0.0034 = 0.34%"
    )
    passed: bool
    sample_size: int = Field(..., ge=0)
    method: Literal["historical_simulation"] = "historical_simulation"
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def _zero_sample_implies_fail(self) -> "VarResult":
        if self.sample_size == 0 and self.passed:
            raise ValueError("sample_size=0 时 passed 必须为 False")
        return self


class SoftJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: list[SoftFlag] = Field(..., min_length=1)
    adjustment: Adjustment
    adjusted_position_pct: float = Field(..., ge=0.0, le=10.0)
    reasoning: str = Field(..., min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _reject_implies_zero_position(self) -> "SoftJudgment":
        if self.adjustment == "reject" and self.adjusted_position_pct != 0:
            raise ValueError("adjustment=reject 时 adjusted_position_pct 必须为 0")
        return self


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    adjusted_position_pct: float = Field(..., ge=0.0, le=10.0)
    hard_checks: list[HardCheckResult] = Field(..., min_length=1)
    var_result: VarResult
    soft_judgment: SoftJudgment
    reject_reason: str | None = None

    @model_validator(mode="after")
    def _reject_requires_reason(self) -> "RiskDecision":
        if self.decision == "reject" and not self.reject_reason:
            raise ValueError("decision=reject 时必须填写 reject_reason")
        if self.decision == "reject" and self.adjusted_position_pct != 0:
            raise ValueError("decision=reject 时 adjusted_position_pct 必须为 0")
        return self
