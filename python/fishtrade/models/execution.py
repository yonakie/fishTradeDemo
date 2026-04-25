"""Execution-layer Pydantic schemas (orders + fills + status)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Side = Literal["buy", "sell"]
Mode = Literal["dryrun", "paper", "backtest"]
Status = Literal["generated", "submitted", "filled", "partial", "failed", "skipped"]


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=10)
    side: Side
    qty: int = Field(..., gt=0)
    order_type: Literal["limit"] = "limit"
    limit_price: float = Field(..., gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    tif: Literal["day"] = "day"

    @model_validator(mode="after")
    def _stop_only_on_buy(self) -> "Order":
        if self.side == "sell" and self.stop_price is not None:
            raise ValueError("sell 单不允许设置 stop_price")
        return self


class FillInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    avg_price: float = Field(..., gt=0)
    filled_qty: int = Field(..., gt=0)
    fill_time: str = Field(..., min_length=1)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode
    order: Order | None
    status: Status
    fill_info: FillInfo | None = None
    error: str | None = None
    broker_order_id: str | None = None

    @model_validator(mode="after")
    def _failed_requires_error(self) -> "ExecutionResult":
        if self.status == "failed" and not self.error:
            raise ValueError("status=failed 时必须填写 error")
        if self.status in ("filled", "partial") and self.fill_info is None:
            raise ValueError(f"status={self.status} 时必须填写 fill_info")
        if self.status == "skipped" and self.order is not None:
            raise ValueError("status=skipped 时 order 必须为 None")
        return self
