from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FactorName = Literal["momentum", "reversal", "volatility", "sma_ratio", "volume_zscore"]


class DateRangeModel(BaseModel):
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "DateRangeModel":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        return self


class MarketSnapshotArguments(DateRangeModel):
    symbol: str = Field(min_length=1, max_length=32)
    limit: int = Field(default=120, ge=1, le=5000)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized or normalized.lower() in {"nan", "none", "null"}:
            raise ValueError("symbol cannot be empty")
        return normalized


class ResearchRequest(DateRangeModel):
    symbols: list[str] | None = Field(default=None, max_length=5000)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = sorted({symbol.strip().upper() for symbol in value if symbol.strip()})
        if not normalized:
            raise ValueError("symbols cannot be empty when provided")
        if any(len(symbol) > 32 or symbol.lower() in {"nan", "none", "null"} for symbol in normalized):
            raise ValueError("symbols contains an invalid symbol")
        return normalized


class FactorRequest(ResearchRequest):
    factor: FactorName = "momentum"
    lookback: int = Field(default=20, ge=5, le=252)


class BacktestRequest(FactorRequest):
    top_k: int = Field(default=2, ge=1, le=5000)
    rebalance_days: int = Field(default=5, ge=1, le=63)
    transaction_cost_bps: float = Field(default=5.0, ge=0, le=100)


class AgentRequest(BacktestRequest):
    query: str = Field(min_length=1, max_length=1000)
    factor: FactorName | None = None
    lookback: int | None = Field(default=None, ge=5, le=252)
    top_k: int | None = Field(default=None, ge=1, le=5000)
    rebalance_days: int | None = Field(default=None, ge=1, le=63)
    transaction_cost_bps: float | None = Field(default=None, ge=0, le=100)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = None
    request_id: str
