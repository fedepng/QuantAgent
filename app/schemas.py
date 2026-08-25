from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator
from typing import Literal


FactorName = Literal["momentum", "reversal", "volatility", "sma_ratio", "volume_zscore"]


class MarketSnapshotArguments(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    limit: int = Field(default=120, ge=1, le=5000)
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class FactorRequest(BaseModel):
    factor: FactorName = "momentum"
    lookback: int = Field(default=20, ge=5, le=252)
    symbols: list[str] | None = Field(default=None, max_length=5000)
    start_date: date | None = None
    end_date: date | None = None


class BacktestRequest(BaseModel):
    factor: FactorName = "momentum"
    lookback: int = Field(default=20, ge=5, le=252)
    top_k: int = Field(default=2, ge=1, le=5000)
    rebalance_days: int = Field(default=5, ge=1, le=63)
    transaction_cost_bps: float = Field(default=5.0, ge=0, le=100)
    symbols: list[str] | None = Field(default=None, max_length=5000)
    start_date: date | None = None
    end_date: date | None = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    factor: FactorName | None = None
    lookback: int | None = Field(default=None, ge=5, le=252)
    top_k: int | None = Field(default=None, ge=1, le=5000)
    rebalance_days: int | None = Field(default=None, ge=1, le=63)
    transaction_cost_bps: float | None = Field(default=None, ge=0, le=100)
    symbols: list[str] | None = Field(default=None, max_length=5000)
    start_date: date | None = None
    end_date: date | None = None
