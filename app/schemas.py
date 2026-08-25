from __future__ import annotations

from pydantic import BaseModel, Field


class FactorRequest(BaseModel):
    factor: str = "momentum"
    lookback: int = Field(default=20, ge=5, le=252)


class BacktestRequest(BaseModel):
    factor: str = "momentum"
    lookback: int = Field(default=20, ge=5, le=252)
    top_k: int = Field(default=2, ge=1, le=20)
    rebalance_days: int = Field(default=5, ge=1, le=63)
    transaction_cost_bps: float = Field(default=5.0, ge=0, le=100)


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    source: str = Field(default="manual", max_length=200)


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    factor: str | None = None
    lookback: int | None = Field(default=None, ge=5, le=252)
    top_k: int | None = Field(default=None, ge=1, le=20)
    rebalance_days: int | None = Field(default=None, ge=1, le=63)
    transaction_cost_bps: float | None = Field(default=None, ge=0, le=100)

