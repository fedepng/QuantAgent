from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from app.db import Database
from app.quant.backtest import BacktestEngine, BacktestParameters
from app.quant.factors import FACTOR_NAMES, FactorEngine
from app.quant.market import MarketDataService
from app.schemas import BacktestRequest, FactorRequest, MarketSnapshotArguments


class ToolRegistry:
    def __init__(
        self,
        market: MarketDataService,
        factors: FactorEngine,
        backtests: BacktestEngine,
        database: Database,
    ) -> None:
        self.market = market
        self.factors = factors
        self.backtests = backtests
        self.database = database
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "market_snapshot": self.market_snapshot,
            "factor_snapshot": self.factor_snapshot,
            "run_backtest": self.run_backtest,
        }
        self._validators = {
            "market_snapshot": MarketSnapshotArguments,
            "factor_snapshot": FactorRequest,
            "run_backtest": BacktestRequest,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "market_snapshot",
                "description": "查询当前数据集中指定资产的 OHLCV 行情。",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": f"当前数据集中的股票代码，例如：{', '.join(self.market.symbols()[:12])}",
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                        "start_date": {"type": ["string", "null"], "format": "date"},
                        "end_date": {"type": ["string", "null"], "format": "date"},
                    },
                    "required": ["symbol", "limit", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "factor_snapshot",
                "description": "计算全部资产的最新因子值并生成横截面排名。",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string", "enum": sorted(FACTOR_NAMES)},
                        "lookback": {"type": "integer", "minimum": 5, "maximum": 252},
                        "symbols": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "start_date": {"type": ["string", "null"], "format": "date"},
                        "end_date": {"type": ["string", "null"], "format": "date"},
                    },
                    "required": ["factor", "lookback", "symbols", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "run_backtest",
                "description": "运行信号滞后一日、定期调仓并扣除交易成本的横截面 Top-K 因子回测。",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string", "enum": sorted(FACTOR_NAMES)},
                        "lookback": {"type": "integer", "minimum": 5, "maximum": 252},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5000},
                        "rebalance_days": {"type": "integer", "minimum": 1, "maximum": 63},
                        "transaction_cost_bps": {"type": "number", "minimum": 0, "maximum": 100},
                        "symbols": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "start_date": {"type": ["string", "null"], "format": "date"},
                        "end_date": {"type": ["string", "null"], "format": "date"},
                    },
                    "required": ["factor", "lookback", "top_k", "rebalance_days", "transaction_cost_bps", "symbols", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
        ]

    def validate_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._validators[name].model_validate(arguments).model_dump(
            mode="json", exclude_none=True
        )

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = self.validate_call(name, arguments)
        return self._tools[name](**validated)

    def _market_slice(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frame = self.market.frame
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        if symbols:
            normalized = sorted({symbol.strip().upper() for symbol in symbols})
            unknown = sorted(set(normalized) - set(self.market.symbols()))
            if unknown:
                raise ValueError(f"Unknown symbols: {unknown}")
            frame = frame[frame["symbol"].isin(normalized)]
        if start_date:
            frame = frame[frame["date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame[frame["date"] <= pd.Timestamp(end_date)]
        if frame.empty:
            raise ValueError("No market data matches the selected range")
        return frame.reset_index(drop=True)

    def market_snapshot(
        self,
        symbol: str = "ALPHA",
        limit: int = 20,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol.upper(),
            "dataset": self.market.dataset,
            "prices": self.market.prices(symbol, limit, start_date, end_date),
        }

    def factor_snapshot(
        self,
        factor: str = "momentum",
        lookback: int = 20,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        frame = self._market_slice(symbols, start_date, end_date)
        return {
            "factor": factor,
            "lookback": lookback,
            "dataset": self.market.dataset,
            "description": FACTOR_NAMES.get(factor),
            "ranking": self.factors.snapshot(frame, factor, lookback),
        }

    def run_backtest(
        self,
        factor: str = "momentum",
        lookback: int = 20,
        top_k: int = 2,
        rebalance_days: int = 5,
        transaction_cost_bps: float = 5.0,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        frame = self._market_slice(symbols, start_date, end_date)
        parameters = BacktestParameters(
            factor=factor,
            lookback=lookback,
            top_k=top_k,
            rebalance_days=rebalance_days,
            transaction_cost_bps=transaction_cost_bps,
        )
        result = self.backtests.run(frame, parameters)
        actual_symbols = sorted(frame["symbol"].unique().tolist())
        actual_start = frame["date"].min().strftime("%Y-%m-%d")
        actual_end = frame["date"].max().strftime("%Y-%m-%d")
        result["parameters"].update(
            {"symbols": actual_symbols, "start_date": actual_start, "end_date": actual_end}
        )
        dataset = self.market.dataset
        provenance = {
            "dataset_id": dataset.get("id"),
            "dataset_hash": dataset.get("content_hash"),
            "dataset_name": dataset.get("name"),
            "symbols": actual_symbols,
            "start_date": actual_start,
            "end_date": actual_end,
            "code_version": "2.0.0",
        }
        run_id = self.database.save_backtest(
            result["strategy"], result["parameters"], result["metrics"], result["series"], provenance
        )
        result["id"] = run_id
        result["provenance"] = provenance
        return result
