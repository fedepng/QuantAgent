from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from app.config import ROOT_DIR
from app.db import Database
from app.errors import QuantAgentError
from app.quant.backtest import BACKTEST_METHODOLOGY_VERSION, BacktestEngine, BacktestParameters
from app.quant.factors import FACTOR_DEFINITION_VERSION, FACTOR_NAMES, FactorEngine
from app.quant.market import MarketDataService
from app.quant.risk import RISK_METHODOLOGY_VERSION
from app.schemas import BacktestRequest, FactorRequest, MarketSnapshotArguments
from app.version import code_version


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
        date_properties = {
            "start_date": {"type": ["string", "null"], "format": "date"},
            "end_date": {"type": ["string", "null"], "format": "date"},
        }
        symbols_property = {"type": ["array", "null"], "items": {"type": "string"}}
        return [
            {
                "type": "function", "name": "market_snapshot",
                "description": "查询当前数据集中指定资产的 OHLCV 行情。",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "当前数据集中的必填股票代码"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                        **date_properties,
                    },
                    "required": ["symbol", "limit", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function", "name": "factor_snapshot",
                "description": "在唯一研究日期计算横截面因子排名。",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string", "enum": sorted(FACTOR_NAMES)},
                        "lookback": {"type": "integer", "minimum": 5, "maximum": 252},
                        "symbols": symbols_property, **date_properties,
                    },
                    "required": ["factor", "lookback", "symbols", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function", "name": "run_backtest",
                "description": "运行信号滞后一日、权重漂移并扣除交易成本的横截面 Top-K 回测。",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string", "enum": sorted(FACTOR_NAMES)},
                        "lookback": {"type": "integer", "minimum": 5, "maximum": 252},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5000},
                        "rebalance_days": {"type": "integer", "minimum": 1, "maximum": 63},
                        "transaction_cost_bps": {"type": "number", "minimum": 0, "maximum": 100},
                        "symbols": symbols_property, **date_properties,
                    },
                    "required": [
                        "factor", "lookback", "top_k", "rebalance_days",
                        "transaction_cost_bps", "symbols", "start_date", "end_date",
                    ],
                    "additionalProperties": False,
                },
            },
        ]

    def validate_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise QuantAgentError("UNKNOWN_TOOL", f"Unknown tool: {name}")
        return self._validators[name].model_validate(arguments).model_dump(mode="json", exclude_none=True)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = self.validate_call(name, arguments)
        return self._tools[name](**validated)

    def _symbol_frame(self, symbols: list[str] | None = None) -> pd.DataFrame:
        frame = self.market.frame
        if symbols:
            unknown = sorted(set(symbols) - set(self.market.symbols()))
            if unknown:
                raise QuantAgentError(
                    "UNKNOWN_SYMBOL", "One or more symbols do not exist",
                    status_code=404, details={"symbols": unknown},
                )
            frame = frame[frame["symbol"].isin(symbols)]
        return frame.reset_index(drop=True)

    def market_snapshot(
        self,
        symbol: str,
        limit: int = 20,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        try:
            prices = self.market.prices(symbol, limit, start_date, end_date)
        except KeyError as error:
            raise QuantAgentError("UNKNOWN_SYMBOL", str(error.args[0]), status_code=404) from error
        except LookupError as error:
            raise QuantAgentError("NO_DATA_IN_RANGE", str(error), status_code=404) from error
        return {"symbol": symbol.upper(), "dataset": self.market.dataset, "prices": prices}

    def factor_snapshot(
        self,
        factor: str = "momentum",
        lookback: int = 20,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        frame = self._symbol_frame(symbols)
        try:
            snapshot = self.factors.snapshot(frame, factor, lookback, start_date, end_date)
        except ValueError as error:
            raise QuantAgentError(
                "INSUFFICIENT_FACTOR_DATA" if "Insufficient" in str(error) or "At least" in str(error) else "INVALID_PARAMETERS",
                str(error),
            ) from error
        return {
            "factor": factor, "lookback": lookback, "dataset": self.market.dataset,
            "description": FACTOR_NAMES.get(factor), "factor_definition_version": FACTOR_DEFINITION_VERSION,
            **snapshot,
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
        frame = self._symbol_frame(symbols)
        parameters = BacktestParameters(factor, lookback, top_k, rebalance_days, transaction_cost_bps)
        try:
            result = self.backtests.run(frame, parameters, start_date, end_date)
        except ValueError as error:
            message = str(error)
            code = "INSUFFICIENT_MARKET_DATA" if any(
                token in message for token in ["Insufficient", "Missing held", "No market data"]
            ) else "INVALID_BACKTEST"
            raise QuantAgentError(code, message) from error
        actual_symbols = sorted(frame["symbol"].unique().tolist())
        series_start = result["series"][0]["date"]
        series_end = result["series"][-1]["date"]
        result["parameters"].update(
            {"symbols": actual_symbols, "start_date": start_date or series_start, "end_date": end_date or series_end}
        )
        dataset = self.market.dataset
        provenance = {
            "dataset_id": dataset.get("id"),
            "dataset_name": dataset.get("name"),
            "market": dataset.get("market"),
            "adjustment": dataset.get("adjustment"),
            "raw_file_hash": dataset.get("raw_file_hash"),
            "normalized_data_hash": dataset.get("normalized_data_hash") or dataset.get("content_hash"),
            "dataset_hash": dataset.get("normalized_data_hash") or dataset.get("content_hash"),
            "symbols": actual_symbols,
            "start_date": series_start,
            "end_date": series_end,
            "factor_definition_version": FACTOR_DEFINITION_VERSION,
            "risk_methodology_version": RISK_METHODOLOGY_VERSION,
            "backtest_methodology_version": BACKTEST_METHODOLOGY_VERSION,
            **code_version(ROOT_DIR),
        }
        run_id = self.database.save_backtest(
            result["strategy"], result["parameters"], result["metrics"], result["series"],
            provenance, result["methodology"],
        )
        result["id"] = run_id
        result["provenance"] = provenance
        return result
