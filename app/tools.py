from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from app.db import Database
from app.quant.backtest import BacktestEngine, BacktestParameters
from app.quant.factors import FACTOR_NAMES, FactorEngine
from app.quant.market import MarketDataService
from app.rag.service import RagService


class ToolRegistry:
    def __init__(
        self,
        market: MarketDataService,
        factors: FactorEngine,
        backtests: BacktestEngine,
        rag: RagService,
        database: Database,
    ) -> None:
        self.market = market
        self.factors = factors
        self.backtests = backtests
        self.rag = rag
        self.database = database
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "market_snapshot": self.market_snapshot,
            "factor_snapshot": self.factor_snapshot,
            "run_backtest": self.run_backtest,
            "knowledge_search": self.knowledge_search,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "market_snapshot",
                "description": "Return recent OHLCV data for a symbol.",
                "parameters": {"symbol": "string", "limit": "integer"},
            },
            {
                "name": "factor_snapshot",
                "description": "Rank all assets by a deterministic factor.",
                "parameters": {"factor": sorted(FACTOR_NAMES), "lookback": "integer"},
            },
            {
                "name": "run_backtest",
                "description": "Run a lagged cross-sectional factor backtest with costs.",
                "parameters": {
                    "factor": sorted(FACTOR_NAMES),
                    "lookback": "integer",
                    "top_k": "integer",
                    "rebalance_days": "integer",
                    "transaction_cost_bps": "number",
                },
            },
            {
                "name": "knowledge_search",
                "description": "Retrieve evidence from uploaded reports and notes.",
                "parameters": {"query": "string", "top_k": "integer"},
            },
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**arguments)

    def market_snapshot(self, symbol: str = "ALPHA", limit: int = 20) -> dict[str, Any]:
        return {"symbol": symbol.upper(), "prices": self.market.prices(symbol, limit)}

    def factor_snapshot(self, factor: str = "momentum", lookback: int = 20) -> dict[str, Any]:
        return {
            "factor": factor,
            "lookback": lookback,
            "description": FACTOR_NAMES.get(factor),
            "ranking": self.factors.snapshot(self.market.frame, factor, lookback),
        }

    def run_backtest(
        self,
        factor: str = "momentum",
        lookback: int = 20,
        top_k: int = 2,
        rebalance_days: int = 5,
        transaction_cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        parameters = BacktestParameters(
            factor=factor,
            lookback=lookback,
            top_k=top_k,
            rebalance_days=rebalance_days,
            transaction_cost_bps=transaction_cost_bps,
        )
        result = self.backtests.run(self.market.frame, parameters)
        run_id = self.database.save_backtest(
            result["strategy"], asdict(parameters), result["metrics"], result["series"]
        )
        result["id"] = run_id
        return result

    def knowledge_search(self, query: str, top_k: int = 4) -> dict[str, Any]:
        return self.rag.search(query, top_k)

