from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import FACTOR_NAMES, FactorEngine
from .risk import calculate_metrics


@dataclass(frozen=True)
class BacktestParameters:
    factor: str = "momentum"
    lookback: int = 20
    top_k: int = 2
    rebalance_days: int = 5
    transaction_cost_bps: float = 5.0


class BacktestEngine:
    def __init__(self, factor_engine: FactorEngine) -> None:
        self.factor_engine = factor_engine

    def run(self, market: pd.DataFrame, parameters: BacktestParameters) -> dict[str, object]:
        if parameters.factor not in FACTOR_NAMES:
            raise ValueError(f"Unsupported factor: {parameters.factor}")
        symbols = sorted(market["symbol"].unique())
        if not 1 <= parameters.top_k <= len(symbols):
            raise ValueError(f"top_k must be between 1 and {len(symbols)}")
        if parameters.rebalance_days < 1:
            raise ValueError("rebalance_days must be positive")

        factor_frame = self.factor_engine.compute(market, parameters.lookback)
        returns = factor_frame.pivot(index="date", columns="symbol", values="return_1d")
        raw_signal = factor_frame.pivot(index="date", columns="symbol", values=parameters.factor)
        signal = raw_signal.shift(1)  # use only information available before today's return

        current_weights = pd.Series(0.0, index=returns.columns)
        strategy_returns: list[float] = []
        turnovers: list[float] = []
        weights_history: list[dict[str, float]] = []
        valid_started = False

        for index, date in enumerate(returns.index):
            row_signal = signal.loc[date].dropna()
            rebalance = index % parameters.rebalance_days == 0
            if rebalance and len(row_signal) >= parameters.top_k:
                selected = row_signal.nlargest(parameters.top_k).index
                target = pd.Series(0.0, index=returns.columns)
                target.loc[selected] = 1 / parameters.top_k
                turnover = float((target - current_weights).abs().sum())
                current_weights = target
                valid_started = True
            else:
                turnover = 0.0

            daily_asset_returns = returns.loc[date].fillna(0.0)
            gross_return = float((current_weights * daily_asset_returns).sum())
            cost = turnover * parameters.transaction_cost_bps / 10_000
            strategy_return = gross_return - cost if valid_started else 0.0
            strategy_returns.append(strategy_return)
            turnovers.append(turnover)
            weights_history.append(
                {symbol: round(float(weight), 6) for symbol, weight in current_weights.items() if weight > 0}
            )

        result = pd.DataFrame(
            {
                "date": returns.index,
                "return": strategy_returns,
                "turnover": turnovers,
                "weights": weights_history,
            }
        )
        result = result.iloc[parameters.lookback + 2 :].reset_index(drop=True)
        metrics = calculate_metrics(result["return"], result["turnover"])
        equity = (1 + result["return"]).cumprod()
        result["equity"] = equity
        result["drawdown"] = equity / equity.cummax() - 1
        series = [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "return": round(float(row.return_), 8),
                "equity": round(float(row.equity), 6),
                "drawdown": round(float(row.drawdown), 6),
                "turnover": round(float(row.turnover), 6),
                "weights": row.weights,
            }
            for row in result.rename(columns={"return": "return_"}).itertuples()
        ]
        return {
            "strategy": f"cross_sectional_{parameters.factor}",
            "parameters": parameters.__dict__,
            "metrics": metrics,
            "series": series,
            "methodology": {
                "signal_lag": 1,
                "rebalance": f"every {parameters.rebalance_days} trading days",
                "weighting": "equal weight among top-k symbols",
                "cost_model": "turnover multiplied by one-way transaction cost",
            },
        }
