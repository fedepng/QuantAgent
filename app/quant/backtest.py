from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import FACTOR_NAMES, FactorEngine
from .risk import calculate_metrics, equity_and_drawdown

BACKTEST_METHODOLOGY_VERSION = "backtest-v2.0.1"


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

    def run(
        self,
        market: pd.DataFrame,
        parameters: BacktestParameters,
        performance_start: str | None = None,
        performance_end: str | None = None,
    ) -> dict[str, object]:
        if parameters.factor not in FACTOR_NAMES:
            raise ValueError(f"Unsupported factor: {parameters.factor}")
        symbols = sorted(market["symbol"].unique())
        if not 1 <= parameters.top_k <= len(symbols):
            raise ValueError(f"top_k must be between 1 and {len(symbols)}")
        if parameters.rebalance_days < 1:
            raise ValueError("rebalance_days must be positive")

        factor_frame = self.factor_engine.compute(market, parameters.lookback)
        returns = factor_frame.pivot(index="date", columns="symbol", values="return_1d").sort_index()
        raw_signal = factor_frame.pivot(index="date", columns="symbol", values=parameters.factor).sort_index()
        signal = raw_signal.shift(1)
        if performance_start:
            returns = returns[returns.index >= pd.Timestamp(performance_start)]
            signal = signal.reindex(returns.index)
        if performance_end:
            returns = returns[returns.index <= pd.Timestamp(performance_end)]
            signal = signal.reindex(returns.index)
        if returns.empty:
            raise ValueError("No market data matches the requested performance range")

        current_weights = pd.Series(0.0, index=returns.columns, dtype=float)
        rows: list[dict[str, object]] = []
        active_day = 0
        first_signal_date: pd.Timestamp | None = None
        first_holding_date: pd.Timestamp | None = None
        skipped_dates_before_start = 0
        incomplete_signal_counts: dict[str, int] = {str(symbol): 0 for symbol in returns.columns}

        for date in returns.index:
            row_signal = signal.loc[date]
            for symbol in returns.columns[row_signal.isna()]:
                incomplete_signal_counts[str(symbol)] += 1
            valid_signal = row_signal.dropna()
            can_form = len(valid_signal) >= parameters.top_k
            if first_holding_date is None and not can_form:
                skipped_dates_before_start += 1
                continue

            rebalance = first_holding_date is None or active_day % parameters.rebalance_days == 0
            turnover = 0.0
            if rebalance:
                if not can_form:
                    raise ValueError(
                        f"Insufficient factor coverage on scheduled rebalance date {date:%Y-%m-%d}"
                    )
                selected = valid_signal.nlargest(parameters.top_k).index
                target = pd.Series(0.0, index=returns.columns, dtype=float)
                target.loc[selected] = 1.0 / parameters.top_k
                turnover = float((target - current_weights).abs().sum())
                current_weights = target
                if first_signal_date is None:
                    signal_position = raw_signal.index.get_loc(date) - 1
                    first_signal_date = pd.Timestamp(raw_signal.index[signal_position])
                    first_holding_date = pd.Timestamp(date)

            held = current_weights[current_weights > 0].index
            daily_asset_returns = returns.loc[date]
            missing_held = [str(symbol) for symbol in held if pd.isna(daily_asset_returns[symbol])]
            if missing_held:
                raise ValueError(
                    f"Missing held-asset return on {date:%Y-%m-%d}: {', '.join(missing_held)}"
                )
            held_returns = daily_asset_returns.loc[held].astype(float)
            if not np.isfinite(held_returns.to_numpy()).all():
                raise ValueError(f"Non-finite held-asset return on {date:%Y-%m-%d}")
            gross_return = float((current_weights.loc[held] * held_returns).sum())
            cost = turnover * parameters.transaction_cost_bps / 10_000.0
            net_return = gross_return - cost
            if net_return <= -1:
                raise ValueError(f"Portfolio equity became non-positive on {date:%Y-%m-%d}")
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "return": net_return,
                    "gross_return": gross_return,
                    "cost": cost,
                    "turnover": turnover,
                    "weights": {
                        str(symbol): round(float(weight), 8)
                        for symbol, weight in current_weights.items()
                        if weight > 0
                    },
                }
            )
            denominator = 1.0 + gross_return
            drifted = current_weights.copy()
            drifted.loc[held] = current_weights.loc[held] * (1.0 + held_returns) / denominator
            current_weights = drifted
            active_day += 1

        if not rows or first_holding_date is None or first_signal_date is None:
            raise ValueError("Insufficient warm-up data: no valid signal can form the first portfolio")
        result = pd.DataFrame(rows)
        metrics = calculate_metrics(result["return"], result["turnover"])
        curve = equity_and_drawdown(result["return"])
        result[["equity", "drawdown"]] = curve[["equity", "drawdown"]]
        series = [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "return": round(float(row.return_), 8),
                "gross_return": round(float(row.gross_return), 8),
                "cost": round(float(row.cost), 8),
                "equity": round(float(row.equity), 8),
                "drawdown": round(float(row.drawdown), 8),
                "turnover": round(float(row.turnover), 8),
                "weights": row.weights,
            }
            for row in result.rename(columns={"return": "return_"}).itertuples()
        ]
        skipped_symbols = {
            symbol: count for symbol, count in incomplete_signal_counts.items() if count > 0
        }
        return {
            "strategy": f"cross_sectional_{parameters.factor}",
            "parameters": parameters.__dict__.copy(),
            "metrics": metrics,
            "series": series,
            "methodology": {
                "version": BACKTEST_METHODOLOGY_VERSION,
                "signal_timing": "T close signal; position applied to the next research day's close-to-close return",
                "signal_lag": 1,
                "rebalance": f"every {parameters.rebalance_days} effective holding days",
                "weighting": "equal weight among top-k symbols at rebalance; weights drift with returns between rebalances",
                "cost_model": "one-way bps multiplied by two-sided turnover; initial entry charged, final liquidation not charged",
                "missing_price_policy": "fail when a held asset lacks a close-to-close return; incomplete non-held signals are excluded",
                "first_signal_date": first_signal_date.strftime("%Y-%m-%d"),
                "first_holding_date": first_holding_date.strftime("%Y-%m-%d"),
                "effective_trading_days": len(series),
                "skipped_dates_before_first_holding": skipped_dates_before_start,
                "incomplete_signal_days_by_symbol": skipped_symbols,
            },
        }
