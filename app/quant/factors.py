from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_NAMES = {
    "momentum": "N-day close-to-close momentum",
    "reversal": "Short-term reversal",
    "volatility": "Annualized rolling volatility (lower is better)",
    "sma_ratio": "Close relative to moving average",
    "volume_zscore": "Rolling volume z-score",
}


class FactorEngine:
    def compute(self, market: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        if lookback < 5:
            raise ValueError("lookback must be at least 5")
        frame = market.sort_values(["symbol", "date"]).copy()
        grouped = frame.groupby("symbol", group_keys=False)
        frame["return_1d"] = grouped["close"].pct_change()
        frame["momentum"] = grouped["close"].pct_change(lookback)
        frame["reversal"] = -grouped["close"].pct_change(5)
        rolling_vol = grouped["return_1d"].rolling(lookback).std().reset_index(level=0, drop=True)
        frame["volatility"] = -rolling_vol * np.sqrt(252)
        rolling_mean = grouped["close"].rolling(lookback).mean().reset_index(level=0, drop=True)
        frame["sma_ratio"] = frame["close"] / rolling_mean - 1
        volume_mean = grouped["volume"].rolling(lookback).mean().reset_index(level=0, drop=True)
        volume_std = grouped["volume"].rolling(lookback).std().reset_index(level=0, drop=True)
        frame["volume_zscore"] = (frame["volume"] - volume_mean) / volume_std.replace(0, np.nan)
        return frame

    def snapshot(
        self,
        market: pd.DataFrame,
        factor: str = "momentum",
        lookback: int = 20,
    ) -> list[dict[str, object]]:
        if factor not in FACTOR_NAMES:
            raise ValueError(f"Unsupported factor: {factor}")
        frame = self.compute(market, lookback)
        latest = frame.dropna(subset=[factor]).groupby("symbol", as_index=False).tail(1)
        latest = latest.sort_values(factor, ascending=False)
        return [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "symbol": row.symbol,
                "factor": factor,
                "value": round(float(getattr(row, factor)), 6),
                "rank": rank,
            }
            for rank, row in enumerate(latest.itertuples(), 1)
        ]

