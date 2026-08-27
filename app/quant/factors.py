from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_DEFINITION_VERSION = "factor-v2.0.1"
MIN_CROSS_SECTION_COVERAGE = 0.8
FACTOR_NAMES = {
    "momentum": "N-day close-to-close momentum",
    "reversal": "Negative N-day close-to-close return",
    "volatility": "Annualized rolling volatility (lower is better)",
    "sma_ratio": "Close relative to N-day moving average",
    "volume_zscore": "N-day rolling volume z-score",
}


class FactorEngine:
    def compute(self, market: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        if lookback < 5:
            raise ValueError("lookback must be at least 5")
        frame = market.sort_values(["symbol", "date"]).copy()
        grouped = frame.groupby("symbol", group_keys=False)
        frame["return_1d"] = grouped["close"].pct_change(fill_method=None)
        frame["momentum"] = grouped["close"].pct_change(lookback, fill_method=None)
        frame["reversal"] = -grouped["close"].pct_change(lookback, fill_method=None)
        rolling_vol = grouped["return_1d"].rolling(lookback).std().reset_index(level=0, drop=True)
        frame["volatility_raw"] = rolling_vol * np.sqrt(252)
        frame["volatility"] = -frame["volatility_raw"]
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
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        if factor not in FACTOR_NAMES:
            raise ValueError(f"Unsupported factor: {factor}")
        requested_symbols = sorted(market["symbol"].unique().tolist())
        if len(requested_symbols) < 2:
            raise ValueError("At least two symbols are required for a cross-sectional ranking")
        frame = self.compute(market, lookback)
        candidates = frame
        if start_date:
            candidates = candidates[candidates["date"] >= pd.Timestamp(start_date)]
        if end_date:
            candidates = candidates[candidates["date"] <= pd.Timestamp(end_date)]
        valid = candidates.dropna(subset=[factor])
        coverage = valid.groupby("date")["symbol"].nunique() / len(requested_symbols)
        eligible_dates = coverage[coverage >= MIN_CROSS_SECTION_COVERAGE].index
        if len(eligible_dates) == 0:
            raise ValueError(
                "Insufficient effective cross-section: no date meets the 80% coverage threshold"
            )
        as_of_date = pd.Timestamp(max(eligible_dates))
        snapshot = valid[valid["date"] == as_of_date].copy()
        if len(snapshot) < 2:
            raise ValueError("Insufficient effective cross-section: fewer than two valid symbols")
        snapshot = snapshot.sort_values([factor, "symbol"], ascending=[False, True])
        effective = set(snapshot["symbol"].tolist())
        excluded = [
            {"symbol": symbol, "reason": "factor_value_unavailable_on_as_of_date"}
            for symbol in requested_symbols
            if symbol not in effective
        ]
        direction = "lower_is_better" if factor == "volatility" else "higher_is_better"
        ranking = []
        for rank, row in enumerate(snapshot.itertuples(), 1):
            score = float(getattr(row, factor))
            raw_value = float(row.volatility_raw) if factor == "volatility" else score
            ranking.append(
                {
                    "date": as_of_date.strftime("%Y-%m-%d"),
                    "symbol": row.symbol,
                    "factor": factor,
                    "raw_value": round(raw_value, 6),
                    "score": round(score, 6),
                    "value": round(score, 6),
                    "direction": direction,
                    "rank": rank,
                }
            )
        return {
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "requested_symbol_count": len(requested_symbols),
            "effective_symbol_count": len(snapshot),
            "coverage_ratio": round(len(snapshot) / len(requested_symbols), 6),
            "coverage_threshold": MIN_CROSS_SECTION_COVERAGE,
            "exclusions": excluded,
            "ranking": ranking,
        }
