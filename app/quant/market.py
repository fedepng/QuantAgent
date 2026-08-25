from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}


def generate_demo_market(seed: int, periods: int = 520) -> pd.DataFrame:
    """Generate a deterministic multi-asset OHLCV data set for offline demos."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-08-21", periods=periods)
    profiles = {
        "ALPHA": (0.00045, 0.012),
        "BETA": (0.00025, 0.016),
        "GAMMA": (0.00010, 0.010),
        "DELTA": (-0.00005, 0.018),
        "EPSILON": (0.00035, 0.014),
        "OMEGA": (0.00015, 0.020),
    }
    frames: list[pd.DataFrame] = []
    for offset, (symbol, (drift, volatility)) in enumerate(profiles.items()):
        market_cycle = 0.0008 * np.sin(np.linspace(0, 8 * np.pi, periods) + offset)
        shocks = rng.normal(drift + market_cycle, volatility, periods)
        close = (70 + offset * 9) * np.exp(np.cumsum(shocks))
        overnight = rng.normal(0, volatility * 0.25, periods)
        open_price = close * np.exp(overnight)
        spread = np.abs(rng.normal(volatility * 0.55, volatility * 0.18, periods))
        high = np.maximum(open_price, close) * (1 + spread)
        low = np.minimum(open_price, close) * (1 - spread)
        volume = rng.lognormal(mean=13.4 + offset * 0.04, sigma=0.28, size=periods)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume.astype("int64"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = REQUIRED_COLUMNS - set(frame.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    frame.columns = frame.columns.str.lower()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("CSV contains missing required values")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("CSV contains duplicate date/symbol rows")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    invalid_range = (
        (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame["high"])
    )
    if invalid_range.any():
        rows = (invalid_range[invalid_range].index + 2).tolist()[:10]
        raise ValueError(f"Invalid OHLC relationship at CSV rows: {rows}")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_market_csv(path: Path | BinaryIO | BytesIO) -> pd.DataFrame:
    return normalize_market_frame(pd.read_csv(path))


def market_quality(frame: pd.DataFrame) -> dict[str, object]:
    counts = frame.groupby("symbol")["date"].count()
    dates = frame["date"].drop_duplicates().sort_values()
    expected = len(dates)
    missing_by_symbol = {
        str(symbol): int(expected - count)
        for symbol, count in counts.items()
        if count < expected
    }
    return {
        "duplicate_rows": 0,
        "missing_values": int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum()),
        "trading_days": expected,
        "missing_trading_days_by_symbol": missing_by_symbol,
    }


class MarketDataService:
    def __init__(self, seed: int) -> None:
        self._frame = generate_demo_market(seed)
        self._dataset = {
            "id": None,
            "name": "内置模拟数据",
            "market": "DEMO",
            "adjustment": "none",
            "content_hash": f"demo-{seed}",
            "is_demo": True,
        }

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def dataset(self) -> dict[str, object]:
        return dict(self._dataset)

    def replace(self, frame: pd.DataFrame, dataset: dict[str, object]) -> None:
        self._frame = normalize_market_frame(frame)
        self._dataset = {**dataset, "is_demo": False}

    def replace_from_csv(self, path: Path) -> None:
        self._frame = load_market_csv(path)

    def symbols(self) -> list[str]:
        return sorted(self._frame["symbol"].unique().tolist())

    def prices(
        self,
        symbol: str,
        limit: int = 120,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        symbol = symbol.upper()
        frame = self._frame[self._frame["symbol"] == symbol].copy()
        if start_date:
            frame = frame[frame["date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame[frame["date"] <= pd.Timestamp(end_date)]
        frame = frame.tail(limit)
        if frame.empty:
            raise KeyError(f"Unknown symbol: {symbol}")
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        return frame.to_dict(orient="records")
