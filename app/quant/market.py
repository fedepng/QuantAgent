from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from app.errors import DataValidationError

REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}
OPTIONAL_COLUMNS = {"amount", "turnover"}
MAX_SYMBOL_LENGTH = 32


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


def _error(row: int | None, field: str, raw: object, reason: str) -> dict[str, object]:
    return {
        "row": row,
        "field": field,
        "raw": None if raw is None else str(raw)[:200],
        "reason": reason,
    }


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    original_columns = [str(column) for column in frame.columns]
    normalized_columns = [column.strip().lower() for column in original_columns]
    errors: list[dict[str, object]] = []
    duplicates = sorted({column for column in normalized_columns if normalized_columns.count(column) > 1})
    for column in duplicates:
        errors.append(_error(1, column, column, "duplicate column after trim/lower normalization"))
    missing = sorted(REQUIRED_COLUMNS - set(normalized_columns))
    for column in missing:
        errors.append(_error(1, column, None, "required column is missing"))
    unknown = sorted(set(normalized_columns) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
    for column in unknown:
        errors.append(_error(1, column, column, "unrecognized column"))
    if errors:
        raise DataValidationError(errors)
    frame.columns = normalized_columns

    raw_dates = frame["date"].copy()
    parsed_dates: list[pd.Timestamp | pd.NaT] = []
    for index, raw in raw_dates.items():
        try:
            parsed = pd.Timestamp(raw)
            if parsed.tzinfo is not None:
                raise ValueError("timezone-aware values are not accepted")
            parsed_dates.append(parsed.normalize())
        except (TypeError, ValueError, OverflowError):
            parsed_dates.append(pd.NaT)
            errors.append(_error(int(index) + 2, "date", raw, "invalid timezone-naive trade date"))
    frame["date"] = pd.to_datetime(pd.Series(parsed_dates, index=frame.index))

    raw_symbols = frame["symbol"].copy()
    symbols = raw_symbols.astype(str).str.strip().str.upper()
    invalid_symbol = (
        symbols.eq("")
        | symbols.str.lower().isin({"nan", "none", "null"})
        | symbols.str.len().gt(MAX_SYMBOL_LENGTH)
    )
    for index in frame.index[invalid_symbol]:
        reason = "symbol is empty or reserved" if len(symbols.at[index]) <= MAX_SYMBOL_LENGTH else "symbol exceeds 32 characters"
        errors.append(_error(int(index) + 2, "symbol", raw_symbols.at[index], reason))
    frame["symbol"] = symbols

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        raw_values = frame[column].copy()
        converted = pd.to_numeric(raw_values, errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        for index in frame.index[invalid]:
            errors.append(_error(int(index) + 2, column, raw_values.at[index], "must be a finite number"))
        frame[column] = converted.astype(float)

    finite_rows = frame[numeric_columns].notna().all(axis=1)
    for column in ["open", "high", "low", "close"]:
        invalid = finite_rows & (frame[column] <= 0)
        for index in frame.index[invalid]:
            errors.append(_error(int(index) + 2, column, frame.at[index, column], "price must be greater than zero"))
    invalid_volume = finite_rows & (frame["volume"] < 0)
    for index in frame.index[invalid_volume]:
        errors.append(_error(int(index) + 2, "volume", frame.at[index, "volume"], "volume must be non-negative"))

    valid_ohlc = finite_rows & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    invalid_range = valid_ohlc & (
        (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame["high"])
    )
    for index in frame.index[invalid_range]:
        errors.append(
            _error(
                int(index) + 2,
                "ohlc",
                {
                    column: frame.at[index, column]
                    for column in ["open", "high", "low", "close"]
                },
                "low/high must contain both open and close",
            )
        )

    duplicate_mask = frame["date"].notna() & ~invalid_symbol & frame.duplicated(["date", "symbol"], keep=False)
    for index in frame.index[duplicate_mask]:
        errors.append(
            _error(
                int(index) + 2,
                "date,symbol",
                f"{raw_dates.at[index]},{raw_symbols.at[index]}",
                "duplicate natural trade date and normalized symbol",
            )
        )
    if errors:
        raise DataValidationError(errors, len(errors))
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_market_csv(path: Path | BinaryIO | BytesIO) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DataValidationError([_error(None, "file", None, f"CSV cannot be parsed: {error}")]) from error
    return normalize_market_frame(frame)


def market_quality(frame: pd.DataFrame) -> dict[str, object]:
    all_dates = frame["date"].drop_duplicates().sort_values()
    expected = len(all_dates)
    counts = frame.groupby("symbol")["date"].count()
    missing_by_symbol = {
        str(symbol): int(expected - count)
        for symbol, count in counts.items()
        if count < expected
    }
    return {
        "duplicate_rows": 0,
        "missing_values": 0,
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
            "raw_file_hash": None,
            "normalized_data_hash": f"demo-{seed}",
            "content_hash": f"demo-{seed}",
            "is_demo": True,
        }
        self.degraded_reason: str | None = None

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def dataset(self) -> dict[str, object]:
        return dict(self._dataset)

    def replace(self, frame: pd.DataFrame, dataset: dict[str, object]) -> None:
        self._frame = normalize_market_frame(frame)
        self._dataset = {**dataset, "is_demo": False}
        self.degraded_reason = None

    def mark_degraded(self, reason: str) -> None:
        self.degraded_reason = reason

    def symbols(self) -> list[str]:
        return sorted(self._frame["symbol"].unique().tolist())

    def prices(
        self,
        symbol: str,
        limit: int = 120,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        symbol = symbol.strip().upper()
        if symbol not in self.symbols():
            raise KeyError(f"Unknown symbol: {symbol}")
        frame = self._frame[self._frame["symbol"] == symbol].copy()
        if start_date:
            frame = frame[frame["date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame[frame["date"] <= pd.Timestamp(end_date)]
        frame = frame.tail(limit)
        if frame.empty:
            raise LookupError(f"No data for {symbol} in the selected date range")
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        return frame.to_dict(orient="records")
