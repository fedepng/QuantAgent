from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
RISK_METHODOLOGY_VERSION = "risk-v2.0.1"


def equity_and_drawdown(returns: pd.Series) -> pd.DataFrame:
    """Build the authoritative equity/drawdown series from daily returns."""
    clean = pd.Series(returns, copy=True).astype(float)
    if clean.empty:
        raise ValueError("No returns available for risk calculation")
    values = clean.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Returns must contain only finite values")
    if (values <= -1).any():
        raise ValueError("A daily return less than or equal to -100% makes equity non-positive")

    equity = (1.0 + clean).cumprod()
    if not np.isfinite(equity.to_numpy()).all() or (equity <= 0).any():
        raise ValueError("Equity must remain finite and positive")
    peaks = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)], ignore_index=True).cummax()
    drawdown = equity.reset_index(drop=True) / peaks.iloc[1:].reset_index(drop=True) - 1.0
    return pd.DataFrame(
        {"equity": equity.to_numpy(dtype=float), "drawdown": drawdown.to_numpy(dtype=float)},
        index=clean.index,
    )


def calculate_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    clean = pd.Series(returns, copy=True).astype(float)
    curve = equity_and_drawdown(clean)
    periods = len(clean)
    years = periods / TRADING_DAYS_PER_YEAR
    annualized_log = math.log(float(curve["equity"].iloc[-1])) / years
    if annualized_log > math.log(np.finfo(float).max):
        raise ValueError("Annualized return is not finite for this short/extreme series")
    annual_return = math.exp(annualized_log) - 1.0
    daily_std = float(clean.std(ddof=1)) if periods > 1 else 0.0
    annual_volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        float(clean.mean()) / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_std > 0
        else 0.0
    )
    max_drawdown = float(curve["drawdown"].min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    var_95 = float(clean.quantile(0.05))
    tail = clean[clean <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95
    metrics = {
        "total_return": float(curve["equity"].iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "win_rate": float((clean > 0).mean()),
    }
    if turnover is not None:
        values = pd.Series(turnover, copy=True).astype(float)
        if len(values) != periods or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Turnover must be finite and aligned with returns")
        metrics["average_daily_turnover"] = float(values.mean())
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Risk calculation produced a non-finite metric")
    return {key: round(value, 6) for key, value in metrics.items()}
