from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("No returns available for risk calculation")
    equity = (1 + clean).cumprod()
    years = max(len(clean) / 252, 1 / 252)
    annual_return = equity.iloc[-1] ** (1 / years) - 1
    annual_volatility = clean.std(ddof=1) * math.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    var_95 = float(clean.quantile(0.05))
    tail = clean[clean <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95
    metrics = {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "win_rate": float((clean > 0).mean()),
    }
    if turnover is not None:
        metrics["average_daily_turnover"] = float(turnover.mean())
    return {key: round(value, 6) for key, value in metrics.items()}

