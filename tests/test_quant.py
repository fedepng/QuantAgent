from __future__ import annotations

import numpy as np

from app.quant.backtest import BacktestEngine, BacktestParameters
from app.quant.factors import FactorEngine
from app.quant.market import generate_demo_market
from app.quant.risk import calculate_metrics


def test_demo_market_is_reproducible() -> None:
    first = generate_demo_market(42, periods=80)
    second = generate_demo_market(42, periods=80)
    assert first.equals(second)
    assert first["symbol"].nunique() == 6


def test_factor_snapshot_returns_all_symbols() -> None:
    market = generate_demo_market(42, periods=100)
    snapshot = FactorEngine().snapshot(market, "momentum", 20)
    ranking = snapshot["ranking"]
    assert len(ranking) == 6
    assert [item["rank"] for item in ranking] == list(range(1, 7))
    assert all(np.isfinite(item["value"]) for item in ranking)


def test_backtest_applies_costs_and_returns_metrics() -> None:
    market = generate_demo_market(42, periods=260)
    engine = BacktestEngine(FactorEngine())
    free = engine.run(market, BacktestParameters(transaction_cost_bps=0))
    costly = engine.run(market, BacktestParameters(transaction_cost_bps=20))
    assert costly["metrics"]["total_return"] <= free["metrics"]["total_return"]
    assert costly["methodology"]["signal_lag"] == 1
    assert {"annual_return", "sharpe_ratio", "max_drawdown"} <= set(costly["metrics"])


def test_future_price_change_does_not_change_past_equity() -> None:
    market = generate_demo_market(42, periods=180)
    engine = BacktestEngine(FactorEngine())
    parameters = BacktestParameters()
    baseline = engine.run(market, parameters)["series"]
    changed = market.copy()
    last_date = changed["date"].max()
    changed.loc[(changed["date"] == last_date) & (changed["symbol"] == "ALPHA"), "close"] *= 10
    altered = engine.run(changed, parameters)["series"]
    assert [item["equity"] for item in baseline[:-1]] == [item["equity"] for item in altered[:-1]]


def test_risk_metrics_have_expected_drawdown() -> None:
    returns = __import__("pandas").Series([0.1, -0.2, 0.05])
    metrics = calculate_metrics(returns)
    assert metrics["max_drawdown"] == -0.2
    assert metrics["daily_cvar_95"] <= metrics["daily_var_95"]
