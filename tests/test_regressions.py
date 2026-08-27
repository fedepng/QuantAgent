from __future__ import annotations

import math
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.agent import AgentExecutionError, ResearchAgent
from app.config import ROOT_DIR, Settings
from app.db import Database
from app.main import create_app
from app.quant.backtest import BacktestEngine, BacktestParameters
from app.quant.factors import FactorEngine
from app.quant.market import MarketDataService, load_market_csv
from app.quant.risk import calculate_metrics, equity_and_drawdown
from app.tools import ToolRegistry


def make_settings(root: Path) -> Settings:
    return Settings(
        database_path=root / "quantagent.db",
        dataset_path=root / "datasets",
        random_seed=42,
    )


def valid_csv() -> bytes:
    return (
        "date,symbol,open,high,low,close,volume\n"
        "2026-01-02, a ,10,12,9,11,100.5\n"
        "2026-01-03,A,11,13,10,12,101\n"
    ).encode()


def test_initial_equity_is_used_for_first_day_drawdown() -> None:
    returns = pd.Series([-0.20, 0.10])
    curve = equity_and_drawdown(returns)
    metrics = calculate_metrics(returns)
    assert curve["drawdown"].tolist() == pytest.approx([-0.20, -0.12])
    assert metrics["max_drawdown"] == -0.20
    assert metrics["sharpe_ratio"] == pytest.approx(
        returns.mean() / returns.std(ddof=1) * math.sqrt(252), abs=1e-6
    )


def test_risk_boundaries_are_finite_or_controlled() -> None:
    assert calculate_metrics(pd.Series([0.01]))["annual_volatility"] == 0
    assert calculate_metrics(pd.Series([0.01, 0.01]))["sharpe_ratio"] == 0
    for values in ([], [np.inf], [-1.0]):
        with pytest.raises(ValueError):
            calculate_metrics(pd.Series(values, dtype=float))


def test_factor_snapshot_uses_one_common_date_and_requested_reversal_window() -> None:
    market = MarketDataService(42).frame
    last = market["date"].max()
    shortened = market[~((market["symbol"] == "ALPHA") & (market["date"] == last))]
    snapshot = FactorEngine().snapshot(shortened, "reversal", 15)
    assert {item["date"] for item in snapshot["ranking"]} == {snapshot["as_of_date"]}
    assert snapshot["effective_symbol_count"] >= math.ceil(
        snapshot["requested_symbol_count"] * snapshot["coverage_threshold"]
    )
    computed = FactorEngine().compute(shortened, 15)
    row = computed[
        (computed["symbol"] == snapshot["ranking"][0]["symbol"])
        & (computed["date"] == pd.Timestamp(snapshot["as_of_date"]))
    ].iloc[0]
    assert snapshot["ranking"][0]["score"] == pytest.approx(row["reversal"], abs=1e-6)


def test_low_volatility_exposes_raw_value_score_and_direction() -> None:
    snapshot = FactorEngine().snapshot(MarketDataService(42).frame, "volatility", 20)
    for item in snapshot["ranking"]:
        assert item["raw_value"] >= 0
        assert item["score"] == pytest.approx(-item["raw_value"], abs=2e-6)
        assert item["direction"] == "lower_is_better"


def test_csv_normalizes_natural_dates_symbols_and_accepts_bom() -> None:
    frame = load_market_csv(BytesIO(b"\xef\xbb\xbf" + valid_csv()))
    assert frame["symbol"].tolist() == ["A", "A"]
    assert frame["volume"].dtype.kind == "f"


@pytest.mark.parametrize(
    "content,field",
    [
        (
            b"date,symbol,open,high,low,close,volume\n"
            b"2026-01-02 00:00,A,10,11,9,10,1\n"
            b"2026-01-02 12:00,a,10,11,9,10,1\n",
            "date,symbol",
        ),
        (
            b"date,symbol,open,high,low,close,volume\n"
            b"2026-01-02,A,10,inf,9,10,1\n",
            "high",
        ),
        (
            b"date,symbol,Open, open,high,low,close,volume\n"
            b"2026-01-02,A,10,10,11,9,10,1\n",
            "open",
        ),
        (
            b"date,symbol,open,high,low,close,volume\n"
            b"2026-01-02,   ,10,11,9,10,1\n",
            "symbol",
        ),
    ],
)
def test_csv_contract_returns_structured_errors(content: bytes, field: str) -> None:
    with pytest.raises(Exception) as captured:
        load_market_csv(BytesIO(content))
    details = captured.value.details
    assert details["total_errors"] >= 1
    assert any(item["field"] == field for item in details["errors"])


def test_backtest_golden_accounting_includes_entry_cost_and_weight_drift() -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)
    closes = {
        "A": [100, 100, 100, 100, 100, 110, 121, 133.1],
        "B": [100, 100, 100, 100, 100, 105, 105, 105],
        "C": [100, 100, 100, 100, 100, 90, 90, 90],
    }
    rows = []
    for symbol, values in closes.items():
        for date, close in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": date, "symbol": symbol, "open": close, "high": close,
                    "low": close, "close": close, "volume": 100,
                }
            )
    result = BacktestEngine(FactorEngine()).run(
        pd.DataFrame(rows),
        BacktestParameters(lookback=5, top_k=2, rebalance_days=2, transaction_cost_bps=10),
    )
    first, second = result["series"]
    assert first["turnover"] == 1
    assert first["cost"] == 0.001
    assert first["gross_return"] == pytest.approx(0.05)
    assert first["return"] == pytest.approx(0.049)
    assert second["weights"]["A"] == pytest.approx(0.5 * 1.1 / 1.05)
    assert second["weights"]["B"] == pytest.approx(0.5 / 1.05)
    assert second["gross_return"] == pytest.approx((0.5 * 1.1 / 1.05) * 0.1)
    assert result["methodology"]["first_holding_date"] == dates[6].strftime("%Y-%m-%d")


def test_backtest_uses_warmup_before_requested_start() -> None:
    market = MarketDataService(42).frame
    start = market["date"].drop_duplicates().sort_values().iloc[100].strftime("%Y-%m-%d")
    engine = BacktestEngine(FactorEngine())
    full = engine.run(market, BacktestParameters(lookback=20), performance_start=start)
    assert full["series"][0]["date"] >= start
    assert full["methodology"]["skipped_dates_before_first_holding"] == 0


def test_api_error_contract_distinguishes_invalid_unknown_and_empty_range(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        invalid = client.get("/api/market/prices", params={"symbol": "ALPHA", "start_date": "bad"})
        assert invalid.status_code == 422
        assert set(invalid.json()) == {"code", "message", "details", "request_id"}
        unknown = client.get("/api/market/prices", params={"symbol": "NOT_REAL"})
        assert unknown.status_code == 404
        assert unknown.json()["code"] == "UNKNOWN_SYMBOL"
        empty = client.get(
            "/api/market/prices",
            params={"symbol": "ALPHA", "start_date": "2099-01-01"},
        )
        assert empty.status_code == 404
        assert empty.json()["code"] == "NO_DATA_IN_RANGE"


def test_dataset_hash_mismatch_is_reported_as_degraded(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    frame = MarketDataService(7).frame.head(80)
    content = frame.to_csv(index=False).encode()
    with TestClient(create_app(settings)) as client:
        dataset = client.post(
            "/api/datasets/import",
            files={"file": ("market.csv", content, "text/csv")},
        ).json()
    database = Database(settings.database_path)
    stored = database.get_dataset(dataset["id"])
    path = settings.dataset_path / stored["storage_path"]
    damaged = pd.read_parquet(path)
    damaged.loc[0, "volume"] += 1
    damaged.to_parquet(path, index=False)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert "HASH_MISMATCH" in health["degraded_reason"]


def test_relative_dataset_storage_survives_project_move(tmp_path: Path) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    settings = make_settings(original)
    content = MarketDataService(7).frame.head(80).to_csv(index=False).encode()
    with TestClient(create_app(settings)) as client:
        assert client.post(
            "/api/datasets/import", files={"file": ("market.csv", content, "text/csv")}
        ).status_code == 201
    moved.mkdir()
    shutil.copy2(settings.database_path, moved / "quantagent.db")
    shutil.copytree(settings.dataset_path, moved / "datasets")
    with TestClient(create_app(make_settings(moved))) as client:
        assert client.get("/health").json()["status"] == "ok"


class InvalidToolResponses:
    def __init__(self) -> None:
        self.responses = self

    def create(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            id="bad", output_text="", usage=None,
            output=[
                SimpleNamespace(
                    type="function_call", call_id="bad-call",
                    name="missing_tool", arguments="{}",
                )
            ],
        )


def test_agent_failure_persists_independent_tool_trace(tmp_path: Path) -> None:
    database = Database(tmp_path / "agent.db")
    database.initialize()
    market = MarketDataService(42)
    factors = FactorEngine()
    agent = ResearchAgent(
        ToolRegistry(market, factors, BacktestEngine(factors), database),
        database, None, "test", "https://example.test",
        client=InvalidToolResponses(), max_tool_rounds=1,
    )
    with pytest.raises(AgentExecutionError):
        agent.run("call a bad tool")
    task = database.list_tasks(1)[0]
    assert task["status"] == "failed"
    assert task["tool_calls"][0]["status"] == "failed"
    assert task["tool_calls"][0]["error_code"] == "UNKNOWN_TOOL"
    assert task["tool_calls"][0]["duration_ms"] >= 0


def test_web_has_no_implicit_backtest_or_demo_symbol_hardcoding() -> None:
    script = (ROOT_DIR / "app" / "static" / "app.js").read_text(encoding="utf-8")
    html = (ROOT_DIR / "app" / "static" / "index.html").read_text(encoding="utf-8")
    initializer = script.split("(async () => {", 1)[1]
    assert "await runBacktest()" not in initializer
    assert "ALPHA" not in script
    assert "ALPHA" not in html
    assert "button.disabled = true" in script
