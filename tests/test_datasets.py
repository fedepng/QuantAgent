from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.quant.market import generate_demo_market


def dataset_csv() -> bytes:
    frame = generate_demo_market(7, periods=90)
    frame = frame[frame["symbol"].isin(["ALPHA", "BETA", "GAMMA"])].copy()
    frame["symbol"] = frame["symbol"].map(
        {"ALPHA": "000001.SZ", "BETA": "600000.SH", "GAMMA": "600519.SH"}
    )
    return frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")


def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "quantagent.db",
        random_seed=42,
        dataset_path=tmp_path / "datasets",
    )


def test_import_activate_and_restore_real_dataset(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    with TestClient(create_app(app_settings)) as client:
        response = client.post(
            "/api/datasets/import",
            files={"file": ("a-share.csv", dataset_csv(), "text/csv")},
            data={
                "name": "A股样例",
                "market_name": "CN",
                "adjustment": "qfq",
                "source": "test",
                "activate": "true",
            },
        )
        assert response.status_code == 201
        dataset = response.json()
        assert dataset["symbol_count"] == 3
        assert dataset["row_count"] == 270
        assert dataset["active"] is True
        assert dataset["quality"]["duplicate_rows"] == 0
        assert "storage_path" not in dataset

        assert client.get("/api/market/symbols").json() == [
            "000001.SZ", "600000.SH", "600519.SH"
        ]
        health = client.get("/health").json()
        assert health["dataset"]["id"] == dataset["id"]
        assert health["dataset"]["is_demo"] is False

        factor = client.post(
            "/api/factors/analyze",
            json={"factor": "momentum", "lookback": 10, "symbols": ["000001.SZ", "600519.SH"]},
        )
        assert factor.status_code == 200
        assert {item["symbol"] for item in factor.json()["ranking"]} == {
            "000001.SZ", "600519.SH"
        }

        backtest = client.post(
            "/api/backtests",
            json={
                "factor": "momentum",
                "lookback": 10,
                "top_k": 2,
                "rebalance_days": 5,
                "transaction_cost_bps": 5,
            },
        )
        assert backtest.status_code == 200
        result = backtest.json()
        assert result["provenance"]["dataset_id"] == dataset["id"]
        assert result["provenance"]["dataset_hash"] == dataset["content_hash"]
        assert result["provenance"]["symbols"] == ["000001.SZ", "600000.SH", "600519.SH"]

    with TestClient(create_app(app_settings)) as restored:
        health = restored.get("/health").json()
        assert health["dataset"]["name"] == "A股样例"
        assert restored.get("/api/market/symbols").json() == [
            "000001.SZ", "600000.SH", "600519.SH"
        ]


def test_invalid_ohlc_is_rejected_without_partial_dataset(tmp_path: Path) -> None:
    content = (
        "date,symbol,open,high,low,close,volume\n"
        "2026-01-02,000001.SZ,10,9,8,11,1000\n"
    ).encode()
    with TestClient(create_app(settings(tmp_path))) as client:
        response = client.post(
            "/api/datasets/import",
            files={"file": ("invalid.csv", content, "text/csv")},
            data={"name": "invalid"},
        )
        assert response.status_code == 422
        assert "Invalid OHLC relationship" in response.json()["detail"]
        assert client.get("/api/datasets").json() == []


def test_duplicate_file_is_rejected(tmp_path: Path) -> None:
    content = dataset_csv()
    with TestClient(create_app(settings(tmp_path))) as client:
        first = client.post(
            "/api/datasets/import",
            files={"file": ("market.csv", content, "text/csv")},
            data={"name": "first"},
        )
        second = client.post(
            "/api/datasets/import",
            files={"file": ("market.csv", content, "text/csv")},
            data={"name": "second"},
        )
        assert first.status_code == 201
        assert second.status_code == 422
        assert "already exists" in second.json()["detail"]
