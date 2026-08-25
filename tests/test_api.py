from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(database_path=tmp_path / "api.db", random_seed=42, embedding_dim=128)
    return TestClient(create_app(settings))


def test_health_and_tool_schemas(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["symbols"] == 6
        tools = client.get("/api/tools").json()
        assert {item["name"] for item in tools} == {
            "market_snapshot", "factor_snapshot", "run_backtest", "knowledge_search"
        }


def test_backtest_and_persistence(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/backtests",
            json={"factor": "momentum", "lookback": 20, "top_k": 2, "rebalance_days": 5, "transaction_cost_bps": 5},
        )
        assert response.status_code == 200
        result = response.json()
        saved = client.get(f"/api/backtests/{result['id']}")
        assert saved.status_code == 200
        assert saved.json()["metrics"] == result["metrics"]


def test_document_rag_and_agent_workflow(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        document = client.post(
            "/api/documents",
            json={"title": "因子说明", "content": "反转因子关注短期超跌后的均值回归。", "source": "test"},
        )
        assert document.status_code == 201
        search = client.post("/api/rag/search", json={"query": "什么是反转因子", "top_k": 3})
        assert search.status_code == 200
        assert search.json()["citations"]
        agent = client.post(
            "/api/agent/run",
            json={"query": "回测20日动量因子并给出夏普比率和最大回撤"},
        )
        assert agent.status_code == 200
        body = agent.json()
        assert body["plan"][0]["tool"] == "run_backtest"
        assert "夏普比率" in body["answer"]
        tasks = client.get("/api/tasks").json()
        assert tasks[0]["status"] == "completed"

