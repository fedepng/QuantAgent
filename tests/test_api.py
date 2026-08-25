from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeResponses:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            call = SimpleNamespace(
                type="function_call",
                name="run_backtest",
                call_id="call_backtest",
                arguments=(
                    '{"factor":"momentum","lookback":20,"top_k":2,'
                    '"rebalance_days":5,"transaction_cost_bps":5.0}'
                ),
            )
            return SimpleNamespace(
                id="resp_plan",
                output=[call],
                output_text="",
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
            )
        assert any(
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == "call_backtest"
            for item in kwargs["input"]
        )
        return SimpleNamespace(
            id="resp_answer",
            output=[],
            output_text="回测完成，结果包含夏普比率和最大回撤。",
            usage=SimpleNamespace(input_tokens=200, output_tokens=30, total_tokens=230),
        )


class FakeDeepSeek:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def make_client(tmp_path: Path, deepseek_client=None) -> TestClient:
    settings = Settings(database_path=tmp_path / "api.db", random_seed=42)
    return TestClient(create_app(settings, deepseek_client=deepseek_client))


def test_health_and_tool_schemas(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["symbols"] == 6
        assert health.json()["agent"] == {
            "provider": "deepseek", "model": "deepseek-v4-flash", "configured": False
        }
        tools = client.get("/api/tools").json()
        assert {item["name"] for item in tools} == {
            "market_snapshot", "factor_snapshot", "run_backtest"
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


def test_agent_workflow(tmp_path: Path) -> None:
    fake = FakeDeepSeek()
    with make_client(tmp_path, fake) as client:
        agent = client.post(
            "/api/agent/run",
            json={"query": "回测20日动量因子并给出夏普比率和最大回撤"},
        )
        assert agent.status_code == 200
        body = agent.json()
        assert body["plan"][0]["tool"] == "run_backtest"
        assert body["agent"] == "deepseek"
        assert "夏普比率" in body["answer"]
        assert len(fake.responses.requests) == 2
        tasks = client.get("/api/tasks").json()
        assert tasks[0]["status"] == "completed"


def test_agent_requires_deepseek_configuration(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post("/api/agent/run", json={"query": "回测动量因子"})
        assert response.status_code == 503
        assert "DEEPSEEK_API_KEY" in response.json()["detail"]
