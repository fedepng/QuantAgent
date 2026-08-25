from pathlib import Path
from types import SimpleNamespace
import json

import httpx
from openai import OpenAI

from app.agent import DeepSeekResearchAgent
from app.db import Database
from app.quant.backtest import BacktestEngine
from app.quant.factors import FactorEngine
from app.quant.market import MarketDataService
from app.tools import ToolRegistry


class ScriptedResponses:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        index = len(self.requests)
        if index == 1:
            return SimpleNamespace(
                id="response_1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="run_backtest",
                        call_id="backtest_1",
                        arguments=(
                            '{"factor":"momentum","lookback":30,"top_k":3,'
                            '"rebalance_days":10,"transaction_cost_bps":8.0}'
                        ),
                    )
                ],
                output_text="",
                usage=None,
            )
        if index == 2:
            previous_output = next(
                item for item in kwargs["input"]
                if isinstance(item, dict) and item.get("type") == "function_call_output"
            )
            assert previous_output["call_id"] == "backtest_1"
            assert '"series"' not in previous_output["output"]
            return SimpleNamespace(
                id="response_2",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="factor_snapshot",
                        call_id="factor_1",
                        arguments='{"factor":"momentum","lookback":30}',
                    )
                ],
                output_text="",
                usage=None,
            )
        return SimpleNamespace(
            id="response_3",
            output=[],
            output_text="30 日动量回测已完成，并结合当前因子排名说明了结果。",
            usage=None,
        )


class ScriptedDeepSeek:
    def __init__(self) -> None:
        self.responses = ScriptedResponses()


def test_deepseek_agent_runs_multiple_tool_rounds(tmp_path: Path) -> None:
    database = Database(tmp_path / "agent.db")
    database.initialize()
    market = MarketDataService(42)
    factors = FactorEngine()
    tools = ToolRegistry(market, factors, BacktestEngine(factors), database)
    client = ScriptedDeepSeek()
    agent = DeepSeekResearchAgent(tools, database, None, "deepseek-v4-flash", client=client)

    result = agent.run("使用30日动量因子，每10日调仓，选择前三并解释震荡风险")

    assert [item["tool"] for item in result["plan"]] == ["run_backtest", "factor_snapshot"]
    assert result["plan"][0]["arguments"] == {
        "factor": "momentum",
        "lookback": 30,
        "top_k": 3,
        "rebalance_days": 10,
        "transaction_cost_bps": 8.0,
    }
    assert result["steps"][1]["output"]["ranking"]
    assert len(client.responses.requests) == 3
    task = database.list_tasks(1)[0]
    assert task["status"] == "completed"
    assert len(task["plan"]) == 2


def test_deepseek_responses_protocol_with_openai_sdk(tmp_path: Path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload["model"] == "deepseek-v4-flash"
            assert all(tool["type"] == "function" and tool["strict"] for tool in payload["tools"])
            output = [
                {
                    "type": "function_call",
                    "id": "fc_sdk",
                    "call_id": "call_sdk",
                    "name": "factor_snapshot",
                    "arguments": '{"factor":"reversal","lookback":15}',
                    "status": "completed",
                }
            ]
        else:
            assert any(item.get("type") == "function_call" for item in payload["input"])
            assert any(item.get("type") == "function_call_output" for item in payload["input"])
            output = [
                {
                    "type": "message",
                    "id": "msg_sdk",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "反转因子排名已经计算完成。",
                            "annotations": [],
                        }
                    ],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"resp_sdk_{len(requests)}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "deepseek-v4-flash",
                "output": output,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": payload["tools"],
            },
        )

    database = Database(tmp_path / "sdk.db")
    database.initialize()
    market = MarketDataService(42)
    factors = FactorEngine()
    tools = ToolRegistry(market, factors, BacktestEngine(factors), database)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(api_key="test-key", base_url="https://api.deepseek.test", http_client=http_client)
    agent = DeepSeekResearchAgent(tools, database, None, "deepseek-v4-flash", client=client)

    result = agent.run("分析15日反转因子排名")

    assert result["answer"] == "反转因子排名已经计算完成。"
    assert result["plan"][0]["arguments"] == {"factor": "reversal", "lookback": 15}
    assert len(requests) == 2
