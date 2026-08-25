from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.db import Database
from app.tools import ToolRegistry


AGENT_INSTRUCTIONS = """
你是 QuantAgent，一名量化研究助手。你的职责是理解用户需求、调用工具并基于工具结果作答。

规则：
1. 涉及行情、因子、回测、收益率或风险指标时必须调用对应工具，禁止自行编造或心算金融数值。
2. 用户在自然语言中给出的周期、持仓数量、调仓间隔和交易成本应准确转换为工具参数；未指定时采用 momentum、20 日回看、Top-2、每 5 日调仓和 5 bps 交易成本，不得自行更改默认值。
3. 可以连续调用多个工具，例如先回测，再查询因子排名或行情解释结果。
4. 最终回答应说明数据集、采用的参数、关键结果和必要的风险提示。
5. 本系统用于研究演示，不提供个性化投资建议，不承诺收益。
""".strip()


class AgentUnavailableError(RuntimeError):
    pass


class AgentExecutionError(RuntimeError):
    pass


class DeepSeekResearchAgent:
    def __init__(
        self,
        tools: ToolRegistry,
        database: Database,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.deepseek.com",
        max_tool_rounds: int = 6,
        client: Any | None = None,
    ) -> None:
        self.tools = tools
        self.database = database
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.client = client or (
            OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        )

    @property
    def configured(self) -> bool:
        return self.client is not None

    def _user_input(self, query: str, options: dict[str, Any]) -> str:
        explicit = {key: value for key, value in options.items() if value is not None}
        context = {
            "dataset": self.tools.market.dataset,
            "available_symbols": self.tools.market.symbols(),
        }
        parts = [query, f"当前研究上下文：{json.dumps(context, ensure_ascii=False)}"]
        if explicit:
            parts.append(
                "以下是用户通过界面显式指定的参数，调用工具时必须优先采用："
                f"{json.dumps(explicit, ensure_ascii=False)}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _usage(response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return {
            key: getattr(usage, key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if hasattr(usage, key)
        }

    @staticmethod
    def _model_tool_output(name: str, output: dict[str, Any]) -> dict[str, Any]:
        if name != "run_backtest":
            return output
        return {
            "id": output["id"],
            "strategy": output["strategy"],
            "parameters": output["parameters"],
            "metrics": output["metrics"],
            "methodology": output["methodology"],
            "latest_observations": output["series"][-5:],
        }

    def _create_response(self, input_items: list[Any]) -> Any:
        return self.client.responses.create(
            model=self.model,
            instructions=AGENT_INSTRUCTIONS,
            tools=self.tools.schemas,
            input=input_items,
        )

    def run(self, query: str, **options: Any) -> dict[str, Any]:
        if not self.configured:
            raise AgentUnavailableError(
                "DeepSeek Agent 尚未配置。请在 .env 中设置 DEEPSEEK_API_KEY 后重启服务。"
            )

        plan: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        usages: list[dict[str, Any]] = []
        task_id = self.database.create_task(
            query, [{"agent": "deepseek", "model": self.model, "status": "planning"}]
        )
        input_items: list[Any] = [
            {"role": "user", "content": self._user_input(query, options)}
        ]

        try:
            response = self._create_response(input_items)
            for round_index in range(self.max_tool_rounds + 1):
                usage = self._usage(response)
                if usage:
                    usages.append(usage)

                calls = [
                    item
                    for item in response.output
                    if getattr(item, "type", None) == "function_call"
                ]
                if not calls:
                    answer = (response.output_text or "").strip()
                    if not answer:
                        raise AgentExecutionError("DeepSeek Agent 未返回最终文本。")
                    result = {
                        "task_id": task_id,
                        "query": query,
                        "agent": "deepseek",
                        "model": self.model,
                        "response_id": getattr(response, "id", None),
                        "plan": plan,
                        "steps": steps,
                        "answer": answer,
                        "usage": usages,
                    }
                    self.database.update_task_plan(task_id, plan)
                    self.database.finish_task(task_id, result)
                    return result

                if round_index >= self.max_tool_rounds:
                    raise AgentExecutionError(
                        f"DeepSeek Agent 超过最大工具调用轮数 {self.max_tool_rounds}。"
                    )

                input_items.extend(response.output)
                for call in calls:
                    try:
                        raw_arguments = json.loads(call.arguments)
                    except json.JSONDecodeError as error:
                        raise AgentExecutionError(
                            f"工具 {call.name} 的参数不是合法 JSON。"
                        ) from error

                    arguments = self.tools.validate_call(call.name, raw_arguments)
                    item = {
                        "round": round_index + 1,
                        "call_id": call.call_id,
                        "tool": call.name,
                        "arguments": arguments,
                    }
                    plan.append(item)
                    self.database.update_task_plan(task_id, plan)

                    output = self.tools.call(call.name, arguments)
                    steps.append({**item, "output": output})
                    model_output = self._model_tool_output(call.name, output)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(model_output, ensure_ascii=False),
                        }
                    )

                response = self._create_response(input_items)
        except AgentUnavailableError:
            raise
        except AgentExecutionError as error:
            self.database.update_task_plan(task_id, plan)
            self.database.fail_task(task_id, str(error))
            raise
        except Exception as error:
            message = f"DeepSeek Agent 调用失败：{error}"
            self.database.update_task_plan(task_id, plan)
            self.database.fail_task(task_id, message)
            raise AgentExecutionError(message) from error

        raise AgentExecutionError("DeepSeek Agent 未能完成任务。")
