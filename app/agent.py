from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from openai import OpenAI
from pydantic import ValidationError

from app.db import Database
from app.errors import QuantAgentError
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


@dataclass(frozen=True)
class NormalizedToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class ModelTurn:
    response_id: str | None
    answer: str
    calls: list[NormalizedToolCall]
    usage: dict[str, Any] | None
    raw: Any


def _usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    names = (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    return {name: getattr(usage, name) for name in names if hasattr(usage, name)}


def _chat_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for schema in schemas:
        function = {
            key: schema[key]
            for key in ("name", "description", "parameters", "strict")
            if key in schema
        }
        result.append({"type": "function", "function": function})
    return result


def _chat_message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = call.function
        calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": function.name,
                    "arguments": function.arguments,
                },
            }
        )
    result: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None),
    }
    if calls:
        result["tool_calls"] = calls
    return result


class ProtocolAdapter(Protocol):
    name: str

    def initial_state(self, user_input: str) -> list[Any]: ...

    def request(
        self,
        client: Any,
        model: str,
        schemas: list[dict[str, Any]],
        state: list[Any],
    ) -> ModelTurn: ...

    def append_model_turn(self, state: list[Any], turn: ModelTurn) -> None: ...

    def append_tool_output(
        self,
        state: list[Any],
        call: NormalizedToolCall,
        output: dict[str, Any],
    ) -> None: ...


class ResponsesAdapter:
    name = "responses"

    def initial_state(self, user_input: str) -> list[Any]:
        return [{"role": "user", "content": user_input}]

    def request(
        self,
        client: Any,
        model: str,
        schemas: list[dict[str, Any]],
        state: list[Any],
    ) -> ModelTurn:
        response = client.responses.create(
            model=model,
            instructions=AGENT_INSTRUCTIONS,
            tools=schemas,
            input=state,
        )
        calls = [
            NormalizedToolCall(item.call_id, item.name, item.arguments)
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]
        return ModelTurn(
            response_id=getattr(response, "id", None),
            answer=(getattr(response, "output_text", "") or "").strip(),
            calls=calls,
            usage=_usage(response),
            raw=response.output,
        )

    def append_model_turn(self, state: list[Any], turn: ModelTurn) -> None:
        state.extend(turn.raw)

    def append_tool_output(
        self,
        state: list[Any],
        call: NormalizedToolCall,
        output: dict[str, Any],
    ) -> None:
        state.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(output, ensure_ascii=False),
            }
        )


class ChatCompletionsAdapter:
    name = "chat_completions"

    def initial_state(self, user_input: str) -> list[Any]:
        return [
            {"role": "system", "content": AGENT_INSTRUCTIONS},
            {"role": "user", "content": user_input},
        ]

    def request(
        self,
        client: Any,
        model: str,
        schemas: list[dict[str, Any]],
        state: list[Any],
    ) -> ModelTurn:
        response = client.chat.completions.create(
            model=model,
            messages=state,
            tools=_chat_tools(schemas),
            tool_choice="auto",
        )
        message = response.choices[0].message
        calls = [
            NormalizedToolCall(call.id, call.function.name, call.function.arguments)
            for call in (getattr(message, "tool_calls", None) or [])
            if getattr(call, "type", "function") == "function"
        ]
        content = getattr(message, "content", "") or ""
        answer = content.strip() if isinstance(content, str) else str(content).strip()
        return ModelTurn(
            response_id=getattr(response, "id", None),
            answer=answer,
            calls=calls,
            usage=_usage(response),
            raw=message,
        )

    def append_model_turn(self, state: list[Any], turn: ModelTurn) -> None:
        state.append(_chat_message_dict(turn.raw))

    def append_tool_output(
        self,
        state: list[Any],
        call: NormalizedToolCall,
        output: dict[str, Any],
    ) -> None:
        state.append(
            {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": json.dumps(output, ensure_ascii=False),
            }
        )


def create_protocol_adapter(protocol: str) -> ProtocolAdapter:
    normalized = protocol.strip().lower().replace("-", "_")
    if normalized == "responses":
        return ResponsesAdapter()
    if normalized in {"chat", "chat_completions"}:
        return ChatCompletionsAdapter()
    raise ValueError(
        f"不支持的 LLM 协议：{protocol}。可选值为 responses、chat_completions。"
    )


class ResearchAgent:
    def __init__(
        self,
        tools: ToolRegistry,
        database: Database,
        api_key: str | None,
        model: str,
        base_url: str,
        provider: str = "custom",
        protocol: str = "responses",
        max_tool_rounds: int = 6,
        client: Any | None = None,
    ) -> None:
        self.tools = tools
        self.database = database
        self.provider = provider.strip().lower() or "custom"
        self.model = model
        self.base_url = base_url
        self.adapter = create_protocol_adapter(protocol)
        self.protocol = self.adapter.name
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

    def run(self, query: str, **options: Any) -> dict[str, Any]:
        if not self.configured:
            raise AgentUnavailableError(
                "自然语言研究功能尚未配置。请在 .env 中设置 LLM_API_KEY 后重启服务。"
            )

        plan: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        usages: list[dict[str, Any]] = []
        research_metadata: dict[str, Any] = {
            "dataset": self.tools.market.dataset,
            "backtest_ids": [],
        }
        def persisted_plan() -> list[dict[str, Any]]:
            return [
                {key: value for key, value in item.items() if key != "arguments"}
                for item in plan
            ]
        task_id = self.database.create_task(
            query,
            [
                {
                    "provider": self.provider,
                    "protocol": self.protocol,
                    "model": self.model,
                    "status": "planning",
                }
            ],
        )
        state = self.adapter.initial_state(self._user_input(query, options))

        try:
            for round_index in range(self.max_tool_rounds + 1):
                turn = self.adapter.request(
                    self.client, self.model, self.tools.schemas, state
                )
                if turn.usage:
                    usages.append(turn.usage)

                if not turn.calls:
                    if not turn.answer:
                        raise AgentExecutionError("模型未返回最终文本。")
                    result = {
                        "task_id": task_id,
                        "query": query,
                        "agent": self.provider,
                        "provider": self.provider,
                        "protocol": self.protocol,
                        "model": self.model,
                        "response_id": turn.response_id,
                        "plan": plan,
                        "steps": steps,
                        "answer": turn.answer,
                        "usage": usages,
                        "research_metadata": research_metadata,
                    }
                    self.database.update_task_plan(task_id, persisted_plan())
                    persisted = {
                        "task_id": task_id,
                        "answer": turn.answer,
                        "call_ids": [item["call_record_id"] for item in plan],
                        "research_metadata": research_metadata,
                        "usage": usages,
                    }
                    self.database.finish_task(task_id, persisted)
                    return result

                if round_index >= self.max_tool_rounds:
                    raise AgentExecutionError(
                        f"模型超过最大工具调用轮数 {self.max_tool_rounds}。"
                    )

                self.adapter.append_model_turn(state, turn)
                for call in turn.calls:
                    started = perf_counter()
                    record_id = self.database.start_tool_call(
                        task_id, round_index + 1, call.call_id, call.name, None
                    )
                    try:
                        raw_arguments = json.loads(call.arguments)
                    except json.JSONDecodeError as error:
                        self.database.finish_tool_call(
                            record_id,
                            duration_ms=(perf_counter() - started) * 1000,
                            error_code="INVALID_TOOL_ARGUMENT_JSON",
                            error_message=f"工具 {call.name} 的参数不是合法 JSON。",
                        )
                        raise AgentExecutionError(
                            f"工具 {call.name} 的参数不是合法 JSON。"
                        ) from error
                    try:
                        arguments = self.tools.validate_call(call.name, raw_arguments)
                        with self.database.connect() as connection:
                            connection.execute(
                                "UPDATE tool_calls SET arguments_json=? WHERE id=?",
                                (json.dumps(arguments, ensure_ascii=False), record_id),
                            )
                        item = {
                            "round": round_index + 1,
                            "call_id": call.call_id,
                            "tool": call.name,
                            "call_record_id": record_id,
                            "arguments": arguments,
                        }
                        plan.append(item)
                        self.database.update_task_plan(task_id, persisted_plan())
                        output = self.tools.call(call.name, arguments)
                        model_output = self._model_tool_output(call.name, output)
                        backtest_id = output.get("id") if call.name == "run_backtest" else None
                        if backtest_id is not None:
                            research_metadata["backtest_ids"].append(backtest_id)
                        summary = {
                            key: model_output[key]
                            for key in ("id", "strategy", "parameters", "metrics", "methodology", "as_of_date")
                            if key in model_output
                        }
                        if call.name == "market_snapshot":
                            summary = {
                                "symbol": output["symbol"],
                                "observation_count": len(output["prices"]),
                                "start_date": output["prices"][0]["date"],
                                "end_date": output["prices"][-1]["date"],
                            }
                        self.database.finish_tool_call(
                            record_id,
                            duration_ms=(perf_counter() - started) * 1000,
                            summary=summary,
                            backtest_id=backtest_id,
                        )
                    except Exception as error:
                        if isinstance(error, QuantAgentError):
                            code = error.code
                        elif isinstance(error, ValidationError):
                            code = "INVALID_TOOL_ARGUMENTS"
                        else:
                            code = "TOOL_EXECUTION_FAILED"
                        self.database.finish_tool_call(
                            record_id,
                            duration_ms=(perf_counter() - started) * 1000,
                            error_code=code,
                            error_message=str(error),
                        )
                        raise
                    steps.append({"call_record_id": record_id, "tool": call.name, "output": output})
                    self.adapter.append_tool_output(
                        state,
                        call,
                        model_output,
                    )
        except AgentUnavailableError:
            raise
        except AgentExecutionError as error:
            self.database.update_task_plan(task_id, persisted_plan())
            self.database.fail_task(task_id, str(error))
            raise
        except Exception as error:
            message = f"模型调用失败：{error}"
            self.database.update_task_plan(task_id, persisted_plan())
            self.database.fail_task(task_id, message)
            raise AgentExecutionError(message) from error

        raise AgentExecutionError("模型未能完成任务。")
