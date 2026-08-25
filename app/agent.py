from __future__ import annotations

import re
from typing import Any

from app.db import Database
from app.quant.factors import FACTOR_NAMES
from app.tools import ToolRegistry


FACTOR_ALIASES = {
    "动量": "momentum",
    "momentum": "momentum",
    "反转": "reversal",
    "reversal": "reversal",
    "波动率": "volatility",
    "volatility": "volatility",
    "均线": "sma_ratio",
    "sma": "sma_ratio",
    "成交量": "volume_zscore",
    "volume": "volume_zscore",
}


class ResearchAgent:
    """A deterministic planner that exposes the same boundary as an LLM tool agent.

    Numeric work is always delegated to validated tools. This avoids fabricated
    financial figures and makes every result reproducible in tests.
    """

    def __init__(self, tools: ToolRegistry, database: Database) -> None:
        self.tools = tools
        self.database = database

    def _factor(self, query: str, explicit: str | None) -> str:
        if explicit:
            if explicit not in FACTOR_NAMES:
                raise ValueError(f"Unsupported factor: {explicit}")
            return explicit
        lowered = query.lower()
        return next((value for key, value in FACTOR_ALIASES.items() if key in lowered), "momentum")

    @staticmethod
    def _extract_symbol(query: str) -> str:
        match = re.search(r"\b(ALPHA|BETA|GAMMA|DELTA|EPSILON|OMEGA)\b", query.upper())
        return match.group(1) if match else "ALPHA"

    def plan(self, query: str, options: dict[str, Any]) -> list[dict[str, Any]]:
        lowered = query.lower()
        factor = self._factor(query, options.get("factor"))
        common = {
            "factor": factor,
            "lookback": options.get("lookback") or 20,
        }
        plan: list[dict[str, Any]] = []
        if any(word in lowered for word in ("回测", "收益", "夏普", "回撤", "backtest", "risk")):
            plan.append(
                {
                    "tool": "run_backtest",
                    "arguments": {
                        **common,
                        "top_k": options.get("top_k") or 2,
                        "rebalance_days": options.get("rebalance_days") or 5,
                        "transaction_cost_bps": options.get("transaction_cost_bps") if options.get("transaction_cost_bps") is not None else 5.0,
                    },
                }
            )
        elif any(word in lowered for word in ("因子", "排名", "factor")):
            plan.append({"tool": "factor_snapshot", "arguments": common})
        elif any(word in lowered for word in ("行情", "价格", "k线", "price", "market")):
            plan.append(
                {
                    "tool": "market_snapshot",
                    "arguments": {"symbol": self._extract_symbol(query), "limit": 30},
                }
            )
        if any(word in lowered for word in ("研报", "文档", "依据", "引用", "知识库", "为什么", "research")):
            plan.append({"tool": "knowledge_search", "arguments": {"query": query, "top_k": 4}})
        if not plan:
            plan.append({"tool": "knowledge_search", "arguments": {"query": query, "top_k": 4}})
        return plan

    def run(self, query: str, **options: Any) -> dict[str, Any]:
        plan = self.plan(query, options)
        task_id = self.database.create_task(query, plan)
        try:
            steps = []
            for item in plan:
                output = self.tools.call(item["tool"], item["arguments"])
                steps.append({**item, "output": output})
            result = {
                "task_id": task_id,
                "query": query,
                "plan": plan,
                "steps": steps,
                "answer": self._summarize(steps),
            }
            self.database.finish_task(task_id, result)
            return result
        except Exception as error:
            self.database.fail_task(task_id, str(error))
            raise

    @staticmethod
    def _summarize(steps: list[dict[str, Any]]) -> str:
        sentences = []
        for step in steps:
            output = step["output"]
            if step["tool"] == "run_backtest":
                metrics = output["metrics"]
                sentences.append(
                    "回测完成：累计收益 {total:.2%}，年化收益 {annual:.2%}，"
                    "夏普比率 {sharpe:.2f}，最大回撤 {drawdown:.2%}。".format(
                        total=metrics["total_return"],
                        annual=metrics["annual_return"],
                        sharpe=metrics["sharpe_ratio"],
                        drawdown=metrics["max_drawdown"],
                    )
                )
            elif step["tool"] == "factor_snapshot":
                top = output["ranking"][0]
                sentences.append(f"{output['factor']} 因子当前排名第一的是 {top['symbol']}。")
            elif step["tool"] == "market_snapshot":
                last = output["prices"][-1]
                sentences.append(f"{output['symbol']} 最新收盘价为 {last['close']:.2f}。")
            elif step["tool"] == "knowledge_search":
                sentences.append(output["answer"])
        return " ".join(sentences)
