from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import ResearchAgent
from app.config import ROOT_DIR, Settings, load_settings
from app.db import Database
from app.quant.backtest import BacktestEngine, BacktestParameters
from app.quant.factors import FACTOR_NAMES, FactorEngine
from app.quant.market import MarketDataService
from app.rag.service import RagService
from app.schemas import AgentRequest, BacktestRequest, DocumentRequest, FactorRequest, RagSearchRequest
from app.tools import ToolRegistry


SEED_RESEARCH_NOTE = """
动量因子研究说明

动量策略按照过去一段时间的累计收益对资产排序，并在固定调仓日持有排名靠前的资产。
为避免未来函数，交易日 t 的持仓只能使用 t-1 日收盘后已经得到的因子值。本项目对因子矩阵整体滞后一日，
再计算当日持仓收益。回测同时按照权重变化绝对值计算换手率，并扣除换手率乘以单边交易成本。

风险指标说明

年化收益率按净值期末值和有效交易日数量复合年化；年化波动率使用日收益标准差乘以 sqrt(252)；
夏普比率使用年化收益率除以年化波动率。最大回撤通过净值除以历史累计最高净值再减一计算。
VaR 采用日收益 5% 分位数，CVaR 采用不高于该分位数的尾部收益均值。
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.database_path)
    database.initialize()
    market = MarketDataService(settings.random_seed)
    factors = FactorEngine()
    backtests = BacktestEngine(factors)
    rag = RagService(database, settings.embedding_dim)
    if not rag.list_documents():
        rag.add_document("内置策略研究说明", SEED_RESEARCH_NOTE, "built-in")
    tools = ToolRegistry(market, factors, backtests, rag, database)
    agent = ResearchAgent(tools, database)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="QuantAgent",
        description="Reproducible quantitative research agent with RAG and deterministic tools.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.database = database
    app.state.market = market
    app.state.factors = factors
    app.state.backtests = backtests
    app.state.rag = rag
    app.state.tools = tools
    app.state.agent = agent

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": app.version,
            "symbols": len(market.symbols()),
            "documents": len(rag.list_documents()),
            "vector_backend": rag.index.backend,
        }

    @app.get("/api/tools")
    def tool_schemas() -> list[dict[str, Any]]:
        return tools.schemas

    @app.get("/api/market/symbols")
    def symbols() -> list[str]:
        return market.symbols()

    @app.get("/api/market/prices")
    def prices(symbol: str = "ALPHA", limit: int = Query(120, ge=1, le=520)) -> dict[str, Any]:
        try:
            return tools.market_snapshot(symbol, limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/factors/analyze")
    def analyze_factor(request: FactorRequest) -> dict[str, Any]:
        try:
            return tools.factor_snapshot(request.factor, request.lookback)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/backtests")
    def run_backtest(request: BacktestRequest) -> dict[str, Any]:
        try:
            return tools.run_backtest(**request.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/backtests/{run_id}")
    def get_backtest(run_id: int) -> dict[str, Any]:
        result = database.get_backtest(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return result

    @app.get("/api/documents")
    def documents() -> list[dict[str, object]]:
        return rag.list_documents()

    @app.post("/api/documents", status_code=201)
    def add_document(request: DocumentRequest) -> dict[str, object]:
        try:
            return rag.add_document(request.title, request.content, request.source)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: int) -> dict[str, bool]:
        if not rag.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True}

    @app.post("/api/rag/search")
    def search_rag(request: RagSearchRequest) -> dict[str, object]:
        try:
            return rag.search(request.query, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/agent/run")
    def run_agent(request: AgentRequest) -> dict[str, Any]:
        try:
            values = request.model_dump(exclude={"query"})
            return agent.run(request.query, **values)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/tasks")
    def tasks(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return database.list_tasks(limit)

    static_dir = ROOT_DIR / "app" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()

