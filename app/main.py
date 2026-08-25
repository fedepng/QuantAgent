from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import AgentExecutionError, AgentUnavailableError, DeepSeekResearchAgent
from app.config import ROOT_DIR, Settings, load_settings
from app.db import Database
from app.datasets import DatasetService
from app.quant.backtest import BacktestEngine, BacktestParameters
from app.quant.factors import FACTOR_NAMES, FactorEngine
from app.quant.market import MarketDataService
from app.schemas import AgentRequest, BacktestRequest, FactorRequest
from app.tools import ToolRegistry


def create_app(settings: Settings | None = None, deepseek_client: Any | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.database_path)
    database.initialize()
    market = MarketDataService(settings.random_seed)
    dataset_root = settings.dataset_path or (settings.database_path.parent / "datasets")
    datasets = DatasetService(database, market, dataset_root)
    factors = FactorEngine()
    backtests = BacktestEngine(factors)
    tools = ToolRegistry(market, factors, backtests, database)
    agent = DeepSeekResearchAgent(
        tools,
        database,
        settings.deepseek_api_key,
        settings.deepseek_model,
        settings.deepseek_base_url,
        settings.deepseek_max_tool_rounds,
        deepseek_client,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="QuantAgent",
        description="DeepSeek tool-calling quantitative research agent with deterministic tools.",
        version="2.0.0",
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
    app.state.datasets = datasets
    app.state.factors = factors
    app.state.backtests = backtests
    app.state.tools = tools
    app.state.agent = agent

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": app.version,
            "symbols": len(market.symbols()),
            "dataset": market.dataset,
            "agent": {
                "provider": "deepseek",
                "model": agent.model,
                "configured": agent.configured,
            },
        }

    @app.get("/api/tools")
    def tool_schemas() -> list[dict[str, Any]]:
        return tools.schemas

    @app.get("/api/market/symbols")
    def symbols(dataset_id: int | None = None) -> list[str]:
        try:
            return datasets.symbols(dataset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/datasets/import", status_code=201)
    async def import_dataset(
        file: UploadFile = File(...),
        name: str = Form(""),
        market_name: str = Form("CN"),
        adjustment: str = Form("qfq"),
        source: str = Form("upload"),
        activate: bool = Form(True),
    ) -> dict[str, object]:
        try:
            content = await file.read()
            return datasets.import_csv(
                content,
                file.filename or "market.csv",
                name,
                market_name,
                adjustment,
                source,
                activate,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, object]]:
        return datasets.list()

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: int) -> dict[str, object]:
        dataset = datasets.get(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset

    @app.post("/api/datasets/{dataset_id}/activate")
    def activate_dataset(dataset_id: int) -> dict[str, object]:
        try:
            return datasets.activate(dataset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/datasets/{dataset_id}/symbols")
    def dataset_symbols(dataset_id: int) -> list[str]:
        try:
            return datasets.symbols(dataset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/market/prices")
    def prices(
        symbol: str = "ALPHA",
        limit: int = Query(120, ge=1, le=5000),
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        try:
            return tools.market_snapshot(symbol, limit, start_date, end_date)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/factors/analyze")
    def analyze_factor(request: FactorRequest) -> dict[str, Any]:
        try:
            return tools.factor_snapshot(**request.model_dump(mode="json"))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/backtests")
    def run_backtest(request: BacktestRequest) -> dict[str, Any]:
        try:
            return tools.run_backtest(**request.model_dump(mode="json"))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/backtests/{run_id}")
    def get_backtest(run_id: int) -> dict[str, Any]:
        result = database.get_backtest(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return result

    @app.get("/api/backtests")
    def list_backtests(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return database.list_backtests(limit)

    @app.post("/api/agent/run")
    def run_agent(request: AgentRequest) -> dict[str, Any]:
        try:
            values = request.model_dump(exclude={"query"}, mode="json")
            return agent.run(request.query, **values)
        except AgentUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except AgentExecutionError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/tasks")
    def tasks(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return database.list_tasks(limit)

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: int) -> dict[str, Any]:
        task = database.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    static_dir = ROOT_DIR / "app" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
