from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent import AgentExecutionError, AgentUnavailableError, ResearchAgent
from app.config import ROOT_DIR, Settings, load_settings
from app.datasets import MAX_UPLOAD_BYTES, DatasetService
from app.db import Database
from app.errors import QuantAgentError
from app.quant.backtest import BacktestEngine
from app.quant.factors import FactorEngine
from app.quant.market import MarketDataService
from app.schemas import (
    AgentRequest,
    BacktestRequest,
    ErrorResponse,
    FactorRequest,
    MarketSnapshotArguments,
)
from app.tools import ToolRegistry
from app.version import BUILD_VERSION

logger = logging.getLogger(__name__)
ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Resource or symbol not found"},
    409: {"model": ErrorResponse, "description": "Dataset state conflict"},
    413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
    422: {"model": ErrorResponse, "description": "Invalid parameters or insufficient data"},
    502: {"model": ErrorResponse, "description": "Upstream model failure"},
    503: {"model": ErrorResponse, "description": "Model is not configured"},
}


def _error_body(request: Request, code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
        "request_id": getattr(request.state, "request_id", "unknown"),
    }


def create_app(settings: Settings | None = None, llm_client: Any | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.database_path)
    database.initialize()
    market = MarketDataService(settings.random_seed)
    dataset_root = settings.dataset_path or (settings.database_path.parent / "datasets")
    datasets = DatasetService(database, market, dataset_root)
    factors = FactorEngine()
    backtests = BacktestEngine(factors)
    tools = ToolRegistry(market, factors, backtests, database)
    agent = ResearchAgent(
        tools=tools,
        database=database,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        provider=settings.llm_provider,
        protocol=settings.llm_protocol,
        max_tool_rounds=settings.llm_max_tool_rounds,
        client=llm_client,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="QuantAgent",
        description="Reproducible local quantitative research workbench with deterministic tools.",
        version=BUILD_VERSION,
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(QuantAgentError)
    async def quant_error(request: Request, error: QuantAgentError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_error_body(request, error.code, error.message, error.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "reason": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_body(request, "INVALID_PARAMETERS", "Request validation failed", details),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unexpected request failure id=%s", request.state.request_id)
        return JSONResponse(
            status_code=500,
            content=_error_body(request, "INTERNAL_ERROR", "Unexpected server error"),
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
            "status": "degraded" if market.degraded_reason else "ok",
            "degraded_reason": market.degraded_reason,
            "version": app.version,
            "symbols": len(market.symbols()),
            "dataset": market.dataset,
            "agent": {
                "provider": agent.provider,
                "protocol": agent.protocol,
                "model": agent.model,
                "configured": agent.configured,
            },
        }

    @app.get("/api/tools")
    def tool_schemas() -> list[dict[str, Any]]:
        return tools.schemas

    @app.get("/api/market/symbols", responses=ERROR_RESPONSES)
    def symbols(dataset_id: int | None = None) -> list[str]:
        return datasets.symbols(dataset_id)

    @app.post("/api/datasets/import", status_code=201, responses=ERROR_RESPONSES)
    async def import_dataset(
        file: UploadFile = File(...),
        name: str = Form(""),
        market_name: str = Form("CN"),
        adjustment: str = Form("qfq"),
        source: str = Form("upload"),
        activate: bool = Form(True),
    ) -> dict[str, object]:
        chunks: list[bytes] = []
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise QuantAgentError(
                    "UPLOAD_TOO_LARGE", "CSV file exceeds the 50 MB limit", status_code=413
                )
            chunks.append(chunk)
        return datasets.import_csv(
            b"".join(chunks), file.filename or "market.csv", name,
            market_name, adjustment, source, activate,
        )

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, object]]:
        return datasets.list()

    @app.get("/api/datasets/{dataset_id}", responses=ERROR_RESPONSES)
    def get_dataset(dataset_id: int) -> dict[str, object]:
        dataset = datasets.get(dataset_id)
        if dataset is None:
            raise QuantAgentError("DATASET_NOT_FOUND", "Dataset not found", status_code=404)
        return dataset

    @app.post("/api/datasets/{dataset_id}/activate", responses=ERROR_RESPONSES)
    def activate_dataset(dataset_id: int) -> dict[str, object]:
        return datasets.activate(dataset_id)

    @app.get("/api/datasets/{dataset_id}/symbols", responses=ERROR_RESPONSES)
    def dataset_symbols(dataset_id: int) -> list[str]:
        return datasets.symbols(dataset_id)

    @app.get("/api/market/prices", responses=ERROR_RESPONSES)
    def prices(
        request: Annotated[MarketSnapshotArguments, Depends()],
    ) -> dict[str, Any]:
        return tools.market_snapshot(**request.model_dump(mode="json", exclude_none=True))

    @app.post("/api/factors/analyze", responses=ERROR_RESPONSES)
    def analyze_factor(request: FactorRequest) -> dict[str, Any]:
        return tools.factor_snapshot(**request.model_dump(mode="json", exclude_none=True))

    @app.post("/api/backtests", responses=ERROR_RESPONSES)
    def run_backtest(request: BacktestRequest) -> dict[str, Any]:
        return tools.run_backtest(**request.model_dump(mode="json", exclude_none=True))

    @app.get("/api/backtests/{run_id}", responses=ERROR_RESPONSES)
    def get_backtest(run_id: int) -> dict[str, Any]:
        result = database.get_backtest(run_id)
        if result is None:
            raise QuantAgentError("BACKTEST_NOT_FOUND", "Backtest not found", status_code=404)
        return result

    @app.get("/api/backtests")
    def list_backtests(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return database.list_backtests(limit)

    @app.post("/api/agent/run", responses=ERROR_RESPONSES)
    def run_agent(request: AgentRequest) -> dict[str, Any]:
        try:
            values = request.model_dump(exclude={"query"}, mode="json", exclude_none=True)
            return agent.run(request.query, **values)
        except AgentUnavailableError as error:
            raise QuantAgentError("MODEL_NOT_CONFIGURED", str(error), status_code=503) from error
        except AgentExecutionError as error:
            raise QuantAgentError("UPSTREAM_MODEL_FAILURE", str(error), status_code=502) from error

    @app.get("/api/tasks")
    def tasks(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return database.list_tasks(limit)

    @app.get("/api/tasks/{task_id}", responses=ERROR_RESPONSES)
    def get_task(task_id: int) -> dict[str, Any]:
        task = database.get_task(task_id)
        if task is None:
            raise QuantAgentError("TASK_NOT_FOUND", "Task not found", status_code=404)
        return task

    static_dir = ROOT_DIR / "app" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
