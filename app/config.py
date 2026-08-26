from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    database_path: Path
    random_seed: int
    dataset_path: Path | None = None
    llm_provider: str = "deepseek"
    llm_protocol: str = "responses"
    llm_api_key: str | None = None
    llm_model: str = "deepseek-v4-flash"
    llm_base_url: str = "https://api.deepseek.com"
    llm_max_tool_rounds: int = 6


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def load_settings() -> Settings:
    database_value = os.getenv("QUANTAGENT_DB_PATH", "data/quantagent.db")
    database_path = Path(database_value)
    if not database_path.is_absolute():
        database_path = ROOT_DIR / database_path
    dataset_value = os.getenv("QUANTAGENT_DATASET_PATH", "data/datasets")
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path
    return Settings(
        database_path=database_path,
        random_seed=int(os.getenv("QUANTAGENT_SEED", "20260825")),
        dataset_path=dataset_path,
        llm_provider=_first_env("LLM_PROVIDER", default="deepseek") or "deepseek",
        llm_protocol=_first_env("LLM_PROTOCOL", default="responses") or "responses",
        llm_api_key=_first_env("LLM_API_KEY", "DEEPSEEK_API_KEY"),
        llm_model=_first_env(
            "LLM_MODEL", "DEEPSEEK_MODEL", default="deepseek-v4-flash"
        )
        or "deepseek-v4-flash",
        llm_base_url=_first_env(
            "LLM_BASE_URL", "DEEPSEEK_BASE_URL", default="https://api.deepseek.com"
        )
        or "https://api.deepseek.com",
        llm_max_tool_rounds=int(
            _first_env("LLM_MAX_TOOL_ROUNDS", "DEEPSEEK_MAX_TOOL_ROUNDS", default="6")
            or "6"
        ),
    )
