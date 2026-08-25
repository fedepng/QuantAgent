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
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_max_tool_rounds: int = 6


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
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_max_tool_rounds=int(os.getenv("DEEPSEEK_MAX_TOOL_ROUNDS", "6")),
    )
