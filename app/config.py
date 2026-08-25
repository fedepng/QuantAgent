from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_path: Path
    random_seed: int
    embedding_dim: int


def load_settings() -> Settings:
    database_value = os.getenv("QUANTAGENT_DB_PATH", "data/quantagent.db")
    database_path = Path(database_value)
    if not database_path.is_absolute():
        database_path = ROOT_DIR / database_path
    return Settings(
        database_path=database_path,
        random_seed=int(os.getenv("QUANTAGENT_SEED", "20260825")),
        embedding_dim=int(os.getenv("QUANTAGENT_EMBEDDING_DIM", "384")),
    )

