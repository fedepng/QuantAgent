from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from app.db import Database
from app.quant.market import MarketDataService, load_market_csv, market_quality


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class DatasetService:
    def __init__(self, database: Database, market: MarketDataService, root: Path) -> None:
        self.database = database
        self.market = market
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.restore_active()

    @staticmethod
    def _public(dataset: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in dataset.items() if key != "storage_path"}

    def restore_active(self) -> None:
        dataset_id = self.database.get_active_dataset_id()
        if dataset_id is None:
            return
        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            self.database.set_active_dataset(None)
            return
        path = Path(str(dataset["storage_path"]))
        if not path.exists():
            self.database.set_active_dataset(None)
            return
        self.market.replace(pd.read_parquet(path), self._public(dataset))

    def import_csv(
        self,
        content: bytes,
        filename: str,
        name: str,
        market: str,
        adjustment: str,
        source: str,
        activate: bool = True,
    ) -> dict[str, object]:
        if not content:
            raise ValueError("CSV file is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("CSV file exceeds the 50 MB limit")
        if Path(filename).suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")
        content_hash = sha256(content).hexdigest()
        existing = self.database.find_dataset_by_hash(content_hash)
        if existing:
            raise ValueError(f"The same file already exists as dataset #{existing['id']}")

        frame = load_market_csv(BytesIO(content))
        quality = market_quality(frame)
        key = uuid4().hex
        directory = self.root / key
        directory.mkdir(parents=False, exist_ok=False)
        raw_path = directory / "original.csv"
        parquet_path = directory / "market.parquet"
        try:
            raw_path.write_bytes(content)
            frame.to_parquet(parquet_path, index=False)
            metadata = {
                "name": name.strip() or Path(filename).stem,
                "market": market.strip().upper() or "UNKNOWN",
                "adjustment": adjustment.strip().lower() or "unknown",
                "source": source.strip() or "upload",
                "original_filename": Path(filename).name,
                "storage_path": str(parquet_path.resolve()),
                "content_hash": content_hash,
                "start_date": frame["date"].min().strftime("%Y-%m-%d"),
                "end_date": frame["date"].max().strftime("%Y-%m-%d"),
                "symbol_count": int(frame["symbol"].nunique()),
                "row_count": int(len(frame)),
                "quality": quality,
            }
            dataset_id = self.database.create_dataset(metadata)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            raise RuntimeError("Dataset metadata was not persisted")
        if activate:
            self.activate(dataset_id)
            dataset["active"] = True
        else:
            dataset["active"] = False
        return self._public(dataset)

    def list(self) -> list[dict[str, object]]:
        return [self._public(item) for item in self.database.list_datasets()]

    def get(self, dataset_id: int) -> dict[str, object] | None:
        dataset = self.database.get_dataset(dataset_id)
        return self._public(dataset) if dataset else None

    def activate(self, dataset_id: int) -> dict[str, object]:
        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        path = Path(str(dataset["storage_path"]))
        if not path.exists():
            raise FileNotFoundError("Dataset storage file is missing")
        self.market.replace(pd.read_parquet(path), self._public(dataset))
        self.database.set_active_dataset(dataset_id)
        dataset["active"] = True
        return self._public(dataset)

    def symbols(self, dataset_id: int | None = None) -> list[str]:
        if dataset_id is None or dataset_id == self.market.dataset.get("id"):
            return self.market.symbols()
        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return sorted(pd.read_parquet(str(dataset["storage_path"]), columns=["symbol"])["symbol"].unique().tolist())
