from __future__ import annotations

import shutil
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.db import Database
from app.errors import QuantAgentError
from app.quant.market import MarketDataService, load_market_csv, market_quality

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def normalized_frame_hash(frame: pd.DataFrame) -> str:
    canonical = frame.sort_values(["date", "symbol"]).copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(
        index=False,
        columns=sorted(canonical.columns),
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class DatasetService:
    def __init__(self, database: Database, market: MarketDataService, root: Path) -> None:
        self.database = database
        self.market = market
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.restore_active()

    @staticmethod
    def _public(dataset: dict[str, object]) -> dict[str, object]:
        result = {key: value for key, value in dataset.items() if key != "storage_path"}
        result["content_hash"] = result.get("normalized_data_hash") or result.get("content_hash")
        return result

    def _path(self, dataset: dict[str, object]) -> Path:
        stored = Path(str(dataset["storage_path"]))
        path = stored if stored.is_absolute() else self.root / stored
        resolved = path.resolve()
        if self.root not in resolved.parents:
            raise QuantAgentError(
                "DATASET_STORAGE_INVALID",
                "Dataset storage path is outside the configured dataset root",
                status_code=409,
            )
        return resolved

    def _verified_frame(self, dataset: dict[str, object]) -> pd.DataFrame:
        path = self._path(dataset)
        if not path.is_file():
            raise QuantAgentError(
                "DATASET_FILE_MISSING",
                "Dataset storage file is missing",
                status_code=409,
                details={"dataset_id": dataset["id"]},
            )
        try:
            frame = pd.read_parquet(path)
            normalized = load_market_csv(
                BytesIO(frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"))
            )
        except Exception as error:
            raise QuantAgentError(
                "DATASET_FILE_DAMAGED",
                "Dataset storage file cannot be read",
                status_code=409,
                details={"dataset_id": dataset["id"]},
            ) from error
        actual_hash = normalized_frame_hash(normalized)
        expected_hash = dataset.get("normalized_data_hash") or dataset.get("content_hash")
        if expected_hash and actual_hash != expected_hash:
            raise QuantAgentError(
                "DATASET_HASH_MISMATCH",
                "Dataset normalized hash does not match its metadata",
                status_code=409,
                details={"dataset_id": dataset["id"]},
            )
        return normalized

    def restore_active(self) -> None:
        dataset_id = self.database.get_active_dataset_id()
        if dataset_id is None:
            return
        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            self.market.mark_degraded(f"Active dataset #{dataset_id} metadata is missing")
            return
        try:
            frame = self._verified_frame(dataset)
            self.market.replace(frame, self._public(dataset))
        except QuantAgentError as error:
            self.market.mark_degraded(f"{error.code}: {error.message}")

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
            raise QuantAgentError("EMPTY_UPLOAD", "CSV file is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise QuantAgentError("UPLOAD_TOO_LARGE", "CSV file exceeds the 50 MB limit", status_code=413)
        if Path(filename).suffix.lower() != ".csv":
            raise QuantAgentError("UNSUPPORTED_FILE_TYPE", "Only CSV files are supported")
        raw_hash = sha256(content).hexdigest()
        existing = self.database.find_dataset_by_hash(raw_hash)
        if existing:
            raise QuantAgentError(
                "DUPLICATE_DATASET",
                f"The same file already exists as dataset #{existing['id']}",
                status_code=409,
            )

        frame = load_market_csv(BytesIO(content))
        normalized_hash = normalized_frame_hash(frame)
        quality = market_quality(frame)
        key = uuid4().hex
        directory = self.root / key
        directory.mkdir(parents=False, exist_ok=False)
        raw_path = directory / "original.csv"
        parquet_path = directory / "market.parquet"
        dataset_id: int | None = None
        previous_active = self.database.get_active_dataset_id()
        try:
            raw_path.write_bytes(content)
            frame.to_parquet(parquet_path, index=False)
            stored_frame = pd.read_parquet(parquet_path)
            if normalized_frame_hash(stored_frame) != normalized_hash:
                raise RuntimeError("Parquet verification hash mismatch")
            metadata = {
                "name": name.strip() or Path(filename).stem,
                "market": market.strip().upper() or "UNKNOWN",
                "adjustment": adjustment.strip().lower() or "unknown",
                "source": source.strip() or "upload",
                "original_filename": Path(filename).name,
                "storage_path": str(Path(key) / "market.parquet"),
                "raw_file_hash": raw_hash,
                "normalized_data_hash": normalized_hash,
                "start_date": frame["date"].min().strftime("%Y-%m-%d"),
                "end_date": frame["date"].max().strftime("%Y-%m-%d"),
                "symbol_count": int(frame["symbol"].nunique()),
                "row_count": int(len(frame)),
                "quality": quality,
            }
            dataset_id = self.database.create_dataset(metadata)
            dataset = self.database.get_dataset(dataset_id)
            if dataset is None:
                raise RuntimeError("Dataset metadata was not persisted")
            if activate:
                self.market.replace(frame, self._public(dataset))
                self.database.set_active_dataset(dataset_id)
                dataset["active"] = True
            else:
                dataset["active"] = False
            return self._public(dataset)
        except Exception:
            if dataset_id is not None:
                self.database.set_active_dataset(previous_active)
                self.database.delete_dataset(dataset_id)
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def list(self) -> list[dict[str, object]]:
        return [self._public(item) for item in self.database.list_datasets()]

    def get(self, dataset_id: int) -> dict[str, object] | None:
        dataset = self.database.get_dataset(dataset_id)
        return self._public(dataset) if dataset else None

    def activate(self, dataset_id: int) -> dict[str, object]:
        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            raise QuantAgentError("DATASET_NOT_FOUND", "Dataset not found", status_code=404)
        frame = self._verified_frame(dataset)
        self.market.replace(frame, self._public(dataset))
        self.database.set_active_dataset(dataset_id)
        dataset["active"] = True
        return self._public(dataset)

    def symbols(self, dataset_id: int | None = None) -> list[str]:
        if dataset_id is None or dataset_id == self.market.dataset.get("id"):
            return self.market.symbols()
        dataset = self.database.get_dataset(dataset_id)
        if dataset is None:
            raise QuantAgentError("DATASET_NOT_FOUND", "Dataset not found", status_code=404)
        return sorted(self._verified_frame(dataset)["symbol"].unique().tolist())
