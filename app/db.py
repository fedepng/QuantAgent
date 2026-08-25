from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            status TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            series_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            adjustment TEXT NOT NULL,
            source TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            storage_path TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL UNIQUE,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            symbol_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            quality_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_backtests_created ON backtest_runs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at DESC);
        """
        with self.connect() as connection:
            connection.executescript(schema)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(backtest_runs)").fetchall()
            }
            additions = {
                "dataset_id": "INTEGER",
                "dataset_hash": "TEXT",
                "symbols_json": "TEXT",
                "start_date": "TEXT",
                "end_date": "TEXT",
                "code_version": "TEXT",
            }
            for name, column_type in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE backtest_runs ADD COLUMN {name} {column_type}"
                    )

    def create_task(self, query: str, plan: list[dict[str, Any]]) -> int:
        timestamp = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks(query, status, plan_json, created_at, updated_at)
                VALUES (?, 'running', ?, ?, ?)
                """,
                (query, json.dumps(plan, ensure_ascii=False), timestamp, timestamp),
            )
            return int(cursor.lastrowid)

    def finish_task(self, task_id: int, result: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='completed', result_json=?, updated_at=? WHERE id=?",
                (json.dumps(result, ensure_ascii=False), utc_now(), task_id),
            )

    def update_task_plan(self, task_id: int, plan: list[dict[str, Any]]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET plan_json=?, updated_at=? WHERE id=?",
                (json.dumps(plan, ensure_ascii=False), utc_now(), task_id),
            )

    def fail_task(self, task_id: int, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='failed', error=?, updated_at=? WHERE id=?",
                (error, utc_now(), task_id),
            )

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["plan"] = json.loads(item.pop("plan_json"))
            result_json = item.pop("result_json")
            item["result"] = json.loads(result_json) if result_json else None
            result.append(item)
        return result

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["plan"] = json.loads(item.pop("plan_json"))
        result_json = item.pop("result_json")
        item["result"] = json.loads(result_json) if result_json else None
        return item

    def create_dataset(self, metadata: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO datasets(
                    name, market, adjustment, source, original_filename, storage_path,
                    content_hash, start_date, end_date, symbol_count, row_count,
                    quality_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["name"], metadata["market"], metadata["adjustment"],
                    metadata["source"], metadata["original_filename"],
                    metadata["storage_path"], metadata["content_hash"],
                    metadata["start_date"], metadata["end_date"],
                    metadata["symbol_count"], metadata["row_count"],
                    json.dumps(metadata["quality"], ensure_ascii=False), utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _dataset_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["quality"] = json.loads(item.pop("quality_json"))
        return item

    def list_datasets(self) -> list[dict[str, Any]]:
        active_id = self.get_active_dataset_id()
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
        result = []
        for row in rows:
            item = self._dataset_row(row)
            item["active"] = item["id"] == active_id
            result.append(item)
        return result

    def get_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._dataset_row(row) if row else None

    def find_dataset_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE content_hash=?", (content_hash,)
            ).fetchone()
        return self._dataset_row(row) if row else None

    def set_active_dataset(self, dataset_id: int | None) -> None:
        value = "" if dataset_id is None else str(dataset_id)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO app_state(key, value) VALUES ('active_dataset_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (value,),
            )

    def get_active_dataset_id(self) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_state WHERE key='active_dataset_id'"
            ).fetchone()
        if row is None or not row["value"]:
            return None
        return int(row["value"])

    def save_backtest(
        self,
        strategy: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        series: list[dict[str, Any]],
        provenance: dict[str, Any] | None = None,
    ) -> int:
        provenance = provenance or {}
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backtest_runs(
                    strategy, parameters_json, metrics_json, series_json, created_at,
                    dataset_id, dataset_hash, symbols_json, start_date, end_date, code_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy,
                    json.dumps(parameters, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    json.dumps(series, ensure_ascii=False),
                    utc_now(),
                    provenance.get("dataset_id"),
                    provenance.get("dataset_hash"),
                    json.dumps(provenance.get("symbols", []), ensure_ascii=False),
                    provenance.get("start_date"),
                    provenance.get("end_date"),
                    provenance.get("code_version", "2.0.0"),
                ),
            )
            return int(cursor.lastrowid)

    def get_backtest(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM backtest_runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "strategy": row["strategy"],
            "parameters": json.loads(row["parameters_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "series": json.loads(row["series_json"]),
            "provenance": {
                "dataset_id": row["dataset_id"],
                "dataset_hash": row["dataset_hash"],
                "symbols": json.loads(row["symbols_json"] or "[]"),
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "code_version": row["code_version"],
            },
            "created_at": row["created_at"],
        }

    def list_backtests(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, strategy, parameters_json, metrics_json, dataset_id, "
                "dataset_hash, start_date, end_date, code_version, created_at "
                "FROM backtest_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "strategy": row["strategy"],
                "parameters": json.loads(row["parameters_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "dataset_id": row["dataset_id"],
                "dataset_hash": row["dataset_hash"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "code_version": row["code_version"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
