from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _add_missing(
        connection: sqlite3.Connection, table: str, additions: dict[str, str]
    ) -> None:
        columns = Database._columns(connection, table)
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL DEFAULT '[]',
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
                    raw_file_hash TEXT,
                    normalized_data_hash TEXT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    symbol_count INTEGER NOT NULL,
                    row_count INTEGER NOT NULL,
                    quality_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    series_json TEXT NOT NULL,
                    methodology_json TEXT NOT NULL DEFAULT '{}',
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    dataset_id INTEGER REFERENCES datasets(id) ON DELETE RESTRICT,
                    dataset_hash TEXT,
                    symbols_json TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    code_version TEXT
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    round_number INTEGER NOT NULL,
                    call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    result_summary_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    backtest_id INTEGER REFERENCES backtest_runs(id) ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_backtests_created ON backtest_runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id, id);
                """
            )
            self._add_missing(
                connection,
                "datasets",
                {"raw_file_hash": "TEXT", "normalized_data_hash": "TEXT"},
            )
            self._add_missing(
                connection,
                "backtest_runs",
                {
                    "dataset_id": "INTEGER",
                    "dataset_hash": "TEXT",
                    "symbols_json": "TEXT",
                    "start_date": "TEXT",
                    "end_date": "TEXT",
                    "code_version": "TEXT",
                    "methodology_json": "TEXT NOT NULL DEFAULT '{}'",
                    "provenance_json": "TEXT NOT NULL DEFAULT '{}'",
                },
            )
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS backtest_dataset_fk_insert
                BEFORE INSERT ON backtest_runs
                WHEN NEW.dataset_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM datasets WHERE id = NEW.dataset_id)
                BEGIN SELECT RAISE(ABORT, 'unknown dataset_id'); END;
                CREATE TRIGGER IF NOT EXISTS backtest_dataset_fk_update
                BEFORE UPDATE OF dataset_id ON backtest_runs
                WHEN NEW.dataset_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM datasets WHERE id = NEW.dataset_id)
                BEGIN SELECT RAISE(ABORT, 'unknown dataset_id'); END;
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def create_task(self, query: str, plan: list[dict[str, Any]]) -> int:
        timestamp = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks(query,status,plan_json,created_at,updated_at) VALUES (?,'running',?,?,?)",
                (query, json.dumps(plan, ensure_ascii=False), timestamp, timestamp),
            )
            return int(cursor.lastrowid)

    def finish_task(self, task_id: int, result: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='completed',result_json=?,updated_at=? WHERE id=?",
                (json.dumps(result, ensure_ascii=False), utc_now(), task_id),
            )

    def update_task_plan(self, task_id: int, plan: list[dict[str, Any]]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET plan_json=?,updated_at=? WHERE id=?",
                (json.dumps(plan, ensure_ascii=False), utc_now(), task_id),
            )

    def fail_task(self, task_id: int, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='failed',error=?,updated_at=? WHERE id=?",
                (error, utc_now(), task_id),
            )

    def _task(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["plan"] = json.loads(item.pop("plan_json") or "[]")
        result_json = item.pop("result_json")
        item["result"] = json.loads(result_json) if result_json else None
        item["tool_calls"] = self.list_tool_calls(item["id"])
        return item

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._task(row) for row in rows]

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def start_tool_call(
        self,
        task_id: int,
        round_number: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_calls(
                    task_id,round_number,call_id,tool_name,arguments_json,started_at,status
                ) VALUES (?,?,?,?,?,?,'running')
                """,
                (
                    task_id,
                    round_number,
                    call_id,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False) if arguments is not None else None,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def finish_tool_call(
        self,
        record_id: int,
        *,
        duration_ms: float,
        summary: dict[str, Any] | None = None,
        backtest_id: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        status = "failed" if error_code else "completed"
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE tool_calls SET finished_at=?,duration_ms=?,status=?,
                    result_summary_json=?,error_code=?,error_message=?,backtest_id=?
                WHERE id=?
                """,
                (
                    utc_now(),
                    round(duration_ms, 3),
                    status,
                    json.dumps(summary, ensure_ascii=False) if summary is not None else None,
                    error_code,
                    error_message,
                    backtest_id,
                    record_id,
                ),
            )

    def list_tool_calls(self, task_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_calls WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            arguments = item.pop("arguments_json")
            summary = item.pop("result_summary_json")
            item["arguments"] = json.loads(arguments) if arguments else None
            item["result_summary"] = json.loads(summary) if summary else None
            result.append(item)
        return result

    def create_dataset(self, metadata: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO datasets(
                    name,market,adjustment,source,original_filename,storage_path,
                    content_hash,raw_file_hash,normalized_data_hash,start_date,end_date,
                    symbol_count,row_count,quality_json,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    metadata["name"], metadata["market"], metadata["adjustment"],
                    metadata["source"], metadata["original_filename"], metadata["storage_path"],
                    metadata["raw_file_hash"], metadata["raw_file_hash"],
                    metadata["normalized_data_hash"], metadata["start_date"], metadata["end_date"],
                    metadata["symbol_count"], metadata["row_count"],
                    json.dumps(metadata["quality"], ensure_ascii=False), utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _dataset_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["quality"] = json.loads(item.pop("quality_json"))
        item["raw_file_hash"] = item.get("raw_file_hash") or item.get("content_hash")
        item["normalized_data_hash"] = item.get("normalized_data_hash") or item.get("content_hash")
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
                "SELECT * FROM datasets WHERE content_hash=? OR raw_file_hash=?",
                (content_hash, content_hash),
            ).fetchone()
        return self._dataset_row(row) if row else None

    def delete_dataset(self, dataset_id: int) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))

    def set_active_dataset(self, dataset_id: int | None) -> None:
        value = "" if dataset_id is None else str(dataset_id)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO app_state(key,value) VALUES ('active_dataset_id',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (value,),
            )

    def get_active_dataset_id(self) -> int | None:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM app_state WHERE key='active_dataset_id'").fetchone()
        return int(row["value"]) if row and row["value"] else None

    def save_backtest(
        self,
        strategy: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        series: list[dict[str, Any]],
        provenance: dict[str, Any] | None = None,
        methodology: dict[str, Any] | None = None,
    ) -> int:
        provenance = provenance or {}
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backtest_runs(
                    strategy,parameters_json,metrics_json,series_json,methodology_json,
                    provenance_json,created_at,dataset_id,dataset_hash,symbols_json,
                    start_date,end_date,code_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    strategy, json.dumps(parameters, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False), json.dumps(series, ensure_ascii=False),
                    json.dumps(methodology or {}, ensure_ascii=False),
                    json.dumps(provenance, ensure_ascii=False), utc_now(),
                    provenance.get("dataset_id"), provenance.get("normalized_data_hash"),
                    json.dumps(provenance.get("symbols", []), ensure_ascii=False),
                    provenance.get("start_date"), provenance.get("end_date"),
                    provenance.get("code_version", "unknown"),
                ),
            )
            return int(cursor.lastrowid)

    def get_backtest(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM backtest_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        provenance = json.loads(row["provenance_json"] or "{}")
        if not provenance:
            provenance = {
                "dataset_id": row["dataset_id"], "normalized_data_hash": row["dataset_hash"],
                "symbols": json.loads(row["symbols_json"] or "[]"), "start_date": row["start_date"],
                "end_date": row["end_date"], "code_version": row["code_version"],
            }
        return {
            "id": row["id"], "strategy": row["strategy"],
            "parameters": json.loads(row["parameters_json"]),
            "metrics": json.loads(row["metrics_json"]), "series": json.loads(row["series_json"]),
            "methodology": json.loads(row["methodology_json"] or "{}"),
            "provenance": provenance, "created_at": row["created_at"],
        }

    def list_backtests(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,strategy,parameters_json,metrics_json,provenance_json,dataset_id,"
                "dataset_hash,start_date,end_date,code_version,created_at "
                "FROM backtest_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"], "strategy": row["strategy"],
                "parameters": json.loads(row["parameters_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "provenance": json.loads(row["provenance_json"] or "{}"),
                "dataset_id": row["dataset_id"], "dataset_hash": row["dataset_hash"],
                "start_date": row["start_date"], "end_date": row["end_date"],
                "code_version": row["code_version"], "created_at": row["created_at"],
            }
            for row in rows
        ]
