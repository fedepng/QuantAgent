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
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            UNIQUE(document_id, chunk_index)
        );
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
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_backtests_created ON backtest_runs(created_at DESC);
        """
        with self.connect() as connection:
            connection.executescript(schema)

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

    def save_backtest(
        self,
        strategy: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        series: list[dict[str, Any]],
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backtest_runs(
                    strategy, parameters_json, metrics_json, series_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    strategy,
                    json.dumps(parameters, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    json.dumps(series, ensure_ascii=False),
                    utc_now(),
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
            "created_at": row["created_at"],
        }

