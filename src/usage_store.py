from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def current_month_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def validate_month(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise ValueError("month must use YYYY-MM format") from exc
    return parsed.strftime("%Y-%m")


class MonthlyUsageStore:
    """Persistent per-company monthly request and provider token totals."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=10,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=10000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_usage (
                month_utc TEXT NOT NULL,
                company_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                requests INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    month_utc,
                    company_id,
                    provider,
                    model,
                    operation,
                    status
                )
            )
            """
        )
        self._connection.commit()

    def record(
        self,
        *,
        company_id: str,
        provider: str,
        model: str,
        operation: str,
        status: str = "success",
        requests: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        month_utc: str | None = None,
    ) -> None:
        month_utc = validate_month(month_utc or current_month_utc())
        values = (
            max(int(requests), 0),
            max(int(input_tokens), 0),
            max(int(output_tokens), 0),
            max(int(total_tokens), 0),
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO monthly_usage (
                    month_utc,
                    company_id,
                    provider,
                    model,
                    operation,
                    status,
                    requests,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    month_utc,
                    company_id,
                    provider,
                    model,
                    operation,
                    status
                )
                DO UPDATE SET
                    requests = requests + excluded.requests,
                    input_tokens = input_tokens + excluded.input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    total_tokens = total_tokens + excluded.total_tokens,
                    updated_at = excluded.updated_at
                """,
                (
                    month_utc,
                    company_id,
                    provider,
                    model,
                    operation,
                    status,
                    *values,
                    now,
                ),
            )
            self._connection.commit()

    def summary(
        self,
        company_id: str,
        month_utc: str | None = None,
    ) -> dict[str, Any]:
        month_utc = validate_month(month_utc or current_month_utc())
        with self._lock:
            rows = [
                dict(row)
                for row in self._connection.execute(
                    """
                    SELECT
                        provider,
                        model,
                        operation,
                        status,
                        requests,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        updated_at
                    FROM monthly_usage
                    WHERE company_id = ? AND month_utc = ?
                    ORDER BY provider, model, operation, status
                    """,
                    (company_id, month_utc),
                ).fetchall()
            ]
        return {
            "company_id": company_id,
            "month_utc": month_utc,
            "requests": sum(row["requests"] for row in rows),
            "searches": sum(
                row["requests"]
                for row in rows
                if row["operation"] == "search"
            ),
            "model_requests": sum(
                row["requests"]
                for row in rows
                if row["operation"] != "search"
            ),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "total_tokens": sum(row["total_tokens"] for row in rows),
            "breakdown": rows,
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
