from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    id: int
    source_name: str
    source_path: str
    status: str
    stages: dict[str, str]


class JobDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT UNIQUE NOT NULL,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_stages (
                    job_id INTEGER NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (job_id, stage_name)
                )
                """
            )
            conn.commit()

    def add_job(self, source_name: str, source_path: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM jobs WHERE source_name = ?",
                (source_name,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                return existing[0]
            cursor = conn.execute(
                "INSERT INTO jobs (source_name, source_path, status) VALUES (?, ?, ?)",
                (source_name, source_path, "pending"),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def mark_stage(self, source_name: str, stage_name: str, status: str, details: str | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            job_id = self._get_job_id(conn, source_name)
            if job_id is None:
                raise ValueError(f"Job {source_name} not found")
            conn.execute(
                """
                INSERT INTO job_stages (job_id, stage_name, status, details)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id, stage_name) DO UPDATE SET
                    status = excluded.status,
                    details = excluded.details,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (job_id, stage_name, status, details),
            )
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                ("processing" if status in {"in_progress", "retrying"} else "completed", job_id),
            )
            conn.commit()

    def get_stages(self, source_name: str) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT stage_name, status FROM job_stages WHERE job_id = (SELECT id FROM jobs WHERE source_name = ?)",
                (source_name,),
            ).fetchall()
        return {stage_name: status for stage_name, status in rows}

    def get_job(self, source_name: str) -> JobRecord:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, source_name, source_path, status FROM jobs WHERE source_name = ?",
                (source_name,),
            ).fetchone()
        if row is None:
            raise KeyError(source_name)
        return JobRecord(id=row[0], source_name=row[1], source_path=row[2], status=row[3], stages=self.get_stages(source_name))

    def _get_all_job_names(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT source_name FROM jobs ORDER BY id").fetchall()
        return [row[0] for row in rows]

    def _get_job_id(self, conn: sqlite3.Connection, source_name: str) -> int | None:
        row = conn.execute("SELECT id FROM jobs WHERE source_name = ?", (source_name,)).fetchone()
        return int(row[0]) if row else None
