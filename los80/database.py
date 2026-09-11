from __future__ import annotations

import sqlite3
import json
import time
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
                """CREATE TABLE IF NOT EXISTS stage_cache (
                    cache_key TEXT PRIMARY KEY, source_fingerprint TEXT NOT NULL,
                    stage_name TEXT NOT NULL, config_hash TEXT NOT NULL,
                    value_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_hit_at TEXT, hit_count INTEGER NOT NULL DEFAULT 0
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER,
                    stage_name TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
                    units_processed REAL, units_name TEXT, rate REAL,
                    cpu_percent REAL, ram_bytes INTEGER, gpu_utilization REAL,
                    gpu_vram_bytes INTEGER, cache_hit INTEGER NOT NULL DEFAULT 0,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS job_leases (
                    job_id INTEGER PRIMARY KEY, owner TEXT NOT NULL,
                    expires_at REAL NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media_analysis (
                    job_id INTEGER PRIMARY KEY,
                    source_fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    scenes_json TEXT NOT NULL,
                    detected_issues_json TEXT NOT NULL DEFAULT '[]',
                    recommendations_json TEXT NOT NULL,
                    processing_decisions_json TEXT NOT NULL DEFAULT '{}',
                    quality_score REAL NOT NULL,
                    previews_json TEXT NOT NULL DEFAULT '{}',
                    analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(media_analysis)")}
            if "detected_issues_json" not in columns:
                conn.execute("ALTER TABLE media_analysis ADD COLUMN detected_issues_json TEXT NOT NULL DEFAULT '[]'")
            if "processing_decisions_json" not in columns:
                conn.execute("ALTER TABLE media_analysis ADD COLUMN processing_decisions_json TEXT NOT NULL DEFAULT '{}'")
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
            if status in {"in_progress", "retrying"}:
                job_status = "processing"
            elif status == "failed":
                job_status = "failed"
            elif stage_name == "completed" and status == "completed":
                job_status = "completed"
            else:
                job_status = None
            if job_status is not None:
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_status, job_id),
                )
            conn.commit()

    def mark_job_status(self, source_name: str, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            job_id = self._get_job_id(conn, source_name)
            if job_id is None:
                raise ValueError(f"Job {source_name} not found")
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, job_id),
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

    def save_analysis(self, source_name: str, analysis: dict[str, Any]) -> None:
        """Insert or replace the complete, versionable analysis for a job."""
        with sqlite3.connect(self.db_path) as conn:
            job_id = self._get_job_id(conn, source_name)
            if job_id is None:
                raise ValueError(f"Job {source_name} not found")
            conn.execute(
                """
                INSERT INTO media_analysis
                    (job_id, source_fingerprint, metadata_json, scenes_json,
                     detected_issues_json, recommendations_json, processing_decisions_json,
                     quality_score, previews_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    source_fingerprint=excluded.source_fingerprint,
                    metadata_json=excluded.metadata_json,
                    scenes_json=excluded.scenes_json,
                    detected_issues_json=excluded.detected_issues_json,
                    recommendations_json=excluded.recommendations_json,
                    processing_decisions_json=excluded.processing_decisions_json,
                    quality_score=excluded.quality_score,
                    previews_json=excluded.previews_json,
                    analyzed_at=CURRENT_TIMESTAMP
                """,
                (job_id, analysis["source_fingerprint"],
                 json.dumps(analysis.get("metadata", {})),
                 json.dumps(analysis.get("scenes", {})),
                 json.dumps(analysis.get("detected_issues", [])),
                 json.dumps(analysis.get("recommendations", {})),
                 json.dumps(analysis.get("processing_decisions", analysis.get("recommendations", {}))),
                 float(analysis.get("quality_score", 0)),
                 json.dumps(analysis.get("previews", {}))),
            )
            conn.commit()

    def get_analysis(self, source_name: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT source_fingerprint, metadata_json, scenes_json,
                          detected_issues_json, recommendations_json, processing_decisions_json,
                          quality_score, previews_json,
                          analyzed_at
                   FROM media_analysis
                   WHERE job_id=(SELECT id FROM jobs WHERE source_name=?)""",
                (source_name,),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_fingerprint": row[0], "metadata": json.loads(row[1]),
            "scenes": json.loads(row[2]), "detected_issues": json.loads(row[3]),
            "recommendations": json.loads(row[4]), "processing_decisions": json.loads(row[5]),
            "quality_score": row[6], "previews": json.loads(row[7]),
            "analyzed_at": row[8],
        }

    def has_analysis(self, source_name: str, source_fingerprint: str | None = None) -> bool:
        result = self.get_analysis(source_name)
        return result is not None and (source_fingerprint is None or result["source_fingerprint"] == source_fingerprint)

    def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value_json FROM stage_cache WHERE cache_key=?", (cache_key,)).fetchone()
            if row:
                conn.execute("UPDATE stage_cache SET hit_count=hit_count+1,last_hit_at=CURRENT_TIMESTAMP WHERE cache_key=?", (cache_key,))
                conn.commit()
        return json.loads(row[0]) if row else None

    def put_cache(self, cache_key: str, fingerprint: str, stage_name: str,
                  config_hash: str, value: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO stage_cache(cache_key,source_fingerprint,stage_name,config_hash,value_json)
                   VALUES(?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET
                   value_json=excluded.value_json,created_at=CURRENT_TIMESTAMP,last_hit_at=NULL,hit_count=0""",
                (cache_key, fingerprint, stage_name, config_hash, json.dumps(value)),
            )
            conn.commit()

    def acquire_lease(self, source_name: str, owner: str, ttl_seconds: float = 3600) -> bool:
        now = time.time()
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("BEGIN IMMEDIATE")
            job_id = self._get_job_id(conn, source_name)
            if job_id is None:
                raise ValueError(f"Job {source_name} not found")
            current = conn.execute("SELECT owner,expires_at FROM job_leases WHERE job_id=?", (job_id,)).fetchone()
            if current and current[0] != owner and float(current[1]) > now:
                conn.rollback()
                return False
            conn.execute(
                """INSERT INTO job_leases(job_id,owner,expires_at) VALUES(?,?,?)
                   ON CONFLICT(job_id) DO UPDATE SET owner=excluded.owner,
                   expires_at=excluded.expires_at,updated_at=CURRENT_TIMESTAMP""",
                (job_id, owner, now + ttl_seconds),
            )
            conn.commit()
        return True

    def release_lease(self, source_name: str, owner: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM job_leases WHERE job_id=(SELECT id FROM jobs WHERE source_name=?) AND owner=?", (source_name, owner))
            conn.commit()

    def recover_interrupted(self) -> int:
        """Make interrupted stages retryable while preserving completed stages."""
        with sqlite3.connect(self.db_path) as conn:
            interrupted_ids = [row[0] for row in conn.execute(
                "SELECT DISTINCT job_id FROM job_stages WHERE status IN ('in_progress','retrying')"
            )]
            cursor = conn.execute(
                "UPDATE job_stages SET status='pending',details='recovered after interruption',updated_at=CURRENT_TIMESTAMP WHERE status IN ('in_progress','retrying')"
            )
            if interrupted_ids:
                placeholders = ",".join("?" for _ in interrupted_ids)
                conn.execute(f"UPDATE jobs SET status='pending',updated_at=CURRENT_TIMESTAMP WHERE id IN ({placeholders})", interrupted_ids)
                conn.execute(f"DELETE FROM job_leases WHERE job_id IN ({placeholders})", interrupted_ids)
            conn.execute("DELETE FROM job_leases WHERE expires_at <= ?", (time.time(),))
            conn.commit()
            return cursor.rowcount

    def record_benchmark(self, source_name: str | None, stage_name: str,
                         elapsed_seconds: float, **metrics: Any) -> None:
        units = metrics.get("units_processed")
        rate = (float(units) / elapsed_seconds) if units is not None and elapsed_seconds > 0 else metrics.get("rate")
        with sqlite3.connect(self.db_path) as conn:
            job_id = self._get_job_id(conn, source_name) if source_name else None
            conn.execute(
                """INSERT INTO benchmark_history
                   (job_id,stage_name,elapsed_seconds,units_processed,units_name,rate,
                    cpu_percent,ram_bytes,gpu_utilization,gpu_vram_bytes,cache_hit,details_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job_id, stage_name, elapsed_seconds, units, metrics.get("units_name"), rate,
                 metrics.get("cpu_percent"), metrics.get("ram_bytes"), metrics.get("gpu_utilization"),
                 metrics.get("gpu_vram_bytes"), int(bool(metrics.get("cache_hit"))),
                 json.dumps(metrics.get("details", {}))),
            )
            conn.commit()

    def performance_summary(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT stage_name,COUNT(*),AVG(elapsed_seconds),AVG(rate),SUM(cache_hit) FROM benchmark_history GROUP BY stage_name"
            ).fetchall()
            totals = conn.execute("SELECT COUNT(*),COALESCE(SUM(cache_hit),0) FROM benchmark_history").fetchone()
            average_total = conn.execute("SELECT COALESCE(AVG(elapsed_seconds),0) FROM benchmark_history WHERE stage_name='total'").fetchone()[0]
            gpu_efficiency = conn.execute("SELECT AVG(gpu_utilization) FROM benchmark_history WHERE stage_name='upscaling'").fetchone()[0]
            encoder_efficiency = conn.execute("SELECT AVG(rate) FROM benchmark_history WHERE stage_name='encoding'").fetchone()[0]
        stages = {row[0]: {"runs": row[1], "average_seconds": round(row[2], 3),
                           "average_rate": round(row[3], 3) if row[3] is not None else None,
                           "cache_hits": row[4]} for row in rows}
        count, hits = totals or (0, 0)
        bottleneck = max(stages, key=lambda name: stages[name]["average_seconds"], default="n/a")
        return {"average_processing_time": round(average_total, 3),
                "bottleneck": bottleneck, "cache_hit_rate": round(hits / count, 3) if count else 0,
                "skipped_work": hits, "estimated_savings_seconds": round(sum(v["average_seconds"] * v["cache_hits"] for v in stages.values()), 3),
                "gpu_efficiency": round(gpu_efficiency, 2) if gpu_efficiency is not None else None,
                "encoder_efficiency_fps": round(encoder_efficiency, 2) if encoder_efficiency is not None else None,
                "stages": stages}
