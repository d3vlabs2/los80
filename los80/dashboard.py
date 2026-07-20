from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DashboardSummary:
    queue: int
    progress: int
    transcription_progress: int
    upscaling_progress: int
    encoding_progress: int
    current_encoder: str
    eta: str
    gpu: str
    completed: int
    failed: int
    skipped: int
    download_progress: int
    upload_progress: int
    transfer_speed: str
    completed_transfers: int
    skipped_transfers: int
    failed_transfers: int


class Dashboard:
    def __init__(self, jobs: Iterable[dict[str, object]]) -> None:
        self.jobs = list(jobs)

    def summary(self) -> DashboardSummary:
        completed = sum(1 for job in self.jobs if str(job.get("status", "")).lower() == "completed")
        failed = sum(1 for job in self.jobs if str(job.get("status", "")).lower() == "failed")
        skipped = sum(1 for job in self.jobs if str(job.get("status", "")).lower() == "skipped")
        queue = max(len(self.jobs) - completed - failed - skipped, 0)
        transcription_completed = sum(
            1
            for job in self.jobs
            if isinstance(job.get("stages", {}), dict)
            and str(job.get("stages", {}).get("subtitles", "")).lower() == "completed"
        )
        upscaling_completed = sum(
            1
            for job in self.jobs
            if isinstance(job.get("stages", {}), dict)
            and str(job.get("stages", {}).get("upscaling", "")).lower() == "completed"
        )
        encoding_completed = sum(
            1
            for job in self.jobs
            if isinstance(job.get("stages", {}), dict)
            and str(job.get("stages", {}).get("encoding", "")).lower() == "completed"
        )
        return DashboardSummary(
            queue=queue,
            progress=int((completed / len(self.jobs) * 100) if self.jobs else 0),
            transcription_progress=int((transcription_completed / len(self.jobs) * 100) if self.jobs else 0),
            upscaling_progress=int((upscaling_completed / len(self.jobs) * 100) if self.jobs else 0),
            encoding_progress=int((encoding_completed / len(self.jobs) * 100) if self.jobs else 0),
            current_encoder="libx265",
            eta="n/a",
            gpu="cpu",
            completed=completed,
            failed=failed,
            skipped=skipped,
            download_progress=0,
            upload_progress=0,
            transfer_speed="0 B/s",
            completed_transfers=0,
            skipped_transfers=0,
            failed_transfers=0,
        )
