from __future__ import annotations

import time
from pathlib import Path

import pytest

from los80.database import JobDatabase
from los80.performance import (ResourceMonitor, StageScheduler, adaptive_batch_size,
                               cache_key, cleanup_job_directory, cleanup_temporary_files,
                               profile_action)


def test_cache_invalidates_for_source_or_configuration(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.sqlite")
    key, digest = cache_key("translation", "source-a", {"target": "en"})
    db.put_cache(key, "source-a", "translation", digest, {"output": "a.srt"})
    assert db.get_cache(key) == {"output": "a.srt"}
    changed_source, _ = cache_key("translation", "source-b", {"target": "en"})
    changed_config, _ = cache_key("translation", "source-a", {"target": "fr"})
    assert db.get_cache(changed_source) is None
    assert db.get_cache(changed_config) is None


def test_lease_prevents_duplicate_processing_and_expired_work_recovers(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.sqlite")
    db.add_job("video.mp4", "/tmp/video.mp4")
    assert db.acquire_lease("video.mp4", "worker-a", ttl_seconds=60)
    assert not db.acquire_lease("video.mp4", "worker-b", ttl_seconds=60)
    db.mark_stage("video.mp4", "encoding", "in_progress")
    assert db.recover_interrupted() == 1
    assert db.get_stages("video.mp4")["encoding"] == "pending"
    db.release_lease("video.mp4", "worker-a")
    assert db.acquire_lease("video.mp4", "worker-b", ttl_seconds=60)


def test_scheduler_separates_queues_and_rejects_duplicate_work() -> None:
    scheduler = StageScheduler(cpu_workers=2, io_workers=2)
    gate = scheduler.submit("gpu", "video:gpu", lambda: (time.sleep(.02), "gpu")[1])
    with pytest.raises(RuntimeError, match="already scheduled"):
        scheduler.submit("gpu", "video:gpu", lambda: None)
    cpu = scheduler.submit("cpu", "video:cpu", lambda: "cpu")
    io = scheduler.submit("io", "video:io", lambda: "io")
    assert {gate.result(), cpu.result(), io.result()} == {"gpu", "cpu", "io"}
    scheduler.close()


def test_resource_monitor_batching_cleanup_and_profiling(tmp_path: Path) -> None:
    monitor = ResourceMonitor()
    snapshot = monitor.snapshot(tmp_path)
    assert snapshot["disk_free_bytes"] > 0
    assert adaptive_batch_size(8 * 1024**3, configured=1, maximum=8) == 4
    stale = tmp_path / "old.tmp"
    stale.write_bytes(b"temporary")
    assert cleanup_temporary_files(tmp_path, max_age_seconds=0) == 1
    job_dir = tmp_path / "working" / "video.mp4"
    job_dir.mkdir(parents=True)
    (job_dir / "intermediate.bin").write_bytes(b"data")
    assert cleanup_job_directory(tmp_path / "working", "video.mp4")
    assert not job_dir.exists()
    outputs = profile_action(lambda: sum(range(100)), tmp_path / "profile")
    assert set(outputs) == {"pstats", "flamegraph", "slowest_functions"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())


def test_benchmark_storage_regression(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.sqlite")
    db.add_job("video.mp4", "/tmp/video.mp4")
    started = time.perf_counter()
    for _ in range(200):
        db.record_benchmark("video.mp4", "encoding", .1,
                            units_processed=3, units_name="frames",
                            cpu_percent=25, ram_bytes=1024)
    elapsed = time.perf_counter() - started
    summary = db.performance_summary()
    assert summary["stages"]["encoding"]["runs"] == 200
    assert summary["encoder_efficiency_fps"] == 30
    assert elapsed < 5.0
