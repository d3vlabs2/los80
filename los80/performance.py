from __future__ import annotations

import cProfile
import hashlib
import json
import os
import pstats
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from los80.database import JobDatabase


def config_hash(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def cache_key(stage: str, fingerprint: str, settings: dict[str, Any]) -> tuple[str, str]:
    digest = config_hash(settings)
    return hashlib.sha256(f"{stage}:{fingerprint}:{digest}".encode()).hexdigest(), digest


class ResourceMonitor:
    def snapshot(self, path: str | Path = ".") -> dict[str, float | int | None]:
        disk = shutil.disk_usage(Path(path))
        cpu: float | None = None
        ram: int | None = None
        try:
            import psutil  # type: ignore
            cpu = float(psutil.cpu_percent(interval=None))
            ram = int(psutil.virtual_memory().used)
        except ImportError:
            if hasattr(os, "getloadavg"):
                cpu = min(100.0, os.getloadavg()[0] / max(os.cpu_count() or 1, 1) * 100)
        gpu, vram, vram_free = self._gpu_snapshot()
        return {"disk_free_bytes": disk.free, "disk_total_bytes": disk.total,
                "cpu_percent": cpu, "ram_bytes": ram,
                "gpu_utilization": gpu, "gpu_vram_bytes": vram,
                "gpu_vram_free_bytes": vram_free}

    @staticmethod
    def _gpu_snapshot() -> tuple[float | None, int | None, int | None]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                utilization, memory_mb, free_mb = result.stdout.splitlines()[0].split(",")
                return (float(utilization.strip()), int(float(memory_mb.strip()) * 1024 * 1024),
                        int(float(free_mb.strip()) * 1024 * 1024))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return None, None, None

    def ensure_disk_space(self, path: str | Path, required_bytes: int, reserve_bytes: int = 2 * 1024**3) -> None:
        free = shutil.disk_usage(Path(path)).free
        if free < required_bytes + reserve_bytes:
            raise RuntimeError(f"Insufficient disk space: {free} bytes free, {required_bytes + reserve_bytes} required")


def adaptive_batch_size(available_vram_bytes: int | None, configured: int = 1,
                        maximum: int = 8) -> int:
    if available_vram_bytes is None or available_vram_bytes <= 0:
        return max(1, min(configured, maximum))
    suggested = max(1, available_vram_bytes // (2 * 1024**3))
    return int(min(maximum, max(configured, suggested)))


class StageScheduler:
    """Bounded CPU, single-GPU, and I/O queues for safe stage overlap."""
    def __init__(self, cpu_workers: int = 2, io_workers: int = 2) -> None:
        self.cpu = ThreadPoolExecutor(max_workers=max(1, cpu_workers), thread_name_prefix="los80-cpu")
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="los80-gpu")
        self.io = ThreadPoolExecutor(max_workers=max(1, io_workers), thread_name_prefix="los80-io")
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, queue: str, unique_key: str, action: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        with self._lock:
            if unique_key in self._active:
                raise RuntimeError(f"Work already scheduled: {unique_key}")
            self._active.add(unique_key)
        executor = {"cpu": self.cpu, "gpu": self.gpu, "io": self.io}[queue]
        future = executor.submit(action, *args, **kwargs)
        future.add_done_callback(lambda _: self._discard(unique_key))
        return future

    def _discard(self, key: str) -> None:
        with self._lock:
            self._active.discard(key)

    def close(self) -> None:
        for executor in (self.cpu, self.gpu, self.io):
            executor.shutdown(wait=True, cancel_futures=False)


class BenchmarkTimer:
    def __init__(self, database: JobDatabase, monitor: ResourceMonitor | None = None) -> None:
        self.database, self.monitor = database, monitor or ResourceMonitor()

    @contextmanager
    def measure(self, source_name: str, stage: str, units: float | None = None,
                units_name: str | None = None, cache_hit: bool = False) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            resources = self.monitor.snapshot(self.database.db_path.parent)
            self.database.record_benchmark(source_name, stage, elapsed,
                units_processed=units, units_name=units_name, cache_hit=cache_hit,
                cpu_percent=resources["cpu_percent"], ram_bytes=resources["ram_bytes"],
                gpu_utilization=resources["gpu_utilization"], gpu_vram_bytes=resources["gpu_vram_bytes"])


def profile_action(action: Callable[[], Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw, text = output / "los80-profile.pstats", output / "los80-profile.txt"
    folded = output / "los80-profile.folded"
    profiler = cProfile.Profile()
    profiler.enable()
    action()
    profiler.disable()
    profiler.dump_stats(raw)
    with text.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats("cumulative")
        stats.print_stats(50)
    stats = pstats.Stats(profiler)
    with folded.open("w", encoding="utf-8") as handle:
        for (filename, line, function), values in stats.stats.items():
            handle.write(f"{Path(filename).name}:{line}:{function} {max(1, int(values[3] * 1_000_000))}\n")
    return {"pstats": raw, "flamegraph": folded, "slowest_functions": text}


def cleanup_temporary_files(directory: str | Path, max_age_seconds: float = 86400) -> int:
    root, now, removed = Path(directory), time.time(), 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in {".tmp", ".part"} or path.name.endswith(".upscaled.mp4")):
            if now - path.stat().st_mtime >= max_age_seconds:
                path.unlink()
                removed += 1
    return removed


def cleanup_job_directory(working_root: str | Path, source_name: str) -> bool:
    root = Path(working_root).resolve()
    target = (root / source_name).resolve()
    if target.parent != root or not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
