from __future__ import annotations

from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def scan_videos(input_dir: str | Path, extensions: Iterable[str] | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        return []
    exts = set(extensions or SUPPORTED_EXTENSIONS)
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts
    )
