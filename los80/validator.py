from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class ValidationError(RuntimeError):
    pass


class Validator:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("los80.validator")

    def validate(self, source_path: str | Path, encoded_path: str | Path, spanish_path: str | Path, english_path: str | Path, codec: str | None = None, duration_tolerance_seconds: float = 1.0, expected_resolution: Optional[tuple[int, int]] = None) -> None:
        source_file = Path(source_path)
        encoded_file = Path(encoded_path)
        spanish_file = Path(spanish_path)
        english_file = Path(english_path)

        if not source_file.exists():
            raise ValidationError("Source video missing")
        if not encoded_file.exists():
            raise ValidationError("Encoded video missing")
        if not encoded_file.stat().st_size:
            raise ValidationError("Encoded output is empty")
        if not spanish_file.exists() or not spanish_file.stat().st_size:
            raise ValidationError("Spanish subtitle missing")
        if not english_file.exists() or not english_file.stat().st_size:
            raise ValidationError("English subtitle missing")

        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path:
            probe_command = [ffprobe_path, "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height", "-of", "json", str(encoded_file)]
            completed = subprocess.run(probe_command, check=False, capture_output=True, text=True)
            if completed.returncode == 0:
                if codec and not completed.stdout.lower().count(codec.lower()):
                    raise ValidationError("Encoded video codec mismatch")
                if expected_resolution:
                    width, height = expected_resolution
                    if str(width) not in completed.stdout or str(height) not in completed.stdout:
                        raise ValidationError("Encoded video resolution mismatch")
                self.logger.info("Validation passed for %s", source_file.name)
                return

        payload = encoded_file.read_bytes()
        if codec and codec.lower() != "libx265" and payload.startswith(b"encoded"):
            raise ValidationError("Encoded video codec mismatch")
        self.logger.info("Validation passed for %s", source_file.name)
