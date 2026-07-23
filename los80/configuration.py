from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@dataclass
class AppConfig:
    input_dir: str = "./input"
    output_dir: str = "./output"
    archive_dir: str = "./archive"
    working_dir: str = "./working"
    database_path: str = "./los80.sqlite"
    reports_dir: str = "./reports"
    max_retries: int = 2
    include_subtitles: bool = True
    include_translation: bool = True
    enable_duplicate_detection: bool = False
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    upscaler_model: str = "RealESRGAN_x4plus"
    target_width: int = 3840
    target_height: int = 2160
    tile_size: int = 0
    tile_padding: int = 10
    face_enhance: bool = False
    denoise: bool = False
    sharpen: bool = False
    skip_if_target_reached: bool = True
    encoder_codec: str = "libx265"
    encoder_preset: str = "medium"
    encoder_crf: int = 28
    encoder_bitrate: str = "0"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    hardware_acceleration: str = "auto"
    embed_subtitles: bool = False
    subtitle_language: str = "eng"
    duration_tolerance_seconds: float = 1.0
    supported_extensions: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi")
    drive_enabled: bool = False
    drive_parent_id: str | None = None
    drive_client_id: str | None = None
    drive_client_secret: str | None = None
    drive_token_path: str | None = None
    drive_credentials_path: str | None = None
    drive_scopes: tuple[str, ...] = ("https://www.googleapis.com/auth/drive",)
    drive_use_colab: bool = False
    drive_input_folder_name: str = "INPUT"
    drive_output_folder_name: str = "OUTPUT"
    drive_archive_folder_name: str = "ARCHIVE"
    drive_logs_folder_name: str = "LOGS"
    drive_reports_folder_name: str = "REPORTS"
    drive_temp_folder_name: str = "TEMP"
    drive_chunk_size: int = 1024 * 1024
    stages: list[str] = field(
        default_factory=lambda: [
            "scan",
            "audio_extraction",
            "subtitles",
            "translation",
            "upscaling",
            "encoding",
            "validation",
            "upload",
            "archive",
        ]
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load LOS80 settings from YAML.

    ``config/config.yaml`` is the canonical project configuration. Passing an
    explicit path remains supported for callers that manage their own config
    location.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config = AppConfig()
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        for key, value in raw.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config
