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
    analysis_enabled: bool = True
    analysis_force: bool = False
    analysis_generate_gif: bool = False
    analysis_quality_threshold: float = 80.0
    analysis_deinterlace: bool | None = None
    analysis_denoise: bool | None = None
    analysis_sharpen: bool | None = None
    analysis_model: str | None = None
    analysis_tile_size: int | None = None
    analysis_encoder_preset: str | None = None
    analysis_skip: bool | None = None
    parallel_enabled: bool = False
    cpu_workers: int = 2
    io_workers: int = 2
    adaptive_batching: bool = True
    max_batch_size: int = 8
    disk_reserve_bytes: int = 2147483648
    cleanup_enabled: bool = True
    cleanup_max_age_seconds: int = 86400
    job_lease_seconds: int = 21600
    max_retries: int = 2
    include_subtitles: bool = True
    include_translation: bool = True
    translation_model: str = "facebook/nllb-200-distilled-600M"
    translation_device: str = "auto"
    translation_batch_size: int = 8
    enable_duplicate_detection: bool = False
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    upscaler_model: str = "RealESRGAN_x4plus"
    realesrgan_backend_path: str | None = None
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
            "analysis",
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
        # The canonical documented configuration groups this setting with AI
        # upscaling, while flat files remain supported for compatibility.
        ai_upscaling = raw.get("ai_upscaling", {})
        if isinstance(ai_upscaling, dict) and "backend_path" in ai_upscaling:
            config.realesrgan_backend_path = ai_upscaling["backend_path"]
        translation = raw.get("translation", {})
        if isinstance(translation, dict):
            for yaml_key, attribute in {
                "model": "translation_model",
                "device": "translation_device",
                "batch_size": "translation_batch_size",
            }.items():
                if yaml_key in translation:
                    setattr(config, attribute, translation[yaml_key])
    return config
