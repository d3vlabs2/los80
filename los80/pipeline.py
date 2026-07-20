from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from los80.dashboard import Dashboard

from los80.database import JobDatabase
from los80.drive import GoogleDriveClient
from los80.encoder import EncoderService
from los80.scanner import scan_videos
from los80.subtitles import SubtitlesService
from los80.translator import TranslationService
from los80.upscaler import AIUpscaler
from los80.validator import Validator
from los80.utilities import ensure_directory


class Pipeline:
    def __init__(self, config: object, database: Optional[JobDatabase] = None) -> None:
        self.config = config
        self.database = database or JobDatabase(getattr(config, "database_path", "./los80.sqlite"))
        self.logger = logging.getLogger("los80.pipeline")
        self.subtitles_service = SubtitlesService(
            self.logger,
            getattr(config, "whisper_model", "base"),
            getattr(config, "whisper_device", "cpu"),
            getattr(config, "whisper_compute_type", "int8"),
        )
        self.translation_service = TranslationService(self.logger)
        self.upscaler = AIUpscaler(self.logger)
        self.encoder = EncoderService(self.logger)
        self.validator = Validator(self.logger)
        self.drive = GoogleDriveClient(self.logger)

    def run(self) -> None:
        ensure_directory(self.config.input_dir)
        ensure_directory(self.config.output_dir)
        ensure_directory(self.config.archive_dir)
        ensure_directory(self.config.working_dir)
        ensure_directory(self.config.reports_dir)

        for video_path in scan_videos(self.config.input_dir, self.config.supported_extensions):
            source_name = video_path.name
            self.database.add_job(source_name, str(video_path))
            self.database.mark_stage(source_name, "scan", "completed", details="discovered")
            working_dir = Path(self.config.working_dir) / source_name
            working_dir.mkdir(parents=True, exist_ok=True)

            spanish_path = Path(self.config.output_dir) / f"{video_path.stem}.es.srt"
            english_path = Path(self.config.output_dir) / f"{video_path.stem}.en.srt"
            upscaled_path = working_dir / f"{video_path.stem}.upscaled{video_path.suffix}"
            encoded_path = Path(self.config.output_dir) / f"{video_path.stem}.mp4"
            encoder_config = {
                "codec": getattr(self.config, "encoder_codec", "libx265"),
                "preset": getattr(self.config, "encoder_preset", "medium"),
                "crf": getattr(self.config, "encoder_crf", 28),
                "bitrate": getattr(self.config, "encoder_bitrate", "0"),
                "audio_codec": getattr(self.config, "audio_codec", "aac"),
                "audio_bitrate": getattr(self.config, "audio_bitrate", "128k"),
                "hardware_preference": getattr(self.config, "hardware_preference", getattr(self.config, "hardware_acceleration", "auto")),
                "embed_subtitles": getattr(self.config, "embed_subtitles", False),
                "copy_metadata": getattr(self.config, "copy_metadata", True),
                "preserve_chapters": getattr(self.config, "preserve_chapters", True),
                "preserve_color_space": getattr(self.config, "preserve_color_space", True),
                "overwrite_existing": getattr(self.config, "overwrite_existing", True),
                "subtitle_path": str(english_path),
            }
            upscaler_config = {
                "model": getattr(self.config, "upscaler_model", "RealESRGAN_x4plus"),
                "target_width": getattr(self.config, "target_width", 3840),
                "target_height": getattr(self.config, "target_height", 2160),
                "tile_size": getattr(self.config, "tile_size", 0),
                "tile_padding": getattr(self.config, "tile_padding", 10),
                "face_enhance": getattr(self.config, "face_enhance", False),
                "denoise": getattr(self.config, "denoise", False),
                "sharpen": getattr(self.config, "sharpen", False),
                "device": self.upscaler.detect_device(),
                "skip_if_target_reached": getattr(self.config, "skip_if_target_reached", True),
            }

            stages = self.database.get_stages(source_name)
            if self.config.include_subtitles and stages.get("subtitles") != "completed":
                if spanish_path.exists() and english_path.exists():
                    self.database.mark_stage(source_name, "subtitles", "completed", details="existing")
                else:
                    for attempt in range(self.config.max_retries + 1):
                        try:
                            self._run_stage(source_name, "subtitles", lambda: self.subtitles_service.generate_subtitles(video_path, spanish_path, english_path), details="generated")
                            break
                        except Exception as exc:  # pragma: no cover - runtime path
                            self.logger.warning("Subtitle generation failed for %s (attempt %s): %s", source_name, attempt + 1, exc)
                            self.database.mark_stage(source_name, "subtitles", "retrying", details=str(exc))
                            if attempt >= self.config.max_retries:
                                raise RuntimeError(f"Subtitle generation failed for {source_name}: {exc}") from exc
            elif self.config.include_subtitles and spanish_path.exists() and english_path.exists():
                self.database.mark_stage(source_name, "subtitles", "completed", details="existing")

            if self.config.include_translation and stages.get("translation") != "completed":
                if english_path.exists():
                    self._run_stage(source_name, "translation", lambda: self.translation_service.translate_subtitles(spanish_path, english_path), details="translated")
                else:
                    self.database.mark_stage(source_name, "translation", "skipped", details="missing english output")
            elif self.config.include_translation and english_path.exists():
                self.database.mark_stage(source_name, "translation", "completed", details="existing")

            if stages.get("upscaling") != "completed" or not upscaled_path.exists():
                self._run_stage(source_name, "upscaling", lambda: self.upscaler.upscale(video_path, upscaled_path, upscaler_config), details="upscaled")
            else:
                self.database.mark_stage(source_name, "upscaling", "completed", details="existing")

            if stages.get("encoding") != "completed" or not encoded_path.exists():
                self.database.mark_stage(source_name, "encoding", "in_progress", details="encoding")
                selected_encoder = self.encoder.backend_cls().detect_hardware_encoder(encoder_config["hardware_preference"]) if hasattr(self.encoder.backend_cls(), "detect_hardware_encoder") else "libx265"
                self.logger.info("Encoding %s with %s", source_name, selected_encoder)
                self._run_stage(source_name, "encoding", lambda: self.encoder.encode(upscaled_path, encoded_path, encoder_config), details=selected_encoder)
            else:
                self.database.mark_stage(source_name, "encoding", "completed", details="existing")

            self._run_stage(source_name, "validation", lambda: self.validator.validate(video_path, encoded_path, spanish_path, english_path, codec=encoder_config["codec"], duration_tolerance_seconds=getattr(self.config, "duration_tolerance_seconds", 1.0)), details="validated")
            report_html = Path(self.config.reports_dir) / f"{video_path.stem}.html"
            report_csv = Path(self.config.reports_dir) / f"{video_path.stem}.csv"
            upload_targets = [
                (encoded_path, encoded_path.name),
                (spanish_path, f"{video_path.stem}.es.srt"),
                (english_path, f"{video_path.stem}.en.srt"),
                (report_html, report_html.name),
                (report_csv, report_csv.name),
            ]
            transfer_results = []
            if getattr(self.config, "drive_enabled", False):
                for local_path, remote_name in upload_targets:
                    if not local_path.exists():
                        continue
                    self.database.mark_stage(source_name, "upload", "in_progress", details=f"uploading {remote_name}")
                    result = self.drive.upload(local_path, remote_name, parent_id=getattr(self.config, "drive_parent_id", None), verify=True)
                    transfer_results.append(result)
                    if result.get("status") == "uploaded" and result.get("verified"):
                        self.database.mark_stage(source_name, "upload", "completed", details=f"uploaded {remote_name}")
                    elif result.get("status") == "skipped":
                        self.database.mark_stage(source_name, "upload", "completed", details=f"skipped {remote_name}")
                    else:
                        self.database.mark_stage(source_name, "upload", "failed", details=f"failed {remote_name}")
                        raise RuntimeError(f"Drive upload failed for {remote_name}")
                if any(result.get("verified") for result in transfer_results):
                    archived_path = self.drive.archive_source(video_path, self.config.archive_dir)
                    self.database.mark_stage(source_name, "archive", "completed", details=f"archived to {archived_path}")
                else:
                    self.database.mark_stage(source_name, "archive", "skipped", details="upload verification failed")
            else:
                self.database.mark_stage(source_name, "upload", "completed", details="drive disabled")
                archived_path = self.drive.archive_source(video_path, self.config.archive_dir)
                self.database.mark_stage(source_name, "archive", "completed", details=f"archived to {archived_path}")
            self.database.mark_stage(source_name, "completed", "completed", details="finished")

    def _run_stage(self, source_name: str, stage_name: str, action, *, details: str) -> None:
        self.database.mark_stage(source_name, stage_name, "in_progress", details="starting")
        started_at = datetime.now()
        try:
            action()
        except Exception as exc:  # pragma: no cover - runtime path
            self.database.mark_stage(source_name, stage_name, "failed", details=str(exc))
            self.logger.exception("Stage %s failed for %s: %s", stage_name, source_name, exc)
            raise
        elapsed = (datetime.now() - started_at).total_seconds()
        self.logger.info("Stage %s completed for %s in %.2fs", stage_name, source_name, elapsed)
        self.database.mark_stage(source_name, stage_name, "completed", details=f"{details} ({elapsed:.2f}s)")

    def dashboard_summary(self) -> dict[str, object]:
        jobs = [self.database.get_job(job_name).__dict__ for job_name in self.database._get_all_job_names()]
        return Dashboard(jobs).summary().__dict__
