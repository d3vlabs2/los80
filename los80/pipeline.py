from __future__ import annotations

import logging
import subprocess
import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from los80.dashboard import Dashboard
from los80.analysis import MediaAnalyzer, source_fingerprint

from los80.database import JobDatabase
from los80.drive import GoogleDriveClient
from los80.encoder import EncoderService
from los80.scanner import scan_videos
from los80.subtitles import SubtitlesService
from los80.translator import TranslationService
from los80.upscaler import AIUpscaler
from los80.validator import Validator
from los80.utilities import ensure_directory
from los80.reports import write_csv_report, write_html_report
from los80.performance import (BenchmarkTimer, ResourceMonitor, StageScheduler,
                               adaptive_batch_size, cache_key, cleanup_job_directory,
                               cleanup_temporary_files)


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
        self.translation_service = TranslationService(
            self.logger,
            getattr(config, "translation_model", "facebook/nllb-200-distilled-600M"),
            getattr(config, "translation_device", "auto"),
            getattr(config, "translation_batch_size", 8),
        )
        self.upscaler = AIUpscaler(self.logger)
        self.encoder = EncoderService(self.logger)
        self.validator = Validator(self.logger)
        self.drive = GoogleDriveClient(self.logger)
        self.analyzer = MediaAnalyzer(self.logger)
        self.monitor = ResourceMonitor()
        self.benchmarks = BenchmarkTimer(self.database, self.monitor)
        self.scheduler = StageScheduler(getattr(config, "cpu_workers", 2), getattr(config, "io_workers", 2))
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex}"

    def run(self) -> dict[str, object]:
        run_started = time.perf_counter()
        ensure_directory(self.config.input_dir)
        ensure_directory(self.config.output_dir)
        ensure_directory(self.config.archive_dir)
        ensure_directory(self.config.working_dir)
        ensure_directory(self.config.reports_dir)
        self.database.recover_interrupted()
        if getattr(self.config, "cleanup_enabled", True):
            cleanup_temporary_files(self.config.working_dir, getattr(self.config, "cleanup_max_age_seconds", 86400))

        videos = scan_videos(self.config.input_dir, self.config.supported_extensions)
        prefetches = {
            path: self.scheduler.submit("io", f"prefetch:{path}", self._prefetch, path)
            for path in videos
        }
        finalizers = []
        for video_path in videos:
            source_name = video_path.name
            try:
                prefetches[video_path].result()
                self.database.add_job(source_name, str(video_path))
                if not self.database.acquire_lease(source_name, self.worker_id, getattr(self.config, "job_lease_seconds", 21600)):
                    self.logger.info("Skipping leased job %s", source_name)
                    self.database.mark_job_status(source_name, "skipped")
                    print(f"\u2022 {source_name}: skipped (active lease)")
                    continue
                print(f"[{source_name}] scan: completed")
                self.database.mark_stage(source_name, "scan", "completed", details="discovered")
                job_started = time.perf_counter()
                working_dir = Path(self.config.working_dir) / source_name
                working_dir.mkdir(parents=True, exist_ok=True)
                self.monitor.ensure_disk_space(
                    working_dir, video_path.stat().st_size * 2,
                    reserve_bytes=getattr(self.config, "disk_reserve_bytes", 2 * 1024**3),
                )

                analysis = self.database.get_analysis(source_name)
                fingerprint = source_fingerprint(video_path)
                overrides = {
                        "deinterlace": getattr(self.config, "analysis_deinterlace", None),
                        "denoise": getattr(self.config, "analysis_denoise", None),
                        "sharpen": getattr(self.config, "analysis_sharpen", None),
                        "model": getattr(self.config, "analysis_model", None),
                        "tile_size": getattr(self.config, "analysis_tile_size", None),
                        "encoder_preset": getattr(self.config, "analysis_encoder_preset", None),
                        "skip": getattr(self.config, "analysis_skip", None),
                }
                analysis_settings = {**overrides,
                    "quality_threshold": getattr(self.config, "analysis_quality_threshold", 80.0),
                    "generate_gif": getattr(self.config, "analysis_generate_gif", False)}
                analysis_cache_key, analysis_config_hash = cache_key("analysis", fingerprint, analysis_settings)
                cached_analysis = None if getattr(self.config, "analysis_force", False) else self.database.get_cache(analysis_cache_key)
                print(f"[{source_name}] analysis: starting")
                if cached_analysis is not None:
                    analysis = cached_analysis
                    self.database.save_analysis(source_name, analysis)
                    self.database.mark_stage(source_name, "analysis", "completed", details="cache hit")
                    self.database.record_benchmark(source_name, "analysis", 0, cache_hit=True)
                elif getattr(self.config, "analysis_enabled", True):
                    try:
                        self.database.mark_stage(source_name, "analysis", "in_progress", details="analyzing")
                        analysis = self.analyzer.analyze(
                            video_path, overrides,
                            quality_threshold=float(getattr(self.config, "analysis_quality_threshold", 80.0)),
                        )
                        analysis["previews"] = self.analyzer.generate_previews(
                            video_path, self.config.reports_dir,
                            animated=getattr(self.config, "analysis_generate_gif", False),
                        )
                        self.database.save_analysis(source_name, analysis)
                        self.database.put_cache(analysis_cache_key, fingerprint, "analysis", analysis_config_hash, analysis)
                        self.database.mark_stage(source_name, "analysis", "completed", details="analyzed")
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        # Preserve previous-phase compatibility for unreadable inputs while
                        # making the unavailable preflight visible in job state.
                        self.logger.warning("Analysis unavailable for %s: %s", source_name, exc)
                        self.database.mark_stage(source_name, "analysis", "skipped", details=str(exc))
                elif analysis is not None:
                    self.database.mark_stage(source_name, "analysis", "completed", details="existing")
                print(f"[{source_name}] analysis: {self.database.get_stages(source_name).get('analysis', 'skipped')}")

                recommendations = dict(analysis.get("recommendations", {})) if analysis else {}
                media_duration = float((analysis or {}).get("metadata", {}).get("duration", 0))
                media_frames = float((analysis or {}).get("metadata", {}).get("frame_count", 0))
                if not media_frames:
                    media_frames = media_duration * float((analysis or {}).get("metadata", {}).get("fps", 0))
                if analysis is not None:
                    # Analysis data is reusable, while policy overrides remain live and
                    # can be changed without forcing an expensive media rescan.
                    override_attributes = {
                        "deinterlace": "analysis_deinterlace",
                        "denoise": "analysis_denoise",
                        "sharpen": "analysis_sharpen",
                        "model": "analysis_model",
                        "tile_size": "analysis_tile_size",
                        "encoder_preset": "analysis_encoder_preset",
                    }
                    for decision, attribute in override_attributes.items():
                        override = getattr(self.config, attribute, None)
                        if override is not None:
                            recommendations[decision] = override
                    skip_override = getattr(self.config, "analysis_skip", None)
                    recommendations["skip"] = (
                        bool(skip_override) if skip_override is not None
                        else float(analysis.get("quality_score", 0))
                        >= float(getattr(self.config, "analysis_quality_threshold", 80.0))
                    )

                spanish_path = Path(self.config.output_dir) / f"{video_path.stem}.es.srt"
                english_path = Path(self.config.output_dir) / f"{video_path.stem}.en.srt"
                upscaled_path = working_dir / f"{video_path.stem}.upscaled{video_path.suffix}"
                encoded_path = Path(self.config.output_dir) / f"{video_path.stem}.mp4"
                encoder_config = {
                    "codec": getattr(self.config, "encoder_codec", "libx265"),
                    "preset": recommendations.get("encoder_preset", getattr(self.config, "encoder_preset", "medium")),
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
                    "deinterlace": recommendations.get("deinterlace", False),
                }
                upscaler_config = {
                    "backend_path": getattr(self.config, "realesrgan_backend_path", None),
                    "model": recommendations.get("model", getattr(self.config, "upscaler_model", "RealESRGAN_x4plus")),
                    "target_width": getattr(self.config, "target_width", 3840),
                    "target_height": getattr(self.config, "target_height", 2160),
                    "tile_size": recommendations.get("tile_size", getattr(self.config, "tile_size", 0)),
                    "tile_padding": getattr(self.config, "tile_padding", 10),
                    "face_enhance": getattr(self.config, "face_enhance", False),
                    "denoise": recommendations.get("denoise", getattr(self.config, "denoise", False)),
                    "sharpen": recommendations.get("sharpen", getattr(self.config, "sharpen", False)),
                    "deinterlace": recommendations.get("deinterlace", False),
                    "device": self.upscaler.detect_device(),
                    "skip_if_target_reached": getattr(self.config, "skip_if_target_reached", True),
                }
                if getattr(self.config, "adaptive_batching", True):
                    resources = self.monitor.snapshot(working_dir)
                    upscaler_config["batch_size"] = adaptive_batch_size(
                        resources.get("gpu_vram_free_bytes"),
                        configured=int(getattr(self.config, "batch_size", 1)),
                        maximum=int(getattr(self.config, "max_batch_size", 8)),
                    )
                if analysis is not None:
                    analysis["processing_decisions"] = {
                        "skip_upscaling": bool(recommendations.get("skip", False)),
                        "deinterlace": bool(encoder_config["deinterlace"]),
                        "denoise": bool(upscaler_config["denoise"]),
                        "sharpen": bool(upscaler_config["sharpen"]),
                        "model": upscaler_config["model"],
                        "tile_size": upscaler_config["tile_size"],
                        "encoder_preset": encoder_config["preset"],
                    }
                    self.database.save_analysis(source_name, analysis)

                stages = self.database.get_stages(source_name)
                subtitle_settings = {"model": self.config.whisper_model, "device": self.config.whisper_device,
                                     "compute_type": self.config.whisper_compute_type}
                subtitle_key, subtitle_hash = cache_key("transcription", fingerprint, subtitle_settings)
                subtitle_cached = self.database.get_cache(subtitle_key)
                print(f"[{source_name}] Spanish subtitles / English subtitles: starting")
                if self.config.include_subtitles and subtitle_cached and spanish_path.exists() and english_path.exists():
                    self.database.mark_stage(source_name, "subtitles", "completed", details="cache hit")
                    self.database.record_benchmark(source_name, "subtitles", 0, cache_hit=True)
                elif self.config.include_subtitles:
                    if not (spanish_path.exists() and english_path.exists() and stages.get("subtitles") == "completed"):
                        for attempt in range(self.config.max_retries + 1):
                            try:
                                self._run_stage(source_name, "subtitles", lambda: self.scheduler.submit(
                                    "cpu", f"cpu:subtitles:{source_name}", self.subtitles_service.generate_subtitles,
                                    video_path, spanish_path, english_path).result(), details="generated",
                                    units=media_duration, units_name="media_seconds")
                                self.database.put_cache(subtitle_key, fingerprint, "transcription", subtitle_hash,
                                                        {"spanish": str(spanish_path), "english": str(english_path)})
                                break
                            except Exception as exc:  # pragma: no cover - runtime path
                                self.logger.warning("Subtitle generation failed for %s (attempt %s): %s", source_name, attempt + 1, exc)
                                self.database.mark_stage(source_name, "subtitles", "retrying", details=str(exc))
                                if attempt >= self.config.max_retries:
                                    raise RuntimeError(f"Subtitle generation failed for {source_name}: {exc}") from exc
                    else:
                        self.database.put_cache(subtitle_key, fingerprint, "transcription", subtitle_hash,
                                                {"spanish": str(spanish_path), "english": str(english_path)})
                subtitle_status = self.database.get_stages(source_name).get("subtitles", "skipped")
                print(f"[{source_name}] Spanish subtitles / English subtitles: {subtitle_status}")

                translation_settings = {"source_language": "es", "target_language": "en"}
                translation_key, translation_hash = cache_key("translation", fingerprint, translation_settings)
                translation_cached = self.database.get_cache(translation_key)
                print(f"[{source_name}] English subtitle translation: starting")
                if self.config.include_translation and translation_cached and english_path.exists():
                    self.database.mark_stage(source_name, "translation", "completed", details="cache hit")
                    self.database.record_benchmark(source_name, "translation", 0, cache_hit=True)
                elif self.config.include_translation and stages.get("translation") != "completed":
                    if english_path.exists():
                        self._run_stage(source_name, "translation", lambda: self.scheduler.submit(
                            "cpu", f"cpu:translation:{source_name}", self.translation_service.translate_subtitles,
                            spanish_path, english_path).result(), details="translated")
                        self.database.put_cache(translation_key, fingerprint, "translation", translation_hash,
                                                {"output": str(english_path)})
                    else:
                        self.database.mark_stage(source_name, "translation", "skipped", details="missing english output")
                elif self.config.include_translation and english_path.exists():
                    self.database.mark_stage(source_name, "translation", "completed", details="existing")
                translation_status = self.database.get_stages(source_name).get("translation", "skipped")
                print(f"[{source_name}] English subtitle translation: {translation_status}")

                processing_input = upscaled_path
                if recommendations.get("skip", False):
                    processing_input = video_path
                    self.database.mark_stage(source_name, "upscaling", "skipped", details="source already high quality")
                    print(f"[{source_name}] upscaling: skipped")
                elif stages.get("upscaling") != "completed" or not upscaled_path.exists():
                    if upscaled_path.exists() and stages.get("upscaling") != "completed":
                        upscaled_path.unlink()
                    self._run_stage(source_name, "upscaling", lambda: self.scheduler.submit(
                        "gpu", f"gpu:{source_name}", self.upscaler.upscale,
                        video_path, upscaled_path, upscaler_config).result(), details="upscaled",
                        units=media_frames, units_name="frames")
                else:
                    self.database.mark_stage(source_name, "upscaling", "completed", details="existing")
                    print(f"[{source_name}] upscaling: completed (resume)")

                if stages.get("encoding") != "completed" or not encoded_path.exists():
                    if encoded_path.exists() and stages.get("encoding") != "completed":
                        encoded_path.unlink()
                    self.database.mark_stage(source_name, "encoding", "in_progress", details="encoding")
                    selected_encoder = self.encoder.backend_cls().detect_hardware_encoder(encoder_config["hardware_preference"]) if hasattr(self.encoder.backend_cls(), "detect_hardware_encoder") else "libx265"
                    self.logger.info("Encoding %s with %s", source_name, selected_encoder)
                    self._run_stage(source_name, "encoding", lambda: self.encoder.encode(processing_input, encoded_path, encoder_config), details=selected_encoder,
                                    units=media_frames, units_name="frames")
                else:
                    self.database.mark_stage(source_name, "encoding", "completed", details="existing")
                    print(f"[{source_name}] encoding: completed (resume)")

                self._run_stage(source_name, "validation", lambda: self.validator.validate(video_path, encoded_path, spanish_path, english_path, codec=encoder_config["codec"], duration_tolerance_seconds=getattr(self.config, "duration_tolerance_seconds", 1.0)), details="validated")
                report_html = Path(self.config.reports_dir) / f"{video_path.stem}.html"
                report_csv = Path(self.config.reports_dir) / f"{video_path.stem}.csv"
                if analysis is not None:
                    required_previews = {"thumbnail", "contact_sheet", "comparison"}
                    if getattr(self.config, "analysis_generate_gif", False):
                        required_previews.add("animated_gif")
                    previews = analysis.setdefault("previews", {})
                    if not required_previews.issubset(previews) or not all(Path(previews[key]).exists() for key in required_previews if key in previews):
                        previews.update(self.analyzer.generate_previews(
                            video_path, self.config.reports_dir, processed=encoded_path,
                            animated=getattr(self.config, "analysis_generate_gif", False),
                        ))
                    self.database.save_analysis(source_name, analysis)
                    self.database.put_cache(analysis_cache_key, fingerprint, "analysis", analysis_config_hash, analysis)
                report_job = self.database.get_job(source_name).__dict__
                report_job["analysis"] = self.database.get_analysis(source_name)
                report_job["performance"] = self.database.performance_summary()
                write_html_report(report_html, [report_job])
                write_csv_report(report_csv, [report_job])
                upload_targets = [
                    (encoded_path, encoded_path.name),
                    (spanish_path, f"{video_path.stem}.es.srt"),
                    (english_path, f"{video_path.stem}.en.srt"),
                    (report_html, report_html.name),
                    (report_csv, report_csv.name),
                ]
                finalize_args = (source_name, video_path, upload_targets, job_started)
                if getattr(self.config, "parallel_enabled", False):
                    future = self.scheduler.submit("io", f"finalize:{source_name}", self._finalize_job, *finalize_args)
                    finalizers.append((source_name, future))
                else:
                    self._finalize_job(*finalize_args)
            except Exception as exc:  # Keep the batch moving after one failed video.
                self.logger.exception("Processing failed for %s: %s", source_name, exc)
                try:
                    self.database.add_job(source_name, str(video_path))
                    self.database.mark_stage(source_name, "failed", "failed", details=str(exc))
                    self.database.release_lease(source_name, self.worker_id)
                except Exception:
                    self.logger.exception("Could not record failure for %s", source_name)
                print(f"\u2717 {source_name}: failed: {exc}")
        for source_name, future in finalizers:
            try:
                future.result()
            except Exception as exc:
                self.logger.exception("Finalization failed for %s: %s", source_name, exc)
                self.database.mark_stage(source_name, "failed", "failed", details=str(exc))
                self.database.release_lease(source_name, self.worker_id)
                print(f"\u2717 {source_name}: finalization failed: {exc}")
        self.scheduler.close()
        summary = self.dashboard_summary()
        current_statuses = [self.database.get_job(path.name).status for path in videos]
        summary["completed"] = current_statuses.count("completed")
        summary["skipped"] = current_statuses.count("skipped")
        summary["failed"] = current_statuses.count("failed")
        summary["elapsed_time"] = time.perf_counter() - run_started
        return summary

    def _run_stage(self, source_name: str, stage_name: str, action, *, details: str,
                   units: float | None = None, units_name: str | None = None) -> None:
        print(f"[{source_name}] {stage_name}: starting")
        self.database.mark_stage(source_name, stage_name, "in_progress", details="starting")
        started_at = datetime.now()
        try:
            with self.benchmarks.measure(source_name, stage_name, units=units, units_name=units_name):
                action()
        except Exception as exc:  # pragma: no cover - runtime path
            self.database.mark_stage(source_name, stage_name, "failed", details=str(exc))
            self.logger.exception("Stage %s failed for %s: %s", stage_name, source_name, exc)
            print(f"[{source_name}] {stage_name}: failed")
            raise
        elapsed = (datetime.now() - started_at).total_seconds()
        self.logger.info("Stage %s completed for %s in %.2fs", stage_name, source_name, elapsed)
        self.database.mark_stage(source_name, stage_name, "completed", details=f"{details} ({elapsed:.2f}s)")
        print(f"[{source_name}] {stage_name}: completed ({elapsed:.2f}s)")

    @staticmethod
    def _prefetch(path: Path, chunk_size: int = 1024 * 1024) -> int:
        """Warm filesystem caches without retaining large videos in RAM."""
        with path.open("rb") as handle:
            return len(handle.read(chunk_size))

    def _finalize_job(self, source_name: str, video_path: Path,
                      upload_targets: list[tuple[Path, str]], job_started: float) -> None:
        print(f"[{source_name}] upload: starting")
        transfer_results = []
        if getattr(self.config, "drive_enabled", False):
            for local_path, remote_name in upload_targets:
                if not local_path.exists():
                    continue
                self.database.mark_stage(source_name, "upload", "in_progress", details=f"uploading {remote_name}")
                started = time.perf_counter()
                result = self.drive.upload(local_path, remote_name, parent_id=getattr(self.config, "drive_parent_id", None), verify=True)
                self.database.record_benchmark(source_name, "upload", time.perf_counter() - started,
                    units_processed=local_path.stat().st_size, units_name="bytes",
                    cache_hit=result.get("status") == "skipped")
                transfer_results.append(result)
                if result.get("status") in {"uploaded", "skipped"} and result.get("verified"):
                    self.database.mark_stage(source_name, "upload", "completed", details=f"{result['status']} {remote_name}")
                else:
                    self.database.mark_stage(source_name, "upload", "failed", details=f"failed {remote_name}")
                    raise RuntimeError(f"Drive upload failed for {remote_name}")
            if not any(result.get("verified") for result in transfer_results):
                self.database.mark_stage(source_name, "archive", "skipped", details="upload verification failed")
                return
        else:
            self.database.mark_stage(source_name, "upload", "completed", details="drive disabled")
        print(f"[{source_name}] upload: completed")
        print(f"[{source_name}] archive: starting")
        archived_path = self.drive.archive_source(video_path, self.config.archive_dir)
        self.database.mark_stage(source_name, "archive", "completed", details=f"archived to {archived_path}")
        self.database.mark_stage(source_name, "completed", "completed", details="finished")
        print(f"[{source_name}] archive: completed")
        self.database.record_benchmark(source_name, "total", time.perf_counter() - job_started)
        if getattr(self.config, "cleanup_enabled", True):
            cleanup_job_directory(self.config.working_dir, source_name)
        self.database.release_lease(source_name, self.worker_id)

    def dashboard_summary(self) -> dict[str, object]:
        jobs = []
        for job_name in self.database._get_all_job_names():
            job = self.database.get_job(job_name).__dict__
            job["analysis"] = self.database.get_analysis(job_name)
            jobs.append(job)
        return Dashboard(jobs).summary().__dict__
