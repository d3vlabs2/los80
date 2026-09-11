from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from los80.realesrgan_runtime import RealESRGANRuntime, RealESRGANRuntimeError


class UpscalingError(RuntimeError):
    pass


class RealESRGANBackend:
    def upscale(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            runtime = RealESRGANRuntime(config.get("backend_path")).ensure()
        except RealESRGANRuntimeError as exc:
            raise UpscalingError(str(exc)) from exc

        ffmpeg = str(config.get("ffmpeg_binary", "ffmpeg"))
        ffprobe = str(config.get("ffprobe_binary", "ffprobe"))
        if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
            raise UpscalingError("FFmpeg and FFprobe are required for frame-based Real-ESRGAN upscaling")

        model_name = str(config.get("model", "RealESRGAN_x4plus"))
        if model_name == "RealESRGAN_x4plus":
            model_name = "realesrgan-x4plus"
        missing = [runtime.model_dir / f"{model_name}{suffix}" for suffix in (".bin", ".param")
                   if not (runtime.model_dir / f"{model_name}{suffix}").is_file()]
        if missing:
            raise UpscalingError("Required model files are missing: " + ", ".join(map(str, missing)))

        work_dir = output_path.parent / f".{output_path.name}.realesrgan-frames"
        source_frames = work_dir / "source"
        upscaled_frames = work_dir / "upscaled"
        source_frames.mkdir(parents=True, exist_ok=True)
        upscaled_frames.mkdir(parents=True, exist_ok=True)
        metadata = self._probe(input_path, ffprobe)
        extraction_marker = work_dir / ".extraction-complete"
        if not extraction_marker.exists():
            self._run([
                ffmpeg, "-y", "-copyts", "-i", str(input_path), "-map", "0:v:0",
                "-fps_mode", "passthrough", "-frame_pts", "1",
                str(source_frames / "frame-%020d.png"),
            ], "extract video frames")
            if not any(source_frames.glob("frame-*.png")):
                raise UpscalingError(f"FFmpeg extracted no frames from {input_path}")
            extraction_marker.write_text("complete\n", encoding="utf-8")

        source_paths = sorted(source_frames.glob("frame-*.png"))
        pending = [path for path in source_paths if not (upscaled_frames / path.name).is_file()]
        batch_size = max(int(config.get("batch_size", 8)), 1)
        batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]
        gpu_count = self._gpu_count() if config.get("device") == "cuda" else 0
        worker_count = min(max(gpu_count, 1), len(batches)) if batches else 0
        if batches:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="realesrgan") as executor:
                futures = {
                    executor.submit(
                        self._upscale_batch, batch, number, number % gpu_count if gpu_count else None,
                        work_dir, upscaled_frames, runtime.executable, runtime.model_dir,
                        model_name, config,
                    ): number
                    for number, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    future.result()
        missing_outputs = [path.name for path in source_paths if not (upscaled_frames / path.name).is_file()]
        if missing_outputs:
            raise UpscalingError(f"Real-ESRGAN did not produce {len(missing_outputs)} upscaled frames")

        concat_file = work_dir / "frames.ffconcat"
        self._write_concat(concat_file, source_paths, upscaled_frames, metadata)
        video_codec = str(config.get("intermediate_codec", "libx264"))
        start_time = str(metadata.get("start_time", "0"))
        command = [
            ffmpeg, "-y", "-copyts", "-itsoffset", start_time,
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(input_path), "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
            "-map_metadata", "1", "-map_chapters", "1", "-c:v", video_codec,
        ]
        if video_codec == "libx264":
            command.extend(["-crf", "0", "-preset", "medium"])
        command.extend(["-c:a", "copy", "-c:s", "copy", "-fps_mode", "vfr"])
        if metadata.get("sample_aspect_ratio") not in {None, "", "N/A", "1:1"}:
            command.extend(["-vf", f"setsar={metadata['sample_aspect_ratio']}"])
        if metadata.get("pix_fmt") and metadata["pix_fmt"] != "unknown":
            command.extend(["-pix_fmt", str(metadata["pix_fmt"])])
        for option, value in self._color_options(metadata).items():
            command.extend([option, value])
        command.append(str(output_path))
        self._run(command, "reassemble upscaled video")
        if not output_path.exists():
            raise UpscalingError(f"FFmpeg did not produce upscaled video: {output_path}")
        shutil.rmtree(work_dir)
        return output_path

    def _upscale_batch(self, frames: list[Path], batch_number: int, gpu_id: int | None,
                       work_dir: Path, output_dir: Path, executable: Path, model_dir: Path,
                       model_name: str, config: dict[str, object]) -> None:
        batch_dir = work_dir / "batches" / f"batch-{batch_number:06d}"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True)
        for frame in frames:
            target = batch_dir / frame.name
            try:
                os.link(frame, target)
            except OSError:
                shutil.copy2(frame, target)
        command = [str(executable), "-i", str(batch_dir), "-o", str(output_dir),
                   "-m", str(model_dir), "-n", model_name, "-f", "png"]
        if config.get("tile_size"):
            command.extend(["-t", str(config["tile_size"])])
        if gpu_id is not None:
            command.extend(["-g", str(gpu_id)])
        self._run(command, f"upscale frame batch {batch_number + 1}")
        shutil.rmtree(batch_dir)

    @staticmethod
    def _run(command: list[str], operation: str) -> None:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise UpscalingError(f"Failed to {operation}: {message}")

    @staticmethod
    def _probe(input_path: Path, ffprobe: str) -> dict[str, object]:
        completed = subprocess.run([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
            "-show_format", "-of", "json", str(input_path),
        ], check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise UpscalingError(f"Failed to inspect video: {completed.stderr.strip()}")
        try:
            data = json.loads(completed.stdout)
            metadata = dict(data["streams"][0])
            if "start_time" not in metadata and isinstance(data.get("format"), dict):
                metadata["start_time"] = data["format"].get("start_time", "0")
            return metadata
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise UpscalingError(f"FFprobe returned invalid video metadata for {input_path}") from exc

    @staticmethod
    def _gpu_count() -> int:
        if not shutil.which("nvidia-smi"):
            return 0
        completed = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
        return sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("GPU "))

    @staticmethod
    def _write_concat(concat_file: Path, source_paths: list[Path], output_dir: Path,
                      metadata: dict[str, object]) -> None:
        numerator, denominator = _parse_ratio(str(metadata.get("time_base", "1/25")))
        fallback = 1.0 / _ratio(str(metadata.get("avg_frame_rate", "25/1")), 25.0)
        pts = [int(path.stem.split("-")[-1]) for path in source_paths]
        lines = ["ffconcat version 1.0"]
        for index, source in enumerate(source_paths):
            path = (output_dir / source.name).resolve()
            escaped = str(path).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
            duration = (pts[index + 1] - pts[index]) * numerator / denominator if index + 1 < len(pts) else fallback
            lines.append(f"duration {max(duration, 0.000001):.9f}")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _color_options(metadata: dict[str, object]) -> dict[str, str]:
        mapping = {"color_primaries": "-color_primaries", "color_transfer": "-color_trc",
                   "color_space": "-colorspace", "color_range": "-color_range"}
        return {option: str(metadata[key]) for key, option in mapping.items()
                if metadata.get(key) and metadata[key] != "unknown"}


class AIUpscaler:
    def __init__(self, logger: Optional[logging.Logger] = None, backend_cls: Optional[type] = None) -> None:
        self.logger = logger or logging.getLogger("los80.upscaler")
        self.backend_cls = backend_cls or RealESRGANBackend

    def upscale(self, video_path: str | Path, output_path: str | Path, config: Optional[dict[str, object]] = None) -> Path:
        input_path = Path(video_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            self.logger.info("Using existing upscaled intermediate %s", output_file)
            return output_file

        backend = self.backend_cls()
        if hasattr(backend, "upscale"):
            self.logger.info("Using %s backend", self.backend_cls.__name__)
            return backend.upscale(input_path, output_file, config or {})

        raise UpscalingError("No AI upscaler backend available")

    def detect_device(self) -> str:
        if shutil.which("nvidia-smi"):
            return "cuda"
        return "cpu"


def _parse_ratio(value: str) -> tuple[int, int]:
    try:
        numerator, denominator = value.split("/", 1)
        return int(numerator), max(int(denominator), 1)
    except (ValueError, ZeroDivisionError):
        return 1, 25


def _ratio(value: str, default: float) -> float:
    numerator, denominator = _parse_ratio(value)
    result = numerator / denominator
    return result if result > 0 else default
