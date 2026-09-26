from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from time import perf_counter

from los80.realesrgan_runtime import RealESRGANRuntime, RealESRGANRuntimeError


class UpscalingError(RuntimeError):
    pass


class RealESRGANBackend:
    _UNSUPPORTED_OPTIONS = ("tile_padding", "denoise", "sharpen", "face_enhance", "deinterlace")

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("los80.upscaler.ncnn")

    def upscale(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runtime, model_name = self.prepare(config)
        ffmpeg = str(config.get("ffmpeg_binary", "ffmpeg"))
        ffprobe = str(config.get("ffprobe_binary", "ffprobe"))
        if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
            raise UpscalingError("FFmpeg and FFprobe are required for frame-based Real-ESRGAN upscaling")

        work_dir = output_path.parent / f".{output_path.name}.realesrgan-frames"
        if config.get("resume_identity") is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
            manifest = work_dir / "resume.json"
            identity = dict(config["resume_identity"])
            if hasattr(self, "half"):
                identity["precision"] = "FP16" if self.half else "FP32"
            if manifest.exists():
                if json.loads(manifest.read_text()) != identity:
                    raise UpscalingError("Motion resume settings/source changed; choose a new --output path")
            elif any(path.name != "resume.partial.json" for path in work_dir.iterdir()):
                raise UpscalingError("Frame directory has no resume identity; choose a new --output path")
            else:
                temporary_manifest = work_dir / "resume.partial.json"
                temporary_manifest.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
                temporary_manifest.replace(manifest)
        source_frames = work_dir / "source"
        upscaled_frames = work_dir / "upscaled"
        source_frames.mkdir(parents=True, exist_ok=True)
        upscaled_frames.mkdir(parents=True, exist_ok=True)
        metadata = self._probe(input_path, ffprobe)
        if config.get("frame_manifest") is not None:
            source_paths = self._manifest_source_paths(input_path, work_dir, ffmpeg, config["frame_manifest"],
                                                       float(config.get("scale", 4)))
            expected_names = {path.name for path in source_paths}
            stale = sum(path.name not in expected_names for path in upscaled_frames.glob("*.png"))
            stale_source = sum(path.name not in expected_names for path in source_frames.glob("*.png"))
            if config.get("progress"):
                print(f"frame manifest: {len(source_paths)} expected; ignoring {stale} stale upscaled PNGs "
                      f"and {stale_source} stale source PNGs", flush=True)
        else:
            extraction_marker = work_dir / ".extraction-complete"
            if not extraction_marker.exists():
                self._run([
                    ffmpeg, "-y", "-copyts", "-i", str(input_path), "-map", "0:v:0",
                    "-fps_mode", "passthrough", "-frame_pts", "1",
                    "-enc_time_base", str(metadata.get("time_base", "1/25")),
                    str(source_frames / "frame-%020d.png"),
                ], "extract video frames")
                if not any(source_frames.glob("frame-*.png")):
                    raise UpscalingError(f"FFmpeg extracted no frames from {input_path}")
                extraction_marker.write_text("complete\n", encoding="utf-8")

            source_paths = sorted(source_frames.glob("frame-*.png"))
        if not source_paths:
            raise UpscalingError("No source frames available for resume")
        pending = [path for path in source_paths if not (upscaled_frames / path.name).is_file()]
        self._process_frames(pending, work_dir, upscaled_frames, runtime, model_name,
                             {**config, "source_frame_paths": source_paths})
        missing_outputs = [path.name for path in source_paths if not (upscaled_frames / path.name).is_file()]
        if missing_outputs:
            raise UpscalingError(f"Real-ESRGAN did not produce {len(missing_outputs)} upscaled frames")

        concat_file = work_dir / "frames.ffconcat"
        self._write_concat(concat_file, source_paths, upscaled_frames, metadata,
                           single_images=config.get("frame_manifest") is not None)
        video_codec = str(config.get("intermediate_codec", "libx264"))
        start_time = str(metadata.get("start_time", "0"))
        command = [
            ffmpeg, "-y", "-copyts", "-itsoffset", start_time,
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(input_path), "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
            "-map_metadata", "1", "-map_chapters", "1", "-c:v", video_codec,
        ]
        if video_codec == "libx264":
            command.extend(["-crf", str(config.get("intermediate_crf", 0)),
                            "-preset", str(config.get("intermediate_preset", "medium"))])
        command.extend(["-c:a", "copy", "-c:s", "copy", "-fps_mode",
                        "passthrough" if config.get("frame_manifest") is not None else "vfr"])
        if metadata.get("sample_aspect_ratio") not in {None, "", "N/A", "1:1"}:
            command.extend(["-vf", "setsar=" + str(metadata["sample_aspect_ratio"]).replace(":", "/")])
        if metadata.get("pix_fmt") and metadata["pix_fmt"] != "unknown":
            command.extend(["-pix_fmt", str(metadata["pix_fmt"])])
        for option, value in self._color_options(metadata).items():
            command.extend([option, value])
        if config.get("faststart"):
            command.extend(["-movflags", "+faststart"])
        temporary_video = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
        command.append(str(temporary_video))
        try:
            self._run(command, "reassemble upscaled video")
            if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
                raise UpscalingError(f"FFmpeg did not produce upscaled video: {output_path}")
            if config.get("output_validator"):
                config["output_validator"](temporary_video)
            temporary_video.replace(output_path)
        finally:
            temporary_video.unlink(missing_ok=True)
        if not config.get("keep_frames", False):
            shutil.rmtree(work_dir)
        return output_path

    def _manifest_source_paths(self, input_path, work_dir, ffmpeg, manifest, scale=4):
        """Migrate legacy caches using source PTS; never enumerate cache files as frames."""
        source_dir = work_dir / "source"
        paths = [source_dir / frame["name"] for frame in manifest["frames"]]
        if not paths or len({path.name for path in paths}) != len(paths):
            raise UpscalingError("Invalid or duplicate source frame identities")
        if any(frame["name"] != f"frame-{frame['pts']:020d}.png" for frame in manifest["frames"]):
            raise UpscalingError("Noncanonical source frame identity")
        from los80.cache_migration import finish_migration, migrate_legacy_frames, has_legacy_pairs
        finish_migration(work_dir, manifest)
        needs_reassociation = (any(not (work_dir / "upscaled" / path.name).is_file() for path in paths)
                               and has_legacy_pairs(work_dir, manifest))
        if any(not path.is_file() for path in paths) or needs_reassociation:
            # Decode by ordinal, independent of FFmpeg's image encoder time base
            # and filename padding. Associate ordinals with ffprobe's source PTS.
            with tempfile.TemporaryDirectory(prefix=".extract-", dir=work_dir) as directory:
                staged = Path(directory)
                self._run([
                    ffmpeg, "-y", "-copyts", "-i", str(input_path), "-map", "0:v:0",
                    "-fps_mode", "passthrough", "-start_number", "0",
                    str(staged / "decoded-%09d.png"),
                ], "extract video frames")
                decoded = [staged / f"decoded-{index:09d}.png" for index in range(len(paths))]
                if any(not path.is_file() for path in decoded) or len(list(staged.glob("decoded-*.png"))) != len(paths):
                    raise UpscalingError(f"Extraction frame count differs from manifest ({len(paths)} expected)")
                migrated = migrate_legacy_frames(work_dir, manifest, decoded, scale)
                if not migrated:
                    for decoded_path, path in zip(decoded, paths):
                        decoded_path.replace(path)
        temporary = work_dir / "extraction-manifest.partial.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(work_dir / "extraction-manifest.json")
        (work_dir / ".extraction-complete").write_text("complete\n", encoding="utf-8")
        return paths

    def prepare(self, config):
        try:
            runtime = RealESRGANRuntime(config.get("backend_path")).ensure()
        except RealESRGANRuntimeError as exc:
            raise UpscalingError(str(exc)) from exc
        model_name = str(config.get("model", "RealESRGAN_x4plus"))
        if model_name == "RealESRGAN_x4plus":
            model_name = "realesrgan-x4plus"
        missing = [runtime.model_dir / f"{model_name}{suffix}" for suffix in (".bin", ".param")
                   if not (runtime.model_dir / f"{model_name}{suffix}").is_file()]
        if missing:
            raise UpscalingError("Required model files are missing: " + ", ".join(map(str, missing)))
        self._warn_unsupported_options(config)
        return runtime, model_name

    def _process_frames(self, pending, work_dir, upscaled_frames, runtime, model_name, config):
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
        command = self._build_ncnn_command(
            executable, batch_dir, output_dir, model_dir, model_name, gpu_id, config,
        )
        self._run(command, f"upscale frame batch {batch_number + 1}")
        shutil.rmtree(batch_dir)

    @staticmethod
    def _build_ncnn_command(executable: Path, input_dir: Path, output_dir: Path,
                            model_dir: Path, model_name: str, gpu_id: int | None,
                            config: dict[str, object]) -> list[str]:
        command = [
            str(executable), "-i", str(input_dir), "-o", str(output_dir),
            "-m", str(model_dir), "-n", model_name,
        ]
        if config.get("scale") is not None:
            command.extend(["-s", str(config["scale"])])
        if config.get("tile_size") is not None:
            command.extend(["-t", str(config["tile_size"])])
        if gpu_id is not None:
            command.extend(["-g", str(gpu_id)])
        command.extend(["-f", str(config.get("output_format", "png"))])
        if config.get("verbose"):
            command.append("-v")
        return command

    def _warn_unsupported_options(self, config: dict[str, object]) -> None:
        for option in self._UNSUPPORTED_OPTIONS:
            value = config.get(option)
            if value not in (None, False, 0, "", "0"):
                self.logger.warning(
                    "Ignoring unsupported Real-ESRGAN NCNN option %s=%r; "
                    "the value remains available to other upscaler backends",
                    option, value,
                )

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
                      metadata: dict[str, object], single_images: bool = False) -> None:
        numerator, denominator = _parse_ratio(str(metadata.get("time_base", "1/25")))
        fallback = 1.0 / _ratio(str(metadata.get("avg_frame_rate", "25/1")), 25.0)
        pts = [int(path.stem[len("frame-"):]) for path in source_paths]
        lines = ["ffconcat version 1.0"]
        for index, source in enumerate(source_paths):
            path = (output_dir / source.name).resolve()
            escaped = str(path).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
            if single_images:
                lines.extend(["option pattern_type none", "option loop 0"])
            frame_rate = str(metadata.get("avg_frame_rate", "25/1"))
            if _ratio(frame_rate, 0) <= 0:
                frame_rate = "25/1"
            lines.append(f"option framerate {frame_rate}")
            duration = (pts[index + 1] - pts[index]) * numerator / denominator if index + 1 < len(pts) else fallback
            lines.append(f"duration {max(duration, 0.000001):.9f}")
        temporary = concat_file.with_suffix(".partial.ffconcat")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(concat_file)

    @staticmethod
    def _color_options(metadata: dict[str, object]) -> dict[str, str]:
        mapping = {"color_primaries": "-color_primaries", "color_transfer": "-color_trc",
                   "color_space": "-colorspace", "color_range": "-color_range"}
        return {option: str(metadata[key]) for key, option in mapping.items()
                if metadata.get(key) and metadata[key] != "unknown"}


class AIUpscaler:
    def __init__(self, logger: Optional[logging.Logger] = None, backend_cls: Optional[type] = None) -> None:
        self.logger = logger or logging.getLogger("los80.upscaler")
        self.backend_cls = backend_cls

    def upscale(self, video_path: str | Path, output_path: str | Path, config: Optional[dict[str, object]] = None) -> Path:
        input_path = Path(video_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            self.logger.info("Using existing upscaled intermediate %s", output_file)
            return output_file

        backend_cls = self.backend_cls or select_backend()
        backend = backend_cls(self.logger) if issubclass(backend_cls, RealESRGANBackend) else backend_cls()
        if hasattr(backend, "upscale"):
            self.logger.info("Using %s backend", backend_cls.__name__)
            return backend.upscale(input_path, output_file, config or {})

        raise UpscalingError("No AI upscaler backend available")

    def detect_device(self) -> str:
        return "cuda" if cuda_available() else "cpu"


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def select_backend():
    # Never silently fall back to Vulkan after a CUDA inference/setup failure.
    return TorchRealESRGANBackend if cuda_available() else RealESRGANBackend


class TorchRealESRGANBackend(RealESRGANBackend):
    """PyTorch frame inference; extraction and remux stay in the existing backend."""

    def prepare(self, config):
        from los80.torch_realesrgan import prepare_engine
        self.engine, self.half = prepare_engine(config)
        if config.get("progress"):
            print(f"precision: {'FP16' if self.half else 'FP32'}", flush=True)
        return None, str(config.get("model", "RealESRGAN_x4plus"))

    def upscale_frame(self, input_path, output_path, config):
        import cv2
        temporary = output_path.with_name(output_path.stem + ".partial.png")
        try:
            frame = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise UpscalingError(f"Cannot decode frame: {input_path}")
            from los80.torch_realesrgan import restoration_variant, validate_restoration_settings
            scale = float(config.get("scale", 4))
            strength = float(config.get("restoration_strength", 1))
            validate_restoration_settings(scale, strength)
            started = perf_counter()
            result, _ = self.engine.enhance(frame, outscale=4 if strength != 1 else scale)
            inference_seconds = perf_counter() - started
            if strength != 1:
                result = restoration_variant(frame, result, scale, strength)
            expected = tuple(int(v * float(config.get("scale", 4))) for v in frame.shape[:2])
            if result.shape[:2] != expected:
                raise UpscalingError(f"Unexpected output resolution for {input_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(temporary), result):
                raise UpscalingError(f"Cannot write frame: {output_path}")
            temporary.replace(output_path)
            self.inference_seconds = getattr(self, "inference_seconds", 0.0) + inference_seconds
            self.processed_frames = getattr(self, "processed_frames", 0) + 1
        except Exception as exc:
            raise UpscalingError(f"CUDA frame failed: {input_path}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return output_path

    def _process_frames(self, pending, work_dir, upscaled_frames, runtime, model_name, config):
        # A single model instance owns the CUDA device; do not share it across threads.
        from PIL import Image
        sources = config.get("source_frame_paths")
        if sources is None:
            sources = sorted((work_dir / "source").glob("frame-*.png"))
        self.frame_count = len(sources)
        self.processed_frames = 0
        self.resumed_frames = 0
        self.inference_seconds = 0.0
        for index, source in enumerate(sources, 1):
            output = upscaled_frames / source.name
            try:
                with Image.open(source) as image:
                    expected = tuple(int(v * float(config.get("scale", 4))) for v in image.size)
                with Image.open(output) as image:
                    image.load()
                    if image.size == expected:
                        self.resumed_frames += 1
                        continue
            except (OSError, ValueError):
                pass
            self.upscale_frame(source, output, config)
            if config.get("progress") and (index == 1 or index % 10 == 0 or index == self.frame_count):
                print(f"frames: {index}/{self.frame_count} ({self.resumed_frames} resumed)", flush=True)


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
