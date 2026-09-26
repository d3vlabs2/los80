"""Short-video diagnostic using LOS80's existing CUDA video/frame backend."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from los80.torch_realesrgan import validate_restoration_settings
from los80.upscaler import TorchRealESRGANBackend, UpscalingError, cuda_available


def _probe_motion(path: Path, ffprobe: str):
    completed = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,time_base,sample_aspect_ratio:frame=best_effort_timestamp,best_effort_timestamp_time",
        "-of", "json", str(path),
    ], check=False, capture_output=True, text=True)
    if completed.returncode:
        raise UpscalingError(f"Cannot inspect motion clip {path}: {completed.stderr.strip()}")
    try:
        data = json.loads(completed.stdout)
        video = data["streams"][0]
        pts = [int(frame["best_effort_timestamp"]) for frame in data["frames"]]
        video["frame_pts"] = pts
        timestamps = [value * Fraction(video["time_base"]) for value in pts]
        if not timestamps or Fraction(video["avg_frame_rate"]) <= 0:
            raise ValueError("No frames or invalid FPS")
        return video, timestamps
    except (KeyError, ValueError, IndexError, ZeroDivisionError) as exc:
        raise UpscalingError(f"Invalid motion metadata for {path}") from exc


def _audio_packets(path: Path, ffprobe: str):
    completed = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "a", "-show_packets", "-show_data_hash", "sha256",
        "-show_entries", "packet=stream_index,pts_time,duration_time,data_hash", "-of", "json", str(path),
    ], check=False, capture_output=True, text=True)
    if completed.returncode:
        raise UpscalingError(f"Cannot inspect audio in {path}: {completed.stderr.strip()}")
    return json.loads(completed.stdout).get("packets", [])


def _validate_motion(path, source_video, source_times, source_audio, scale, ffprobe):
    video, times = _probe_motion(path, ffprobe)
    expected = (int(source_video["width"] * scale), int(source_video["height"] * scale))
    if (video["width"], video["height"]) != expected or video["codec_name"] != "h264":
        raise UpscalingError("Motion output resolution/codec does not match the requested H.264 restoration")
    if Fraction(video["avg_frame_rate"]) != Fraction(source_video["avg_frame_rate"]):
        raise UpscalingError("Motion output FPS differs from the source")
    if len(times) != len(source_times):
        raise UpscalingError(f"Motion output frame count differs from the source: expected {len(source_times)}, got {len(times)}")
    tolerance = 2 * max(Fraction(video["time_base"]), Fraction(source_video["time_base"]))
    if any(abs(output - source) > tolerance for output, source in zip(times, source_times)):
        raise UpscalingError("Motion output frame timestamps differ from the source")
    def sar(stream):
        value = stream.get("sample_aspect_ratio", "1:1")
        return Fraction(value.replace(":", "/")) if value not in {"N/A", "0:1"} else Fraction(1)
    if sar(video) != sar(source_video):
        raise UpscalingError("Motion output sample aspect ratio differs from the source")
    if _audio_packets(path, ffprobe) != source_audio:
        raise UpscalingError("Motion output audio packets/timing differ from the source")
    return video, times


def smoke_motion(input_path: Path, output_path: Path | None = None, config: dict | None = None,
                 require_cuda: bool = False):
    config = dict(config or {})
    scale = float(config.get("scale", 2))
    strength = float(config.get("restoration_strength", 0.25))
    validate_restoration_settings(scale, strength)
    output_path = output_path or input_path.parent / f"restored_{scale:g}x_strength{strength:.2f}.mp4"
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Motion input and output must be different paths")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Motion smoke output must be an MP4")
    # Motion smoke deliberately exercises CUDA. NCNN remains available through
    # the normal pipeline and still-image commands, with no fallback on failure.
    if not cuda_available():
        raise UpscalingError("CUDA unavailable; motion smoke requires a Colab GPU runtime")
    import torch
    gpu = torch.cuda.get_device_name(0)
    model = "RealESRGAN_x4plus"
    ffprobe = str(config.get("ffprobe_binary", "ffprobe"))
    started = perf_counter()
    source_video, source_times = _probe_motion(input_path, ffprobe)
    frame_pts = source_video["frame_pts"]
    if any(right <= left for left, right in zip(frame_pts, frame_pts[1:])):
        raise UpscalingError("Motion source frame timestamps must be unique and increasing")
    frame_manifest = {"schema": 1, "time_base": source_video["time_base"],
                      "fps": source_video["avg_frame_rate"],
                      "frames": [{"pts": pts, "name": f"frame-{pts:020d}.png"} for pts in frame_pts]}
    source_audio = _audio_packets(input_path, ffprobe)
    digest = hashlib.sha256()
    with input_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    identity = {"schema": 1, "source_sha256": digest.hexdigest(), "model": model,
                "native_scale": 4, "scale": scale, "strength": strength, "half": True,
                "tile_size": int(config.get("tile_size", 256)), "tile_padding": int(config.get("tile_padding", 10))}
    validated = {}
    def validate(path):
        validated["video"], validated["times"] = _validate_motion(
            path, source_video, source_times, source_audio, scale, ffprobe)
    options = {**config, "model": model, "half": True, "face_enhance": False,
               "scale": scale, "restoration_strength": strength,
               "tile_size": identity["tile_size"], "tile_padding": identity["tile_padding"],
               "intermediate_codec": "libx264", "intermediate_crf": 16,
               "intermediate_preset": "slow", "faststart": True, "keep_frames": True,
               "resume_identity": identity, "frame_manifest": frame_manifest,
               "output_validator": validate, "progress": True}
    backend = TorchRealESRGANBackend()
    print(f"backend: {type(backend).__name__}\nGPU: {gpu}\nmodel: {model}", flush=True)
    print(f"input resolution: {source_video['width']}x{source_video['height']}\n"
          f"source FPS: {source_video['avg_frame_rate']}\nnumber of source frames: {len(source_times)}", flush=True)
    backend.upscale(input_path, output_path, options)
    video = validated["video"]
    average = (f"{backend.inference_seconds / backend.processed_frames:.3f}s"
               if backend.processed_frames else "not measured (all frames resumed)")
    report = {"backend": type(backend).__name__, "GPU": gpu, "model": model,
              "precision": "FP16" if backend.half else "FP32",
              "input resolution": f"{source_video['width']}x{source_video['height']}",
              "output resolution": f"{video['width']}x{video['height']}",
              "source FPS": source_video["avg_frame_rate"], "output FPS": video["avg_frame_rate"],
              "number of frames": len(validated["times"]), "frames inferred this run": backend.processed_frames,
              "frames resumed": backend.resumed_frames, "processing time": f"{perf_counter()-started:.3f}s",
              "average inference time per new frame": average, "output path": str(output_path)}
    for key, value in report.items():
        print(f"{key}: {value}")
    return report
