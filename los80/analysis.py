from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


def source_fingerprint(path: str | Path) -> str:
    path = Path(path)
    stat = path.stat()
    value = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode()).hexdigest()


def _ratio(value: str | None) -> float:
    try:
        left, right = (value or "0/1").split("/", 1)
        return float(left) / float(right)
    except (ValueError, ZeroDivisionError):
        return 0.0


class MediaAnalyzer:
    """Fast, deterministic preflight analysis using FFprobe/FFmpeg."""

    def __init__(self, logger: logging.Logger | None = None,
                 runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
                 ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg") -> None:
        self.logger = logger or logging.getLogger("los80.analysis")
        self.runner, self.ffprobe, self.ffmpeg = runner or subprocess.run, ffprobe, ffmpeg

    def extract_metadata(self, path: str | Path) -> dict[str, Any]:
        source = Path(path)
        result = self.runner([
            self.ffprobe, "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(source)
        ], capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        fmt = probe.get("format", {})
        width, height = int(video.get("width", 0)), int(video.get("height", 0))
        fps = _ratio(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        return {
            "filename": source.name, "resolution": f"{width}x{height}",
            "width": width, "height": height, "fps": round(fps, 3),
            "duration": float(fmt.get("duration") or video.get("duration") or 0),
            "codec": video.get("codec_name", "unknown"),
            "audio_codec": audio.get("codec_name", "none"),
            "audio_channels": int(audio.get("channels", 0)),
            "bitrate": int(fmt.get("bit_rate") or video.get("bit_rate") or 0),
            "aspect_ratio": video.get("display_aspect_ratio") or (f"{width}:{height}" if height else "unknown"),
            "color_space": video.get("color_space") or video.get("pix_fmt", "unknown"),
            "container_format": fmt.get("format_name", source.suffix.lstrip(".")),
            "file_size": int(fmt.get("size") or source.stat().st_size),
            "frame_count": int(video.get("nb_frames") or 0),
            "field_order": video.get("field_order", "unknown"),
        }

    def analyze_scenes(self, path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        # A bounded sample makes analysis practical even for feature-length sources.
        duration = min(float(metadata.get("duration", 0)), 300.0)
        cmd = [self.ffmpeg, "-hide_banner", "-t", str(duration), "-i", str(path),
               "-vf", "cropdetect=24:16:0,idet,blackdetect=d=0.5:pix_th=0.10,mpdecimate,freezedetect=n=-50dB:d=2,scdet=t=10",
               "-an", "-f", "null", "-"]
        result = self.runner(cmd, capture_output=True, text=True, check=False)
        log = (result.stderr or "") + (result.stdout or "")
        crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", log)
        black = re.findall(r"black_start:([\d.]+).*?black_end:([\d.]+)", log)
        scene_times = [float(v) for v in re.findall(r"lavfi\.scd\.time: ([\d.]+)", log)]
        freezes = [float(v) for v in re.findall(r"freeze_start: ([\d.]+)", log)]
        repeated = sum(int(v) for v in re.findall(r"drop_count:(\d+)", log))
        tff = sum(int(v) for v in re.findall(r"TFF:(\d+)", log))
        bff = sum(int(v) for v in re.findall(r"BFF:(\d+)", log))
        total_frames = max(int(metadata.get("frame_count", 0)), 1)
        bpp = int(metadata.get("bitrate", 0)) / max(metadata.get("width", 1) * metadata.get("height", 1) * metadata.get("fps", 1), 1)
        borders = False
        if crops:
            cw, ch, _, _ = map(int, crops[-1])
            borders = cw < int(metadata.get("width", cw)) - 4 or ch < int(metadata.get("height", ch)) - 4
        return {
            "black_borders": borders, "crop": ":".join(crops[-1]) if crops else None,
            "interlaced": metadata.get("field_order") not in (None, "unknown", "progressive") or (tff + bff) > 5,
            "duplicate_frames": repeated, "duplicate_ratio": round(repeated / total_frames, 4),
            "heavy_compression_artifacts": bpp < 0.04, "excessive_noise": bpp > 0.35,
            "blur_score": 35.0 if int(metadata.get("height", 0)) < 720 else 15.0,
            "scene_changes": scene_times, "static_scenes": freezes,
            "black_segments": [{"start": float(a), "end": float(b)} for a, b in black],
            "credits": bool(freezes and max(freezes) > float(metadata.get("duration", 0)) * .8),
            "intro_sequence": len([t for t in scene_times if t <= 120]) >= 3,
        }

    @staticmethod
    def quality_score(metadata: dict[str, Any], scenes: dict[str, Any]) -> float:
        height = int(metadata.get("height", 0))
        resolution = min(35.0, height / 1080 * 35)
        pixels_per_second = max(int(metadata.get("width", 0)) * height * float(metadata.get("fps", 0)), 1)
        bitrate_density = int(metadata.get("bitrate", 0)) / pixels_per_second
        compression = min(25.0, bitrate_density / .12 * 25)
        noise = 10.0 if scenes.get("excessive_noise") else 15.0
        blur = max(0.0, 15.0 - float(scenes.get("blur_score", 0)) * .15)
        consistency = max(0.0, 10.0 - float(scenes.get("duplicate_ratio", 0)) * 100)
        return round(max(0.0, min(100.0, resolution + compression + noise + blur + consistency)), 1)

    @staticmethod
    def recommend(metadata: dict[str, Any], scenes: dict[str, Any], score: float,
                  overrides: dict[str, Any] | None = None,
                  quality_threshold: float = 80.0) -> dict[str, Any]:
        decisions = {
            "deinterlace": bool(scenes.get("interlaced")),
            "denoise": bool(scenes.get("excessive_noise") or score < 45),
            "sharpen": bool(float(scenes.get("blur_score", 0)) >= 25),
            "model": "RealESRGAN_x4plus_anime_6B" if metadata.get("codec") in {"gif"} else "RealESRGAN_x4plus",
            "tile_size": 256 if int(metadata.get("width", 0)) >= 1920 else 512,
            "encoder_preset": "slow" if score >= 65 else "medium",
            "skip": bool(score >= quality_threshold),
        }
        for key, value in (overrides or {}).items():
            if value is not None and key in decisions:
                decisions[key] = value
        return decisions

    @staticmethod
    def detected_issues(scenes: dict[str, Any]) -> list[str]:
        issue_fields = {
            "black_borders": "black borders", "interlaced": "interlaced content",
            "heavy_compression_artifacts": "heavy compression artifacts",
            "excessive_noise": "excessive noise", "credits": "credits",
            "intro_sequence": "intro sequence",
        }
        issues = [label for key, label in issue_fields.items() if scenes.get(key)]
        if int(scenes.get("duplicate_frames", 0)) > 0:
            issues.append("duplicate frames")
        if scenes.get("static_scenes"):
            issues.append("static scenes")
        if float(scenes.get("blur_score", 0)) >= 25:
            issues.append("blur")
        return issues

    def analyze(self, path: str | Path, overrides: dict[str, Any] | None = None,
                quality_threshold: float = 80.0) -> dict[str, Any]:
        metadata = self.extract_metadata(path)
        scenes = self.analyze_scenes(path, metadata)
        score = self.quality_score(metadata, scenes)
        recs = self.recommend(metadata, scenes, score, overrides, quality_threshold)
        duration = float(metadata.get("duration", 0))
        recs["estimated_processing_time_seconds"] = round(duration * (0.15 if recs["skip"] else 3.0), 1)
        recs["estimated_output_size_bytes"] = int(duration * max(int(metadata.get("bitrate", 0)) * .65, 1) / 8)
        return {"source_fingerprint": source_fingerprint(path), "metadata": metadata,
                "scenes": scenes, "detected_issues": self.detected_issues(scenes),
                "recommendations": recs, "processing_decisions": dict(recs),
                "quality_score": score, "previews": {}}

    def generate_previews(self, source: str | Path, reports_dir: str | Path,
                          processed: str | Path | None = None, animated: bool = False) -> dict[str, str]:
        source, reports = Path(source), Path(reports_dir)
        reports.mkdir(parents=True, exist_ok=True)
        stem = source.stem
        outputs: dict[str, str] = {}
        jobs = {
            "thumbnail": (["-ss", "00:00:05", "-i", str(source), "-frames:v", "1", "-vf", "scale=640:-2"], reports / f"{stem}.thumbnail.jpg"),
            "contact_sheet": (["-i", str(source), "-vf", "fps=1/60,scale=320:-2,tile=4x3", "-frames:v", "1"], reports / f"{stem}.contact-sheet.jpg"),
        }
        if processed is not None and Path(processed).exists():
            jobs["comparison"] = (["-i", str(source), "-i", str(processed), "-filter_complex",
                "[0:v]select='eq(n,0)',scale=640:-2[a];[1:v]select='eq(n,0)',scale=640:-2[b];[a][b]hstack", "-frames:v", "1"], reports / f"{stem}.comparison.jpg")
        if animated:
            jobs["animated_gif"] = (["-ss", "00:00:05", "-t", "5", "-i", str(source), "-vf", "fps=5,scale=480:-2:flags=lanczos"], reports / f"{stem}.preview.gif")
        for name, (args, output) in jobs.items():
            result = self.runner([self.ffmpeg, "-hide_banner", "-y", *args, str(output)], capture_output=True, text=True, check=False)
            if result.returncode == 0 and output.exists():
                outputs[name] = str(output)
        return outputs
