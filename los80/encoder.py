from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence


class EncodingError(RuntimeError):
    pass


class FFmpegBackend:
    def __init__(self, ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe", logger: Optional[logging.Logger] = None) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.logger = logger or logging.getLogger("los80.encoder")

    def detect_binary(self, binary_name: str) -> Optional[str]:
        resolved = shutil.which(binary_name)
        return str(resolved) if resolved else None

    def _run_command(self, command: Sequence[str]) -> str:
        completed = subprocess.run(list(command), check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise EncodingError(completed.stderr.strip() or completed.stdout.strip() or "ffmpeg command failed")
        return completed.stdout.strip()

    def detect_hardware_encoder(self, preference: str = "auto") -> str:
        encoders_output = self._run_command([self.ffmpeg_binary, "-encoders"]) if self.detect_binary(self.ffmpeg_binary) else ""
        candidates = ["hevc_nvenc", "hevc_qsv", "hevc_vaapi", "hevc_videotoolbox", "libx265"]
        if preference in {"nvenc", "qsv", "vaapi", "videotoolbox"}:
            candidates = [f"hevc_{preference}"] + [candidate for candidate in candidates if candidate != f"hevc_{preference}"]
        for candidate in candidates:
            if candidate in encoders_output:
                return candidate
        return "libx265"

    def encode(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = self.detect_binary(self.ffmpeg_binary)
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg is not installed or not available on PATH")

        ffprobe_path = self.detect_binary(self.ffprobe_binary)
        if not ffprobe_path:
            self.logger.warning("FFprobe is not available; continuing without stream probing")

        codec = str(config.get("codec", "libx265"))
        preset = str(config.get("preset", "medium"))
        crf = str(config.get("crf", 28))
        bitrate = str(config.get("bitrate", "0"))
        audio_codec = str(config.get("audio_codec", "aac"))
        audio_bitrate = str(config.get("audio_bitrate", "128k"))
        hardware_preference = str(config.get("hardware_preference", "auto")).lower()
        embed_subtitles = bool(config.get("embed_subtitles", False))
        copy_metadata = bool(config.get("copy_metadata", True))
        preserve_chapters = bool(config.get("preserve_chapters", True))
        preserve_color_space = bool(config.get("preserve_color_space", True))
        overwrite_existing = bool(config.get("overwrite_existing", True))
        subtitle_path = Path(str(config.get("subtitle_path", ""))) if config.get("subtitle_path") else None

        try:
            selected_encoder = self.detect_hardware_encoder(hardware_preference)
        except TypeError:
            selected_encoder = self.detect_hardware_encoder()
        command: list[str] = [ffmpeg_path, "-y" if overwrite_existing else "-n", "-i", str(input_path)]
        if selected_encoder in {"hevc_nvenc", "hevc_qsv", "hevc_vaapi", "hevc_videotoolbox"}:
            command.extend(["-c:v", selected_encoder])
        else:
            command.extend(["-c:v", codec])
        if preset:
            command.extend(["-preset", preset])
        if crf:
            command.extend(["-crf", crf])
        if bitrate and bitrate != "0":
            command.extend(["-b:v", bitrate])
        command.extend(["-c:a", audio_codec])
        if audio_bitrate:
            command.extend(["-b:a", audio_bitrate])
        if copy_metadata:
            command.append("-map_metadata")
            command.append("0")
        if preserve_chapters:
            command.append("-map_chapters")
            command.append("0")
        if preserve_color_space:
            command.append("-pix_fmt")
            command.append("yuv420p")
        if embed_subtitles and subtitle_path and subtitle_path.exists():
            command.extend(["-i", str(subtitle_path)])
            command.extend(["-map", "0:v:0", "-map", "0:a?", "-map", "0:s?", "-map", "1:s?", "-c:s", "mov_text"])
        else:
            command.extend(["-map", "0:v:0", "-map", "0:a?", "-map", "0:s?"])
        command.extend(["-movflags", "+faststart", str(output_path)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise EncodingError(completed.stderr.strip() or completed.stdout.strip() or f"FFmpeg encoding failed for {input_path}")
        if not output_path.exists():
            raise EncodingError(f"FFmpeg did not produce output: {output_path}")
        return output_path


class EncoderService:
    def __init__(self, logger: Optional[logging.Logger] = None, backend_cls: Optional[type] = None) -> None:
        self.logger = logger or logging.getLogger("los80.encoder")
        self.backend_cls = backend_cls or FFmpegBackend

    def encode(self, video_path: str | Path, output_path: str | Path, config: Optional[dict[str, object]] = None) -> Path:
        input_path = Path(video_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            self.logger.info("Using existing intermediate encoded file %s", output_file)
            return output_file
        backend = self.backend_cls()
        if hasattr(backend, "encode"):
            return backend.encode(input_path, output_file, config or {})
        raise RuntimeError("FFmpeg encoder backend does not implement encode")
