from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class SubtitlesGenerationError(RuntimeError):
    pass


class SubtitlesService:
    def __init__(self, logger: Optional[logging.Logger] = None, model_name: str = "base", device: str = "cpu", compute_type: str = "int8") -> None:
        self.logger = logger or logging.getLogger("los80.subtitles")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    def generate_subtitles(self, video_path: str | Path, spanish_output: str | Path, english_output: str | Path) -> tuple[Path, Path]:
        input_path = Path(video_path)
        spanish_file = Path(spanish_output)
        english_file = Path(english_output)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        spanish_file.parent.mkdir(parents=True, exist_ok=True)
        english_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting subtitle generation for %s", input_path.name)
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            self.logger.error("faster-whisper is required for subtitle generation: %s", exc)
            raise SubtitlesGenerationError("faster-whisper is required for subtitle generation. Install it with `pip install faster-whisper`.") from exc

        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        segments, info = model.transcribe(str(input_path), task="transcribe")
        detected_language = getattr(info, "language", "en")
        if detected_language != "es":
            self.logger.info("Detected language %s for %s; producing subtitles in the detected source language", detected_language, input_path.name)

        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            start = self._format_timestamp(segment.start)
            end = self._format_timestamp(segment.end)
            text = getattr(segment, "text", "").strip()
            if text:
                lines.append(f"{index}\n{start} --> {end}\n{text}\n")

        if not lines:
            raise SubtitlesGenerationError(f"No subtitle segments were generated for {input_path.name}")

        spanish_text = "\n".join(lines).strip() + "\n"
        spanish_file.write_text(spanish_text, encoding="utf-8")
        english_file.write_text(spanish_text, encoding="utf-8")
        self.logger.info("Completed subtitle generation for %s", input_path.name)
        return spanish_file, english_file

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_ms = int(seconds * 1000)
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
