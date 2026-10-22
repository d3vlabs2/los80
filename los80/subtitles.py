from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np


class SubtitlesGenerationError(RuntimeError):
    pass


class SubtitlesService:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.logger = logger or logging.getLogger("los80.subtitles")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    def generate_subtitles(
        self,
        video_path: str | Path,
        spanish_output: str | Path,
        english_output: str | Path,
    ) -> tuple[Path, Path]:
        input_path = Path(video_path)
        spanish_file = Path(spanish_output)
        english_file = Path(english_output)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        spanish_file.parent.mkdir(parents=True, exist_ok=True)
        english_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting subtitle generation for %s", input_path.name)

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise SubtitlesGenerationError(
                "faster-whisper is required for subtitle generation. "
                "Install it with `pip install faster-whisper`."
            ) from exc

        # Decode with FFmpeg ourselves so Faster-Whisper never invokes PyAV.
        with tempfile.TemporaryDirectory(prefix="los80-whisper-") as tmp:
            wav_path = Path(tmp) / "audio.wav"

            command = [
                "ffmpeg",
                "-y",
                "-v", "error",
                "-i", str(input_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ]

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

            if completed.returncode != 0 or not wav_path.exists():
                raise SubtitlesGenerationError(
                    f"FFmpeg audio extraction failed for {input_path.name}: "
                    f"{completed.stderr.strip()}"
                )

            with wave.open(str(wav_path), "rb") as wav:
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getframerate() != 16000
                ):
                    raise SubtitlesGenerationError(
                        f"Unexpected Whisper WAV format for {input_path.name}"
                    )

                audio = np.frombuffer(
                    wav.readframes(wav.getnframes()),
                    dtype=np.int16,
                ).astype(np.float32) / 32768.0

        if audio.size == 0:
            raise SubtitlesGenerationError(
                f"No audio decoded from {input_path.name}"
            )

        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        segments, info = model.transcribe(
            audio,
            language="es",
            task="transcribe",
        )

        detected_language = getattr(info, "language", "es")
        self.logger.info(
            "Whisper language %s for %s",
            detected_language,
            input_path.name,
        )

        lines: list[str] = []

        for index, segment in enumerate(segments, start=1):
            start = self._format_timestamp(segment.start)
            end = self._format_timestamp(segment.end)
            segment_text = getattr(segment, "text", "").strip()

            if segment_text:
                lines.append(
                    f"{index}\n{start} --> {end}\n{segment_text}\n"
                )

        if not lines:
            raise SubtitlesGenerationError(
                f"No subtitle segments were generated for {input_path.name}"
            )

        spanish_text = "\n".join(lines).strip() + "\n"

        spanish_file.write_text(spanish_text, encoding="utf-8")

        # English subtitles are created by TranslationService.
        # Do not create a Spanish placeholder here because the translator
        # treats an existing English output as already completed.

        self.logger.info(
            "Completed subtitle generation for %s",
            input_path.name,
        )

        return spanish_file, english_file

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_ms = int(seconds * 1000)
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
