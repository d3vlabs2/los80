from pathlib import Path

import pytest

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.encoder import EncoderService, FFmpegBackend
from los80.pipeline import Pipeline
from los80.validator import ValidationError, Validator


class FakeBackend:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def encode(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.write_bytes(b"encoded")
        return output_path


def test_ffmpeg_detection_uses_available_binary(monkeypatch) -> None:
    backend = FFmpegBackend()
    monkeypatch.setattr("los80.encoder.shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert backend.detect_binary("ffmpeg") == "/usr/bin/ffmpeg"


def test_detect_hardware_encoder_prefers_hevc(monkeypatch) -> None:
    backend = FFmpegBackend()
    monkeypatch.setattr(backend, "_run_command", lambda *args, **kwargs: "hevc_nvenc\nhevc_qsv\nlibx265\n")
    assert backend.detect_hardware_encoder() == "hevc_nvenc"


def test_encoder_falls_back_to_libx265(monkeypatch) -> None:
    backend = FFmpegBackend()
    monkeypatch.setattr(backend, "detect_hardware_encoder", lambda: "libx265")
    monkeypatch.setattr("los80.encoder.shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr("los80.encoder.subprocess.run", lambda *args, **kwargs: type("Result", (), {"returncode": 0})())
    output = backend.encode(Path("/tmp/input.mp4"), Path("/tmp/out.mp4"), {"codec": "libx265", "hardware_preference": "auto"})
    assert output.exists()


def test_validation_failure_when_output_missing(tmp_path: Path) -> None:
    validator = Validator()
    with pytest.raises(ValidationError):
        validator.validate(tmp_path / "source.mp4", tmp_path / "missing.mp4", tmp_path / "es.srt", tmp_path / "en.srt")


def test_pipeline_skips_encoding_for_completed_stage(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    video_path = input_dir / "clip.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    encoded_path = output_dir / "clip.mp4"
    encoded_path.write_bytes(b"encoded")
    (output_dir / "clip.es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
    (output_dir / "clip.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    config = AppConfig(input_dir=str(input_dir), output_dir=str(output_dir), archive_dir=str(tmp_path / "archive"), working_dir=str(tmp_path / "working"), reports_dir=str(tmp_path / "reports"), database_path=str(tmp_path / "jobs.sqlite"))
    database = JobDatabase(config.database_path)
    database.add_job("clip.mp4", str(video_path))
    database.mark_stage("clip.mp4", "scan", "completed", "discovered")
    database.mark_stage("clip.mp4", "encoding", "completed", "existing")

    pipeline = Pipeline(config, database)
    pipeline.encoder.backend_cls = FakeBackend
    pipeline.run()

    assert database.get_stages("clip.mp4")["encoding"] == "completed"
