import sys
import types
from pathlib import Path

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.pipeline import Pipeline
from los80.subtitles import SubtitlesService


class FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    def __init__(self, language: str) -> None:
        self.language = language


class FakeModel:
    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path: str, language: str | None = None, task: str | None = None):
        return [FakeSegment(0.0, 1.0, "Hola mundo"), FakeSegment(1.0, 2.0, "Adiós")], FakeInfo("es")


def test_subtitles_service_detects_spanish_and_writes_srt(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    service = SubtitlesService(model_name="base")
    spanish_path = tmp_path / "clip.es.srt"
    english_path = tmp_path / "clip.en.srt"

    service.generate_subtitles(video_path, spanish_path, english_path)

    assert spanish_path.exists()
    assert english_path.exists()
    assert "Hola mundo" in spanish_path.read_text(encoding="utf-8")
    assert "Hola mundo" in english_path.read_text(encoding="utf-8")


def test_pipeline_skips_existing_subtitle_files(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    video_path = input_dir / "clip.mp4"
    video_path.write_bytes(b"video")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "clip.es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
    (output_dir / "clip.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    config = AppConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        archive_dir=str(tmp_path / "archive"),
        working_dir=str(tmp_path / "working"),
        reports_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "jobs.sqlite"),
    )
    database = JobDatabase(config.database_path)
    database.add_job("clip.mp4", str(video_path))
    database.mark_stage("clip.mp4", "scan", "completed", "discovered")
    database.mark_stage("clip.mp4", "subtitles", "completed", "existing")
    database.mark_stage("clip.mp4", "translation", "completed", "existing")

    pipeline = Pipeline(config, database)

    def fail(*args, **kwargs):
        raise AssertionError("subtitle generation should be skipped")

    monkeypatch.setattr(pipeline.subtitles_service, "generate_subtitles", fail)
    monkeypatch.setattr(pipeline.translation_service, "translate_subtitles", fail)
    monkeypatch.setattr(pipeline.upscaler, "upscale", lambda *args, **kwargs: Path("/tmp/out.mp4"))
    monkeypatch.setattr(pipeline.encoder, "encode", lambda *args, **kwargs: Path("/tmp/out.mp4"))
    monkeypatch.setattr(pipeline.validator, "validate", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.drive, "upload", lambda *args, **kwargs: Path("/tmp/out.mp4"))

    pipeline.run()
