from pathlib import Path

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.encoder import EncoderService
from los80.pipeline import Pipeline
from los80.validator import ValidationError, Validator


class FakeBackend:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def encode(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.write_bytes(b"encoded")
        return output_path


def test_encoder_skips_existing_intermediate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "encoded.mp4"
    output_path.write_bytes(b"encoded")

    encoder = EncoderService(backend_cls=FakeBackend)
    result = encoder.encode(input_path, output_path, {"codec": "libx265"})

    assert result == output_path
    assert output_path.read_bytes() == b"encoded"


def test_validation_requires_expected_codec(tmp_path: Path) -> None:
    validator = Validator()
    source_path = tmp_path / "source.mp4"
    encoded_path = tmp_path / "encoded.mp4"
    source_path.write_bytes(b"video")
    encoded_path.write_bytes(b"encoded")
    (tmp_path / "spanish.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
    (tmp_path / "english.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    try:
        validator.validate(source_path, encoded_path, tmp_path / "spanish.srt", tmp_path / "english.srt", codec="av1")
    except ValidationError as exc:
        assert "codec" in str(exc).lower()
    else:
        raise AssertionError("Expected validation failure")


def test_pipeline_resumes_encoding_from_existing_output(
    tmp_path: Path, monkeypatch
) -> None:
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

    config = AppConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        archive_dir=str(tmp_path / "archive"),
        working_dir=str(tmp_path / "working"),
        reports_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "jobs.sqlite"),
        include_subtitles=False,
        include_translation=False,
    )
    database = JobDatabase(config.database_path)
    database.add_job("clip.mp4", str(video_path))
    database.mark_stage("clip.mp4", "scan", "completed", "discovered")
    database.mark_stage("clip.mp4", "encoding", "completed", "existing")

    pipeline = Pipeline(config, database)
    monkeypatch.setattr(
        pipeline.upscaler,
        "upscale",
        lambda input_path, output_path, options: output_path.write_bytes(b"upscaled")
        or output_path,
    )
    monkeypatch.setattr(
        pipeline.encoder,
        "encode",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed encoding stage should be skipped")
        ),
    )
    monkeypatch.setattr(pipeline.validator, "validate", lambda *args, **kwargs: None)
    pipeline.run()

    assert database.get_stages("clip.mp4")["encoding"] == "completed"
