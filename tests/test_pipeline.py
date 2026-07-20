from pathlib import Path

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.pipeline import Pipeline


def test_pipeline_runs_and_tracks_scan(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "clip.mp4").write_bytes(b"video")

    config = AppConfig(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        archive_dir=str(tmp_path / "archive"),
        working_dir=str(tmp_path / "working"),
        reports_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "jobs.sqlite"),
    )
    database = JobDatabase(config.database_path)
    pipeline = Pipeline(config, database)

    class FakeEncoderBackend:
        def detect_hardware_encoder(self, preference: str = "auto") -> str:
            return "libx265"

    def generate_subtitles(
        video_path: Path, spanish_path: Path, english_path: Path
    ) -> tuple[Path, Path]:
        spanish_path.write_text("Hola\n", encoding="utf-8")
        english_path.write_text("Hello\n", encoding="utf-8")
        return spanish_path, english_path

    def write_output(input_path: Path, output_path: Path, *args, **kwargs) -> Path:
        output_path.write_bytes(b"encoded")
        return output_path

    monkeypatch.setattr(
        pipeline.subtitles_service, "generate_subtitles", generate_subtitles
    )
    monkeypatch.setattr(
        pipeline.translation_service,
        "translate_subtitles",
        lambda source_path, output_path: output_path,
    )
    monkeypatch.setattr(pipeline.upscaler, "upscale", write_output)
    monkeypatch.setattr(pipeline.encoder, "encode", write_output)
    pipeline.encoder.backend_cls = FakeEncoderBackend
    monkeypatch.setattr(pipeline.validator, "validate", lambda *args, **kwargs: None)

    pipeline.run()

    job = database.get_job("clip.mp4")
    assert job.status == "completed"
    assert database.get_stages("clip.mp4")["scan"] == "completed"
