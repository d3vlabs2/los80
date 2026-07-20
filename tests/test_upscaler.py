from pathlib import Path

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.pipeline import Pipeline
from los80.upscaler import AIUpscaler


class FakeBackend:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def upscale(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.write_bytes(b"upscaled")
        return output_path


def test_upscaler_skips_existing_intermediate(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "clip.upscaled.mp4"
    output_path.write_bytes(b"already")

    upscaler = AIUpscaler(backend_cls=FakeBackend)
    result = upscaler.upscale(input_path, output_path, {"skip_if_target_reached": False})

    assert result == output_path
    assert output_path.read_bytes() == b"already"


def test_pipeline_resumes_upscaling_from_existing_intermediate(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    video_path = input_dir / "clip.mp4"
    video_path.write_bytes(b"video")

    working_dir = tmp_path / "working"
    working_dir.mkdir(parents=True)
    upscaled_path = working_dir / "clip.mp4" / "clip.upscaled.mp4"
    upscaled_path.parent.mkdir(parents=True)
    upscaled_path.write_bytes(b"upscaled")

    config = AppConfig(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        archive_dir=str(tmp_path / "archive"),
        working_dir=str(working_dir),
        reports_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "jobs.sqlite"),
        max_retries=1,
    )
    database = JobDatabase(config.database_path)
    database.add_job("clip.mp4", str(video_path))
    database.mark_stage("clip.mp4", "scan", "completed", "discovered")
    database.mark_stage("clip.mp4", "upscaling", "completed", "existing")

    pipeline = Pipeline(config, database)
    pipeline.upscaler.backend_cls = FakeBackend
    pipeline.run()

    assert database.get_stages("clip.mp4")["upscaling"] == "completed"
