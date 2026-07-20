from pathlib import Path

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.pipeline import Pipeline


def test_pipeline_runs_and_tracks_scan(tmp_path: Path) -> None:
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

    pipeline.run()

    job = database.get_job("clip.mp4")
    assert job.status == "completed"
    assert database.get_stages("clip.mp4")["scan"] == "completed"
