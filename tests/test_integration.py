import importlib.util
import shutil
from pathlib import Path

import pytest

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.pipeline import Pipeline


REQUIRED_BINARIES = ("ffmpeg", "ffprobe", "realesrgan-ncnn-vulkan")


@pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is None
    or any(shutil.which(binary) is None for binary in REQUIRED_BINARIES),
    reason="full pipeline dependencies are not installed",
)
def test_full_pipeline_runs_with_real_dependencies(tmp_path: Path) -> None:
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    archive_dir = tmp_path / 'archive'
    working_dir = tmp_path / 'working'
    reports_dir = tmp_path / 'reports'
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    working_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    sample_video = input_dir / 'sample.mp4'
    sample_video.write_bytes(b'video')

    config = AppConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        archive_dir=str(archive_dir),
        working_dir=str(working_dir),
        reports_dir=str(reports_dir),
        database_path=str(tmp_path / 'jobs.sqlite'),
        max_retries=0,
        include_translation=False,
        include_subtitles=True,
        drive_enabled=False,
    )
    database = JobDatabase(config.database_path)
    pipeline = Pipeline(config, database)

    pipeline.run()

    assert (output_dir / 'sample.es.srt').exists()
    assert (output_dir / 'sample.en.srt').exists()
    assert (output_dir / 'sample.mp4').exists()
    assert (archive_dir / 'sample.mp4').exists()
