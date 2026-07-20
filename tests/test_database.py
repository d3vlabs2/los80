from pathlib import Path

from los80.database import JobDatabase


def test_database_tracks_stage_progress(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite"
    database = JobDatabase(db_path)

    database.add_job("sample.mp4", "/tmp/sample.mp4")
    database.mark_stage("sample.mp4", "scan", "completed", details="scanned")
    database.mark_stage("sample.mp4", "subtitles", "in_progress", details="starting")

    stages = database.get_stages("sample.mp4")
    assert stages["scan"] == "completed"
    assert stages["subtitles"] == "in_progress"
    assert database.get_job("sample.mp4").status == "processing"


def test_database_skips_duplicate_job(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite"
    database = JobDatabase(db_path)

    job_id_a = database.add_job("duplicate.mp4", "/tmp/duplicate.mp4")
    job_id_b = database.add_job("duplicate.mp4", "/tmp/duplicate.mp4")

    assert job_id_a == job_id_b
