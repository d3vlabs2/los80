import json
from pathlib import Path
from subprocess import CompletedProcess

from los80.analysis import MediaAnalyzer, source_fingerprint
from los80.database import JobDatabase
from los80.dashboard import Dashboard
from los80.reports import write_csv_report, write_html_report


def _runner(command, **kwargs):
    if command[0] == "ffprobe":
        payload = {"streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 720, "height": 480,
             "avg_frame_rate": "30000/1001", "display_aspect_ratio": "4:3", "pix_fmt": "yuv420p",
             "field_order": "tt", "nb_frames": "3000"},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2},
        ], "format": {"duration": "100", "bit_rate": "800000", "size": "10000000", "format_name": "mov,mp4"}}
        return CompletedProcess(command, 0, json.dumps(payload), "")
    log = "crop=704:464:8:8 TFF:20 BFF:0 black_start:0 black_end:1 freeze_start: 90 lavfi.scd.time: 10 lavfi.scd.time: 20 lavfi.scd.time: 30 drop_count:12"
    return CompletedProcess(command, 0, "", log)


def test_metadata_scene_recommendation_and_score(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"media")
    analyzer = MediaAnalyzer(runner=_runner)
    result = analyzer.analyze(video)
    assert result["metadata"]["resolution"] == "720x480"
    assert result["metadata"]["audio_channels"] == 2
    assert result["scenes"]["black_borders"] is True
    assert result["scenes"]["interlaced"] is True
    assert result["recommendations"]["deinterlace"] is True
    assert 0 <= result["quality_score"] <= 100


def test_analysis_resume_storage(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"media")
    db = JobDatabase(tmp_path / "jobs.sqlite")
    db.add_job(video.name, str(video))
    analysis = MediaAnalyzer(runner=_runner).analyze(video)
    db.save_analysis(video.name, analysis)
    assert db.has_analysis(video.name, source_fingerprint(video))
    video.write_bytes(b"changed")
    assert not db.has_analysis(video.name, source_fingerprint(video))


def test_extended_reports(tmp_path: Path) -> None:
    job = {"source_name": "source.mp4", "status": "completed", "analysis": {
        "metadata": {"resolution": "720x480", "codec": "h264"},
        "quality_score": 61.5, "detected_issues": ["noise"],
        "recommendations": {"denoise": True},
        "processing_decisions": {"denoise": False},
        "previews": {"thumbnail": "source.thumbnail.jpg"},
    }}
    html = write_html_report(tmp_path / "report.html", [job]).read_text()
    csv = write_csv_report(tmp_path / "report.csv", [job]).read_text()
    assert "61.5" in html and "720x480" in html
    assert "quality_score" in csv and "source.thumbnail.jpg" in csv
    assert "detected_issues" in csv and "noise" in html
    assert 'denoise&quot;: false' in html


def test_dashboard_exposes_analysis_details() -> None:
    jobs = [{"source_name": "source.mp4", "status": "processing",
             "stages": {"analysis": "completed"}, "analysis": {
                 "quality_score": 72.0, "detected_issues": ["interlaced content"],
                 "recommendations": {"estimated_processing_time_seconds": 300.0},
                 "processing_decisions": {"deinterlace": True},
             }}]
    media = Dashboard(jobs).summary().media[0]
    assert media["quality_score"] == 72.0
    assert media["detected_issues"] == ["interlaced content"]
    assert media["recommended_processing_profile"]["deinterlace"] is True
    assert media["estimated_processing_time"] == 300.0
    assert media["analysis_status"] == "completed"


def test_preview_generation_creates_all_requested_assets(tmp_path: Path) -> None:
    source, processed = tmp_path / "source.mp4", tmp_path / "processed.mp4"
    source.write_bytes(b"source")
    processed.write_bytes(b"processed")

    def preview_runner(command, **kwargs):
        Path(command[-1]).write_bytes(b"preview")
        return CompletedProcess(command, 0, "", "")

    previews = MediaAnalyzer(runner=preview_runner).generate_previews(
        source, tmp_path / "reports", processed=processed, animated=True,
    )
    assert set(previews) == {"thumbnail", "contact_sheet", "comparison", "animated_gif"}
    assert all(Path(path).exists() for path in previews.values())


def test_quality_threshold_controls_skip_recommendation() -> None:
    metadata = {"width": 3840, "height": 2160, "codec": "h265"}
    scenes = {"blur_score": 0}
    assert MediaAnalyzer.recommend(metadata, scenes, 85, quality_threshold=80)["skip"] is True
    assert MediaAnalyzer.recommend(metadata, scenes, 85, quality_threshold=90)["skip"] is False


def test_recommendation_overrides_are_honored() -> None:
    decisions = MediaAnalyzer.recommend(
        {"width": 720, "height": 480, "codec": "h264"},
        {"interlaced": True, "excessive_noise": True, "blur_score": 40}, 30,
        overrides={"deinterlace": False, "denoise": False, "sharpen": False,
                   "model": "custom-model", "tile_size": 128,
                   "encoder_preset": "fast", "skip": True},
    )
    assert decisions == {
        "deinterlace": False, "denoise": False, "sharpen": False,
        "model": "custom-model", "tile_size": 128,
        "encoder_preset": "fast", "skip": True,
    }
