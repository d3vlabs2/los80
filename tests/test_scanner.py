from pathlib import Path

from los80.scanner import scan_videos


def test_scan_videos_recursively(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    (input_dir / "clip1.mp4").write_bytes(b"video")
    (nested_dir / "clip2.mkv").write_bytes(b"video")
    (nested_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    files = scan_videos(input_dir)

    paths = [path.name for path in files]
    assert "clip1.mp4" in paths
    assert "clip2.mkv" in paths
    assert "notes.txt" not in paths
