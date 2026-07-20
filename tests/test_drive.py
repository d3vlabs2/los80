from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from los80.drive import GoogleDriveClient, TransferError


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeTransport:
    def __init__(self, *, files: dict | None = None, fail_once: bool = False) -> None:
        self.files = files or {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail_once = fail_once
        self._failed = False

    def request(self, method: str, url: str, *, headers: dict | None = None, params: dict | None = None, data: bytes | None = None, json_body: dict | None = None) -> FakeResponse:
        self.calls.append((method, url, json_body))
        if self.fail_once and not self._failed:
            self._failed = True
            raise RuntimeError("temporary failure")
        if method == "POST" and url.endswith("/files"):
            return FakeResponse(200, {"id": "file-1", "name": "sample.mp4", "size": 3, "md5Checksum": hashlib.md5(b"abc").hexdigest(), "kind": "drive#file"})
        if method == "PATCH":
            return FakeResponse(200, {"id": "file-1", "name": "sample.mp4", "size": 3, "md5Checksum": hashlib.md5(b"abc").hexdigest(), "kind": "drive#file"})
        if method == "GET":
            if self.files:
                return FakeResponse(200, {"files": [{"id": "existing", "name": "clip.mp4", "size": 3, "md5Checksum": hashlib.md5(b"abc").hexdigest()}]})
            return FakeResponse(200, {"files": []})
        return FakeResponse(200, {})


def test_authentication_uses_explicit_token(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    client = GoogleDriveClient(token_path=token_path, access_token="abc-token")

    token = client.authenticate()

    assert token == "abc-token"
    assert token_path.exists()


def test_upload_verifies_and_reports_success(tmp_path: Path) -> None:
    local_path = tmp_path / "clip.mp4"
    local_path.write_bytes(b"abc")
    transport = FakeTransport()
    client = GoogleDriveClient(transport=transport, access_token="token")

    result = client.upload_file(local_path, "clip.mp4", parent_id="root", verify=True)

    assert result["status"] == "uploaded"
    assert result["verified"] is True
    assert result["remote_id"] == "file-1"


def test_download_skips_existing_cached_file(tmp_path: Path) -> None:
    local_path = tmp_path / "clip.mp4"
    local_path.write_bytes(b"abc")
    transport = FakeTransport()
    client = GoogleDriveClient(transport=transport, access_token="token")

    result = client.download_file("file-1", local_path, verify=True)

    assert result["status"] == "skipped"
    assert len(transport.calls) == 0


def test_resume_uses_existing_upload_session(tmp_path: Path) -> None:
    local_path = tmp_path / "clip.mp4"
    local_path.write_bytes(b"abc")
    session_file = tmp_path / "clip.mp4.part"
    session_file.write_text(json.dumps({"session_uri": "https://example.test/session", "offset": 0}), encoding="utf-8")
    transport = FakeTransport()
    client = GoogleDriveClient(transport=transport, access_token="token")

    result = client.upload_file(local_path, "clip.mp4", parent_id="root", verify=True)

    assert result["status"] == "uploaded"
    assert any(call[0] == "PATCH" for call in transport.calls)


def test_duplicate_detection_skips_upload(tmp_path: Path) -> None:
    local_path = tmp_path / "clip.mp4"
    local_path.write_bytes(b"abc")
    transport = FakeTransport(files={"clip.mp4": {"id": "existing", "size": 3, "md5Checksum": hashlib.md5(b"abc").hexdigest()}})
    client = GoogleDriveClient(transport=transport, access_token="token")

    result = client.upload_file(local_path, "clip.mp4", parent_id="root", verify=True)

    assert result["status"] == "skipped"
    assert result["reason"] == "duplicate"


def test_verification_failure_raises(tmp_path: Path) -> None:
    local_path = tmp_path / "clip.mp4"
    local_path.write_bytes(b"abc")
    transport = FakeTransport(files={"clip.mp4": {"id": "existing", "size": 4, "md5Checksum": hashlib.md5(b"xyz").hexdigest()}})
    client = GoogleDriveClient(transport=transport, access_token="token")

    with pytest.raises(TransferError):
        client.upload_file(local_path, "clip.mp4", parent_id="root", verify=True)


def test_retry_logic_retries_failed_transfers(tmp_path: Path) -> None:
    local_path = tmp_path / "clip.mp4"
    local_path.write_bytes(b"abc")
    transport = FakeTransport(fail_once=True)
    client = GoogleDriveClient(transport=transport, access_token="token", max_retries=2)

    result = client.upload_file(local_path, "clip.mp4", parent_id="root", verify=True)

    assert result["status"] == "uploaded"
    assert len(transport.calls) >= 2


def test_archive_moves_file_only_after_success(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"abc")
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    client = GoogleDriveClient(access_token="token")

    archived_path = client.archive_source(source_path, archive_dir)

    assert archived_path.exists()
    assert not source_path.exists()
    assert archived_path.parent == archive_dir
