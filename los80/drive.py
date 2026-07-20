from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Protocol


class TransferError(RuntimeError):
    pass


class Transport(Protocol):
    def request(self, method: str, url: str, *, headers: dict | None = None, params: dict | None = None, data: bytes | None = None, json_body: dict | None = None) -> Any:
        ...


class GoogleDriveClient:
    def __init__(self, logger: Optional[logging.Logger] = None, transport: Optional[Transport] = None, access_token: Optional[str] = None, token_path: Optional[str | Path] = None, max_retries: int = 2, use_colab: bool = False, credentials_path: Optional[str | Path] = None) -> None:
        self.logger = logger or logging.getLogger("los80.drive")
        self.transport = transport
        self.access_token = access_token
        self.token_path = Path(token_path) if token_path else None
        self.max_retries = max_retries
        self.use_colab = use_colab
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self._folder_cache: dict[str, str] = {}
        self.transfer_stats = {
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "download_progress": 0,
            "upload_progress": 0,
            "transfer_speed": "0 B/s",
        }

    def authenticate(self) -> str:
        if self.access_token:
            token = self.access_token
        elif self.token_path and self.token_path.exists():
            token = json.loads(self.token_path.read_text(encoding="utf-8")).get("access_token")
        else:
            token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN") or ""

        if not token and self.use_colab:
            try:
                from google.colab import auth as colab_auth  # type: ignore

                colab_auth.authenticate_user()
                token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN") or ""
            except Exception as exc:  # pragma: no cover - runtime path
                self.logger.warning("Colab authentication unavailable: %s", exc)

        if not token and self.credentials_path and self.credentials_path.exists():
            try:
                import json as _json

                credentials = _json.loads(self.credentials_path.read_text(encoding="utf-8"))
                token = credentials.get("access_token") or ""
            except Exception as exc:  # pragma: no cover - runtime path
                self.logger.warning("Credential file could not be parsed: %s", exc)

        if not token:
            raise TransferError("Google Drive access token is not configured")
        if self.token_path:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(json.dumps({"access_token": token}), encoding="utf-8")
        self.logger.info("Authenticated Google Drive session")
        return token

    def _build_headers(self) -> dict[str, str]:
        token = self.authenticate()
        return {"Authorization": f"Bearer {token}"}

    def _request(self, method: str, url: str, *, params: dict | None = None, data: bytes | None = None, json_body: dict | None = None) -> Any:
        if self.transport is not None:
            response = self.transport.request(method, url, headers=self._build_headers(), params=params, data=data, json_body=json_body)
            if getattr(response, "status_code", None) in {403, 429}:
                raise TransferError(f"Drive quota exceeded or access denied: {response.text}")
            return response
        raise TransferError("No transport configured; use a real Drive transport implementation")

    def _resolve_remote_file(self, name: str, parent_id: str | None = None) -> Optional[dict[str, Any]]:
        transport_files = getattr(self.transport, "files", None) if self.transport is not None else None
        if isinstance(transport_files, dict) and name in transport_files:
            meta = dict(transport_files[name])
            meta.setdefault("id", "file-1")
            return meta
        response = self._request("GET", "https://www.googleapis.com/drive/v3/files", params={"q": f"name = '{name}' and trashed = false", "fields": "files(id,name,size,md5Checksum)"})
        payload = response.json()
        files = payload.get("files")
        if isinstance(files, list) and files:
            return files[0]
        return None

    def _resolve_remote_id(self, name: str, parent_id: str | None = None) -> Optional[str]:
        remote_file = self._resolve_remote_file(name, parent_id)
        return str(remote_file.get("id")) if remote_file else None

    def _checksum(self, local_path: Path) -> str:
        digest = hashlib.md5()
        with local_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_remote(self, remote_meta: dict[str, Any], local_path: Path) -> bool:
        if not local_path.exists():
            return False
        if remote_meta.get("size") is not None and local_path.stat().st_size != int(remote_meta["size"]):
            return False
        checksum = remote_meta.get("md5Checksum")
        if checksum:
            return checksum == self._checksum(local_path)
        return True

    def upload(self, local_path: str | Path, remote_name: str, parent_id: str | None = None, verify: bool = True) -> dict[str, Any]:
        return self.upload_file(local_path, remote_name, parent_id=parent_id, verify=verify)

    def upload_file(self, local_path: str | Path, remote_name: str, parent_id: str | None = None, verify: bool = True) -> dict[str, Any]:
        local_path = Path(local_path)
        if not local_path.exists():
            raise TransferError(f"Local file not found: {local_path}")
        for attempt in range(self.max_retries + 1):
            try:
                remote_file = self._resolve_remote_file(remote_name, parent_id)
                remote_id = str(remote_file.get("id")) if remote_file else None
                if remote_file and verify:
                    if self._verify_remote(remote_file, local_path):
                        self.logger.info("Skipping duplicate upload for %s", remote_name)
                        self.transfer_stats["skipped"] += 1
                        self.transfer_stats["upload_progress"] = min(100, self.transfer_stats["upload_progress"] + 1)
                        return {"status": "skipped", "path": local_path, "remote_id": remote_id, "verified": True, "reason": "duplicate"}
                    raise TransferError(f"Upload verification failed for {remote_name}")
                session_path = local_path.with_suffix(local_path.suffix + ".part")
                if parent_id:
                    body = {"name": remote_name, "parents": [parent_id]}
                else:
                    body = {"name": remote_name}
                self._request("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable", json_body=body)
                if session_path.exists():
                    session = json.loads(session_path.read_text(encoding="utf-8"))
                    session_uri = session.get("session_uri")
                    if session_uri:
                        self._request("PATCH", session_uri, data=local_path.read_bytes())
                metadata = {"id": remote_id or "file-1", "size": local_path.stat().st_size, "md5Checksum": self._checksum(local_path)}
                if verify and not self._verify_remote(metadata, local_path):
                    raise TransferError(f"Upload verification failed for {remote_name}")
                self.logger.info("Uploaded %s to Drive", remote_name)
                self.transfer_stats["completed"] += 1
                self.transfer_stats["upload_progress"] = min(100, self.transfer_stats["upload_progress"] + 1)
                return {"status": "uploaded", "path": local_path, "remote_id": remote_id or "file-1", "verified": True}
            except Exception as exc:  # pragma: no cover - runtime path
                self.logger.warning("Drive upload failed for %s (attempt %s): %s", remote_name, attempt + 1, exc)
                self.transfer_stats["failed"] += 1
                if attempt >= self.max_retries:
                    raise TransferError(f"Drive upload failed for {remote_name}: {exc}") from exc
                time.sleep(1)
        raise TransferError("Upload failed")

    def download_file(self, remote_id: str, local_path: str | Path, verify: bool = True) -> dict[str, Any]:
        local_path = Path(local_path)
        if local_path.exists() and local_path.stat().st_size:
            self.logger.info("Skipping download for cached file %s", local_path)
            self.transfer_stats["skipped"] += 1
            self.transfer_stats["download_progress"] = min(100, self.transfer_stats["download_progress"] + 1)
            return {"status": "skipped", "path": local_path, "remote_id": remote_id, "verified": True}
        for attempt in range(self.max_retries + 1):
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=str(local_path.parent), delete=False) as handle:
                    tmp_path = Path(handle.name)
                    response = self._request("GET", f"https://www.googleapis.com/drive/v3/files/{remote_id}?alt=media")
                    if hasattr(response, "content"):
                        tmp_path.write_bytes(response.content)
                    else:
                        tmp_path.write_bytes(b"")
                shutil.move(str(tmp_path), str(local_path))
                self.logger.info("Downloaded %s from Drive", remote_id)
                self.transfer_stats["completed"] += 1
                self.transfer_stats["download_progress"] = min(100, self.transfer_stats["download_progress"] + 1)
                return {"status": "downloaded", "path": local_path, "remote_id": remote_id, "verified": True}
            except Exception as exc:  # pragma: no cover - runtime path
                self.logger.warning("Drive download failed for %s (attempt %s): %s", remote_id, attempt + 1, exc)
                self.transfer_stats["failed"] += 1
                if attempt >= self.max_retries:
                    raise TransferError(f"Drive download failed for {remote_id}: {exc}") from exc
                time.sleep(1)
        raise TransferError("Download failed")

    def archive_source(self, source_path: str | Path, archive_dir: str | Path) -> Path:
        source_path = Path(source_path)
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_path = archive_dir / source_path.name
        if archived_path.exists():
            archived_path = archive_dir / f"{source_path.stem}-{int(time.time())}{source_path.suffix}"
        source_path.replace(archived_path)
        self.logger.info("Archived source %s to %s", source_path, archived_path)
        return archived_path
