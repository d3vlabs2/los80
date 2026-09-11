from __future__ import annotations

import os
import platform
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_VERSION = "v0.2.5.0"
RELEASE_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
EXECUTABLE_NAME = "realesrgan-ncnn-vulkan"
REQUIRED_MODELS = ("realesrgan-x4plus.bin", "realesrgan-x4plus.param")


class RealESRGANRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeInfo:
    executable: Path
    model_dir: Path
    version: str
    cache_dir: Path


def default_cache_dir() -> Path:
    if Path("/content").is_dir() or "COLAB_RELEASE_TAG" in os.environ:
        return Path("/content/.cache/los80/realesrgan")
    return Path.home() / ".cache" / "los80" / "realesrgan"


class RealESRGANRuntime:
    def __init__(self, backend_path: str | Path | None = None, cache_dir: str | Path | None = None) -> None:
        self.backend_path = Path(backend_path).expanduser() if backend_path else None
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()

    @property
    def version_dir(self) -> Path:
        return self.cache_dir / SUPPORTED_VERSION

    def ensure(self) -> RuntimeInfo:
        if self.backend_path:
            return self._inspect_override()
        existing = self.inspect()
        if existing:
            return existing
        self._install()
        installed = self.inspect()
        if not installed:
            raise RealESRGANRuntimeError(
                f"Real-ESRGAN {SUPPORTED_VERSION} was extracted but its executable or model weights are missing in {self.version_dir}"
            )
        return installed

    def inspect(self) -> RuntimeInfo | None:
        root = self.version_dir
        executable = _find_file(root, EXECUTABLE_NAME)
        model_dir = _find_model_dir(root)
        marker = root / ".version"
        if not executable or not model_dir or not marker.is_file():
            return None
        if marker.read_text(encoding="utf-8").strip() != SUPPORTED_VERSION:
            return None
        _ensure_executable(executable)
        return RuntimeInfo(executable, model_dir, SUPPORTED_VERSION, self.cache_dir)

    def _inspect_override(self) -> RuntimeInfo:
        assert self.backend_path is not None
        executable = self.backend_path
        root = executable.parent
        if executable.is_dir():
            root = executable
            executable = _find_file(root, EXECUTABLE_NAME) or root / EXECUTABLE_NAME
        if not executable.is_file():
            raise RealESRGANRuntimeError(f"Configured Real-ESRGAN executable does not exist: {executable}")
        model_dir = _find_model_dir(root)
        if not model_dir:
            raise RealESRGANRuntimeError(
                f"Real-ESRGAN model weights are missing near {executable} (required: {', '.join(REQUIRED_MODELS)})"
            )
        _ensure_executable(executable)
        return RuntimeInfo(executable, model_dir, "configured", self.cache_dir)

    def _install(self) -> None:
        if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
            raise RealESRGANRuntimeError(
                "Automatic Real-ESRGAN installation supports Linux x86_64; set realesrgan_backend_path in config.yaml"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="realesrgan-", dir=self.cache_dir) as temporary:
                temporary_dir = Path(temporary)
                archive = temporary_dir / "release.zip"
                try:
                    urllib.request.urlretrieve(RELEASE_URL, archive)
                except (OSError, urllib.error.URLError) as exc:
                    raise RealESRGANRuntimeError(
                        f"Failed to download Real-ESRGAN {SUPPORTED_VERSION} from {RELEASE_URL}: {exc}"
                    ) from exc
                extracted = temporary_dir / "extracted"
                extracted.mkdir()
                try:
                    with zipfile.ZipFile(archive) as bundle:
                        _safe_extract(bundle, extracted)
                except (OSError, zipfile.BadZipFile) as exc:
                    raise RealESRGANRuntimeError(f"Failed to extract Real-ESRGAN archive {archive}: {exc}") from exc
                if not _find_file(extracted, EXECUTABLE_NAME) or not _find_model_dir(extracted):
                    raise RealESRGANRuntimeError(
                        "Downloaded Real-ESRGAN archive is incomplete (executable or required model weights missing)"
                    )
                (extracted / ".version").write_text(SUPPORTED_VERSION + "\n", encoding="utf-8")
                if self.version_dir.exists():
                    shutil.rmtree(self.version_dir)
                extracted.replace(self.version_dir)
        except RealESRGANRuntimeError:
            raise
        except OSError as exc:
            raise RealESRGANRuntimeError(f"Failed to install Real-ESRGAN in {self.cache_dir}: {exc}") from exc


def _find_file(root: Path, name: str) -> Path | None:
    if not root.is_dir():
        return None
    return next((path for path in root.rglob(name) if path.is_file()), None)


def _find_model_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    for param in root.rglob(REQUIRED_MODELS[1]):
        if all((param.parent / name).is_file() for name in REQUIRED_MODELS):
            return param.parent
    return None


def _ensure_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        if not mode & stat.S_IXUSR:
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as exc:
        raise RealESRGANRuntimeError(f"Cannot set executable permissions on {path}: {exc}") from exc
    if not os.access(path, os.X_OK):
        raise RealESRGANRuntimeError(f"Real-ESRGAN executable is not executable: {path}")


def _safe_extract(bundle: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in bundle.infolist():
        target = (destination / member.filename).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise RealESRGANRuntimeError(f"Unsafe path in Real-ESRGAN archive: {member.filename}")
    bundle.extractall(destination)
