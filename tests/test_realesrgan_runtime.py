from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from los80.realesrgan_runtime import (
    REQUIRED_MODELS,
    SUPPORTED_VERSION,
    RealESRGANRuntime,
    RealESRGANRuntimeError,
)


def _release_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("release/realesrgan-ncnn-vulkan", "binary")
        for model in REQUIRED_MODELS:
            bundle.writestr(f"release/models/{model}", "weights")


def test_runtime_downloads_once_and_sets_executable(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.zip"
    _release_archive(source)
    downloads = []

    def download(url: str, destination: Path) -> None:
        downloads.append(url)
        Path(destination).write_bytes(source.read_bytes())

    monkeypatch.setattr("urllib.request.urlretrieve", download)
    runtime = RealESRGANRuntime(cache_dir=tmp_path / "cache")

    first = runtime.ensure()
    second = runtime.ensure()

    assert first == second
    assert first.version == SUPPORTED_VERSION
    assert os.access(first.executable, os.X_OK)
    assert len(downloads) == 1


def test_override_requires_model_weights(tmp_path: Path) -> None:
    executable = tmp_path / "realesrgan-ncnn-vulkan"
    executable.write_text("binary", encoding="utf-8")

    with pytest.raises(RealESRGANRuntimeError, match="model weights are missing"):
        RealESRGANRuntime(executable).ensure()
