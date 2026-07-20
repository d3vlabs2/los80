from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class UpscalingError(RuntimeError):
    pass


class RealESRGANBackend:
    def upscale(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        binary = shutil.which("realesrgan-ncnn-vulkan") or shutil.which("realesrgan")
        if not binary:
            raise UpscalingError("Real-ESRGAN is not installed or not available on PATH")

        command = [binary, "-i", str(input_path), "-o", str(output_path)]
        if config.get("model"):
            command.extend(["-m", str(config["model"])])
        if config.get("tile_size"):
            command.extend(["-t", str(config["tile_size"])])
        if config.get("tile_padding"):
            command.extend(["-p", str(config["tile_padding"])])
        if config.get("face_enhance"):
            command.append("-f")
        if config.get("denoise"):
            command.append("-dn")
        if config.get("sharpen"):
            command.append("-s")
        if config.get("device") == "cuda":
            command.append("-g")
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise UpscalingError(completed.stderr.strip() or completed.stdout.strip() or f"Real-ESRGAN failed for {input_path}")
        if not output_path.exists():
            raise UpscalingError(f"Real-ESRGAN did not produce output: {output_path}")
        return output_path


class AIUpscaler:
    def __init__(self, logger: Optional[logging.Logger] = None, backend_cls: Optional[type] = None) -> None:
        self.logger = logger or logging.getLogger("los80.upscaler")
        self.backend_cls = backend_cls or RealESRGANBackend

    def upscale(self, video_path: str | Path, output_path: str | Path, config: Optional[dict[str, object]] = None) -> Path:
        input_path = Path(video_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            self.logger.info("Using existing upscaled intermediate %s", output_file)
            return output_file

        backend = self.backend_cls()
        if hasattr(backend, "upscale"):
            self.logger.info("Using %s backend", self.backend_cls.__name__)
            return backend.upscale(input_path, output_file, config or {})

        raise UpscalingError("No AI upscaler backend available")

    def detect_device(self) -> str:
        if shutil.which("nvidia-smi"):
            return "cuda"
        return "cpu"
