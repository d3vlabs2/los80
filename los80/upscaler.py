from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from los80.realesrgan_runtime import RealESRGANRuntime, RealESRGANRuntimeError


class UpscalingError(RuntimeError):
    pass


class RealESRGANBackend:
    def upscale(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            runtime = RealESRGANRuntime(config.get("backend_path")).ensure()
        except RealESRGANRuntimeError as exc:
            raise UpscalingError(str(exc)) from exc

        command = [str(runtime.executable), "-i", str(input_path), "-o", str(output_path), "-m", str(runtime.model_dir)]
        if config.get("model"):
            model_name = str(config["model"])
            if model_name == "RealESRGAN_x4plus":
                model_name = "realesrgan-x4plus"
            missing = [
                runtime.model_dir / f"{model_name}{suffix}"
                for suffix in (".bin", ".param")
                if not (runtime.model_dir / f"{model_name}{suffix}").is_file()
            ]
            if missing:
                raise UpscalingError(
                    f"Required model files for {model_name} are missing: "
                    + ", ".join(str(path) for path in missing)
                )
            command.extend(["-n", model_name])
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
