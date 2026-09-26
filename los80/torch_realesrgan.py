"""Lazy loading of the official Real-ESRGAN PyTorch inference engine."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from los80.realesrgan_runtime import default_cache_dir


def prepare_engine(config):
    from los80.upscaler import UpscalingError
    try:
        import torch
        if not torch.cuda.is_available():
            raise UpscalingError("CUDA is required for the PyTorch backend")
        # BasicSR 1.4.2 imports this module removed by recent torchvision.
        try:
            importlib.import_module("torchvision.transforms.functional_tensor")
        except ModuleNotFoundError as exc:
            if exc.name != "torchvision.transforms.functional_tensor":
                raise
            sys.modules["torchvision.transforms.functional_tensor"] = importlib.import_module(
                "torchvision.transforms.functional")
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        model_name = str(config.get("model", "RealESRGAN_x4plus"))
        if model_name not in {"RealESRGAN_x4plus", "realesrgan-x4plus"}:
            raise UpscalingError(f"Unsupported CUDA model: {model_name}; use RealESRGAN_x4plus")
        scale = float(config.get("scale", 4))
        if not 0 < scale <= 4:
            raise UpscalingError("Real-ESRGAN output scale must be greater than zero and at most 4")
        model_path = config.get("model_path")
        if not model_path:
            model_path = load_file_from_url(
                url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model_dir=str(default_cache_dir() / "torch"), progress=True,
                file_name="RealESRGAN_x4plus.pth")
        half = bool(config.get("half", True))
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        engine = RealESRGANer(scale=4, model_path=str(Path(model_path)), model=model,
                             tile=int(config.get("tile_size", 0)),
                             tile_pad=int(config.get("tile_padding", 10)), pre_pad=0,
                             half=half, device=torch.device("cuda"))
        # Upstream tile_process catches RuntimeError and can reuse the previous
        # tile after an OOM. Convert model failures so they always abort the frame.
        forward = engine.model.forward
        def checked_forward(*args, **kwargs):
            try:
                return forward(*args, **kwargs)
            except RuntimeError as exc:
                raise UpscalingError(f"CUDA model inference failed: {exc}") from exc
        engine.model.forward = checked_forward
        return engine, half
    except UpscalingError:
        raise
    except Exception as exc:
        raise UpscalingError(
            f"Cannot initialize CUDA Real-ESRGAN: {exc}. Install LOS80 with pip install -e '.[cuda]'"
        ) from exc
