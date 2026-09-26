"""Real-ESRGAN inference via Spandrel, without BasicSR's training dependencies.

Keep the enhance(image, outscale) interface used by TorchRealESRGANBackend.
Heavy imports remain lazy so NCNN-only installations can still import LOS80.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from los80.realesrgan_runtime import default_cache_dir

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_FILENAME = "RealESRGAN_x4plus.pth"


def _model_path(config: dict) -> Path:
    if config.get("model_path"):
        return Path(config["model_path"]).expanduser()
    import torch
    cache = default_cache_dir() / "torch"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / MODEL_FILENAME
    if target.is_file() and target.stat().st_size:
        return target
    # A failed/interrupted download must not become a cached model. Unique
    # temporary files also allow independent workers to populate the cache.
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".pth", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.hub.download_url_to_file(MODEL_URL, str(temporary), progress=True)
        if not temporary.stat().st_size:
            raise ValueError("Downloaded Real-ESRGAN weights are empty")
        # Validate before publishing; never load arbitrary checkpoint objects.
        torch.load(temporary, map_location="cpu", weights_only=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _load_model(path: Path):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a RealESRGAN_x4plus state dictionary")
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model = ModelLoader().load_from_state_dict(state)
    if (not isinstance(model, ImageModelDescriptor) or model.architecture.id != "ESRGAN"
            or model.scale != 4 or model.input_channels != 3 or model.output_channels != 3
            or not {"64nf", "23nb"}.issubset(model.tags)):
        raise ValueError("Weights must describe RealESRGAN_x4plus (RGB, 4x, 64 features, 23 RRDB blocks)")
    return model


def prepare_engine(config):
    from los80.upscaler import UpscalingError
    try:
        import torch
        if not torch.cuda.is_available():
            raise UpscalingError("CUDA is required for the PyTorch backend")
        model_name = str(config.get("model", "RealESRGAN_x4plus"))
        if model_name not in {"RealESRGAN_x4plus", "realesrgan-x4plus"}:
            raise UpscalingError(f"Unsupported CUDA model: {model_name}; use RealESRGAN_x4plus")
        scale = float(config.get("scale", 4))
        if not math.isfinite(scale) or not 0 < scale <= 4:
            raise UpscalingError("Real-ESRGAN output scale must be greater than zero and at most 4")
        tile, padding = int(config.get("tile_size", 0)), int(config.get("tile_padding", 10))
        if tile < 0 or padding < 0:
            raise UpscalingError("Tile size and padding must be nonnegative")
        model = _load_model(_model_path(config))
        device = torch.device("cuda")
        half = bool(config.get("half", True)) and model.supports_half and torch.cuda.get_device_capability(device) >= (5, 3)
        model.to(device=device, dtype=torch.float16 if half else torch.float32).eval()
        return SpandrelEngine(model, tile, padding), half
    except UpscalingError:
        raise
    except Exception as exc:
        raise UpscalingError(
            f"Cannot initialize CUDA Real-ESRGAN: {exc}. Install LOS80 with pip install -e '.[cuda]'"
        ) from exc


class SpandrelEngine:
    """Image conversion and padded tiles around a maintained RRDBNet loader.

    Stitch on CPU and transfer only one padded tile at a time to CUDA, keeping
    tile_size effective for T4 memory use. Model errors propagate immediately.
    """

    def __init__(self, model, tile: int = 0, tile_pad: int = 10):
        self.model = model
        self.tile = tile
        self.tile_pad = tile_pad

    def _infer_rgb(self, rgb):
        import numpy as np
        import torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
        height, width = rgb.shape[:2]
        scale = self.model.scale

        def infer(patch):
            prediction = self.model(patch.to(device=self.model.device, dtype=self.model.dtype))
            if not torch.isfinite(prediction).all():
                raise ValueError("Real-ESRGAN produced non-finite pixels; retry with half=False")
            return prediction.float().cpu()

        with torch.inference_mode():
            if not self.tile:
                result = infer(tensor)
            else:
                result = torch.empty((1, 3, height * scale, width * scale), dtype=torch.float32)
                for y in range(0, height, self.tile):
                    for x in range(0, width, self.tile):
                        right, bottom = min(x + self.tile, width), min(y + self.tile, height)
                        left_pad, top_pad = max(x - self.tile_pad, 0), max(y - self.tile_pad, 0)
                        right_pad, bottom_pad = min(right + self.tile_pad, width), min(bottom + self.tile_pad, height)
                        prediction = infer(tensor[:, :, top_pad:bottom_pad, left_pad:right_pad])
                        crop_x, crop_y = (x - left_pad) * scale, (y - top_pad) * scale
                        result[:, :, y * scale:bottom * scale, x * scale:right * scale] = prediction[
                            :, :, crop_y:crop_y + (bottom-y)*scale, crop_x:crop_x + (right-x)*scale]
            return result.squeeze(0).clamp_(0, 1).permute(1, 2, 0).numpy()

    def enhance(self, image, outscale=4):
        import cv2
        import numpy as np
        if image.dtype not in (np.uint8, np.uint16):
            raise ValueError("Real-ESRGAN input must be an 8-bit or 16-bit image")
        height, width = image.shape[:2]
        if not math.isfinite(outscale) or not 0 < outscale <= 4 or min(int(width*outscale), int(height*outscale)) < 1:
            raise ValueError("Invalid output scale for this image")
        maximum = float(np.iinfo(image.dtype).max)
        normalized = image.astype(np.float32) / maximum
        alpha = None
        if image.ndim == 2:
            mode = "L"
            rgb = np.repeat(normalized[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            mode = "RGBA" if image.shape[2] == 4 else "RGB"
            rgb = normalized[:, :, :3][:, :, ::-1]  # OpenCV BGR -> model RGB
            if mode == "RGBA":
                alpha = normalized[:, :, 3]
        else:
            raise ValueError("Unsupported input image channels")
        result = self._infer_rgb(rgb)[:, :, ::-1]  # RGB -> OpenCV BGR
        if mode == "L":
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        elif alpha is not None:
            alpha_rgb = self._infer_rgb(np.repeat(alpha[:, :, None], 3, axis=2))
            alpha_result = cv2.cvtColor(alpha_rgb, cv2.COLOR_RGB2GRAY)
            result = np.dstack((result, alpha_result))
        output = np.rint(result * maximum).astype(image.dtype)
        return restoration_variant(image, output, outscale), mode


def validate_restoration_settings(scale, strength):
    """Validate postprocessing before any expensive model work."""
    if not math.isfinite(scale) or not 0 < scale <= 4:
        raise ValueError("Output scale must be greater than zero and at most 4")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Restoration strength must be between 0 and 1")


def restoration_variant(original, neural, scale=4, strength=1.0):
    """Downsample native inference, then blend with a Lanczos source resize.

    This is image-space blending, not model denoise/weight interpolation.
    Arrays retain OpenCV channel order, bit depth and dimensions. The original
    is never resized before neural inference. Strength 1 preserves normal output.
    """
    import cv2
    import numpy as np
    validate_restoration_settings(scale, strength)
    if original.dtype not in (np.uint8, np.uint16) or original.dtype != neural.dtype:
        raise ValueError("Source and neural image must have matching uint8/uint16 types")
    if original.shape[2:] != neural.shape[2:]:
        raise ValueError("Source and neural image channels must match")
    height, width = original.shape[:2]
    target = (int(width * scale), int(height * scale))
    if min(target) < 1 or neural.shape[1] < target[0] or neural.shape[0] < target[1]:
        raise ValueError("Neural output must be at least as large as the requested output")
    restored = neural if neural.shape[:2] == target[::-1] else cv2.resize(
        neural, target, interpolation=cv2.INTER_LANCZOS4)
    if strength == 1:
        return restored
    baseline = cv2.resize(original, target, interpolation=cv2.INTER_LANCZOS4)
    if strength == 0:
        return baseline
    blended = strength * restored.astype(np.float32) + (1 - strength) * baseline.astype(np.float32)
    return np.rint(blended).clip(0, np.iinfo(original.dtype).max).astype(original.dtype)
