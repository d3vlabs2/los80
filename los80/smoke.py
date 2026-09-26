"""One-frame diagnostics using the same backend as the video pipeline."""
from pathlib import Path
import tempfile
from time import perf_counter

from los80.torch_realesrgan import restoration_variant, validate_restoration_settings
from los80.upscaler import RealESRGANBackend, TorchRealESRGANBackend, UpscalingError, cuda_available, select_backend


def _prepare_backend(config, require_cuda):
    available = cuda_available()
    if require_cuda and not available:
        raise UpscalingError("CUDA unavailable; select a GPU runtime in Colab")
    backend = select_backend()()
    gpu = "none"
    if available:
        import torch
        gpu = torch.cuda.get_device_name(0)
    print(f"backend selected: {type(backend).__name__}", flush=True)
    print(f"CUDA availability: {available}\nGPU name: {gpu}", flush=True)
    runtime, model = backend.prepare(config)
    return backend, runtime, model


def _upscale_frame(backend, runtime, model, input_path, output_path, config):
    if isinstance(backend, TorchRealESRGANBackend):
        return backend.upscale_frame(input_path, output_path, config)
    from PIL import Image
    temporary = output_path.with_name(output_path.stem + ".partial.png")
    try:
        backend._run(RealESRGANBackend._build_ncnn_command(
            runtime.executable, input_path, temporary, runtime.model_dir, model, None,
            {**config, "output_format": "png"}), "upscale smoke frame")
        with Image.open(input_path) as source, Image.open(temporary) as image:
            image.load()
            expected = tuple(int(v * float(config.get("scale", 4))) for v in source.size)
            if image.size != expected:
                raise UpscalingError("Unexpected NCNN output resolution")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _read_image(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise UpscalingError(f"Cannot decode frame: {path}")
    return image


def _atomic_png(path, image):
    """Write only complete PNGs, keeping an existing output on write failure."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if not cv2.imwrite(str(temporary), image):
            raise UpscalingError(f"Cannot write frame: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _precision(backend):
    return ("FP16" if backend.half else "FP32") if isinstance(backend, TorchRealESRGANBackend) else "backend default"


def smoke_frame(input_path: Path, output_path: Path, config: dict, require_cuda=False):
    from PIL import Image
    if input_path.resolve() == output_path.resolve():
        raise UpscalingError("Smoke input and output must be different paths")
    scale, strength = float(config.get("scale", 4)), float(config.get("restoration_strength", 1))
    validate_restoration_settings(scale, strength)
    with Image.open(input_path) as image:
        image.load()
        input_size = image.size
    start = perf_counter()
    backend, runtime, model = _prepare_backend(config, require_cuda)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if strength == 1:
        _upscale_frame(backend, runtime, model, input_path, output_path, config)
    else:
        with tempfile.TemporaryDirectory(prefix=".smoke-", dir=output_path.parent) as directory:
            native_path = Path(directory) / "native.png"
            _upscale_frame(backend, runtime, model, input_path, native_path, {**config, "scale": 4})
            result = restoration_variant(_read_image(input_path), _read_image(native_path), scale, strength)
            _atomic_png(output_path, result)
    elapsed = perf_counter() - start
    with Image.open(output_path) as image:
        image.load()
        output_size = image.size
    report = {"model used": model, "precision": _precision(backend),
              "input resolution": f"{input_size[0]}x{input_size[1]}",
              "output resolution": f"{output_size[0]}x{output_size[1]}",
              "scale factor": config.get("scale", 4), "restoration strength (output blend)": strength,
              "processing time (including model setup)": f"{elapsed:.3f}s", "output path": str(output_path)}
    for key, value in report.items():
        print(f"{key}: {value}")
    return report


def _strength_label(value):
    return f"{value:.2f}" if round(value, 2) == value else str(value)


def _contact_sheet(input_path, variants, destination, crop=None):
    """Common display size for fair comparison; crop coordinates refer to input."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import cv2
    import numpy as np
    entries = [(input_path, "Original (Lanczos display only)", 1)] + [
        (item["path"], f"{item['model']} | {item['scale']}x | blend {_strength_label(item['strength'])}", item["scale"])
        for item in variants]
    with Image.open(input_path) as original:
        source_width, source_height = original.size
    width, height = (crop[2], crop[3]) if crop else (source_width, source_height)
    cell_width, label_height, columns = 512, 64, 4
    image_height = max(1, min(512, round(cell_width * height / width)))
    sheet = Image.new("RGB", (columns * cell_width, ((len(entries) + columns - 1) // columns) * (image_height + label_height)), "#202020")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for index, (path, label, scale) in enumerate(entries):
        array = _read_image(path)
        if array.dtype == np.uint16:
            array = np.rint(array.astype(np.float32) / 257).astype(np.uint8)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA if array.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        image = Image.fromarray(array).convert("RGB")
        if crop:
            x, y, w, h = crop
            image = image.crop((x * scale, y * scale, (x+w) * scale, (y+h) * scale))
        image = ImageOps.contain(image, (cell_width, image_height), Image.Resampling.LANCZOS)
        x, y = (index % columns) * cell_width, (index // columns) * (image_height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x+8, y+6), label, fill="white", font=font)
        draw.text((x+8, y+30), f"Source crop {crop}" if crop else "Full frame - open individual PNG for 100% view", fill="#cccccc", font=font)
    _atomic_png(destination, cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR))


def smoke_compare(input_path: Path, output_dir: Path, config: dict, scales=(2, 3, 4),
                  strengths=(0.25, 0.5, 0.75, 1.0), require_cuda=False, contact_sheet=False, crop=None):
    """One native 4x inference, shared by every requested scale/strength pair."""
    scales, strengths = list(dict.fromkeys(scales)), list(dict.fromkeys(strengths))
    if not scales or not strengths or any(scale not in (2, 3, 4) for scale in scales):
        raise ValueError("Comparison requires scales from 2, 3, 4 and at least one strength")
    for strength in strengths:
        validate_restoration_settings(4, strength)
    original = _read_image(input_path)
    height, width = original.shape[:2]
    if crop is not None:
        if not contact_sheet:
            raise ValueError("--crop requires --contact-sheet")
        x, y, w, h = crop
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
            raise ValueError("Contact-sheet crop must fit inside the original frame")
    model = str(config.get("model", "RealESRGAN_x4plus"))
    if model not in {"RealESRGAN_x4plus", "realesrgan-x4plus"}:
        raise ValueError("Comparison supports RealESRGAN_x4plus")
    model = "RealESRGAN_x4plus"
    variants = [{"model": model, "scale": scale, "strength": strength,
                 "path": output_dir / f"{model}_scale{scale}_strength{_strength_label(strength)}.png"}
                for scale in scales for strength in strengths]
    sheet_path = output_dir / ("contact-sheet-crop.png" if crop else "contact-sheet.png")
    targets = [item["path"] for item in variants] + ([sheet_path] if contact_sheet else [])
    if any(path.resolve() == input_path.resolve() for path in targets):
        raise ValueError("Comparison output must not overwrite the input")
    start = perf_counter()
    native_config = {**config, "model": model, "scale": 4}
    backend, runtime, selected_model = _prepare_backend(native_config, require_cuda)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=output_dir) as directory:
        native_path = Path(directory) / "native.png"
        _upscale_frame(backend, runtime, selected_model, input_path, native_path, native_config)
        neural = _read_image(native_path)
        for item in variants:
            output = restoration_variant(original, neural, item["scale"], item["strength"])
            _atomic_png(item["path"], output)
            print(f"{output.shape[1]}x{output.shape[0]} | {item['scale']}x | strength {item['strength']:.2f} | {item['path']}")
    if contact_sheet:
        _contact_sheet(input_path, variants, sheet_path, crop)
        print(f"contact sheet: {sheet_path}")
    print(f"model used: {model}\nprecision: {_precision(backend)}\ninput resolution: {width}x{height}")
    print("Neural inference: one native 4x pass; strengths are output blends with Lanczos, not native denoise settings")
    print(f"processing time (including model setup and comparisons): {perf_counter()-start:.3f}s")
    return {"variants": variants, "contact_sheet": sheet_path if contact_sheet else None}
