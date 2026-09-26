"""One-frame diagnostic using the same backend as the video pipeline."""
from pathlib import Path
from time import perf_counter

from los80.upscaler import RealESRGANBackend, TorchRealESRGANBackend, UpscalingError, cuda_available, select_backend


def smoke_frame(input_path: Path, output_path: Path, config: dict, require_cuda=False):
    from PIL import Image
    if input_path.resolve() == output_path.resolve():
        raise UpscalingError("Smoke input and output must be different paths")
    with Image.open(input_path) as image:
        image.load()
        input_size = image.size
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
    start = perf_counter()
    runtime, model = backend.prepare(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(backend, TorchRealESRGANBackend):
        backend.upscale_frame(input_path, output_path, config)
    else:
        temporary = output_path.with_name(output_path.stem + ".partial.png")
        try:
            backend._run(RealESRGANBackend._build_ncnn_command(
                runtime.executable, input_path, temporary, runtime.model_dir, model, None,
                {**config, "output_format": "png"}), "upscale smoke frame")
            with Image.open(temporary) as image:
                image.load()
                expected = tuple(int(v * float(config.get("scale", 4))) for v in input_size)
                if image.size != expected:
                    raise UpscalingError("Unexpected NCNN output resolution")
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
    elapsed = perf_counter() - start
    with Image.open(output_path) as image:
        image.load()
        output_size = image.size
    report = {"model used": model, "precision": ("FP16" if backend.half else "FP32") if isinstance(backend, TorchRealESRGANBackend) else "backend default",
              "input resolution": f"{input_size[0]}x{input_size[1]}",
              "output resolution": f"{output_size[0]}x{output_size[1]}",
              "scale factor": config.get("scale", 4), "processing time (including model setup)": f"{elapsed:.3f}s",
              "output path": str(output_path)}
    for key, value in report.items():
        print(f"{key}: {value}")
    return report
