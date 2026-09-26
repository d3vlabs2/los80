from __future__ import annotations

from pathlib import Path
import argparse
import logging

from los80.configuration import DEFAULT_CONFIG_PATH, load_config
from los80.database import JobDatabase
from los80.scanner import scan_videos
from los80.analysis import MediaAnalyzer, source_fingerprint
from los80.performance import profile_action
from los80.realesrgan_runtime import RealESRGANRuntime, RealESRGANRuntimeError
from los80.pipeline import Pipeline
from los80.upscaler import UpscalingError, cuda_available, TorchRealESRGANBackend
from los80.translator import TranslationError, TranslationService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LOS80 batch pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command")
    analyze = subparsers.add_parser("analyze", help="Analyze discovered videos before processing")
    analyze.add_argument("--force", action="store_true", help="Replace existing analysis results")
    profile = subparsers.add_parser("profile", help="Profile LOS80 discovery and database operations")
    profile.add_argument("--output-dir", type=Path, default=None, help="Directory for profiling artifacts")
    subparsers.add_parser("doctor", help="Check and prepare the Real-ESRGAN runtime")
    smoke = subparsers.add_parser("smoke", help="Upscale one image without running the video pipeline")
    smoke.add_argument("--input", type=Path, default=Path("/content/los80_smoke/input.png"))
    smoke.add_argument("--output", type=Path, default=Path("/content/los80_smoke/output.png"))
    smoke.add_argument("--require-cuda", action="store_true")
    smoke.add_argument("--tile-size", type=int, default=256)
    smoke.add_argument("--scale", type=float, default=None, help="Output scale (neural model remains 4x)")
    smoke.add_argument("--strength", type=float, default=1.0, help="Output blend: 0=Lanczos original, 1=neural result")
    compare = subparsers.add_parser("smoke-compare", help="Compare one frame with one shared 4x inference")
    compare.add_argument("--input", type=Path, default=Path("/content/los80_smoke/input.png"))
    compare.add_argument("--output-dir", type=Path, default=Path("/content/los80_smoke/comparison"))
    compare.add_argument("--require-cuda", action="store_true")
    compare.add_argument("--tile-size", type=int, default=256)
    compare.add_argument("--scales", nargs="+", type=int, choices=(2, 3, 4), default=[2, 3, 4])
    compare.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.0])
    compare.add_argument("--contact-sheet", action="store_true")
    compare.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                         help="Contact-sheet crop in source pixels; full frame still used for inference")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command in {"smoke", "smoke-compare"}:
        from los80.smoke import smoke_frame, smoke_compare
        options = {
            "backend_path": config.realesrgan_backend_path,
            "model": config.upscaler_model, "scale": config.upscaler_scale,
            "tile_size": args.tile_size, "tile_padding": config.tile_padding,
        }
        try:
            if args.command == "smoke-compare":
                smoke_compare(args.input, args.output_dir, options, scales=args.scales,
                              strengths=args.strengths, require_cuda=args.require_cuda,
                              contact_sheet=args.contact_sheet, crop=args.crop)
            else:
                options["scale"] = args.scale if args.scale is not None else config.upscaler_scale
                options["restoration_strength"] = args.strength
                smoke_frame(args.input, args.output, options, require_cuda=args.require_cuda)
        except (UpscalingError, OSError, ValueError) as exc:
            raise SystemExit(f"Smoke test failed: {exc}") from exc
        return
    if args.command == "doctor":
        failed = False
        runtime = RealESRGANRuntime(getattr(config, "realesrgan_backend_path", None))
        try:
            if cuda_available():
                backend = TorchRealESRGANBackend()
                backend.prepare({"model": config.upscaler_model, "tile_size": config.tile_size})
                info = None
                print(f"Real-ESRGAN: CUDA/PyTorch ({'FP16' if backend.half else 'FP32'})")
            else:
                info = runtime.ensure()
        except (RealESRGANRuntimeError, UpscalingError) as exc:
            print(f"Real-ESRGAN failed: {exc}")
            failed = True
        else:
            if info is not None:
                print(f"Real-ESRGAN executable: {info.executable}")
                print(f"Model weights: {info.model_dir}")
                print(f"Version: {info.version}")
                print(f"Cache location: {info.cache_dir}")
        translator = TranslationService(
            model_name=getattr(config, "translation_model", "facebook/nllb-200-distilled-600M"),
            device=getattr(config, "translation_device", "auto"),
            batch_size=getattr(config, "translation_batch_size", 8),
        )
        try:
            translation = translator.prepare()
        except TranslationError as exc:
            print(f"\u2717 Translation model: {exc}")
            failed = True
        else:
            print(f"\u2713 Translation model installed: {translation['model']}")
            print(f"\u2713 Translation cache location: {translation['cache_dir']}")
            print(f"\u2713 Translation device: {translation['device']}")
        if failed:
            raise SystemExit(1)
        return
    if args.command is None and not cuda_available():
        try:
            RealESRGANRuntime(getattr(config, "realesrgan_backend_path", None)).ensure()
        except RealESRGANRuntimeError as exc:
            raise SystemExit(f"Real-ESRGAN startup check failed: {exc}") from exc
    database = JobDatabase(config.database_path)

    if args.command is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        summary = Pipeline(config, database).run()
        elapsed = float(summary.get("elapsed_time", 0.0))
        print("\nLOS80 processing summary")
        print(f"completed: {summary.get('completed', 0)}")
        print(f"skipped: {summary.get('skipped', 0)}")
        print(f"failed: {summary.get('failed', 0)}")
        print(f"elapsed time: {elapsed:.2f}s")
        return

    analyzer = MediaAnalyzer()

    if args.command == "profile":
        output_dir = args.output_dir or Path(config.reports_dir) / "performance"

        def workload() -> None:
            for video_path in scan_videos(config.input_dir, config.supported_extensions):
                database.add_job(video_path.name, str(video_path))
                database.get_stages(video_path.name)

        outputs = profile_action(workload, output_dir)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    for video_path in scan_videos(config.input_dir):
        source_name = video_path.name
        job_id = database.add_job(source_name, str(video_path))
        database.mark_stage(source_name, "scan", "completed", details="discovered")
        if args.command == "analyze":
            fingerprint = source_fingerprint(video_path)
            if not args.force and database.has_analysis(source_name, fingerprint):
                print(f"Skipped {source_name} (analysis exists)")
                continue
            overrides = {
                "deinterlace": getattr(config, "analysis_deinterlace", None),
                "denoise": getattr(config, "analysis_denoise", None),
                "sharpen": getattr(config, "analysis_sharpen", None),
                "model": getattr(config, "analysis_model", None),
                "tile_size": getattr(config, "analysis_tile_size", None),
                "encoder_preset": getattr(config, "analysis_encoder_preset", None),
                "skip": getattr(config, "analysis_skip", None),
            }
            result = analyzer.analyze(
                video_path, overrides,
                quality_threshold=float(getattr(config, "analysis_quality_threshold", 80.0)),
            )
            result["previews"] = analyzer.generate_previews(
                video_path, config.reports_dir,
                animated=getattr(config, "analysis_generate_gif", False),
            )
            database.save_analysis(source_name, result)
            database.mark_stage(source_name, "analysis", "completed", details="analyzed")
            print(f"Analyzed {source_name} (quality={result['quality_score']})")
            continue


if __name__ == "__main__":
    main()
