from __future__ import annotations

from pathlib import Path
import argparse

from los80.configuration import DEFAULT_CONFIG_PATH, load_config
from los80.database import JobDatabase
from los80.scanner import scan_videos
from los80.analysis import MediaAnalyzer, source_fingerprint
from los80.performance import profile_action


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    database = JobDatabase(config.database_path)
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
        print(f"Queued {source_name} (job_id={job_id})")


if __name__ == "__main__":
    main()
