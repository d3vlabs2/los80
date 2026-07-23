from __future__ import annotations

from pathlib import Path
import argparse

from los80.configuration import DEFAULT_CONFIG_PATH, load_config
from los80.database import JobDatabase
from los80.scanner import scan_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LOS80 batch pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    database = JobDatabase(config.database_path)

    for video_path in scan_videos(config.input_dir):
        source_name = video_path.name
        job_id = database.add_job(source_name, str(video_path))
        database.mark_stage(source_name, "scan", "completed", details="discovered")
        print(f"Queued {source_name} (job_id={job_id})")


if __name__ == "__main__":
    main()
