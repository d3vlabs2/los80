from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable


REPORT_FIELDS = [
    "source_name", "status", "metadata", "resolution", "fps", "duration", "codec",
    "audio_codec", "audio_channels", "bitrate", "aspect_ratio", "color_space",
    "container_format", "file_size", "quality_score", "detected_issues", "recommendations",
    "processing_decisions", "preview_paths", "performance",
]


def _report_row(job: dict[str, object]) -> dict[str, object]:
    analysis = job.get("analysis") if isinstance(job.get("analysis"), dict) else {}
    metadata = analysis.get("metadata", {}) if isinstance(analysis, dict) else {}
    recommendations = analysis.get("recommendations", {}) if isinstance(analysis, dict) else {}
    return {
        "source_name": job.get("source_name", ""), "status": job.get("status", ""),
        "metadata": json.dumps(metadata, sort_keys=True),
        **{key: metadata.get(key, "") for key in REPORT_FIELDS if key in metadata},
        "quality_score": analysis.get("quality_score", "") if isinstance(analysis, dict) else "",
        "detected_issues": json.dumps(analysis.get("detected_issues", []), sort_keys=True) if isinstance(analysis, dict) else "[]",
        "recommendations": json.dumps(recommendations, sort_keys=True),
        "processing_decisions": json.dumps(analysis.get("processing_decisions", recommendations), sort_keys=True) if isinstance(analysis, dict) else "{}",
        "preview_paths": json.dumps(analysis.get("previews", {}), sort_keys=True) if isinstance(analysis, dict) else "{}",
        "performance": json.dumps(job.get("performance", {}), sort_keys=True),
    }


def write_html_report(path: str | Path, jobs: Iterable[dict[str, object]]) -> Path:
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = list(jobs)
    normalized = [_report_row(row) for row in rows]
    html_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in REPORT_FIELDS) + "</tr>" for row in normalized)
    headings = "".join(f"<th>{html.escape(field.replace('_', ' ').title())}</th>" for field in REPORT_FIELDS)
    performance = next((row.get("performance") for row in rows if isinstance(row.get("performance"), dict)), {})
    performance_html = html.escape(json.dumps(performance, indent=2, sort_keys=True))
    content = f"""<!doctype html>
<html>
  <head><meta charset='utf-8'><title>LOS80 Report</title></head>
  <body>
    <h1>LOS80 Processing Report</h1>
    <table>
      <tr>{headings}</tr>
      {html_rows}
    </table>
    <h2>Performance</h2>
    <pre>{performance_html}</pre>
  </body>
</html>"""
    output_file.write_text(content, encoding="utf-8")
    return output_file


def write_csv_report(path: str | Path, jobs: Iterable[dict[str, object]]) -> Path:
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for job in jobs:
            writer.writerow(_report_row(job))
    return output_file


def write_json_report(path: str | Path, jobs: Iterable[dict[str, object]]) -> Path:
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(list(jobs), indent=2), encoding="utf-8")
    return output_file
