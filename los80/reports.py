from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable


def write_html_report(path: str | Path, jobs: Iterable[dict[str, object]]) -> Path:
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = list(jobs)
    html_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('source_name', '')))}</td><td>{html.escape(str(row.get('status', '')))}</td></tr>"
        for row in rows
    )
    content = f"""<!doctype html>
<html>
  <head><meta charset='utf-8'><title>LOS80 Report</title></head>
  <body>
    <h1>LOS80 Processing Report</h1>
    <table>
      <tr><th>Source</th><th>Status</th></tr>
      {html_rows}
    </table>
  </body>
</html>"""
    output_file.write_text(content, encoding="utf-8")
    return output_file


def write_csv_report(path: str | Path, jobs: Iterable[dict[str, object]]) -> Path:
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_name", "status"])
        writer.writeheader()
        for job in jobs:
            writer.writerow({"source_name": job.get("source_name", ""), "status": job.get("status", "")})
    return output_file


def write_json_report(path: str | Path, jobs: Iterable[dict[str, object]]) -> Path:
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(list(jobs), indent=2), encoding="utf-8")
    return output_file
