"""
ReportGenerator
===============
Aggregates all pipeline outputs into a unified final report:
  - Data quality stats
  - Cleaning summary
  - Augmentation summary
  - Alignment stats
  - Evaluation metrics
  - Error analysis
  - Linguistic resource summary

Outputs HTML and Markdown versions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive pipeline reports."""

    def __init__(self, project_name: str = "runyoro-nmt-v1"):
        self.project_name = project_name
        self.sections: Dict[str, str] = {}
        self.metadata: Dict = {
            "project": project_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def add_section(self, title: str, content: str) -> None:
        self.sections[title] = content

    def add_data_stats(self, stats: Dict) -> None:
        lines = [
            "## Data Pipeline Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
        ]
        for k, v in stats.items():
            lines.append(f"| {k} | {v} |")
        self.add_section("data_stats", "\n".join(lines))

    def add_eval_results(self, results: Dict) -> None:
        lines = [
            "## Evaluation Results (runyoro-nmt-v1)",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]
        for k, v in results.items():
            lines.append(f"| {k} | {v} |")
        self.add_section("eval_results", "\n".join(lines))

    def generate_markdown(self, output_path: Optional[str] = None) -> str:
        header = f"""# {self.project_name} — Pipeline Report

**Generated:** {self.metadata['generated_at']}

---
"""
        body = "\n\n".join(self.sections.values())
        report = header + body

        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("Report saved: %s", output_path)

        return report

    def generate_html(self, output_path: Optional[str] = None) -> str:
        """Wrap the markdown report in a minimal HTML shell for easy viewing."""
        md = self.generate_markdown()
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.project_name} Report</title>
  <style>
    body {{ font-family: Inter, sans-serif; max-width: 900px; margin: 40px auto;
            padding: 0 20px; color: #191c1e; background: #f7f9fb; }}
    h1 {{ color: #070235; }} h2 {{ color: #070235; border-bottom: 2px solid #86f2e4; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #c8c5d0; padding: 8px 12px; text-align: left; }}
    th {{ background: #eceef0; }}
    pre {{ background: #f2f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; }}
    code {{ background: #e6e8ea; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
<pre>{md}</pre>
</body>
</html>"""

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("HTML report saved: %s", output_path)
        return html
