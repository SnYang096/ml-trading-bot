#!/usr/bin/env python3
"""
Regenerate path-primitives training report artifacts from an existing run directory.

Use case:
- You updated report rendering logic (HTML + summary), and want to re-render
  without retraining the model.

Expected run_dir layout (produced by scripts/train_path_primitives_mlp.py):
  - meta.json
  - metrics.json
  - pred_sample.csv (optional)
  - report.html (will be overwritten)
  - metrics_summary.md (will be overwritten)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import sys

# Ensure project root on sys.path so `import src.*` works when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.models.nn.path_primitives_reporting import (
    render_html_dashboard,
    _metrics_summary_md,  # type: ignore
)


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing meta.json/metrics.json",
    )
    ap.add_argument(
        "--out-html",
        default=None,
        help="Optional output html path (default: <run-dir>/report.html)",
    )
    ap.add_argument(
        "--out-summary",
        default=None,
        help="Optional output md path (default: <run-dir>/metrics_summary.md)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta_p = run_dir / "meta.json"
    metrics_p = run_dir / "metrics.json"
    if not meta_p.exists() or not metrics_p.exists():
        raise SystemExit(f"Missing meta/metrics in run_dir: {run_dir}")

    meta = _load_json(meta_p)
    metrics = _load_json(metrics_p)

    df_pred_sample: Optional[pd.DataFrame] = None
    sample_p = run_dir / "pred_sample.csv"
    if sample_p.exists():
        df_pred_sample = pd.read_csv(sample_p, index_col=0)

    html = render_html_dashboard(
        meta=meta, metrics=metrics, df_pred_sample=df_pred_sample
    )
    out_html = Path(args.out_html) if args.out_html else (run_dir / "report.html")
    out_html.write_text(html, encoding="utf-8")

    summary = _metrics_summary_md(metrics)  # type: ignore
    out_md = (
        Path(args.out_summary) if args.out_summary else (run_dir / "metrics_summary.md")
    )
    out_md.write_text(summary, encoding="utf-8")

    print("✅ Wrote:", out_html.as_posix())
    print("✅ Wrote:", out_md.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
