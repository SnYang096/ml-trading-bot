"""Emit rd_loop / research scan suggestions when drift monitor ALERTs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _probe_grid_from_item(item: Dict[str, Any]) -> str:
    p25 = item.get("window_p25")
    p50 = item.get("window_p50")
    p75 = item.get("window_p75")
    vals = [v for v in (p25, p50, p75) if isinstance(v, (int, float))]
    if len(vals) >= 2:
        lo, hi = min(vals), max(vals)
        mid = vals[len(vals) // 2]
        return f"{lo:.6g},{mid:.6g},{hi:.6g}"
    return "0,0.5,1"


def build_rd_loop_snippet(
    *,
    strategy: str,
    drift_items: List[Dict[str, Any]],
    features_parquet: str,
    output_dir: str = "results/drift_suggestions",
) -> Dict[str, Any]:
    """Build a minimal rd_loop yaml fragment for human review after drift ALERT."""
    scans: List[Dict[str, Any]] = []
    for item in drift_items:
        if item.get("status") not in ("DRIFT", "MISSING_FEATURE"):
            continue
        feature = item.get("feature")
        if not feature:
            continue
        grid = _probe_grid_from_item(item)
        p50 = item.get("window_p50")
        scans.append(
            {
                "mode": "feature-plateau",
                "strategy": strategy,
                "layer": "regime",
                "features_parquet": features_parquet,
                "feature": feature,
                "operator": "<=",
                "grid": grid,
                "out": f"quick_scan/drift_{feature}.md",
                "comment": "drift follow-up plateau scan",
            }
        )
        if isinstance(p50, (int, float)):
            scans.append(
                {
                    "mode": "condition-set",
                    "strategy": strategy,
                    "layer": "regime",
                    "features_parquet": features_parquet,
                    "condition": [
                        f"probe_below_p50: {feature}<={p50:.6g}",
                        f"probe_above_p50: {feature}>={p50:.6g}",
                    ],
                    "out": f"quick_scan/drift_{feature}_conditions.md",
                    "comment": "quantile probes only — production τ from plateau/lift",
                }
            )

    snippet = {
        "topic": f"{strategy}_drift_followup",
        "strategy": strategy,
        "output_dir": f"{output_dir}/{strategy}",
        "research_scans": scans,
        "variant_grid": None,
        "decision_doc": None,
    }
    return snippet


def write_drift_suggestions(
    report: List[Dict[str, Any]],
    *,
    features_parquet: str,
    out_dir: Path,
) -> List[Path]:
    """Write one yaml+json suggestion file per strategy with ALERT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for entry in report:
        if not entry.get("any_alert"):
            continue
        strategy = str(entry.get("strategy", "unknown"))
        snippet = build_rd_loop_snippet(
            strategy=strategy,
            drift_items=entry.get("items") or [],
            features_parquet=features_parquet,
        )
        if not snippet.get("research_scans"):
            continue
        json_path = out_dir / f"rd_loop_{strategy}_drift_snippet.json"
        json_path.write_text(
            json.dumps(snippet, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        yaml_path = out_dir / f"rd_loop_{strategy}_drift_snippet.yaml"
        try:
            import yaml

            yaml_path.write_text(
                yaml.safe_dump(snippet, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception:
            yaml_path = json_path
        cmd_path = out_dir / f"run_{strategy}_drift_rd_loop.sh"
        cmd_path.write_text(
            f"#!/usr/bin/env bash\n"
            f"set -euo pipefail\n"
            f"python scripts/rd_loop.py --hypothesis-yaml {yaml_path}\n",
            encoding="utf-8",
        )
        written.extend([json_path, yaml_path, cmd_path])
    return written
