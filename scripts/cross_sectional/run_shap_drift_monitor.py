#!/usr/bin/env python3
"""
Monitor SHAP drift compared with a historical baseline.

Example:
    python scripts/cross_sectional/run_shap_drift_monitor.py \
        --current results/cross_sectional/shap_reports/manifest.json \
        --baseline results/cross_sectional/shap_baseline.json \
        --threshold 0.5 \
        --output results/cross_sectional/shap_drift_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current SHAP metrics to baseline and flag drift."
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current SHAP manifest.json (from run_shap_analysis.py).",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help=(
            "Baseline JSON file storing previous SHAP metrics "
            "(expected format similar to manifest['final_selection']['metrics'])."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Relative change threshold to flag drift (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_sectional/shap_drift_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--update-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite baseline with current metrics when drift is acceptable.",
    )
    return parser.parse_args()


def load_metrics(path: str) -> Dict[str, Dict[str, float]]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(file)
    data = json.loads(file.read_text(encoding="utf-8"))
    return data


def extract_metrics(manifest: Dict) -> Dict[str, Dict[str, float]]:
    final_selection = manifest.get("final_selection") or {}
    metrics = final_selection.get("metrics", {})
    return metrics


def compute_drift(
    current: Dict[str, Dict[str, float]],
    baseline: Dict[str, Dict[str, float]],
    threshold: float,
) -> Tuple[List[Dict[str, object]], float]:
    alerts: List[Dict[str, object]] = []
    overall_change = 0.0
    # for factors present in either baseline or current
    factors = set(baseline.keys()).union(current.keys())
    for factor in factors:
        base = baseline.get(factor)
        curr = current.get(factor)
        if not base or not curr:
            alerts.append(
                {
                    "factor": factor,
                    "status": "missing",
                    "message": "Factor missing in baseline or current.",
                }
            )
            continue

        base_mean = base.get("mean_shap", 0.0)
        curr_mean = curr.get("mean_shap", 0.0)
        base_abs = base.get("abs_shap", abs(base_mean))
        curr_abs = curr.get("abs_shap", abs(curr_mean))

        change = 0.0
        if base_abs > 0:
            change = abs(curr_abs - base_abs) / base_abs
        overall_change += change
        if change > threshold:
            alerts.append(
                {
                    "factor": factor,
                    "status": "drift",
                    "message": f"Absolute SHAP changed by {change:.2f} (> {threshold}).",
                    "baseline_abs": base_abs,
                    "current_abs": curr_abs,
                }
            )
    average_change = overall_change / max(len(factors), 1)
    return alerts, average_change


def write_report(
    alerts: List[Dict[str, object]],
    average_change: float,
    threshold: float,
    path: str,
) -> None:
    lines = [
        "# SHAP Drift Report",
        f"- Average absolute change: {average_change:.4f}",
        f"- Threshold: {threshold}",
        "",
    ]
    if not alerts:
        lines.append("✅ No significant drift detected.")
    else:
        lines.append("⚠️ Drift Alerts:")
        for alert in alerts:
            lines.append(f"## {alert['factor']}")
            lines.append(f"- Status: {alert['status']}")
            lines.append(f"- Message: {alert['message']}")
            if "baseline_abs" in alert:
                lines.append(f"- Baseline |abs SHAP|: {alert['baseline_abs']:.4f}")
            if "current_abs" in alert:
                lines.append(f"- Current |abs SHAP|: {alert['current_abs']:.4f}")
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_metrics(args.current)
    current_metrics = extract_metrics(manifest)
    baseline_data = load_metrics(args.baseline)

    if "metrics" in baseline_data and "factors" in baseline_data:
        baseline_metrics = baseline_data["metrics"]
    else:
        baseline_metrics = baseline_data

    alerts, avg_change = compute_drift(
        current_metrics, baseline_metrics, args.threshold
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(alerts, avg_change, args.threshold, str(output_path))
    print(f"✅ SHAP drift report written to {output_path}")

    if args.update_baseline and not alerts:
        baseline_path = Path(args.baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(
                {
                    "metrics": current_metrics,
                    "factors": list(current_metrics.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"🔄 Baseline updated at {baseline_path}")


if __name__ == "__main__":
    main()
