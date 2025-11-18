#!/usr/bin/env python3
"""
Validate SHAP signs against expected economic logic.

Example:
    python scripts/cross_sectional/run_factor_logic_check.py \
        --shap-manifest results/cross_sectional/shap_reports/manifest.json \
        --expectations configs/factor_expectations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check SHAP average/sign against expected economic logic."
    )
    parser.add_argument(
        "--shap-manifest",
        required=True,
        help="Path to SHAP manifest.json produced by run_shap_analysis.py.",
    )
    parser.add_argument(
        "--expectations",
        required=True,
        help=(
            "JSON file mapping factor name -> expected logic "
            "(e.g. {'factor': {'expected_sign': '+', 'description': 'Higher value should boost return'}})."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Allowed deviation from expected sign (default: 0 means exact match).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_sectional/shap_logic_report.md",
        help="Output markdown report file.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(file)
    return json.loads(file.read_text(encoding="utf-8"))


def check_logic(
    shap_summary: Dict[str, Dict[str, float]],
    expectations: Dict[str, Dict[str, str]],
    tolerance: float,
) -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    for factor, expectation in expectations.items():
        expected_sign = expectation.get("expected_sign")
        desc = expectation.get("description", "")
        shap_entry = shap_summary.get(factor)
        if not shap_entry:
            report[factor] = {
                "status": "missing",
                "message": "Factor not present among SHAP top features.",
                "expected_sign": expected_sign,
                "description": desc,
            }
            continue

        shap_mean = shap_entry.get("mean_shap", 0.0)
        shap_sign = np.sign(shap_mean)
        expected = None
        if expected_sign == "+":
            expected = 1.0
        elif expected_sign == "-":
            expected = -1.0
        elif expected_sign == "0":
            expected = 0.0

        status = "unknown"
        message = ""
        if expected is None:
            status = "unknown"
            message = "Expected sign not specified (+/-/0)."
        else:
            diff = abs(shap_sign - expected)
            if diff <= tolerance:
                status = "ok"
                message = f"SHAP sign {shap_sign} matches expected {expected} within tolerance."
            else:
                status = "alert"
                message = (
                    f"SHAP sign {shap_sign:.1f} deviates from expected {expected:.1f}."
                )
        report[factor] = {
            "status": status,
            "message": message,
            "expected_sign": expected_sign,
            "mean_shap": shap_mean,
            "description": desc,
        }
    return report


def write_report(report: Dict[str, Dict[str, object]], path: str) -> None:
    output = ["# SHAP Economic Logic Validation\n"]
    for factor, details in report.items():
        output.append(f"## {factor}")
        output.append(f"- Status: {details.get('status', 'unknown')}")
        output.append(f"- Expected sign: {details.get('expected_sign', 'N/A')}")
        output.append(f"- Observed mean SHAP: {details.get('mean_shap', 0.0):.4f}")
        output.append(f"- Message: {details.get('message', '')}")
        desc = details.get("description")
        if desc:
            output.append(f"- Description: {desc}")
        output.append("")
    Path(path).write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.shap_manifest)
    shap_metrics = manifest.get("final_selection") or {}
    shap_summary = shap_metrics.get("metrics", {})
    expectations = load_json(args.expectations)

    report = check_logic(shap_summary, expectations, args.tolerance)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, str(output_path))
    print(f"✅ SHAP logic report written to {output_path}")


if __name__ == "__main__":
    main()
