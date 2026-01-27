#!/usr/bin/env python3
"""
Validate FBF optimization results vs baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_mean(vals):
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FBF optimization results")
    parser.add_argument(
        "--optimization", required=True, help="optimization results json"
    )
    parser.add_argument("--baseline", required=True, help="baseline json")
    parser.add_argument("--out", required=True, help="output json path")
    args = parser.parse_args()

    with open(args.optimization, "r") as f:
        opt = json.load(f)
    with open(args.baseline, "r") as f:
        baseline = json.load(f)

    summary = {"archetype": opt.get("archetype"), "rules": {}}
    for rule_id, payload in opt.get("rules", {}).items():
        best = payload.get("best", {})
        groups = best.get("groups", {})
        execution_ratio = _safe_mean(
            [g.get("execution_fbf_ratio", 0.0) for g in groups.values()]
        )
        semantic_ratio = _safe_mean(
            [g.get("semantic_fbf_ratio", 0.0) for g in groups.values()]
        )
        consistency = _safe_mean(
            [g.get("label_consistency_ratio", 0.0) for g in groups.values()]
        )
        precision = _safe_mean([g.get("gate_precision", 0.0) for g in groups.values()])
        recall = _safe_mean([g.get("gate_recall", 0.0) for g in groups.values()])
        noise = _safe_mean([g.get("noise_rate", 0.0) for g in groups.values()])

        summary["rules"][rule_id] = {
            "candidate": best.get("candidate"),
            "score": best.get("score"),
            "execution_fbf_ratio": execution_ratio,
            "semantic_fbf_ratio": semantic_ratio,
            "label_consistency_ratio": consistency,
            "gate_precision": precision,
            "gate_recall": recall,
            "noise_rate": noise,
            "baseline": baseline,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
