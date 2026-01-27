#!/usr/bin/env python3
"""
Apply FBF optimized thresholds to execution_archetypes.yaml (when-then schema).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import yaml


def _extract_best_candidates(opt: Dict[str, Any]) -> Dict[str, float]:
    best = {}
    rules = opt.get("rules", {})
    for rule_id, payload in rules.items():
        best_candidate = payload.get("best", {}).get("candidate")
        if best_candidate is not None:
            best[rule_id] = float(best_candidate)
    return best


def _apply_thresholds(arch_cfg: Dict[str, Any], thresholds: Dict[str, float]) -> int:
    updated = 0
    for rule in arch_cfg.get("when_then_rules", []):
        rule_id = rule.get("id")
        if rule_id not in thresholds:
            continue
        when = rule.get("when", {})
        for k, v in when.items():
            if isinstance(v, dict) and "quantile_gt" in v:
                v["quantile_gt"] = thresholds[rule_id]
                updated += 1
            elif isinstance(v, dict) and "quantile_lt" in v:
                v["quantile_lt"] = thresholds[rule_id]
                updated += 1
        rule["when"] = when
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply FBF optimized params to config")
    parser.add_argument(
        "--optimization",
        required=True,
        help="optimization json from optimize_fbf_label_based.py",
    )
    parser.add_argument(
        "--config", required=True, help="execution_archetypes.yaml path"
    )
    parser.add_argument("--out", required=True, help="output config path")
    parser.add_argument(
        "--archetype", default="FailedBreakoutFade", help="archetype name"
    )
    args = parser.parse_args()

    with open(args.optimization, "r") as f:
        opt = json.load(f)
    thresholds = _extract_best_candidates(opt)
    if not thresholds:
        raise ValueError("no optimized thresholds found in optimization json")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    arches = config.get("archetypes") if isinstance(config, dict) else None
    if arches is None:
        arches = config
    arch_cfg = arches.get(args.archetype) if isinstance(arches, dict) else None
    if not arch_cfg:
        raise KeyError(f"archetype not found: {args.archetype}")

    updated = _apply_thresholds(arch_cfg, thresholds)
    if updated == 0:
        raise RuntimeError(
            "no thresholds updated; check rule ids in optimization output"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
