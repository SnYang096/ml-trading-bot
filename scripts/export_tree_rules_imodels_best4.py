#!/usr/bin/env python3
"""
Batch export: export rules (md/json/py if-else) for the 4 tree strategies using their best-known configs.

This script:
- Picks the best run per strategy (prefer C over B over A) using the same logic as tree_model_finalize.py
- Exports rules for BOTH:
  - features_full.yaml (exact best config)
  - features_lite.yaml (trimmed, faster to iterate)

Outputs under: results/rules_export/tree_best4/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tree_model_finalize import TREE_STRATEGIES_DEFAULT, _pick_best_run


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategies",
        default=",".join(TREE_STRATEGIES_DEFAULT),
        help="Comma-separated tree strategies (config/strategies/<name>)",
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--max-rules", type=int, default=50)
    ap.add_argument("--max-conditions", type=int, default=3)
    ap.add_argument("--min-support", type=float, default=0.01)
    ap.add_argument("--max-rule-len", type=int, default=160)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--variants",
        default="lite,full",
        help="Comma-separated variants to export: lite,full",
    )
    args = ap.parse_args()

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    out_root = ROOT / "results" / "rules_export" / "tree_best4"
    out_root.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    variants = [v for v in variants if v in {"lite", "full"}]
    if not variants:
        variants = ["lite"]

    for s in strategies:
        pick = _pick_best_run(s)
        if pick is None or pick.suggested_yaml is None:
            print(f"⚠️  skip {s}: no best run / suggested_yaml found")
            continue

        export_dir = (
            ROOT
            / "config"
            / "strategies_exported"
            / "tree_best"
            / f"{s}__{pick.tag}__{pick.stage}"
        )
        full_yaml = export_dir / "features_full.yaml"
        lite_yaml = export_dir / "features_lite.yaml"
        # Fallback to suggested_yaml if exported ones don't exist
        if not full_yaml.exists():
            full_yaml = pick.suggested_yaml
        if not lite_yaml.exists():
            lite_yaml = pick.suggested_yaml

        mapping = {"full": full_yaml, "lite": lite_yaml}
        for variant in variants:
            feat_yaml = mapping[variant]
            out_dir = out_root / f"{s}__{pick.tag}__{pick.stage}__{variant}"
            cmd = [
                sys.executable,
                "scripts/export_tree_rules_imodels.py",
                "--strategy-config",
                str(ROOT / "config" / "strategies" / s),
                "--features-yaml",
                str(feat_yaml),
                "--symbol",
                str(args.symbol),
                "--timeframe",
                str(args.timeframe),
                "--start-date",
                str(args.start_date),
                "--end-date",
                str(args.end_date),
                "--test-size",
                str(args.test_size),
                "--output-dir",
                str(out_dir),
                "--max-rules",
                str(args.max_rules),
                "--max-conditions",
                str(args.max_conditions),
                "--min-support",
                str(args.min_support),
                "--max-rule-len",
                str(args.max_rule_len),
                "--random-state",
                str(args.random_state),
            ]
            print("▶️", " ".join(cmd))
            _run(cmd)

    print(f"✅ Done. Rules exported under: {out_root}")


if __name__ == "__main__":
    main()
