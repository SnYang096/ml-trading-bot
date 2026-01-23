#!/usr/bin/env python3
"""
批量优化所有archetype的gate规则阈值

使用平坦高原方法，逐个优化每个archetype的关键gate规则。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from scripts.optimize_gate_plateau import (
    _scan_threshold,
    _find_plateau,
    BucketConfig,
    OptimizationConfig,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Optimize all archetype gate rules using plateau method"
    )
    p.add_argument("--gated-logs", required=True, help="Gated logs file (parquet)")
    p.add_argument("--raw-logs", default=None, help="Raw logs file (parquet, optional)")
    p.add_argument(
        "--output-dir", required=True, help="Output directory for optimization results"
    )
    p.add_argument(
        "--min-trade-rate",
        type=float,
        default=0.005,
        help="Minimum trade rate threshold",
    )
    p.add_argument(
        "--min-trades-per-bucket",
        type=int,
        default=10,
        help="Minimum trades per bucket",
    )
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Path to execution_archetypes.yaml",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gated_df = pd.read_parquet(args.gated_logs)
    raw_df = pd.read_parquet(args.raw_logs) if args.raw_logs else gated_df

    # Load archetypes
    arches = load_execution_archetypes_registry(args.execution_archetypes)

    # Filter out VolMeanCompressionExpansionReversion
    arches = {
        k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"
    }

    # Configuration
    bucket_config = BucketConfig()
    opt_config = OptimizationConfig(
        min_trade_rate=args.min_trade_rate,
        min_trades_per_bucket=args.min_trades_per_bucket,
    )

    all_results: Dict[str, Any] = {}

    # Optimize each archetype
    for arch_name, arch_config in arches.items():
        print(f"\n{'=' * 80}")
        print(f"Optimizing archetype: {arch_name}")
        print(f"{'=' * 80}\n")

        if not arch_config.gate_rules:
            print(f"  No gate rules found for {arch_name}, skipping...")
            continue

        arch_results: List[Dict[str, Any]] = []

        # Get rules list and deny_if list
        all_rules = arch_config.gate_rules.get("rules", [])
        deny_if_names = arch_config.gate_rules.get("deny_if", [])

        if not all_rules or not deny_if_names:
            print(f"  No gate rules or deny_if list found for {arch_name}, skipping...")
            continue

        # Filter to quantile-based rules that are in deny_if list
        quantile_rules = [
            r
            for r in all_rules
            if r.get("name") in deny_if_names
            and r.get("kind")
            in ("quantile_lt", "quantile_gt", "quantile_lte", "quantile_gte")
        ]

        if not quantile_rules:
            print(f"  No quantile-based rules found for {arch_name}, skipping...")
            continue

        for rule in quantile_rules:
            rule_name = rule.get("name", "unknown")
            feature_key = rule.get("key")
            rule_kind = rule.get("kind")
            current_threshold = rule.get("quantile") or rule.get("value", 0.5)

            if not feature_key or not rule_kind:
                continue

            print(f"  Optimizing rule: {rule_name} ({feature_key}, {rule_kind})")

            # Scan thresholds
            scan_results = _scan_threshold(
                raw_df,
                rule_name,
                feature_key,
                rule_kind,
                opt_config.threshold_range,
                opt_config.threshold_step,
                bucket_config,
                opt_config,
                args.execution_archetypes,
                base_gated_df=gated_df if args.raw_logs else None,
            )

            if len(scan_results) == 0:
                print(f"    No valid thresholds found")
                continue

            # Find plateau
            plateau = _find_plateau(scan_results, opt_config.min_sharpe_threshold)

            if plateau:
                plateau_start, plateau_end, plateau_median = plateau
                print(
                    f"    Plateau found: [{plateau_start:.4f}, {plateau_end:.4f}], recommended: {plateau_median:.4f}"
                )
            else:
                # Use best threshold
                best_idx = scan_results["robustness_score"].idxmax()
                plateau_median = scan_results.loc[best_idx, "threshold"]
                plateau_start = plateau_median
                plateau_end = plateau_median
                print(
                    f"    No plateau found, using best threshold: {plateau_median:.4f}"
                )

            # Get best result
            best_idx = scan_results["robustness_score"].idxmax()
            best_result = scan_results.loc[best_idx].to_dict()

            arch_results.append(
                {
                    "rule_name": rule_name,
                    "feature_key": feature_key,
                    "rule_kind": rule_kind,
                    "current_threshold": current_threshold,
                    "plateau_start": float(plateau_start),
                    "plateau_end": float(plateau_end),
                    "recommended_threshold": float(plateau_median),
                    "robustness_score": float(best_result["robustness_score"]),
                    "trade_rate": float(best_result["trade_rate"]),
                    "min_coverage": int(best_result["min_coverage"]),
                    "worst_bucket": best_result.get("worst_bucket", "unknown"),
                }
            )

        all_results[arch_name] = arch_results

        # Save per-archetype results
        arch_output = output_dir / f"{arch_name}_optimization.json"
        with open(arch_output, "w") as f:
            json.dump(arch_results, f, indent=2)
        print(f"  Saved results to: {arch_output}")

    # Save summary
    summary_output = output_dir / "all_archetypes_optimization_summary.json"
    with open(summary_output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
