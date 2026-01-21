#!/usr/bin/env python3
"""
Diagnose Gate Filtering

This script analyzes how Gate filtering affects trades, showing:
1. Which trades are filtered by regime
2. Which trades are filtered by archetype availability
3. Which trades are filtered by Gate rules
4. Final archetype distribution after Gate filtering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.apply_tree_gate_3action import (
    _enabled_archetypes,
    load_execution_archetypes_registry,
)
from src.time_series_model.live.meta_router_config import load_meta_router_live_config


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose Gate filtering effects.")
    p.add_argument("--logs", required=True, help="logs_3action.parquet")
    p.add_argument("--regime", required=True, help="physics_regime parquet")
    p.add_argument("--live-config", required=True, help="meta_router_live_config.yaml")
    p.add_argument("--output-md", required=True, help="Output Markdown report")
    args = p.parse_args()

    # Load data
    logs_df = pd.read_parquet(args.logs)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

    regime_df = pd.read_parquet(args.regime)
    regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])

    # Merge
    merged = logs_df.merge(
        regime_df[["symbol", "timestamp", "regime"]],
        on=["symbol", "timestamp"],
        how="inner",
    )

    # Load configs
    live_cfg = load_meta_router_live_config(args.live_config)
    arches = load_execution_archetypes_registry()

    # Statistics
    stats = {
        "total_trades": 0,
        "filtered_by_no_trade_regime": 0,
        "filtered_by_disabled_archetype": 0,
        "passed_gate": 0,
        "by_archetype": {},
        "by_regime": {},
    }

    # Process each trade
    for _, row in merged.iterrows():
        mode = str(row.get("mode", "NO_TRADE")).upper()
        if mode == "NO_TRADE":
            continue

        stats["total_trades"] += 1
        regime = str(row.get("regime", "")).upper()

        # Track by regime
        if regime not in stats["by_regime"]:
            stats["by_regime"][regime] = 0
        stats["by_regime"][regime] += 1

        # Filter 1: NO_TRADE regime
        if regime == "NO_TRADE":
            stats["filtered_by_no_trade_regime"] += 1
            continue

        # Filter 2: Check enabled archetypes
        regime_for_lookup = (
            "TREND"
            if regime in ("TC_REGIME", "TE_REGIME")
            else regime.replace("_REGIME", "")
        )
        enabled = _enabled_archetypes(
            live_cfg_path=args.live_config,
            regime=regime_for_lookup,
            archetypes=arches,
        )

        if not enabled:
            stats["filtered_by_disabled_archetype"] += 1
            continue

        # Infer archetype
        if regime == "TE_REGIME" and mode == "TREND":
            arch = "TE"
        elif regime == "TC_REGIME" and mode == "TREND":
            arch = "TC"
        elif regime == "MEAN_REGIME" and mode == "MEAN":
            arch = "FR"
        else:
            arch = "TC"  # Fallback

        if arch not in stats["by_archetype"]:
            stats["by_archetype"][arch] = 0
        stats["by_archetype"][arch] += 1
        stats["passed_gate"] += 1

    # Generate report
    lines = []
    lines.append("# Gate Filtering Diagnosis Report\n")
    lines.append(f"- logs: `{args.logs}`\n")
    lines.append(f"- regime: `{args.regime}`\n")
    lines.append(f"- live-config: `{args.live_config}`\n")

    lines.append("\n## Summary\n")
    lines.append("| metric | count | percentage |\n|---|---|---|\n")
    lines.append(f"| Total trades | {stats['total_trades']} | 100.0% |\n")
    lines.append(
        f"| Filtered by NO_TRADE regime | {stats['filtered_by_no_trade_regime']} | "
        f"{stats['filtered_by_no_trade_regime'] / max(stats['total_trades'], 1) * 100:.1f}% |\n"
    )
    lines.append(
        f"| Filtered by disabled archetype | {stats['filtered_by_disabled_archetype']} | "
        f"{stats['filtered_by_disabled_archetype'] / max(stats['total_trades'], 1) * 100:.1f}% |\n"
    )
    lines.append(
        f"| Passed Gate | {stats['passed_gate']} | "
        f"{stats['passed_gate'] / max(stats['total_trades'], 1) * 100:.1f}% |\n"
    )

    lines.append("\n## By Regime (Before Gate)\n")
    lines.append("| regime | trade_count |\n|---|---|\n")
    for regime, count in sorted(stats["by_regime"].items()):
        lines.append(f"| {regime} | {count} |\n")

    lines.append("\n## By Archetype (After Gate)\n")
    lines.append("| archetype | trade_count |\n|---|---|\n")
    for arch, count in sorted(stats["by_archetype"].items()):
        lines.append(f"| {arch} | {count} |\n")

    # Check why TE/ET are missing
    lines.append("\n## Why TE/ET Are Missing?\n")
    if "TE" not in stats["by_archetype"]:
        lines.append("⚠️ **TE archetype is missing**\n")
        lines.append("Possible reasons:\n")
        lines.append("1. TE_REGIME trades were filtered by Gate rules\n")
        lines.append("2. TE_REGIME trades don't have required evidence\n")
        lines.append("3. Gate rules for TrendExpansionTE are too strict\n")
    else:
        lines.append(f"✓ TE archetype has {stats['by_archetype']['TE']} trades\n")

    if "ET" not in stats["by_archetype"]:
        lines.append("\n⚠️ **ET archetype is missing**\n")
        lines.append("Possible reasons:\n")
        lines.append("1. ET is only enabled in MEAN regime (currently disabled)\n")
        lines.append("2. No trades match ET conditions\n")
    else:
        lines.append(f"\n✓ ET archetype has {stats['by_archetype']['ET']} trades\n")

    # Enabled archetypes info
    lines.append("\n## Enabled Archetypes (from live config)\n")
    for regime_name in ["TREND", "MEAN", "NO_TRADE"]:
        enabled = _enabled_archetypes(
            live_cfg_path=args.live_config,
            regime=regime_name,
            archetypes=arches,
        )
        lines.append(
            f"- **{regime_name}**: {enabled if enabled else '[] (disabled)'}\n"
        )

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
