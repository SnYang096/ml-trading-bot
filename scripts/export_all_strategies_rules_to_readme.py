#!/usr/bin/env python3
"""
Batch export rules to all four main tree strategies' README.md files.

This script will:
1. Generate rules for each strategy using export_tree_rules_imodels.py
2. Parse the generated rules.md files
3. Append rules section to each strategy's README.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "sr_breakout",
    "compression_breakout",
    "trend_following",
]


def main() -> None:
    for strategy in STRATEGIES:
        print(f"\n{'='*80}")
        print(f"Processing strategy: {strategy}")
        print(f"{'='*80}\n")

        strategy_config = ROOT / "config" / "strategies" / strategy
        features_yaml = strategy_config / "features_suggested_20260128.yaml"

        if not strategy_config.exists():
            print(f"⚠️  Strategy config not found: {strategy_config}")
            continue

        if not features_yaml.exists():
            print(f"⚠️  Features YAML not found: {features_yaml}")
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "export_strategy_rules_to_readme.py"),
            "--strategy-config",
            str(strategy_config),
            "--features-yaml",
            str(features_yaml),
            "--generate-rules",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "240T",
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2025-12-31",
            "--max-rules",
            "20",
            "--min-support",
            "0.01",
            "--max-conditions",
            "3",
            "--max-rule-len",
            "120",
            "--random-state",
            "42",
        ]

        try:
            result = subprocess.run(cmd, cwd=str(ROOT), check=True)
            print(f"✅ Successfully exported rules for {strategy}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to export rules for {strategy}: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted. Stopping batch export.")
            sys.exit(1)

    print(f"\n{'='*80}")
    print("✅ Batch export completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
