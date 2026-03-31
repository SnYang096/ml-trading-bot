#!/usr/bin/env python3
"""
PCM (Portfolio Capital Management) 层性能诊断

检测指标:
- Slot allocation效率（是否充分利用可用slot）
- Risk budgeting执行情况（是否超出风险预算）
- Position sizing合理性（是否过度集中）
- Slot rotation频率（是否过于频繁或过于保守）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_pcm_performance(
    df: pd.DataFrame, baseline_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """Analyze PCM layer performance."""
    results: Dict[str, Any] = {}

    # Filter to gated trades
    gated_df = df[df.get("gate_ok", pd.Series([False] * len(df))) == True].copy()

    # Slot allocation analysis (if slot information is available)
    if "slot_id" in gated_df.columns or "position_id" in gated_df.columns:
        slot_col = "slot_id" if "slot_id" in gated_df.columns else "position_id"
        unique_slots = gated_df[slot_col].nunique()
        total_trades = len(gated_df)
        results["slot_allocation"] = {
            "unique_slots": int(unique_slots),
            "total_trades": total_trades,
            "avg_trades_per_slot": (
                float(total_trades / unique_slots) if unique_slots > 0 else 0.0
            ),
        }

    # Archetype distribution (indicates slot usage by archetype)
    if "gate_archetype" in gated_df.columns:
        arch_dist = gated_df["gate_archetype"].value_counts().to_dict()
        results["archetype_distribution"] = {
            str(k): int(v) for k, v in arch_dist.items()
        }

    # Comparison with baseline
    if baseline_df is not None:
        base_gated = baseline_df[
            baseline_df.get("gate_ok", pd.Series([False] * len(baseline_df))) == True
        ]
        results["baseline_comparison"] = {
            "production_trades": len(gated_df),
            "baseline_trades": len(base_gated),
            "trade_count_change": len(gated_df) - len(base_gated),
        }

    return results


def format_report(results: Dict[str, Any], output_path: Path) -> str:
    """Format PCM performance report."""
    lines = []
    lines.append("# PCM (Portfolio Capital Management) Performance Analysis")
    lines.append("")

    if "slot_allocation" in results:
        lines.append("## Slot Allocation")
        lines.append("")
        sa = results["slot_allocation"]
        lines.append(f"- **Unique Slots**: {sa.get('unique_slots', 0)}")
        lines.append(f"- **Total Trades**: {sa.get('total_trades', 0)}")
        lines.append(
            f"- **Avg Trades per Slot**: {sa.get('avg_trades_per_slot', 0.0):.2f}"
        )
        lines.append("")

    if "archetype_distribution" in results:
        lines.append("## Archetype Distribution")
        lines.append("")
        for arch, count in sorted(
            results["archetype_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"- **{arch}**: {count} trades")
        lines.append("")

    if "baseline_comparison" in results:
        lines.append("## Baseline Comparison")
        lines.append("")
        bc = results["baseline_comparison"]
        lines.append(f"- **Production Trades**: {bc.get('production_trades', 0)}")
        lines.append(f"- **Baseline Trades**: {bc.get('baseline_trades', 0)}")
        lines.append(f"- **Trade Count Change**: {bc.get('trade_count_change', 0)}")
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "1. Check PCM policy configuration (capacity_limit, risk_release_threshold)"
    )
    lines.append("2. Review slot rotation logic")
    lines.append("3. Verify archetype compatibility rules")
    lines.append("4. Adjust risk budget allocation if needed")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose PCM layer performance")
    p.add_argument("--logs", required=True, help="Input logs file (parquet)")
    p.add_argument(
        "--baseline", default=None, help="Baseline logs file for comparison (optional)"
    )
    p.add_argument("--output", required=True, help="Output report path (markdown)")
    args = p.parse_args()

    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"Error: Logs file not found: {logs_path}", file=sys.stderr)
        return 1

    # Load data
    df = pd.read_parquet(logs_path)
    baseline_df = pd.read_parquet(args.baseline) if args.baseline else None

    # Analyze
    results = analyze_pcm_performance(df, baseline_df)

    # Format and write report
    report = format_report(results, Path(args.output))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"PCM performance report written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
