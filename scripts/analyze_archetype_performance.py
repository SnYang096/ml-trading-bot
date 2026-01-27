#!/usr/bin/env python3
"""
分析各archetype的性能指标

生成详细的archetype性能报告，包括：
- 每个archetype的Sharpe、交易数、胜率、平均收益
- 多archetype同时触发的统计
- CVD判断的效果统计
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sharpe(returns: pd.Series) -> float:
    """Calculate Sharpe ratio (annualized, using Daily factor sqrt(252))."""
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    # Use Daily annualization factor (sqrt(252)) instead of 4H factor (sqrt(6*365))
    # This is more conservative and standard for financial metrics
    return float(mean / std * np.sqrt(252)) if std > 1e-12 else 0.0


def _archetype_return(row: pd.Series, ret_mean_col: str, ret_trend_col: str) -> float:
    """Select ret_mean or ret_trend based on archetype."""
    archetype = str(row.get("gate_archetype") or row.get("archetype") or "").upper()
    if not archetype:
        return 0.0

    # TC/TE → ret_trend
    if "TC" in archetype or "TE" in archetype:
        return float(row.get(ret_trend_col, 0.0) or 0.0)

    # FR/ET → ret_mean
    if "FR" in archetype or "ET" in archetype:
        return float(row.get(ret_mean_col, 0.0) or 0.0)

    return 0.0


def analyze_archetype_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance by archetype."""
    df = df.copy()

    # Add archetype return column
    ret_mean_col = "ret_mean"
    ret_trend_col = "ret_trend"
    df["archetype_ret"] = df.apply(
        lambda r: _archetype_return(r, ret_mean_col, ret_trend_col), axis=1
    )

    # Filter to only rows with gate_ok=True
    gated_df = df[df.get("gate_ok", pd.Series([False] * len(df))) == True].copy()

    results: Dict[str, Any] = {
        "overall": {},
        "by_archetype": {},
        "multi_archetype_stats": {},
        "cvd_effect": {},
    }

    # Overall stats
    if len(gated_df) > 0:
        overall_returns = gated_df["archetype_ret"]
        results["overall"] = {
            "total_trades": len(gated_df),
            "sharpe": _sharpe(overall_returns),
            "win_rate": (
                float((overall_returns > 0).sum() / len(overall_returns))
                if len(overall_returns) > 0
                else 0.0
            ),
            "avg_return": (
                float(overall_returns.mean()) if len(overall_returns) > 0 else 0.0
            ),
            "total_return": (
                float(overall_returns.sum()) if len(overall_returns) > 0 else 0.0
            ),
        }

    # By archetype
    archetype_col = "gate_archetype"
    if archetype_col in gated_df.columns:
        for arch_name in gated_df[archetype_col].dropna().unique():
            arch_df = gated_df[gated_df[archetype_col] == arch_name]
            arch_returns = arch_df["archetype_ret"]
            if len(arch_returns) > 0:
                results["by_archetype"][arch_name] = {
                    "trades": len(arch_df),
                    "sharpe": _sharpe(arch_returns),
                    "win_rate": float((arch_returns > 0).sum() / len(arch_returns)),
                    "avg_return": float(arch_returns.mean()),
                    "total_return": float(arch_returns.sum()),
                }

    # Multi-archetype statistics (from gate_reasons)
    if "gate_reasons" in df.columns:
        multi_arch_stats: Dict[str, int] = defaultdict(int)
        for reasons in df["gate_reasons"].dropna():
            if isinstance(reasons, str):
                # Parse reasons string (e.g., "multiple_archetypes_no_trade:TC,TE")
                if "multiple_archetypes" in reasons.lower():
                    parts = reasons.split(":")
                    if len(parts) > 1:
                        archs = parts[1].strip()
                        multi_arch_stats[archs] += 1
            elif isinstance(reasons, list):
                # Check if it's a multi-archetype case
                arch_names = [
                    r
                    for r in reasons
                    if isinstance(r, str)
                    and any(a in r.upper() for a in ["TC", "TE", "FR", "ET"])
                ]
                if len(arch_names) > 1:
                    key = "+".join(sorted(set(arch_names)))
                    multi_arch_stats[key] += 1

        results["multi_archetype_stats"] = dict(multi_arch_stats)

    # CVD effect (if cvd_change_5 is available)
    if "cvd_change_5" in gated_df.columns:
        cvd_col = "cvd_change_5"
        cvd_median = gated_df[cvd_col].median()
        positive_cvd = gated_df[gated_df[cvd_col] > cvd_median]
        negative_cvd = gated_df[gated_df[cvd_col] <= cvd_median]

        results["cvd_effect"] = {
            "positive_cvd": (
                {
                    "trades": len(positive_cvd),
                    "sharpe": (
                        _sharpe(positive_cvd["archetype_ret"])
                        if len(positive_cvd) > 0
                        else 0.0
                    ),
                }
                if len(positive_cvd) > 0
                else {}
            ),
            "negative_cvd": (
                {
                    "trades": len(negative_cvd),
                    "sharpe": (
                        _sharpe(negative_cvd["archetype_ret"])
                        if len(negative_cvd) > 0
                        else 0.0
                    ),
                }
                if len(negative_cvd) > 0
                else {}
            ),
        }

    return results


def format_markdown_report(results: Dict[str, Any]) -> str:
    """Format results as Markdown report."""
    lines = []
    lines.append("# Archetype Performance Analysis")
    lines.append("")

    # Overall stats
    lines.append("## Overall Performance")
    lines.append("")
    if results["overall"]:
        overall = results["overall"]
        lines.append(f"- **Total Trades**: {overall.get('total_trades', 0)}")
        lines.append(f"- **Sharpe Ratio**: {overall.get('sharpe', 0.0):.4f}")
        lines.append(f"- **Win Rate**: {overall.get('win_rate', 0.0):.2%}")
        lines.append(f"- **Average Return**: {overall.get('avg_return', 0.0):.6f}")
        lines.append(f"- **Total Return**: {overall.get('total_return', 0.0):.6f}")
    lines.append("")

    # By archetype
    lines.append("## Performance by Archetype")
    lines.append("")
    if results["by_archetype"]:
        lines.append(
            "| Archetype | Trades | Sharpe | Win Rate | Avg Return | Total Return |"
        )
        lines.append(
            "|-----------|--------|--------|----------|------------|--------------|"
        )
        for arch_name, arch_stats in sorted(results["by_archetype"].items()):
            lines.append(
                f"| {arch_name} | {arch_stats.get('trades', 0)} | "
                f"{arch_stats.get('sharpe', 0.0):.4f} | "
                f"{arch_stats.get('win_rate', 0.0):.2%} | "
                f"{arch_stats.get('avg_return', 0.0):.6f} | "
                f"{arch_stats.get('total_return', 0.0):.6f} |"
            )
    else:
        lines.append("No archetype data available.")
    lines.append("")

    # Multi-archetype stats
    lines.append("## Multi-Archetype Statistics")
    lines.append("")
    if results["multi_archetype_stats"]:
        lines.append("| Combination | Count |")
        lines.append("|-------------|-------|")
        for combo, count in sorted(
            results["multi_archetype_stats"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"| {combo} | {count} |")
    else:
        lines.append("No multi-archetype combinations found.")
    lines.append("")

    # CVD effect
    lines.append("## CVD Effect Analysis")
    lines.append("")
    if results["cvd_effect"]:
        if (
            "positive_cvd" in results["cvd_effect"]
            and results["cvd_effect"]["positive_cvd"]
        ):
            pos = results["cvd_effect"]["positive_cvd"]
            lines.append(
                f"**Positive CVD**: {pos.get('trades', 0)} trades, Sharpe: {pos.get('sharpe', 0.0):.4f}"
            )
        if (
            "negative_cvd" in results["cvd_effect"]
            and results["cvd_effect"]["negative_cvd"]
        ):
            neg = results["cvd_effect"]["negative_cvd"]
            lines.append(
                f"**Negative CVD**: {neg.get('trades', 0)} trades, Sharpe: {neg.get('sharpe', 0.0):.4f}"
            )
    else:
        lines.append("CVD data not available.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze archetype performance")
    p.add_argument("--logs", required=True, help="Input logs file (parquet)")
    p.add_argument("--output", required=True, help="Output markdown report path")
    args = p.parse_args()

    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"Error: Logs file not found: {logs_path}", file=sys.stderr)
        return 1

    # Read logs
    df = pd.read_parquet(logs_path)

    # Analyze
    results = analyze_archetype_performance(df)

    # Format and write report
    report = format_markdown_report(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Archetype performance report written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
