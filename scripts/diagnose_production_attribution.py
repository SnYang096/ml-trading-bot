#!/usr/bin/env python3
"""
实盘归因和调整流程

自动检测连续亏损和Sharpe下降，分层诊断，定位问题层，生成修复建议。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sharpe(returns: pd.Series) -> float:
    """Calculate Sharpe ratio (annualized)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    return float(mean / std * np.sqrt(252)) if std > 1e-12 else 0.0


def _archetype_return(row: pd.Series, ret_mean_col: str, ret_trend_col: str) -> float:
    """Select ret_mean or ret_trend based on archetype."""
    archetype = str(row.get("gate_archetype") or row.get("archetype") or "").upper()
    if not archetype:
        return 0.0
    if "TC" in archetype or "TE" in archetype:
        return float(row.get(ret_trend_col, 0.0) or 0.0)
    if "FR" in archetype or "ET" in archetype:
        return float(row.get(ret_mean_col, 0.0) or 0.0)
    return 0.0


def detect_degradation(
    production_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    alert_thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect performance degradation."""
    production_df = production_df.copy()
    baseline_df = baseline_df.copy()

    # Calculate returns
    production_df["archetype_ret"] = production_df.apply(
        lambda r: _archetype_return(r, "ret_mean", "ret_trend"), axis=1
    )
    baseline_df["archetype_ret"] = baseline_df.apply(
        lambda r: _archetype_return(r, "ret_mean", "ret_trend"), axis=1
    )

    # Filter to gated trades
    prod_gated = production_df[
        production_df.get("gate_ok", pd.Series([False] * len(production_df))) == True
    ]
    base_gated = baseline_df[
        baseline_df.get("gate_ok", pd.Series([False] * len(baseline_df))) == True
    ]

    prod_returns = prod_gated["archetype_ret"]
    base_returns = base_gated["archetype_ret"]

    # Calculate metrics
    prod_sharpe = _sharpe(prod_returns)
    base_sharpe = _sharpe(base_returns)
    sharpe_drop = prod_sharpe - base_sharpe

    prod_trades = len(prod_gated)
    base_trades = len(base_gated)
    trade_count_drop = (
        (prod_trades - base_trades) / base_trades if base_trades > 0 else 0.0
    )

    # Detect consecutive losses
    consecutive_losses = 0
    max_consecutive = 0
    for ret in prod_returns:
        if ret < 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0

    # Check alerts
    alerts = []
    if max_consecutive >= alert_thresholds.get("consecutive_losses", 5):
        alerts.append(
            f"Consecutive losses: {max_consecutive} >= {alert_thresholds.get('consecutive_losses', 5)}"
        )
    if sharpe_drop <= alert_thresholds.get("sharpe_drop", -0.5):
        alerts.append(
            f"Sharpe drop: {sharpe_drop:.4f} <= {alert_thresholds.get('sharpe_drop', -0.5)}"
        )
    if trade_count_drop <= alert_thresholds.get("trade_count_drop", -0.2):
        alerts.append(
            f"Trade count drop: {trade_count_drop:.2%} <= {alert_thresholds.get('trade_count_drop', -0.2):.2%}"
        )

    return {
        "production_sharpe": prod_sharpe,
        "baseline_sharpe": base_sharpe,
        "sharpe_drop": sharpe_drop,
        "production_trades": prod_trades,
        "baseline_trades": base_trades,
        "trade_count_drop": trade_count_drop,
        "max_consecutive_losses": max_consecutive,
        "alerts": alerts,
        "degradation_detected": len(alerts) > 0,
    }


def format_report(degradation: Dict[str, Any], output_dir: Path) -> str:
    """Format diagnostic report."""
    lines = []
    lines.append("# Production Attribution Analysis")
    lines.append("")

    lines.append("## Performance Degradation Detection")
    lines.append("")
    lines.append(f"- **Production Sharpe**: {degradation['production_sharpe']:.4f}")
    lines.append(f"- **Baseline Sharpe**: {degradation['baseline_sharpe']:.4f}")
    lines.append(f"- **Sharpe Drop**: {degradation['sharpe_drop']:.4f}")
    lines.append(f"- **Production Trades**: {degradation['production_trades']}")
    lines.append(f"- **Baseline Trades**: {degradation['baseline_trades']}")
    lines.append(f"- **Trade Count Drop**: {degradation['trade_count_drop']:.2%}")
    lines.append(
        f"- **Max Consecutive Losses**: {degradation['max_consecutive_losses']}"
    )
    lines.append("")

    if degradation["degradation_detected"]:
        lines.append("## ⚠️ Alerts")
        lines.append("")
        for alert in degradation["alerts"]:
            lines.append(f"- {alert}")
        lines.append("")
        lines.append("## Recommended Actions")
        lines.append("")
        lines.append("1. **Layer 1: NN Path Head** - Check IC, Rank IC, Calibration")
        lines.append("2. **Layer 2: Gate** - Check gate rules performance")
        lines.append("3. **Layer 3: Archetype** - Check archetype stability")
        lines.append("4. **Layer 4: Execution** - Check R-multiple, MAE control")
        lines.append("5. **Layer 5: PCM** - Check slot allocation, risk budgeting")
        lines.append("6. **Layer 6: Outcome** - Check realized vs predicted PnL")
    else:
        lines.append("✅ No degradation detected")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Production attribution analysis")
    p.add_argument(
        "--production-logs", required=True, help="Production logs file (parquet)"
    )
    p.add_argument(
        "--baseline-logs", required=True, help="Baseline logs file (parquet)"
    )
    p.add_argument(
        "--output-dir", required=True, help="Output directory for diagnostics"
    )
    p.add_argument(
        "--alert-thresholds",
        default='{"consecutive_losses": 5, "sharpe_drop": -0.5, "trade_count_drop": -0.2}',
        help="JSON string with alert thresholds",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    production_df = pd.read_parquet(args.production_logs)
    baseline_df = pd.read_parquet(args.baseline_logs)

    # Parse alert thresholds
    alert_thresholds = json.loads(args.alert_thresholds)

    # Detect degradation
    degradation = detect_degradation(production_df, baseline_df, alert_thresholds)

    # Save JSON
    json_path = output_dir / "degradation_report.json"
    with open(json_path, "w") as f:
        json.dump(degradation, f, indent=2)

    # Format and save report
    report = format_report(degradation, output_dir)
    md_path = output_dir / "degradation_report.md"
    md_path.write_text(report, encoding="utf-8")

    print(f"Production attribution analysis complete:")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")

    if degradation["degradation_detected"]:
        print("\n⚠️  Degradation detected! See report for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
