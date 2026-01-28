#!/usr/bin/env python3
"""
Experiment script to compare regime filtering and gate veto effects.

Runs 4 configurations:
1. Baseline: with regime filter + with gate veto
2. No regime filter: without regime filter + with gate veto
3. No gate veto: with regime filter + without gate veto
4. Both disabled: without regime filter + without gate veto

For each configuration:
- Runs apply_archetype_gate
- Runs diagnose_e2e_kpi
- Generates comparison report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run regime and gate experiments with 4 configurations."
    )
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (must contain symbol, timestamp, regime, ret_mean, ret_trend)",
    )
    p.add_argument(
        "--output-dir",
        default="results/experiments",
        help="Output directory for experiment results",
    )
    p.add_argument(
        "--features-store-root",
        default="feature_store",
        help="FeatureStore root directory",
    )
    p.add_argument(
        "--features-store-layer",
        required=True,
        help="FeatureStore layer name",
    )
    p.add_argument("--symbols", default=None, help="Comma-separated symbols")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
    )
    p.add_argument(
        "--db-path",
        default=os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", "data/order_management.db"),
    )
    p.add_argument("--evidence-quantiles", default=None)
    p.add_argument(
        "--physics-regime",
        default=None,
        help="Path to physics_regime parquet (for semantic scores). If provided and auto-compute-semantic-floors is set, will auto-compute floors.",
    )
    p.add_argument(
        "--semantic-score-floors",
        default=None,
        help="Path to semantic score floors JSON. If not provided but physics_regime is provided, will auto-compute.",
    )
    p.add_argument(
        "--auto-compute-semantic-floors",
        action="store_true",
        help="Auto-compute semantic score floors from physics_regime if not provided.",
    )
    p.add_argument(
        "--ret-mean-col", default="ret_mean", help="Column name for mean returns"
    )
    p.add_argument(
        "--ret-trend-col", default="ret_trend", help="Column name for trend returns"
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define 7 configurations (added: only_gate_rules)
    configs = [
        {
            "name": "baseline",
            "description": "With regime filter + with gate veto + with semantic veto (baseline)",
            "disable_regime_filter": False,
            "disable_gate_veto": False,
            "disable_semantic_veto": False,
            "use_semantic_floors": True,  # Enable semantic floors for baseline
        },
        {
            "name": "only_gate_rules",
            "description": "With regime filter + with gate veto only (no semantic veto)",
            "disable_regime_filter": False,
            "disable_gate_veto": False,
            "disable_semantic_veto": True,
            "use_semantic_floors": False,
        },
        {
            "name": "no_regime_filter",
            "description": "Without regime filter + with gate veto + with semantic veto",
            "disable_regime_filter": True,
            "disable_gate_veto": False,
            "disable_semantic_veto": False,
            "use_semantic_floors": True,
        },
        {
            "name": "no_gate_veto",
            "description": "With regime filter + without gate veto + with semantic veto",
            "disable_regime_filter": False,
            "disable_gate_veto": True,
            "disable_semantic_veto": False,
            "use_semantic_floors": True,
        },
        {
            "name": "no_semantic_veto",
            "description": "With regime filter + with gate veto + without semantic veto (legacy, same as only_gate_rules)",
            "disable_regime_filter": False,
            "disable_gate_veto": False,
            "disable_semantic_veto": True,
            "use_semantic_floors": False,
        },
        {
            "name": "no_regime_no_veto",
            "description": "Without regime filter + without gate veto + with semantic veto",
            "disable_regime_filter": True,
            "disable_gate_veto": True,
            "disable_semantic_veto": False,
            "use_semantic_floors": True,
        },
        {
            "name": "all_veto_off",
            "description": "Without regime filter + without gate veto + without semantic veto (all veto off)",
            "disable_regime_filter": True,
            "disable_gate_veto": True,
            "disable_semantic_veto": True,
            "use_semantic_floors": False,
        },
    ]

    results: Dict[str, Dict[str, Any]] = {}

    for config in configs:
        config_name = config["name"]
        print(f"\n{'#' * 80}")
        print(f"# Configuration: {config_name}")
        print(f"# {config['description']}")
        print(f"{'#' * 80}\n")

        # Step 1: Apply gate
        gated_file = output_dir / f"{config_name}_gated.parquet"
        gate_cmd = [
            sys.executable,
            "scripts/apply_archetype_gate.py",
            "--logs",
            str(args.logs),
            "--out",
            str(gated_file),
            "--features-store-root",
            str(args.features_store_root),
            "--features-store-layer",
            str(args.features_store_layer),
            "--timeframe",
            str(args.timeframe),
            "--execution-archetypes",
            str(args.execution_archetypes),
            "--live-config",
            str(args.db_path),
        ]

        if args.symbols:
            gate_cmd.extend(["--symbols", str(args.symbols)])
        if args.start_date:
            gate_cmd.extend(["--start-date", str(args.start_date)])
        if args.end_date:
            gate_cmd.extend(["--end-date", str(args.end_date)])
        if args.evidence_quantiles:
            gate_cmd.extend(["--evidence-quantiles", str(args.evidence_quantiles)])
        if args.physics_regime:
            gate_cmd.extend(["--physics-regime", str(args.physics_regime)])

        # Handle semantic score floors
        semantic_floors_path = args.semantic_score_floors
        if not semantic_floors_path and config.get("use_semantic_floors", False):
            # Auto-compute semantic score floors if needed but not provided
            if args.physics_regime and args.auto_compute_semantic_floors:
                temp_floors = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                )
                temp_floors_path = temp_floors.name
                temp_floors.close()
                # Compute semantic score floors
                compute_cmd = [
                    sys.executable,
                    "scripts/compute_semantic_score_floors.py",
                    "--physics-regime",
                    str(args.physics_regime),
                    "--output",
                    temp_floors_path,
                ]
                compute_result = subprocess.run(
                    compute_cmd, check=True, cwd=str(PROJECT_ROOT)
                )
                if compute_result.returncode == 0:
                    semantic_floors_path = temp_floors_path
                    print(f"✅ Auto-computed semantic score floors: {temp_floors_path}")

        if semantic_floors_path:
            gate_cmd.extend(["--semantic-score-floors", str(semantic_floors_path)])

        if config["disable_regime_filter"]:
            gate_cmd.append("--disable-regime-filter")
        if config["disable_gate_veto"]:
            gate_cmd.append("--disable-gate-veto")
        if config["disable_semantic_veto"]:
            gate_cmd.append("--disable-semantic-veto")

        exit_code = run_command(gate_cmd, f"Apply gate for {config_name}")
        if exit_code != 0:
            print(f"❌ Gate application failed for {config_name}")
            return exit_code

        # Step 2: Run E2E KPI diagnostics
        kpi_json = output_dir / f"{config_name}_kpi.json"
        kpi_md = output_dir / f"{config_name}_kpi.md"
        kpi_cmd = [
            sys.executable,
            "scripts/diagnose_e2e_kpi.py",
            "--logs",
            str(args.logs),
            "--gate",
            str(gated_file),
            "--output-json",
            str(kpi_json),
            "--output-md",
            str(kpi_md),
            "--ret-mean-col",
            str(args.ret_mean_col),
            "--ret-trend-col",
            str(args.ret_trend_col),
        ]

        exit_code = run_command(kpi_cmd, f"E2E KPI diagnostics for {config_name}")
        if exit_code != 0:
            print(f"❌ E2E KPI diagnostics failed for {config_name}")
            return exit_code

        # Load KPI results
        if kpi_json.exists():
            with open(kpi_json, "r") as f:
                results[config_name] = json.load(f)
                results[config_name]["description"] = config["description"]
                results[config_name]["gated_file"] = str(gated_file)
                results[config_name]["kpi_file"] = str(kpi_md)

    # Step 3: Generate comparison report
    comparison_file = output_dir / "regime_gate_comparison.md"
    generate_comparison_report(results, comparison_file)

    print(f"\n{'=' * 80}")
    print(f"✅ Experiment completed!")
    print(f"   Results directory: {output_dir}")
    print(f"   Comparison report: {comparison_file}")
    print(f"{'=' * 80}\n")

    return 0


def generate_comparison_report(
    results: Dict[str, Dict[str, Any]], output_file: Path
) -> None:
    """Generate a markdown comparison report."""
    lines = [
        "# Regime and Gate Experiment Comparison Report",
        "",
        "## Experiment Overview",
        "",
        "This report compares 6 configurations:",
        "",
        "1. **Baseline**: With regime filter + with gate veto + with semantic veto",
        "2. **No Regime Filter**: Without regime filter + with gate veto + with semantic veto",
        "3. **No Gate Veto**: With regime filter + without gate veto + with semantic veto",
        "4. **No Semantic Veto**: With regime filter + with gate veto + without semantic veto",
        "5. **No Regime No Veto**: Without regime filter + without gate veto + with semantic veto",
        "6. **All Veto Off**: Without regime filter + without gate veto + without semantic veto",
        "",
        "---",
        "",
        "## Overall KPI Comparison",
        "",
        "| Configuration | Sharpe | Trades | Win Rate | Profit/Loss Ratio |",
        "|--------------|--------|--------|----------|-------------------|",
    ]

    for config_name, data in results.items():
        overall = data.get("overall", {})
        sharpe = overall.get("sharpe", 0.0)
        trades = overall.get("trade_count", 0)
        win_rate = overall.get("win_rate", 0.0)
        pl_ratio = overall.get("profit_loss_ratio", 0.0)

        lines.append(
            f"| {config_name} | {sharpe:.3f} | {trades} | {win_rate:.1%} | {pl_ratio:.2f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Detailed Results",
            "",
        ]
    )

    for config_name, data in results.items():
        lines.extend(
            [
                f"### {config_name}",
                "",
                f"**Description**: {data.get('description', 'N/A')}",
                "",
                f"- Gated file: `{data.get('gated_file', 'N/A')}`",
                f"- KPI report: `{data.get('kpi_file', 'N/A')}`",
                "",
            ]
        )

        # Add per-symbol summary if available
        by_symbol = data.get("by_symbol", {})
        if by_symbol:
            lines.extend(
                [
                    "#### Per Symbol Summary",
                    "",
                    "| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |",
                    "|--------|--------|--------|----------|-------------------|",
                ]
            )
            for symbol, metrics in by_symbol.items():
                sharpe = metrics.get("sharpe", 0.0)
                trades = metrics.get("trade_count", 0)
                win_rate = metrics.get("win_rate", 0.0)
                pl_ratio = metrics.get("profit_loss_ratio", 0.0)
                lines.append(
                    f"| {symbol} | {sharpe:.3f} | {trades} | {win_rate:.1%} | {pl_ratio:.2f} |"
                )
            lines.append("")

        # Add per-archetype summary if available
        by_archetype = data.get("by_archetype", {})
        if by_archetype:
            lines.extend(
                [
                    "#### Per Archetype Summary",
                    "",
                    "| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |",
                    "|-----------|--------|--------|----------|-------------------|",
                ]
            )
            for archetype, metrics in by_archetype.items():
                sharpe = metrics.get("sharpe", 0.0)
                trades = metrics.get("trade_count", 0)
                win_rate = metrics.get("win_rate", 0.0)
                pl_ratio = metrics.get("profit_loss_ratio", 0.0)
                lines.append(
                    f"| {archetype} | {sharpe:.3f} | {trades} | {win_rate:.1%} | {pl_ratio:.2f} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Add conclusions
    lines.extend(
        [
            "## Conclusions",
            "",
            "### Key Findings",
            "",
            "1. **Regime Filter Impact**: Compare baseline vs no_regime_filter",
            "2. **Gate Veto Impact**: Compare baseline vs no_gate_veto",
            "3. **Combined Impact**: Compare baseline vs no_regime_no_veto",
            "",
            "### FR/ET Trading Statistics",
            "",
            "Check the detailed KPI reports for FR/ET archetype trading statistics.",
            "",
        ]
    )

    output_file.write_text("\n".join(lines))
    print(f"✅ Comparison report saved: {output_file}")


if __name__ == "__main__":
    raise SystemExit(main())
