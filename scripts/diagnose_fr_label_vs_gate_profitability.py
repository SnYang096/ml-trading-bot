#!/usr/bin/env python3
"""
Experiment 5: Label vs Gate Profitability Analysis

Question: Are labels profitable but gates filtering out profitable trades?

This script:
1. Computes label-based profitability (ret_mean > 0) for ALL rows
2. Computes label-based profitability for gate-passed rows
3. Compares: label-profitable vs gate-passed overlap
4. Identifies: which gate rules filter out profitable labels?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec
from src.time_series_model.core.constitution.execution_evidence import (
    load_evidence_quantiles,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


def _sharpe(returns: pd.Series) -> float:
    """Compute Sharpe ratio"""
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def analyze_label_profitability(
    df: pd.DataFrame,
    ret_col: str = "ret_mean",
) -> Dict[str, any]:
    """Analyze label profitability for all rows"""
    profitable = df[df[ret_col] > 0]
    unprofitable = df[df[ret_col] <= 0]

    return {
        "total_rows": len(df),
        "profitable_count": len(profitable),
        "unprofitable_count": len(unprofitable),
        "profitable_rate": len(profitable) / len(df) if len(df) > 0 else 0.0,
        "profitable_sharpe": (
            _sharpe(profitable[ret_col]) if len(profitable) > 0 else 0.0
        ),
        "profitable_mean_return": (
            float(profitable[ret_col].mean()) if len(profitable) > 0 else 0.0
        ),
        "unprofitable_mean_return": (
            float(unprofitable[ret_col].mean()) if len(unprofitable) > 0 else 0.0
        ),
        "overall_sharpe": _sharpe(df[ret_col]),
        "overall_mean_return": float(df[ret_col].mean()),
    }


def _read_feature_store_range(
    *,
    features_store_root: str,
    layer: str,
    symbols: List[str],
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """Load features from FeatureStore"""
    store = FeatureStore(str(features_store_root))
    parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(timeframe)
        )
        start_ts = pd.Timestamp(start) if start else pd.Timestamp("1970-01-01")
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2100-01-01")
        df_sym = store.read_range(spec, start=start_ts, end=end_ts)
        if df_sym.empty:
            print(f"⚠️  Empty FeatureStore read for symbol={sym}, layer={layer}")
            continue
        if "symbol" not in df_sym.columns:
            df_sym = df_sym.copy()
            df_sym["symbol"] = sym
        parts.append(df_sym)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=0, ignore_index=False)
    if "timestamp" not in df.columns:
        if getattr(df.index, "name", None) == "timestamp":
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df.index, utc=False, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["timestamp"] = df.index
        else:
            raise KeyError(
                "Expected FeatureStore data to have a 'timestamp' column or DatetimeIndex"
            )
    # Ensure timestamp is a column (not index)
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=False)
            if "timestamp" not in df.columns:
                df["timestamp"] = df.index
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["timestamp"] = df.index
    return df


def apply_gate_to_df(
    df: pd.DataFrame,
    archetype_name: str,
    arches: Dict,
    quantiles: Optional[Dict] = None,
    features_store_root: Optional[str] = None,
    features_layer: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> pd.Series:
    """Apply gate rules to DataFrame, return gate_ok Series"""
    arch = arches.get(archetype_name)
    if not arch:
        return pd.Series(False, index=df.index)

    gate_cfg = {
        "when_then_rules": list(getattr(arch, "when_then_rules", []) or []),
        "default_action": str(getattr(arch, "default_action", "deny")),
    }

    # Load features from FeatureStore if needed
    if features_store_root and features_layer and timeframe:
        print("📊 Loading features from FeatureStore...")
        symbols = df["symbol"].unique().tolist()
        start = df["timestamp"].min().isoformat() if "timestamp" in df.columns else None
        end = df["timestamp"].max().isoformat() if "timestamp" in df.columns else None
        features_df = _read_feature_store_range(
            features_store_root=features_store_root,
            layer=features_layer,
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        if not features_df.empty:
            # Ensure timestamp is a column in df
            if "timestamp" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index(drop=False)
                    if "timestamp" not in df.columns:
                        df["timestamp"] = df.index
            # Handle features_df: if timestamp is in index but not in columns, reset index
            # But if timestamp is already in columns, don't reset
            if "timestamp" not in features_df.columns:
                if getattr(
                    features_df.index, "name", None
                ) == "timestamp" or isinstance(features_df.index, pd.DatetimeIndex):
                    features_df = features_df.reset_index(drop=False)
            # If timestamp is in both index and columns, drop from index
            elif getattr(features_df.index, "name", None) == "timestamp":
                features_df = features_df.reset_index(drop=True)
            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            features_df["timestamp"] = pd.to_datetime(
                features_df["timestamp"], errors="coerce"
            )
            # Merge features
            merge_cols = ["symbol", "timestamp"]
            df = df.merge(
                features_df,
                on=merge_cols,
                how="left",
                suffixes=("", "_feat"),
            )
            print(f"✅ Merged {len(features_df.columns)} features from FeatureStore")

    gate_ok_list = []
    for idx, row in df.iterrows():
        features = row.to_dict()
        # Remove _feat suffix columns
        features = {k: v for k, v in features.items() if not k.endswith("_feat")}
        ok, _ = apply_gate_rules(
            gate_rules=gate_cfg,
            features=features,
            quantiles=quantiles,
        )
        gate_ok_list.append(ok)

    return pd.Series(gate_ok_list, index=df.index)


def analyze_gate_filtering_effect(
    df: pd.DataFrame,
    gate_ok: pd.Series,
    ret_col: str = "ret_mean",
) -> Dict[str, any]:
    """Analyze how gate filtering affects profitability"""
    passed = df[gate_ok]
    filtered = df[~gate_ok]

    passed_profitable = analyze_label_profitability(passed, ret_col)
    filtered_profitable = analyze_label_profitability(filtered, ret_col)
    all_profitable = analyze_label_profitability(df, ret_col)

    # Overlap analysis
    profitable_labels = df[df[ret_col] > 0]
    profitable_labels_passed = profitable_labels[gate_ok[profitable_labels.index]]
    profitable_labels_filtered = profitable_labels[~gate_ok[profitable_labels.index]]

    return {
        "all_rows": all_profitable,
        "gate_passed": passed_profitable,
        "gate_filtered": filtered_profitable,
        "overlap": {
            "profitable_labels_total": len(profitable_labels),
            "profitable_labels_passed": len(profitable_labels_passed),
            "profitable_labels_filtered": len(profitable_labels_filtered),
            "profitable_labels_pass_rate": (
                len(profitable_labels_passed) / len(profitable_labels)
                if len(profitable_labels) > 0
                else 0.0
            ),
            "profitable_labels_filter_rate": (
                len(profitable_labels_filtered) / len(profitable_labels)
                if len(profitable_labels) > 0
                else 0.0
            ),
        },
        "gate_stats": {
            "total_passed": len(passed),
            "total_filtered": len(filtered),
            "pass_rate": len(passed) / len(df) if len(df) > 0 else 0.0,
        },
    }


def analyze_rule_impact_on_profitable_labels(
    df: pd.DataFrame,
    archetype_name: str,
    arches: Dict,
    quantiles: Optional[Dict] = None,
    ret_col: str = "ret_mean",
    features_store_root: Optional[str] = None,
    features_layer: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, any]:
    """Analyze which rules filter out profitable labels"""
    arch = arches.get(archetype_name)
    if not arch:
        return {}

    profitable_labels = df[df[ret_col] > 0].copy()
    if len(profitable_labels) == 0:
        return {}

    # Load features from FeatureStore if needed
    if features_store_root and features_layer and timeframe:
        symbols = profitable_labels["symbol"].unique().tolist()
        start = (
            profitable_labels["timestamp"].min().isoformat()
            if "timestamp" in profitable_labels.columns
            else None
        )
        end = (
            profitable_labels["timestamp"].max().isoformat()
            if "timestamp" in profitable_labels.columns
            else None
        )
        features_df = _read_feature_store_range(
            features_store_root=features_store_root,
            layer=features_layer,
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        if not features_df.empty:
            # Handle timestamp in features_df
            if "timestamp" not in features_df.columns:
                if getattr(
                    features_df.index, "name", None
                ) == "timestamp" or isinstance(features_df.index, pd.DatetimeIndex):
                    features_df = features_df.reset_index(drop=False)
            elif getattr(features_df.index, "name", None) == "timestamp":
                features_df = features_df.reset_index(drop=True)
            # Ensure timestamp is datetime
            features_df["timestamp"] = pd.to_datetime(
                features_df["timestamp"], errors="coerce"
            )
            profitable_labels["timestamp"] = pd.to_datetime(
                profitable_labels["timestamp"], errors="coerce"
            )
            merge_cols = ["symbol", "timestamp"]
            profitable_labels = profitable_labels.merge(
                features_df,
                on=merge_cols,
                how="left",
                suffixes=("", "_feat"),
            )

    rules = list(getattr(arch, "when_then_rules", []) or [])
    rule_impacts = []

    # Test each rule individually
    for rule in rules:
        rule_id = rule.get("id", "")
        phase = rule.get("phase", "")
        reason = rule.get("reason", "")

        # Create a gate config with only this rule
        test_gate_cfg = {
            "when_then_rules": [rule],
            "default_action": "allow",  # Default allow to see rule's effect
        }

        gate_ok_list = []
        for idx, row in profitable_labels.iterrows():
            features = row.to_dict()
            # Remove _feat suffix columns
            features = {k: v for k, v in features.items() if not k.endswith("_feat")}
            ok, _ = apply_gate_rules(
                gate_rules=test_gate_cfg,
                features=features,
                quantiles=quantiles,
            )
            gate_ok_list.append(ok)

        gate_ok_series = pd.Series(gate_ok_list, index=profitable_labels.index)
        passed = profitable_labels[gate_ok_series]
        filtered = profitable_labels[~gate_ok_series]

        rule_impacts.append(
            {
                "rule_id": rule_id,
                "phase": phase,
                "reason": reason,
                "action": rule.get("then", {}).get("action", ""),
                "profitable_labels_passed": len(passed),
                "profitable_labels_filtered": len(filtered),
                "filter_rate": (
                    len(filtered) / len(profitable_labels)
                    if len(profitable_labels) > 0
                    else 0.0
                ),
                "passed_sharpe": _sharpe(passed[ret_col]) if len(passed) > 0 else 0.0,
                "filtered_sharpe": (
                    _sharpe(filtered[ret_col]) if len(filtered) > 0 else 0.0
                ),
                "passed_mean_return": (
                    float(passed[ret_col].mean()) if len(passed) > 0 else 0.0
                ),
                "filtered_mean_return": (
                    float(filtered[ret_col].mean()) if len(filtered) > 0 else 0.0
                ),
            }
        )

    return {
        "total_profitable_labels": len(profitable_labels),
        "rule_impacts": sorted(
            rule_impacts, key=lambda x: x["filter_rate"], reverse=True
        ),
    }


def generate_report(
    results: Dict[str, any],
    output_md: Path,
    output_json: Path,
) -> None:
    """Generate Markdown and JSON reports"""
    md = "# FR Label vs Gate Profitability Analysis (Experiment 5)\n\n"
    md += "## Question\n\n"
    md += "Are labels profitable but gates filtering out profitable trades?\n\n"
    md += "---\n\n"

    # Section 1: Overall Label Profitability
    md += "## 1. Overall Label Profitability (All Rows)\n\n"
    all_stats = results["all_rows"]
    md += f"- **Total Rows**: {all_stats['total_rows']:,}\n"
    md += f"- **Profitable Labels** (ret_mean > 0): {all_stats['profitable_count']:,} ({all_stats['profitable_rate']:.1%})\n"
    md += f"- **Unprofitable Labels** (ret_mean <= 0): {all_stats['unprofitable_count']:,} ({1 - all_stats['profitable_rate']:.1%})\n"
    md += f"- **Overall Sharpe**: {all_stats['overall_sharpe']:.4f}\n"
    md += f"- **Overall Mean Return**: {all_stats['overall_mean_return']:.6f}\n"
    md += f"- **Profitable Labels Sharpe**: {all_stats['profitable_sharpe']:.4f}\n"
    md += f"- **Profitable Labels Mean Return**: {all_stats['profitable_mean_return']:.6f}\n"
    md += f"- **Unprofitable Labels Mean Return**: {all_stats['unprofitable_mean_return']:.6f}\n\n"

    # Section 2: Gate Filtering Effect
    md += "## 2. Gate Filtering Effect\n\n"
    gate_stats = results["gate_stats"]
    md += f"- **Gate Pass Rate**: {gate_stats['pass_rate']:.1%} ({gate_stats['total_passed']:,} / {gate_stats['total_passed'] + gate_stats['total_filtered']:,})\n"
    md += f"- **Gate Filter Rate**: {1 - gate_stats['pass_rate']:.1%} ({gate_stats['total_filtered']:,} / {gate_stats['total_passed'] + gate_stats['total_filtered']:,})\n\n"

    passed_stats = results["gate_passed"]
    filtered_stats = results["gate_filtered"]
    md += "### Gate-Passed Rows\n"
    md += f"- **Count**: {passed_stats['total_rows']:,}\n"
    md += f"- **Profitable Rate**: {passed_stats['profitable_rate']:.1%}\n"
    md += f"- **Sharpe**: {passed_stats['overall_sharpe']:.4f}\n"
    md += f"- **Mean Return**: {passed_stats['overall_mean_return']:.6f}\n\n"

    md += "### Gate-Filtered Rows\n"
    md += f"- **Count**: {filtered_stats['total_rows']:,}\n"
    md += f"- **Profitable Rate**: {filtered_stats['profitable_rate']:.1%}\n"
    md += f"- **Sharpe**: {filtered_stats['overall_sharpe']:.4f}\n"
    md += f"- **Mean Return**: {filtered_stats['overall_mean_return']:.6f}\n\n"

    # Section 3: Overlap Analysis
    md += "## 3. Profitable Labels vs Gate Overlap\n\n"
    overlap = results["overlap"]
    md += f"- **Total Profitable Labels**: {overlap['profitable_labels_total']:,}\n"
    md += f"- **Profitable Labels Passed Gate**: {overlap['profitable_labels_passed']:,} ({overlap['profitable_labels_pass_rate']:.1%})\n"
    md += f"- **Profitable Labels Filtered by Gate**: {overlap['profitable_labels_filtered']:,} ({overlap['profitable_labels_filter_rate']:.1%})\n\n"

    md += "### Key Finding\n\n"
    if overlap["profitable_labels_filter_rate"] > 0.3:
        md += f"⚠️ **WARNING**: Gate is filtering out {overlap['profitable_labels_filter_rate']:.1%} of profitable labels!\n\n"
    else:
        md += f"✅ Gate filtering is relatively selective (only {overlap['profitable_labels_filter_rate']:.1%} of profitable labels filtered).\n\n"

    # Section 4: Rule Impact Analysis
    md += "## 4. Individual Rule Impact on Profitable Labels\n\n"
    rule_impacts = results.get("rule_impacts", {})
    if rule_impacts:
        md += (
            f"Total Profitable Labels: {rule_impacts['total_profitable_labels']:,}\n\n"
        )
        md += "| Rule ID | Phase | Reason | Action | Filtered | Filter Rate | Passed Sharpe | Filtered Sharpe |\n"
        md += "|---------|-------|--------|--------|----------|-------------|---------------|-----------------|\n"

        for impact in rule_impacts.get("rule_impacts", []):
            md += (
                f"| {impact['rule_id']} | {impact['phase']} | {impact['reason']} | "
                f"{impact['action']} | {impact['profitable_labels_filtered']} | "
                f"{impact['filter_rate']:.1%} | {impact['passed_sharpe']:.4f} | "
                f"{impact['filtered_sharpe']:.4f} |\n"
            )

        md += "\n### Rules Filtering Out Most Profitable Labels\n\n"
        top_filters = sorted(
            rule_impacts.get("rule_impacts", []),
            key=lambda x: x["filter_rate"],
            reverse=True,
        )[:5]
        for impact in top_filters:
            md += f"- **{impact['rule_id']}** ({impact['phase']}): Filters {impact['filter_rate']:.1%} of profitable labels\n"
            md += f"  - Passed Sharpe: {impact['passed_sharpe']:.4f}, Filtered Sharpe: {impact['filtered_sharpe']:.4f}\n"

    md += "\n---\n\n"
    md += "## 5. Conclusions\n\n"

    # Generate conclusions
    conclusions = []
    if overlap["profitable_labels_filter_rate"] > 0.3:
        conclusions.append(
            f"Gate is filtering out {overlap['profitable_labels_filter_rate']:.1%} of profitable labels. "
            "This suggests gates may be too strict or targeting wrong criteria."
        )
    else:
        conclusions.append(
            f"Gate filtering is relatively selective ({overlap['profitable_labels_filter_rate']:.1%} of profitable labels filtered)."
        )

    if passed_stats["overall_sharpe"] < 0:
        conclusions.append(
            f"Even gate-passed rows have negative Sharpe ({passed_stats['overall_sharpe']:.4f}). "
            "This suggests the problem is not just gate filtering, but the underlying labels or strategy."
        )
    else:
        conclusions.append(
            f"Gate-passed rows have positive Sharpe ({passed_stats['overall_sharpe']:.4f}). "
            "Gate filtering is working correctly."
        )

    if filtered_stats["profitable_rate"] > passed_stats["profitable_rate"]:
        conclusions.append(
            f"Filtered rows have higher profitable rate ({filtered_stats['profitable_rate']:.1%}) than passed rows ({passed_stats['profitable_rate']:.1%}). "
            "This suggests gates are filtering out profitable opportunities."
        )

    for conclusion in conclusions:
        md += f"- {conclusion}\n"

    md += "\n"

    # Write reports
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Report written to: {output_md}")
    print(f"✅ JSON written to: {output_json}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Label vs Gate Profitability Analysis"
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="logs_execution.parquet path",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml path",
    )
    parser.add_argument(
        "--evidence-quantiles",
        default=None,
        help="evidence_quantiles.json path",
    )
    parser.add_argument(
        "--features-store-root",
        default=None,
        help="FeatureStore root directory",
    )
    parser.add_argument(
        "--features-layer",
        default="physical",
        help="FeatureStore layer (e.g., 'physical')",
    )
    parser.add_argument(
        "--timeframe",
        default="4h",
        help="Timeframe for FeatureStore (e.g., '4h')",
    )
    parser.add_argument(
        "--archetype",
        default="FailureReversionFR",
        help="Archetype to analyze",
    )
    parser.add_argument(
        "--output-md",
        default="results/fr_label_vs_gate_profitability_report.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--output-json",
        default="results/fr_label_vs_gate_profitability_report.json",
        help="Output JSON report path",
    )

    args = parser.parse_args()

    print("📊 Loading data...")
    df = pd.read_parquet(args.logs)
    # Reset index to avoid duplicate index issues
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows")

    print("📊 Loading archetype config...")
    arches = load_execution_archetypes_registry(args.execution_archetypes)

    print("📊 Loading evidence quantiles...")
    quantiles = None
    if args.evidence_quantiles and Path(args.evidence_quantiles).exists():
        quantiles_raw = load_evidence_quantiles(args.evidence_quantiles)
        # Handle per-symbol quantiles
        if isinstance(quantiles_raw, dict) and "BTCUSDT" in quantiles_raw:
            # Per-symbol format, we'll use symbol-specific quantiles in apply_gate_to_df
            quantiles = quantiles_raw
        else:
            quantiles = quantiles_raw
        print("✅ Evidence quantiles loaded")
    else:
        print("⚠️  No evidence quantiles provided")

    print("\n🔍 Experiment 5: Label vs Gate Profitability Analysis\n")

    # Step 1: Analyze overall label profitability
    print("Step 1: Analyzing overall label profitability...")
    all_stats = analyze_label_profitability(df)

    # Step 2: Apply gate and analyze filtering effect
    print("Step 2: Applying gate rules...")
    # For per-symbol quantiles, we need to apply gate per symbol
    gate_ok_list = []
    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].copy()
        sym_quantiles = (
            quantiles.get(str(symbol))
            if isinstance(quantiles, dict) and str(symbol) in quantiles
            else quantiles
        )
        sym_gate_ok = apply_gate_to_df(
            sym_df,
            args.archetype,
            arches,
            sym_quantiles,
            features_store_root=args.features_store_root,
            features_layer=args.features_layer,
            timeframe=args.timeframe,
        )
        gate_ok_list.append(sym_gate_ok)

    gate_ok = pd.concat(gate_ok_list)
    # Ensure index alignment with original df
    # Create a mapping from (symbol, timestamp) to gate_ok value
    if len(gate_ok) == len(df):
        # If lengths match, try direct alignment
        gate_ok.index = df.index
    else:
        # Otherwise, use merge on symbol+timestamp
        df_with_gate = df.copy()
        df_with_gate["_temp_idx"] = range(len(df_with_gate))
        gate_df = pd.DataFrame({"gate_ok": gate_ok})
        gate_df["symbol"] = (
            df.loc[gate_ok.index, "symbol"].values if "symbol" in df.columns else None
        )
        gate_df["timestamp"] = (
            df.loc[gate_ok.index, "timestamp"].values
            if "timestamp" in df.columns
            else None
        )
        merged = df_with_gate.merge(
            gate_df, on=["symbol", "timestamp"], how="left", suffixes=("", "_gate")
        )
        merged = merged.sort_values("_temp_idx")
        gate_ok = merged["gate_ok"].fillna(False)
        gate_ok.index = df.index

    print("Step 3: Analyzing gate filtering effect...")
    gate_effect = analyze_gate_filtering_effect(df, gate_ok)

    # Step 4: Analyze individual rule impacts
    print("Step 4: Analyzing individual rule impacts...")
    rule_impacts = analyze_rule_impact_on_profitable_labels(
        df,
        args.archetype,
        arches,
        quantiles,
        features_store_root=args.features_store_root,
        features_layer=args.features_layer,
        timeframe=args.timeframe,
    )

    # Combine results
    results = {
        "all_rows": all_stats,
        "gate_passed": gate_effect["gate_passed"],
        "gate_filtered": gate_effect["gate_filtered"],
        "overlap": gate_effect["overlap"],
        "gate_stats": gate_effect["gate_stats"],
        "rule_impacts": rule_impacts,
    }

    # Generate report
    print("\n📝 Generating report...")
    generate_report(
        results,
        Path(args.output_md),
        Path(args.output_json),
    )

    print("\n✅ Experiment 5 complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
