#!/usr/bin/env python3
"""
FR Sharpe Root Cause Diagnostic Script

Implements 5 experiments to identify root causes of negative FR Sharpe:
1. Period Analysis: Compare FR performance across time periods
2. Gate Regime Filtering: Test gate effectiveness
3. Threshold Scanning: Find optimal thresholds
4. Rule Complexity: Test minimal vs full gate configurations
5. Label Profitability: Compare label vs gate-filtered profitability (already done separately)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

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
    if "timestamp" not in df.columns:
        if getattr(df.index, "name", None) == "timestamp":
            df = df.reset_index(drop=False)
            if "timestamp" not in df.columns:
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
    gate_config_override: Optional[Dict] = None,
) -> pd.Series:
    """Apply gate rules to DataFrame, return gate_ok Series"""
    arch = arches.get(archetype_name)
    if not arch:
        return pd.Series(False, index=df.index)

    # Use override config if provided, otherwise use arch config
    if gate_config_override:
        gate_cfg = gate_config_override
    else:
        gate_cfg = {
            "when_then_rules": list(getattr(arch, "when_then_rules", []) or []),
            "default_action": str(getattr(arch, "default_action", "deny")),
        }

    # Load features from FeatureStore if needed
    if features_store_root and features_layer and timeframe:
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
            if "timestamp" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index(drop=False)
                    if "timestamp" not in df.columns:
                        df["timestamp"] = df.index
            if "timestamp" not in features_df.columns:
                if getattr(
                    features_df.index, "name", None
                ) == "timestamp" or isinstance(features_df.index, pd.DatetimeIndex):
                    features_df = features_df.reset_index(drop=False)
            elif getattr(features_df.index, "name", None) == "timestamp":
                features_df = features_df.reset_index(drop=True)
            features_df["timestamp"] = pd.to_datetime(
                features_df["timestamp"], errors="coerce"
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.merge(
                features_df,
                on=["symbol", "timestamp"],
                how="left",
                suffixes=("", "_feat"),
            )

    gate_ok_list = []
    for idx, row in df.iterrows():
        features = row.to_dict()
        features = {k: v for k, v in features.items() if not k.endswith("_feat")}
        ok, _ = apply_gate_rules(
            gate_rules=gate_cfg,
            features=features,
            quantiles=quantiles,
        )
        gate_ok_list.append(ok)

    return pd.Series(gate_ok_list, index=df.index)


# ============================================================================
# Experiment 1: Period Analysis
# ============================================================================


def experiment1_period_analysis(
    df: pd.DataFrame,
    ret_col: str = "ret_mean",
) -> Dict[str, Any]:
    """Experiment 1: Analyze FR performance by time period"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Split by quarters
    df["quarter"] = df["timestamp"].dt.to_period("Q")
    df["month"] = df["timestamp"].dt.to_period("M")

    results = {
        "by_quarter": [],
        "by_month": [],
        "overall": {},
    }

    # Overall stats
    profitable = df[df[ret_col] > 0]
    results["overall"] = {
        "total_rows": len(df),
        "profitable_count": len(profitable),
        "profitable_rate": len(profitable) / len(df) if len(df) > 0 else 0.0,
        "sharpe": _sharpe(df[ret_col]),
        "mean_return": float(df[ret_col].mean()),
        "win_rate": float((df[ret_col] > 0).mean()),
    }

    # By quarter
    for quarter, group in df.groupby("quarter"):
        profitable_q = group[group[ret_col] > 0]
        results["by_quarter"].append(
            {
                "period": str(quarter),
                "total_rows": len(group),
                "profitable_count": len(profitable_q),
                "profitable_rate": (
                    len(profitable_q) / len(group) if len(group) > 0 else 0.0
                ),
                "sharpe": _sharpe(group[ret_col]),
                "mean_return": float(group[ret_col].mean()),
                "win_rate": float((group[ret_col] > 0).mean()),
            }
        )

    # By month
    for month, group in df.groupby("month"):
        profitable_m = group[group[ret_col] > 0]
        results["by_month"].append(
            {
                "period": str(month),
                "total_rows": len(group),
                "profitable_count": len(profitable_m),
                "profitable_rate": (
                    len(profitable_m) / len(group) if len(group) > 0 else 0.0
                ),
                "sharpe": _sharpe(group[ret_col]),
                "mean_return": float(group[ret_col].mean()),
                "win_rate": float((group[ret_col] > 0).mean()),
            }
        )

    return results


def generate_exp1_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate Experiment 1 report"""
    md = "# Experiment 1: Period Analysis\n\n"
    md += "## Question\n\n"
    md += "Is 2024 bull market inherently bad for FR (mean reversion)?\n\n"
    md += "---\n\n"

    md += "## Overall Statistics\n\n"
    overall = results["overall"]
    md += f"- **Total Rows**: {overall['total_rows']:,}\n"
    md += f"- **Profitable Rate**: {overall['profitable_rate']:.1%}\n"
    md += f"- **Sharpe**: {overall['sharpe']:.4f}\n"
    md += f"- **Mean Return**: {overall['mean_return']:.6f}\n"
    md += f"- **Win Rate**: {overall['win_rate']:.1%}\n\n"

    md += "## Performance by Quarter\n\n"
    md += "| Quarter | Rows | Profitable Rate | Sharpe | Mean Return | Win Rate |\n"
    md += "|---------|------|-----------------|--------|-------------|----------|\n"
    for q in sorted(results["by_quarter"], key=lambda x: x["period"]):
        md += f"| {q['period']} | {q['total_rows']} | {q['profitable_rate']:.1%} | {q['sharpe']:.4f} | {q['mean_return']:.6f} | {q['win_rate']:.1%} |\n"

    md += "\n## Performance by Month\n\n"
    md += "| Month | Rows | Profitable Rate | Sharpe | Mean Return | Win Rate |\n"
    md += "|-------|------|-----------------|--------|-------------|----------|\n"
    for m in sorted(results["by_month"], key=lambda x: x["period"]):
        md += f"| {m['period']} | {m['total_rows']} | {m['profitable_rate']:.1%} | {m['sharpe']:.4f} | {m['mean_return']:.6f} | {m['win_rate']:.1%} |\n"

    md += "\n## Conclusions\n\n"
    sharpe_by_quarter = [q["sharpe"] for q in results["by_quarter"]]
    if sharpe_by_quarter:
        best_q = max(results["by_quarter"], key=lambda x: x["sharpe"])
        worst_q = min(results["by_quarter"], key=lambda x: x["sharpe"])
        md += (
            f"- **Best Quarter**: {best_q['period']} (Sharpe: {best_q['sharpe']:.4f})\n"
        )
        md += f"- **Worst Quarter**: {worst_q['period']} (Sharpe: {worst_q['sharpe']:.4f})\n"
        if all(s < 0 for s in sharpe_by_quarter):
            md += "- ⚠️ **All quarters show negative Sharpe**, suggesting the problem is not period-specific.\n"
        else:
            md += "- ✅ Some quarters show positive Sharpe, suggesting period-specific factors may be at play.\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# ============================================================================
# Experiment 2: Gate Regime Filtering
# ============================================================================


def experiment2_gate_regime_filtering(
    df: pd.DataFrame,
    archetype_name: str,
    arches: Dict,
    quantiles: Optional[Dict] = None,
    features_store_root: Optional[str] = None,
    features_layer: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Experiment 2: Test gate regime filtering effectiveness"""
    results = {}

    # Baseline: No gate (all rows)
    results["no_gate"] = {
        "total_rows": len(df),
        "trade_count": len(df),
        "sharpe": _sharpe(df["ret_mean"]),
        "mean_return": float(df["ret_mean"].mean()),
        "win_rate": float((df["ret_mean"] > 0).mean()),
        "profitable_rate": float((df["ret_mean"] > 0).mean()),
    }

    # Regime-only: Only exclusions (jump_risk, adx)
    arch = arches.get(archetype_name)
    if arch:
        regime_only_rules = []
        for rule in getattr(arch, "when_then_rules", []) or []:
            phase = rule.get("phase", "")
            if phase in ("safety", "exclusions"):
                regime_only_rules.append(rule)

        regime_only_cfg = {
            "when_then_rules": regime_only_rules,
            "default_action": "allow",  # Allow by default, only deny on exclusions
        }

        gate_ok_regime = apply_gate_to_df(
            df,
            archetype_name,
            arches,
            quantiles,
            features_store_root,
            features_layer,
            timeframe,
            gate_config_override=regime_only_cfg,
        )

        passed_regime = df[gate_ok_regime]
        results["regime_only"] = {
            "total_rows": len(df),
            "trade_count": len(passed_regime),
            "pass_rate": len(passed_regime) / len(df) if len(df) > 0 else 0.0,
            "sharpe": (
                _sharpe(passed_regime["ret_mean"]) if len(passed_regime) > 0 else 0.0
            ),
            "mean_return": (
                float(passed_regime["ret_mean"].mean())
                if len(passed_regime) > 0
                else 0.0
            ),
            "win_rate": (
                float((passed_regime["ret_mean"] > 0).mean())
                if len(passed_regime) > 0
                else 0.0
            ),
            "profitable_rate": (
                float((passed_regime["ret_mean"] > 0).mean())
                if len(passed_regime) > 0
                else 0.0
            ),
        }

    # Full gate: Current when-then rules
    gate_ok_full = apply_gate_to_df(
        df,
        archetype_name,
        arches,
        quantiles,
        features_store_root,
        features_layer,
        timeframe,
    )

    # Align indices
    if len(gate_ok_full) == len(df):
        gate_ok_full.index = df.index
    else:
        # Merge approach
        df_with_idx = df.copy()
        df_with_idx["_temp_idx"] = range(len(df_with_idx))
        gate_df = pd.DataFrame({"gate_ok": gate_ok_full})
        if "symbol" in df.columns and "timestamp" in df.columns:
            gate_df["symbol"] = (
                df.loc[gate_ok_full.index, "symbol"].values
                if all(i in df.index for i in gate_ok_full.index)
                else None
            )
            gate_df["timestamp"] = (
                df.loc[gate_ok_full.index, "timestamp"].values
                if all(i in df.index for i in gate_ok_full.index)
                else None
            )
            if gate_df["symbol"].notna().any() and gate_df["timestamp"].notna().any():
                merged = df_with_idx.merge(
                    gate_df,
                    on=["symbol", "timestamp"],
                    how="left",
                    suffixes=("", "_gate"),
                )
                merged = merged.sort_values("_temp_idx")
                gate_ok_full = merged["gate_ok"].fillna(False)
                gate_ok_full.index = df.index

    passed_full = df[gate_ok_full.reindex(df.index, fill_value=False)]
    results["full_gate"] = {
        "total_rows": len(df),
        "trade_count": len(passed_full),
        "pass_rate": len(passed_full) / len(df) if len(df) > 0 else 0.0,
        "sharpe": _sharpe(passed_full["ret_mean"]) if len(passed_full) > 0 else 0.0,
        "mean_return": (
            float(passed_full["ret_mean"].mean()) if len(passed_full) > 0 else 0.0
        ),
        "win_rate": (
            float((passed_full["ret_mean"] > 0).mean()) if len(passed_full) > 0 else 0.0
        ),
        "profitable_rate": (
            float((passed_full["ret_mean"] > 0).mean()) if len(passed_full) > 0 else 0.0
        ),
    }

    return results


def generate_exp2_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate Experiment 2 report"""
    md = "# Experiment 2: Gate Regime Filtering\n\n"
    md += "## Question\n\n"
    md += "Are gates failing to exclude non-FR regimes?\n\n"
    md += "---\n\n"

    md += "## Comparison: No Gate vs Regime-Only vs Full Gate\n\n"
    md += "| Configuration | Trade Count | Pass Rate | Sharpe | Mean Return | Win Rate | Profitable Rate |\n"
    md += "|---------------|------------|-----------|--------|-------------|----------|-----------------|\n"

    for config_name in ["no_gate", "regime_only", "full_gate"]:
        if config_name not in results:
            continue
        cfg = results[config_name]
        md += f"| {config_name.replace('_', ' ').title()} | {cfg['trade_count']} | "
        md += f"{cfg.get('pass_rate', 1.0):.1%} | {cfg['sharpe']:.4f} | "
        md += f"{cfg['mean_return']:.6f} | {cfg['win_rate']:.1%} | {cfg['profitable_rate']:.1%} |\n"

    md += "\n## Conclusions\n\n"
    no_gate_sharpe = results.get("no_gate", {}).get("sharpe", 0)
    regime_sharpe = results.get("regime_only", {}).get("sharpe", 0)
    full_sharpe = results.get("full_gate", {}).get("sharpe", 0)

    if no_gate_sharpe < 0:
        md += "- ⚠️ **No-gate baseline has negative Sharpe**, suggesting the problem is not just gate filtering.\n"
    if regime_sharpe > full_sharpe:
        md += f"- ✅ **Regime-only filtering improves Sharpe** ({regime_sharpe:.4f} vs {full_sharpe:.4f}), suggesting preconditions/evidence rules may be too strict.\n"
    elif full_sharpe > regime_sharpe:
        md += f"- ✅ **Full gate improves Sharpe** ({full_sharpe:.4f} vs {regime_sharpe:.4f}), suggesting all rules are beneficial.\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# ============================================================================
# Experiment 3: Threshold Scanning
# ============================================================================


def experiment3_threshold_scanning(
    df: pd.DataFrame,
    archetype_name: str,
    arches: Dict,
    quantiles: Optional[Dict] = None,
    features_store_root: Optional[str] = None,
    features_layer: Optional[str] = None,
    timeframe: Optional[str] = None,
    scan_rules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Experiment 3: Scan threshold space for optimal Sharpe"""
    arch = arches.get(archetype_name)
    if not arch:
        return {}

    rules = list(getattr(arch, "when_then_rules", []) or [])
    if scan_rules:
        rules = [r for r in rules if r.get("id") in scan_rules]

    results = {
        "rule_scans": [],
    }

    # For each rule with quantile thresholds, scan different values
    for rule in rules:
        rule_id = rule.get("id", "")
        when = rule.get("when", {})

        # Find quantile thresholds in when clause
        quantile_params = []
        if isinstance(when, dict):
            for key, value in when.items():
                if isinstance(value, dict):
                    for op, threshold in value.items():
                        if op.startswith("quantile_"):
                            quantile_params.append((key, op, threshold))

        if not quantile_params:
            continue

        # Scan thresholds for this rule
        rule_results = {
            "rule_id": rule_id,
            "phase": rule.get("phase", ""),
            "scans": [],
        }

        # For simplicity, scan one parameter at a time
        for feat_key, op, current_threshold in quantile_params[
            :1
        ]:  # Only scan first param
            test_thresholds = np.linspace(0.0, 1.0, 21)  # 0.0 to 1.0 in 0.05 steps

            for test_thresh in test_thresholds:
                # Create modified rule
                modified_rule = rule.copy()
                modified_when = when.copy()
                if feat_key in modified_when:
                    modified_when[feat_key] = {op: test_thresh}
                modified_rule["when"] = modified_when

                # Create gate config with this single modified rule
                test_gate_cfg = {
                    "when_then_rules": [modified_rule],
                    "default_action": "allow",  # Allow by default to see rule's effect
                }

                # Apply gate
                gate_ok = apply_gate_to_df(
                    df,
                    archetype_name,
                    arches,
                    quantiles,
                    features_store_root,
                    features_layer,
                    timeframe,
                    gate_config_override=test_gate_cfg,
                )

                # Align indices
                if len(gate_ok) != len(df):
                    gate_ok = gate_ok.reindex(df.index, fill_value=False)

                passed = df[gate_ok]
                if len(passed) > 0:
                    rule_results["scans"].append(
                        {
                            "threshold": float(test_thresh),
                            "trade_count": len(passed),
                            "trade_rate": len(passed) / len(df),
                            "sharpe": _sharpe(passed["ret_mean"]),
                            "mean_return": float(passed["ret_mean"].mean()),
                            "win_rate": float((passed["ret_mean"] > 0).mean()),
                        }
                    )

        if rule_results["scans"]:
            results["rule_scans"].append(rule_results)

    return results


def generate_exp3_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate Experiment 3 report"""
    md = "# Experiment 3: Threshold Scanning\n\n"
    md += "## Question\n\n"
    md += "Are thresholds too loose, allowing bad trades?\n\n"
    md += "---\n\n"

    for rule_scan in results.get("rule_scans", []):
        md += f"## Rule: {rule_scan['rule_id']} ({rule_scan['phase']})\n\n"
        md += "| Threshold | Trade Count | Trade Rate | Sharpe | Mean Return | Win Rate |\n"
        md += (
            "|-----------|------------|-----------|--------|-------------|----------|\n"
        )

        scans = sorted(rule_scan["scans"], key=lambda x: x["threshold"])
        for scan in scans:
            md += f"| {scan['threshold']:.2f} | {scan['trade_count']} | {scan['trade_rate']:.1%} | "
            md += f"{scan['sharpe']:.4f} | {scan['mean_return']:.6f} | {scan['win_rate']:.1%} |\n"

        # Find best threshold
        if scans:
            best = max(scans, key=lambda x: x["sharpe"])
            md += f"\n**Best Threshold**: {best['threshold']:.2f} (Sharpe: {best['sharpe']:.4f}, Trade Rate: {best['trade_rate']:.1%})\n\n"

    md += "\n## Conclusions\n\n"
    md += "- Review threshold scans above to identify optimal values for each rule.\n"
    md += "- Consider tightening thresholds if current values allow too many bad trades.\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# ============================================================================
# Experiment 4: Rule Complexity
# ============================================================================


def experiment4_rule_complexity(
    df: pd.DataFrame,
    archetype_name: str,
    arches: Dict,
    quantiles: Optional[Dict] = None,
    features_store_root: Optional[str] = None,
    features_layer: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Experiment 4: Test rule complexity (minimal vs full)"""
    arch = arches.get(archetype_name)
    if not arch:
        return {}

    all_rules = list(getattr(arch, "when_then_rules", []) or [])
    results = {}

    # Minimal: Only core rules (fr_looks_trendy + fr_reversal_evidence)
    minimal_rules = [
        r
        for r in all_rules
        if r.get("id") in ["fr_looks_trendy", "fr_reversal_evidence"]
    ]
    minimal_cfg = {
        "when_then_rules": minimal_rules,
        "default_action": "allow",
    }

    gate_ok_minimal = apply_gate_to_df(
        df,
        archetype_name,
        arches,
        quantiles,
        features_store_root,
        features_layer,
        timeframe,
        gate_config_override=minimal_cfg,
    )

    if len(gate_ok_minimal) != len(df):
        gate_ok_minimal = gate_ok_minimal.reindex(df.index, fill_value=False)

    passed_minimal = df[gate_ok_minimal]
    results["minimal"] = {
        "trade_count": len(passed_minimal),
        "pass_rate": len(passed_minimal) / len(df) if len(df) > 0 else 0.0,
        "sharpe": (
            _sharpe(passed_minimal["ret_mean"]) if len(passed_minimal) > 0 else 0.0
        ),
        "mean_return": (
            float(passed_minimal["ret_mean"].mean()) if len(passed_minimal) > 0 else 0.0
        ),
        "win_rate": (
            float((passed_minimal["ret_mean"] > 0).mean())
            if len(passed_minimal) > 0
            else 0.0
        ),
    }

    # No-reflexivity: Remove safety phase
    no_reflex_rules = [r for r in all_rules if r.get("phase") != "safety"]
    no_reflex_cfg = {
        "when_then_rules": no_reflex_rules,
        "default_action": str(getattr(arch, "default_action", "deny")),
    }

    gate_ok_no_reflex = apply_gate_to_df(
        df,
        archetype_name,
        arches,
        quantiles,
        features_store_root,
        features_layer,
        timeframe,
        gate_config_override=no_reflex_cfg,
    )

    if len(gate_ok_no_reflex) != len(df):
        gate_ok_no_reflex = gate_ok_no_reflex.reindex(df.index, fill_value=False)

    passed_no_reflex = df[gate_ok_no_reflex]
    results["no_reflexivity"] = {
        "trade_count": len(passed_no_reflex),
        "pass_rate": len(passed_no_reflex) / len(df) if len(df) > 0 else 0.0,
        "sharpe": (
            _sharpe(passed_no_reflex["ret_mean"]) if len(passed_no_reflex) > 0 else 0.0
        ),
        "mean_return": (
            float(passed_no_reflex["ret_mean"].mean())
            if len(passed_no_reflex) > 0
            else 0.0
        ),
        "win_rate": (
            float((passed_no_reflex["ret_mean"] > 0).mean())
            if len(passed_no_reflex) > 0
            else 0.0
        ),
    }

    # Extreme-reflexivity: Raise reflexivity threshold to 0.95
    extreme_reflex_rules = []
    for rule in all_rules:
        if (
            rule.get("phase") == "safety"
            and "reflexivity" in rule.get("id", "").lower()
        ):
            modified_rule = rule.copy()
            when = rule.get("when", {}).copy()
            for key, value in when.items():
                if isinstance(value, dict):
                    for op, threshold in value.items():
                        if op.startswith("quantile_"):
                            when[key] = {op: 0.95}
            modified_rule["when"] = when
            extreme_reflex_rules.append(modified_rule)
        else:
            extreme_reflex_rules.append(rule)

    extreme_reflex_cfg = {
        "when_then_rules": extreme_reflex_rules,
        "default_action": str(getattr(arch, "default_action", "deny")),
    }

    gate_ok_extreme = apply_gate_to_df(
        df,
        archetype_name,
        arches,
        quantiles,
        features_store_root,
        features_layer,
        timeframe,
        gate_config_override=extreme_reflex_cfg,
    )

    if len(gate_ok_extreme) != len(df):
        gate_ok_extreme = gate_ok_extreme.reindex(df.index, fill_value=False)

    passed_extreme = df[gate_ok_extreme]
    results["extreme_reflexivity"] = {
        "trade_count": len(passed_extreme),
        "pass_rate": len(passed_extreme) / len(df) if len(df) > 0 else 0.0,
        "sharpe": (
            _sharpe(passed_extreme["ret_mean"]) if len(passed_extreme) > 0 else 0.0
        ),
        "mean_return": (
            float(passed_extreme["ret_mean"].mean()) if len(passed_extreme) > 0 else 0.0
        ),
        "win_rate": (
            float((passed_extreme["ret_mean"] > 0).mean())
            if len(passed_extreme) > 0
            else 0.0
        ),
    }

    # Full gate: Current configuration
    gate_ok_full = apply_gate_to_df(
        df,
        archetype_name,
        arches,
        quantiles,
        features_store_root,
        features_layer,
        timeframe,
    )

    if len(gate_ok_full) != len(df):
        gate_ok_full = gate_ok_full.reindex(df.index, fill_value=False)

    passed_full = df[gate_ok_full]
    results["full"] = {
        "trade_count": len(passed_full),
        "pass_rate": len(passed_full) / len(df) if len(df) > 0 else 0.0,
        "sharpe": _sharpe(passed_full["ret_mean"]) if len(passed_full) > 0 else 0.0,
        "mean_return": (
            float(passed_full["ret_mean"].mean()) if len(passed_full) > 0 else 0.0
        ),
        "win_rate": (
            float((passed_full["ret_mean"] > 0).mean()) if len(passed_full) > 0 else 0.0
        ),
    }

    return results


def generate_exp4_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate Experiment 4 report"""
    md = "# Experiment 4: Rule Complexity\n\n"
    md += "## Question\n\n"
    md += "Are too many rules causing conflicts or over-filtering?\n\n"
    md += "---\n\n"

    md += "## Comparison: Minimal vs No-Reflexivity vs Extreme-Reflexivity vs Full\n\n"
    md += "| Configuration | Trade Count | Pass Rate | Sharpe | Mean Return | Win Rate |\n"
    md += (
        "|---------------|------------|-----------|--------|-------------|----------|\n"
    )

    for config_name in ["minimal", "no_reflexivity", "extreme_reflexivity", "full"]:
        if config_name not in results:
            continue
        cfg = results[config_name]
        md += f"| {config_name.replace('_', ' ').title()} | {cfg['trade_count']} | "
        md += f"{cfg['pass_rate']:.1%} | {cfg['sharpe']:.4f} | "
        md += f"{cfg['mean_return']:.6f} | {cfg['win_rate']:.1%} |\n"

    md += "\n## Conclusions\n\n"
    minimal_sharpe = results.get("minimal", {}).get("sharpe", 0)
    no_reflex_sharpe = results.get("no_reflexivity", {}).get("sharpe", 0)
    extreme_reflex_sharpe = results.get("extreme_reflexivity", {}).get("sharpe", 0)
    full_sharpe = results.get("full", {}).get("sharpe", 0)

    best_config = max(
        [
            ("minimal", minimal_sharpe),
            ("no_reflexivity", no_reflex_sharpe),
            ("extreme_reflexivity", extreme_reflex_sharpe),
            ("full", full_sharpe),
        ],
        key=lambda x: x[1],
    )

    md += f"- **Best Configuration**: {best_config[0].replace('_', ' ').title()} (Sharpe: {best_config[1]:.4f})\n"

    if minimal_sharpe > full_sharpe:
        md += "- ✅ **Minimal gate performs better**, suggesting full gate may be over-filtering.\n"
    if no_reflex_sharpe > full_sharpe:
        md += "- ✅ **No-reflexivity performs better**, suggesting reflexivity rules may be too strict.\n"
    if extreme_reflex_sharpe > full_sharpe:
        md += "- ✅ **Extreme-reflexivity performs better**, suggesting current reflexivity thresholds are too low.\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FR Sharpe Root Cause Diagnostic Script"
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
        help="FeatureStore layer",
    )
    parser.add_argument(
        "--timeframe",
        default="4h",
        help="Timeframe for FeatureStore",
    )
    parser.add_argument(
        "--archetype",
        default="FailureReversionFR",
        help="Archetype to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fr_sharpe_diagnosis",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--experiments",
        default="1,2,3,4",
        help="Comma-separated list of experiments to run (1,2,3,4,5)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Loading data...")
    df = pd.read_parquet(args.logs)
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows")

    print("📊 Loading archetype config...")
    arches = load_execution_archetypes_registry(args.execution_archetypes)

    print("📊 Loading evidence quantiles...")
    quantiles = None
    if args.evidence_quantiles and Path(args.evidence_quantiles).exists():
        quantiles = load_evidence_quantiles(args.evidence_quantiles)
        print("✅ Evidence quantiles loaded")
    else:
        print("⚠️  No evidence quantiles provided")

    experiments_to_run = [int(x.strip()) for x in args.experiments.split(",")]

    # Experiment 1: Period Analysis
    if 1 in experiments_to_run:
        print("\n🔍 Experiment 1: Period Analysis")
        exp1_results = experiment1_period_analysis(df)
        generate_exp1_report(
            exp1_results, output_dir / "experiment1_period_analysis.md"
        )
        with open(output_dir / "experiment1_period_analysis.json", "w") as f:
            json.dump(exp1_results, f, indent=2, default=str)
        print("✅ Experiment 1 complete")

    # Experiment 2: Gate Regime Filtering
    if 2 in experiments_to_run:
        print("\n🔍 Experiment 2: Gate Regime Filtering")
        exp2_results = experiment2_gate_regime_filtering(
            df,
            args.archetype,
            arches,
            quantiles,
            args.features_store_root,
            args.features_layer,
            args.timeframe,
        )
        generate_exp2_report(
            exp2_results, output_dir / "experiment2_gate_regime_filtering.md"
        )
        with open(output_dir / "experiment2_gate_regime_filtering.json", "w") as f:
            json.dump(exp2_results, f, indent=2, default=str)
        print("✅ Experiment 2 complete")

    # Experiment 3: Threshold Scanning
    if 3 in experiments_to_run:
        print("\n🔍 Experiment 3: Threshold Scanning")
        exp3_results = experiment3_threshold_scanning(
            df,
            args.archetype,
            arches,
            quantiles,
            args.features_store_root,
            args.features_layer,
            args.timeframe,
        )
        generate_exp3_report(
            exp3_results, output_dir / "experiment3_threshold_scanning.md"
        )
        with open(output_dir / "experiment3_threshold_scanning.json", "w") as f:
            json.dump(exp3_results, f, indent=2, default=str)
        print("✅ Experiment 3 complete")

    # Experiment 4: Rule Complexity
    if 4 in experiments_to_run:
        print("\n🔍 Experiment 4: Rule Complexity")
        exp4_results = experiment4_rule_complexity(
            df,
            args.archetype,
            arches,
            quantiles,
            args.features_store_root,
            args.features_layer,
            args.timeframe,
        )
        generate_exp4_report(
            exp4_results, output_dir / "experiment4_rule_complexity.md"
        )
        with open(output_dir / "experiment4_rule_complexity.json", "w") as f:
            json.dump(exp4_results, f, indent=2, default=str)
        print("✅ Experiment 4 complete")

    # Generate summary report
    print("\n📝 Generating summary report...")
    summary_md = "# FR Sharpe Root Cause Diagnosis - Summary\n\n"
    summary_md += "This report summarizes findings from 5 diagnostic experiments.\n\n"
    summary_md += "## Experiments\n\n"
    summary_md += "1. **Period Analysis**: Analyze FR performance by time period\n"
    summary_md += "2. **Gate Regime Filtering**: Test gate effectiveness\n"
    summary_md += "3. **Threshold Scanning**: Find optimal thresholds\n"
    summary_md += "4. **Rule Complexity**: Test minimal vs full gate configurations\n"
    summary_md += "5. **Label Profitability**: Compare label vs gate-filtered profitability (see separate report)\n\n"
    summary_md += "## Reports\n\n"
    summary_md += "- Experiment 1: `experiment1_period_analysis.md`\n"
    summary_md += "- Experiment 2: `experiment2_gate_regime_filtering.md`\n"
    summary_md += "- Experiment 3: `experiment3_threshold_scanning.md`\n"
    summary_md += "- Experiment 4: `experiment4_rule_complexity.md`\n"
    summary_md += "- Experiment 5: `../fr_label_vs_gate_profitability_report.md`\n"

    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("✅ All experiments complete!")
    print(f"📁 Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
