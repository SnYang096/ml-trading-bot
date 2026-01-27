#!/usr/bin/env python3
"""
Label-based plateau scan for FBF rules (dual-window metrics).
Scans candidate thresholds per rule and evaluates stability/consistency.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_fr_label_vs_gate_profitability import apply_gate_to_df
from src.time_series_model.core.constitution.execution_evidence import (
    load_evidence_quantiles,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


DEFAULT_RULE_RANGES = {
    "fbf_false_breakout": np.linspace(0.1, 0.5, 9).tolist(),
    "fbf_volume_profile_rejection": np.linspace(0.2, 0.6, 9).tolist(),
    "fbf_wick_scene_long_wick": np.linspace(0.4, 0.8, 9).tolist(),
    "fbf_fp_aggressive_but_stuck:vpin": np.linspace(0.4, 0.8, 9).tolist(),
    "fbf_fp_aggressive_but_stuck:cvd": np.linspace(0.2, 0.6, 9).tolist(),
    "fbf_vpin_spike": np.linspace(0.5, 0.9, 9).tolist(),
}


def _gate_metrics(df: pd.DataFrame, gate_ok: pd.Series) -> Dict[str, float]:
    gate_ok = gate_ok.reindex(df.index, fill_value=False)
    is_fbf = df["fbf_label"] == "FBF"
    if gate_ok.sum() == 0:
        return {
            "trade_rate": 0.0,
            "gate_precision": 0.0,
            "gate_recall": 0.0,
            "noise_rate": 0.0,
        }
    trade_rate = float(gate_ok.mean())
    precision = float((gate_ok & is_fbf).sum() / gate_ok.sum())
    recall = float((gate_ok & is_fbf).sum() / max(is_fbf.sum(), 1))
    noise_rate = float(
        (gate_ok & (df["fbf_label"] == "unlabeled")).sum() / gate_ok.sum()
    )
    return {
        "trade_rate": trade_rate,
        "gate_precision": precision,
        "gate_recall": recall,
        "noise_rate": noise_rate,
    }


def _label_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "execution_fbf_ratio": float((df["execution_label"] == "FBF").mean()),
        "semantic_fbf_ratio": float((df["semantic_label"] == "FBF").mean()),
        "label_consistency_ratio": float(df["label_consistency"].mean()),
    }


def _stability_score(
    groups: Dict[str, Dict[str, float]], holding_median_drift: float
) -> float:
    exec_ratios = [v["execution_fbf_ratio"] for v in groups.values()]
    sem_ratios = [v["semantic_fbf_ratio"] for v in groups.values()]
    consistency = [v["label_consistency_ratio"] for v in groups.values()]
    trade_rates = [v["trade_rate"] for v in groups.values()]
    precisions = [v["gate_precision"] for v in groups.values()]
    recalls = [v["gate_recall"] for v in groups.values()]
    noise = [v["noise_rate"] for v in groups.values()]

    def _safe_std(vals: List[float]) -> float:
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    label_stability = 1 - _safe_std(exec_ratios + sem_ratios)
    consistency_avg = float(np.mean(consistency)) if consistency else 0.0
    gate_accuracy = (
        float(np.mean(precisions) * 0.4 + np.mean(recalls) * 0.4 - np.mean(noise) * 0.2)
        if precisions
        else 0.0
    )
    coverage_stability = 1 / (1 + _safe_std(trade_rates)) if trade_rates else 0.0

    # Penalize holding time drift (not optimized directly)
    holding_penalty = max(0.0, holding_median_drift)

    score = (
        0.2 * float(np.mean(exec_ratios) if exec_ratios else 0.0)
        + 0.2 * float(np.mean(sem_ratios) if sem_ratios else 0.0)
        + 0.15 * consistency_avg
        + 0.15 * label_stability
        + 0.2 * gate_accuracy
        + 0.05 * coverage_stability
        - 0.05 * holding_penalty
    )
    return float(score)


def _update_rule_threshold(rule: Dict, rule_id: str, new_value: float) -> Dict:
    rule = deepcopy(rule)
    if rule["id"] == rule_id:
        when = rule.get("when", {})
        for k, v in when.items():
            if isinstance(v, dict) and "quantile_gt" in v:
                v["quantile_gt"] = float(new_value)
            elif isinstance(v, dict) and "quantile_lt" in v:
                v["quantile_lt"] = float(new_value)
        rule["when"] = when
    return rule


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label-based plateau scan for FBF rules"
    )
    parser.add_argument(
        "--labels", required=True, help="labeled parquet from generate_fbf_labels.py"
    )
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument(
        "--archetype", default="FailedBreakoutFade", help="archetype name"
    )
    parser.add_argument(
        "--rules", default=None, help="comma-separated rule ids to scan"
    )
    parser.add_argument("--quantiles", default=None, help="evidence_quantiles.json")
    parser.add_argument("--feature-store-root", default=None, help="FeatureStore root")
    parser.add_argument(
        "--feature-store-layer", default=None, help="FeatureStore layer"
    )
    parser.add_argument("--timeframe", default=None, help="timeframe for FeatureStore")
    parser.add_argument("--grid-json", default=None, help="override rule grid JSON")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument(
        "--regime-cols", default=None, help="comma-separated regime columns (optional)"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    for col in ["execution_label", "semantic_label", "fbf_label", "label_consistency"]:
        if col not in df.columns:
            raise KeyError(f"missing label column: {col}")

    arches = load_execution_archetypes_registry(
        path=str(PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml")
    )
    arch = arches.get(args.archetype)
    if not arch:
        raise KeyError(f"archetype not found: {args.archetype}")

    gate_cfg = {
        "when_then_rules": list(getattr(arch, "when_then_rules", []) or []),
        "default_action": str(getattr(arch, "default_action", "deny")),
    }

    quantiles = load_evidence_quantiles(args.quantiles) if args.quantiles else None
    rule_ids = [r["id"] for r in gate_cfg["when_then_rules"]]

    grid = DEFAULT_RULE_RANGES
    if args.grid_json:
        with open(args.grid_json, "r") as f:
            grid = json.load(f)

    target_rules = rule_ids
    if args.rules:
        target_rules = [r.strip() for r in args.rules.split(",") if r.strip()]

    regime_cols = (
        [c.strip() for c in args.regime_cols.split(",")] if args.regime_cols else []
    )

    # baseline holding median for drift penalty
    base_gate_ok = apply_gate_to_df(
        df=df.copy(),
        archetype_name=args.archetype,
        arches=arches,
        quantiles=quantiles,
        features_store_root=args.feature_store_root,
        features_layer=args.feature_store_layer,
        timeframe=args.timeframe,
    )
    base_holding_median = (
        float(df.loc[base_gate_ok, "holding_bars"].median())
        if base_gate_ok.any()
        else 0.0
    )

    results = {"archetype": args.archetype, "rules": {}}

    for rule_id in target_rules:
        if rule_id not in rule_ids:
            continue
        candidates = grid.get(rule_id, grid.get(f"{rule_id}:vpin")) or []
        rule_results = []
        for cand in candidates:
            updated_rules = []
            for rule in gate_cfg["when_then_rules"]:
                if rule["id"] == rule_id:
                    updated_rules.append(
                        _update_rule_threshold(rule, rule_id, float(cand))
                    )
                else:
                    updated_rules.append(deepcopy(rule))

            # Create updated gate config dict (not modifying frozen dataclass)
            gate_cfg_updated = {
                "when_then_rules": updated_rules,
                "default_action": gate_cfg["default_action"],
            }
            # Create temporary archetype dict for apply_gate_to_df
            arches_tmp = deepcopy(arches)
            from dataclasses import replace

            arch_orig = arches_tmp[args.archetype]
            arch_tmp = replace(arch_orig, when_then_rules=updated_rules)
            arches_tmp[args.archetype] = arch_tmp
            gate_ok = apply_gate_to_df(
                df=df.copy(),
                archetype_name=args.archetype,
                arches=arches_tmp,
                quantiles=quantiles,
                features_store_root=args.feature_store_root,
                features_layer=args.feature_store_layer,
                timeframe=args.timeframe,
            )

            groups = {}
            if regime_cols:
                for key, sub in df.groupby(regime_cols):
                    if len(sub) < args.min_samples:
                        continue
                    metrics = {}
                    metrics.update(_label_metrics(sub))
                    metrics.update(_gate_metrics(sub, gate_ok.loc[sub.index]))
                    groups[str(key)] = metrics
            else:
                metrics = {}
                metrics.update(_label_metrics(df))
                metrics.update(_gate_metrics(df, gate_ok))
                groups["overall"] = metrics

            holding_median = (
                float(df.loc[gate_ok, "holding_bars"].median())
                if gate_ok.any()
                else 0.0
            )
            holding_drift = (
                abs(holding_median - base_holding_median) / (base_holding_median + 1e-9)
                if base_holding_median > 0
                else 0.0
            )
            score = _stability_score(groups, holding_drift)

            rule_results.append(
                {
                    "rule_id": rule_id,
                    "candidate": float(cand),
                    "score": score,
                    "holding_median": holding_median,
                    "holding_drift": holding_drift,
                    "groups": groups,
                }
            )

        if rule_results:
            best = max(rule_results, key=lambda r: r["score"])
            results["rules"][rule_id] = {"best": best, "candidates": rule_results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
