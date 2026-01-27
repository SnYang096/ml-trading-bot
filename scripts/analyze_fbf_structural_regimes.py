#!/usr/bin/env python3
"""
Analyze FBF labels across structural regimes.

Reports:
- execution/semantic label ratios
- label consistency ratios
- gate precision/recall/noise_rate (optional)
- execution metrics (holding_bars, exit_reason)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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


def _bucket_series(
    values: pd.Series, bins: List[float], labels: List[str]
) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True)


def _compute_gate_metrics(df: pd.DataFrame, gate_ok: pd.Series) -> Dict[str, float]:
    if gate_ok is None or gate_ok.empty:
        return {}
    gate_ok = gate_ok.reindex(df.index, fill_value=False)
    is_fbf = df["fbf_label"] == "FBF"
    if gate_ok.sum() == 0:
        return {"gate_precision": 0.0, "gate_recall": 0.0, "noise_rate": 0.0}
    precision = float((gate_ok & is_fbf).sum() / gate_ok.sum())
    recall = float((gate_ok & is_fbf).sum() / max(is_fbf.sum(), 1))
    noise_rate = float(
        (gate_ok & (df["fbf_label"] == "unlabeled")).sum() / gate_ok.sum()
    )
    return {
        "gate_precision": precision,
        "gate_recall": recall,
        "noise_rate": noise_rate,
    }


def _summarize_group(
    df: pd.DataFrame, gate_ok: Optional[pd.Series]
) -> Dict[str, float]:
    exec_ratio = float((df["execution_label"] == "FBF").mean())
    sem_ratio = float((df["semantic_label"] == "FBF").mean())
    consistency = float(df["label_consistency"].mean())
    holding_median = float(df["holding_bars"].median()) if "holding_bars" in df else 0.0
    exit_reason_counts = (
        df.get("exit_reason", pd.Series(dtype=str)).value_counts().to_dict()
    )
    metrics = {
        "count": int(len(df)),
        "execution_fbf_ratio": exec_ratio,
        "semantic_fbf_ratio": sem_ratio,
        "label_consistency_ratio": consistency,
        "median_holding_bars": holding_median,
        "exit_reason_counts": exit_reason_counts,
    }
    if gate_ok is not None:
        metrics.update(_compute_gate_metrics(df, gate_ok.loc[df.index]))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze FBF labels by structural regimes"
    )
    parser.add_argument(
        "--labels", required=True, help="labeled parquet from generate_fbf_labels.py"
    )
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument(
        "--archetype", default="FailedBreakoutFade", help="archetype name"
    )
    parser.add_argument("--quantiles", default=None, help="evidence_quantiles.json")
    parser.add_argument("--feature-store-root", default=None, help="FeatureStore root")
    parser.add_argument(
        "--feature-store-layer", default=None, help="FeatureStore layer"
    )
    parser.add_argument("--timeframe", default=None, help="timeframe for FeatureStore")
    parser.add_argument("--prior-trend-col", default="path_efficiency_pct")
    parser.add_argument("--volatility-col", default="atr_percentile")
    parser.add_argument("--sr-distance-col", default="sr_distance_normalized")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    for col in ["execution_label", "semantic_label", "fbf_label", "label_consistency"]:
        if col not in df.columns:
            raise KeyError(f"missing label column: {col}")

    gate_ok = None
    if args.quantiles:
        quantiles = load_evidence_quantiles(args.quantiles)
        arches = load_execution_archetypes_registry(
            path=str(PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml")
        )
        gate_ok = apply_gate_to_df(
            df=df.copy(),
            archetype_name=args.archetype,
            arches=arches,
            quantiles=quantiles,
            features_store_root=args.feature_store_root,
            features_layer=args.feature_store_layer,
            timeframe=args.timeframe,
        )

    results = {"overall": _summarize_group(df, gate_ok)}

    # Regime buckets (separate dimensions to avoid sparsity)
    if args.prior_trend_col in df.columns:
        trend_bucket = _bucket_series(
            df[args.prior_trend_col],
            bins=[-np.inf, 0.4, 0.6, np.inf],
            labels=["weak", "mid", "strong"],
        )
        results["prior_trend"] = {}
        for label, sub in df.groupby(trend_bucket):
            if label is None or len(sub) < args.min_samples:
                continue
            results["prior_trend"][str(label)] = _summarize_group(sub, gate_ok)

    if args.volatility_col in df.columns:
        vol_bucket = _bucket_series(
            df[args.volatility_col],
            bins=[-np.inf, 0.3, 0.7, np.inf],
            labels=["low", "mid", "high"],
        )
        results["volatility"] = {}
        for label, sub in df.groupby(vol_bucket):
            if label is None or len(sub) < args.min_samples:
                continue
            results["volatility"][str(label)] = _summarize_group(sub, gate_ok)

    if args.sr_distance_col in df.columns:
        sr_bucket = _bucket_series(
            df[args.sr_distance_col],
            bins=[-np.inf, 0.2, 0.5, np.inf],
            labels=["near", "mid", "far"],
        )
        results["sr_distance"] = {}
        for label, sub in df.groupby(sr_bucket):
            if label is None or len(sub) < args.min_samples:
                continue
            results["sr_distance"][str(label)] = _summarize_group(sub, gate_ok)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
