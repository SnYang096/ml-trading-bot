#!/usr/bin/env python3
"""
Diagnose gate loosening based on labels.

Goal:
- Identify which rules veto FBF labels (precision/recall diagnosis).
- Recommend looser thresholds using label distribution.

This script is archetype-agnostic and reusable for other archetypes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_fr_label_vs_gate_profitability import (  # noqa: E402
    _read_feature_store_range,
)
from src.time_series_model.core.constitution.execution_evidence import (  # noqa: E402
    load_evidence_quantiles,
)
from src.time_series_model.live.tree_gate import (  # noqa: E402
    _eval_when_clause,
    apply_when_then_rules,
)


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=False)
            if "timestamp" not in df.columns:
                df["timestamp"] = df.index
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["timestamp"] = df.index
    elif getattr(df.index, "name", None) == "timestamp":
        # Avoid ambiguity when timestamp exists as both index and column.
        df = df.reset_index(drop=True)
    return df


def _get_archetype_cfg(cfg: Dict[str, Any], archetype: str) -> Dict[str, Any]:
    arches = cfg.get("archetypes") if isinstance(cfg, dict) else None
    if arches is None:
        arches = cfg
    if not isinstance(arches, dict) or archetype not in arches:
        raise KeyError(f"archetype not found: {archetype}")
    return arches[archetype]


def _select_quantiles(
    quantiles_raw: Optional[Dict[str, Any]], symbol: str
) -> Optional[Dict[str, Any]]:
    if not isinstance(quantiles_raw, dict):
        return None
    sym_q = quantiles_raw.get(str(symbol))
    return sym_q if isinstance(sym_q, dict) else quantiles_raw


def _extract_leaf_conditions(when: Any) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    if when is None:
        return leaves
    if isinstance(when, list):
        for item in when:
            leaves.extend(_extract_leaf_conditions(item))
        return leaves
    if not isinstance(when, dict):
        return leaves

    if "not" in when:
        return _extract_leaf_conditions(when.get("not"))
    if "all_of" in when:
        for item in when.get("all_of") or []:
            leaves.extend(_extract_leaf_conditions(item))
        return leaves
    if "any_of" in when:
        for item in when.get("any_of") or []:
            leaves.extend(_extract_leaf_conditions(item))
        return leaves
    if "key" in when and "op" in when:
        leaves.append(
            {
                "key": str(when.get("key") or ""),
                "op": str(when.get("op") or ""),
                "value": when.get("value"),
            }
        )
        return leaves
    if "any_key_contains" in when:
        leaves.append(
            {"key": "", "op": "any_key_contains", "value": when.get("any_key_contains")}
        )
        return leaves
    if len(when) == 1:
        key = next(iter(when.keys()))
        cond = when.get(key) or {}
        if isinstance(cond, dict) and len(cond) == 1:
            op = next(iter(cond.keys()))
            val = cond.get(op)
            leaves.append({"key": str(key), "op": str(op), "value": val})
    return leaves


def _parse_quantile_map(qmap: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[float, float]] = []
    for k, v in (qmap or {}).items():
        try:
            q = (
                float(str(k).replace("q", "")) / 100.0
                if str(k).startswith("q")
                else float(k)
            )
            pairs.append((q, float(v)))
        except Exception:
            continue
    if not pairs:
        return np.array([]), np.array([])
    qs, vals = zip(*pairs)
    qs = np.array(qs, dtype=float)
    vals = np.array(vals, dtype=float)
    order = np.argsort(vals)
    return qs[order], vals[order]


def _quantile_rank(values: np.ndarray, qmap: Dict[str, Any]) -> np.ndarray:
    qs, vals = _parse_quantile_map(qmap)
    if len(qs) == 0 or len(vals) == 0:
        return np.full_like(values, np.nan, dtype=float)
    return np.interp(values, vals, qs, left=qs[0], right=qs[-1])


def _series_quantile_rank(
    df: pd.DataFrame, key: str, quantiles_raw: Optional[Dict[str, Any]]
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    if "symbol" not in df.columns:
        return out
    for sym, sub in df.groupby("symbol"):
        sym_q = _select_quantiles(quantiles_raw, str(sym))
        if not sym_q or key not in sym_q:
            continue
        values = pd.to_numeric(sub.get(key), errors="coerce").astype(float).values
        out.loc[sub.index] = _quantile_rank(values, sym_q.get(key) or {})
    return out


def _recommend_threshold(
    *,
    action: str,
    op: str,
    fbf_values: np.ndarray,
    target_pass: float,
    target_veto: float,
) -> Optional[float]:
    if fbf_values.size == 0:
        return None
    if action == "require":
        if op in ("quantile_gt", "quantile_gte", "value_gt", "value_gte"):
            q = np.quantile(fbf_values, max(0.0, min(1.0, 1 - target_pass)))
        else:
            q = np.quantile(fbf_values, max(0.0, min(1.0, target_pass)))
    elif action == "deny":
        if op in ("quantile_gt", "quantile_gte", "value_gt", "value_gte"):
            q = np.quantile(fbf_values, max(0.0, min(1.0, 1 - target_veto)))
        else:
            q = np.quantile(fbf_values, max(0.0, min(1.0, target_veto)))
    else:
        return None
    return float(q)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose gate loosening using labels")
    parser.add_argument(
        "--labels", required=True, help="labeled parquet from generate_fbf_labels.py"
    )
    parser.add_argument(
        "--config", required=True, help="execution_archetypes.yaml path"
    )
    parser.add_argument("--archetype", required=True, help="archetype name")
    parser.add_argument(
        "--quantiles", default=None, help="evidence_quantiles.json path"
    )
    parser.add_argument("--feature-store-root", default=None, help="FeatureStore root")
    parser.add_argument(
        "--feature-store-layer", default=None, help="FeatureStore layer"
    )
    parser.add_argument("--timeframe", default=None, help="timeframe for FeatureStore")
    parser.add_argument(
        "--label-col", default="fbf_label", help="label column for target"
    )
    parser.add_argument("--label-value", default="FBF", help="label value for target")
    parser.add_argument(
        "--target-fbf-pass", type=float, default=0.6, help="target pass rate on labels"
    )
    parser.add_argument(
        "--target-fbf-veto", type=float, default=0.1, help="target veto rate on labels"
    )
    parser.add_argument(
        "--min-samples", type=int, default=200, help="min samples to suggest threshold"
    )
    parser.add_argument("--out", required=True, help="output json path")
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    df = _ensure_timestamp(df)

    if args.feature_store_root and args.feature_store_layer and args.timeframe:
        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
        start = df["timestamp"].min().isoformat() if "timestamp" in df.columns else None
        end = df["timestamp"].max().isoformat() if "timestamp" in df.columns else None
        features_df = _read_feature_store_range(
            features_store_root=args.feature_store_root,
            layer=args.feature_store_layer,
            symbols=symbols,
            timeframe=args.timeframe,
            start=start,
            end=end,
        )
        if not features_df.empty:
            features_df = _ensure_timestamp(features_df)
            merge_cols = ["symbol", "timestamp"]
            df = df.merge(features_df, on=merge_cols, how="left")

    quantiles_raw = load_evidence_quantiles(args.quantiles)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    arch_cfg = _get_archetype_cfg(cfg, args.archetype)
    when_then_rules = list(arch_cfg.get("when_then_rules") or [])
    default_action = str(arch_cfg.get("default_action") or "deny").lower()

    label_mask = df[args.label_col] == args.label_value
    near_miss_mask = (df.get("semantic_label") == args.label_value) & (
        df.get("execution_label") != args.label_value
    )

    # Gate-level precision/recall
    gate_ok = []
    for _, row in df.iterrows():
        quantiles = _select_quantiles(quantiles_raw, str(row.get("symbol")))
        ok, _ = apply_when_then_rules(
            when_then_rules=when_then_rules,
            features=row.to_dict(),
            quantiles=quantiles,
            default_action=default_action,
        )
        gate_ok.append(bool(ok))
    gate_ok = pd.Series(gate_ok, index=df.index)
    gate_precision = float((gate_ok & label_mask).sum() / max(gate_ok.sum(), 1))
    gate_recall = float((gate_ok & label_mask).sum() / max(label_mask.sum(), 1))

    results: Dict[str, Any] = {
        "archetype": args.archetype,
        "rows": int(len(df)),
        "label_col": args.label_col,
        "label_value": args.label_value,
        "label_ratio": float(label_mask.mean()) if len(df) > 0 else 0.0,
        "near_miss_ratio": float(near_miss_mask.mean()) if len(df) > 0 else 0.0,
        "gate_precision": gate_precision,
        "gate_recall": gate_recall,
        "rules": {},
    }

    for rule in when_then_rules:
        rule_id = str(rule.get("id") or rule.get("name") or "")
        if not rule_id:
            continue
        phase = str(rule.get("phase") or "")
        action = str(rule.get("then", {}).get("action") or "").lower()
        reason = str(rule.get("reason") or "")
        when = rule.get("when")

        matched = []
        for _, row in df.iterrows():
            quantiles = _select_quantiles(quantiles_raw, str(row.get("symbol")))
            matched.append(
                bool(
                    _eval_when_clause(when, features=row.to_dict(), quantiles=quantiles)
                )
            )
        matched = pd.Series(matched, index=df.index)

        if action == "require":
            fbf_veto_rate = (
                float((~matched & label_mask).mean()) if label_mask.any() else 0.0
            )
        elif action == "deny":
            fbf_veto_rate = (
                float((matched & label_mask).mean()) if label_mask.any() else 0.0
            )
        else:
            fbf_veto_rate = float("nan")

        rule_info: Dict[str, Any] = {
            "phase": phase,
            "action": action,
            "reason": reason,
            "matched_rate_all": float(matched.mean()) if len(matched) else 0.0,
            "matched_rate_label": (
                float(matched[label_mask].mean()) if label_mask.any() else 0.0
            ),
            "matched_rate_near_miss": (
                float(matched[near_miss_mask].mean()) if near_miss_mask.any() else 0.0
            ),
            "fbf_veto_rate": fbf_veto_rate,
            "leaf_suggestions": [],
        }

        for leaf in _extract_leaf_conditions(when):
            key = leaf.get("key") or ""
            op = leaf.get("op") or ""
            current_val = leaf.get("value")
            if not key:
                continue
            series = pd.to_numeric(df.get(key), errors="coerce")
            if op.startswith("quantile_"):
                q_series = _series_quantile_rank(df, key, quantiles_raw)
                fbf_vals = q_series[label_mask].dropna().values
                recommended = _recommend_threshold(
                    action=action,
                    op=op,
                    fbf_values=fbf_vals,
                    target_pass=args.target_fbf_pass,
                    target_veto=args.target_fbf_veto,
                )
            else:
                fbf_vals = series[label_mask].dropna().values
                recommended = _recommend_threshold(
                    action=action,
                    op=op,
                    fbf_values=fbf_vals,
                    target_pass=args.target_fbf_pass,
                    target_veto=args.target_fbf_veto,
                )

            sample_size = int(np.isfinite(fbf_vals).sum()) if fbf_vals.size else 0
            if sample_size < args.min_samples:
                recommended = None

            rule_info["leaf_suggestions"].append(
                {
                    "key": key,
                    "op": op,
                    "current": current_val,
                    "recommended": recommended,
                    "sample_size": sample_size,
                    "note": "recommendation is approximate for composite rules",
                }
            )

        results["rules"][rule_id] = rule_info

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
