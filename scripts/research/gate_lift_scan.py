"""Single-feature Gate lift plateau (deny semantics) for ``mlbot research plateau --kpi lift``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.research.gate_when import (
    is_feature_allowed_for_gate_deny,
    load_allowed_gate_deny_features,
    resolve_gate_deny_operator,
)
from src.research.labels import derive_is_good_from_forward_rr
from src.research.stat_kernels.gate_lift import scan_thresholds_for_lift
from src.research.stat_kernels.plateau import find_stable_lift_plateau
from src.research.stat_kernels.robustness import (
    UnifiedOptimizationConfig,
    compute_robustness_score,
)


def _resolve_label_col(df: pd.DataFrame, label_col: str) -> str:
    if label_col in df.columns:
        if df[label_col].dtype == bool:
            df["_is_good"] = df[label_col].astype(int)
            return "_is_good"
        return label_col
    derive_is_good_from_forward_rr(df, label_col=label_col)
    return label_col


def _threshold_range(
    df: pd.DataFrame,
    feature_col: str,
    grid: Optional[List[float]],
) -> Tuple[Tuple[float, float], float]:
    if grid and len(grid) >= 2:
        low, high = float(min(grid)), float(max(grid))
        step = (high - low) / max(len(grid) - 1, 1)
        return (low, high), max(step, 1e-6)
    if "_pct" in feature_col or "quantile" in feature_col:
        return (0.05, 0.95), 0.05
    q_low = float(df[feature_col].quantile(0.05))
    q_high = float(df[feature_col].quantile(0.95))
    step = max((q_high - q_low) / 20, 1e-6)
    return (q_low, q_high), step


def gate_lift_plateau_payload(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,
    *,
    base_mask: pd.Series,
    label_col: str = "is_good",
    grid: Optional[List[float]] = None,
    config: Optional[UnifiedOptimizationConfig] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan deny-threshold lift grid and detect stable plateau."""
    cfg = config or UnifiedOptimizationConfig()
    work = df.loc[base_mask].copy()
    if feature_col not in work.columns:
        raise ValueError(f"feature '{feature_col}' missing from parquet")

    if strategy:
        patterns = load_allowed_gate_deny_features(strategy)
        if patterns and not is_feature_allowed_for_gate_deny(feature_col, patterns):
            raise ValueError(
                f"feature '{feature_col}' not in features_gate.yaml whitelist "
                f"for strategy {strategy!r}"
            )

    use_label = _resolve_label_col(work, label_col)
    deny_op = resolve_gate_deny_operator(operator)
    if deny_op not in ("lt", "le", "gt", "ge"):
        raise ValueError(
            f"unsupported operator for gate lift: {operator!r} (deny={deny_op!r})"
        )

    th_range, step = _threshold_range(work, feature_col, grid)
    results = scan_thresholds_for_lift(
        work, feature_col, deny_op, th_range, step, use_label
    )

    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "threshold": r["threshold"],
                "lift": r.get("lift"),
                "lift_valid": r.get("lift_valid", False),
                "pass_rate_all": r.get("pass_rate_all"),
                "pass_rate_good": r.get("pass_rate_good"),
                "pass_rate_bad": r.get("pass_rate_bad"),
                "n_passed": r.get("n_passed"),
            }
        )

    stable = find_stable_lift_plateau(results, cfg, actual_step=step)
    payload: Dict[str, Any] = {
        "kpi": "lift",
        "feature": feature_col,
        "operator": deny_op,
        "input_operator": operator,
        "deny_operator": deny_op,
        "rows": rows,
        "threshold_range": {"low": th_range[0], "high": th_range[1], "step": step},
    }

    if stable is None:
        payload.update(
            {
                "is_plateau": False,
                "reason": "no stable lift plateau",
                "recommended": None,
            }
        )
        return payload

    rec = stable["recommended_threshold"]
    rob = compute_robustness_score(
        work, feature_col, deny_op, float(rec), label_col=use_label, config=cfg
    )
    payload.update(
        {
            "is_plateau": True,
            "start_threshold": stable["plateau_start"],
            "end_threshold": stable["plateau_end"],
            "recommended": stable["plateau_mid"],
            "recommended_threshold": rec,
            "plateau_mid": stable["plateau_mid"],
            "lift_mean": stable["lift_mean"],
            "lift_at_mid": stable["lift_at_mid"],
            "pass_rate_at_mid": stable["pass_rate_at_mid"],
            "robustness_score": rob.to_dict(),
            "confidence": "stable_plateau",
        }
    )
    return payload


def format_gate_lift_report(payload: Dict[str, Any]) -> str:
    feature = payload["feature"]
    md = [f"# gate_lift_plateau · {feature} deny({payload.get('deny_operator')})"]
    md.append("")
    md.append("| threshold | lift | pass_rate |")
    md.append("|---:|---:|---:|")
    for row in payload.get("rows", []):
        lift = row.get("lift")
        lift_s = f"{lift:.4f}" if lift is not None and np.isfinite(lift) else "nan"
        md.append(
            f"| {row['threshold']:.4g} | {lift_s} | {row.get('pass_rate_all', 0):.4f} |"
        )
    md.append("")
    if payload.get("is_plateau"):
        md.append(
            f"**Plateau**: [{payload.get('start_threshold')}, {payload.get('end_threshold')}] "
            f"recommended={payload.get('recommended')} "
            f"robustness={payload.get('robustness_score', {}).get('overall_score')}"
        )
    else:
        md.append(f"**No plateau**: {payload.get('reason', 'n/a')}")
    return "\n".join(md)
