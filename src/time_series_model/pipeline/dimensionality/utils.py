"""Utility functions for dimensionality comparison."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


def _slugify(value: str, default: str = "unknown") -> str:
    """Create a filesystem-friendly slug."""
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    # Replace commas with hyphens first to keep multi-symbol ordering visible
    value = value.replace(",", "-")
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "-", value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug or default


def _get_primary_metric(perf: Dict) -> Tuple[str, Optional[float]]:
    """Return the primary evaluation metric name/value for a stage."""
    if not perf:
        return "", None

    financial = perf.get("financial_metrics") or {}
    win_rate = financial.get("win_rate")
    if win_rate is not None:
        return "win_rate", float(win_rate)

    classification = perf.get("classification_metrics") or {}
    for key in ("f1_macro", "f1_weighted", "accuracy"):
        val = classification.get(key)
        if val is not None:
            return key, float(val)

    return "r2", perf.get("r2")


def _derive_feature_insights(stage_baseline: Dict,
                             stage_candidate: Dict) -> Dict:
    """Summarise whether representative features improve over baseline."""
    metric_name_base, metric_base = _get_primary_metric(stage_baseline)
    metric_name_cand, metric_cand = _get_primary_metric(stage_candidate)

    metric_name = metric_name_cand or metric_name_base or "r2"
    delta = None
    if metric_base is not None and metric_cand is not None:
        delta = float(metric_cand) - float(metric_base)

    effective = delta is not None and delta > 0

    return {
        "metric_name": metric_name,
        "baseline_value": float(metric_base) if metric_base is not None else None,
        "candidate_value": float(metric_cand) if metric_cand is not None else None,
        "delta": delta,
        "effective": effective,
        "baseline_stage": "stage1_all_features",
        "candidate_stage": "stage3_representatives",
        "recommended_stage": "stage3_representatives" if effective else "stage1_all_features",
    }
