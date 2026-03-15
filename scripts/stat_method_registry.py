#!/usr/bin/env python3
"""Shared statistical method registry for pipeline scripts.

This module centralizes:
1) canonical method naming
2) fallback list de-duplication
3) shared RR split method evaluation
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


CANONICAL_METHODS = (
    "distribution_ks",
    "mean_effect",
    "welch_ttest",
    "tail_bad_rate_ratio",
    "upside_positive_rate_ratio",
)


def canonicalize_method_name(method: str | None) -> str:
    """Return canonical method name (lower-cased token)."""
    if not method:
        return "distribution_ks"
    return str(method).strip().lower()


def standardize_method_list(
    methods: Iterable[str] | None, default: List[str]
) -> List[str]:
    """Standardize + de-duplicate fallback list while keeping order."""
    source = list(methods or default)
    canonical: List[str] = []
    seen = set()
    for method in source:
        key = canonicalize_method_name(method)
        if key in seen:
            continue
        seen.add(key)
        canonical.append(key)
    return canonical or list(default)


def get_canonical_methods() -> List[str]:
    """Expose canonical methods for CLI choices."""
    return list(CANONICAL_METHODS)


# Backward-compatible function aliases (for module import stability only)
normalize_method = canonicalize_method_name
normalize_fallback_methods = standardize_method_list


def evaluate_rr_split_method(
    method: str,
    rr_pass: np.ndarray,
    rr_reject: np.ndarray,
    thresholds: Dict[str, float] | None = None,
) -> Tuple[bool, float, Dict[str, float]]:
    """Evaluate one canonical statistical method on pass/reject RR splits.

    Returns:
      (passed, score, extra_metrics)
    """
    from scipy import stats as _stats

    th = thresholds or {}
    method = canonicalize_method_name(method)

    mean_pass = float(np.mean(rr_pass))
    mean_reject = float(np.mean(rr_reject))
    effect = mean_pass - mean_reject

    min_effect = float(th.get("min_effect", 0.02))
    min_ks_stat = float(th.get("min_ks_statistic", 0.05))
    max_ks_pval = float(th.get("max_ks_pvalue", 0.05))
    t_test_alpha = float(th.get("t_test_alpha", 0.05))
    min_bad_rate_lift = float(th.get("min_bad_rate_lift", 1.05))
    min_positive_lift = float(th.get("min_positive_lift", 1.20))
    bad_rr_threshold = float(th.get("bad_rr_threshold", 0.0))
    positive_rr_threshold = float(th.get("positive_rr_threshold", 0.8))

    extra: Dict[str, float] = {
        "mean_pass": round(mean_pass, 6),
        "mean_reject": round(mean_reject, 6),
        "effect": round(effect, 6),
    }

    if method == "mean_effect":
        passed = effect > min_effect and mean_pass > mean_reject
        return passed, float(effect), extra

    if method == "distribution_ks":
        ks_stat, ks_pval = _stats.ks_2samp(rr_pass, rr_reject)
        extra["ks_stat"] = round(float(ks_stat), 6)
        extra["ks_pval"] = round(float(ks_pval), 8)
        passed = (
            float(ks_stat) >= min_ks_stat
            and float(ks_pval) < max_ks_pval
            and mean_pass > mean_reject
        )
        return passed, float(ks_stat), extra

    if method == "welch_ttest":
        t_stat, t_p_two = _stats.ttest_ind(rr_pass, rr_reject, equal_var=False)
        t_p = t_p_two / 2 if t_stat > 0 else 1.0 - t_p_two / 2
        extra["t_stat"] = round(float(t_stat), 6)
        extra["p_value"] = round(float(t_p), 8)
        passed = float(t_p) < t_test_alpha and mean_pass > mean_reject
        return passed, float(t_stat), extra

    if method == "tail_bad_rate_ratio":
        bad_rate_pass = float(np.mean(rr_pass < bad_rr_threshold))
        bad_rate_reject = float(np.mean(rr_reject < bad_rr_threshold))
        bad_rate_lift = (
            bad_rate_reject / max(bad_rate_pass, 1e-9) if bad_rate_reject > 0 else 1.0
        )
        extra["bad_rate_pass"] = round(bad_rate_pass, 6)
        extra["bad_rate_reject"] = round(bad_rate_reject, 6)
        extra["bad_rate_lift"] = round(float(bad_rate_lift), 6)
        passed = bad_rate_lift >= min_bad_rate_lift and mean_pass > mean_reject
        return passed, float(bad_rate_lift), extra

    if method == "upside_positive_rate_ratio":
        pr_pass = float(np.mean(rr_pass > positive_rr_threshold))
        pr_reject = float(np.mean(rr_reject > positive_rr_threshold))
        positive_lift = pr_pass / max(pr_reject, 1e-9) if pr_pass > 0 else 0.0
        extra["positive_rate_pass"] = round(pr_pass, 6)
        extra["positive_rate_reject"] = round(pr_reject, 6)
        extra["positive_lift"] = round(float(positive_lift), 6)
        passed = positive_lift >= min_positive_lift and mean_pass > mean_reject
        return passed, float(positive_lift), extra

    return False, 0.0, extra
