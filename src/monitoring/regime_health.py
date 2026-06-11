"""Regime-aware monitor helpers (read regime.yaml, not hardcoded EMA only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.time_series_model.archetype.loader import RegimeConfig

_DEFAULT_LABELS = ("bull", "bear", "neutral")


def has_labeled_regime_schema(regime_yaml: Dict[str, Any]) -> bool:
    ar = regime_yaml.get("allowed_regimes")
    if not isinstance(ar, dict):
        return False
    return any(isinstance(v, dict) and v.get("rules") for v in ar.values())


def resolve_baseline_regime_shares(
    *,
    regime_yaml: Dict[str, Any],
    baseline_entry: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, float]]:
    """Baseline shares from watchdog JSON or regime.yaml last_calibration."""
    entry = baseline_entry or {}
    raw = entry.get("regime_shares")
    if isinstance(raw, dict) and raw:
        return {str(k): float(v) for k, v in raw.items()}

    lc = regime_yaml.get("last_calibration") or {}
    raw = lc.get("regime_shares")
    if isinstance(raw, dict) and raw:
        return {str(k): float(v) for k, v in raw.items()}

    bull = entry.get("bull_share")
    if bull is not None:
        try:
            return {"bull": float(bull)}
        except (TypeError, ValueError):
            return None
    return None


def regime_shares_from_window(
    window_df: pd.DataFrame,
    regime_yaml: Dict[str, Any],
) -> Dict[str, float]:
    """Classify each row with RegimeConfig.classify() → label shares."""
    cfg = RegimeConfig.from_mapping(regime_yaml)
    if window_df.empty:
        return {label: 0.0 for label in _DEFAULT_LABELS}

    labels: List[str] = []
    for _, row in window_df.iterrows():
        feats = {c: row[c] for c in window_df.columns if pd.notna(row[c])}
        labels.append(cfg.classify(feats))

    series = pd.Series(labels)
    n = max(len(series), 1)
    return {label: float((series == label).sum()) / n for label in _DEFAULT_LABELS}


def evaluate_regime_share_drift(
    *,
    strategy: str,
    regime_yaml: Dict[str, Any],
    window_df: pd.DataFrame,
    baseline_entry: Optional[Dict[str, Any]] = None,
    share_tol: float = 0.10,
) -> Dict[str, Any]:
    """Compare current regime label mix vs baseline (labeled regime schema)."""
    current = regime_shares_from_window(window_df, regime_yaml)
    baseline = resolve_baseline_regime_shares(
        regime_yaml=regime_yaml,
        baseline_entry=baseline_entry,
    )
    item = {
        "kind": "regime_shares",
        "current": current,
        "baseline": baseline,
        "share_tol": share_tol,
    }
    if baseline is None:
        return {
            "strategy": strategy,
            "any_alert": False,
            "status": "BASELINE_MISSING",
            "skipped": (
                "regime_shares baseline missing — add last_calibration.regime_shares "
                "to regime.yaml or regime_watchdog_baseline.json after Tier-0"
            ),
            "items": [item],
        }

    alerts: List[str] = []
    for label, base_share in baseline.items():
        cur = current.get(label, 0.0)
        delta = cur - base_share
        if abs(delta) > share_tol:
            alerts.append(
                f"REGIME_SHARE_DRIFT: {label} {cur:.1%} vs baseline {base_share:.1%}"
                f" (delta={delta:+.1%}, tol={share_tol:+.1%})"
            )

    return {
        "strategy": strategy,
        "any_alert": bool(alerts),
        "status": "ALERT" if alerts else "OK",
        "alerts": alerts,
        "items": [item],
    }
