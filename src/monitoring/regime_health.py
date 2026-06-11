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


def has_multileg_regime_schema(regime_yaml: Dict[str, Any]) -> bool:
    ext = regime_yaml.get("extensions") or {}
    ml = ext.get("multileg")
    if not isinstance(ml, dict):
        return False
    return bool(str(ml.get("entry_feature") or "").strip()) and ml.get("entry_min") is not None


def multileg_config(regime_yaml: Dict[str, Any]) -> Dict[str, Any]:
    ext = regime_yaml.get("extensions") or {}
    ml = ext.get("multileg")
    return ml if isinstance(ml, dict) else {}


def resolve_multileg_baseline(
    *,
    strategy: str,
    regime_yaml: Dict[str, Any],
    baseline_entry: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, float]]:
    """Baseline entry pass metrics from watchdog JSON or regime last_calibration."""
    entry = baseline_entry or {}
    raw = entry.get("multileg_baseline")
    if isinstance(raw, dict):
        row = raw.get(strategy) or raw
        if isinstance(row, dict) and row.get("entry_pass_rate") is not None:
            return {
                "entry_pass_rate": float(row["entry_pass_rate"]),
                "median_entry_feature": float(row.get("median_entry_feature") or 0.0),
            }

    lc = regime_yaml.get("last_calibration") or {}
    raw = lc.get("multileg_baseline")
    if isinstance(raw, dict):
        row = raw.get(strategy) or raw
        if isinstance(row, dict) and row.get("entry_pass_rate") is not None:
            return {
                "entry_pass_rate": float(row["entry_pass_rate"]),
                "median_entry_feature": float(row.get("median_entry_feature") or 0.0),
            }
    return None


def entry_pass_rate_from_window(
    window_df: pd.DataFrame,
    *,
    entry_feature: str,
    entry_min: float,
    min_samples: int = 20,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Return (pass_rate, median_feature, skip_reason)."""
    if entry_feature not in window_df.columns:
        return None, None, f"entry_feature {entry_feature!r} not in window parquet"
    series = pd.to_numeric(window_df[entry_feature], errors="coerce").dropna()
    if len(series) < min_samples:
        return None, None, f"insufficient samples ({len(series)} < {min_samples})"
    rate = float((series >= float(entry_min)).sum()) / len(series)
    return rate, float(series.median()), None


def evaluate_multileg_entry_health(
    *,
    strategy: str,
    regime_yaml: Dict[str, Any],
    window_df: pd.DataFrame,
    baseline_entry: Optional[Dict[str, Any]] = None,
    pass_rate_tol: float = 0.10,
    min_samples: int = 20,
) -> Dict[str, Any]:
    """Compare extensions.multileg entry pass rate vs baseline."""
    if not has_multileg_regime_schema(regime_yaml):
        return {
            "strategy": strategy,
            "any_alert": False,
            "status": "UNSUPPORTED",
            "skipped": "extensions.multileg schema missing in regime.yaml",
            "items": [],
        }

    ml = multileg_config(regime_yaml)
    entry_feature = str(ml.get("entry_feature") or "")
    entry_min = float(ml.get("entry_min"))
    pass_rate, median_feat, skip = entry_pass_rate_from_window(
        window_df,
        entry_feature=entry_feature,
        entry_min=entry_min,
        min_samples=min_samples,
    )
    baseline = resolve_multileg_baseline(
        strategy=strategy,
        regime_yaml=regime_yaml,
        baseline_entry=baseline_entry,
    )
    item: Dict[str, Any] = {
        "kind": "multileg_entry",
        "entry_feature": entry_feature,
        "entry_min": entry_min,
        "current_pass_rate": pass_rate,
        "current_median_entry_feature": median_feat,
        "baseline": baseline,
        "pass_rate_tol": pass_rate_tol,
    }
    if skip:
        return {
            "strategy": strategy,
            "any_alert": False,
            "status": "SKIPPED",
            "skipped": skip,
            "items": [item],
        }
    if baseline is None:
        return {
            "strategy": strategy,
            "any_alert": False,
            "status": "BASELINE_MISSING",
            "skipped": (
                "multileg_baseline missing — add last_calibration.multileg_baseline "
                f"for {strategy} after Tier-0"
            ),
            "items": [item],
        }

    base_rate = float(baseline["entry_pass_rate"])
    delta = float(pass_rate) - base_rate
    alerts: List[str] = []
    if abs(delta) > pass_rate_tol:
        alerts.append(
            f"MULTILEG_PASS_RATE_DRIFT: {pass_rate:.1%} vs baseline {base_rate:.1%}"
            f" (delta={delta:+.1%}, tol={pass_rate_tol:+.1%})"
        )

    return {
        "strategy": strategy,
        "any_alert": bool(alerts),
        "status": "ALERT" if alerts else "OK",
        "alerts": alerts,
        "items": [item],
    }
