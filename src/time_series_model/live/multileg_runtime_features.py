"""Runtime-only multileg columns (aliases / gates) not emitted by the feature DAG."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional, Set

import pandas as pd

from src.features.semantic_chop import (
    as_finite_float,
    resolve_semantic_chop,
    set_canonical_semantic_chop,
)
from src.time_series_model.live.regime_box_prefilter import is_stable_box_bar

# Default box_prefilter thresholds (match archetypes regime.yaml extensions.multileg).
_DEFAULT_BOX_PREFILTER = {
    "stability_min": 0.85,
    "width_min": 0.04,
    "width_max": 0.30,
    "touches_min": 5,
}


def _wants(wanted: Optional[Set[str]], key: str) -> bool:
    """Add a runtime column only when the plan expects it (or no plan is set)."""
    return (not wanted) or (key in wanted)


def enrich_multileg_runtime_features(
    features: MutableMapping[str, Any],
    *,
    wanted: Optional[Set[str]] = None,
) -> None:
    """Back-fill alias and gate columns expected in ``live_feature_set``.

    Only columns present in ``wanted`` are added so non-multileg feature
    computers (e.g. pure TPC/BPC) are not polluted with spurious gate columns.
    """
    if _wants(wanted, "semantic_chop"):
        set_canonical_semantic_chop(features)

    if _wants(wanted, "trend_confidence_f"):
        conf = as_finite_float(features.get("trend_confidence"))
        if conf is not None and "trend_confidence_f" not in features:
            features["trend_confidence_f"] = conf

    if _wants(wanted, "box_prefilter") and "box_prefilter" not in features:
        stable = False
        for window in (60, 120):
            if is_stable_box_bar(
                features,
                _DEFAULT_BOX_PREFILTER,
                box_window=window,
            ):
                stable = True
                break
        features["box_prefilter"] = 1.0 if stable else 0.0

    if _wants(wanted, "trend_direction"):
        coerce_trend_direction_for_bus(features)


def trend_direction_to_sign(value: Any) -> float:
    """Map UP/DOWN labels or raw sign to +1 / -1 (NaN when unknown)."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        return 1.0 if value.strip().upper() != "DOWN" else -1.0
    val = as_finite_float(value)
    if val is None:
        return float("nan")
    if val == 0.0:
        return float("nan")
    return 1.0 if val > 0.0 else -1.0


def normalize_trend_direction_column(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed string/float ``trend_direction`` history to float64 for parquet."""
    if df.empty or "trend_direction" not in df.columns:
        return df
    out = df.copy()
    out["trend_direction"] = out["trend_direction"].map(trend_direction_to_sign)
    return out.astype({"trend_direction": "float64"}, errors="ignore")


def coerce_trend_direction_for_bus(features: MutableMapping[str, Any]) -> None:
    """Publish ``trend_direction`` as numeric sign (+1 UP / -1 DOWN) for parquet."""
    raw = as_finite_float(features.get("trend_direction_raw"))
    if raw is not None and raw != 0.0:
        features["trend_direction"] = 1.0 if raw > 0.0 else -1.0
        return
    sign = trend_direction_to_sign(features.get("trend_direction"))
    if math.isfinite(sign):
        features["trend_direction"] = sign


def trend_direction_label(features: Mapping[str, Any], *, default: str = "UP") -> str:
    """Decode bus/engine features to UP/DOWN."""
    td = features.get("trend_direction")
    if isinstance(td, str):
        label = td.strip().upper()
        if label in {"UP", "DOWN"}:
            return label
    sign = as_finite_float(features.get("trend_direction_raw"))
    if sign is None:
        sign = as_finite_float(td)
    if sign is None:
        return default
    return "UP" if sign >= 0.0 else "DOWN"


def live_feature_satisfied(
    key: str,
    features: Mapping[str, Any],
) -> bool:
    """True when ``key`` is present or covered by a runtime alias/source column."""
    if key in features:
        return True
    if (
        key == "semantic_chop"
        and resolve_semantic_chop(features, default=None) is not None
    ):
        return True
    if (
        key == "trend_confidence_f"
        and as_finite_float(features.get("trend_confidence")) is not None
    ):
        return True
    if key == "box_prefilter":
        for window in (60, 120):
            w = max(1, int(window))
            if features.get(f"box_stability_{w}") is not None:
                return True
    if key == "trend_direction":
        if as_finite_float(features.get("trend_direction")) is not None:
            return True
        if as_finite_float(features.get("trend_direction_raw")) is not None:
            return True
        td = features.get("trend_direction")
        return isinstance(td, str) and bool(str(td).strip())
    return False
