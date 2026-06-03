"""Runtime-only multileg columns (aliases / gates) not emitted by the feature DAG."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Set

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

_STRING_LIVE_COLUMNS = frozenset({"trend_direction"})


def is_string_live_column(name: str) -> bool:
    return str(name) in _STRING_LIVE_COLUMNS


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
    return False
