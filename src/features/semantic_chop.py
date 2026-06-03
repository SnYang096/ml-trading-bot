"""Canonical semantic_chop resolution across archetype-prefixed alias columns."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

SEMANTIC_CHOP_COLUMNS: Tuple[str, ...] = (
    "semantic_chop",
    "tpc_semantic_chop",
    "bpc_semantic_chop",
    "me_semantic_chop",
    "bpt_semantic_chop",
)

_MULTILEG_RUNTIME_ALIASES: dict[str, tuple[str, ...]] = {
    "semantic_chop": SEMANTIC_CHOP_COLUMNS[1:],
    "bpc_semantic_chop": ("semantic_chop", "tpc_semantic_chop", "me_semantic_chop", "bpt_semantic_chop"),
    "trend_confidence": ("trend_confidence_f",),
}


def multileg_feature_aliases(feature: str) -> tuple[str, ...]:
    """Runtime alias columns for a multileg regime/gate feature name."""
    key = str(feature or "").strip()
    if not key:
        return ()
    return _MULTILEG_RUNTIME_ALIASES.get(key, ())


def as_finite_float(value: Any) -> Optional[float]:
    """Return float when finite; None for missing, NaN, or inf."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def resolve_semantic_chop(
    features: Mapping[str, Any],
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    """First finite value among semantic chop aliases."""
    for key in SEMANTIC_CHOP_COLUMNS:
        val = as_finite_float(features.get(key))
        if val is not None:
            return val
    return default


def set_canonical_semantic_chop(features: MutableMapping[str, Any]) -> None:
    """Add canonical ``semantic_chop`` only; never rewrite prefixed alias columns.

    Used on the feature-bus publish path so producers do not mutate the values of
    archetype-prefixed columns (e.g. ``bpc_semantic_chop``) that other consumers
    may read on their own semantics.
    """
    resolved = resolve_semantic_chop(features, default=None)
    if resolved is None:
        return
    features["semantic_chop"] = resolved


def normalize_semantic_chop_aliases(features: MutableMapping[str, Any]) -> None:
    """Set canonical ``semantic_chop`` and back-fill present alias keys in-place.

    Intended for transient consumer-side dicts (not persisted), where collapsing
    every prefix alias to the resolved value simplifies downstream reads.
    """
    resolved = resolve_semantic_chop(features, default=None)
    if resolved is None:
        return
    features["semantic_chop"] = resolved
    for key in SEMANTIC_CHOP_COLUMNS[1:]:
        if key in features:
            features[key] = resolved


def resolve_feature_float(
    features: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: float = 0.0,
) -> float:
    """Resolve first finite float from ordered keys."""
    for key in keys:
        val = as_finite_float(features.get(key))
        if val is not None:
            return val
    return float(default)
