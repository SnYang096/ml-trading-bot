"""Shared regime.yaml parsing for TPC (B-system) and multileg strategies.

File format (TPC-shaped, both strategy families share the same YAML schema):

    allowed_regimes: [bull, bear, neutral]   # TPC regime-label mask
    allowed_sides:   [long, short]           # direction mask
    rules:                                   # per-bar RegimeConfig rules (TPC: written
      ...                                    # explicitly; multileg: auto-synthesised,
                                             # do NOT write manually)
    extensions:
      multileg:                              # multileg engine params only
        entry_feature: bpc_semantic_chop
        entry_min: 0.52                      # source-of-truth for RegimeConfig rule
        exit_below: 0.33                     # hysteresis exit (no TPC equivalent)
        ...

For TPC/B-system strategies ``extensions.multileg`` is absent and ``rules`` is written
directly.  For multileg (chop_grid, trend_scalp) ``rules`` must be omitted from the YAML
— ``parse_regime_layer`` synthesises them from ``extensions.multileg`` automatically, so
there is no duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from src.time_series_model.archetype.loader import (
    RegimeConfig,
    _DEFAULT_ALLOWED_REGIMES,
    _DEFAULT_ALLOWED_SIDES,
)

_REGIME_LAYER_META_KEYS = frozenset(
    {"last_calibration", "last_multileg_evaluation"}
)

# Keys in extensions.multileg that map to the entry-threshold rule.
# entry_chop_min kept as legacy fallback for old YAML / sweep-script candidate dicts.
_ENTRY_THRESHOLD_KEYS = ("entry_min", "entry_chop_min")


def multileg_regime_section(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Engine/backtest regime block (extensions.multileg, or legacy nested regime:)."""
    extensions = raw.get("extensions")
    if isinstance(extensions, dict):
        multileg = extensions.get("multileg")
        if isinstance(multileg, dict):
            return dict(multileg)
    # Legacy: nested ``regime:`` block written directly in the YAML.
    nested = raw.get("regime")
    if isinstance(nested, dict):
        return dict(nested)
    return {}


def _synthesise_rules_from_multileg(multileg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Auto-generate RegimeConfig rules from extensions.multileg entry threshold.

    The exit threshold (exit_below) is *not* expressed as a rule
    because RegimeConfig.evaluate() is stateless — hysteresis is the engine's job.
    """
    feature = str(multileg.get("entry_feature") or "bpc_semantic_chop").strip()
    for key in _ENTRY_THRESHOLD_KEYS:
        value = multileg.get(key)
        if value is not None:
            return [
                {
                    "feature": feature,
                    "operator": ">=",
                    "value": float(value),
                    "locked": True,
                    "lock_reason": (
                        f"synthesised from extensions.multileg.{key} "
                        "— edit extensions.multileg, not this rule"
                    ),
                }
            ]
    return []


def parse_regime_layer(raw: Mapping[str, Any]) -> Tuple[RegimeConfig, Dict[str, Any]]:
    """Return (RegimeConfig, multileg engine dict).

    RegimeConfig rules come from:
    - ``rules`` (TPC / explicit) if non-empty, OR
    - auto-synthesised from ``extensions.multileg`` entry threshold (multileg strategies).
    """
    multileg = multileg_regime_section(raw)
    explicit_rules = list(raw.get("rules") or [])
    if not explicit_rules and multileg:
        rules = _synthesise_rules_from_multileg(multileg)
    else:
        rules = explicit_rules
    config = RegimeConfig(
        rules=rules,
        allowed_regimes=list(
            raw.get("allowed_regimes") or list(_DEFAULT_ALLOWED_REGIMES)
        ),
        allowed_sides=list(
            raw.get("allowed_sides") or list(_DEFAULT_ALLOWED_SIDES)
        ),
    )
    return config, multileg


def regime_layer_effective_fragment(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Build merge fragment for ``load_multileg_effective_config`` / diagnostics."""
    if not raw:
        return {}
    config, multileg = parse_regime_layer(raw)
    out: Dict[str, Any] = {}
    if multileg:
        out["regime"] = multileg
    if config.rules:
        out["regime_rules"] = list(config.rules)
    if tuple(config.allowed_regimes) != _DEFAULT_ALLOWED_REGIMES:
        out["allowed_regimes"] = list(config.allowed_regimes)
    if tuple(config.allowed_sides) != _DEFAULT_ALLOWED_SIDES:
        out["allowed_sides"] = list(config.allowed_sides)
    for key in _REGIME_LAYER_META_KEYS:
        if key in raw:
            out[key] = raw[key]
    return out


def load_regime_layer(path: Path) -> Tuple[RegimeConfig, Dict[str, Any]]:
    if not path.exists():
        return RegimeConfig(), {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return parse_regime_layer(raw)


def multileg_extensions_section(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Mutable ``extensions.multileg`` dict; migrates legacy ``regime:`` block in-place."""
    extensions = doc.setdefault("extensions", {})
    multileg: Optional[Dict[str, Any]] = extensions.get("multileg")
    if isinstance(multileg, dict):
        return multileg
    legacy = doc.pop("regime", None)
    new_multileg: Dict[str, Any] = dict(legacy) if isinstance(legacy, dict) else {}
    extensions["multileg"] = new_multileg
    return new_multileg
