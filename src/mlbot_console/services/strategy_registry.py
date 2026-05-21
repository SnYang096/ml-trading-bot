"""Console strategy registry aligned with constitution.yaml (trend / spot / multi-leg)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import yaml

from live_data_stream.constitution_config import (
    enabled_archetypes_from_constitution,
    load_constitution_dict,
    multi_leg_strategies_from_constitution,
    spot_strategies_from_constitution,
)
from mlbot_console.config import SETTINGS

# Display titles for known strategy slugs (constitution id -> UI label).
STRATEGY_TITLES: Dict[str, str] = {
    "tpc": "TPC",
    "bpc": "BPC",
    "me": "ME",
    "srb": "SRB",
    "spot_accum_simple": "Spot",
    "chop_grid": "Chop Grid",
    "trend_scalp": "Trend Scalp",
}

_FALLBACK_STRATEGIES: List[Dict[str, str]] = [
    {"id": "tpc", "account_layer": "trend", "title": "TPC"},
    {"id": "bpc", "account_layer": "trend", "title": "BPC"},
    {"id": "me", "account_layer": "trend", "title": "ME"},
    {"id": "srb", "account_layer": "trend", "title": "SRB"},
    {"id": "spot_accum_simple", "account_layer": "spot", "title": "Spot"},
    {"id": "chop_grid", "account_layer": "multi_leg", "title": "Chop Grid"},
    {"id": "trend_scalp", "account_layer": "multi_leg", "title": "Trend Scalp"},
]


def _title(strategy_id: str) -> str:
    sid = str(strategy_id or "").strip().lower()
    return STRATEGY_TITLES.get(sid, sid.replace("_", " ").title() or sid)


def strategies_from_constitution_cfg(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for sid in enabled_archetypes_from_constitution(cfg):
        slug = str(sid).strip().lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append({"id": slug, "account_layer": "trend", "title": _title(slug)})
    for sid in spot_strategies_from_constitution(cfg):
        slug = str(sid).strip().lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append({"id": slug, "account_layer": "spot", "title": _title(slug)})
    for sid in multi_leg_strategies_from_constitution(cfg):
        slug = str(sid).strip().lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append({"id": slug, "account_layer": "multi_leg", "title": _title(slug)})
    return out


@lru_cache(maxsize=1)
def get_console_strategies() -> List[Dict[str, str]]:
    path = SETTINGS.constitution_yaml
    if path.is_file():
        try:
            cfg = load_constitution_dict(str(path))
            rows = strategies_from_constitution_cfg(cfg)
            if rows:
                return rows
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            pass
    return list(_FALLBACK_STRATEGIES)


def strategy_title(strategy_id: str) -> str:
    return _title(strategy_id)


def strategies_for_account_layer(account_layer: str) -> List[Dict[str, str]]:
    layer = str(account_layer or "").strip().lower()
    return [s for s in get_console_strategies() if s.get("account_layer") == layer]
