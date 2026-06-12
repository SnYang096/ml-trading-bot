"""Console strategy id → account layer (B / A / C)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

from time_series_model.live.feature_stage_taxonomy import CONSOLE_STRATEGIES

_STRATEGY_LAYER: Dict[str, str] = {
    str(meta["id"]).lower(): str(meta["account_layer"])
    for meta in CONSOLE_STRATEGIES
}

_LAYER_LABELS: Dict[str, str] = {
    "trend": "B·Trend",
    "spot": "A·Spot",
    "multi_leg": "C·Multi-leg",
}


def strategy_account_layer(strategy_id: str) -> str:
    """Return trend | spot | multi_leg; unknown trend strategies default to trend."""
    sid = str(strategy_id or "").strip().lower()
    if not sid:
        return "trend"
    if sid in _STRATEGY_LAYER:
        return _STRATEGY_LAYER[sid]
    if sid in ("spot", "multi_leg", "trend"):
        return sid
    if "spot" in sid:
        return "spot"
    if sid in ("chop", "grid", "multileg", "multi_leg"):
        return "multi_leg"
    return "trend"


def account_layer_label(layer: str) -> str:
    return _LAYER_LABELS.get(str(layer or ""), str(layer or ""))


def known_strategy_ids() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGY_LAYER.keys()))


def strategies_for_layer(layer: str) -> tuple[str, ...]:
    lay = str(layer or "").strip().lower()
    return tuple(sid for sid, acct in _STRATEGY_LAYER.items() if acct == lay)


def default_spot_strategy_id() -> str:
    ids = strategies_for_layer("spot")
    return ids[0] if ids else "spot_accum_simple"


def spot_strategy_ids() -> tuple[str, ...]:
    return strategies_for_layer("spot")


def strategy_has_deployed_archetypes(
    strategy_id: str, *, strategies_root: Union[str, Path]
) -> bool:
    """True when live/highcap/config/strategies/<id>/archetypes exists."""
    sid = str(strategy_id or "").strip().lower()
    if not sid:
        return False
    arch = strategies_root / sid / "archetypes"
    return arch.is_dir()


@lru_cache(maxsize=1)
def get_live_console_strategies() -> List[Dict[str, str]]:
    """Constitution-enabled strategies with deployed live archetype trees."""
    meta_by_id = {s["id"]: s for s in get_console_strategies()}
    try:
        from mlbot_console.config import SETTINGS
        from src.live_data_stream.constitution_config import (
            console_live_strategies_from_constitution,
            load_constitution_dict,
            resolve_constitution_yaml_path,
        )

        explicit = str(SETTINGS.constitution_yaml or "").strip()
        if explicit:
            if not os.path.isfile(explicit):
                return []
            path = explicit
        else:
            path = resolve_constitution_yaml_path()
            if not path or not os.path.isfile(path):
                return []
        cfg = load_constitution_dict(path)
        if not cfg:
            return []
        out: List[Dict[str, str]] = []
        for sid in console_live_strategies_from_constitution(cfg):
            key = str(sid).strip().lower()
            if not key:
                continue
            if key in meta_by_id:
                row = dict(meta_by_id[key])
                # Trade Map / taxonomy: show constitution strategy slug, not friendly aliases.
                row["title"] = key
                out.append(row)
            else:
                out.append(
                    {
                        "id": key,
                        "account_layer": strategy_account_layer(key),
                        "title": key,
                    }
                )
        deployed = [
            row
            for row in out
            if strategy_has_deployed_archetypes(
                str(row.get("id") or ""), strategies_root=SETTINGS.strategies_root
            )
        ]
        return sorted(deployed, key=lambda x: x["id"])
    except Exception:
        # Fail closed: do not show research archetypes when constitution is unreadable.
        return []


@lru_cache(maxsize=1)
def get_console_strategies() -> List[Dict[str, str]]:
    """Strategy registry for taxonomy / constitution summary (id, account_layer, title)."""
    by_id: Dict[str, Dict[str, str]] = {
        str(meta["id"]): {
            "id": str(meta["id"]),
            "account_layer": str(meta["account_layer"]),
            "title": str(meta.get("title") or meta["id"]),
        }
        for meta in CONSOLE_STRATEGIES
    }
    try:
        from mlbot_console.config import SETTINGS
        from src.live_data_stream.constitution_config import (
            load_constitution_dict,
            resolve_constitution_yaml_path,
            strategies_for_slot_metrics_from_constitution,
        )

        path = resolve_constitution_yaml_path(override=str(SETTINGS.constitution_yaml))
        cfg = load_constitution_dict(path)
        for sid in strategies_for_slot_metrics_from_constitution(cfg):
            if sid in by_id:
                continue
            by_id[sid] = {
                "id": sid,
                "account_layer": strategy_account_layer(sid),
                "title": sid,
            }
    except Exception:
        pass
    return sorted(by_id.values(), key=lambda x: x["id"])


def layer_for_funnel_filter(
    account_layer: str,
    strategy: str,
) -> Optional[str]:
    """Resolve effective account layer when API passes layer and/or strategy filters."""
    strat = str(strategy or "").strip().lower()
    layer = str(account_layer or "").strip().lower()
    if strat:
        return strategy_account_layer(strat)
    if layer in _LAYER_LABELS:
        return layer
    return None
