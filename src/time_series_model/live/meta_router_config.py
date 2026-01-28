"""
Live config: read from config/live/live_config_defaults.yaml at startup.
No database; single source of truth is the YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaRouterLiveConfig:
    enabled_archetypes: List[str]
    size_multipliers: Dict[str, float]
    nnmultihead_inference: Dict[str, Any]
    window_minutes: int
    min_order_interval_minutes: int


_DEFAULT_CONFIG_PATH = "config/live/live_config_defaults.yaml"


def load_meta_router_live_config(
    *,
    config_path: Optional[str] = None,
    archetype_registry_path: Optional[str] = None,
) -> MetaRouterLiveConfig:
    """
    Load live config from YAML file. Used at startup; no database.
    """
    path = Path(config_path or _DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Live config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # enabled_archetypes
    enabled_raw = raw.get("enabled_archetypes", "ALL")
    if isinstance(enabled_raw, str) and enabled_raw.strip().upper() == "ALL":
        reg_path = (
            archetype_registry_path or "config/nnmultihead/execution_archetypes.yaml"
        )
        from src.time_series_model.nnmultihead.strategy_profile import (
            load_execution_archetypes_registry,
        )

        arches = load_execution_archetypes_registry(reg_path)
        enabled_archetypes = list(arches.keys())
    elif isinstance(enabled_raw, list):
        enabled_archetypes = [str(x) for x in enabled_raw]
    else:
        enabled_archetypes = []

    # size_multipliers: fill missing with 1.0
    size_multipliers = raw.get("size_multipliers") or {}
    for arch in enabled_archetypes:
        if arch not in size_multipliers:
            size_multipliers[arch] = 1.0
    size_multipliers = {str(k): float(v) for k, v in size_multipliers.items()}

    window_minutes = int(raw.get("window_minutes", 15))
    min_order_interval_minutes = int(raw.get("min_order_interval_minutes", 10))
    nnmultihead_inference = dict(raw.get("nnmultihead_inference") or {})

    return MetaRouterLiveConfig(
        enabled_archetypes=enabled_archetypes,
        size_multipliers=size_multipliers,
        nnmultihead_inference=nnmultihead_inference,
        window_minutes=window_minutes,
        min_order_interval_minutes=min_order_interval_minutes,
    )


def select_first_enabled_archetype(cfg: MetaRouterLiveConfig) -> Optional[str]:
    return cfg.enabled_archetypes[0] if cfg.enabled_archetypes else None
