from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MetaRouterLiveConfig:
    enabled_archetypes: List[str]
    size_multipliers: Dict[str, float]
    nnmultihead_inference: Dict[str, Any]
    window_minutes: int
    min_order_interval_minutes: int


def load_meta_router_live_config(
    *, db_path: Optional[str] = None
) -> MetaRouterLiveConfig:
    from src.order_management.storage import Storage

    storage = Storage(db_path=db_path or "data/order_management.db")
    cfg = storage.get_live_config()
    if cfg is None:
        raise ValueError("live_config not initialized in database")

    enabled = cfg.get("enabled_archetypes") or []
    size_multipliers = cfg.get("size_multipliers") or {}
    nnmh = cfg.get("nnmultihead_inference") or {}
    return MetaRouterLiveConfig(
        enabled_archetypes=[str(x) for x in enabled],
        size_multipliers={str(k): float(v) for k, v in size_multipliers.items()},
        nnmultihead_inference=dict(nnmh),
        window_minutes=int(cfg.get("window_minutes", 15)),
        min_order_interval_minutes=int(cfg.get("min_order_interval_minutes", 10)),
    )


def select_first_enabled_archetype(cfg: MetaRouterLiveConfig) -> Optional[str]:
    return cfg.enabled_archetypes[0] if cfg.enabled_archetypes else None
