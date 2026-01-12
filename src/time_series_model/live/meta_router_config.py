from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class VolMeanOverlayConfig:
    enabled: bool
    archetype_id: str
    size_multiplier: float


@dataclass(frozen=True)
class MetaRouterLiveConfig:
    version: int
    name: str
    enabled_archetypes: Dict[str, List[str]]
    size_multipliers: Dict[str, float]
    vol_mean: VolMeanOverlayConfig
    router_thresholds: Dict[str, Any]
    nnmultihead_inference: Dict[str, Any]
    decision_loop: Dict[str, Any]


def load_meta_router_live_config(path: str | Path) -> MetaRouterLiveConfig:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ena = obj.get("enabled_archetypes") or {}
    sm = obj.get("size_multipliers") or {}
    vm = obj.get("vol_mean") or {}
    rt = obj.get("router_thresholds") or {}
    nni = obj.get("nnmultihead_inference") or {}
    dl = obj.get("decision_loop") or {}
    enabled_archetypes: Dict[str, List[str]] = {}
    if isinstance(ena, dict):
        for k, v in ena.items():
            if isinstance(v, list):
                enabled_archetypes[str(k).upper()] = [str(x) for x in v]
    size_multipliers: Dict[str, float] = {}
    if isinstance(sm, dict):
        for k, v in sm.items():
            try:
                size_multipliers[str(k)] = float(v)
            except Exception:
                continue
    return MetaRouterLiveConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "meta_router_live_config")),
        enabled_archetypes=enabled_archetypes,
        size_multipliers=size_multipliers,
        vol_mean=VolMeanOverlayConfig(
            enabled=bool(vm.get("enabled", False)),
            archetype_id=str(
                vm.get("archetype_id", "VolMeanCompressionExpansionReversion")
            ),
            size_multiplier=float(vm.get("size_multiplier", 0.05)),
        ),
        router_thresholds=dict(rt) if isinstance(rt, dict) else {},
        nnmultihead_inference=dict(nni) if isinstance(nni, dict) else {},
        decision_loop=dict(dl) if isinstance(dl, dict) else {},
    )


def select_first_enabled_archetype(
    cfg: MetaRouterLiveConfig, *, regime: str
) -> Optional[str]:
    rr = str(regime).upper()
    xs = cfg.enabled_archetypes.get(rr) or []
    return xs[0] if xs else None
