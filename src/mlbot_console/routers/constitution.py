from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.strategy_registry import get_console_strategies
from mlbot_console.services.universe import load_universe_symbols

router = APIRouter(tags=["constitution"])


@router.get("/api/constitution/summary")
def constitution_summary() -> dict:
    path = SETTINGS.constitution_yaml
    if not path.is_file():
        return ok({"path": str(path), "loaded": False})
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ra = raw.get("resource_allocation") or {}
    spot = raw.get("spot") or {}
    multi = raw.get("multi_leg") or {}
    return ok(
        {
            "path": str(path),
            "loaded": True,
            "enabled_archetypes": ra.get("enabled_archetypes") or [],
            "multi_leg_strategies": multi.get("strategies") or [],
            "spot_strategies": spot.get("strategies") or [],
            "console_strategies": get_console_strategies(),
            "symbols": load_universe_symbols(SETTINGS.universe_yaml),
        }
    )
