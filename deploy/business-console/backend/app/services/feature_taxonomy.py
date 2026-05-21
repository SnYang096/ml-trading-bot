"""Load strategy×stage feature taxonomy from repo archetype YAML."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import SETTINGS

_REPO = SETTINGS.repo_root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.time_series_model.live.feature_stage_taxonomy import (  # noqa: E402
    build_console_feature_taxonomy,
)


def strategies_config_root() -> Path:
    return _REPO / "config" / "strategies"


@lru_cache(maxsize=1)
def get_feature_taxonomy() -> Dict[str, Any]:
    return build_console_feature_taxonomy(strategies_config_root())


def lookup_column_meta(column: str) -> Optional[Dict[str, str]]:
    """Best match for a bus column (exact name, or strip _f suffix)."""
    tax = get_feature_taxonomy()
    index: Dict[str, List[Dict[str, str]]] = tax.get("index") or {}
    col = str(column or "").strip()
    if not col:
        return None
    hits = index.get(col)
    if hits:
        return hits[0]
    if col.endswith("_f"):
        hits = index.get(col[:-2])
        if hits:
            return hits[0]
    base = col[:-2] if col.endswith("_f") else col
    for key, recs in index.items():
        if key.endswith("_f") and key[:-2] == base and recs:
            return recs[0]
    return None


def enrich_columns_with_taxonomy(columns: List[str]) -> Dict[str, Any]:
    tax = get_feature_taxonomy()
    index = tax.get("index") or {}
    enriched: Dict[str, List[Dict[str, str]]] = {}
    for col in columns:
        meta = lookup_column_meta(col)
        if meta:
            enriched[col] = index.get(meta["column"], [meta])
        elif col in index:
            enriched[col] = index[col]
    return {
        "taxonomy": tax,
        "column_meta": enriched,
    }
