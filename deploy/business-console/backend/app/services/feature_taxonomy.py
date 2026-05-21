"""Load strategy×stage feature taxonomy from archetype YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import SETTINGS
from app.services.feature_stage_taxonomy import build_console_feature_taxonomy

_EMPTY_TAXONOMY: Dict[str, Any] = {
    "strategies": [],
    "index": {},
    "stage_order": [],
    "stage_labels": {},
    "account_layer_labels": {},
}


def strategies_config_root() -> Path:
    return SETTINGS.strategies_root


@lru_cache(maxsize=1)
def get_feature_taxonomy() -> Dict[str, Any]:
    root = strategies_config_root()
    if not root.is_dir():
        return dict(_EMPTY_TAXONOMY)
    try:
        return build_console_feature_taxonomy(root)
    except Exception:
        return dict(_EMPTY_TAXONOMY)


def lookup_column_meta(column: str) -> Optional[Dict[str, str]]:
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
