"""Load strategy×stage feature taxonomy from archetype YAML."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from mlbot_console.config import SETTINGS
from mlbot_console.services.strategy_registry import (
    account_layer_label,
    get_live_console_strategies,
)
from time_series_model.live.feature_stage_taxonomy import (
    ACCOUNT_LAYER_LABELS,
    build_console_feature_taxonomy,
)

_EMPTY_TAXONOMY: Dict[str, Any] = {
    "strategies": [],
    "index": {},
    "stage_order": [],
    "stage_labels": {},
    "account_layer_labels": {},
    "live_strategy_ids": [],
    "live_strategies": [],
}


def _live_strategy_taxonomy_entries(
    live: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for meta in live:
        layer = str(meta.get("account_layer") or "trend")
        out.append(
            {
                "id": str(meta["id"]),
                "account_layer": layer,
                "account_layer_title": account_layer_label(layer),
                "title": str(meta["id"]),
                "stages": {},
            }
        )
    return out


def _constitution_taxonomy_fields(live: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "live_strategy_ids": [str(s["id"]) for s in live],
        "live_strategies": [dict(s) for s in live],
        "constitution_source": str(SETTINGS.constitution_yaml),
        "account_layer_labels": dict(ACCOUNT_LAYER_LABELS),
    }


def _merge_live_into_taxonomy(
    tax: Dict[str, Any], live: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Ensure every constitution-enabled strategy appears in strategies (even without YAML)."""
    by_id = {str(s["id"]): s for s in tax.get("strategies") or []}
    for entry in _live_strategy_taxonomy_entries(live):
        sid = entry["id"]
        if sid not in by_id:
            by_id[sid] = entry
        else:
            existing = by_id[sid]
            if not existing.get("stages"):
                existing["stages"] = entry.get("stages") or {}
            if not existing.get("title"):
                existing["title"] = entry["title"]
    tax["strategies"] = sorted(by_id.values(), key=lambda s: str(s["id"]))
    return tax


@lru_cache(maxsize=1)
def get_feature_taxonomy() -> Dict[str, Any]:
    live = get_live_console_strategies()
    const_fields = _constitution_taxonomy_fields(live)
    root = SETTINGS.strategies_root
    if not root.is_dir():
        return {
            **_EMPTY_TAXONOMY,
            **const_fields,
            "strategies": _live_strategy_taxonomy_entries(live),
        }
    try:
        tax = build_console_feature_taxonomy(root, strategies=live)
        tax = _merge_live_into_taxonomy(tax, live)
        tax.update(const_fields)
        return tax
    except Exception:
        return {
            **_EMPTY_TAXONOMY,
            **const_fields,
            "strategies": _live_strategy_taxonomy_entries(live),
        }


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
