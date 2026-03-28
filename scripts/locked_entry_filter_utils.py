from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _filter_signature(flt: Dict[str, Any]) -> str:
    """Prefer explicit id; fallback to normalized single-condition signature."""
    fid = str(flt.get("id", "")).strip()
    if fid:
        return fid
    conds = flt.get("conditions") or []
    if isinstance(conds, list) and conds:
        c0 = conds[0] if isinstance(conds[0], dict) else {}
        return (
            f"{c0.get('feature','')}|{c0.get('operator','')}|" f"{c0.get('value','')}"
        )
    return "unknown"


def load_locked_entry_filters(path: Path) -> List[Dict[str, Any]]:
    """Load locked: true filters from entry_filters.yaml."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    filters = raw.get("filters") or []
    if not isinstance(filters, list):
        return []
    return [
        copy.deepcopy(f)
        for f in filters
        if isinstance(f, dict) and bool(f.get("locked", False))
    ]


def merge_locked_entry_filters(
    target_path: Path,
    locked_filters: List[Dict[str, Any]],
    *,
    default_enabled: bool = False,
) -> Dict[str, int]:
    """Ensure locked entry filters are not lost in target file.

    Missing locked filters are appended and default to disabled (enabled=false)
    so later runs can retune thresholds and decide to re-enable.
    """
    if not locked_filters:
        return {"added": 0, "total": 0}

    raw: Dict[str, Any] = {}
    if target_path.exists():
        raw = yaml.safe_load(target_path.read_text(encoding="utf-8")) or {}

    current = raw.get("filters") or []
    if not isinstance(current, list):
        current = []

    seen = {_filter_signature(f) for f in current if isinstance(f, dict)}
    merged = [copy.deepcopy(f) for f in current if isinstance(f, dict)]
    added = 0

    for lf in locked_filters:
        sig = _filter_signature(lf)
        if sig in seen:
            continue
        nlf = copy.deepcopy(lf)
        nlf["locked"] = True
        nlf["enabled"] = bool(default_enabled)
        nlf.setdefault("notes", "locked feature pool: preserved for threshold retuning")
        merged.append(nlf)
        seen.add(sig)
        added += 1

    raw["filters"] = merged
    raw["combination_mode"] = raw.get("combination_mode", "or")
    if added > 0 or not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return {"added": added, "total": len(merged)}
