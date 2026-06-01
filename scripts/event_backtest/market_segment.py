"""Load canonical backtest windows from config/market_segment.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "config" / "market_segment.yaml"


def load_market_segments(
    path: Optional[Path | str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return segment id → {start_date, end_date, label, purpose, ...}."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"market_segment config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = data.get("segments") or []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        if not sid:
            continue
        if sid in out:
            raise ValueError(f"duplicate market_segment id: {sid!r}")
        out[str(sid)] = dict(row)
    if not out:
        raise ValueError(f"no segments in {p}")
    return out


def resolve_segment_run(
    run: Dict[str, Any],
    *,
    grid: Optional[Dict[str, Any]] = None,
    segments: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fill start_date / end_date from ``segment`` when omitted."""
    seg_id = run.get("segment")
    if not seg_id:
        if "start_date" not in run or "end_date" not in run:
            raise ValueError("run needs start_date/end_date or segment")
        return run

    if segments is None:
        path = (grid or {}).get("market_segment_path") or _DEFAULT_PATH
        segments = load_market_segments(path)

    seg_id = str(seg_id)
    if seg_id not in segments:
        known = ", ".join(sorted(segments))
        raise KeyError(f"unknown segment {seg_id!r}; known: {known}")

    seg = segments[seg_id]
    merged = dict(run)
    merged["start_date"] = str(seg["start_date"])
    merged["end_date"] = str(seg["end_date"])
    merged.setdefault("segment_label", seg.get("label"))
    merged.setdefault("segment_purpose", seg.get("purpose"))
    return merged


def expand_segment_matrix(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand ``segment_matrix`` × ``variants`` into flat runs (optional grid helper)."""
    matrix = grid.get("segment_matrix")
    if not matrix:
        return list(grid.get("runs") or [])

    seg_ids = matrix.get("segments")
    if seg_ids == "all":
        path = grid.get("market_segment_path") or _DEFAULT_PATH
        seg_ids = list(load_market_segments(path))
    if not isinstance(seg_ids, list) or not seg_ids:
        raise ValueError("segment_matrix.segments must be a list or 'all'")

    variants = matrix.get("variants") or []
    if not variants:
        raise ValueError("segment_matrix.variants required")

    out_runs: List[Dict[str, Any]] = []
    for vid in seg_ids:
        for var in variants:
            if not isinstance(var, dict):
                continue
            suffix = str(var.get("suffix") or var.get("variant") or "run")
            run: Dict[str, Any] = {
                "variant": f"{suffix}_{vid}",
                "segment": vid,
                **{k: v for k, v in var.items() if k not in ("suffix", "variant")},
            }
            out_dir = var.get("output_dir")
            if out_dir:
                run["output_dir"] = f"{out_dir.rstrip('/')}/{vid}"
            out_runs.append(run)
    return out_runs
