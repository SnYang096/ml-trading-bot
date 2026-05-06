"""Layer statistics for dashboard chips (full scan; separate API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .scan import scan_flat_runs, scan_rolling_ledgers


def _layer_stats_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按策略 tab 统计 pipeline / mode / run_kind。"""

    def stats_one(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        pipe: Dict[str, int] = {}
        mode: Dict[str, int] = {}
        rk: Dict[str, int] = {}
        for r in subset:
            pd = r.get("pipeline_dir") or "?"
            pipe[pd] = pipe.get(pd, 0) + 1
            m = str(r.get("mode") or "?")
            mode[m] = mode.get(m, 0) + 1
            k = str(r.get("run_kind") or "?")
            rk[k] = rk.get(k, 0) + 1
        return {"pipeline": pipe, "mode": mode, "run_kind": rk}

    out: Dict[str, Any] = {"__all__": stats_one(rows)}
    strats = sorted(set(r.get("strategy") or "" for r in rows))
    for s in strats:
        out[s] = stats_one([r for r in rows if (r.get("strategy") or "") == s])
    return out


def build_layer_stats_for_dashboard(
    results_root: Path,
    *,
    strategy_filter: Optional[str],
    q: Optional[str],
) -> Dict[str, Any]:
    """与看板相同的扫描范围；统计 JSON 经 ``/api/dashboard-stats.json`` 返回，避免塞进 HTML 引发解析错误。"""
    results_root = results_root.resolve()
    rolling_rows = scan_rolling_ledgers(results_root, strategy_filter=None, q=q)
    flat_rows = scan_flat_runs(results_root, strategy_filter=None, q=q)
    for r in rolling_rows:
        r["run_kind"] = "rolling"
    for r in flat_rows:
        r["run_kind"] = "flat"
    rows = rolling_rows + flat_rows
    rows.sort(key=lambda r: r.get("mtime") or 0.0, reverse=True)
    return _layer_stats_payload(rows)
