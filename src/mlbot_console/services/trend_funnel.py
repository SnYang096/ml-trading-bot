"""Read signal funnel snapshots from live_monitor.db stats_15min."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services.strategy_registry import (
    layer_for_funnel_filter,
    strategy_account_layer,
)


def fetch_funnel_snapshots(
    db_path: Path,
    *,
    symbol: str = "",
    limit: int = 96,
) -> List[Dict[str, Any]]:
    """Return recent stats_15min rows (newest first), parsing by_strategy JSON."""
    if not db_path.is_file():
        return []
    sym = str(symbol or "").strip().upper()
    sql = """
        SELECT timestamp, symbol, bars_processed, direction_assigned,
               gate_passed, entry_filter_passed, evidence_passed,
               pcm_selected, orders_placed, by_strategy, regime
        FROM stats_15min
    """
    params: tuple = ()
    if sym and sym not in ("*", "ALL"):
        sql += " WHERE UPPER(symbol) = ?"
        params = (sym,)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params = params + (int(limit),)

    from mlbot_console.services.db import query_rows

    rows = query_rows(db_path, sql, params)
    out: List[Dict[str, Any]] = []
    for row in rows:
        bys = row.get("by_strategy")
        if isinstance(bys, str):
            try:
                bys = json.loads(bys) if bys else {}
            except json.JSONDecodeError:
                bys = {}
        if not isinstance(bys, dict):
            bys = {}
        out.append(
            {
                "timestamp": row.get("timestamp"),
                "symbol": row.get("symbol") or "",
                "bars_processed": int(row.get("bars_processed") or 0),
                "direction_assigned": int(row.get("direction_assigned") or 0),
                "gate_passed": int(row.get("gate_passed") or 0),
                "entry_filter_passed": int(row.get("entry_filter_passed") or 0),
                "evidence_passed": int(row.get("evidence_passed") or 0),
                "pcm_selected": int(row.get("pcm_selected") or 0),
                "orders_placed": int(row.get("orders_placed") or 0),
                "pcm_regime_label": row.get("regime"),
                "by_strategy": bys,
            }
        )
    return out


def aggregate_funnel_by_strategy(
    snapshots: List[Dict[str, Any]],
    *,
    symbol: str = "",
    account_layer: str = "",
    strategy: str = "",
) -> Dict[str, Dict[str, int]]:
    """Sum regime/prefilter/direction/gate counters per strategy across snapshots."""
    sym = str(symbol or "").strip().upper()
    layer_filter = layer_for_funnel_filter(account_layer, strategy)
    strat_filter = str(strategy or "").strip().lower()
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    int_keys = (
        "evals",
        "regime_passed",
        "regime_denied",
        "prefilter_passed",
        "prefilter_denied",
        "direction",
        "gate_passed",
        "gate_rejected",
        "entry_filter_passed",
        "signals",
        "pcm_selected",
        "orders",
    )
    for snap in snapshots:
        snap_sym = str(snap.get("symbol") or "").upper()
        if sym and sym not in ("*", "ALL") and snap_sym != sym:
            continue
        bys = snap.get("by_strategy") or {}
        if not isinstance(bys, dict):
            continue
        for strat, raw in bys.items():
            sid = str(strat).lower()
            if strat_filter and sid != strat_filter:
                continue
            if layer_filter and strategy_account_layer(sid) != layer_filter:
                continue
            if not isinstance(raw, dict):
                continue
            bucket = totals[sid]
            for k in int_keys:
                v = raw.get(k)
                if isinstance(v, (int, float)):
                    bucket[k] += int(v)
    return {k: dict(v) for k, v in totals.items()}
