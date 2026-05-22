"""Read trend signal funnel snapshots from live_monitor.db stats_15min."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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
