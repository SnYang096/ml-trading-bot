"""Spot buy eligibility summary from feature bus + spot order DB (P2 fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.db import query_one, query_rows
from app.services.feature_overlay import load_feature_overlay


def _pending_spot_orders(spot_db: Path, symbol: str) -> List[Dict[str, Any]]:
    if not spot_db.is_file():
        return []
    sql = """
        SELECT order_id, side, status, price, quantity, created_at, updated_at
        FROM spot_orders
        WHERE symbol = ?
          AND LOWER(TRIM(status)) IN ('pending', 'open', 'new', 'submitted')
        ORDER BY created_at DESC
        LIMIT 20
    """
    return query_rows(spot_db, sql, (symbol.upper(),))


def spot_eligibility_summary(
    *,
    feature_bus_root: Path,
    spot_db: Path,
    symbol: str,
    timeframe: str = "2h",
) -> Dict[str, Any]:
    sym = symbol.upper()
    blockers: List[str] = []
    overlay = load_feature_overlay(
        feature_bus_root,
        sym,
        timeframe,
        column="weekly_ema_200_position",
    )
    weekly_pos = overlay.get("latest")
    if weekly_pos is not None and float(weekly_pos) < 0:
        blockers.append("weekly_ema_200_position_below_zero")
    elif weekly_pos is None and not overlay.get("available"):
        blockers.append("weekly_ema_200_position_unavailable")

    pending = _pending_spot_orders(spot_db, sym)
    if pending:
        blockers.append("pending_spot_orders")

    filled_today = 0
    if spot_db.is_file():
        row = query_one(
            spot_db,
            """
            SELECT COUNT(*) AS cnt FROM spot_orders
            WHERE symbol = ?
              AND LOWER(TRIM(status)) IN ('filled', 'closed')
              AND side = 'buy'
            """,
            (sym,),
        )
        if row:
            filled_today = int(row.get("cnt") or 0)

    return {
        "symbol": sym,
        "weekly_ema_200_position": weekly_pos,
        "feature_available": overlay.get("available", False),
        "blockers": blockers,
        "pending_orders": pending,
        "filled_buy_count": filled_today,
        "can_buy": len(blockers) == 0,
        "notes": [
            "deploy_schedule and day_limit require runtime logs or snapshots (P2 fallback).",
        ],
    }
