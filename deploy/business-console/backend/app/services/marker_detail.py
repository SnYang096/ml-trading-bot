"""Resolve marker id to linked DB rows for the detail drawer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.services.db import query_one


def parse_marker_id(marker_id: str) -> Optional[Dict[str, str]]:
    parts = marker_id.split(":", 2)
    if len(parts) != 3:
        return None
    return {"scope": parts[0], "source": parts[1], "key": parts[2]}


def marker_detail(
    marker_id: str,
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
) -> Dict[str, Any]:
    parsed = parse_marker_id(marker_id)
    if not parsed:
        return {"found": False, "marker_id": marker_id, "error": "invalid_id"}
    scope = parsed["scope"]
    source = parsed["source"]
    key = parsed["key"]

    if scope == "trend" and trend_db.is_file():
        return _trend_detail(trend_db, source, key, marker_id)
    if scope == "spot" and spot_db.is_file():
        return _spot_detail(spot_db, key, marker_id)
    if scope == "multi_leg" and multi_leg_db.is_file():
        return _multi_leg_detail(multi_leg_db, source, key, marker_id)
    return {"found": False, "marker_id": marker_id, "error": "db_missing"}


def _trend_detail(db: Path, source: str, key: str, marker_id: str) -> Dict[str, Any]:
    if source == "positions":
        base, evt = key.rsplit(":", 1) if ":" in key else (key, "entry")
        row = query_one(
            db,
            "SELECT * FROM positions WHERE position_id = ?",
            (base,),
        )
        return {"found": row is not None, "marker_id": marker_id, "table": "positions", "row": row, "event": evt}
    if source == "orders":
        row = query_one(db, "SELECT * FROM orders WHERE order_id = ?", (key,))
        return {"found": row is not None, "marker_id": marker_id, "table": "orders", "row": row}
    if source == "position_operations":
        row = query_one(
            db,
            "SELECT * FROM position_operations WHERE operation_id = ?",
            (key,),
        )
        return {
            "found": row is not None,
            "marker_id": marker_id,
            "table": "position_operations",
            "row": row,
        }
    return {"found": False, "marker_id": marker_id, "error": "unknown_source"}


def _spot_detail(db: Path, key: str, marker_id: str) -> Dict[str, Any]:
    row = query_one(db, "SELECT * FROM spot_orders WHERE order_id = ?", (key,))
    return {"found": row is not None, "marker_id": marker_id, "table": "spot_orders", "row": row}


def _multi_leg_detail(db: Path, source: str, key: str, marker_id: str) -> Dict[str, Any]:
    if source == "multi_leg_orders":
        row = query_one(
            db,
            "SELECT * FROM multi_leg_orders WHERE local_order_id = ?",
            (key,),
        )
        return {
            "found": row is not None,
            "marker_id": marker_id,
            "table": "multi_leg_orders",
            "row": row,
        }
    if source == "multi_leg_execution_reports":
        row = query_one(
            db,
            "SELECT * FROM multi_leg_execution_reports WHERE event_id = ?",
            (key,),
        )
        return {
            "found": row is not None,
            "marker_id": marker_id,
            "table": "multi_leg_execution_reports",
            "row": row,
        }
    return {"found": False, "marker_id": marker_id, "error": "unknown_source"}
