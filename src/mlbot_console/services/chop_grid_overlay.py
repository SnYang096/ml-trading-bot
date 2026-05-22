"""Chop grid price ladder + regime bands for Trade Map."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from mlbot_console.services.db import query_rows
from mlbot_console.services.feature_overlay import _resolve_feature_path
from mlbot_console.services.multileg_order_links import leg_group_key, row_group_key

_TP_RE = re.compile(r"_(L|S)(\d+)_tp$", re.I)
_DEFAULT_MAX_LEVELS = 2
_CHOP_ENTRY_MIN = 0.50


def _parse_ts(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)):
            v = float(raw)
            if v > 1e12:
                return int(v / 1000)
            if v > 1e9:
                return int(v)
        from datetime import datetime, timezone

        s = str(raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (TypeError, ValueError):
        return None


def _load_engine_state(engine_data_root: Path, symbol: str) -> Optional[Dict[str, Any]]:
    sym = symbol.upper()
    path = engine_data_root / "multi_leg_live" / "state" / f"chop_grid_{sym}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _orders_for_symbol(db_path: Path, symbol: str) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    sql = """
        SELECT local_order_id, leg_id, side, purpose, status, price, average_price,
               filled_quantity, created_at, filled_at, strategy
        FROM multi_leg_orders
        WHERE symbol = ? AND lower(strategy) = 'chop_grid'
    """
    rows = query_rows(db_path, sql, (symbol.upper(),))
    for row in rows:
        row["order_id"] = row.get("local_order_id")
    return rows


def _inventory_legs(
    engine_state: Optional[Dict[str, Any]],
    db_path: Path,
    symbol: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if engine_state:
        for inv in engine_state.get("inventory") or []:
            leg_id = str(inv.get("leg_id") or "").strip()
            if not leg_id:
                continue
            out[leg_id] = {
                "leg_id": leg_id,
                "side": str(inv.get("side") or "").upper(),
                "entry_price": float(inv.get("entry_price") or 0),
                "quantity": float(inv.get("quantity") or 0),
                "source": "engine",
            }
    if db_path.is_file():
        sql = """
            SELECT leg_id, side, entry_price, quantity
            FROM multi_leg_positions
            WHERE symbol = ? AND lower(strategy) = 'chop_grid'
              AND lower(trim(coalesce(status, ''))) = 'open'
        """
        for row in query_rows(db_path, sql, (symbol.upper(),)):
            leg_id = str(row.get("leg_id") or "").strip()
            if not leg_id:
                continue
            out.setdefault(
                leg_id,
                {
                    "leg_id": leg_id,
                    "side": str(row.get("side") or "").upper(),
                    "entry_price": float(row.get("entry_price") or 0),
                    "quantity": float(row.get("quantity") or 0),
                    "source": "db",
                },
            )
    return out


def _order_by_leg(orders: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_leg: Dict[str, List[Dict[str, Any]]] = {}
    for row in orders:
        for field in ("leg_id", "order_id", "local_order_id"):
            val = str(row.get(field) or "")
            if not val:
                continue
            by_leg.setdefault(val, []).append(row)
            gk = leg_group_key(val)
            if gk:
                m = _TP_RE.search(val) or re.search(r"_(L|S)(\d+)$", val, re.I)
                if m:
                    base = f"{gk}_{m.group(1).upper()}{m.group(2)}"
                    by_leg.setdefault(base, []).append(row)
    return by_leg


def _entry_status(
    rows: List[Dict[str, Any]],
    inventory: Dict[str, Dict[str, Any]],
    *,
    leg_key: str = "",
) -> str:
    if leg_key:
        for lid in inventory:
            if str(lid).endswith(leg_key):
                return "filled"
    leg_id = ""
    for row in rows:
        oid = str(row.get("order_id") or "")
        purpose = str(row.get("purpose") or "").lower()
        if "take_profit" in purpose or _TP_RE.search(oid):
            continue
        leg_id = str(row.get("leg_id") or oid)
        break
    if leg_id and leg_id in inventory:
        return "filled"
    for row in rows:
        purpose = str(row.get("purpose") or "").lower()
        if "take_profit" in purpose:
            continue
        status = str(row.get("status") or "").lower()
        filled = float(row.get("filled_quantity") or 0)
        if status in {"filled", "closed"} or filled > 0:
            return "filled"
        if status in {"open", "pending", "new", "submitted", "shadow"}:
            return "open"
    return "missing"


def _tp_info(rows: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    for row in rows:
        purpose = str(row.get("purpose") or "").lower()
        oid = str(row.get("order_id") or "")
        if "take_profit" not in purpose and not _TP_RE.search(oid):
            continue
        price = row.get("average_price") or row.get("price")
        try:
            px = float(price) if price is not None else None
        except (TypeError, ValueError):
            px = None
        status = str(row.get("status") or "").lower()
        if px is not None and px == px:
            return px, status or "unknown"
    return None, ""


def _level_rows(
    *,
    center: float,
    spacing: float,
    max_levels: int,
    orders: List[Dict[str, Any]],
    inventory: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_leg = _order_by_leg(orders)
    levels: List[Dict[str, Any]] = []
    for side_code, side_name in (("L", "long"), ("S", "short")):
        for idx in range(1, max_levels + 1):
            leg_key = f"_{side_code}{idx}"
            price = (
                center - spacing * idx if side_code == "L" else center + spacing * idx
            )
            matched: List[Dict[str, Any]] = []
            seen_rows: Set[int] = set()
            inv_row = None

            def extend_unique(rows: List[Dict[str, Any]]) -> None:
                for row in rows:
                    row_id = id(row)
                    if row_id in seen_rows:
                        continue
                    seen_rows.add(row_id)
                    matched.append(row)

            for lid, inv in inventory.items():
                if lid.endswith(leg_key):
                    inv_row = inv
                    extend_unique(by_leg.get(lid, []))
            for oid, rows in by_leg.items():
                if oid.endswith(leg_key):
                    extend_unique(rows)
            entry_px = None
            if inv_row and inv_row.get("entry_price"):
                entry_px = float(inv_row["entry_price"])
            else:
                for row in matched:
                    if "take_profit" in str(row.get("purpose") or "").lower():
                        continue
                    val = row.get("average_price") or row.get("price")
                    if val is not None:
                        entry_px = float(val)
                        break
            tp_px, tp_st = _tp_info(matched)
            levels.append(
                {
                    "leg": f"{side_code}{idx}",
                    "side": side_name,
                    "grid_price": round(price, 6),
                    "entry_price": entry_px,
                    "entry_status": _entry_status(
                        matched, inventory, leg_key=leg_key
                    ),
                    "tp_price": tp_px,
                    "tp_status": tp_st,
                }
            )
    return levels


def _batch_from_state(
    state: Dict[str, Any],
    orders: List[Dict[str, Any]],
    inventory: Dict[str, Dict[str, Any]],
    *,
    max_levels: int,
) -> Optional[Dict[str, Any]]:
    center = float(state.get("center") or 0)
    spacing = float(state.get("spacing") or 0)
    if center <= 0 or spacing <= 0:
        return None
    grid_id = str(state.get("grid_id") or "")
    batch_orders = [
        r for r in orders if row_group_key(r) == (leg_group_key(grid_id) or grid_id)
    ]
    if not batch_orders and grid_id:
        batch_orders = [r for r in orders if grid_id in str(r.get("order_id") or "")]
    grid_group = leg_group_key(grid_id) or grid_id
    batch_inventory = {k: v for k, v in inventory.items() if grid_group and grid_group in k}
    return {
        "grid_id": grid_id,
        "center": center,
        "spacing": spacing,
        "active": bool(state.get("active")),
        "levels": _level_rows(
            center=center,
            spacing=spacing,
            max_levels=max_levels,
            orders=batch_orders,
            inventory=batch_inventory,
        ),
    }


def load_chop_grid_map_overlay(
    *,
    multi_leg_db: Path,
    engine_data_root: Optional[Path],
    symbol: str,
    max_levels_per_side: int = _DEFAULT_MAX_LEVELS,
) -> Dict[str, Any]:
    """Return {batches: [...]} for active/recent chop_grid ladders."""
    sym = symbol.upper()
    orders = _orders_for_symbol(multi_leg_db, sym)
    state = (
        _load_engine_state(engine_data_root, sym) if engine_data_root is not None else None
    )
    inventory = _inventory_legs(state, multi_leg_db, sym)
    batches: List[Dict[str, Any]] = []
    seen_batches: Set[str] = set()

    if state:
        batch = _batch_from_state(
            state, orders, inventory, max_levels=max_levels_per_side
        )
        if batch and batch.get("grid_id"):
            batches.append(batch)
            seen_batches.add(str(batch["grid_id"]))

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in orders:
        gk = row_group_key(row)
        if not gk:
            continue
        by_group.setdefault(gk, []).append(row)

    for gk, group_orders in by_group.items():
        if gk in seen_batches:
            continue
        inv_in_group = {k: v for k, v in inventory.items() if gk in k}
        if not inv_in_group and not any(
            str(r.get("status") or "").lower() in {"open", "pending", "submitted", "new"}
            for r in group_orders
        ):
            continue
        prices = []
        for row in group_orders:
            for key in ("price", "average_price"):
                val = row.get(key)
                if val is not None:
                    try:
                        prices.append(float(val))
                    except (TypeError, ValueError):
                        pass
        if not prices:
            continue
        center = sum(prices) / len(prices)
        spacing = 0.0
        if len(prices) >= 2:
            spacing = (max(prices) - min(prices)) / max(len(prices) - 1, 1)
        if spacing <= 0:
            spacing = center * 0.01
        batches.append(
            {
                "grid_id": gk,
                "center": center,
                "spacing": spacing,
                "active": False,
                "levels": _level_rows(
                    center=center,
                    spacing=spacing,
                    max_levels=max_levels_per_side,
                    orders=group_orders,
                    inventory=inv_in_group,
                ),
            }
        )
        seen_batches.add(gk)

    return {"batches": batches}


def load_chop_regime_regions(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    entry_min: float = _CHOP_ENTRY_MIN,
) -> List[Dict[str, Any]]:
    """Time spans where bpc_semantic_chop >= entry_min (for chart shading)."""
    import pandas as pd

    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    if path is None:
        return []
    try:
        df = pd.read_parquet(
            path, columns=["timestamp", "bpc_semantic_chop", "semantic_chop"]
        )
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return []
    col = None
    for name in ("bpc_semantic_chop", "semantic_chop"):
        if name in df.columns:
            col = name
            break
    if col is None:
        return []
    if "timestamp" not in df.columns:
        return []
    df = df.sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if df.empty:
        return []

    regions: List[Dict[str, Any]] = []
    in_region = False
    region_start: Optional[int] = None
    prev_ts: Optional[int] = None
    for _, row in df.iterrows():
        ts = _parse_ts(row["timestamp"])
        if ts is None:
            continue
        try:
            val = float(row[col])
        except (TypeError, ValueError):
            val = 0.0
        active = val >= entry_min
        if active and not in_region:
            in_region = True
            region_start = ts
        elif not active and in_region and region_start is not None:
            regions.append(
                {
                    "start": region_start,
                    "end": prev_ts if prev_ts is not None else ts,
                    "entry_min": entry_min,
                    "feature": col,
                }
            )
            in_region = False
            region_start = None
        prev_ts = ts
    if in_region and region_start is not None and prev_ts is not None:
        regions.append(
            {
                "start": region_start,
                "end": prev_ts,
                "entry_min": entry_min,
                "feature": col,
            }
        )
    return regions
