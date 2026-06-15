"""Open positions across trend / spot / multi-leg for CMS holdings view."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mlbot_console.services.exchange_balances import _parse_float
from mlbot_console.services.account_summary import (
    _trend_entry_qty_by_position,
    _trend_position_qty,
    _trend_unrealized_pnl_usdt,
    latest_close_prices,
)
from mlbot_console.services.db import query_rows
from mlbot_console.services.spot_pnl import compute_spot_order_pnl
from mlbot_console.services.symbols import is_all_symbols
from mlbot_console.services.trade_markers import _marker_id, _parse_ts

logger = logging.getLogger(__name__)

_OPEN_ORDER_STATUSES = frozenset(
    {"open", "new", "pending", "submitted", "partially_filled", "shadow"}
)


def _is_filled_row(row: Dict[str, Any]) -> bool:
    from mlbot_console.services.multileg_order_links import _is_filled_row as ml_filled

    return ml_filled(row)


def _normalize_side(raw: Any) -> str:
    side = str(raw or "").lower()
    if side in {"buy", "long"}:
        return "long"
    if side in {"sell", "short"}:
        return "short"
    return side or "long"


def _row_matches_strategy(row: Dict[str, Any], strategy: Optional[str]) -> bool:
    want = str(strategy or "").strip().lower()
    if not want:
        return True
    got = str(row.get("strategy") or row.get("strategy_id") or "").strip().lower()
    return got == want


def _trend_pending_exit_counts(db_path: Path, symbol: Optional[str]) -> Dict[str, int]:
    if not db_path.is_file():
        return {}
    where = " WHERE o.position_id IS NOT NULL AND lower(o.status) IN ({})".format(
        ",".join("?" for _ in _OPEN_ORDER_STATUSES)
    )
    params: tuple[Any, ...] = tuple(sorted(_OPEN_ORDER_STATUSES))
    if symbol and not is_all_symbols(symbol):
        where += " AND o.symbol = ?"
        params = (*params, symbol.upper())
    rows = query_rows(
        db_path,
        f"""
        SELECT o.position_id, o.side, p.side AS pos_side
        FROM orders o
        INNER JOIN positions p ON p.position_id = o.position_id
        {where}
        """,
        params,
    )
    out: Dict[str, int] = {}
    for row in rows:
        pid = str(row.get("position_id") or "")
        if not pid:
            continue
        pos_side = _normalize_side(row.get("pos_side"))
        o_side = str(row.get("side") or "").upper()
        closing = (pos_side == "long" and o_side == "SELL") or (
            pos_side == "short" and o_side == "BUY"
        )
        if not closing:
            continue
        out[pid] = out.get(pid, 0) + 1
    return out


def _multileg_pending_exit_counts(
    db_path: Path, symbol: Optional[str]
) -> Dict[str, int]:
    if not db_path.is_file():
        return {}
    where = """
        WHERE lower(trim(coalesce(status, ''))) IN ({})
          AND lower(trim(coalesce(purpose, ''))) IN (
              'take_profit', 'stop_loss', 'market_exit'
          )
          AND coalesce(trim(leg_id), '') != ''
    """.format(",".join("?" for _ in _OPEN_ORDER_STATUSES))
    params: tuple[Any, ...] = tuple(sorted(_OPEN_ORDER_STATUSES))
    if symbol and not is_all_symbols(symbol):
        where += " AND symbol = ?"
        params = (*params, symbol.upper())
    rows = query_rows(
        db_path,
        f"""
        SELECT leg_id
        FROM multi_leg_orders
        {where}
        """,
        params,
    )
    out: Dict[str, int] = {}
    for row in rows:
        leg = str(row.get("leg_id") or "").strip()
        if not leg:
            continue
        out[leg] = out.get(leg, 0) + 1
    return out


def _get_multileg_tp_sl_orders(
    db_path: Path, symbol: Optional[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Get TP/SL order details for each position.
    
    Returns dict mapping position_id to list of {order_type, price, order_id}.
    """
    if not db_path.is_file():
        return {}
    
    where = """
        WHERE lower(trim(coalesce(status, ''))) IN ({})
          AND lower(trim(coalesce(purpose, ''))) IN (
              'take_profit', 'stop_loss'
          )
          AND coalesce(trim(leg_id), '') != ''
    """.format(",".join("?" for _ in _OPEN_ORDER_STATUSES))
    params: tuple[Any, ...] = tuple(sorted(_OPEN_ORDER_STATUSES))
    if symbol and not is_all_symbols(symbol):
        where += " AND symbol = ?"
        params = (*params, symbol.upper())
    
    rows = query_rows(
        db_path,
        f"""
        SELECT leg_id, purpose, price, exchange_order_id
        FROM multi_leg_orders
        {where}
        ORDER BY leg_id, purpose
        """,
        params,
    )
    
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        leg = str(row.get("leg_id") or "").strip()
        if not leg:
            continue
        if leg not in out:
            out[leg] = []
        out[leg].append({
            "order_type": str(row.get("purpose") or "").upper(),
            "price": float(row.get("price") or 0),
            "order_id": str(row.get("exchange_order_id") or ""),
        })
    return out


def _multileg_open_leg_ids(db_path: Path, symbol: Optional[str]) -> Set[str]:
    """Return leg_ids that have an open position in multi_leg_positions table.
    
    Used to filter ghost entries from multi_leg_orders that are filled
    but whose position has already been closed/reconciled away.
    """
    if not db_path.is_file():
        return set()
    where = "WHERE lower(trim(coalesce(status, ''))) = 'open'"
    params: tuple[Any, ...] = ()
    if symbol and not is_all_symbols(symbol):
        where += " AND symbol = ?"
        params = (symbol.upper(),)
    rows = query_rows(
        db_path,
        f"""
        SELECT leg_id
        FROM multi_leg_positions
        {where}
        """,
        params,
    )
    return {str(row.get("leg_id") or "").strip() for row in rows if str(row.get("leg_id") or "").strip()}


def _trend_open_rows(
    db_path: Path,
    *,
    symbol: Optional[str],
    mark_prices: Dict[str, float],
    pending_exits: Dict[str, int],
) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    where = """
        WHERE (lower(trim(coalesce(status, ''))) = 'open' OR exit_time IS NULL)
          AND coalesce(trim(exit_reason), '') != 'exchange_sync_duplicate'
    """
    params: tuple[Any, ...] = ()
    if symbol and not is_all_symbols(symbol):
        where += " AND symbol = ?"
        params = (symbol.upper(),)
    rows = query_rows(
        db_path,
        f"""
        SELECT position_id, symbol, side, current_size, entry_time, exit_time,
               entry_price, status, strategy_id
        FROM positions
        {where}
        """,
        params,
    )
    entry_qty_by_pid = _trend_entry_qty_by_position(
        db_path, symbol if symbol and not is_all_symbols(symbol) else None
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("exit_time"):
            st = str(row.get("status") or "").lower()
            if st != "open":
                continue
        pid = str(row.get("position_id") or "")
        sym_u = str(row.get("symbol") or "").upper()
        qty = _trend_position_qty(row, entry_qty_by_pid)
        if qty <= 0:
            continue
        entry_px = float(row.get("entry_price") or 0.0)
        mark_px = float(mark_prices.get(sym_u) or 0.0)
        pos_row = dict(row)
        pos_row["pos_side"] = row.get("side")
        upnl = _trend_unrealized_pnl_usdt(
            pos_row, mark_px, entry_qty_by_pid=entry_qty_by_pid
        )
        entry_ts = _parse_ts(row.get("entry_time"))
        side = _normalize_side(row.get("side"))
        strat = str(row.get("strategy_id") or "trend").lower()
        out.append(
            {
                "position_id": pid,
                "symbol": sym_u,
                "scope": "trend",
                "strategy": strat,
                "side": side,
                "quantity": qty,
                "entry_price": entry_px,
                "mark_price": mark_px if mark_px > 0 else None,
                "unrealized_pnl_usdt": upnl,
                "entry_time": entry_ts,
                "pending_exit_orders": int(pending_exits.get(pid) or 0),
                "entry_marker_id": _marker_id("trend", "positions", f"{pid}:entry"),
            }
        )
    return out


def _spot_open_rows(
    db_path: Path,
    *,
    symbol: Optional[str],
    mark_prices: Dict[str, float],
) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    pnl_map = compute_spot_order_pnl(
        db_path,
        symbol=symbol if symbol and not is_all_symbols(symbol) else None,
        mark_prices=mark_prices,
    )
    out: List[Dict[str, Any]] = []
    for oid, rec in pnl_map.items():
        if rec.get("unrealized_pnl") is None:
            continue
        sym_u = str(rec.get("symbol") or "").upper()
        if symbol and not is_all_symbols(symbol) and sym_u != symbol.upper():
            continue
        qty = float(rec.get("lot_qty") or 0.0)
        if qty <= 0:
            continue
        strat = str(rec.get("strategy") or "spot").lower()
        out.append(
            {
                "position_id": oid,
                "symbol": sym_u,
                "scope": "spot",
                "strategy": strat,
                "side": "long",
                "quantity": qty,
                "entry_price": float(rec.get("entry_price") or 0.0),
                "mark_price": rec.get("mark_price"),
                "unrealized_pnl_usdt": float(rec.get("unrealized_pnl") or 0.0),
                "entry_time": rec.get("entry_time"),
                "pending_exit_orders": 0,
                "entry_marker_id": _marker_id("spot", "spot_orders", oid),
            }
        )
    return out


def _multileg_is_open_entry_row(row: Dict[str, Any]) -> bool:
    from mlbot_console.services.multileg_order_links import is_entry_row

    if not _is_filled_row(row):
        return False
    purpose = str(row.get("purpose") or "").lower()
    if is_entry_row(row):
        return True
    if purpose in {"inventory", "entry"}:
        return True
    return False


def _multileg_open_rows(
    db_path: Path,
    *,
    symbol: Optional[str],
    mark_prices: Dict[str, float],
    pending_exits: Dict[str, int],
) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    from mlbot_console.services.account_summary import _discover_symbols
    from mlbot_console.services.multileg_leg_pnl import (
        _order_key,
        _unrealized_pnl_usdt,
        pair_multileg_entry_exits,
    )
    from mlbot_console.services.multileg_order_links import (
        _is_filled_row as ml_is_filled,
        hydrate_multileg_fill_fields,
    )

    symbols: List[str]
    if symbol and not is_all_symbols(symbol):
        symbols = [symbol.upper()]
    else:
        symbols = _discover_symbols(
            trend_db=Path("/dev/null"),
            spot_db=Path("/dev/null"),
            multi_leg_db=db_path,
        )

    # Get TP/SL order details for all positions
    tp_sl_map = _get_multileg_tp_sl_orders(db_path, symbol)

    # Get active leg_ids from positions table (ground truth after reconcile)
    active_leg_ids = _multileg_open_leg_ids(db_path, symbol)

    out: List[Dict[str, Any]] = []
    for sym in symbols:
        raw = query_rows(
            db_path,
            """
            SELECT local_order_id, strategy, purpose, status, side, position_side,
                   filled_quantity, quantity, average_price, price, filled_at, created_at,
                   leg_id
            FROM multi_leg_orders
            WHERE symbol = ?
            """,
            (sym,),
        )
        hydrated: List[Dict[str, Any]] = []
        for row in raw:
            item = dict(row)
            item["order_id"] = row.get("local_order_id")
            hydrate_multileg_fill_fields(item)
            hydrated.append(item)

        closed_entries = {
            _order_key(entry)
            for entry, exit_row in pair_multileg_entry_exits(hydrated)
            if _order_key(entry) and ml_is_filled(exit_row)
        }
        mark_px = float(mark_prices.get(sym) or 0.0)

        for row in hydrated:
            if not _multileg_is_open_entry_row(row):
                continue
            oid = str(row.get("local_order_id") or "")
            if not oid or oid in closed_entries:
                continue
            qty = float(row.get("filled_quantity") or row.get("quantity") or 0.0)
            if qty <= 0:
                continue
            entry_px = float(row.get("average_price") or row.get("price") or 0.0)
            upnl = _unrealized_pnl_usdt(row, mark_px) if mark_px > 0 else None
            ps = str(row.get("position_side") or row.get("side") or "").upper()
            if ps in {"LONG", "BUY"}:
                side = "long"
            elif ps in {"SHORT", "SELL"}:
                side = "short"
            else:
                side = _normalize_side(row.get("side"))
            leg_key = str(row.get("leg_id") or oid)
            
            # Filter ghost positions: only show if leg_id is in active positions table
            # If positions table has data (active_leg_ids is non-empty), skip
            # entries whose leg_id is not tracked as open.
            if active_leg_ids and leg_key not in active_leg_ids:
                continue
            entry_ts = _parse_ts(row.get("filled_at")) or _parse_ts(
                row.get("created_at")
            )
            strat = str(row.get("strategy") or "multi_leg").lower()
            
            # Extract TP/SL info for this position
            tp_sl_orders = tp_sl_map.get(leg_key, [])
            tp_info = next((o for o in tp_sl_orders if o["order_type"] == "TAKE_PROFIT"), None)
            sl_info = next((o for o in tp_sl_orders if o["order_type"] == "STOP_LOSS"), None)
            
            out.append(
                {
                    "position_id": oid,
                    "leg": leg_key,
                    "symbol": sym,
                    "scope": "multi_leg",
                    "strategy": strat,
                    "side": side,
                    "quantity": qty,
                    "entry_price": entry_px,
                    "mark_price": mark_px if mark_px > 0 else None,
                    "unrealized_pnl_usdt": upnl,
                    "entry_time": entry_ts,
                    "pending_exit_orders": int(pending_exits.get(leg_key) or 0),
                    "entry_marker_id": _marker_id("multi_leg", "multi_leg_orders", oid),
                    # TP/SL fields
                    "tp_price": tp_info["price"] if tp_info else None,
                    "tp_order_id": tp_info["order_id"] if tp_info else None,
                    "sl_price": sl_info["price"] if sl_info else None,
                    "sl_order_id": sl_info["order_id"] if sl_info else None,
                }
            )
    return out


def _exchange_position_map(
    exchange_ledger: Optional[Dict[str, Any]],
) -> Dict[tuple, float]:
    """Extract per-(symbol, side) net position from exchange ledger.

    Returns {(symbol_upper, 'long'|'short'): abs(positionAmt)}
    """
    out: Dict[tuple, float] = {}
    if not exchange_ledger:
        return out
    for acct in exchange_ledger.get("accounts") or []:
        for pos in acct.get("exchange_open_positions") or []:
            sym = str(pos.get("symbol") or "").upper()
            side = "long" if float(pos.get("position_amt") or 0) > 0 else "short"
            qty = abs(float(pos.get("position_amt") or 0))
            if sym and qty > 0:
                key = (sym, side)
                out[key] = out.get(key, 0.0) + qty
    return out


def _enrich_with_exchange_legs(
    rows: List[Dict[str, Any]],
    exchange_ledger: Optional[Dict[str, Any]],
) -> None:
    """Enrich local open positions with exchange-level margin/leverage data."""
    if not exchange_ledger:
        return

    ex_legs: Dict[tuple, Dict[str, Any]] = {}
    for acct in exchange_ledger.get("accounts") or []:
        scope = acct.get("scope", "")
        for pos in acct.get("exchange_open_positions") or []:
            sym = str(pos.get("symbol") or "").upper()
            side = "long" if float(pos.get("position_amt") or 0) > 0 else "short"
            key = (scope, sym, side)
            if key not in ex_legs:
                ex_legs[key] = pos

    # Group local rows sharing the same exchange leg for pro-rata margin split.
    groups: Dict[tuple, List[int]] = {}
    for idx, row in enumerate(rows):
        scope = str(row.get("scope") or "")
        sym = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or "long").lower()
        groups.setdefault((scope, sym, side), []).append(idx)

    for (scope, sym, side), indices in groups.items():
        key = (scope, sym, side)
        ex_pos = ex_legs.get(key)
        if not ex_pos:
            for acct_scope in ["trend", "multi_leg"]:
                fallback_key = (acct_scope, sym, side)
                if fallback_key in ex_legs:
                    ex_pos = ex_legs[fallback_key]
                    break
        if not ex_pos:
            continue

        total_margin = _parse_float(ex_pos.get("initial_margin_usdt"))
        total_qty = sum(float(rows[i].get("quantity") or 0.0) for i in indices)
        n = len(indices)
        for i in indices:
            row = rows[i]
            row_qty = float(row.get("quantity") or 0.0)
            share = (row_qty / total_qty) if total_qty > 0 else (1.0 / max(n, 1))
            row["exchange_leverage"] = ex_pos.get("leverage")
            row["exchange_notional_usdt"] = (
                round(_parse_float(ex_pos.get("notional_usdt")) * share, 4)
                if ex_pos.get("notional_usdt") is not None
                else None
            )
            if total_margin > 0:
                row["exchange_initial_margin_usdt"] = round(total_margin * share, 4)
                row["exchange_margin_allocated"] = n > 1
            else:
                row["exchange_initial_margin_usdt"] = ex_pos.get("initial_margin_usdt")
            maint = _parse_float(ex_pos.get("maint_margin_usdt"))
            row["exchange_maint_margin_usdt"] = (
                round(maint * share, 4) if maint > 0 and n > 1 else ex_pos.get("maint_margin_usdt")
            )
            row["exchange_liquidation_price"] = ex_pos.get("liquidation_price")
            row["exchange_margin_type"] = ex_pos.get("margin_type")



def _exchange_has_position(
    ex_map: Dict[tuple, float],
    symbol: str,
    side: str,
    min_qty: float = 0.0001,
) -> bool:
    """True when exchange reports a non-dust position for (symbol, side)."""
    return ex_map.get((symbol.upper(), side.lower()), 0.0) >= min_qty


def collect_open_positions(
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    symbol: str,
    scopes: List[str],
    limit: int = 200,
    feature_bus_root: Optional[Path] = None,
    strategy: Optional[str] = None,
    exchange_ledger: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    scope_set = {s.strip().lower() for s in scopes if s.strip()}
    if not scope_set:
        scope_set = {"trend", "spot", "multi_leg"}

    sym_filter: Optional[str] = None
    if symbol and not is_all_symbols(symbol):
        sym_filter = symbol.upper()

    marks: Dict[str, float] = {}
    if feature_bus_root is not None and feature_bus_root.is_dir():
        if sym_filter:
            marks = latest_close_prices(feature_bus_root, [sym_filter])
        else:
            from mlbot_console.services.account_summary import _discover_symbols

            syms = _discover_symbols(
                trend_db=trend_db,
                spot_db=spot_db,
                multi_leg_db=multi_leg_db,
            )
            marks = latest_close_prices(feature_bus_root, syms)

    merged: List[Dict[str, Any]] = []
    if "trend" in scope_set:
        # P4: Try TTS list_open_projections first (read-only SQLite projection).
        # Fallback to _trend_open_rows + dedup if TTS unavailable.
        # Distinguish "successfully got []" (zero holdings) from "projection failed".
        _projection_ok = False
        try:
            from src.order_management.storage import Storage
            from src.order_management.trend_position_truth_sync import (
                TrendPositionTruthSync,
            )
            _storage = Storage(str(trend_db))
            projection = TrendPositionTruthSync.list_open_projections(
                _storage, symbol=sym_filter,
            )
            _projection_ok = True  # list_open_projections succeeded (even if [])
            if projection:
                pending = _trend_pending_exit_counts(trend_db, sym_filter)
            for row in projection:
                # Normalize entry_time to int timestamp (same as _trend_open_rows)
                row["entry_time"] = _parse_ts(row.get("entry_time")) or 0
                sym = str(row.get("symbol") or "")
                mark = marks.get(sym)
                if mark and mark > 0 and float(row.get("entry_price") or 0) > 0:
                    qty = float(row.get("quantity") or 0)
                    side = str(row.get("side") or "long")
                    entry = float(row["entry_price"])
                    if side == "long":
                        row["unrealized_pnl_usdt"] = (mark - entry) * qty
                    else:
                        row["unrealized_pnl_usdt"] = (entry - mark) * qty
                    row["mark_price"] = mark
                pid = str(row.get("position_id") or "")
                row["pending_exit_orders"] = int(pending.get(pid) or 0)
                row["entry_marker_id"] = _marker_id(
                    "trend", "positions", pid,
                )
            merged.extend(projection)
            logger.debug(
                "Used TTS projection for trend open positions (rows=%d)",
                len(projection),
            )
        except Exception:
            logger.warning("P4 TTS projection path failed, falling back", exc_info=True)

        if not _projection_ok:
            pending = _trend_pending_exit_counts(trend_db, sym_filter)
            merged.extend(
                _trend_open_rows(
                    trend_db,
                    symbol=sym_filter,
                    mark_prices=marks,
                    pending_exits=pending,
                )
            )
    if "spot" in scope_set:
        merged.extend(_spot_open_rows(spot_db, symbol=sym_filter, mark_prices=marks))
    if "multi_leg" in scope_set:
        pending = _multileg_pending_exit_counts(multi_leg_db, sym_filter)
        merged.extend(
            _multileg_open_rows(
                multi_leg_db,
                symbol=sym_filter,
                mark_prices=marks,
                pending_exits=pending,
            )
        )

    if strategy:
        merged = [r for r in merged if _row_matches_strategy(r, strategy)]

    # ── Cross-reference with exchange: drop local entries for symbols that
    #     have zero exchange position (stale / fully-closed but unpaired). ──
    if exchange_ledger:
        ex_map = _exchange_position_map(exchange_ledger)
        logger.debug(
            "exchange_ledger cross-validation: ex_map size=%d, merged rows=%d",
            len(ex_map),
            len(merged),
        )
        if ex_map:
            filtered: List[Dict[str, Any]] = []
            dropped_count = 0
            for row in merged:
                scope = str(row.get("scope") or "")
                if scope == "spot":
                    # Spot holdings are always real; trust local computation.
                    filtered.append(row)
                    continue
                sym = str(row.get("symbol") or "")
                side = str(row.get("side") or "long")
                has_pos = _exchange_has_position(ex_map, sym, side)
                if not has_pos:
                    dropped_count += 1
                    logger.debug(
                        "Dropping stale position: scope=%s symbol=%s side=%s (exchange=0)",
                        scope,
                        sym,
                        side,
                    )
                else:
                    filtered.append(row)
            logger.debug(
                "exchange_ledger filter: dropped=%d, kept=%d",
                dropped_count,
                len(filtered),
            )
            merged = filtered
        else:
            logger.warning("exchange_ledger provided but ex_map is empty!")
    else:
        logger.warning("exchange_ledger is None - skipping cross-validation")

    # ── Deduplicate: same (scope, symbol, side) with identical qty → keep
    #     most recent entry_time (fixes exchange-sync + bootstrap dupes).
    #     P4 门禁: P1 唯一写入口 + P3 连续 3 日无 duplicate_position_row_closed > 0
    #     满足后可删除此 dedup 逻辑。 ──
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for row in merged:
        scope = str(row.get("scope") or "")
        # spot: batch-level; multi_leg: leg-level — trust upstream dedup.
        if scope not in ("trend",):
            continue
        sym = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or "long")
        dedup_key = (scope, sym, side)
        ts = int(row.get("entry_time") or 0)
        existing = dedup.get(dedup_key)
        if existing is None or ts > int(existing.get("entry_time") or 0):
            dedup[dedup_key] = row
    if dedup:
        keep_ids = {str(r.get("position_id") or "") for r in dedup.values()}
        merged = [
            r
            for r in merged
            if str(r.get("scope") or "") != "trend"
            or str(r.get("position_id") or "") in keep_ids
        ]

    merged.sort(
        key=lambda r: (
            int(r.get("entry_time") or 0),
            str(r.get("position_id") or ""),
        ),
        reverse=True,
    )
    
    # Enrich with exchange margin/leverage data
    _enrich_with_exchange_legs(merged, exchange_ledger)
    
    return merged[: max(int(limit), 1)]
