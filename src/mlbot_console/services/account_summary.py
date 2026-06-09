"""Account-level PnL and order stats for the business console."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from mlbot_console.services.db import query_rows
from mlbot_console.services.spot_pnl import compute_spot_order_pnl, spot_holdings_from_orders
from mlbot_console.services.strategy_registry import (
    default_spot_strategy_id,
    spot_strategy_ids,
)
from mlbot_console.services.exchange_balances import build_exchange_ledger
from mlbot_console.services.trade_links import multi_leg_trade_links
from mlbot_console.services.trade_markers import _parse_ts

_SCOPE_LABELS = {
    "trend": "B·Trend",
    "spot": "A·Spot",
    "multi_leg": "C·Multi-leg",
}

def _is_all_symbols(symbol: str) -> bool:
    return str(symbol or "").strip().upper() in {"", "*", "ALL", "__ALL__"}


def latest_close_prices(feature_bus_root: Path, symbols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    bars_root = feature_bus_root / "bars_1min"
    if not bars_root.is_dir():
        return out
    for sym in symbols:
        s = str(sym).upper()
        path = bars_root / f"{s}.parquet"
        if not path.is_file():
            continue
        try:
            df = pd.read_parquet(path, columns=["close"])
            if df.empty:
                continue
            out[s] = float(df["close"].iloc[-1])
        except Exception:
            continue
    return out


def _discover_symbols(
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
) -> List[str]:
    found: set[str] = set()
    for db, table, col in (
        (trend_db, "positions", "symbol"),
        (trend_db, "orders", "symbol"),
        (spot_db, "spot_orders", "symbol"),
        (multi_leg_db, "multi_leg_orders", "symbol"),
    ):
        if not db.is_file():
            continue
        try:
            for row in query_rows(
                db, f"SELECT DISTINCT {col} AS symbol FROM {table} WHERE {col} IS NOT NULL"
            ):
                sym = str(row.get("symbol") or "").upper()
                if sym:
                    found.add(sym)
        except Exception:
            continue
    return sorted(found)


def _link_pnl_usdt(entry_row: Dict[str, Any], exit_row: Dict[str, Any]) -> Optional[float]:
    qty = float(entry_row.get("filled_quantity") or entry_row.get("quantity") or 0.0)
    if qty <= 0:
        return None
    entry_px = float(entry_row.get("average_price") or entry_row.get("price") or 0.0)
    exit_px = float(exit_row.get("average_price") or exit_row.get("price") or 0.0)
    if entry_px <= 0 or exit_px <= 0:
        return None
    side = str(entry_row.get("side") or "").lower()
    if side in {"buy", "long"}:
        return (exit_px - entry_px) * qty
    return (entry_px - exit_px) * qty


def _trend_entry_qty_by_position(trend_db: Path, sym: Optional[str]) -> Dict[str, float]:
    """Best-effort entry size for PnL when positions.current_size is zero after close."""
    if not trend_db.is_file():
        return {}
    sym_clause = " AND o.symbol = ?" if sym else ""
    params: tuple[Any, ...] = (sym,) if sym else ()
    out: Dict[str, float] = {}
    for row in query_rows(
        trend_db,
        f"""
        SELECT o.position_id, o.side, o.filled_quantity, o.quantity, p.side AS pos_side
        FROM orders o
        INNER JOIN positions p ON p.position_id = o.position_id
        WHERE lower(o.status) = 'filled'{sym_clause}
        ORDER BY COALESCE(o.filled_at, o.created_at) ASC
        """,
        params,
    ):
        pid = str(row.get("position_id") or "")
        if not pid or pid in out:
            continue
        pos_side = str(row.get("pos_side") or "long").lower()
        o_side = str(row.get("side") or "").lower()
        is_entry = (
            o_side in {"buy", "long"} if pos_side == "long" else o_side in {"sell", "short"}
        )
        if not is_entry:
            continue
        qty = float(row.get("filled_quantity") or row.get("quantity") or 0.0)
        if qty > 0:
            out[pid] = qty
    return out


def _trend_position_qty(
    row: Dict[str, Any], entry_qty_by_pid: Dict[str, float]
) -> float:
    qty = float(row.get("current_size") or 0.0)
    if qty > 0:
        return qty
    pid = str(row.get("position_id") or "")
    return float(entry_qty_by_pid.get(pid) or 0.0)


def _trend_realized_pnl_usdt(
    row: Dict[str, Any], *, entry_qty_by_pid: Dict[str, float]
) -> Optional[float]:
    if not row.get("exit_time"):
        return None
    rpnl = row.get("realized_pnl")
    if rpnl is not None:
        try:
            return float(rpnl)
        except (TypeError, ValueError):
            pass
    entry_px = float(row.get("entry_price") or 0.0)
    exit_px = float(row.get("exit_price") or 0.0)
    qty = _trend_position_qty(row, entry_qty_by_pid)
    if qty <= 0 or entry_px <= 0 or exit_px <= 0:
        return None
    side = str(row.get("pos_side") or row.get("side") or "long").lower()
    if side in {"buy", "long"}:
        return (exit_px - entry_px) * qty
    return (entry_px - exit_px) * qty


def _trend_unrealized_pnl_usdt(
    row: Dict[str, Any],
    mark_px: float,
    *,
    entry_qty_by_pid: Dict[str, float],
) -> Optional[float]:
    if row.get("exit_time"):
        return None
    qty = _trend_position_qty(row, entry_qty_by_pid)
    entry_px = float(row.get("entry_price") or 0.0)
    if qty <= 0 or entry_px <= 0 or mark_px <= 0:
        return None
    side = str(row.get("pos_side") or row.get("side") or "long").lower()
    if side in {"buy", "long"}:
        return (mark_px - entry_px) * qty
    return (entry_px - mark_px) * qty


def _trend_realized_rec(pnl: float) -> Dict[str, Any]:
    return {
        "pnl_usdt": float(pnl),
        "realized_pnl": float(pnl),
        "unrealized_pnl": None,
        "pnl_hint": "已实现",
    }


def _trend_unrealized_rec(pnl: float) -> Dict[str, Any]:
    return {
        "pnl_usdt": float(pnl),
        "realized_pnl": None,
        "unrealized_pnl": float(pnl),
        "pnl_hint": "浮盈",
    }


def _multileg_realized_rows(db_path: Path, symbol: str) -> List[Dict[str, Any]]:
    """Realized round-trips from entry-leg PnL map (more reliable than link re-parse)."""
    if not db_path.is_file():
        return []
    from mlbot_console.services.multileg_leg_pnl import multileg_pnl_by_order_id
    from mlbot_console.services.multileg_order_links import _is_filled_row, is_entry_row

    sym = symbol.upper()
    pnl_map = multileg_pnl_by_order_id(db_path, sym)
    if not pnl_map:
        return []

    rows = query_rows(
        db_path,
        """
        SELECT local_order_id, strategy, symbol, side, purpose, status,
               filled_at, created_at
        FROM multi_leg_orders
        WHERE symbol = ?
        """,
        (sym,),
    )
    links, _ = multi_leg_trade_links(db_path, sym)
    exit_ts_by_entry: Dict[str, int] = {}
    for lk in links:
        if str(lk.get("status") or "").lower() != "closed":
            continue
        em = str(lk.get("entry_marker_id") or "")
        entry_key = em.rsplit(":", 1)[-1] if em else ""
        if entry_key:
            exit_ts_by_entry[entry_key] = int(lk.get("exit_time") or 0)

    out: List[Dict[str, Any]] = []
    for row in rows:
        oid = str(row.get("local_order_id") or "")
        rec = pnl_map.get(oid) or {}
        if rec.get("realized_pnl") is None:
            continue
        purpose = str(row.get("purpose") or "").lower()
        if not (
            (is_entry_row(row) and _is_filled_row(row))
            or (purpose == "inventory" and _is_filled_row(row))
        ):
            continue
        pnl = float(rec["realized_pnl"])
        exit_ts = int(exit_ts_by_entry.get(oid) or 0)
        if exit_ts <= 0:
            exit_ts = int(_parse_ts(row.get("filled_at")) or _parse_ts(row.get("created_at")) or 0)
        out.append(
            {
                "strategy": str(row.get("strategy") or "multi_leg").lower(),
                "symbol": sym,
                "scope": "multi_leg",
                "pnl_usdt": pnl,
                "exit_time": exit_ts,
            }
        )
    return out


def _trend_stats(
    trend_db: Path,
    *,
    symbol: Optional[str],
    mark_prices: Mapping[str, float],
    since_ts: Optional[int],
) -> Dict[str, Any]:
    if not trend_db.is_file():
        return _empty_scope_stats("trend")
    where = ""
    params: tuple[Any, ...] = ()
    if symbol and not _is_all_symbols(symbol):
        where = " WHERE symbol = ?"
        params = (symbol.upper(),)
    rows = query_rows(
        trend_db,
        f"""
        SELECT position_id, symbol, side, current_size, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, strategy_id
        FROM positions
        {where}
        """,
        params,
    )
    entry_qty_by_pid = _trend_entry_qty_by_position(
        trend_db, symbol if symbol and not _is_all_symbols(symbol) else None
    )
    realized = 0.0
    unrealized = 0.0
    open_count = 0
    closed_count = 0
    by_strategy: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "closed_trades": 0.0, "open_positions": 0.0}
    )
    daily: Dict[str, float] = defaultdict(float)

    for row in rows:
        exit_ts = _parse_ts(row.get("exit_time"))
        if since_ts is not None and exit_ts is not None and exit_ts < since_ts:
            continue
        strat = str(row.get("strategy_id") or "trend").lower()
        st = str(row.get("status") or "").lower()
        if st == "open" or not row.get("exit_time"):
            open_count += 1
            by_strategy[strat]["open_positions"] += 1
            
            # Calculate unrealized PnL
            sym = str(row.get("symbol") or "").upper()
            qty = float(row.get("current_size") or 0.0)
            entry_px = float(row.get("entry_price") or 0.0)
            mark_px = float(mark_prices.get(sym) or 0.0)
            
            if qty > 0 and entry_px > 0 and mark_px > 0:
                side = str(row.get("side") or "").lower()
                if side in {"buy", "long"}:
                    upnl = (mark_px - entry_px) * qty
                else:
                    upnl = (entry_px - mark_px) * qty
                unrealized += upnl
                by_strategy[strat]["unrealized_pnl"] += upnl
                
            continue
        pnl = _trend_realized_pnl_usdt(row, entry_qty_by_pid=entry_qty_by_pid)
        if pnl is None:
            closed_count += 1
            continue
        realized += pnl
        closed_count += 1
        by_strategy[strat]["realized_pnl"] += pnl
        by_strategy[strat]["closed_trades"] += 1
        if exit_ts:
            day = datetime.fromtimestamp(exit_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            daily[day] += pnl

    return {
        "scope": "trend",
        "label": _SCOPE_LABELS["trend"],
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "open_positions": open_count,
        "closed_trades": closed_count,
        "by_strategy": dict(by_strategy),
        "daily_realized": [{"date": d, "pnl": daily[d]} for d in sorted(daily)],
    }


def _spot_stats(
    spot_db: Path,
    *,
    symbol: Optional[str],
    mark_prices: Mapping[str, float],
    since_ts: Optional[int],
) -> Dict[str, Any]:
    if not spot_db.is_file():
        return _empty_scope_stats("spot")
    per_order = compute_spot_order_pnl(
        spot_db,
        symbol=symbol,
        mark_prices=mark_prices,
    )
    realized = 0.0
    unrealized = 0.0
    closed = 0
    open_lots = 0
    by_strategy: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "closed_trades": 0.0, "open_lots": 0.0}
    )
    daily: Dict[str, float] = defaultdict(float)
    allowed = set(spot_strategy_ids())
    default_strat = default_spot_strategy_id()
    for sid in allowed:
        by_strategy[sid]  # ensure keys exist

    for rec in per_order.values():
        r = rec.get("realized_pnl")
        u = rec.get("unrealized_pnl")
        strat = str(rec.get("strategy") or default_strat).lower()
        if strat not in allowed:
            strat = default_strat
        if r is not None:
            exit_ts = int(rec.get("exit_ts") or 0)
            if since_ts is not None and exit_ts and exit_ts < since_ts:
                continue
            realized += float(r)
            closed += 1
            by_strategy[strat]["realized_pnl"] += float(r)
            by_strategy[strat]["closed_trades"] += 1
            if exit_ts:
                day = datetime.fromtimestamp(exit_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                )
                daily[day] += float(r)
        elif u is not None:
            unrealized += float(u)
            open_lots += 1
            by_strategy[strat]["unrealized_pnl"] += float(u)
            by_strategy[strat]["open_lots"] += 1

    return {
        "scope": "spot",
        "label": _SCOPE_LABELS["spot"],
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "open_positions": open_lots,
        "closed_trades": closed,
        "by_strategy": dict(by_strategy),
        "daily_realized": [{"date": d, "pnl": daily[d]} for d in sorted(daily)],
    }


def _multileg_stats(
    multi_leg_db: Path,
    *,
    symbol: Optional[str],
    mark_prices: Mapping[str, float],
    since_ts: Optional[int],
) -> Dict[str, Any]:
    if not multi_leg_db.is_file():
        return _empty_scope_stats("multi_leg")
    symbols = _discover_symbols(trend_db=Path("/dev/null"), spot_db=Path("/dev/null"), multi_leg_db=multi_leg_db)
    if symbol and not _is_all_symbols(symbol):
        symbols = [symbol.upper()]
    realized = 0.0
    unrealized = 0.0
    closed = 0
    open_positions = 0
    by_strategy: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "closed_trades": 0.0, "open_positions": 0.0}
    )
    daily: Dict[str, float] = defaultdict(float)

    from mlbot_console.services.multileg_leg_pnl import multileg_pnl_by_order_id
    from mlbot_console.services.db import query_rows

    for sym in symbols:
        pnl_map = multileg_pnl_by_order_id(multi_leg_db, sym, mark_prices=dict(mark_prices))

        rows = query_rows(
            multi_leg_db,
            "SELECT local_order_id, strategy, filled_at FROM multi_leg_orders WHERE symbol = ?",
            (sym,),
        )
        order_meta = {str(r["local_order_id"]): r for r in rows if r.get("local_order_id")}

        entry_meta_rows = query_rows(
            multi_leg_db,
            """
            SELECT local_order_id, strategy, purpose, status, filled_quantity, quantity
            FROM multi_leg_orders
            WHERE symbol = ?
            """,
            (sym,),
        )
        from mlbot_console.services.multileg_order_links import (
            _is_filled_row as _ml_is_filled,
            is_entry_row as _ml_is_entry,
        )

        for row in entry_meta_rows:
            oid = str(row.get("local_order_id") or "")
            rec = pnl_map.get(oid) or {}
            purpose = str(row.get("purpose") or "").lower()
            if not (
                (_ml_is_entry(row) and _ml_is_filled(row))
                or (purpose == "inventory" and _ml_is_filled(row))
            ):
                continue
            u = rec.get("unrealized_pnl")
            if u is None:
                continue
            strat = str(row.get("strategy") or "multi_leg").lower()
            unrealized += float(u)
            by_strategy[strat]["unrealized_pnl"] += float(u)

        for item in _multileg_realized_rows(multi_leg_db, sym):
            exit_ts = int(item.get("exit_time") or 0)
            if since_ts is not None and exit_ts < since_ts:
                continue
            pnl = float(item["pnl_usdt"])
            strat = str(item.get("strategy") or "multi_leg").lower()
            realized += pnl
            closed += 1
            by_strategy[strat]["realized_pnl"] += pnl
            by_strategy[strat]["closed_trades"] += 1
            if exit_ts:
                day = datetime.fromtimestamp(exit_ts, tz=timezone.utc).strftime("%Y-%m-%d")
                daily[day] += pnl
                
        # Get open positions count
        open_rows = query_rows(
            multi_leg_db,
            "SELECT strategy, COUNT(*) as cnt FROM multi_leg_positions WHERE status = 'open' AND symbol = ? GROUP BY strategy",
            (sym,)
        )
        for r in open_rows:
            strat = str(r.get("strategy") or "multi_leg").lower()
            cnt = int(r.get("cnt") or 0)
            open_positions += cnt
            by_strategy[strat]["open_positions"] += cnt

    return {
        "scope": "multi_leg",
        "label": _SCOPE_LABELS["multi_leg"],
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "open_positions": open_positions,
        "closed_trades": closed,
        "by_strategy": dict(by_strategy),
        "daily_realized": [{"date": d, "pnl": daily[d]} for d in sorted(daily)],
    }


def _empty_scope_stats(scope: str) -> Dict[str, Any]:
    return {
        "scope": scope,
        "label": _SCOPE_LABELS.get(scope, scope),
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "open_positions": 0,
        "closed_trades": 0,
        "by_strategy": {},
        "daily_realized": [],
    }


def _merge_daily(series_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    daily: Dict[str, float] = defaultdict(float)
    for series in series_list:
        for pt in series:
            daily[str(pt["date"])] += float(pt.get("pnl") or 0.0)
    return [{"date": d, "pnl": daily[d]} for d in sorted(daily)]


def _parse_utc_date(date_str: str) -> datetime:
    return datetime.strptime(str(date_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _iso_week_start(date_str: str) -> str:
    """Monday (UTC) of the ISO week containing date_str (YYYY-MM-DD)."""
    d = _parse_utc_date(date_str)
    monday = d - timedelta(days=d.weekday())
    return monday.strftime("%Y-%m-%d")


def aggregate_weekly_realized(daily: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sum daily realized PnL by calendar week (Mon–Sun, UTC)."""
    weekly_pnl: Dict[str, float] = defaultdict(float)
    week_end: Dict[str, str] = {}
    for pt in daily or []:
        day = str(pt.get("date") or "")
        if not day:
            continue
        ws = _iso_week_start(day)
        weekly_pnl[ws] += float(pt.get("pnl") or 0.0)
        if ws not in week_end or day > week_end[ws]:
            week_end[ws] = day
    rows: List[Dict[str, Any]] = []
    for ws in sorted(weekly_pnl):
        we = week_end.get(ws, ws)
        rows.append(
            {
                "week_start": ws,
                "week_end": we,
                "pnl": weekly_pnl[ws],
                "label": f"{ws} ~ {we}",
            }
        )
    return rows


def cumulative_realized_curve(daily: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Running sum of daily realized PnL (equity curve of closed PnL only)."""
    cum = 0.0
    out: List[Dict[str, Any]] = []
    for pt in daily or []:
        pnl = float(pt.get("pnl") or 0.0)
        cum += pnl
        out.append(
            {
                "date": str(pt.get("date") or ""),
                "pnl": pnl,
                "cumulative": cum,
            }
        )
    return out


def _weekly_realized_kpis(daily: List[Dict[str, Any]]) -> Dict[str, Any]:
    """This week / last week realized totals (UTC, week starts Monday)."""
    now = datetime.now(timezone.utc)
    this_monday = (now - timedelta(days=now.weekday())).date()
    last_monday = this_monday - timedelta(days=7)
    this_week_pnl = 0.0
    last_week_pnl = 0.0
    for pt in daily or []:
        day = str(pt.get("date") or "")
        if not day:
            continue
        d = _parse_utc_date(day).date()
        pnl = float(pt.get("pnl") or 0.0)
        if d >= this_monday:
            this_week_pnl += pnl
        elif last_monday <= d < this_monday:
            last_week_pnl += pnl
    return {
        "this_week_start": this_monday.isoformat(),
        "last_week_start": last_monday.isoformat(),
        "this_week_pnl": this_week_pnl,
        "last_week_pnl": last_week_pnl,
    }


def build_account_summary(
    *,
    trend_db: Path,
    spot_db: Path,
    spot_ledger_db: Path,
    multi_leg_db: Path,
    feature_bus_root: Path,
    symbol: str = "*",
    lookback_days: int = 30,
    scopes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    lookback_days = int(lookback_days)
    since_ts = (
        None
        if lookback_days <= 0
        else int(datetime.now(timezone.utc).timestamp()) - lookback_days * 86400
    )
    symbols = _discover_symbols(
        trend_db=trend_db, spot_db=spot_db, multi_leg_db=multi_leg_db
    )
    
    from mlbot_console.services.mark_prices import fetch_mark_prices
    marks = fetch_mark_prices(feature_bus_root, symbols)

    sym_arg = None if _is_all_symbols(symbol) else symbol
    trend = _trend_stats(trend_db, symbol=sym_arg, mark_prices=marks, since_ts=since_ts)
    spot = _spot_stats(
        spot_db, symbol=sym_arg, mark_prices=marks, since_ts=since_ts
    )
    multileg = _multileg_stats(multi_leg_db, symbol=sym_arg, mark_prices=marks, since_ts=since_ts)

    all_scopes = [trend, spot, multileg]
    allowed = {str(s).strip().lower() for s in (scopes or [])}
    if allowed:
        scope_blocks = [s for s in all_scopes if str(s.get("scope") or "") in allowed]
    else:
        scope_blocks = all_scopes
    total_realized = sum(float(s["realized_pnl"]) for s in scope_blocks)
    total_unrealized = sum(float(s["unrealized_pnl"]) for s in scope_blocks)
    total_open = sum(int(s["open_positions"]) for s in scope_blocks)
    total_closed = sum(int(s["closed_trades"]) for s in scope_blocks)

    strategy_rows: Dict[str, Dict[str, Any]] = {}
    for scope_block in scope_blocks:
        scope_name = scope_block["scope"]
        for strat, agg in (scope_block.get("by_strategy") or {}).items():
            key = f"{scope_name}:{strat}"
            strategy_rows[key] = {
                "scope": scope_name,
                "strategy": strat,
                "realized_pnl": float(agg.get("realized_pnl") or 0.0),
                "unrealized_pnl": float(agg.get("unrealized_pnl") or 0.0),
                "closed_trades": int(agg.get("closed_trades") or 0.0),
                "open_positions": int(
                    agg.get("open_positions") or agg.get("open_lots") or 0.0
                ),
            }

    daily = _merge_daily([s["daily_realized"] for s in scope_blocks])

    ledger = build_exchange_ledger(mark_prices=marks, symbol=symbol)
    exchange_by_scope = {str(a["scope"]): a for a in ledger.get("accounts") or []}
    
    from mlbot_console.services.spot_ledger_book import fetch_spot_ledger_holdings
    spot_ledger_data = fetch_spot_ledger_holdings(spot_ledger_db, marks)
    fifo_holdings = spot_holdings_from_orders(spot_db, mark_prices=marks)
    ledger_by_asset = {h["asset"]: h for h in spot_ledger_data.get("holdings") or []}
    fifo_by_asset = {h["asset"]: h for h in fifo_holdings}

    for scope_block in scope_blocks:
        ex = exchange_by_scope.get(str(scope_block.get("scope") or ""))
        if ex:
            if scope_block["scope"] == "spot":
                ex["ledger_holdings"] = spot_ledger_data["holdings"]
                ex["ledger_holdings_value_usdt"] = spot_ledger_data["holdings_value_usdt"]
                for h in ex.get("holdings") or []:
                    asset = str(h.get("asset") or "")
                    led = ledger_by_asset.get(asset) or fifo_by_asset.get(asset)
                    if not led:
                        continue
                    avg = float(led.get("cost_basis") or led.get("avg_entry_usdt") or 0.0)
                    if avg > 0:
                        h["avg_entry_usdt"] = avg
                        h["cost_notional_usdt"] = float(
                            led.get("deploy_usdt")
                            or led.get("cost_notional_usdt")
                            or avg * float(h.get("qty") or 0.0)
                        )
                    if led.get("unrealized_pnl_usdt") is not None:
                        h["unrealized_pnl_usdt"] = float(led["unrealized_pnl_usdt"])
                    h["entry_source"] = (
                        "ledger" if asset in ledger_by_asset else "fifo_orders"
                    )
            scope_block["exchange"] = ex

    ledger_totals = dict(ledger.get("totals") or {})
    totals = {
        "realized_pnl": total_realized,
        "unrealized_pnl": total_unrealized,
        "open_positions": total_open,
        "closed_trades": total_closed,
        "equity_usdt": ledger_totals.get("equity_usdt"),
        "wallet_balance_usdt": ledger_totals.get("wallet_balance_usdt"),
        "available_usdt": ledger_totals.get("available_usdt"),
        "exchange_unrealized_pnl_usdt": ledger_totals.get(
            "exchange_unrealized_pnl_usdt"
        ),
    }

    last_day_pnl = 0.0
    last_7d_pnl = 0.0
    if daily:
        last_day_pnl = float(daily[-1].get("pnl") or 0.0)
        tail = daily[-7:]
        last_7d_pnl = sum(float(d.get("pnl") or 0.0) for d in tail)

    weekly = aggregate_weekly_realized(daily)
    weekly_kpis = _weekly_realized_kpis(daily)
    cumulative = cumulative_realized_curve(daily)

    return {
        "symbol": "ALL" if _is_all_symbols(symbol) else str(symbol).upper(),
        "lookback_days": lookback_days,
        "since_ts": since_ts,
        "totals": totals,
        "recent_realized": {
            "last_day_pnl": last_day_pnl,
            "last_7d_pnl": last_7d_pnl,
            "last_day": daily[-1]["date"] if daily else None,
            **weekly_kpis,
        },
        "weekly_realized": weekly,
        "cumulative_realized": cumulative,
        "exchange_ledger": ledger,
        "scopes": scope_blocks,
        "strategies": sorted(
            strategy_rows.values(),
            key=lambda r: (r["scope"], r["strategy"]),
        ),
        "daily_realized": daily,
        "mark_prices": marks,
        "notes": [
            "余额/权益来自币安实时 API：Trend→BINANCE_API_KEY，Multi-leg→MULTI_LEG_BINANCE_FUTURES_*，Spot→BINANCE_SPOT_*。",
            "合约权益=totalMarginBalance，钱包余额=totalWalletBalance；现货权益≈USDT+持仓按标记价折算。",
            "总账 equity_usdt 为各账户权益之和（三个独立子账户，非单账户拆分）。",
            "Trend PnL from positions.realized_pnl on closed rows.",
            "Spot PnL uses FIFO buy lots by fill time; sells realize against oldest buys.",
            "Spot open buys show unrealized PnL when bars_1min close is available.",
            "Multi-leg PnL approximated from filled entry/exit leg prices × quantity.",
            (
                "Realized PnL and daily chart include all historical exits."
                if lookback_days <= 0
                else f"Realized PnL and daily chart only include exits within the last {lookback_days} days."
            ),
        ],
    }


def build_order_pnl_maps(
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    feature_bus_root: Optional[Path] = None,
    symbol: str = "*",
    mark_prices: Optional[Dict[str, float]] = None,
    scopes: Optional[Tuple[str, ...]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (trend_by_order_id, spot_by_order_id, multileg_by_order_id) enrichment maps."""
    scope_set = {str(s).strip().lower() for s in (scopes or ()) if str(s).strip()}
    if not scope_set:
        scope_set = {"trend", "spot", "multi_leg"}
    sym = None if _is_all_symbols(symbol) else symbol.upper()
    marks = dict(mark_prices or {})
    if not marks and feature_bus_root is not None and feature_bus_root.is_dir():
        if sym:
            marks = latest_close_prices(feature_bus_root, [sym])
        else:
            symbols = _discover_symbols(
                trend_db=trend_db, spot_db=spot_db, multi_leg_db=multi_leg_db
            )
            marks = latest_close_prices(feature_bus_root, symbols)

    trend_map: Dict[str, Dict[str, Any]] = {}
    if "trend" in scope_set and trend_db.is_file():
        where = ""
        params: tuple[Any, ...] = ()
        if sym:
            where = " WHERE symbol = ?"
            params = (sym,)
        entry_qty_by_pid = _trend_entry_qty_by_position(trend_db, sym)
        positions = query_rows(
            trend_db,
            f"""
            SELECT position_id, symbol, side, current_size, entry_time, exit_time,
                   entry_price, exit_price, realized_pnl, status, strategy_id
            FROM positions
            {where}
            """,
            params,
        )
        for row in positions:
            pid = str(row.get("position_id") or "")
            if not pid:
                continue
            sym_u = str(row.get("symbol") or "").upper()
            mark_px = float(marks.get(sym_u) or 0.0)
            if row.get("exit_time"):
                pnl = _trend_realized_pnl_usdt(row, entry_qty_by_pid=entry_qty_by_pid)
                if pnl is None:
                    continue
                rec = _trend_realized_rec(pnl)
                trend_map[f"{pid}:exit"] = rec
                continue
            upnl = _trend_unrealized_pnl_usdt(
                row, mark_px, entry_qty_by_pid=entry_qty_by_pid
            )
            if upnl is None:
                continue
            rec = _trend_unrealized_rec(upnl)
            trend_map[f"{pid}:entry"] = rec

        sym_filter = " AND o.symbol = ?" if sym else ""
        for row in query_rows(
            trend_db,
            f"""
            SELECT o.order_id, o.side, o.position_id,
                   p.realized_pnl, p.side AS pos_side, p.exit_time,
                   p.entry_price, p.exit_price, p.current_size, p.symbol
            FROM orders o
            INNER JOIN positions p ON p.position_id = o.position_id
            WHERE o.position_id IS NOT NULL{sym_filter}
            """,
            params,
        ):
            oid = str(row.get("order_id") or "")
            pid = str(row.get("position_id") or "")
            if not oid or oid in trend_map:
                continue
            pos_side = str(row.get("pos_side") or "long").lower()
            o_side = str(row.get("side") or "").lower()
            is_long = pos_side == "long"
            is_entry = o_side in {"buy", "long"} if is_long else o_side in {"sell", "short"}
            is_exit = o_side in {"sell", "short"} if is_long else o_side in {"buy", "long"}
            if row.get("exit_time"):
                if not is_exit:
                    continue
                pnl = _trend_realized_pnl_usdt(row, entry_qty_by_pid=entry_qty_by_pid)
                if pnl is None:
                    continue
                trend_map[oid] = _trend_realized_rec(pnl)
                continue
            if not is_entry:
                continue
            sym_u = str(row.get("symbol") or "").upper()
            mark_px = float(marks.get(sym_u) or 0.0)
            upnl = _trend_unrealized_pnl_usdt(
                row, mark_px, entry_qty_by_pid=entry_qty_by_pid
            )
            if upnl is None:
                continue
            existing = trend_map.get(f"{pid}:entry")
            trend_map[oid] = existing if existing else _trend_unrealized_rec(upnl)

    spot_map: Dict[str, Dict[str, Any]] = {}
    if "spot" in scope_set and spot_db.is_file():
        spot_map = compute_spot_order_pnl(
            spot_db, symbol=symbol if sym else None, mark_prices=marks
        )

    multileg_map: Dict[str, Dict[str, Any]] = {}
    if "multi_leg" in scope_set and multi_leg_db.is_file():
        from mlbot_console.services.multileg_leg_pnl import multileg_pnl_by_order_id

        if sym:
            multileg_map = multileg_pnl_by_order_id(
                multi_leg_db, sym, mark_prices=marks
            )
        else:
            for s in _discover_symbols(
                trend_db=Path("/dev/null"),
                spot_db=Path("/dev/null"),
                multi_leg_db=multi_leg_db,
            ):
                multileg_map.update(
                    multileg_pnl_by_order_id(
                        multi_leg_db, s, mark_prices=marks
                    )
                )

    return trend_map, spot_map, multileg_map
