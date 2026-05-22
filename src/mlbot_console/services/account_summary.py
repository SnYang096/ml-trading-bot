"""Account-level PnL and order stats for the business console."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from mlbot_console.services.db import query_rows
from mlbot_console.services.spot_pnl import compute_spot_order_pnl
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


def _multileg_realized_rows(db_path: Path, symbol: str) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    sym = symbol.upper()
    rows = query_rows(
        db_path,
        """
        SELECT local_order_id, strategy, symbol, side, purpose, status, order_type,
               filled_quantity, average_price, filled_at, created_at, price, quantity,
               stop_price, leg_id
        FROM multi_leg_orders
        WHERE symbol = ?
        ORDER BY COALESCE(filled_at, created_at) ASC
        """,
        (sym,),
    )
    by_id = {str(r.get("local_order_id") or ""): r for r in rows}
    out: List[Dict[str, Any]] = []
    for link in multi_leg_trade_links(db_path, sym)[0]:
        if str(link.get("status") or "").lower() != "closed":
            continue
        exit_mid = str(link.get("exit_marker_id") or "")
        entry_mid = str(link.get("entry_marker_id") or "")
        exit_key = exit_mid.rsplit(":", 1)[-1] if exit_mid else ""
        entry_key = entry_mid.rsplit(":", 1)[-1] if entry_mid else ""
        exit_row = by_id.get(exit_key)
        entry_row = by_id.get(entry_key)
        if not exit_row or not entry_row:
            continue
        pnl = _link_pnl_usdt(entry_row, exit_row)
        if pnl is None:
            continue
        out.append(
            {
                "strategy": str(link.get("strategy") or exit_row.get("strategy") or "multi_leg"),
                "symbol": sym,
                "scope": "multi_leg",
                "pnl_usdt": pnl,
                "exit_time": int(link.get("exit_time") or 0),
            }
        )
    return out


def _trend_stats(
    trend_db: Path,
    *,
    symbol: Optional[str],
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
        SELECT position_id, symbol, side, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, strategy_id
        FROM positions
        {where}
        """,
        params,
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
        rpnl = row.get("realized_pnl")
        if st == "open" or not row.get("exit_time"):
            open_count += 1
            by_strategy[strat]["open_positions"] += 1
            continue
        if rpnl is None:
            closed_count += 1
            continue
        pnl = float(rpnl)
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
    since_ts: Optional[int],
) -> Dict[str, Any]:
    if not multi_leg_db.is_file():
        return _empty_scope_stats("multi_leg")
    symbols = _discover_symbols(trend_db=Path("/dev/null"), spot_db=Path("/dev/null"), multi_leg_db=multi_leg_db)
    if symbol and not _is_all_symbols(symbol):
        symbols = [symbol.upper()]
    realized = 0.0
    closed = 0
    by_strategy: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "closed_trades": 0.0, "open_positions": 0.0}
    )
    daily: Dict[str, float] = defaultdict(float)

    for sym in symbols:
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

    return {
        "scope": "multi_leg",
        "label": _SCOPE_LABELS["multi_leg"],
        "realized_pnl": realized,
        "unrealized_pnl": 0.0,
        "open_positions": 0,
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


def build_account_summary(
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    feature_bus_root: Path,
    symbol: str = "*",
    lookback_days: int = 30,
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
    marks = latest_close_prices(feature_bus_root, symbols)

    sym_arg = None if _is_all_symbols(symbol) else symbol
    trend = _trend_stats(trend_db, symbol=sym_arg, since_ts=since_ts)
    spot = _spot_stats(
        spot_db, symbol=sym_arg, mark_prices=marks, since_ts=since_ts
    )
    multileg = _multileg_stats(multi_leg_db, symbol=sym_arg, since_ts=since_ts)

    scopes = [trend, spot, multileg]
    total_realized = sum(float(s["realized_pnl"]) for s in scopes)
    total_unrealized = sum(float(s["unrealized_pnl"]) for s in scopes)
    total_open = sum(int(s["open_positions"]) for s in scopes)
    total_closed = sum(int(s["closed_trades"]) for s in scopes)

    strategy_rows: Dict[str, Dict[str, Any]] = {}
    for scope_block in scopes:
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

    daily = _merge_daily([s["daily_realized"] for s in scopes])

    ledger = build_exchange_ledger(mark_prices=marks)
    exchange_by_scope = {str(a["scope"]): a for a in ledger.get("accounts") or []}
    for scope_block in scopes:
        ex = exchange_by_scope.get(str(scope_block.get("scope") or ""))
        if ex:
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

    return {
        "symbol": "ALL" if _is_all_symbols(symbol) else str(symbol).upper(),
        "lookback_days": lookback_days,
        "since_ts": since_ts,
        "totals": totals,
        "exchange_ledger": ledger,
        "scopes": scopes,
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
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (trend_by_order_id, spot_by_order_id, multileg_by_order_id) enrichment maps."""
    symbols = _discover_symbols(
        trend_db=trend_db, spot_db=spot_db, multi_leg_db=multi_leg_db
    )
    marks = dict(mark_prices or {})
    if not marks and feature_bus_root is not None and feature_bus_root.is_dir():
        marks = latest_close_prices(feature_bus_root, symbols)
    sym = None if _is_all_symbols(symbol) else symbol.upper()

    trend_map: Dict[str, Dict[str, Any]] = {}
    if trend_db.is_file():
        where = ""
        params: tuple[Any, ...] = ()
        if sym:
            where = " WHERE symbol = ?"
            params = (sym,)
        for row in query_rows(
            trend_db,
            f"""
            SELECT position_id, symbol, realized_pnl, status, exit_time, entry_time
            FROM positions
            {where}
            """,
            params,
        ):
            pid = str(row.get("position_id") or "")
            rpnl = row.get("realized_pnl")
            if rpnl is not None and row.get("exit_time"):
                rec = {
                    "pnl_usdt": float(rpnl),
                    "realized_pnl": float(rpnl),
                    "pnl_hint": "已实现",
                }
                trend_map[f"{pid}:exit"] = rec
        sym_filter = " AND o.symbol = ?" if sym else ""
        for row in query_rows(
            trend_db,
            f"""
            SELECT o.order_id, o.side, p.realized_pnl, p.side AS pos_side, p.exit_time
            FROM orders o
            INNER JOIN positions p ON p.position_id = o.position_id
            WHERE p.exit_time IS NOT NULL AND p.realized_pnl IS NOT NULL{sym_filter}
            """,
            params,
        ):
            oid = str(row.get("order_id") or "")
            if not oid or oid in trend_map:
                continue
            o_side = str(row.get("side") or "")
            p_side = str(row.get("pos_side") or "long")
            is_long = p_side.lower() == "long"
            exit_side = o_side.lower() in {"sell", "short"} if is_long else o_side.lower() in {
                "buy",
                "long",
            }
            if not exit_side:
                continue
            rpnl = row.get("realized_pnl")
            if rpnl is None:
                continue
            trend_map[oid] = {
                "pnl_usdt": float(rpnl),
                "realized_pnl": float(rpnl),
                "pnl_hint": "已实现",
            }

    spot_map: Dict[str, Dict[str, Any]] = {}
    if spot_db.is_file():
        spot_map = compute_spot_order_pnl(
            spot_db, symbol=symbol if sym else None, mark_prices=marks
        )

    multileg_map: Dict[str, Dict[str, Any]] = {}
    if multi_leg_db.is_file():
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
