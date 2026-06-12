#!/usr/bin/env python3
"""Reconcile B·Trend local closes vs Binance futures allOrders + userTrades.

Answers: who closed the position (STOP trigger, MARKET, manual), and whether
local DB has a matching filled order.

Usage (inside quant-trend-swing container):

  python3 scripts/reconcile_trend_closes_with_exchange.py \\
    --db /app/data/order_management.db --symbol ETHUSDT --since 2026-05-01

If copied to /tmp, set MLBOT_REPO_ROOT=/app (default fallback).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_repo_root() -> Path:
    env = os.getenv("MLBOT_REPO_ROOT", "").strip()
    if env:
        root = Path(env)
        if (root / "src" / "order_management").is_dir():
            return root
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        root = here.parents[1]
        if (root / "src" / "order_management").is_dir():
            return root
    for candidate in (Path("/app"), Path.cwd(), *here.parents):
        if (candidate / "src" / "order_management").is_dir():
            return candidate
    return here.parents[1]


_REPO_ROOT = _resolve_repo_root()
_SRC = _REPO_ROOT / "src"
for p in (_REPO_ROOT, _SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.order_management.binance_api import BinanceAPI


def _parse_ts(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    raw = str(s).strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def _ms_to_iso(ms: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (TypeError, ValueError, OSError):
        return str(ms)


def _since_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# Binance futures allOrders / userTrades: startTime..endTime window must be < 7 days.
_WINDOW_MS = 6 * 24 * 3600 * 1000


def _iter_time_windows(start_ms: int, end_ms: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        win_end = min(cursor + _WINDOW_MS, end_ms)
        windows.append((cursor, win_end))
        cursor = win_end + 1
    return windows


def _fetch_all_orders(
    api: BinanceAPI, symbol: str, start_ms: int
) -> List[Dict[str, Any]]:
    mid = api._futures_market_id(symbol)
    if not mid:
        return []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for win_start, win_end in _iter_time_windows(start_ms, end_ms):
        print(f"  allOrders window {_ms_to_iso(win_start)} .. {_ms_to_iso(win_end)}")
        batch = api._fapi_signed_get(
            "/fapi/v1/allOrders",
            {
                "symbol": mid,
                "startTime": win_start,
                "endTime": win_end,
                "limit": 1000,
            },
        )
        if not isinstance(batch, list):
            continue
        for row in batch:
            oid = str(row.get("orderId") or "")
            if oid and oid in seen:
                continue
            if oid:
                seen.add(oid)
            out.append(row)
    return out


def _fetch_user_trades(
    api: BinanceAPI, symbol: str, start_ms: int
) -> List[Dict[str, Any]]:
    mid = api._futures_market_id(symbol)
    if not mid:
        return []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for win_start, win_end in _iter_time_windows(start_ms, end_ms):
        print(f"  userTrades window {_ms_to_iso(win_start)} .. {_ms_to_iso(win_end)}")
        batch = api._fapi_signed_get(
            "/fapi/v1/userTrades",
            {
                "symbol": mid,
                "startTime": win_start,
                "endTime": win_end,
                "limit": 1000,
            },
        )
        if not isinstance(batch, list):
            continue
        for row in batch:
            tid = str(row.get("id") or row.get("tradeId") or "")
            if tid and tid in seen:
                continue
            if tid:
                seen.add(tid)
            out.append(row)
    return out


def _local_short_positions(
    conn: sqlite3.Connection, symbol: str
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT position_id, side, entry_time, exit_time, entry_price, exit_price,
               realized_pnl, exit_reason, status
        FROM positions
        WHERE symbol = ? AND lower(side) = 'short'
        ORDER BY entry_time
        """,
        (symbol.upper(),),
    ).fetchall()
    cols = [
        "position_id",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "realized_pnl",
        "exit_reason",
        "status",
    ]
    return [dict(zip(cols, r)) for r in rows]


def _local_orders_by_pid(conn: sqlite3.Connection, pid: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT order_id, binance_order_id, side, status, order_type, quantity,
               average_price, stop_price, created_at, filled_at, error_message
        FROM orders
        WHERE position_id = ?
        ORDER BY created_at
        """,
        (pid,),
    ).fetchall()
    cols = [
        "order_id",
        "binance_order_id",
        "side",
        "status",
        "order_type",
        "quantity",
        "average_price",
        "stop_price",
        "created_at",
        "filled_at",
        "error_message",
    ]
    return [dict(zip(cols, r)) for r in rows]


def _classify_exchange_order(row: Dict[str, Any]) -> str:
    typ = str(row.get("type") or "").upper()
    close_pos = str(row.get("closePosition") or row.get("close_position") or "").lower()
    reduce = str(row.get("reduceOnly") or row.get("reduce_only") or "").lower()
    if close_pos in {"true", "1"}:
        return f"{typ}+closePosition"
    if reduce in {"true", "1"}:
        return f"{typ}+reduceOnly"
    return typ


def _match_trades_to_close(
    trades: List[Dict[str, Any]],
    *,
    position_side: str,
    close_side: str,
    entry_ms: int,
    exit_ms: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in trades:
        if str(t.get("side") or "").upper() != close_side.upper():
            continue
        ps = str(t.get("positionSide") or "BOTH").upper()
        if ps not in {"", "BOTH"} and ps != position_side.upper():
            continue
        t_ms = int(t.get("time") or 0)
        if t_ms < entry_ms:
            continue
        if exit_ms is not None and t_ms > exit_ms + 86400_000:
            continue
        out.append(t)
    return out


def reconcile(
    *,
    db_path: Path,
    symbol: str,
    since: str,
    api: BinanceAPI,
) -> Dict[str, Any]:
    start_ms = _since_ms(since)
    conn = sqlite3.connect(str(db_path))
    local_positions = _local_short_positions(conn, symbol)

    print("Fetching Binance allOrders...")
    all_orders = _fetch_all_orders(api, symbol, start_ms)
    print("Fetching Binance userTrades...")
    user_trades = _fetch_user_trades(api, symbol, start_ms)

    # Exchange closes for SHORT = BUY side filled
    ex_closes = [
        o
        for o in all_orders
        if str(o.get("status") or "").upper() == "FILLED"
        and str(o.get("side") or "").upper() == "BUY"
        and str(o.get("positionSide") or "BOTH").upper() in {"SHORT", "BOTH"}
    ]
    ex_opens_short = [
        o
        for o in all_orders
        if str(o.get("status") or "").upper() == "FILLED"
        and str(o.get("side") or "").upper() == "SELL"
        and str(o.get("positionSide") or "BOTH").upper() in {"SHORT", "BOTH"}
    ]

    stop_filled = [o for o in ex_closes if "STOP" in str(o.get("type") or "").upper()]
    market_closes = [
        o for o in ex_closes if str(o.get("type") or "").upper() == "MARKET"
    ]

    report: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "since": since,
        "local_short_positions": len(local_positions),
        "exchange_all_orders": len(all_orders),
        "exchange_user_trades": len(user_trades),
        "exchange_filled_buy_closes": len(ex_closes),
        "exchange_filled_sell_opens_short": len(ex_opens_short),
        "exchange_filled_stop_closes": len(stop_filled),
        "exchange_filled_market_closes": len(market_closes),
        "positions": [],
        "unmatched_exchange_closes": [],
    }

    local_binance_ids = {
        str(r[0])
        for r in conn.execute(
            "SELECT binance_order_id FROM orders WHERE binance_order_id IS NOT NULL"
        )
        if r[0]
    }

    matched_ex_order_ids: set[str] = set()

    for pos in local_positions:
        pid = str(pos["position_id"])
        entry_ms = _parse_ts(pos.get("entry_time"))
        exit_ms = _parse_ts(pos.get("exit_time"))
        entry_ms_i = (entry_ms or 0) * 1000
        local_orders = _local_orders_by_pid(conn, pid)
        local_filled_close = [
            o
            for o in local_orders
            if str(o.get("side") or "").lower() == "buy"
            and str(o.get("status") or "").lower() == "filled"
        ]

        # Nearest exchange BUY fill after entry (within 14d of local exit if set)
        window_end = (exit_ms or 0) * 1000 + 14 * 86400_000 if exit_ms else None
        candidates = []
        for o in ex_closes:
            t_ms = int(o.get("updateTime") or o.get("time") or 0)
            if t_ms < entry_ms_i:
                continue
            if window_end is not None and t_ms > window_end:
                continue
            candidates.append(o)
        candidates.sort(key=lambda x: int(x.get("updateTime") or x.get("time") or 0))

        best = candidates[0] if candidates else None
        if best is not None:
            matched_ex_order_ids.add(str(best.get("orderId") or ""))

        close_trades = _match_trades_to_close(
            user_trades,
            position_side="SHORT",
            close_side="BUY",
            entry_ms=entry_ms_i,
            exit_ms=(exit_ms * 1000 if exit_ms else None),
        )

        item = {
            "position_id": pid,
            "entry_time": pos.get("entry_time"),
            "exit_time": pos.get("exit_time"),
            "exit_reason": pos.get("exit_reason"),
            "local_filled_close_orders": len(local_filled_close),
            "local_orders_summary": [
                {
                    "side": o.get("side"),
                    "status": o.get("status"),
                    "order_type": o.get("order_type"),
                    "created_at": o.get("created_at"),
                    "error": (o.get("error_message") or "")[:80],
                }
                for o in local_orders
            ],
            "nearest_exchange_close": None,
            "exchange_trades_near_exit": [],
        }
        if best is not None:
            oid = str(best.get("orderId") or "")
            item["nearest_exchange_close"] = {
                "orderId": oid,
                "in_local_db": oid in local_binance_ids,
                "time": _ms_to_iso(best.get("updateTime") or best.get("time")),
                "type": _classify_exchange_order(best),
                "avgPrice": best.get("avgPrice"),
                "executedQty": best.get("executedQty"),
                "status": best.get("status"),
                "clientOrderId": best.get("clientOrderId"),
            }
        if close_trades:
            item["exchange_trades_near_exit"] = [
                {
                    "time": _ms_to_iso(t.get("time")),
                    "price": t.get("price"),
                    "qty": t.get("qty"),
                    "realizedPnl": t.get("realizedPnl"),
                    "buyer": t.get("buyer"),
                    "orderId": t.get("orderId"),
                }
                for t in close_trades[-5:]
            ]
        report["positions"].append(item)

    for o in ex_closes:
        oid = str(o.get("orderId") or "")
        if oid and oid not in matched_ex_order_ids:
            report["unmatched_exchange_closes"].append(
                {
                    "orderId": oid,
                    "time": _ms_to_iso(o.get("updateTime") or o.get("time")),
                    "type": _classify_exchange_order(o),
                    "avgPrice": o.get("avgPrice"),
                    "executedQty": o.get("executedQty"),
                    "in_local_db": oid in local_binance_ids,
                    "clientOrderId": o.get("clientOrderId"),
                }
            )

    conn.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--since", default="2026-05-01")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
        "BINANCE_FUTURES_API_SECRET", ""
    )
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")

    api = BinanceAPI(api_key, api_secret, testnet=False)
    report = reconcile(
        db_path=args.db,
        symbol=str(args.symbol).upper(),
        since=args.since,
        api=api,
    )

    print("=== reconcile_trend_closes_with_exchange ===")
    print(f"symbol={report['symbol']} since={report['since']}")
    print(
        f"local_short_positions={report['local_short_positions']} "
        f"ex_allOrders={report['exchange_all_orders']} "
        f"ex_userTrades={report['exchange_user_trades']}"
    )
    print(
        f"ex_filled_buy_closes={report['exchange_filled_buy_closes']} "
        f"(STOP={report['exchange_filled_stop_closes']} "
        f"MARKET={report['exchange_filled_market_closes']})"
    )
    print()

    for item in report["positions"]:
        print(f"--- {item['position_id']} ---")
        print(
            f"  local exit={item['exit_time']} reason={item['exit_reason']} "
            f"filled_close_orders={item['local_filled_close_orders']}"
        )
        for lo in item["local_orders_summary"]:
            print(
                f"    local: {lo['side']} {lo['status']} {lo['order_type']} @ {lo['created_at']}"
            )
            if lo.get("error"):
                print(f"           err: {lo['error']}")
        ex = item.get("nearest_exchange_close")
        if ex:
            print(
                f"  nearest EX close: {ex['time']} {ex['type']} "
                f"px={ex['avgPrice']} qty={ex['executedQty']} "
                f"in_local_db={ex['in_local_db']} orderId={ex['orderId']}"
            )
        else:
            print("  nearest EX close: (none in window)")
        for tr in item.get("exchange_trades_near_exit") or []:
            print(
                f"    trade: {tr['time']} px={tr['price']} qty={tr['qty']} "
                f"rpnl={tr['realizedPnl']} orderId={tr['orderId']}"
            )
        print()

    if report["unmatched_exchange_closes"]:
        print("=== exchange BUY closes not matched to any local position ===")
        for row in report["unmatched_exchange_closes"][:20]:
            print(
                f"  {row['time']} {row['type']} px={row['avgPrice']} "
                f"qty={row['executedQty']} in_local_db={row['in_local_db']} "
                f"orderId={row['orderId']}"
            )

    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
