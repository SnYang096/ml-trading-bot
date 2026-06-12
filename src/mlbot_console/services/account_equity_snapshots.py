"""Daily Binance wallet/equity snapshots for account curve history."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services.db import query_rows
from mlbot_console.services.exchange_balances import build_exchange_ledger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_equity_daily (
    snapshot_date TEXT NOT NULL,
    scope TEXT NOT NULL,
    wallet_balance_usdt REAL,
    equity_usdt REAL,
    unrealized_pnl_usdt REAL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, scope)
);
CREATE INDEX IF NOT EXISTS idx_account_equity_daily_date
    ON account_equity_daily(snapshot_date);
"""


def ensure_snapshot_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def capture_daily_account_snapshots(
    db_path: Path,
    *,
    mark_prices: Optional[Dict[str, float]] = None,
    snapshot_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch exchange ledger and upsert one row per scope + ``all`` for UTC date."""
    ensure_snapshot_schema(db_path)
    day = str(snapshot_date or _utc_today())
    fetched_at = datetime.now(timezone.utc).isoformat()
    ledger = build_exchange_ledger(mark_prices=mark_prices)
    accounts = list(ledger.get("accounts") or [])
    totals = dict(ledger.get("totals") or {})

    rows: List[Dict[str, Any]] = []
    for acct in accounts:
        if not acct.get("ok"):
            continue
        scope = str(acct.get("scope") or "")
        if not scope:
            continue
        wallet = acct.get("wallet_balance_usdt")
        equity = acct.get("equity_usdt")
        upnl = acct.get("account_unrealized_pnl_usdt")
        if upnl is None:
            upnl = acct.get("unrealized_pnl_usdt")
        rows.append(
            {
                "snapshot_date": day,
                "scope": scope,
                "wallet_balance_usdt": float(wallet) if wallet is not None else None,
                "equity_usdt": float(equity) if equity is not None else None,
                "unrealized_pnl_usdt": float(upnl) if upnl is not None else None,
                "fetched_at": fetched_at,
            }
        )

    rows.append(
        {
            "snapshot_date": day,
            "scope": "all",
            "wallet_balance_usdt": (
                float(totals["wallet_balance_usdt"])
                if totals.get("wallet_balance_usdt") is not None
                else None
            ),
            "equity_usdt": (
                float(totals["equity_usdt"])
                if totals.get("equity_usdt") is not None
                else None
            ),
            "unrealized_pnl_usdt": (
                float(totals["exchange_unrealized_pnl_usdt"])
                if totals.get("exchange_unrealized_pnl_usdt") is not None
                else None
            ),
            "fetched_at": fetched_at,
        }
    )

    conn = sqlite3.connect(str(db_path))
    try:
        for row in rows:
            conn.execute(
                """
                INSERT INTO account_equity_daily (
                    snapshot_date, scope, wallet_balance_usdt, equity_usdt,
                    unrealized_pnl_usdt, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_date, scope) DO UPDATE SET
                    wallet_balance_usdt = excluded.wallet_balance_usdt,
                    equity_usdt = excluded.equity_usdt,
                    unrealized_pnl_usdt = excluded.unrealized_pnl_usdt,
                    fetched_at = excluded.fetched_at
                """,
                (
                    row["snapshot_date"],
                    row["scope"],
                    row["wallet_balance_usdt"],
                    row["equity_usdt"],
                    row["unrealized_pnl_usdt"],
                    row["fetched_at"],
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "snapshot_date": day,
        "fetched_at": fetched_at,
        "rows_written": len(rows),
        "db_path": str(db_path),
        "accounts_ok": int(totals.get("accounts_ok") or 0),
    }


def load_daily_equity_curves(
    db_path: Path,
    *,
    scope: str = "all",
    lookback_days: int = 0,
) -> Dict[str, Any]:
    """Load historical wallet/equity series from daily snapshots."""
    if not db_path.is_file():
        return {"balance": [], "equity": [], "note": "无账户快照 DB"}
    ensure_snapshot_schema(db_path)
    where = " WHERE scope = ?"
    params: tuple[Any, ...] = (scope,)
    if lookback_days > 0:
        since = (
            datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
        ).strftime("%Y-%m-%d")
        where += " AND snapshot_date >= ?"
        params = (scope, since)
    rows = query_rows(
        db_path,
        f"""
        SELECT snapshot_date, wallet_balance_usdt, equity_usdt, unrealized_pnl_usdt
        FROM account_equity_daily
        {where}
        ORDER BY snapshot_date ASC
        """,
        params,
    )
    balance: List[Dict[str, Any]] = []
    equity: List[Dict[str, Any]] = []
    for row in rows:
        d = str(row.get("snapshot_date") or "")
        w = row.get("wallet_balance_usdt")
        e = row.get("equity_usdt")
        if w is not None:
            balance.append({"date": d, "value_usdt": float(w)})
        if e is not None:
            equity.append({"date": d, "value_usdt": float(e)})
    note = (
        f"来自 account_equity 日快照（scope={scope}，{len(balance)} 天）"
        if balance
        else "尚无账户日快照；请运行 snapshot_account_equity 或等待日更任务"
    )
    return {"balance": balance, "equity": equity, "note": note, "scope": scope}


def merge_live_into_curves(
    curves: Dict[str, Any],
    *,
    wallet_usdt: Optional[float],
    equity_usdt: Optional[float],
) -> Dict[str, Any]:
    """Refresh today's (or append) balance/equity with live exchange values."""
    if wallet_usdt is None and equity_usdt is None:
        return curves
    today = _utc_today()
    balance = list(curves.get("balance") or [])
    equity = list(curves.get("equity") or [])

    def _upsert(series: List[Dict[str, Any]], value: float) -> List[Dict[str, Any]]:
        out = list(series)
        if out and str(out[-1].get("date") or "") == today:
            out[-1] = {"date": today, "value_usdt": float(value)}
        else:
            out.append({"date": today, "value_usdt": float(value)})
        return out

    if wallet_usdt is not None:
        balance = _upsert(balance, float(wallet_usdt))
    if equity_usdt is not None:
        equity = _upsert(equity, float(equity_usdt))
    elif wallet_usdt is not None:
        equity = _upsert(equity, float(wallet_usdt))

    merged = dict(curves)
    merged["balance"] = balance
    merged["equity"] = equity
    if balance and "日快照" in str(curves.get("note") or ""):
        merged["note"] = (
            str(curves.get("note") or "")
            + "；最新点已用实时交易所数值刷新"
        ).strip("；")
    return merged
