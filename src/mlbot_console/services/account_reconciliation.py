"""Reconciliation between exchange and local databases for console."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mlbot_console.services.db import query_rows
from mlbot_console.services.exchange_balances import fetch_scope_exchange_balance
from mlbot_console.services.symbols import is_all_symbols

logger = logging.getLogger(__name__)


def _spot_tracked_assets(
    spot_db: Optional[Path], spot_ledger_db: Optional[Path]
) -> Set[str]:
    """Assets the spot strategy ledger / orders care about (not full wallet)."""
    assets: Set[str] = set()
    if spot_ledger_db and spot_ledger_db.is_file():
        from mlbot_console.services.spot_ledger_book import fetch_spot_ledger_holdings

        for h in fetch_spot_ledger_holdings(spot_ledger_db, {}).get("holdings") or []:
            a = str(h.get("asset") or "").upper()
            if a:
                assets.add(a)
    if spot_db and spot_db.is_file():
        rows = query_rows(
            spot_db,
            "SELECT DISTINCT symbol FROM spot_orders WHERE status IN ('open','filled','partial')",
        )
        for row in rows:
            sym = str(row.get("symbol") or "").upper()
            if sym.endswith("USDT"):
                assets.add(sym[:-4])
            elif sym:
                assets.add(sym)
    return assets


def _local_trend_open_positions(
    trend_db: Optional[Path], *, symbol: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not trend_db or not Path(trend_db).is_file():
        return []
    from mlbot_console.services.account_summary import _trend_entry_qty_by_position

    sym = str(symbol or "").strip().upper()
    sym_filter = sym if sym and not is_all_symbols(sym) else None
    entry_qty = _trend_entry_qty_by_position(Path(trend_db), sym_filter)
    where = " WHERE lower(status) = 'open'"
    params: tuple[Any, ...] = ()
    if sym_filter:
        where += " AND symbol = ?"
        params = (sym_filter,)
    rows = query_rows(
        Path(trend_db),
        f"""
        SELECT position_id, symbol, side, current_size, entry_price, status, strategy_id
        FROM positions
        {where}
        """,
        params,
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        sym_u = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or "long").lower()
        if side in {"buy", "sell"}:
            side = "long" if side == "buy" else "short"
        try:
            qty = float(row.get("current_size") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            pid = str(row.get("position_id") or "")
            qty = float(entry_qty.get(pid) or 0.0)
        if qty <= 0:
            continue
        out.append(
            {
                "position_id": str(row.get("position_id") or ""),
                "symbol": sym_u,
                "side": side,
                "quantity": qty,
                "entry_price": float(row.get("entry_price") or 0.0),
                "strategy_id": str(row.get("strategy_id") or ""),
            }
        )
    return sorted(out, key=lambda x: (x["symbol"], x["side"], x["position_id"]))


def reconcile_account(
    scope: str,
    *,
    trend_db: Any = None,
    spot_db: Any = None,
    spot_ledger_db: Any = None,
    multi_leg_db: Any = None,
    mark_prices: Optional[Dict[str, float]] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Reconcile exchange state with local state for a given scope."""

    # Fetch exchange state
    exchange = fetch_scope_exchange_balance(
        scope, mark_prices=mark_prices, symbol=symbol
    )
    if not exchange.get("ok"):
        return {
            "scope": scope,
            "ok": False,
            "error": exchange.get("error"),
            "issues": [],
            "exchange_snapshot": exchange,
            "local_snapshot": {},
        }

    issues = []
    local_snapshot = {}

    if scope == "spot":
        from mlbot_console.services.spot_ledger_book import fetch_spot_ledger_holdings

        local_ledger = fetch_spot_ledger_holdings(spot_ledger_db, mark_prices or {})
        tracked = _spot_tracked_assets(spot_db, spot_ledger_db)
        local_snapshot = {
            **local_ledger,
            "tracked_assets": sorted(tracked),
        }

        exchange_holdings = {h["asset"]: h for h in exchange.get("holdings", [])}
        local_holdings = {h["asset"]: h for h in local_ledger.get("holdings", [])}

        # Only reconcile assets the strategy books; ignore stray wallet balances (e.g. fee ETH).
        compare_assets = tracked if tracked else set(local_holdings.keys())
        sym = str(symbol or "").strip().upper()
        if sym and not is_all_symbols(sym) and sym.endswith("USDT"):
            base = sym[:-4]
            if base:
                compare_assets = (
                    {base} if not compare_assets else compare_assets & {base}
                )

        for asset in compare_assets:
            ex_h = exchange_holdings.get(asset) or {"qty": 0.0, "value_usdt": 0.0}
            loc_h = local_holdings.get(asset) or {"qty": 0.0, "value_usdt": 0.0}

            qty_diff = float(ex_h.get("qty") or 0) - float(loc_h.get("qty") or 0)
            tol = max(
                1e-6,
                max(float(ex_h.get("qty") or 0), float(loc_h.get("qty") or 0)) * 0.001,
            )

            if abs(qty_diff) > tol:
                issues.append(
                    {
                        "kind": "qty_mismatch",
                        "asset": asset,
                        "exchange": ex_h.get("qty"),
                        "local": loc_h.get("qty"),
                        "delta": qty_diff,
                    }
                )

        min_usdt = float(os.getenv("MLBOT_RECON_WALLET_EXTRA_MIN_USDT", "5"))
        for asset, ex_h in exchange_holdings.items():
            if asset in compare_assets:
                continue
            val = float(ex_h.get("value_usdt") or 0)
            if val >= min_usdt:
                issues.append(
                    {
                        "kind": "wallet_extra",
                        "asset": asset,
                        "exchange": ex_h.get("qty"),
                        "local": 0.0,
                        "delta": ex_h.get("qty"),
                        "note": "交易所余额未纳入 spot_accum 母仓",
                    }
                )

    elif scope == "multi_leg":
        if multi_leg_db and multi_leg_db.is_file():
            rows = query_rows(
                multi_leg_db,
                """
                SELECT raw_json, created_at, strategy, symbol, ok
                FROM multi_leg_reconciliation_snapshots
                ORDER BY created_at DESC
                LIMIT 1
                """,
            )
            if not rows:
                local_snapshot = {
                    "note": "尚无引擎对账快照（multi_leg 进程运行后会写入）"
                }
            else:
                row = rows[0]
                local_snapshot = {
                    "last_reconciliation_at": row.get("created_at"),
                    "strategy": row.get("strategy"),
                    "symbol": row.get("symbol"),
                    "engine_ok": bool(row.get("ok")),
                }
                try:
                    report = json.loads(str(row.get("raw_json") or "{}"))
                    for m in report.get("missing_exchange_orders") or []:
                        if not isinstance(m, dict):
                            continue
                        issues.append(
                            {
                                "kind": "missing_exchange_order",
                                "order_id": m.get("order_id"),
                                "symbol": m.get("symbol"),
                                "side": m.get("side"),
                            }
                        )
                    for o in report.get("orphan_exchange_orders") or []:
                        if not isinstance(o, dict):
                            continue
                        issues.append(
                            {
                                "kind": "orphan_exchange_order",
                                "order_id": o.get("order_id") or o.get("orderId"),
                                "symbol": o.get("symbol"),
                                "side": o.get("side"),
                            }
                        )
                    for p in report.get("position_mismatches") or []:
                        if not isinstance(p, dict):
                            continue
                        ex_q = float(p.get("exchange_quantity") or 0)
                        loc_q = float(p.get("local_quantity") or 0)
                        issues.append(
                            {
                                "kind": "position_mismatch",
                                "symbol": p.get("symbol"),
                                "side": p.get("side"),
                                "exchange": ex_q,
                                "local": loc_q,
                                "delta": ex_q - loc_q,
                            }
                        )
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "Failed to parse multi_leg_reconciliation_snapshots: %s", exc
                    )
                    local_snapshot["parse_error"] = str(exc)

    elif scope == "trend":
        local_positions = _local_trend_open_positions(
            Path(str(trend_db)) if trend_db else None,
            symbol=symbol,
        )
        local_snapshot = {"open_positions": local_positions}
        exchange_positions = list(exchange.get("exchange_open_positions") or [])
        sym_u = str(symbol or "").strip().upper()
        if sym_u and not is_all_symbols(sym_u):
            exchange_positions = [
                p
                for p in exchange_positions
                if str(p.get("symbol") or "").upper() == sym_u
            ]
        local_by_key: Dict[str, float] = {}
        for p in local_positions:
            key = f"{p['symbol']}:{p['side']}"
            local_by_key[key] = local_by_key.get(key, 0.0) + float(p["quantity"])
        exchange_by_key: Dict[str, float] = {}
        for p in exchange_positions:
            key = f"{p['symbol']}:{p['side']}"
            exchange_by_key[key] = exchange_by_key.get(key, 0.0) + float(p["quantity"])

        all_keys = set(local_by_key) | set(exchange_by_key)
        for key in sorted(all_keys):
            sym, side = key.split(":", 1)
            ex_q = float(exchange_by_key.get(key) or 0.0)
            loc_q = float(local_by_key.get(key) or 0.0)
            tol = max(1e-6, max(ex_q, loc_q) * 0.02)
            if abs(ex_q - loc_q) <= tol:
                continue
            if ex_q > 0 and loc_q == 0:
                kind = "exchange_position_not_in_local_db"
            elif loc_q > 0 and ex_q == 0:
                kind = "local_position_not_on_exchange"
            else:
                kind = "position_mismatch"
            issues.append(
                {
                    "kind": kind,
                    "symbol": sym,
                    "side": side,
                    "exchange": ex_q,
                    "local": loc_q,
                    "delta": ex_q - loc_q,
                }
            )

    return {
        "scope": scope,
        "ok": len(issues) == 0,
        "issues": issues,
        "exchange_snapshot": exchange,
        "local_snapshot": local_snapshot,
    }


def reconcile_all_accounts(
    *,
    trend_db: Any = None,
    spot_db: Any = None,
    spot_ledger_db: Any = None,
    multi_leg_db: Any = None,
    feature_bus_root: Any = None,
    mark_prices: Optional[Dict[str, float]] = None,
    symbol: str = "*",
    lookback_days: int = 0,
    _account_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run engine reconciliation (spot/multi_leg) + PnL vs exchange + realized PnL for A/B/C."""
    import time

    from mlbot_console.services.account_pnl_reconciliation import (
        reconcile_pnl_vs_exchange,
        reconcile_realized_pnl,
    )
    from mlbot_console.services.account_summary import build_account_summary

    engine_by_scope: Dict[str, Any] = {}
    engine_issues: List[Dict[str, Any]] = []
    for scope in ("spot", "trend", "multi_leg"):
        res = reconcile_account(
            scope,
            trend_db=trend_db,
            spot_db=spot_db,
            spot_ledger_db=spot_ledger_db,
            multi_leg_db=multi_leg_db,
            mark_prices=mark_prices,
            symbol=symbol,
        )
        engine_by_scope[scope] = res
        for issue in res.get("issues") or []:
            engine_issues.append({**issue, "scope": scope, "layer": "engine"})

    pnl = reconcile_pnl_vs_exchange(
        trend_db=Path(str(trend_db)) if trend_db else Path("/dev/null"),
        spot_db=Path(str(spot_db)) if spot_db else Path("/dev/null"),
        spot_ledger_db=(
            Path(str(spot_ledger_db)) if spot_ledger_db else Path("/dev/null")
        ),
        multi_leg_db=Path(str(multi_leg_db)) if multi_leg_db else Path("/dev/null"),
        feature_bus_root=(
            Path(str(feature_bus_root)) if feature_bus_root else Path("/dev/null")
        ),
        symbol=symbol,
        lookback_days=lookback_days,
    )
    pnl_issues = [{**i, "layer": "pnl"} for i in (pnl.get("issues") or [])]

    # Realized PnL reconciliation for multi_leg scope
    now_ms = int(time.time() * 1000)
    effective_lookback = lookback_days if lookback_days > 0 else 90
    start_ms = now_ms - effective_lookback * 86_400_000

    realized_recon: Dict[str, Any] = {"ok": True, "scope": "multi_leg", "issues": []}
    try:
        # Reuse pre-built summary if available to avoid duplicate DB queries
        if _account_summary is None:
            summary = build_account_summary(
                trend_db=Path(str(trend_db)) if trend_db else Path("/dev/null"),
                spot_db=Path(str(spot_db)) if spot_db else Path("/dev/null"),
                spot_ledger_db=(
                    Path(str(spot_ledger_db)) if spot_ledger_db else Path("/dev/null")
                ),
                multi_leg_db=Path(str(multi_leg_db)) if multi_leg_db else Path("/dev/null"),
                feature_bus_root=(
                    Path(str(feature_bus_root)) if feature_bus_root else Path("/dev/null")
                ),
                symbol="*",
                lookback_days=lookback_days,
            )
        else:
            summary = _account_summary
        local_realized = 0.0
        for s in summary.get("scopes") or []:
            if str(s.get("scope")) == "multi_leg":
                local_realized = float(s.get("realized_pnl") or 0.0)
                break

        local_commission = 0.0
        if multi_leg_db:
            from mlbot_console.services.db import query_rows as _qr

            rows = _qr(
                Path(str(multi_leg_db)),
                "SELECT COALESCE(SUM(commission), 0) as total_commission "
                "FROM multi_leg_orders "
                "WHERE lower(status) IN ('filled', 'partially_filled') "
                "AND (error_message IS NULL OR error_message NOT LIKE '%bug%') "
                "AND filled_at >= datetime('now', ?)",
                (f"-{effective_lookback} days",),
            )
            if rows:
                local_commission = abs(float(rows[0].get("total_commission") or 0.0))

        realized_recon = reconcile_realized_pnl(
            scope="multi_leg",
            local_realized_pnl=local_realized,
            local_commission=local_commission,
            symbol="*",
            start_time_ms=start_ms,
            end_time_ms=now_ms,
        )
    except Exception as exc:
        logger.warning("reconcile_all_accounts: realized recon failed: %s", exc)
        realized_recon = {
            "ok": False,
            "scope": "multi_leg",
            "issues": [
                {
                    "kind": "realized_recon_error",
                    "scope": "multi_leg",
                    "message": str(exc),
                }
            ],
        }

    realized_issues = [
        {**i, "layer": "realized"} for i in (realized_recon.get("issues") or [])
    ]

    all_issues = engine_issues + pnl_issues + realized_issues
    ok = len(all_issues) == 0
    if not ok:
        logger.warning(
            "reconcile_all_accounts: %d issue(s) (engine=%d pnl=%d realized=%d)",
            len(all_issues),
            len(engine_issues),
            len(pnl_issues),
            len(realized_issues),
        )

    return {
        "ok": ok,
        "symbol": symbol,
        "lookback_days": lookback_days,
        "issues": all_issues,
        "engine": engine_by_scope,
        "pnl": pnl,
        "realized": realized_recon,
    }
