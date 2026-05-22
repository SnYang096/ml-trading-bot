"""Reconciliation between exchange and local databases for console."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mlbot_console.services.exchange_balances import fetch_scope_exchange_balance

logger = logging.getLogger(__name__)


def reconcile_account(
    scope: str,
    *,
    trend_db: Any = None,
    spot_db: Any = None,
    spot_ledger_db: Any = None,
    multi_leg_db: Any = None,
    mark_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Reconcile exchange state with local state for a given scope."""
    
    # Fetch exchange state
    exchange = fetch_scope_exchange_balance(scope, mark_prices=mark_prices)
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
        local_snapshot = local_ledger
        
        exchange_holdings = {h["asset"]: h for h in exchange.get("holdings", [])}
        local_holdings = {h["asset"]: h for h in local_ledger.get("holdings", [])}
        
        all_assets = set(exchange_holdings.keys()) | set(local_holdings.keys())
        
        for asset in all_assets:
            ex_h = exchange_holdings.get(asset) or {"qty": 0.0, "value_usdt": 0.0}
            loc_h = local_holdings.get(asset) or {"qty": 0.0, "value_usdt": 0.0}
            
            qty_diff = ex_h["qty"] - loc_h["qty"]
            # Tolerance: 1e-6 or 0.1% of qty
            tol = max(1e-6, max(ex_h["qty"], loc_h["qty"]) * 0.001)
            
            if abs(qty_diff) > tol:
                issues.append({
                    "kind": "qty_mismatch",
                    "asset": asset,
                    "exchange": ex_h["qty"],
                    "local": loc_h["qty"],
                    "delta": qty_diff,
                })
                
    elif scope == "multi_leg":
        # For multi-leg, we can read the latest reconciliation snapshot from the DB
        from mlbot_console.services.db import query_rows
        if multi_leg_db and multi_leg_db.is_file():
            rows = query_rows(
                multi_leg_db,
                "SELECT report_json, created_at FROM multi_leg_reconciliation_snapshots ORDER BY created_at DESC LIMIT 1"
            )
            if rows:
                import json
                try:
                    report = json.loads(rows[0]["report_json"])
                    local_snapshot = {"last_reconciliation_at": rows[0]["created_at"]}
                    
                    missing = report.get("missing_exchange_orders", [])
                    orphans = report.get("orphan_exchange_orders", [])
                    mismatches = report.get("position_mismatches", [])
                    
                    for m in missing:
                        issues.append({
                            "kind": "missing_exchange_order",
                            "order_id": m.get("order_id"),
                            "symbol": m.get("symbol"),
                            "side": m.get("side"),
                        })
                        
                    for o in orphans:
                        issues.append({
                            "kind": "orphan_exchange_order",
                            "order_id": o.get("order_id") or o.get("orderId"),
                            "symbol": o.get("symbol"),
                            "side": o.get("side"),
                        })
                        
                    for p in mismatches:
                        issues.append({
                            "kind": "position_mismatch",
                            "symbol": p.get("symbol"),
                            "side": p.get("side"),
                            "exchange": p.get("exchange_quantity"),
                            "local": p.get("local_quantity"),
                            "delta": float(p.get("exchange_quantity") or 0) - float(p.get("local_quantity") or 0),
                        })
                except Exception as e:
                    logger.warning("Failed to parse multi_leg_reconciliation_snapshots: %s", e)
                    
    elif scope == "trend":
        # For trend, we can compare positions table with exchange positions
        # But exchange_balances doesn't fetch positions yet, only account summary
        # So we skip detailed trend reconciliation for now
        pass

    return {
        "scope": scope,
        "ok": len(issues) == 0,
        "issues": issues,
        "exchange_snapshot": exchange,
        "local_snapshot": local_snapshot,
    }
