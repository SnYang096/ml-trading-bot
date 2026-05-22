"""Read spot_accum_ledger to provide ledger-based holdings and equity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from mlbot_console.services.db import query_rows


def _position_row(key: str, lot_data: Mapping[str, Any]) -> tuple[str, float, float, float, float] | None:
    """Parse one spot_accum_ledger position (live runner or legacy lot id)."""
    if not isinstance(lot_data, dict):
        return None
    sym = str(lot_data.get("symbol") or key or "").upper()
    if not sym:
        return None
    if not sym.endswith("USDT") and str(key or "").upper().endswith("USDT"):
        sym = str(key).upper()
    qty = float(lot_data.get("_qty_base") or lot_data.get("qty_base") or 0.0)
    if qty <= 0.0:
        return None
    notional = float(
        lot_data.get("_entry_notional_usdt") or lot_data.get("entry_notional_usdt") or 0.0
    )
    deploy = float(
        lot_data.get("_spot_quote_deployed")
        or lot_data.get("deploy_usdt")
        or notional
    )
    vwap = float(lot_data.get("vwap_entry") or 0.0)
    if vwap <= 0.0 and notional > 0.0:
        vwap = notional / qty
    return sym, qty, notional, deploy, vwap


def fetch_spot_ledger_holdings(
    spot_ledger_db: Path,
    mark_prices: Mapping[str, float],
) -> Dict[str, Any]:
    """Read spot_accum_ledger positions and calculate ledger equity."""
    if not spot_ledger_db.is_file():
        return {
            "holdings": [],
            "holdings_value_usdt": 0.0,
        }

    # Read state_kv positions
    rows = query_rows(spot_ledger_db, "SELECT v FROM state_kv WHERE k='positions'")
    
    holdings = []
    holdings_value_usdt = 0.0
    
    if rows:
        try:
            positions = json.loads(rows[0]["v"])
            if not isinstance(positions, dict):
                positions = {}
            for lot_id, lot_data in positions.items():
                parsed = _position_row(str(lot_id), lot_data)
                if parsed is None:
                    continue
                sym, qty, notional, deploy, vwap = parsed
                asset = sym[:-4] if sym.endswith("USDT") else sym
                px = float(mark_prices.get(sym) or mark_prices.get(asset) or 0.0)
                val = qty * px
                holdings_value_usdt += val
                holdings.append(
                    {
                        "asset": asset,
                        "symbol": sym,
                        "qty": qty,
                        "cost_basis": vwap,
                        "deploy_usdt": deploy,
                        "price_usdt": px,
                        "value_usdt": val,
                        "unrealized_pnl_usdt": val - notional if px > 0 else 0.0,
                    }
                )
        except Exception:
            pass

    return {
        "holdings": sorted(holdings, key=lambda x: x["value_usdt"], reverse=True),
        "holdings_value_usdt": holdings_value_usdt,
    }
