"""Read spot_accum_ledger to provide ledger-based holdings and equity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from mlbot_console.services.db import query_rows


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
            for lot_id, lot_data in positions.items():
                sym = str(lot_data.get("symbol") or "").upper()
                if not sym:
                    continue
                asset = sym[:-4] if sym.endswith("USDT") else sym
                qty = float(lot_data.get("qty_base") or 0.0)
                if qty <= 0:
                    continue
                
                # Try to find mark price
                px = float(mark_prices.get(sym) or mark_prices.get(asset) or 0.0)
                val = qty * px
                holdings_value_usdt += val
                
                holdings.append({
                    "asset": asset,
                    "symbol": sym,
                    "qty": qty,
                    "cost_basis": float(lot_data.get("vwap_entry") or 0.0),
                    "deploy_usdt": float(lot_data.get("entry_notional_usdt") or 0.0),
                    "price_usdt": px,
                    "value_usdt": val,
                    "unrealized_pnl_usdt": val - float(lot_data.get("entry_notional_usdt") or 0.0) if px > 0 else 0.0,
                })
        except Exception:
            pass

    return {
        "holdings": sorted(holdings, key=lambda x: x["value_usdt"], reverse=True),
        "holdings_value_usdt": holdings_value_usdt,
    }
