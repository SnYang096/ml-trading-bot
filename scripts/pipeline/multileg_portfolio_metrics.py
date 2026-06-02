"""Equal-weight portfolio metrics for multi-leg backtests.

Each symbol runs on its own capital bucket (``pnl_per_capital`` per trade is
normalized by that strategy's gross-exposure cap). Summing all trades across
symbols **overstates** portfolio return; the canonical portfolio metric is the
**arithmetic mean** of per-symbol cumulative ``pnl_per_capital``.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def portfolio_pnl_from_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    """Compute pooled vs equal-weight portfolio return from trade rows."""
    if trades.empty or "pnl_per_capital" not in trades.columns:
        return {
            "n_symbols": 0,
            "sum_pnl_per_capital_pooled": 0.0,
            "portfolio_pnl_per_capital": 0.0,
            "return_pct_pooled": 0.0,
            "return_pct": 0.0,
            "per_symbol_pnl_per_capital": {},
        }

    pc = pd.to_numeric(trades["pnl_per_capital"], errors="coerce").fillna(0.0)
    pooled = float(pc.sum())
    if "symbol" in trades.columns:
        per_sym = trades.assign(_pc=pc).groupby("symbol", sort=True)["_pc"].sum()
        n = int(len(per_sym))
        portfolio = float(per_sym.mean()) if n else 0.0
        per_symbol = {str(k): float(v) for k, v in per_sym.items()}
    else:
        n = 1
        portfolio = pooled
        per_symbol = {}

    return {
        "n_symbols": n,
        "sum_pnl_per_capital_pooled": pooled,
        "portfolio_pnl_per_capital": portfolio,
        "return_pct_pooled": pooled * 100.0,
        "return_pct": portfolio * 100.0,
        "per_symbol_pnl_per_capital": per_symbol,
    }


def dual_add_summary_fields(
    trades: pd.DataFrame, segments: pd.DataFrame
) -> Dict[str, Any]:
    """Shared one-row summary fields for dual-add / chop-grid aligned reports."""
    agg = portfolio_pnl_from_trades(trades)
    row: Dict[str, Any] = {
        "segments": len(segments),
        "trades": len(trades),
        "n_symbols": agg["n_symbols"],
        "sum_pnl_per_capital": agg["portfolio_pnl_per_capital"],
        "sum_pnl_per_capital_pooled": agg["sum_pnl_per_capital_pooled"],
        "return_pct": agg["return_pct"],
        "return_pct_pooled": agg["return_pct_pooled"],
    }
    if trades.empty or segments.empty:
        return row
    row.update(
        {
            "trade_win_rate": (trades["pnl_pct"] > 0).mean(),
            "segment_win_rate": (segments["pnl_per_capital"] > 0).mean(),
            "worst_segment": segments["pnl_per_capital"].min(),
            "median_drawdown": segments["max_drawdown"].median(),
        }
    )
    return row
