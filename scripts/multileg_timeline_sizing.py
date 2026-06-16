"""Compound sizing sync for multileg timeline backtest."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.config.multileg_sizing import resolve_multi_leg_unit_notionals
from src.order_management.multi_leg_risk_governor import (
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)


def build_risk_limits(
    *,
    ml: Mapping[str, Any],
    const: Mapping[str, Any],
    equity_usdt: float,
) -> MultiLegRiskLimits:
    rs = ml.get("risk_limits", {}) or {}
    ks = const.get("kill_switch", {}) or {}
    max_gross = float(rs.get("max_gross_notional_pct", 2.70)) * equity_usdt
    max_dd = float(ks.get("max_dd", 0.20))
    acct = ml.get("account", {}) or {}
    if acct.get("max_drawdown_pct") is not None:
        max_dd = min(max_dd, float(acct["max_drawdown_pct"]))
    sym_gross = rs.get("max_symbol_gross_notional_pct")
    sym_net = rs.get("max_symbol_net_notional_pct")
    return MultiLegRiskLimits(
        max_gross_notional=max_gross,
        max_net_notional=float(rs.get("max_net_notional_pct", 0.75)) * equity_usdt,
        max_symbol_gross_notional=(
            float(sym_gross) * equity_usdt if sym_gross is not None else max_gross
        ),
        max_symbol_net_notional=(
            float(sym_net) * equity_usdt if sym_net is not None else max_gross * 0.66
        ),
        max_resting_orders=int(rs.get("max_resting_orders", 60)),
        account_equity_usdt=equity_usdt,
        max_drawdown_pct=max_dd,
        account_risk_limits=const.get("account_risk"),
    )


def sync_multileg_timeline_sizing(
    *,
    engines: Dict[str, Dict[str, Any]],
    ml: Mapping[str, Any],
    const: Mapping[str, Any],
    equity_usdt: float,
    initial_equity: float,
    compound_sizing: bool,
    governors: Optional[Iterable[MultiLegPortfolioRiskGovernor]] = None,
) -> Dict[str, float]:
    """Refresh per-engine notionals and governor limits from current equity."""
    anchor = float(equity_usdt if compound_sizing else initial_equity or equity_usdt)
    anchor = max(1.0, anchor)
    units = resolve_multi_leg_unit_notionals(
        ml,
        equity_usdt=anchor,
        strategies=["chop_grid", "trend_scalp"],
    )
    chop_u = float(units.get("chop_grid", 4000.0))
    trend_u = float(units.get("trend_scalp", 9000.0))
    for engs in engines.values():
        chop = engs.get("chop")
        trend = engs.get("trend")
        if chop is not None:
            chop.level_notional = chop_u
        if trend is not None:
            trend.unit_notional = trend_u

    limits = build_risk_limits(ml=ml, const=const, equity_usdt=anchor)
    if governors is not None:
        for gov in governors:
            object.__setattr__(gov, "limits", limits)
    return {"chop_grid": chop_u, "trend_scalp": trend_u, "anchor_equity": anchor}
