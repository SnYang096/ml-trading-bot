"""Portfolio-level risk checks for multi-leg strategy actions.

This module sits above ``GridExecutionAdapter``. Strategy engines still own
their inventory state, while the governor blocks action batches that would push
the shared account beyond gross/net exposure or resting-order limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


Action = Dict[str, Any]


@dataclass(frozen=True)
class MultiLegRiskLimits:
    """Hard caps for a shared account running multi-leg strategies."""

    max_gross_notional: float
    max_net_notional: float
    max_symbol_gross_notional: Optional[float] = None
    max_symbol_net_notional: Optional[float] = None
    max_resting_orders: Optional[int] = None


@dataclass(frozen=True)
class ExposureSnapshot:
    """Current exchange or local exposure for one symbol/side."""

    symbol: str
    side: str
    quantity: float
    mark_price: float

    @property
    def notional(self) -> float:
        return abs(float(self.quantity) * float(self.mark_price))


@dataclass(frozen=True)
class RiskRejection:
    """One action rejected by the portfolio governor."""

    action: Action
    reason: str


@dataclass(frozen=True)
class RiskCheckResult:
    """Approved actions plus explicit rejection reasons."""

    approved_actions: List[Action] = field(default_factory=list)
    rejected: List[RiskRejection] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


class MultiLegPortfolioRiskGovernor:
    """Apply portfolio exposure limits to multi-leg action batches.

    ``cancel`` and ``market_exit`` actions are always allowed because they reduce
    risk or remove resting orders. ``place`` actions are admitted only if the
    resulting projected exposure remains within configured limits.
    """

    def __init__(self, limits: MultiLegRiskLimits) -> None:
        self.limits = limits

    def check_actions(
        self,
        actions: Iterable[Action],
        *,
        positions: Iterable[ExposureSnapshot] = (),
        open_orders: Iterable[Mapping[str, Any]] = (),
    ) -> RiskCheckResult:
        approved: List[Action] = []
        rejected: List[RiskRejection] = []
        long_by_symbol, short_by_symbol = _exposure_maps(positions)
        resting_orders = len(list(open_orders))

        for action in actions:
            kind = str(action.get("action", "") or "").lower()
            if kind in {"cancel", "market_exit"}:
                approved.append(dict(action))
                if kind == "cancel" and resting_orders > 0:
                    resting_orders -= 1
                continue
            if kind != "place":
                approved.append(dict(action))
                continue

            if (
                self.limits.max_resting_orders is not None
                and resting_orders + 1 > self.limits.max_resting_orders
            ):
                rejected.append(
                    RiskRejection(
                        dict(action),
                        f"max_resting_orders exceeded: {resting_orders + 1} > "
                        f"{self.limits.max_resting_orders}",
                    )
                )
                continue

            symbol = _required_symbol(action)
            side = str(action.get("side", "") or "").upper()
            notional = _action_notional(action)
            next_long = dict(long_by_symbol)
            next_short = dict(short_by_symbol)
            if side == "BUY":
                next_long[symbol] = next_long.get(symbol, 0.0) + notional
            elif side == "SELL":
                next_short[symbol] = next_short.get(symbol, 0.0) + notional
            else:
                rejected.append(RiskRejection(dict(action), f"unsupported place side: {side}"))
                continue

            reason = self._limit_violation(next_long, next_short, symbol)
            if reason:
                rejected.append(RiskRejection(dict(action), reason))
                continue

            approved.append(dict(action))
            long_by_symbol, short_by_symbol = next_long, next_short
            resting_orders += 1

        return RiskCheckResult(approved_actions=approved, rejected=rejected)

    def _limit_violation(
        self,
        long_by_symbol: Mapping[str, float],
        short_by_symbol: Mapping[str, float],
        symbol: str,
    ) -> str:
        gross = sum(long_by_symbol.values()) + sum(short_by_symbol.values())
        net = abs(sum(long_by_symbol.values()) - sum(short_by_symbol.values()))
        if gross > self.limits.max_gross_notional:
            return f"max_gross_notional exceeded: {gross:.8f} > {self.limits.max_gross_notional:.8f}"
        if net > self.limits.max_net_notional:
            return f"max_net_notional exceeded: {net:.8f} > {self.limits.max_net_notional:.8f}"

        sym_gross = long_by_symbol.get(symbol, 0.0) + short_by_symbol.get(symbol, 0.0)
        sym_net = abs(long_by_symbol.get(symbol, 0.0) - short_by_symbol.get(symbol, 0.0))
        if (
            self.limits.max_symbol_gross_notional is not None
            and sym_gross > self.limits.max_symbol_gross_notional
        ):
            return (
                f"max_symbol_gross_notional exceeded for {symbol}: "
                f"{sym_gross:.8f} > {self.limits.max_symbol_gross_notional:.8f}"
            )
        if (
            self.limits.max_symbol_net_notional is not None
            and sym_net > self.limits.max_symbol_net_notional
        ):
            return (
                f"max_symbol_net_notional exceeded for {symbol}: "
                f"{sym_net:.8f} > {self.limits.max_symbol_net_notional:.8f}"
            )
        return ""


def _exposure_maps(
    positions: Iterable[ExposureSnapshot],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    long_by_symbol: Dict[str, float] = {}
    short_by_symbol: Dict[str, float] = {}
    for pos in positions:
        symbol = str(pos.symbol).upper()
        side = str(pos.side).upper()
        if side == "LONG":
            long_by_symbol[symbol] = long_by_symbol.get(symbol, 0.0) + pos.notional
        elif side == "SHORT":
            short_by_symbol[symbol] = short_by_symbol.get(symbol, 0.0) + pos.notional
    return long_by_symbol, short_by_symbol


def _required_symbol(action: Mapping[str, Any]) -> str:
    symbol = str(action.get("symbol", "") or "").upper().strip()
    if not symbol:
        raise ValueError("multi-leg action requires symbol")
    return symbol


def _action_notional(action: Mapping[str, Any]) -> float:
    try:
        quantity = float(action.get("quantity"))
        price = float(action.get("price") or action.get("mark_price"))
    except (TypeError, ValueError) as exc:
        raise ValueError("place action requires numeric quantity and price") from exc
    if quantity <= 0 or price <= 0:
        raise ValueError("place action requires positive quantity and price")
    return abs(quantity * price)
