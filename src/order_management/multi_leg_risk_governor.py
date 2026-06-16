"""Portfolio-level risk checks for multi-leg strategy actions.

This module sits above ``GridExecutionAdapter``. Strategy engines still own
their inventory state, while the governor blocks action batches that would push
the shared account beyond gross/net exposure or resting-order limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from src.order_management.multi_leg_kill_switch import MultiLegKillSwitchTracker
from src.time_series_model.core.constitution.account_risk_guard import (
    AccountRiskSnapshot,
    evaluate_account_risk,
    resolve_account_risk_limits,
    snapshot_for_backtest,
)

logger = logging.getLogger(__name__)


Action = Dict[str, Any]


@dataclass(frozen=True)
class MultiLegRiskLimits:
    """Hard caps for the dedicated multi-leg account.

    Notional fields are USDT exposure ceilings. They do not reuse classic trend
    slots/risk; callers may derive them from a separate account equity budget.
    """

    max_gross_notional: float
    max_net_notional: float
    max_symbol_gross_notional: Optional[float] = None
    max_symbol_net_notional: Optional[float] = None
    max_resting_orders: Optional[int] = None
    account_equity_usdt: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    account_risk_limits: Optional[Dict[str, Any]] = None


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

    def __init__(
        self,
        limits: MultiLegRiskLimits,
        *,
        account_snapshot_provider: Optional[Callable[[], AccountRiskSnapshot]] = None,
        kill_switch_tracker: Optional[MultiLegKillSwitchTracker] = None,
        on_halt_change: Optional[Callable] = None,
    ) -> None:
        self.limits = limits
        self._account_risk_limits = resolve_account_risk_limits(
            limits.account_risk_limits
        )
        self._account_snapshot_provider = account_snapshot_provider
        self._kill_switch_tracker = kill_switch_tracker
        if kill_switch_tracker is not None and on_halt_change is not None:
            kill_switch_tracker.on_halt_change = on_halt_change

    def check_actions(
        self,
        actions: Iterable[Action],
        *,
        positions: Iterable[ExposureSnapshot] = (),
        open_orders: Iterable[Mapping[str, Any]] = (),
        drawdown_pct: Optional[float] = None,
    ) -> RiskCheckResult:
        approved: List[Action] = []
        rejected: List[RiskRejection] = []
        long_by_symbol, short_by_symbol = _exposure_maps(positions)
        resting_orders = len(list(open_orders))

        tracker = self._kill_switch_tracker
        if tracker is not None:
            tracker.begin_batch()
            self._refresh_kill_switch_from_account(tracker)

        effective_drawdown = drawdown_pct
        if tracker is not None and tracker.drawdown_pct is not None:
            effective_drawdown = tracker.drawdown_pct

        for action in actions:
            kind = str(action.get("action", "") or "").lower()
            if kind in {"cancel", "market_exit", "cancel_protection"}:
                approved.append(dict(action))
                if kind == "cancel" and resting_orders > 0:
                    resting_orders -= 1
                continue

            kill_reason = tracker.blocks_action(kind) if tracker is not None else None
            if kill_reason:
                rejected.append(RiskRejection(dict(action), kill_reason))
                continue

            if kind == "place_protection":
                approved.append(dict(action))
                continue

            if kind != "place":
                approved.append(dict(action))
                continue

            if (
                self.limits.max_drawdown_pct is not None
                and effective_drawdown is not None
                and float(effective_drawdown) >= float(self.limits.max_drawdown_pct)
            ):
                rejected.append(
                    RiskRejection(
                        dict(action),
                        f"max_drawdown_pct exceeded: {float(effective_drawdown):.4f} >= "
                        f"{float(self.limits.max_drawdown_pct):.4f}",
                    )
                )
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
                rejected.append(
                    RiskRejection(dict(action), f"unsupported place side: {side}")
                )
                continue

            reason = self._limit_violation(next_long, next_short, symbol)
            if reason:
                rejected.append(RiskRejection(dict(action), reason))
                continue

            account_reason = self._account_risk_violation(
                proposed_notional=notional,
                projected_gross=sum(next_long.values()) + sum(next_short.values()),
            )
            if account_reason:
                rejected.append(RiskRejection(dict(action), account_reason))
                continue

            approved.append(dict(action))
            long_by_symbol, short_by_symbol = next_long, next_short
            resting_orders += 1

        return RiskCheckResult(approved_actions=approved, rejected=rejected)

    def _refresh_kill_switch_from_account(
        self, tracker: MultiLegKillSwitchTracker
    ) -> None:
        if self._account_snapshot_provider is None:
            return
        try:
            snap = self._account_snapshot_provider()
        except Exception:
            logger.warning(
                "multi-leg kill-switch: account snapshot failed", exc_info=True
            )
            return
        if snap is None or float(snap.equity or 0.0) <= 0:
            return
        tracker.update_from_equity(float(snap.equity))

    def _account_risk_violation(
        self,
        *,
        proposed_notional: float,
        projected_gross: float,
    ) -> str:
        if not bool(self._account_risk_limits.get("enabled", False)):
            return ""
        equity = float(self.limits.account_equity_usdt or 0.0)
        snap: Optional[AccountRiskSnapshot] = None
        if self._account_snapshot_provider is not None:
            try:
                snap = self._account_snapshot_provider()
                if snap is not None and float(snap.equity or 0.0) > 0:
                    equity = float(snap.equity)
            except Exception:
                if bool(self._account_risk_limits.get("fail_closed", True)):
                    return "account_risk_snapshot_unavailable"
                return ""
        if equity <= 0:
            if bool(self._account_risk_limits.get("fail_closed", True)):
                return "account_equity_unavailable"
            return ""
        if snap is None:
            snap = snapshot_for_backtest(
                equity_usdt=equity,
                gross_notional=max(0.0, projected_gross - proposed_notional),
                margin_stress_leverage=float(
                    self._account_risk_limits.get("margin_stress_leverage", 5.0) or 5.0
                ),
            )
        violations = evaluate_account_risk(
            limits=self._account_risk_limits,
            snapshot=snap,
            proposed_notional=proposed_notional,
        )
        if violations:
            return "account_risk_limit: " + violations[0]
        return ""

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
        sym_net = abs(
            long_by_symbol.get(symbol, 0.0) - short_by_symbol.get(symbol, 0.0)
        )
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
