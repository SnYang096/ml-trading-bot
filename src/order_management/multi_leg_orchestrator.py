"""Live orchestration for multi-leg strategy engines.

The orchestrator composes strategy-owned state with shared account safety:

Engine -> Portfolio Governor -> Execution Adapter -> Reconciler -> Engine update

It intentionally keeps exchange transport and strategy inventory logic separate.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, runtime_checkable

from src.order_management.grid_execution_adapter import (
    MultiLegExecutionAdapter,
    MultiLegExecutionResult,
)
from src.order_management.multi_leg_reconciliation import (
    LocalOrderSnapshot,
    LocalPositionSnapshot,
    MultiLegReconciler,
    ReconciliationReport,
)
from src.order_management.multi_leg_risk_governor import (
    Action,
    ExposureSnapshot,
    MultiLegPortfolioRiskGovernor,
    RiskCheckResult,
)

logger = logging.getLogger(__name__)


def _reconcile_not_ok_warn_cooldown_s() -> float:
    raw = os.environ.get("MLBOT_MULTI_LEG_RECONCILE_WARN_COOLDOWN_SECONDS", "300")
    try:
        v = float(raw)
        return v if v >= 0.0 else 300.0
    except (TypeError, ValueError):
        return 300.0


@runtime_checkable
class MultiLegEngineProtocol(Protocol):
    """Minimal hooks a chop-grid or dual-add live engine can implement."""

    def local_order_snapshots(self) -> Iterable[LocalOrderSnapshot]:
        """Return strategy-owned pending/open order state."""

    def local_position_snapshots(self) -> Iterable[LocalPositionSnapshot]:
        """Return strategy-owned inventory state."""

    def on_execution_results(
        self, results: Iterable[MultiLegExecutionResult]
    ) -> None:
        """Receive adapter execution results for persisted id mapping."""

    def on_reconciliation_report(self, report: ReconciliationReport) -> None:
        """Receive exchange/local drift diagnostics."""

    def on_execution_report(self, report: Mapping[str, Any]) -> None:
        """Receive normalized user-stream execution reports."""


@dataclass(frozen=True)
class OrchestrationReport:
    """Summary of one orchestrated action/reconciliation pass."""

    risk: RiskCheckResult
    execution_results: List[MultiLegExecutionResult] = field(default_factory=list)
    reconciliation: Optional[ReconciliationReport] = None
    reconciliation_results: List[MultiLegExecutionResult] = field(default_factory=list)


class MultiLegLiveOrchestrator:
    """Coordinate multi-leg engine actions with portfolio risk and reconciliation."""

    def __init__(
        self,
        *,
        engine: object,
        governor: MultiLegPortfolioRiskGovernor,
        adapter: MultiLegExecutionAdapter,
        reconciler: MultiLegReconciler,
        execute_reconciliation_actions: bool = True,
        storage: Optional[Any] = None,
        run_id: Optional[str] = None,
        strategy_name: str = "",
        symbol: str = "",
        drawdown_pct_provider: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self.engine = engine
        self.governor = governor
        self.adapter = adapter
        self.reconciler = reconciler
        self.execute_reconciliation_actions = bool(execute_reconciliation_actions)
        self.storage = storage
        self.run_id = run_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.drawdown_pct_provider = drawdown_pct_provider
        self._last_reconcile_not_ok_warn_at: float = 0.0

    def run_actions(
        self,
        actions: Iterable[Action],
        *,
        exchange_orders: Optional[Iterable[Mapping[str, Any]]] = None,
        exchange_positions: Optional[Iterable[Mapping[str, Any]]] = None,
        reconcile: bool = True,
    ) -> OrchestrationReport:
        """Risk-check, execute, reconcile, then notify engine."""

        action_list = [dict(a) for a in actions]
        if not action_list and not reconcile:
            return OrchestrationReport(risk=RiskCheckResult())

        orders = (
            [dict(o) for o in exchange_orders]
            if exchange_orders is not None
            else self.adapter.sync_open_orders(None)
        )
        positions = (
            [dict(p) for p in exchange_positions]
            if exchange_positions is not None
            else self.adapter.sync_positions(None)
        )
        risk = self.governor.check_actions(
            action_list,
            positions=_exchange_positions_to_exposures(positions),
            open_orders=orders,
            drawdown_pct=self._current_drawdown_pct(),
        )
        execution_results = (
            self.adapter.execute_actions(risk.approved_actions)
            if risk.approved_actions
            else []
        )
        _call_optional(self.engine, "on_execution_results", execution_results)
        reconciliation = None
        reconciliation_results: List[MultiLegExecutionResult] = []
        if reconcile:
            # Reconcile against post-execution exchange truth. Pre-trade snapshots
            # (``orders`` / ``positions`` above) are only for risk projection; passing
            # them here marks freshly placed ids as missing and orphans on the next
            # cancel pass.
            reconcile_orders = orders
            reconcile_positions = positions
            if execution_results:
                reconcile_orders = self.adapter.sync_open_orders(None)
                reconcile_positions = self.adapter.sync_positions(None)
            reconciliation, reconciliation_results = self.reconcile(
                exchange_orders=reconcile_orders,
                exchange_positions=reconcile_positions,
            )
        self._persist_positions()
        return OrchestrationReport(
            risk=risk,
            execution_results=execution_results,
            reconciliation=reconciliation,
            reconciliation_results=reconciliation_results,
        )

    def _current_drawdown_pct(self) -> Optional[float]:
        if self.drawdown_pct_provider is None:
            return None
        try:
            v = self.drawdown_pct_provider()
        except Exception:
            return None
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def reconcile(
        self,
        *,
        exchange_orders: Optional[Iterable[Mapping[str, Any]]] = None,
        exchange_positions: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> tuple[ReconciliationReport, List[MultiLegExecutionResult]]:
        """Compare local engine state with exchange truth and optionally act."""

        orders = (
            [dict(o) for o in exchange_orders]
            if exchange_orders is not None
            else self.adapter.sync_open_orders(None)
        )
        positions = (
            [dict(p) for p in exchange_positions]
            if exchange_positions is not None
            else self.adapter.sync_positions(None)
        )
        report = self.reconciler.reconcile(
            local_orders=_call_snapshot(self.engine, "local_order_snapshots"),
            exchange_orders=orders,
            local_positions=_call_snapshot(self.engine, "local_position_snapshots"),
            exchange_positions=positions,
        )
        if not report.ok:
            cooldown = _reconcile_not_ok_warn_cooldown_s()
            now = time.monotonic()
            if cooldown <= 0.0 or (now - self._last_reconcile_not_ok_warn_at) >= cooldown:
                logger.warning(
                    "multi-leg reconcile not ok: strategy=%s symbol=%s "
                    "missing_exchange_orders=%d orphan_exchange_orders=%d "
                    "position_mismatches=%d",
                    self.strategy_name,
                    self.symbol,
                    len(report.missing_exchange_orders),
                    len(report.orphan_exchange_orders),
                    len(report.position_mismatches),
                )
                self._last_reconcile_not_ok_warn_at = now
            else:
                logger.debug(
                    "multi-leg reconcile not ok (suppressed, cooldown=%.0fs): "
                    "strategy=%s symbol=%s missing_exchange_orders=%d "
                    "orphan_exchange_orders=%d position_mismatches=%d",
                    cooldown,
                    self.strategy_name,
                    self.symbol,
                    len(report.missing_exchange_orders),
                    len(report.orphan_exchange_orders),
                    len(report.position_mismatches),
                )
        _call_optional(self.engine, "on_reconciliation_report", report)
        self._persist_reconciliation(report)

        results: List[MultiLegExecutionResult] = []
        if self.execute_reconciliation_actions and report.suggested_actions:
            logger.info(
                "multi-leg reconcile cancels: strategy=%s symbol=%s count=%d "
                "(orphan open orders on exchange not in engine state)",
                self.strategy_name,
                self.symbol,
                len(report.suggested_actions),
            )
            # Reconciliation actions are cancel-only by construction today. Route
            # them through the same adapter so client logging stays consistent.
            results = self.adapter.execute_actions(report.suggested_actions)
            _call_optional(self.engine, "on_execution_results", results)
        return report, results

    def on_execution_report(self, report: Mapping[str, Any]) -> None:
        """Forward user-stream execution updates and execute follow-up actions."""

        report_dict = dict(report)
        _call_optional(self.engine, "on_execution_report", report_dict)
        self._persist_execution_report(report_dict)
        follow_ups = _call_snapshot(self.engine, "pop_pending_actions")
        if follow_ups:
            results = self.adapter.execute_actions(follow_ups)
            _call_optional(self.engine, "on_execution_results", results)
        self._persist_positions()

    def _persist_reconciliation(self, report: ReconciliationReport) -> None:
        if self.storage is None:
            return
        raw = {
            "ok": report.ok,
            "missing_exchange_orders": [
                getattr(o, "__dict__", str(o)) for o in report.missing_exchange_orders
            ],
            "orphan_exchange_orders": list(report.orphan_exchange_orders),
            "position_mismatches": [
                getattr(m, "__dict__", str(m)) for m in report.position_mismatches
            ],
        }
        self.storage.record_reconciliation_snapshot(
            {
                "run_id": self.run_id,
                "strategy": self.strategy_name,
                "symbol": self.symbol,
                "ok": report.ok,
                "raw": raw,
            }
        )

    def _persist_execution_report(self, report: Mapping[str, Any]) -> None:
        if self.storage is None:
            return
        payload = {
            "run_id": self.run_id,
            "strategy": self.strategy_name,
            "symbol": report.get("symbol") or self.symbol,
            "order_id": report.get("order_id"),
            "client_order_id": report.get("client_order_id"),
            "status": report.get("status"),
            "execution_type": report.get("execution_type"),
            "event_time": report.get("event_time") or report.get("trade_time"),
            "trade_time": report.get("trade_time"),
            "filled_qty": report.get("filled_qty"),
            "avg_price": report.get("avg_price"),
            "last_filled_price": report.get("last_filled_price"),
            "commission": report.get("commission"),
            "commission_asset": report.get("commission_asset"),
            "reject_reason": report.get("reject_reason"),
            "error_message": report.get("error_message"),
            "raw": dict(report),
        }
        self.storage.record_execution_report(payload)
        apply_fn = getattr(self.storage, "apply_execution_report", None)
        if callable(apply_fn):
            apply_fn(payload)

    def _persist_positions(self) -> None:
        if self.storage is None:
            return
        state = getattr(self.engine, "state", None)
        inventory = list(getattr(state, "inventory", []) or [])
        active_leg_ids = []
        for idx, pos in enumerate(inventory):
            leg_id = str(
                getattr(pos, "leg_id", "") or f"{self.strategy_name}_{self.symbol}_{idx}"
            )
            active_leg_ids.append(leg_id)
            self.storage.upsert_position(
                {
                    "run_id": self.run_id,
                    "strategy": self.strategy_name,
                    "leg_id": leg_id,
                    "symbol": getattr(pos, "symbol", self.symbol),
                    "side": getattr(pos, "side", ""),
                    "entry_price": getattr(pos, "entry_price", 0.0),
                    "quantity": getattr(pos, "quantity", 0.0),
                    "status": "open",
                    "protection_order_ids": getattr(pos, "protection_order_ids", []),
                    "raw": getattr(pos, "__dict__", {}),
                }
            )
        close_absent = getattr(self.storage, "close_absent_positions", None)
        if callable(close_absent):
            close_absent(
                strategy=self.strategy_name,
                symbol=self.symbol,
                active_leg_ids=active_leg_ids,
                run_id=self.run_id,
            )


def _call_optional(target: object, method_name: str, arg: object) -> None:
    method = getattr(target, method_name, None)
    if callable(method):
        method(arg)


def _call_snapshot(target: object, method_name: str) -> List[Any]:
    method = getattr(target, method_name, None)
    if not callable(method):
        return []
    return list(method())


def _exchange_positions_to_exposures(
    positions: Iterable[Mapping[str, Any]],
) -> List[ExposureSnapshot]:
    exposures: List[ExposureSnapshot] = []
    for pos in positions:
        symbol = str(pos.get("symbol", "") or "").upper()
        if not symbol:
            continue
        quantity = _position_quantity(pos)
        if quantity <= 0:
            continue
        side = _position_side(pos)
        mark_price = _position_mark_price(pos, quantity)
        if mark_price <= 0:
            continue
        exposures.append(
            ExposureSnapshot(
                symbol=symbol,
                side=side,
                quantity=quantity,
                mark_price=mark_price,
            )
        )
    return exposures


def _position_quantity(pos: Mapping[str, Any]) -> float:
    try:
        return abs(float(pos.get("position_amount") or pos.get("positionAmt") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _position_side(pos: Mapping[str, Any]) -> str:
    side = str(pos.get("position_side") or pos.get("positionSide") or "").upper()
    if side in {"LONG", "SHORT"}:
        return side
    try:
        amount = float(pos.get("position_amount") or pos.get("positionAmt") or 0.0)
    except (TypeError, ValueError):
        amount = 0.0
    return "LONG" if amount >= 0 else "SHORT"


def _position_mark_price(pos: Mapping[str, Any], quantity: float) -> float:
    for key in ("mark_price", "markPrice", "entry_price", "entryPrice"):
        try:
            value = float(pos.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    for key in ("notional", "position_notional", "positionInitialMargin"):
        try:
            value = abs(float(pos.get(key) or 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0 and quantity > 0:
            return value / quantity
    return 0.0
