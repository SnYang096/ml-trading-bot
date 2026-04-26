"""Live orchestration for multi-leg strategy engines.

The orchestrator composes strategy-owned state with shared account safety:

Engine -> Portfolio Governor -> Execution Adapter -> Reconciler -> Engine update

It intentionally keeps exchange transport and strategy inventory logic separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, runtime_checkable

from src.order_management.grid_execution_adapter import (
    GridExecutionAdapter,
    GridExecutionResult,
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


@runtime_checkable
class MultiLegEngineProtocol(Protocol):
    """Minimal hooks a chop-grid or dual-add live engine can implement."""

    def local_order_snapshots(self) -> Iterable[LocalOrderSnapshot]:
        """Return strategy-owned pending/open order state."""

    def local_position_snapshots(self) -> Iterable[LocalPositionSnapshot]:
        """Return strategy-owned inventory state."""

    def on_execution_results(self, results: Iterable[GridExecutionResult]) -> None:
        """Receive adapter execution results for persisted id mapping."""

    def on_reconciliation_report(self, report: ReconciliationReport) -> None:
        """Receive exchange/local drift diagnostics."""

    def on_execution_report(self, report: Mapping[str, Any]) -> None:
        """Receive normalized user-stream execution reports."""


@dataclass(frozen=True)
class OrchestrationReport:
    """Summary of one orchestrated action/reconciliation pass."""

    risk: RiskCheckResult
    execution_results: List[GridExecutionResult] = field(default_factory=list)
    reconciliation: Optional[ReconciliationReport] = None
    reconciliation_results: List[GridExecutionResult] = field(default_factory=list)


class MultiLegLiveOrchestrator:
    """Coordinate multi-leg engine actions with portfolio risk and reconciliation."""

    def __init__(
        self,
        *,
        engine: object,
        governor: MultiLegPortfolioRiskGovernor,
        adapter: GridExecutionAdapter,
        reconciler: MultiLegReconciler,
        execute_reconciliation_actions: bool = True,
    ) -> None:
        self.engine = engine
        self.governor = governor
        self.adapter = adapter
        self.reconciler = reconciler
        self.execute_reconciliation_actions = bool(execute_reconciliation_actions)

    def run_actions(self, actions: Iterable[Action]) -> OrchestrationReport:
        """Risk-check, execute, reconcile, then notify engine."""

        action_list = [dict(a) for a in actions]
        exchange_orders = self.adapter.sync_open_orders(None)
        exchange_positions = self.adapter.sync_positions(None)
        risk = self.governor.check_actions(
            action_list,
            positions=_exchange_positions_to_exposures(exchange_positions),
            open_orders=exchange_orders,
        )
        execution_results = (
            self.adapter.execute_actions(risk.approved_actions)
            if risk.approved_actions
            else []
        )
        _call_optional(self.engine, "on_execution_results", execution_results)
        reconciliation, reconciliation_results = self.reconcile(
            exchange_orders=exchange_orders,
            exchange_positions=exchange_positions,
        )
        return OrchestrationReport(
            risk=risk,
            execution_results=execution_results,
            reconciliation=reconciliation,
            reconciliation_results=reconciliation_results,
        )

    def reconcile(
        self,
        *,
        exchange_orders: Optional[Iterable[Mapping[str, Any]]] = None,
        exchange_positions: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> tuple[ReconciliationReport, List[GridExecutionResult]]:
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
        _call_optional(self.engine, "on_reconciliation_report", report)

        results: List[GridExecutionResult] = []
        if self.execute_reconciliation_actions and report.suggested_actions:
            # Reconciliation actions are cancel-only by construction today. Route
            # them through the same adapter so client logging stays consistent.
            results = self.adapter.execute_actions(report.suggested_actions)
            _call_optional(self.engine, "on_execution_results", results)
        return report, results

    def on_execution_report(self, report: Mapping[str, Any]) -> None:
        """Forward normalized user-stream execution updates to the engine."""

        _call_optional(self.engine, "on_execution_report", dict(report))


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
