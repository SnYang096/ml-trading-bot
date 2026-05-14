from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator
from src.order_management.multi_leg_reconciliation import (
    LocalOrderSnapshot,
    LocalPositionSnapshot,
    MultiLegReconciler,
    ReconciliationPolicy,
)
from src.order_management.multi_leg_risk_governor import (
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)


@dataclass
class FakeEngine:
    orders: list[LocalOrderSnapshot] = field(default_factory=list)
    positions: list[LocalPositionSnapshot] = field(default_factory=list)
    execution_results: list[list[GridExecutionResult]] = field(default_factory=list)
    reconciliation_reports: list[Any] = field(default_factory=list)
    execution_reports: list[dict[str, Any]] = field(default_factory=list)

    def local_order_snapshots(self) -> list[LocalOrderSnapshot]:
        return list(self.orders)

    def local_position_snapshots(self) -> list[LocalPositionSnapshot]:
        return list(self.positions)

    def on_execution_results(self, results: list[GridExecutionResult]) -> None:
        self.execution_results.append(list(results))

    def on_reconciliation_report(self, report: Any) -> None:
        self.reconciliation_reports.append(report)

    def on_execution_report(self, report: dict[str, Any]) -> None:
        self.execution_reports.append(dict(report))


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.sync_open_orders.return_value = []
    adapter.sync_positions.return_value = []
    adapter.execute_actions.side_effect = lambda actions: [
        GridExecutionResult(
            action=a["action"],
            status="ok",
            symbol=a.get("symbol", ""),
            order_id=a.get("exchange_order_id", "ex_1"),
            client_order_id=a.get("client_order_id", "dat_x"),
            raw=dict(a),
        )
        for a in actions
    ]
    return adapter


def _orchestrator(engine: FakeEngine, adapter: MagicMock) -> MultiLegLiveOrchestrator:
    return MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(
            ReconciliationPolicy(client_id_prefixes={"dat_", "cg_"})
        ),
    )


def test_run_actions_executes_approved_actions_and_notifies_engine() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    orchestrator = _orchestrator(engine, adapter)

    report = orchestrator.run_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "price": 50_000.0,
                "client_order_id": "dat_add_1",
            }
        ]
    )

    assert report.risk.ok
    adapter.execute_actions.assert_called_once()
    assert (
        adapter.execute_actions.call_args.args[0][0]["client_order_id"] == "dat_add_1"
    )
    assert engine.execution_results[0][0].action == "place"
    assert engine.reconciliation_reports[0].ok


def test_run_actions_without_actions_can_skip_exchange_sync() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    orchestrator = _orchestrator(engine, adapter)

    report = orchestrator.run_actions([], reconcile=False)

    assert report.risk.ok
    assert report.execution_results == []
    assert report.reconciliation is None
    adapter.sync_open_orders.assert_not_called()
    adapter.sync_positions.assert_not_called()
    adapter.execute_actions.assert_not_called()


def test_run_actions_filters_rejected_opening_actions_before_adapter() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    orchestrator = _orchestrator(engine, adapter)

    report = orchestrator.run_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 1.0,
                "price": 50_000.0,
            }
        ]
    )

    assert not report.risk.ok
    assert report.execution_results == []
    adapter.execute_actions.assert_not_called()
    assert "max_gross_notional exceeded" in report.risk.rejected[0].reason


def test_reconcile_cancels_orphan_exchange_order_and_notifies_engine() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    adapter.sync_open_orders.return_value = [
        {
            "order_id": "ex_orphan",
            "client_order_id": "dat_orphan",
            "symbol": "BTCUSDT",
        }
    ]
    orchestrator = _orchestrator(engine, adapter)

    report, results = orchestrator.reconcile()

    assert len(report.orphan_exchange_orders) == 1
    assert report.suggested_actions[0]["exchange_order_id"] == "ex_orphan"
    assert results[0].action == "cancel"
    assert engine.reconciliation_reports[0] is report
    assert engine.execution_results[0][0].action == "cancel"


def test_reconcile_reports_position_drift_without_suggesting_position_action() -> None:
    engine = FakeEngine(positions=[LocalPositionSnapshot("BTCUSDT", "LONG", 0.01)])
    adapter = _adapter()
    adapter.sync_positions.return_value = [
        {
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "position_amount": 0.02,
            "mark_price": 50_000.0,
        }
    ]
    orchestrator = _orchestrator(engine, adapter)

    report, results = orchestrator.reconcile()

    assert len(report.position_mismatches) == 1
    assert results == []
    assert (
        engine.reconciliation_reports[0].position_mismatches[0].exchange_quantity
        == 0.02
    )


def test_user_stream_execution_report_is_forwarded_to_engine() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    orchestrator = _orchestrator(engine, adapter)

    orchestrator.on_execution_report(
        {
            "client_order_id": "dat_add_1",
            "status": "PARTIALLY_FILLED",
            "last_filled_qty": 0.003,
        }
    )

    assert engine.execution_reports == [
        {
            "client_order_id": "dat_add_1",
            "status": "PARTIALLY_FILLED",
            "last_filled_qty": 0.003,
        }
    ]


def test_run_actions_uses_exchange_positions_for_risk_projection() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    adapter.sync_positions.return_value = [
        {
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "position_amount": 0.015,
            "mark_price": 50_000.0,
        }
    ]
    orchestrator = _orchestrator(engine, adapter)

    report = orchestrator.run_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "price": 50_000.0,
            }
        ]
    )

    assert not report.risk.ok
    assert "max_gross_notional exceeded" in report.risk.rejected[0].reason
    adapter.execute_actions.assert_not_called()


def test_run_actions_passes_drawdown_provider_to_governor() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(
                max_gross_notional=10_000.0,
                max_net_notional=10_000.0,
                max_drawdown_pct=0.12,
            )
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(
            ReconciliationPolicy(client_id_prefixes={"dat_", "cg_"})
        ),
        drawdown_pct_provider=lambda: 0.12,
    )

    report = orchestrator.run_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "price": 50_000.0,
            }
        ]
    )

    assert not report.risk.ok
    assert "max_drawdown_pct exceeded" in report.risk.rejected[0].reason
    adapter.execute_actions.assert_not_called()
