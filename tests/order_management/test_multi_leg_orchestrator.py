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
from src.order_management.multi_leg_storage import MultiLegStorage


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


def test_execution_report_enriched_with_db_purpose(tmp_path) -> None:
    engine = FakeEngine()
    adapter = _adapter()
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["SOLUSDT"],
        account_label="multi_leg_testnet",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "SOLUSDT_grid_L1_sl",
            "leg_id": "SOLUSDT_grid_L1",
            "symbol": "SOLUSDT",
            "side": "LONG",
            "position_side": "LONG",
            "order_type": "stop_market",
            "purpose": "stop_loss",
            "quantity": 1.13,
            "client_order_id": "cg_de1f197df8e3",
            "exchange_order_id": "2000001079676592",
            "status": "filled",
        }
    )
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(
            ReconciliationPolicy(client_id_prefixes={"dat_", "cg_"})
        ),
        storage=storage,
        strategy_name="chop_grid",
        symbol="SOLUSDT",
    )

    orchestrator.on_execution_report(
        {
            "order_id": "2000001079676592",
            "client_order_id": "cg_de1f197df8e3",
            "status": "FILLED",
            "filled_qty": 0.37,
        }
    )

    assert engine.execution_reports[0]["protection_type"] == "stop_loss"


def test_execution_report_enriched_with_order_type_when_db_misses() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    orchestrator = _orchestrator(engine, adapter)

    orchestrator.on_execution_report(
        {
            "order_id": "algo_99",
            "status": "FILLED",
            "filled_qty": 0.5,
            "order_type": "STOP_MARKET",
        }
    )

    assert engine.execution_reports[0]["protection_type"] == "stop_loss"


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


def test_run_actions_reconcile_refreshes_exchange_orders_after_place() -> None:
    """Post-place reconcile must not reuse the pre-trade open-order snapshot."""
    engine = FakeEngine()
    adapter = _adapter()
    sync_calls = {"n": 0}

    def _sync_open_orders(_symbol: Any) -> list[dict[str, Any]]:
        sync_calls["n"] += 1
        if sync_calls["n"] == 1:
            return []
        return [
            {
                "order_id": "ex_new",
                "client_order_id": "cg_new",
                "symbol": "BTCUSDT",
            }
        ]

    adapter.sync_open_orders.side_effect = _sync_open_orders

    def _on_execution_results(results: list[GridExecutionResult]) -> None:
        engine.execution_results.append(list(results))
        if results and results[0].action == "place":
            engine.orders = [
                LocalOrderSnapshot(
                    order_id="local_1",
                    symbol="BTCUSDT",
                    side="BUY",
                    quantity=0.01,
                    price=50_000.0,
                    exchange_order_id="ex_new",
                    client_order_id="cg_new",
                )
            ]

    engine.on_execution_results = _on_execution_results  # type: ignore[method-assign]
    orchestrator = _orchestrator(engine, adapter)

    report = orchestrator.run_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "price": 50_000.0,
                "client_order_id": "cg_new",
            }
        ]
    )

    assert report.reconciliation is not None
    assert report.reconciliation.ok
    assert sync_calls["n"] >= 2
    assert adapter.execute_actions.call_count == 1
    assert engine.orders and engine.orders[0].exchange_order_id == "ex_new"


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


def test_reconcile_merges_db_open_row_when_engine_state_empty(tmp_path) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    engine = FakeEngine(orders=[])
    adapter = _adapter()
    adapter.sync_open_orders.return_value = [
        {
            "order_id": "90489849398",
            "client_order_id": "cg_16738f8fae98",
            "symbol": "BNBUSDT",
        }
    ]
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    run_id = storage.create_run(
        mode="testnet", strategies=["chop_grid"], symbols=["BNBUSDT"]
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_BNBUSDT_S1_tp",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "order_type": "limit",
            "quantity": 0.1,
            "price": 643.0,
            "exchange_order_id": "90489849398",
            "client_order_id": "cg_16738f8fae98",
            "status": "open",
        }
    )
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(ReconciliationPolicy(client_id_prefixes={"cg_"})),
        storage=storage,
        strategy_name="chop_grid",
        symbol="BNBUSDT",
    )

    report, _ = orchestrator.reconcile()

    assert report.ok
    assert not report.orphan_exchange_orders


def test_reconcile_enriches_engine_row_missing_exchange_id(tmp_path) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    engine = FakeEngine(
        orders=[
            LocalOrderSnapshot(
                order_id="cg_BNBUSDT_S1_tp",
                symbol="BNBUSDT",
                side="SELL",
                quantity=0.1,
                price=643.0,
                exchange_order_id="",
                client_order_id="cg_16738f8fae98",
            )
        ]
    )
    adapter = _adapter()
    adapter.sync_open_orders.return_value = [
        {
            "order_id": "90489849398",
            "client_order_id": "cg_16738f8fae98",
            "symbol": "BNBUSDT",
        }
    ]
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    run_id = storage.create_run(
        mode="testnet", strategies=["chop_grid"], symbols=["BNBUSDT"]
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_BNBUSDT_S1_tp",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "order_type": "limit",
            "quantity": 0.1,
            "price": 643.0,
            "exchange_order_id": "90489849398",
            "client_order_id": "cg_16738f8fae98",
            "status": "open",
        }
    )
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(ReconciliationPolicy(client_id_prefixes={"cg_"})),
        storage=storage,
        strategy_name="chop_grid",
        symbol="BNBUSDT",
    )

    report, _ = orchestrator.reconcile()

    assert report.ok
    assert not report.orphan_exchange_orders


def test_reconcile_publishes_unified_reconciliation_metrics(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _capture(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        "src.order_management.multi_leg_orchestrator.publish_reconciliation_metrics",
        _capture,
    )

    engine = FakeEngine(
        orders=[
            LocalOrderSnapshot(
                order_id="local_1",
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.01,
                price=50_000.0,
                exchange_order_id="ex_missing",
                client_order_id="dat_local_1",
            )
        ]
    )
    adapter = _adapter()
    adapter.sync_open_orders.return_value = []
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(
            ReconciliationPolicy(client_id_prefixes={"dat_", "cg_"})
        ),
        strategy_name="dual_add_trend",
        symbol="BTCUSDT",
    )

    report, _ = orchestrator.reconcile()

    assert not report.ok
    assert len(calls) == 1
    assert calls[0]["scope"] == "hedge"
    assert calls[0]["strategy"] == "dual_add_trend"
    assert calls[0]["symbol"] == "BTCUSDT"
    assert calls[0]["ok"] is False
    assert calls[0]["issue_counts"] == {
        "missing_exchange_order": 1,
        "orphan_exchange_order": 0,
        "position_mismatch": 0,
    }
    assert calls[0]["source"] == "multi_leg_orchestrator"


@dataclass
class FollowUpEngine(FakeEngine):
    pending_rounds: list[list[dict[str, Any]]] = field(default_factory=list)
    _pop_idx: int = 0

    def pop_pending_actions(self) -> list[dict[str, Any]]:
        if self._pop_idx >= len(self.pending_rounds):
            return []
        out = list(self.pending_rounds[self._pop_idx])
        self._pop_idx += 1
        return out


def test_follow_up_place_protection_allowed_when_kill_switch_halted(
    tmp_path,
) -> None:
    from datetime import datetime, timezone

    from src.order_management.multi_leg_kill_switch import (
        MultiLegKillSwitchConfig,
        MultiLegKillSwitchTracker,
    )
    from src.time_series_model.core.constitution.account_risk_guard import (
        AccountRiskSnapshot,
    )

    tracker = MultiLegKillSwitchTracker(
        config=MultiLegKillSwitchConfig(
            enabled=True,
            daily_loss_limit=0.06,
            max_dd=0.20,
            cooldown_minutes=0,
        ),
        state_path=tmp_path / "kill_switch_state.json",
    )
    now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(10_000.0, now=now)
    tracker.begin_batch()
    tracker.update_from_equity(9_300.0, now=now)
    assert tracker.is_halted()

    engine = FollowUpEngine(
        pending_rounds=[
            [
                {
                    "action": "place_protection",
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "quantity": 0.01,
                    "price": 49_000.0,
                }
            ]
        ]
    )
    adapter = _adapter()
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(
                max_gross_notional=1_000_000.0, max_net_notional=1_000_000.0
            ),
            account_snapshot_provider=lambda: AccountRiskSnapshot(
                equity=9_300.0, gross_notional=0.0
            ),
            kill_switch_tracker=tracker,
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(
            ReconciliationPolicy(client_id_prefixes={"dat_", "cg_"})
        ),
    )

    report = orchestrator.run_actions(
        [
            {
                "action": "market_exit",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.01,
            }
        ],
        reconcile=False,
    )

    executed = [
        str(a.get("action"))
        for call in adapter.execute_actions.call_args_list
        for a in call.args[0]
    ]
    assert "market_exit" in executed
    assert "place_protection" in executed
    assert not any("kill_switch" in r.reason for r in report.risk.rejected)
