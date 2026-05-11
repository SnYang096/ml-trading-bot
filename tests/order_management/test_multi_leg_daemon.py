from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable
from unittest.mock import MagicMock, patch

from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_daemon import (
    MultiLegBarEvent,
    MultiLegLiveDaemon,
    StrategyRuntime,
)
from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator
from src.order_management.multi_leg_reconciliation import (
    LocalPositionSnapshot,
    MultiLegReconciler,
)
from src.order_management.multi_leg_risk_governor import (
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)


@dataclass
class FakeProvider:
    bars: list[MultiLegBarEvent]

    def latest_closed_bars(self, symbols: Iterable[str]) -> list[MultiLegBarEvent]:
        allowed = set(symbols)
        return [bar for bar in self.bars if bar.symbol in allowed]


@dataclass
class FakeEngine:
    action_price: float = 50_000.0
    calls: int = 0
    timestamps: list[str] = field(default_factory=list)
    results: list[list[GridExecutionResult]] = field(default_factory=list)
    reports: list[object] = field(default_factory=list)

    def on_bar(self, **kwargs):
        self.calls += 1
        self.timestamps.append(str(kwargs["timestamp"]))
        return [
            {
                "action": "place",
                "symbol": kwargs["symbol"],
                "side": "BUY",
                "quantity": 0.01,
                "price": self.action_price,
                "client_order_id": f"dat_{self.calls}",
            }
        ]

    def local_order_snapshots(self):
        return []

    def local_position_snapshots(self):
        return []

    def on_execution_results(self, results):
        self.results.append(list(results))

    def on_reconciliation_report(self, report):
        self.reports.append(report)


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.sync_open_orders.return_value = []
    adapter.sync_positions.return_value = []
    adapter.execute_actions.side_effect = lambda actions: [
        GridExecutionResult(
            action=a["action"], status="shadow", symbol=a["symbol"], raw=a
        )
        for a in actions
    ]
    return adapter


def _runtime(
    name: str, symbol: str, engine: FakeEngine, adapter: MagicMock
) -> StrategyRuntime:
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(),
    )
    return StrategyRuntime(
        name=name, symbol=symbol, engine=engine, orchestrator=orchestrator
    )


def test_orchestrator_persists_execution_report_with_raw_fill_fields() -> None:
    engine = FakeEngine()
    adapter = _adapter()
    storage = MagicMock()
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(),
        storage=storage,
        run_id="run_1",
        strategy_name="dual_add_trend",
        symbol="BTCUSDT",
    )

    orchestrator.on_execution_report(
        {
            "symbol": "BTCUSDT",
            "order_id": "ex_1",
            "client_order_id": "dat_abc",
            "status": "FILLED",
            "execution_type": "TRADE",
            "commission": 0.02,
            "commission_asset": "USDT",
            "fill_slippage_bps": 2.0,
        }
    )

    storage.record_execution_report.assert_called_once()
    payload = storage.record_execution_report.call_args.args[0]
    assert payload["run_id"] == "run_1"
    assert payload["strategy"] == "dual_add_trend"
    assert payload["raw"]["commission"] == 0.02
    assert payload["raw"]["fill_slippage_bps"] == 2.0


def test_daemon_processes_each_runtime_once_per_new_bar() -> None:
    bar = MultiLegBarEvent(
        symbol="BTCUSDT",
        timestamp="2026-01-01 00:00:00+00:00",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={},
    )
    engine = FakeEngine()
    adapter = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=FakeProvider([bar]),
        runtimes=[_runtime("dual_add_trend", "BTCUSDT", engine, adapter)],
    )

    first = daemon.run_once()
    second = daemon.run_once()

    assert first.bars_seen == 1
    assert first.action_count == 1
    assert first.execution_count == 1
    assert second.bars_seen == 0
    assert engine.calls == 1


def test_daemon_processes_multiple_new_bars_for_same_symbol_in_order() -> None:
    bars = [
        MultiLegBarEvent(
            symbol="BTCUSDT",
            timestamp="2026-01-01 00:01:00+00:00",
            high=101.0,
            low=99.0,
            close=100.0,
            atr=2.0,
            features={},
        ),
        MultiLegBarEvent(
            symbol="BTCUSDT",
            timestamp="2026-01-01 00:02:00+00:00",
            high=102.0,
            low=100.0,
            close=101.0,
            atr=2.0,
            features={},
        ),
    ]
    engine = FakeEngine()
    adapter = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=FakeProvider(bars),
        runtimes=[_runtime("dual_add_trend", "BTCUSDT", engine, adapter)],
    )

    first = daemon.run_once()
    second = daemon.run_once()

    assert first.bars_seen == 2
    assert first.action_count == 2
    assert first.execution_count == 2
    assert second.bars_seen == 0
    assert engine.timestamps == [
        "2026-01-01 00:01:00+00:00",
        "2026-01-01 00:02:00+00:00",
    ]


def test_daemon_reports_rejections_from_governor() -> None:
    bar = MultiLegBarEvent(
        symbol="BTCUSDT",
        timestamp="2026-01-01 00:00:00+00:00",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={},
    )
    engine = FakeEngine(action_price=200_000.0)
    adapter = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=FakeProvider([bar]),
        runtimes=[_runtime("dual_add_trend", "BTCUSDT", engine, adapter)],
    )

    report = daemon.run_once()

    assert report.rejected_count == 1
    assert report.execution_count == 0
    adapter.execute_actions.assert_not_called()


def test_daemon_can_route_same_bar_to_two_strategy_runtimes() -> None:
    bar = MultiLegBarEvent(
        symbol="BTCUSDT",
        timestamp="2026-01-01 00:00:00+00:00",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={},
    )
    engine_a = FakeEngine()
    engine_b = FakeEngine()
    adapter_a = _adapter()
    adapter_b = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=FakeProvider([bar]),
        runtimes=[
            _runtime("chop_grid", "BTCUSDT", engine_a, adapter_a),
            _runtime("dual_add_trend", "BTCUSDT", engine_b, adapter_b),
        ],
    )

    report = daemon.run_once()

    assert report.bars_seen == 2
    assert report.action_count == 1
    assert report.rejected_count == 1
    assert engine_a.calls == 1
    assert engine_b.calls == 1


def test_daemon_blocks_opening_actions_when_other_strategy_already_owns_symbol() -> (
    None
):
    bar = MultiLegBarEvent(
        symbol="BTCUSDT",
        timestamp="2026-01-01 00:00:00+00:00",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={},
    )
    engine_a = FakeEngine()
    engine_b = FakeEngine()
    engine_a.local_position_snapshots = lambda: [
        LocalPositionSnapshot(symbol="BTCUSDT", side="LONG", quantity=0.01)
    ]
    adapter_a = _adapter()
    adapter_b = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=FakeProvider([bar]),
        runtimes=[
            _runtime("chop_grid", "BTCUSDT", engine_a, adapter_a),
            _runtime("dual_add_trend", "BTCUSDT", engine_b, adapter_b),
        ],
    )

    report = daemon.run_once()

    assert report.rejected_count >= 1
    adapter_b.execute_actions.assert_not_called()


def test_run_forever_increments_poll_metric_once_per_iteration() -> None:
    bar = MultiLegBarEvent(
        symbol="BTCUSDT",
        timestamp="2026-01-01 00:00:00+00:00",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={},
    )
    engine = FakeEngine()
    adapter = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=FakeProvider([bar]),
        runtimes=[_runtime("dual_add_trend", "BTCUSDT", engine, adapter)],
        poll_seconds=0.01,
    )
    poll_inc = MagicMock()
    with patch(
        "src.order_management.multi_leg_daemon.METRICS.multi_leg_daemon_polls_total"
    ) as poll_ctr:
        poll_ctr.inc = poll_inc
        asyncio.run(daemon.run_forever(max_iterations=3))

    assert poll_inc.call_count == 3
    assert all(c.args[0] == 1 for c in poll_inc.call_args_list)
