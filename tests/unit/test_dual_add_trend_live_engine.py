from __future__ import annotations

from pathlib import Path

from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_reconciliation import (
    PositionMismatch,
    ReconciliationReport,
)
from src.time_series_model.live.dual_add_trend_live_engine import (
    DualAddPosition,
    DualAddTrendLiveEngine,
)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "dual_add.yaml"
    path.write_text(
        """
regime:
  entry_min: 0.80
  exit_below: 0.50
  max_semantic_chop_entry: 0.25
  max_semantic_chop_hold: 0.40
  exclude_box_prefilter: true
inventory:
  flip_action: close_offside_all
  max_adds_per_side: 3
  max_gross_exposure_units: 4
  max_net_exposure_units: 2
add_spacing:
  atr_mult: 0.50
take_profit:
  atr_mult: 0.25
  min_pct: 0.0005
  min_abs: 0.0
risk:
  diagnostic_fee_bps: 4.0
  max_loss_per_segment: 0.01
order_model:
  entry_order_type: marketable_limit
  add_order_type: marketable_limit
  max_slippage_bps: 5.0
  pending_timeout_bars: 1
""",
        encoding="utf-8",
    )
    return path


def test_dual_add_enters_with_initial_long_and_short_orders(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )

    assert [a["side"] for a in actions] == ["BUY", "SELL"]
    assert [a["order_type"] for a in actions] == [
        "marketable_limit",
        "marketable_limit",
    ]
    assert actions[0]["time_in_force"] == "IOC"
    assert actions[0]["reference_price"] == 100.0
    assert actions[0]["price"] == 100.05
    assert actions[1]["price"] == 99.95
    assert len(engine.local_order_snapshots()) == 2
    assert engine.state.active is True


def test_dual_add_maps_execution_result_and_fill_to_position(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )
    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place",
                status="open",
                symbol="BTCUSDT",
                order_id="ex_1",
                client_order_id="dat_abc",
                raw=actions[0],
            )
        ]
    )

    report = {
        "order_id": "ex_1",
        "client_order_id": "dat_abc",
        "status": "FILLED",
        "filled_qty": 1.0,
        "last_filled_price": 100.02,
        "commission": 0.004,
        "commission_asset": "USDT",
        "trade_time": "2026-01-01T00:02:00Z",
    }
    engine.on_execution_report(report)

    assert len(engine.local_order_snapshots()) == 1
    positions = engine.local_position_snapshots()
    assert len(positions) == 1
    assert positions[0].side == "LONG"
    assert report["reference_price"] == 100.0
    assert round(report["fill_slippage_bps"], 6) == 2.0
    assert report["max_slippage_bps"] == 5.0
    follow_ups = engine.pop_pending_actions()
    assert [a["protection_type"] for a in follow_ups] == ["stop_loss"]
    assert all(a["action"] == "place_protection" for a in follow_ups)


def test_dual_add_immediate_closed_place_records_inventory(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )
    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place",
                status="closed",
                symbol="BTCUSDT",
                order_id="ex_ioc_1",
                client_order_id=actions[0].get("client_order_id"),
                raw={
                    **actions[0],
                    "filled": actions[0]["quantity"],
                    "price": 100.02,
                },
            )
        ]
    )

    assert len(engine.local_order_snapshots()) == 1
    positions = engine.local_position_snapshots()
    assert len(positions) == 1
    assert positions[0].side == "LONG"
    assert actions[0]["side"] == "BUY"


def test_dual_add_cancels_stale_pending_orders(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T02:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )

    assert any(
        a["action"] == "cancel" and a["reason"] == "pending_timeout" for a in actions
    )


def test_dual_add_basket_tp_exits_inventory_together(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.center = 100.0
    engine.state.atr = 1.0
    engine.state.trend_side = "LONG"
    engine.state.inventory = [
        DualAddPosition("long_0", "BTCUSDT", "LONG", 100.0, 1.0, 0, "t0"),
        DualAddPosition("short_0", "BTCUSDT", "SHORT", 100.0, 1.0, 0, "t0"),
        DualAddPosition("long_1", "BTCUSDT", "LONG", 100.5, 1.0, 1, "t1"),
    ]

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T02:00:00Z",
        high=102.0,
        low=101.0,
        close=102.0,
        atr=1.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )

    exit_actions = [a for a in actions if a["action"] == "market_exit"]
    assert len(exit_actions) == 3
    assert {a["reason"] for a in exit_actions} == {"basket_tp"}
    assert engine.state.inventory == []
    assert engine.state.active is False


def test_dual_add_trend_flip_exits_offside_inventory(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.center = 100.0
    engine.state.atr = 2.0
    engine.state.trend_side = "LONG"
    engine.state.inventory = [
        DualAddPosition("long_1", "BTCUSDT", "LONG", 100.0, 1.0, 1, "t0"),
        DualAddPosition("short_1", "BTCUSDT", "SHORT", 100.0, 1.0, 1, "t0"),
    ]

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T02:00:00Z",
        high=100.2,
        low=99.8,
        close=99.8,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "DOWN",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )

    exit_actions = [a for a in actions if a["action"] == "market_exit"]
    assert exit_actions[0]["side"] == "LONG"
    assert [p.side for p in engine.state.inventory] == ["SHORT"]


def test_dual_add_keeps_local_only_missing_pending_orders(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )
    before_ids = [o.order_id for o in engine.state.pending_orders]

    engine.on_reconciliation_report(
        ReconciliationReport(
            missing_exchange_orders=[
                engine.local_order_snapshots()[0],
            ]
        )
    )

    after_ids = [o.order_id for o in engine.state.pending_orders]
    assert before_ids == after_ids


def test_dual_add_prunes_mapped_missing_pending_orders(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )
    target = engine.state.pending_orders[0]
    target.exchange_order_id = "ex_1"

    engine.on_reconciliation_report(
        ReconciliationReport(
            missing_exchange_orders=[
                engine.local_order_snapshots()[0],
            ]
        )
    )

    assert all(o.order_id != target.order_id for o in engine.state.pending_orders)


def test_dual_add_records_reconciliation_report(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )

    engine.on_reconciliation_report(
        ReconciliationReport(
            position_mismatches=[
                PositionMismatch("BTCUSDT", "SHORT", 0.0, 0.01),
            ]
        )
    )

    assert engine.state.last_reconciliation_ok is False
    assert engine.state.last_reconciliation_issues == [
        "position_mismatch:BTCUSDT:SHORT:0.0->0.01"
    ]


# =========================================================================
# _order_history / backfill late-fill guard tests
# =========================================================================
from src.time_series_model.live.dual_add_trend_live_engine import (  # noqa: E402
    DualAddOrder,
)


def test_archive_order_stores_by_all_keys(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    order = DualAddOrder(
        order_id="local_1",
        symbol="BTCUSDT",
        side="BUY",
        price=100.0,
        quantity=1.0,
        reason="entry",
        exchange_order_id="ex_123",
        client_order_id="cl_abc",
        reference_price=100.0,
        max_slippage_bps=5.0,
        seq=0,
    )
    engine._archive_order(order)

    hist = engine.state._order_history
    assert hist.get("ex_123") is order
    assert hist.get("local_1") is order
    assert hist.get("cl_abc") is order


def test_find_order_searches_history_after_pending_removal(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )
    order = engine.state.pending_orders[0]
    order.exchange_order_id = "ex_backfill"
    order.client_order_id = "cl_xyz"

    # Archive order (simulates order removed after fill + removed from pending)
    engine._archive_order(order)
    engine.state.pending_orders = []

    # Backfill fill arrives hours later → _find_order finds archived order
    found = engine._find_order(exchange_id="ex_backfill")
    assert found is not None
    assert found.exchange_order_id == "ex_backfill"

    found2 = engine._find_order(client_id="cl_xyz")
    assert found2 is not None


def test_late_fill_after_segment_exit_generates_cleanup_exit(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=101.0,
        low=99.0,
        close=100.0,
        atr=2.0,
        features={
            "trend_confidence": 1.0,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )
    order = engine.state.pending_orders[0]
    order.exchange_order_id = "ex_late"
    order.client_order_id = "cl_late"
    engine._archive_order(order)

    # Simulate: segment exited, pending_orders cleared by exit flow
    engine.state.inventory = []
    engine.state.pending_orders = []
    engine.state.active = False

    # Late fill via backfill
    report = {
        "order_id": "ex_late",
        "client_order_id": "cl_late",
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "filled_qty": 1.0,
        "last_filled_price": 100.02,
        "trade_time": "2026-01-01T02:00:00Z",
    }
    engine.on_execution_report(report)

    positions = engine.local_position_snapshots()
    assert len(positions) == 1
    assert positions[0].side == "LONG"

    exit_actions = [
        a for a in engine.pop_pending_actions() if a.get("action") == "market_exit"
    ]
    assert len(exit_actions) == 1
    assert exit_actions[0]["reason"] == "late_fill_cleanup"


def test_active_segment_fill_does_not_trigger_late_cleanup(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.center = 100.0
    engine.state.atr = 2.0
    engine.state.trend_side = "LONG"
    engine.state.last_timestamp = "2026-01-01T00:00:00Z"

    order = DualAddOrder(
        order_id="local_active",
        symbol="BTCUSDT",
        side="BUY",
        price=100.0,
        quantity=1.0,
        reason="entry",
        exchange_order_id="ex_active",
        client_order_id="cl_active",
        reference_price=100.0,
        max_slippage_bps=5.0,
        seq=0,
    )
    engine.state.pending_orders.append(order)

    report = {
        "order_id": "ex_active",
        "client_order_id": "cl_active",
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "filled_qty": 1.0,
        "last_filled_price": 100.02,
        "trade_time": "2026-01-01T00:02:00Z",
    }
    engine.on_execution_report(report)

    positions = engine.local_position_snapshots()
    assert len(positions) == 1

    cleanup = [
        a
        for a in engine.pop_pending_actions()
        if a.get("reason") == "late_fill_cleanup"
    ]
    assert len(cleanup) == 0


def test_exit_all_preserves_pending_orders(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.center = 100.0
    engine.state.atr = 2.0
    engine.state.trend_side = "LONG"
    engine.state.last_timestamp = "2026-01-01T00:00:00Z"

    order = DualAddOrder(
        order_id="local_exit",
        symbol="BTCUSDT",
        side="BUY",
        price=100.0,
        quantity=1.0,
        reason="entry",
        exchange_order_id="ex_exit",
        client_order_id="cl_exit",
        reference_price=100.0,
        max_slippage_bps=5.0,
        seq=0,
    )
    engine.state.pending_orders.append(order)

    actions = engine._exit_all(100.0, "2026-01-01T01:00:00Z", reason="regime_exit")

    # pending_orders NOT cleared by _exit_all
    assert len(engine.state.pending_orders) == 1
    assert engine.state.pending_orders[0].order_id == "local_exit"

    cancel_actions = [a for a in actions if a.get("action") == "cancel"]
    assert len(cancel_actions) == 1
    assert cancel_actions[0]["exchange_order_id"] == "ex_exit"

    assert engine.state.inventory == []
    assert engine.state.active is False


def test_on_execution_results_cleans_cancelled_and_archives(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    order = DualAddOrder(
        order_id="local_cancel",
        symbol="BTCUSDT",
        side="BUY",
        price=100.0,
        quantity=1.0,
        reason="entry",
        exchange_order_id="ex_cancel_test",
        client_order_id="cl_cancel_test",
        reference_price=100.0,
        max_slippage_bps=5.0,
        seq=0,
    )
    engine.state.pending_orders.append(order)
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="cancel",
                status="canceled",
                symbol="BTCUSDT",
                order_id="ex_cancel_test",
                client_order_id="cl_cancel_test",
                raw={"local_order_id": "local_cancel"},
            )
        ]
    )

    assert len(engine.state.pending_orders) == 0
    # Archived so late fills can still find the order
    assert engine.state._order_history.get("ex_cancel_test") is order


def test_save_state_excludes_order_history(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    order = DualAddOrder(
        order_id="local_save",
        symbol="BTCUSDT",
        side="BUY",
        price=100.0,
        quantity=1.0,
        reason="entry",
        exchange_order_id="ex_save",
        client_order_id="cl_save",
        reference_price=100.0,
        max_slippage_bps=5.0,
        seq=0,
    )
    engine._archive_order(order)
    engine.save_state()

    raw = __import__("json").loads((tmp_path / "state.json").read_text())
    assert "_order_history" not in raw
    assert raw.get("symbol") == ""
