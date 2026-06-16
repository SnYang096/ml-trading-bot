from __future__ import annotations

from pathlib import Path

from src.order_management.grid_execution_adapter import (
    GridExecutionResult,
    derive_multileg_client_order_id,
)
from src.order_management.multi_leg_reconciliation import (
    PositionMismatch,
    ReconciliationReport,
)
from src.time_series_model.live.dual_add_trend_live_engine import (
    DualAddPosition,
    DualAddTrendLiveEngine,
    _normalize_entry_leg_id,
)
from src.time_series_model.live.segment_lifecycle import SegmentState


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
    assert engine.state.segment_state == SegmentState.CLOSING.value
    assert engine.state.active is True
    assert engine.holds_real_grid_slot() is True


def test_exit_all_cancels_position_protection_orders(tmp_path: Path) -> None:
    """Regime exit must cancel the position's live SL/TP so they don't outlive it."""
    engine = _make_engine_with_position(
        tmp_path,
        protection_order_ids=["ex_sl_111", "ex_tp_222"],
    )

    actions = engine._exit_all(100.0, "2026-01-01T01:00:00Z", reason="regime_exit")

    market_exits = [a for a in actions if a.get("action") == "market_exit"]
    cancel_prot = [a for a in actions if a.get("action") == "cancel_protection"]
    assert len(market_exits) == 1
    assert {a["exchange_order_id"] for a in cancel_prot} == {"ex_sl_111", "ex_tp_222"}
    for a in cancel_prot:
        assert a["symbol"] == "BTCUSDT"
        assert a["leg_id"] == "long_0"
    assert engine.state.inventory == []


def test_target_exit_cancels_position_protection_orders(tmp_path: Path) -> None:
    """Internal TP hit must cancel the leg's exchange protection in the same bar."""
    engine = _make_engine_with_position(
        tmp_path,
        protection_order_ids=["ex_sl_333"],
        take_profit_mode="per_leg",
    )
    # entry 100, atr 2, tp_atr_mult 0.25 → tp distance 0.5 → LONG TP at 100.5
    actions = engine._target_exits(
        high=101.0, low=100.0, close=100.6, timestamp="2026-01-01T02:00:00Z"
    )

    market_exits = [a for a in actions if a.get("action") == "market_exit"]
    cancel_prot = [a for a in actions if a.get("action") == "cancel_protection"]
    assert len(market_exits) == 1
    assert [a["exchange_order_id"] for a in cancel_prot] == ["ex_sl_333"]
    assert engine.state.inventory == []


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


def test_on_execution_results_market_exit_clears_late_fill_inventory(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    pos = DualAddPosition(
        leg_id="local_active_fill0",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        quantity=1.0,
        seq=0,
        entry_time="2026-01-01T02:00:00Z",
    )
    engine.state.inventory = [pos]
    engine.state.active = False
    engine.state.symbol = "BTCUSDT"

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="market_exit",
                status="filled",
                symbol="BTCUSDT",
                order_id="ex_cleanup",
                client_order_id="cl_cleanup",
                raw={
                    "leg_id": pos.leg_id,
                    "order_id": f"{pos.leg_id}_exit_late_fill_cleanup_2026-01-01T02:00:01Z",
                    "reason": "late_fill_cleanup",
                },
            )
        ]
    )

    assert engine.state.inventory == []
    assert engine.local_position_snapshots() == []


def test_on_execution_results_market_exit_persists_when_segment_inactive(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        unit_notional=100.0,
    )
    pos = DualAddPosition(
        leg_id="local_active_fill0",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        quantity=1.0,
        seq=0,
        entry_time="2026-01-01T02:00:00Z",
    )
    engine.state.inventory = [pos]
    engine.state.active = False
    engine.state.symbol = "BTCUSDT"
    engine.save_state()

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="market_exit",
                status="filled",
                symbol="BTCUSDT",
                order_id="ex_cleanup",
                client_order_id="cl_cleanup",
                raw={
                    "leg_id": pos.leg_id,
                    "order_id": f"{pos.leg_id}_exit_late_fill_cleanup_2026-01-01T02:00:01Z",
                    "reason": "late_fill_cleanup",
                },
            )
        ]
    )

    reloaded = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        unit_notional=100.0,
    )
    assert reloaded.state.inventory == []


# =========================================================================
# actions_ensure_protection tests
# =========================================================================


def _make_engine_with_position(
    tmp_path: Path,
    *,
    leg_id: str = "long_0",
    side: str = "LONG",
    protection_order_ids: list[str] | None = None,
    active: bool = True,
    segment_state: str = SegmentState.ENTERING.value,
    protection_stop_mode: str = "catastrophic",
    take_profit_mode: str = "basket",
) -> DualAddTrendLiveEngine:
    """Helper: create an engine with one inventory position."""
    cfg_text = f"""
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
  mode: {take_profit_mode}
risk:
  diagnostic_fee_bps: 4.0
  max_loss_per_segment: 0.01
  protection_stop_mode: {protection_stop_mode}
order_model:
  entry_order_type: marketable_limit
  add_order_type: marketable_limit
  max_slippage_bps: 5.0
  pending_timeout_bars: 1
"""
    cfg_path = tmp_path / "dual_add.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    engine = DualAddTrendLiveEngine(
        config_path=cfg_path,
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    engine.state.active = active
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.center = 100.0
    engine.state.atr = 2.0
    engine.state.trend_side = side
    engine.state.segment_state = segment_state
    engine.state.last_timestamp = "2026-01-01T00:00:00Z"
    engine.state.inventory = [
        DualAddPosition(
            leg_id=leg_id,
            symbol="BTCUSDT",
            side=side,
            entry_price=100.0,
            quantity=1.0,
            seq=0,
            entry_time="2026-01-01T00:00:00Z",
            protection_order_ids=list(protection_order_ids or []),
        ),
    ]
    return engine


def test_ensure_protection_returns_sl_when_no_protection_exists(
    tmp_path: Path,
) -> None:
    """Freshly filled position with no SL placed → reconcile re-places it."""
    engine = _make_engine_with_position(tmp_path, protection_order_ids=[])

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert len(actions) == 1
    assert actions[0]["action"] == "place_protection"
    assert actions[0]["protection_type"] == "stop_loss"
    assert actions[0]["leg_id"] == "long_0"
    assert actions[0]["order_id"] == "long_0_sl"


def test_ensure_protection_returns_empty_when_sl_live_on_exchange(
    tmp_path: Path,
) -> None:
    """SL order still live on exchange → no re-place needed."""
    engine = _make_engine_with_position(tmp_path, protection_order_ids=["ex_sl_123"])

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[
            {"orderId": "ex_sl_123", "clientOrderId": "dat_abc", "status": "NEW"},
        ],
    )

    assert actions == []


def test_ensure_protection_returns_sl_when_sl_filled_on_exchange(
    tmp_path: Path,
) -> None:
    """SL was filled/canceled on exchange → not in open orders → re-place."""
    engine = _make_engine_with_position(tmp_path, protection_order_ids=["ex_sl_old"])

    # Exchange has no open orders (SL was filled and removed)
    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert len(actions) == 1
    assert actions[0]["order_id"] == "long_0_sl"


def test_ensure_protection_skips_when_not_active(tmp_path: Path) -> None:
    engine = _make_engine_with_position(tmp_path, active=False)

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert actions == []


def test_ensure_protection_skips_when_winding_down(tmp_path: Path) -> None:
    engine = _make_engine_with_position(
        tmp_path, segment_state=SegmentState.CLOSING.value
    )

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert actions == []


def test_ensure_protection_skips_when_stop_mode_none(tmp_path: Path) -> None:
    engine = _make_engine_with_position(tmp_path, protection_stop_mode="none")

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert actions == []


def test_ensure_protection_basket_mode_no_tp(tmp_path: Path) -> None:
    """basket TP mode → only SL placed, no TP on exchange."""
    engine = _make_engine_with_position(tmp_path, take_profit_mode="basket")

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert len(actions) == 1
    assert actions[0]["protection_type"] == "stop_loss"


def test_ensure_protection_non_basket_mode_sl_and_tp(tmp_path: Path) -> None:
    """Non-basket TP mode → both SL and TP placed on exchange."""
    engine = _make_engine_with_position(tmp_path, take_profit_mode="per_leg")

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    types = {a["protection_type"] for a in actions}
    assert types == {"stop_loss", "take_profit"}


def test_ensure_protection_dedup_by_client_order_id(tmp_path: Path) -> None:
    """If the deterministic client_order_id already exists on exchange, skip."""
    engine = _make_engine_with_position(tmp_path, protection_order_ids=[])

    # The SL action's order_id is "long_0_sl" — derive_multileg_client_order_id
    # hashes it the same way MultiLegExecutionAdapter does.  Simulate the
    # hashed clientOrderId existing on exchange.
    actions_no_dedup = engine.actions_ensure_protection(
        exchange_positions=[], exchange_orders=[]
    )
    assert len(actions_no_dedup) == 1

    # Compute the hashed client_order_id, put it on exchange, re-run
    expected_cid = derive_multileg_client_order_id(actions_no_dedup[0])
    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[
            {"clientOrderId": expected_cid, "status": "NEW"},
        ],
    )
    assert actions == []


def test_ensure_protection_multiple_positions(tmp_path: Path) -> None:
    """Two positions, one has SL, other doesn't → only missing SL returned."""
    engine = _make_engine_with_position(
        tmp_path,
        leg_id="long_0",
        protection_order_ids=["ex_sl_existing"],
    )
    engine.state.inventory.append(
        DualAddPosition(
            leg_id="short_0",
            symbol="BTCUSDT",
            side="SHORT",
            entry_price=101.0,
            quantity=1.0,
            seq=1,
            entry_time="2026-01-01T01:00:00Z",
            protection_order_ids=[],
        )
    )

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[
            {"orderId": "ex_sl_existing", "status": "NEW"},
        ],
    )

    # Only the SHORT position missing SL
    assert len(actions) == 1
    assert actions[0]["leg_id"] == "short_0"
    assert actions[0]["order_id"] == "short_0_sl"


def test_on_execution_results_market_exit_parses_leg_id_from_order_id(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    pos = DualAddPosition(
        leg_id="local_active_fill0",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        quantity=1.0,
        seq=0,
        entry_time="2026-01-01T02:00:00Z",
    )
    engine.state.inventory = [pos]
    engine.state.active = False
    engine.state.symbol = "BTCUSDT"

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="market_exit",
                status="skipped_no_position",
                symbol="BTCUSDT",
                order_id="ex_cleanup",
                client_order_id="cl_cleanup",
                raw={
                    "order_id": f"{pos.leg_id}_exit_late_fill_cleanup_2026-01-01T02:00:01Z",
                    "reason": "late_fill_cleanup",
                },
            )
        ]
    )

    assert engine.state.inventory == []


def test_on_execution_results_market_exit_rejected_keeps_inventory(
    tmp_path: Path,
) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    pos = DualAddPosition(
        leg_id="local_active_fill0",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        quantity=1.0,
        seq=0,
        entry_time="2026-01-01T02:00:00Z",
    )
    engine.state.inventory = [pos]
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="market_exit",
                status="rejected",
                symbol="BTCUSDT",
                order_id="ex_cleanup",
                client_order_id="cl_cleanup",
                raw={
                    "leg_id": pos.leg_id,
                    "reason": "late_fill_cleanup",
                },
            )
        ]
    )

    assert engine.state.inventory == [pos]


def test_on_execution_results_market_exit_shadow_does_not_clear_inventory(
    tmp_path: Path,
) -> None:
    """shadow=True 时 GridExecutionAdapter 不发实单，不应清本地 inventory。"""
    engine = DualAddTrendLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        unit_notional=100.0,
    )
    pos = DualAddPosition(
        leg_id="local_active_fill0",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        quantity=1.0,
        seq=0,
        entry_time="2026-01-01T02:00:00Z",
    )
    engine.state.inventory = [pos]
    engine.state.active = False
    engine.state.symbol = "BTCUSDT"

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="market_exit",
                status="shadow",
                symbol="BTCUSDT",
                order_id="ex_shadow",
                client_order_id="cl_shadow",
                raw={
                    "leg_id": pos.leg_id,
                    "reason": "late_fill_cleanup",
                },
            )
        ]
    )

    # shadow = paper mode, no real fill → inventory must remain
    assert engine.state.inventory == [pos]


# =========================================================================
# _handle_protection_fill tests
# =========================================================================


def _engine_with_position_and_protection(
    tmp_path: Path,
    *,
    leg_id: str = "long_0",
    side: str = "LONG",
    quantity: float = 1.0,
    entry_price: float = 100.0,
    protection_order_ids: list[str] | None = None,
    atr: float = 2.0,
) -> DualAddTrendLiveEngine:
    """Helper: engine with ACTIVE segment + inventory + protection IDs."""
    engine = _make_engine_with_position(
        tmp_path,
        leg_id=leg_id,
        side=side,
        protection_order_ids=list(protection_order_ids or []),
        segment_state=SegmentState.ACTIVE.value,
        take_profit_mode="per_leg",
    )
    engine.state.inventory[0].quantity = quantity
    engine.state.inventory[0].entry_price = entry_price
    engine.state.atr = atr
    return engine


def test_protection_fill_removes_position_on_full_fill(tmp_path: Path) -> None:
    """Full SL fill removes the position from inventory."""
    engine = _engine_with_position_and_protection(
        tmp_path, protection_order_ids=["ex_sl_abc"]
    )
    assert len(engine.state.inventory) == 1

    handled = engine._handle_protection_fill(
        {
            "order_id": "ex_sl_abc",
            "client_order_id": "",
            "status": "FILLED",
            "filled_qty": 1.0,
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is True
    assert engine.state.inventory == []


def test_protection_fill_partial_shrinks_position(tmp_path: Path) -> None:
    """Partial SL fill (filled_qty < position.quantity) shrinks the position."""
    engine = _engine_with_position_and_protection(
        tmp_path, protection_order_ids=["ex_sl_partial"], quantity=1.0
    )
    pos = engine.state.inventory[0]
    assert float(pos.quantity) == 1.0

    handled = engine._handle_protection_fill(
        {
            "order_id": "ex_sl_partial",
            "status": "FILLED",
            "filled_qty": 0.3,
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is True
    assert len(engine.state.inventory) == 1
    assert float(engine.state.inventory[0].quantity) == 0.7
    # Filled protection ID removed from list
    assert "ex_sl_partial" not in engine.state.inventory[0].protection_order_ids


def test_protection_fill_partial_clears_all_protection_ids_for_re_place(
    tmp_path: Path,
) -> None:
    """Partial SL fill: ALL protection IDs cleared — TP must be re-placed at new size."""
    engine = _engine_with_position_and_protection(
        tmp_path,
        protection_order_ids=["ex_sl_aaa", "ex_tp_bbb"],
        quantity=1.0,
    )

    engine._handle_protection_fill(
        {
            "order_id": "ex_sl_aaa",
            "status": "FILLED",
            "filled_qty": 0.4,
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    pos = engine.state.inventory[0]
    # Partial fill shrinks position and clears ALL protection IDs.
    # TP at original size is now unsafe → must be re-placed via ensure_protection.
    assert pos.protection_order_ids == []
    assert float(pos.quantity) == 0.6


def test_protection_fill_leg_hint_match_strips_sl_suffix(tmp_path: Path) -> None:
    """Fill with leg_id="long_0_sl" → strips _sl → matches leg_id="long_0"."""
    engine = _engine_with_position_and_protection(
        tmp_path, leg_id="long_0", protection_order_ids=[]
    )

    # protection_order_ids empty, but leg_id hint matches after stripping _sl
    handled = engine._handle_protection_fill(
        {
            "order_id": "",
            "client_order_id": "",
            "leg_id": "long_0_sl",
            "status": "FILLED",
            "filled_qty": 1.0,
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is True
    assert engine.state.inventory == []


def test_protection_fill_leg_hint_match_strips_tp_suffix(tmp_path: Path) -> None:
    """Fill with leg_id="long_0_tp" → strips _tp → matches leg_id="long_0"."""
    engine = _engine_with_position_and_protection(
        tmp_path, leg_id="short_1", side="SHORT"
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "",
            "client_order_id": "",
            "leg_id": "short_1_tp",
            "status": "FILLED",
            "filled_qty": 1.0,
            "last_filled_price": 102.0,
            "protection_type": "take_profit",
        }
    )

    assert handled is True
    assert engine.state.inventory == []


def test_protection_fill_ignores_non_filled_status(tmp_path: Path) -> None:
    """NEW/CANCELED status → not handled as protection fill."""
    engine = _engine_with_position_and_protection(
        tmp_path, protection_order_ids=["ex_sl_xxx"]
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "ex_sl_xxx",
            "status": "NEW",
            "filled_qty": 1.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is False
    assert len(engine.state.inventory) == 1


def test_protection_fill_full_close_clears_entire_position(tmp_path: Path) -> None:
    """filled_qty >= pos.quantity → full close, position removed."""
    engine = _engine_with_position_and_protection(
        tmp_path,
        protection_order_ids=["ex_sl_full"],
        quantity=0.5,
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "ex_sl_full",
            "status": "FILLED",
            "filled_qty": 0.5,
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is True
    assert engine.state.inventory == []


def test_protection_fill_no_match_no_leg_hint_returns_false(tmp_path: Path) -> None:
    """order_id/cliend_id don't match, leg_id doesn't match → False."""
    engine = _engine_with_position_and_protection(
        tmp_path,
        leg_id="long_5",
        protection_order_ids=["ex_sl_zzz"],
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "ex_other",
            "client_order_id": "cl_other",
            "leg_id": "unrelated_leg",
            "status": "FILLED",
            "filled_qty": 1.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is False
    assert len(engine.state.inventory) == 1


# =========================================================================
# actions_ensure_protection → on_execution_results closed loop
# =========================================================================


def test_ensure_protection_on_execution_results_closed_loop(tmp_path: Path) -> None:
    """Full round-trip: ensure → execute → on_execution_results → ensure again→[]"""
    engine = _make_engine_with_position(
        tmp_path,
        leg_id="loop_leg",
        protection_order_ids=[],
        segment_state=SegmentState.ACTIVE.value,
    )

    # Phase 1: ensure_protection returns SL action
    actions = engine.actions_ensure_protection(
        exchange_positions=[], exchange_orders=[]
    )
    assert len(actions) == 1
    assert actions[0]["leg_id"] == "loop_leg"
    assert actions[0]["protection_type"] == "stop_loss"

    # Phase 2: adapter executed the SL, now feed back via on_execution_results
    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place_protection",
                status="open",
                symbol="BTCUSDT",
                order_id="ex_loop_sl",
                client_order_id=derive_multileg_client_order_id(actions[0]),
                raw={**actions[0], "leg_id": "loop_leg"},
            )
        ]
    )

    # protection_order_ids now has the exchange ID
    pos = engine._find_position("loop_leg")
    assert pos is not None
    assert "ex_loop_sl" in pos.protection_order_ids

    # Phase 3: ensure_protection again → should skip (SL is live on exchange)
    actions2 = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[
            {"orderId": "ex_loop_sl", "status": "NEW"},
        ],
    )
    assert actions2 == []


def test_ensure_protection_closed_loop_multiple_legs(tmp_path: Path) -> None:
    """Two legs, one gets SL placed via ensure→execute→results cycle, other still missing."""
    engine = _make_engine_with_position(
        tmp_path,
        leg_id="leg_a",
        protection_order_ids=[],
        segment_state=SegmentState.ACTIVE.value,
    )
    engine.state.inventory.append(
        DualAddPosition(
            leg_id="leg_b",
            symbol="BTCUSDT",
            side="SHORT",
            entry_price=101.0,
            quantity=1.0,
            seq=1,
            entry_time="2026-01-01T01:00:00Z",
            protection_order_ids=[],
        )
    )

    # Phase 1: ensure → returns SL for both (neither has protection)
    actions = engine.actions_ensure_protection(
        exchange_positions=[], exchange_orders=[]
    )
    action_leg_ids = {a["leg_id"] for a in actions}
    assert action_leg_ids == {"leg_a", "leg_b"}

    # Phase 2: only leg_a's SL got executed successfully
    leg_a_action = next(a for a in actions if a["leg_id"] == "leg_a")
    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place_protection",
                status="open",
                symbol="BTCUSDT",
                order_id="ex_leg_a_sl",
                client_order_id=derive_multileg_client_order_id(leg_a_action),
                raw={**leg_a_action, "leg_id": "leg_a"},
            )
        ]
    )

    # Phase 3: ensure again — only leg_b missing
    actions2 = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[
            {"orderId": "ex_leg_a_sl", "status": "NEW"},
        ],
    )
    assert len(actions2) == 1
    assert actions2[0]["leg_id"] == "leg_b"


def test_on_execution_results_place_protection_rejected_no_order_id(
    tmp_path: Path,
) -> None:
    """Rejected protection placement → no order_id → not added to protection_order_ids."""
    engine = _make_engine_with_position(
        tmp_path,
        leg_id="rejected_leg",
        protection_order_ids=[],
        segment_state=SegmentState.ACTIVE.value,
    )

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place_protection",
                status="rejected",
                symbol="BTCUSDT",
                order_id="",  # no order_id on rejection
                client_order_id="cl_rejected",
                raw={"leg_id": "rejected_leg", "order_id": "rejected_leg_sl"},
            )
        ]
    )

    pos = engine._find_position("rejected_leg")
    assert pos is not None
    assert pos.protection_order_ids == []  # still empty, no ID to add


def test__normalize_entry_leg_id_strips_suffixes() -> None:
    """Unit-test the helper directly."""
    assert _normalize_entry_leg_id("long_0_sl") == "long_0"
    assert _normalize_entry_leg_id("short_1_tp") == "short_1"
    assert _normalize_entry_leg_id("bare_leg") == "bare_leg"
    assert _normalize_entry_leg_id("") == ""
    assert _normalize_entry_leg_id("odd_sl_tp") == "odd_sl"


# =========================================================================
# Kill switch does NOT block place_protection
# =========================================================================


def test_kill_switch_does_not_block_place_protection() -> None:
    """place_protection must not be in _RISK_INCREASING_ACTIONS so that
    existing positions retain exchange-side SL/TP during halt."""
    from src.order_management.multi_leg_kill_switch import (
        MultiLegKillSwitchConfig,
        MultiLegKillSwitchTracker,
        _RISK_INCREASING_ACTIONS,
    )

    assert "place_protection" not in _RISK_INCREASING_ACTIONS
    assert "place" in _RISK_INCREASING_ACTIONS


def test_kill_switch_allows_protection_during_halt(tmp_path: Path) -> None:
    """Even when the kill switch is halted, place_protection is not blocked."""
    from src.order_management.multi_leg_kill_switch import (
        MultiLegKillSwitchConfig,
        MultiLegKillSwitchTracker,
    )

    tracker = MultiLegKillSwitchTracker(
        config=MultiLegKillSwitchConfig(enabled=True, max_dd=0.01),
        state_path=tmp_path / "ks_state.json",
    )
    # Simulate halt by setting peak high and equity low
    tracker.peak_equity = 1000.0
    tracker.last_equity = 1000.0
    tracker.day_start_equity = 1000.0
    tracker.update_from_equity(950.0)  # 5% drawdown > 1% limit → halted
    assert tracker.is_halted()

    # place must be blocked
    block_reason = tracker.blocks_action("place")
    assert block_reason is not None

    # place_protection must NOT be blocked
    prot_reason = tracker.blocks_action("place_protection")
    assert prot_reason is None


# =========================================================================
# leg_hint fallback handles partial fill
# =========================================================================


def test_protection_fill_leg_hint_partial_shrinks_position(tmp_path: Path) -> None:
    """leg_hint fallback with filled_qty < pos.quantity shrinks the position."""
    engine = _engine_with_position_and_protection(
        tmp_path,
        leg_id="hint_part",
        protection_order_ids=[],  # empty → forces leg_hint fallback
        quantity=1.0,
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "",
            "client_order_id": "",
            "leg_id": "hint_part_sl",
            "status": "FILLED",
            "filled_qty": 0.4,
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is True
    assert len(engine.state.inventory) == 1
    assert float(engine.state.inventory[0].quantity) == 0.6
    assert engine.state.inventory[0].protection_order_ids == []


def test_protection_fill_leg_hint_full_fill_removes_position(tmp_path: Path) -> None:
    """leg_hint fallback with filled_qty >= pos.quantity removes the position."""
    engine = _engine_with_position_and_protection(
        tmp_path,
        leg_id="hint_full",
        protection_order_ids=[],
        quantity=0.5,
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "",
            "client_order_id": "",
            "leg_id": "hint_full_tp",
            "status": "FILLED",
            "filled_qty": 0.5,
            "last_filled_price": 102.0,
            "protection_type": "take_profit",
        }
    )

    assert handled is True
    assert engine.state.inventory == []


def test_protection_fill_leg_hint_zero_filled_qty_removes_position(
    tmp_path: Path,
) -> None:
    """leg_hint fallback with filled_qty=0 (default to full close) removes position."""
    engine = _engine_with_position_and_protection(
        tmp_path,
        leg_id="hint_zero",
        protection_order_ids=[],
        quantity=0.3,
    )

    handled = engine._handle_protection_fill(
        {
            "order_id": "",
            "client_order_id": "",
            "leg_id": "hint_zero_sl",
            "status": "FILLED",
            "filled_qty": 0,  # not provided → default to full close
            "last_filled_price": 98.0,
            "protection_type": "stop_loss",
        }
    )

    assert handled is True
    assert engine.state.inventory == []
