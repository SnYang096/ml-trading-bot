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

    engine.on_execution_report(
        {
            "order_id": "ex_1",
            "client_order_id": "dat_abc",
            "status": "FILLED",
            "filled_qty": 1.0,
            "last_filled_price": 100.0,
            "trade_time": "2026-01-01T00:02:00Z",
        }
    )

    assert len(engine.local_order_snapshots()) == 1
    positions = engine.local_position_snapshots()
    assert len(positions) == 1
    assert positions[0].side == "LONG"
    follow_ups = engine.pop_pending_actions()
    assert [a["protection_type"] for a in follow_ups] == ["take_profit", "stop_loss"]
    assert all(a["action"] == "place_protection" for a in follow_ups)


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
