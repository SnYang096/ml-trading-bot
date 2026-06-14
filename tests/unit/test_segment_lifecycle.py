"""Segment lifecycle tests for chop_grid and trend_scalp live engines."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.order_management.chop_grid_concurrency import MultiLegConcurrencyGate
from src.order_management.grid_execution_adapter import GridExecutionResult
from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridPosition,
)
from src.time_series_model.live.dual_add_trend_live_engine import (
    DualAddOrder,
    DualAddPosition,
    DualAddTrendLiveEngine,
)
from src.time_series_model.live.segment_lifecycle import SegmentState


def _chop_config(tmp_path: Path, *, max_replenish: int | None = None) -> Path:
    replenish = "null" if max_replenish is None else str(int(max_replenish))
    path = tmp_path / "chop.yaml"
    path.write_text(
        f"""
regime:
  entry_chop_min: 0.40
  exit_chop_below: 0.25
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 2
  max_replenish_per_level_per_segment: {replenish}
risk:
  fee_bps: 4.0
  max_loss_per_grid: 0.03
  max_open_levels_total: 4
""",
        encoding="utf-8",
    )
    return path


def _trend_config(tmp_path: Path) -> Path:
    path = tmp_path / "trend.yaml"
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
  mode: per_leg
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


def _enter_features() -> dict:
    return {
        "semantic_chop": 0.8,
        "box_prefilter": False,
        "trend_confidence": 1.0,
        "trend_direction": "UP",
    }


def test_trend_tp_protection_fill_deactivates_immediately(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=tmp_path / "trend_state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    pos = DualAddPosition(
        leg_id="long_0",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        quantity=1.0,
        seq=0,
        entry_time="2026-01-01T00:00:00Z",
        protection_order_ids=["tp_ex_1"],
    )
    engine.state.inventory = [pos]

    engine.on_execution_report(
        {
            "order_id": "tp_ex_1",
            "status": "FILLED",
            "filled_qty": 1.0,
            "protection_type": "take_profit",
        }
    )

    assert engine.state.active is False
    assert engine.state.segment_state == SegmentState.IDLE.value
    assert engine.holds_real_grid_slot() is False


def test_chop_tp_protection_fill_deactivates_when_no_replenish(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path, max_replenish=0),
        state_path=tmp_path / "chop_state.json",
        level_notional=100.0,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.grid_id = "grid"
    engine.state.spacing = 1.0
    engine.state.inventory = [
        GridPosition(
            symbol="BTCUSDT",
            side="LONG",
            level=1,
            entry_price=99.0,
            quantity=1.0,
            entry_time="2026-01-01T00:00:00Z",
            leg_id="BTCUSDT_grid_L1",
            protection_order_ids=["tp_1"],
        )
    ]

    engine.on_execution_report(
        {
            "order_id": "tp_1",
            "status": "FILLED",
            "filled_qty": 1.0,
            "leg_id": "BTCUSDT_grid_L1_tp",
        }
    )

    assert engine.state.active is False
    assert engine.state.segment_state == SegmentState.IDLE.value


def test_chop_tp_with_replenish_stays_active(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path),
        state_path=tmp_path / "chop_state.json",
        level_notional=100.0,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.grid_id = "grid"
    engine.state.center = 100.0
    engine.state.spacing = 1.0
    engine.state.inventory = [
        GridPosition(
            symbol="BTCUSDT",
            side="LONG",
            level=1,
            entry_price=99.0,
            quantity=1.0,
            entry_time="2026-01-01T00:00:00Z",
            leg_id="BTCUSDT_grid_L1",
            protection_order_ids=["tp_1"],
        )
    ]

    engine.on_execution_report(
        {
            "order_id": "tp_1",
            "status": "FILLED",
            "filled_qty": 1.0,
            "leg_id": "BTCUSDT_grid_L1_tp",
        }
    )

    assert engine.state.active is True
    assert engine.state.pending_orders


def test_chop_multi_leg_tp_deactivates_when_last_leg_closes(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path, max_replenish=0),
        state_path=tmp_path / "chop_state.json",
        level_notional=100.0,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.grid_id = "grid"
    engine.state.spacing = 1.0
    engine.state.inventory = [
        GridPosition(
            symbol="BTCUSDT",
            side="LONG",
            level=1,
            entry_price=99.0,
            quantity=1.0,
            entry_time="t0",
            leg_id="L1",
            protection_order_ids=["tp_l1"],
        ),
        GridPosition(
            symbol="BTCUSDT",
            side="SHORT",
            level=1,
            entry_price=101.0,
            quantity=1.0,
            entry_time="t0",
            leg_id="S1",
            protection_order_ids=["tp_s1"],
        ),
    ]

    engine.on_execution_report(
        {"order_id": "tp_l1", "status": "FILLED", "filled_qty": 1.0}
    )
    assert engine.state.active is True

    engine.on_execution_report(
        {"order_id": "tp_s1", "status": "FILLED", "filled_qty": 1.0}
    )
    assert engine.state.active is False


def test_trend_sl_protection_fill_deactivates(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=tmp_path / "trend_state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.inventory = [
        DualAddPosition(
            leg_id="long_0",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            quantity=1.0,
            seq=0,
            entry_time="t0",
            protection_order_ids=["sl_1"],
        )
    ]

    engine.on_execution_report(
        {
            "order_id": "sl_1",
            "status": "FILLED",
            "filled_qty": 1.0,
            "protection_type": "stop_loss",
        }
    )

    assert engine.state.active is False


def test_chop_regime_exit_clears_state(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path),
        state_path=tmp_path / "chop_state.json",
        level_notional=100.0,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.grid_id = "grid"
    engine.state.spacing = 1.0
    engine.state.center = 100.0
    engine.state.inventory = [
        GridPosition(
            symbol="BTCUSDT",
            side="LONG",
            level=1,
            entry_price=99.0,
            quantity=1.0,
            entry_time="t0",
            leg_id="L1",
        )
    ]

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T02:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.1, "box_prefilter": False},
    )

    assert any(a["action"] == "market_exit" for a in actions)
    assert engine.state.inventory == []
    assert engine.state.pending_orders == []
    assert engine.state.active is False


def test_trend_regime_exit_enters_closing_then_idles(tmp_path: Path) -> None:
    engine = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=tmp_path / "trend_state.json",
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.center = 100.0
    engine.state.atr = 2.0
    engine.state.trend_side = "LONG"
    engine.state.inventory = [
        DualAddPosition("long_0", "BTCUSDT", "LONG", 100.0, 1.0, 0, "t0")
    ]
    engine.state.pending_orders.append(
        DualAddOrder(
            order_id="pending_cancel",
            symbol="BTCUSDT",
            side="BUY",
            price=100.0,
            quantity=1.0,
            reason="entry",
            exchange_order_id="ex_cancel",
            client_order_id="cl_cancel",
            reference_price=100.0,
            max_slippage_bps=5.0,
            seq=0,
            created_bar=1,
        )
    )
    engine.state.bar_index = 0

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T02:00:00Z",
        high=100.0,
        low=99.0,
        close=99.0,
        atr=2.0,
        features={
            "trend_confidence": 0.1,
            "trend_direction": "UP",
            "semantic_chop": 0.0,
            "box_prefilter": False,
        },
    )

    assert any(a["action"] == "market_exit" for a in actions)
    assert engine.state.segment_state == SegmentState.CLOSING.value
    assert engine.state.inventory == []

    engine.on_execution_results(
        [
            GridExecutionResult(
                action="cancel",
                status="canceled",
                symbol="BTCUSDT",
                order_id="ex_cancel",
                client_order_id="cl_cancel",
                raw={"local_order_id": "pending_cancel"},
            )
        ]
    )
    assert engine.state.active is False
    assert engine.state.segment_state == SegmentState.IDLE.value


def test_ghost_does_not_occupy_slot(tmp_path: Path) -> None:
    gate = MultiLegConcurrencyGate(1)
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path),
        state_path=tmp_path / "ghost.json",
        level_notional=100.0,
    )
    engine.state.symbol = "BTCUSDT"
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    gate.register("BTCUSDT", engine)

    assert engine.holds_real_grid_slot() is False
    assert gate.allow_new_segment("ETHUSDT") is True


def test_bar_simulation_wind_down_deactivates(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path, max_replenish=0),
        state_path=tmp_path / "sim.json",
        level_notional=100.0,
        bar_simulation=True,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=98.0,
        close=100.0,
        atr=2.0,
        features=_enter_features(),
    )
    assert engine.state.active is True

    engine.state.inventory = []
    engine.state.pending_orders = []
    engine._maybe_deactivate_if_fully_closed()

    assert engine.state.inventory == []
    assert engine.state.pending_orders == []
    assert engine.state.active is False
    assert engine.state.segment_state == SegmentState.IDLE.value


def test_deactivate_persists_segment_state(tmp_path: Path) -> None:
    state_path = tmp_path / "persist.json"
    engine = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=state_path,
        unit_notional=100.0,
    )
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.symbol = "BTCUSDT"
    engine.state.segment_id = "seg"
    engine.state.inventory = [
        DualAddPosition(
            leg_id="long_0",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            quantity=1.0,
            seq=0,
            entry_time="t0",
            protection_order_ids=["tp_1"],
        )
    ]

    engine.on_execution_report(
        {"order_id": "tp_1", "status": "FILLED", "filled_qty": 1.0}
    )

    reloaded = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=state_path,
        unit_notional=100.0,
    )
    assert reloaded.state.segment_state == SegmentState.IDLE.value
    assert reloaded.state.active is False


def test_legacy_active_json_migrates_on_load(tmp_path: Path) -> None:
    state_path = tmp_path / "legacy.json"
    state_path.write_text(
        """
{
  "grid_id": "BTCUSDT_old",
  "symbol": "BTCUSDT",
  "active": true,
  "center": 100.0,
  "spacing": 1.0,
  "pending_orders": [],
  "inventory": [{"symbol": "BTCUSDT", "side": "LONG", "level": 1,
    "entry_price": 99.0, "quantity": 1.0, "entry_time": "t0", "leg_id": "L1"}],
  "last_timestamp": "2026-01-01T00:00:00Z",
  "current_regime": "chop_grid"
}
""",
        encoding="utf-8",
    )
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path),
        state_path=state_path,
        level_notional=100.0,
    )
    assert engine.state.segment_state == SegmentState.ACTIVE.value
    assert engine.state.active is True


def test_deactivate_records_segment_strategy_event(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    class _FakeMetrics:
        def record_strategy_event(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _FakeMetrics(),
    )

    engine = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=tmp_path / "event.json",
        unit_notional=100.0,
    )
    engine.state.symbol = "BTCUSDT"
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.inventory = []
    engine.state.pending_orders = []

    engine._deactivate("ghost_cleared")

    assert engine.state.active is False
    assert len(calls) == 1
    assert calls[0]["event"] == "segment_ghost_cleared"
    assert calls[0]["scope"] == "hedge"
    assert calls[0]["symbol"] == "BTCUSDT"


@pytest.mark.parametrize(
    ("reason", "event"),
    [
        ("ghost_cleared", "segment_ghost_cleared"),
        ("fully_closed", "segment_fully_closed"),
        ("regime_exit", "segment_regime_exit"),
    ],
)
def test_deactivate_records_reason_specific_segment_event(
    tmp_path: Path,
    monkeypatch,
    reason: str,
    event: str,
) -> None:
    calls: list[dict[str, str]] = []

    class _FakeMetrics:
        def record_strategy_event(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _FakeMetrics(),
    )

    engine = DualAddTrendLiveEngine(
        config_path=_trend_config(tmp_path),
        state_path=tmp_path / f"event_{reason}.json",
        unit_notional=100.0,
    )
    engine.state.symbol = "BTCUSDT"
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value

    engine._deactivate(reason)

    assert calls[0]["event"] == event


def test_deactivate_still_runs_when_metrics_fail(tmp_path: Path, monkeypatch) -> None:
    gate = MultiLegConcurrencyGate(1)

    class _BrokenMetrics:
        def record_strategy_event(self, **kwargs) -> None:
            raise RuntimeError("metrics down")

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _BrokenMetrics(),
    )

    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path),
        state_path=tmp_path / "metrics_fail.json",
        level_notional=100.0,
    )
    engine.state.symbol = "BTCUSDT"
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
    gate.register("BTCUSDT", engine)

    engine._deactivate("ghost_cleared")

    assert engine.state.active is False
    assert engine.state.segment_state == SegmentState.IDLE.value
    assert gate.allow_new_segment("ETHUSDT") is True


def test_chop_market_exit_shadow_does_not_clear_inventory(
    tmp_path: Path,
) -> None:
    """shadow=True 时 GridExecutionAdapter 不发实单，chop_grid 不应清本地状态。"""
    engine = ChopGridLiveEngine(
        config_path=_chop_config(tmp_path),
        state_path=tmp_path / "shadow_exit.json",
        level_notional=100.0,
    )
    # Simulate a position that would produce a dust exit
    pos = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=100.0,
        quantity=0.001,
        entry_time="2026-01-01T02:00:00Z",
        leg_id="shadow_leg",
    )
    engine.state.inventory = [pos]
    engine.state.active = True
    engine.state.segment_state = SegmentState.ACTIVE.value
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
                    "reason": "dust_exit",
                },
            )
        ]
    )

    # shadow = paper mode, no real fill → inventory must remain
    assert engine.state.inventory == [pos]
