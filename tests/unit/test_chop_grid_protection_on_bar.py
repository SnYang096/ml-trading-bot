"""Test that protection actions from order fills are returned by on_bar."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridPosition,
)
from src.time_series_model.live.segment_lifecycle import SegmentState


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "grid.yaml"
    path.write_text(
        """
regime:
  entry_chop_min: 0.40
  exit_chop_below: 0.25
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 3
risk:
  fee_bps: 4.0
""",
        encoding="utf-8",
    )
    return path


def test_on_bar_returns_protection_actions_from_pending(
    tmp_path: Path,
) -> None:
    """Verify that TP/SL actions generated in on_order_fill are returned by on_bar.

    Regression test for the bug where _pending_actions were never merged into
    on_bar's return value, causing all live positions to lack TP/SL orders.
    """
    state_path = tmp_path / "state.json"
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )

    # Setup: active grid (no position yet — will be created by late fill)
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"
    engine.state.spacing = 100.0  # $100 spacing
    engine.state.center = 50000.0
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.save_state()

    # Simulate order fill → generates TP action in _pending_actions via late fill ingestion
    fill_report = {
        "status": "FILLED",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "filled_qty": 0.1,
        "avg_price": 50000.0,
        "trade_time": "2026-06-15T00:00:00+00:00",
        "local_order_id": "BTCUSDT_grid_L1",  # Must match leg_id format for late fill
    }
    engine.on_execution_report(fill_report)

    # Verify TP action is in pending queue
    assert len(engine._pending_actions) == 1
    assert engine._pending_actions[0]["action"] == "place_protection"
    assert engine._pending_actions[0]["protection_type"] == "take_profit"

    # Call on_bar — should merge pending actions into return value
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-06-15T00:01:00+00:00",
        high=50100.0,
        low=49900.0,
        close=50050.0,
        atr=200.0,
        features={},
    )

    # Verify TP action is included in on_bar return
    tp_actions = [a for a in actions if a.get("action") == "place_protection"]
    assert (
        len(tp_actions) == 1
    ), f"Expected 1 TP action, got {len(tp_actions)}: {actions}"
    assert tp_actions[0]["leg_id"] == "BTCUSDT_grid_L1"
    assert tp_actions[0]["protection_type"] == "take_profit"
    # TP price = entry + spacing * tp_spacing_mult (default 1.0 from config)
    expected_tp_price = 50000.0 + 100.0 * 1.0  # 50100
    assert tp_actions[0]["price"] == pytest.approx(expected_tp_price)

    # Verify pending queue is cleared after pop
    assert engine._pending_actions == []


def test_on_bar_clears_pending_after_merge(
    tmp_path: Path,
) -> None:
    """Verify that pending actions are cleared after being merged into on_bar."""
    state_path = tmp_path / "state.json"
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )

    # Manually add a pending action
    engine._pending_actions.append(
        {
            "action": "place_protection",
            "order_id": "test_tp",
            "leg_id": "TEST_L1",
            "symbol": "ETHUSDT",
            "side": "LONG",
            "quantity": 1.0,
            "price": 3000.0,
            "protection_type": "take_profit",
        }
    )

    # Call on_bar
    actions = engine.on_bar(
        symbol="ETHUSDT",
        timestamp="2026-06-15T00:00:00+00:00",
        high=3000.0,
        low=2900.0,
        close=2950.0,
        atr=50.0,
        features={},
    )

    # Pending should be cleared
    assert engine._pending_actions == []
    # Action should be in return value
    assert any(a.get("order_id") == "test_tp" for a in actions)
