"""Per-leg grid_sl simulation (backtest research path only)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.time_series_model.grid.chop_grid_engine import ChopGridEngine, GridEngineConfig


def test_short_grid_sl_triggers_on_high_not_low():
    idx = pd.date_range("2026-01-01", periods=3, freq="2h", tz="UTC")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0],
            "high": [101.0, 103.0, 113.0],
            "low": [99.0, 101.0, 108.0],
            "close": [100.0, 102.0, 112.0],
            "atr14": [2.0, 2.0, 2.0],
            "semantic_chop": [0.6, 0.6, 0.6],
        },
        index=idx,
    )
    engine = ChopGridEngine(
        GridEngineConfig(
            max_levels_per_side=1,
            grid_atr_mult=1.0,
            grid_min_pct=0.01,
            per_leg_sl_spacing_mult=5.0,
            same_bar_entry_exit=False,
            fee_bps=4.0,
        )
    )
    result = engine.simulate_segment(
        seg,
        symbol="TESTUSDT",
        regime="chop",
        segment_id="t",
    )
    sl_trades = [t for t in result.trades if t.exit_reason == "grid_sl"]
    assert len(sl_trades) == 1
    trade = sl_trades[0]
    assert trade.side == "SHORT"
    assert trade.entry_price == 102.0
    assert trade.exit_price == 112.0


def test_long_emergency_sl_triggers_at_entry_pct():
    idx = pd.date_range("2026-01-01", periods=4, freq="2h", tz="UTC")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0, 97.0, 82.0],
            "high": [100.5, 100.5, 98.0, 83.0],
            "low": [99.5, 96.0, 96.0, 81.0],
            "close": [100.0, 97.0, 97.0, 82.0],
            "atr14": [2.0, 2.0, 2.0, 2.0],
            "semantic_chop": [0.6, 0.6, 0.6, 0.6],
        },
        index=idx,
    )
    engine = ChopGridEngine(
        GridEngineConfig(
            max_levels_per_side=1,
            grid_atr_mult=1.0,
            grid_min_pct=0.01,
            emergency_stop_loss_enabled=True,
            emergency_stop_loss_trigger_pct=0.15,
            same_bar_entry_exit=False,
            fee_bps=4.0,
        )
    )
    result = engine.simulate_segment(
        seg,
        symbol="TESTUSDT",
        regime="chop",
        segment_id="t",
    )
    em_trades = [t for t in result.trades if t.exit_reason == "emergency_sl"]
    assert len(em_trades) == 1
    trade = em_trades[0]
    assert trade.side == "LONG"
    assert trade.entry_price == 98.0
    assert trade.exit_price == pytest.approx(98.0 * 0.85)
