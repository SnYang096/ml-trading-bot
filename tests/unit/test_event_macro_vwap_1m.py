"""Regression: vwap1200 deadband must be evaluable on 1m closes between primary TF bars."""

from __future__ import annotations

import pandas as pd

from scripts.event_backtest import (
    PositionSimulator,
    _sync_macro_tp_vwap_from_feature_row,
)
from src.time_series_model.live.position_logic import enforce_position


def test_frozen_vwap_level_identity_from_feature_row() -> None:
    close = 1.0
    pv = 0.02
    vwap = close * (1.0 - pv)
    assert abs((close - vwap) / close - pv) < 1e-9


def test_live_pv_crosses_deadband_using_frozen_level() -> None:
    sim = PositionSimulator()
    row = pd.Series(
        {"close": 100.0, "macro_tp_vwap_1200_position": 0.02},
    )
    _sync_macro_tp_vwap_from_feature_row(sim, row)
    assert sim._macro_tp_vwap_level == 98.0
    bar_close = 98.2
    live_pv = (bar_close - float(sim._macro_tp_vwap_level)) / bar_close
    assert abs(live_pv) < 0.005


def test_position_simulator_passes_intraminute_pv_to_enforce() -> None:
    sim = PositionSimulator()
    sim._macro_tp_vwap_level = 100.0
    sim._macro_tp_vwap_position = 0.05
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    pos = {
        "side": "LONG",
        "entry_price": 99.0,
        "entry_time": now,
        "atr_at_entry": 1.0,
        "max_holding_bars": 0,
        "bar_minutes": 120,
        "stop_loss_price": 90.0,
        "structural_exit": "vwap1200",
        "vwap_exit_inner_abs": 0.005,
        "breakeven_locked": True,
    }
    sim._positions["p1"] = pos
    closed = sim.update(
        {
            "timestamp": now,
            "open": 100.1,
            "high": 100.2,
            "low": 100.0,
            "close": 100.15,
        }
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "structural_exit_vwap1200"
