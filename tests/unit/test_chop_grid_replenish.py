"""Chop grid per-level replenish (backtest engine)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from src.time_series_model.grid.chop_grid_engine import ChopGridEngine, GridEngineConfig
from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridPosition,
)


def _synthetic_segment() -> pd.DataFrame:
    """Two bars touch L1 twice with a TP bar in between."""
    idx = pd.date_range("2024-01-01", periods=5, freq="2h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 101.0, 100.5, 100.5],
            "low": [98.0, 98.0, 98.0, 98.0, 98.0],
            "close": [99.0, 99.0, 100.5, 99.0, 99.0],
            "atr14": [2.0, 2.0, 2.0, 2.0, 2.0],
            "semantic_chop": [0.55, 0.55, 0.55, 0.55, 0.55],
        },
        index=idx,
    )


def _run_long_l1_tp(max_replenish: int | None) -> int:
    cfg = GridEngineConfig(
        grid_atr_mult=0.5,
        grid_min_pct=0.004,
        max_levels_per_side=1,
        max_replenish_per_level_per_segment=max_replenish,
        same_bar_entry_exit=True,
    )
    engine = ChopGridEngine(cfg)
    result = engine.simulate_segment(
        _synthetic_segment(),
        symbol="BTCUSDT",
        regime="chop",
        segment_id="t1",
    )
    return sum(
        1
        for t in result.trades
        if t.exit_reason == "grid_tp" and t.side == "LONG" and t.level == 1
    )


def test_max_replenish_zero_allows_only_one_grid_tp() -> None:
    assert _run_long_l1_tp(0) == 1


def test_max_replenish_one_allows_two_grid_tps() -> None:
    assert _run_long_l1_tp(1) == 2


def test_max_replenish_unlimited_allows_multiple_grid_tps() -> None:
    assert _run_long_l1_tp(None) >= 2


def _live_config(tmp_path: Path, max_replenish: int) -> Path:
    path = tmp_path / "grid.yaml"
    path.write_text(
        f"""
regime:
  entry_chop_min: 0.40
  exit_chop_below: 0.25
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 1
  max_replenish_per_level_per_segment: {max_replenish}
risk:
  fee_bps: 4.0
  max_open_levels_total: 2
""",
        encoding="utf-8",
    )
    return path


def test_live_replenish_after_tp_when_max_one(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_live_config(tmp_path, 1),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
        bar_simulation=True,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    pos = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=98.0,
        quantity=1.0,
        entry_time="2026-01-01T02:00:00Z",
        leg_id=f"{engine.state.grid_id}_L1",
    )
    engine.state.inventory = [pos]
    engine.state.pending_orders = []
    actions = engine._after_level_tp_closed(pos, "2026-01-01T04:00:00Z")
    assert engine.state.level_replenish_count.get("L1") == 1
    assert any(a.get("action") == "place" for a in actions)
    assert any(o.order_id.endswith("_r1") for o in engine.state.pending_orders)


def test_live_no_phantom_replenish_when_sync_clears_inventory(tmp_path: Path) -> None:
    """If exchange sync clears inventory without a TP execution report,
    `_maybe_replenish_empty_levels` must not place a duplicate limit.

    Regression: max_replenish=0 must preserve current one-shot live behavior
    even when TP detection is lost.
    """
    engine = ChopGridLiveEngine(
        config_path=_live_config(tmp_path, 0),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
        bar_simulation=False,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    engine.state.inventory = []
    engine.state.pending_orders = []
    actions = engine._maybe_replenish_empty_levels("BTCUSDT", "2026-01-01T02:00:00Z")
    assert actions == []
    assert engine.state.pending_orders == []


def test_live_replenish_one_then_blocks_with_max_one(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_live_config(tmp_path, 1),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
        bar_simulation=True,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    pos1 = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=98.0,
        quantity=1.0,
        entry_time="t1",
        leg_id=f"{engine.state.grid_id}_L1",
    )
    engine.state.inventory = [pos1]
    engine.state.pending_orders = []
    acts1 = engine._after_level_tp_closed(pos1, "t2")
    assert acts1 and acts1[0]["action"] == "place"
    pos2 = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=98.0,
        quantity=1.0,
        entry_time="t3",
        leg_id=engine.state.pending_orders[0].order_id,
    )
    engine.state.inventory = [pos2]
    engine.state.pending_orders = []
    acts2 = engine._after_level_tp_closed(pos2, "t4")
    assert acts2 == []
    assert engine.state.level_replenish_count["L1"] == 2


def test_live_no_replenish_when_max_zero(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_live_config(tmp_path, 0),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
        bar_simulation=True,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    pos = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=98.0,
        quantity=1.0,
        entry_time="2026-01-01T02:00:00Z",
        leg_id=f"{engine.state.grid_id}_L1",
    )
    engine.state.inventory = [pos]
    actions = engine._after_level_tp_closed(pos, "2026-01-01T04:00:00Z")
    assert engine.state.level_replenish_count.get("L1") == 1
    assert not actions
