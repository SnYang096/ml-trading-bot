"""Regression: live-aligned exec window must not fill before signal bar confirms."""

from __future__ import annotations

import pandas as pd

from scripts.chop_grid_backtest import (
    ChopGridEngine,
    GridEngineConfig,
    collect_chop_grid_trades_for_symbol,
)
from scripts.diagnose_chop_grid import GridConfig
from src.time_series_model.grid.subbar_replay import (
    merge_signal_features_onto_execution_bars,
    segment_execution_bounds,
    timeframe_to_timedelta,
)


def _synthetic_sol_like_segment() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timedelta]:
    """Entry signal bar has a spike; post-confirm path does not reach short grid."""
    sig_idx = pd.DatetimeIndex(
        [
            "2025-10-27 06:00:00+00:00",
            "2025-10-27 08:00:00+00:00",
            "2025-10-27 10:00:00+00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [204.0, 202.0, 200.0],
            "high": [204.94, 202.4, 200.8],
            "low": [202.0, 198.7, 198.5],
            "close": [202.45, 199.9, 200.2],
            "volume": [1.0, 1.0, 1.0],
            "atr14": [2.16, 2.16, 2.16],
            "semantic_chop": [0.55, 0.55, 0.55],
            "box_prefilter": [False, False, False],
        },
        index=sig_idx,
    )
    delta = timeframe_to_timedelta("2h")
    t_enter, t_exit = segment_execution_bounds(sig_idx, 0, 2, delta)
    exec_idx = pd.date_range(
        t_enter, t_exit - pd.Timedelta(minutes=1), freq="1min", tz="UTC"
    )
    df_exec = pd.DataFrame(
        {
            "open": 202.0,
            "high": 203.08,
            "low": 198.7,
            "close": 200.0,
            "volume": 1.0,
        },
        index=exec_idx,
    )
    df_exec = merge_signal_features_onto_execution_bars(
        df_exec, df, signal_bar_delta=delta
    )
    return df, df_exec, delta


def test_subbar_path_skips_pre_confirm_spike_short_fill():
    df, df_exec, delta = _synthetic_sol_like_segment()
    cfg = GridConfig(
        chop_min=0.50,
        exit_chop_min=0.32,
        grid_atr_mult=1.0,
        grid_pct=0.01,
        max_levels=2,
        min_segment_bars=1,
        max_segment_bars=10,
    )
    engine = ChopGridEngine(
        GridEngineConfig(
            entry_chop_min=0.50,
            exit_chop_below=0.32,
            grid_atr_mult=1.0,
            grid_min_pct=0.01,
            max_levels_per_side=2,
            fee_bps=4.0,
            max_replenish_per_level_per_segment=1,
            same_bar_entry_exit=False,
            min_segment_bars=1,
            max_segment_bars=10,
        )
    )
    trades, summaries, n_seg, _ = collect_chop_grid_trades_for_symbol(
        "SOLUSDT",
        df,
        df_exec,
        delta,
        cfg,
        engine,
        block_stable_box=False,
        exec_timeframe="1min",
    )
    assert n_seg == 1
    assert summaries
    short_trades = [t for t in trades if t.get("side") == "SHORT"]
    assert not short_trades, "short must not fill from pre-confirm spike on subbar path"


def test_legacy_signal_bar_path_can_fill_pre_confirm_spike():
    """Legacy direct signal-bar sim (sweep --no-exec-merge) remains optimistic."""
    df, _, _ = _synthetic_sol_like_segment()
    cfg = GridConfig(
        chop_min=0.50,
        exit_chop_min=0.32,
        grid_atr_mult=1.0,
        grid_pct=0.01,
        max_levels=2,
        min_segment_bars=1,
        max_segment_bars=10,
    )
    engine = ChopGridEngine(
        GridEngineConfig(
            entry_chop_min=0.50,
            exit_chop_below=0.32,
            grid_atr_mult=1.0,
            grid_min_pct=0.01,
            max_levels_per_side=2,
            fee_bps=4.0,
            max_replenish_per_level_per_segment=1,
            min_segment_bars=1,
            max_segment_bars=10,
        )
    )
    trades, _, _, _ = collect_chop_grid_trades_for_symbol(
        "SOLUSDT",
        df,
        None,
        None,
        cfg,
        engine,
        block_stable_box=False,
    )
    short_trades = [t for t in trades if t.get("side") == "SHORT"]
    assert short_trades, "legacy path uses entry bar OHLC including pre-confirm spike"
