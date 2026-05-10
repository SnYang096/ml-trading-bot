from __future__ import annotations

import pandas as pd

from scripts.diagnose_chop_grid import GridConfig, simulate_fixed_grid
from scripts.diagnose_dual_add_trend import DualAddConfig, simulate_dual_add_segment


def test_chop_grid_signal_bar_does_not_fill_levels() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="2h")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [102.0, 100.5],
            "low": [98.0, 99.5],
            "close": [100.0, 100.0],
            "atr14": [1.0, 1.0],
        },
        index=idx,
    )
    cfg = GridConfig(grid_atr_mult=1.0, grid_pct=0.0, max_levels=1, fee_bps=0.0)

    result = simulate_fixed_grid(seg, cfg=cfg)

    assert result["fills"] == 0
    assert result["cycles"] == 0
    assert result["forced_exits"] == 0


def test_dual_add_signal_bar_does_not_take_profit() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="2h")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [102.0, 100.5],
            "low": [98.0, 99.5],
            "close": [100.0, 100.0],
            "atr14": [1.0, 1.0],
            "trend_direction": ["UP", "UP"],
        },
        index=idx,
    )
    cfg = DualAddConfig(
        add_mode="both",
        tp_atr_mult=1.0,
        fee_bps=0.0,
        max_adds_per_side=0,
        max_loss_per_segment=1.0,
    )

    trades, summary = simulate_dual_add_segment(
        seg,
        cfg=cfg,
        symbol="BTCUSDT",
        segment_id="seg0",
        direction="UP",
    )

    assert summary["tp"] == 0
    assert summary["forced"] == 2
    assert len(trades) == 2


def test_dual_add_basket_take_profit_closes_inventory_together() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="2h")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0, 102.0],
            "high": [100.2, 101.0, 103.0],
            "low": [99.8, 100.0, 101.5],
            "close": [100.0, 101.0, 102.0],
            "atr14": [1.0, 1.0, 1.0],
            "trend_direction": ["UP", "UP", "UP"],
        },
        index=idx,
    )
    cfg = DualAddConfig(
        add_mode="trend",
        take_profit_mode="basket",
        tp_atr_mult=0.25,
        fee_bps=0.0,
        max_adds_per_side=1,
        max_loss_per_segment=1.0,
    )

    trades, summary = simulate_dual_add_segment(
        seg,
        cfg=cfg,
        symbol="BTCUSDT",
        segment_id="seg0",
        direction="UP",
    )

    assert summary["tp"] == 3
    assert summary["forced"] == 0
    assert {trade["exit_reason"] for trade in trades} == {"basket_tp"}


def test_dual_add_basket_exit_slippage_makes_backtest_conservative() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="2h")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0, 102.0],
            "high": [100.2, 101.0, 103.0],
            "low": [99.8, 100.0, 101.5],
            "close": [100.0, 101.0, 102.0],
            "atr14": [1.0, 1.0, 1.0],
            "trend_direction": ["UP", "UP", "UP"],
        },
        index=idx,
    )
    base_cfg = DualAddConfig(
        add_mode="trend",
        take_profit_mode="basket",
        tp_atr_mult=0.25,
        fee_bps=0.0,
        max_adds_per_side=1,
        max_loss_per_segment=1.0,
    )
    conservative_cfg = DualAddConfig(
        add_mode="trend",
        take_profit_mode="basket",
        tp_atr_mult=0.25,
        fee_bps=0.0,
        market_exit_slippage_bps=5.0,
        max_adds_per_side=1,
        max_loss_per_segment=1.0,
    )

    _, base_summary = simulate_dual_add_segment(
        seg,
        cfg=base_cfg,
        symbol="BTCUSDT",
        segment_id="base",
        direction="UP",
    )
    trades, conservative_summary = simulate_dual_add_segment(
        seg,
        cfg=conservative_cfg,
        symbol="BTCUSDT",
        segment_id="conservative",
        direction="UP",
    )

    assert conservative_summary["pnl_per_capital"] < base_summary["pnl_per_capital"]
    assert all(trade["slippage_bps_charged"] == 5.0 for trade in trades)


def test_dual_add_intrabar_touch_buffer_requires_extra_penetration() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="2h")
    seg = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.0, 101.02],
            "low": [100.0, 99.8],
            "close": [100.0, 100.5],
            "atr14": [1.0, 1.0],
            "trend_direction": ["UP", "UP"],
        },
        index=idx,
    )
    cfg = DualAddConfig(
        add_mode="trend",
        step_atr_mult=1.0,
        take_profit_mode="basket",
        fee_bps=0.0,
        intrabar_touch_buffer_bps=5.0,
        max_adds_per_side=1,
        max_loss_per_segment=1.0,
        max_gross_exposure=4,
        initial_hedge=False,
    )

    trades, summary = simulate_dual_add_segment(
        seg,
        cfg=cfg,
        symbol="BTCUSDT",
        segment_id="seg0",
        direction="UP",
    )

    assert summary["max_add_long"] == 0
    assert len(trades) == 1
