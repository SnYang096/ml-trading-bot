"""Tests for tpc_macro_pullback_pct feature (formula, causality, regime gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.bpc_features import (
    compute_tpc_macro_pullback_pct_from_series,
)


def _make_frame(
    highs: list[float],
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    ema_pos: list[float] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    n = len(highs)
    if lows is None:
        lows = [h * 0.98 for h in highs]
    if closes is None:
        closes = [(h + lo) / 2 for h, lo in zip(highs, lows)]
    if ema_pos is None:
        ema_pos = [0.20] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    return (
        pd.Series(highs, index=idx, dtype=float),
        pd.Series(lows, index=idx, dtype=float),
        pd.Series(closes, index=idx, dtype=float),
        pd.Series(ema_pos, index=idx, dtype=float),
    )


class TestTpcMacroPullbackPct:
    def test_formula_long_drawdown(self):
        """High 100 → close 85 with stable roll_high → long=0.15."""
        n = 260
        highs = [100.0] * n
        lows = [90.0] * n
        closes = [85.0] * n
        ema = [0.20] * n
        high, low, close, ema_pos = _make_frame(highs, lows, closes, ema)

        out = compute_tpc_macro_pullback_pct_from_series(
            high=high,
            low=low,
            close=close,
            ema_1200_position=ema_pos,
            lookback=240,
        )
        val = out["tpc_macro_pullback_pct_long"].iloc[-1]
        assert val == pytest.approx(0.15, abs=1e-6)

    def test_causal_shift_no_current_bar_high(self):
        """Current bar spike must not enter roll_high anchor (shift(1))."""
        n = 260
        highs = [100.0] * (n - 1) + [200.0]
        lows = [90.0] * n
        closes = [95.0] * n
        ema = [0.20] * n
        high, low, close, ema_pos = _make_frame(highs, lows, closes, ema)

        out = compute_tpc_macro_pullback_pct_from_series(
            high=high,
            low=low,
            close=close,
            ema_1200_position=ema_pos,
            lookback=240,
        )
        val = out["tpc_macro_pullback_pct_long"].iloc[-1]
        assert val == pytest.approx(0.05, abs=1e-6)

    def test_regime_gate_bull_long_only(self):
        high, low, close, ema_pos = _make_frame([100.0] * 260, ema_pos=[0.20] * 260)
        close.iloc[-1] = 85.0

        out = compute_tpc_macro_pullback_pct_from_series(
            high=high,
            low=low,
            close=close,
            ema_1200_position=ema_pos,
            lookback=240,
        )
        assert pd.notna(out["tpc_macro_pullback_pct_long"].iloc[-1])
        assert pd.isna(out["tpc_macro_pullback_pct_short"].iloc[-1])

    def test_regime_gate_bear_short_only(self):
        n = 260
        highs = [110.0] * n
        lows = [100.0] * n
        closes = [115.0] * n
        ema = [-0.15] * n
        high, low, close, ema_pos = _make_frame(highs, lows, closes, ema)

        out = compute_tpc_macro_pullback_pct_from_series(
            high=high,
            low=low,
            close=close,
            ema_1200_position=ema_pos,
            lookback=240,
        )
        assert pd.isna(out["tpc_macro_pullback_pct_long"].iloc[-1])
        assert pd.notna(out["tpc_macro_pullback_pct_short"].iloc[-1])
        assert out["tpc_macro_pullback_pct_short"].iloc[-1] == pytest.approx(
            0.15, abs=1e-6
        )

    def test_regime_gate_dead_zone_both_nan(self):
        high, low, close, ema_pos = _make_frame([100.0] * 260, ema_pos=[0.0] * 260)
        out = compute_tpc_macro_pullback_pct_from_series(
            high=high,
            low=low,
            close=close,
            ema_1200_position=ema_pos,
            lookback=240,
        )
        assert pd.isna(out["tpc_macro_pullback_pct_long"].iloc[-1])
        assert pd.isna(out["tpc_macro_pullback_pct_short"].iloc[-1])

    def test_bounded_0_1(self):
        n = 260
        highs = np.linspace(100, 120, n)
        lows = highs - 5
        closes = lows - 10  # deep drawdown
        ema = [0.25] * n
        high, low, close, ema_pos = _make_frame(
            list(highs), list(lows), list(closes), ema
        )
        out = compute_tpc_macro_pullback_pct_from_series(
            high=high,
            low=low,
            close=close,
            ema_1200_position=ema_pos,
            lookback=240,
        )
        long_vals = out["tpc_macro_pullback_pct_long"].dropna()
        assert (long_vals >= 0).all()
        assert (long_vals <= 1).all()


def test_feature_dependencies_node_exists():
    from pathlib import Path

    import yaml

    path = Path("config/feature_dependencies.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = (data.get("features") or {}).get("tpc_macro_pullback_pct_f")
    assert node is not None
    assert node["compute_func"] == "compute_tpc_macro_pullback_pct_from_series"
    assert node["compute_params"]["lookback"] == 240
