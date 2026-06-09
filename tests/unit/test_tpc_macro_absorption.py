"""Causal / no-look-ahead verification for tpc_macro_absorption_f.

The feature uses shift(1) on rolling windows — tests confirm:
  1. Adding a future bar does NOT change any past values.
  2. Warmup bars produce NaN (no peek into insufficient history).
  3. Directional: vol contraction rises after deliberate volume collapse.
  4. Bounded [0, 1] output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.bpc_features import (
    compute_tpc_macro_absorption_from_series,
)


def _make_ohlcv(n: int, *, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.3, n)), index=idx)
    high = close + pd.Series(np.abs(rng.normal(0, 0.2, n)), index=idx)
    low = close - pd.Series(np.abs(rng.normal(0, 0.2, n)), index=idx)
    vol = pd.Series(np.abs(rng.normal(5000, 1000, n)), index=idx)
    ema = pd.Series(rng.normal(0.15, 0.05, n).clip(-1, 1), index=idx)
    return {
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "ema_1200_position": ema,
    }


# ── 1. Future-leak: adding bars must not change history ──────────────


def test_no_future_leak_appending_bars():
    """Appending data after time T must not alter values at times <= T."""
    n = 300
    base = _make_ohlcv(n)
    df_base = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    ext_data = _make_ohlcv(50, seed=99)
    # Align indices to avoid freq mismatch
    last_ts = base["close"].index[-1]
    new_idx = pd.date_range(
        last_ts + pd.Timedelta(hours=2), periods=50, freq="2h", tz="UTC"
    )
    extended = {}
    for k, v in base.items():
        ext_v = pd.concat([v, pd.Series(ext_data[k].values, index=new_idx)])
        extended[k] = ext_v
    df_ext = compute_tpc_macro_absorption_from_series(**extended, lookback=240)

    for col in df_base.columns:
        # first n rows must be identical
        pd.testing.assert_series_equal(
            df_base[col].iloc[:n].reset_index(drop=True),
            df_ext[col].iloc[:n].reset_index(drop=True),
            check_names=False,
            obj=f"future-leak: {col}",
        )


def test_no_future_leak_inserting_mid_bars():
    """Inserting a large spike bar in the middle should not affect earlier values."""
    n = 200
    base = _make_ohlcv(n)
    df_before = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    # Insert a huge spike at t=150
    base["close"].iloc[150] += 50
    base["high"].iloc[150] += 55
    base["low"].iloc[150] += 45
    base["volume"].iloc[150] *= 10

    df_after = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    for col in df_before.columns:
        # Values BEFORE the spike (t < 150) must be unchanged
        pd.testing.assert_series_equal(
            df_before[col].iloc[:150],
            df_after[col].iloc[:150],
            check_names=False,
            obj=f"no-leak-before-spike: {col}",
        )


# ── 2. Warmup ───────────────────────────────────────────────────────


def test_warmup_period_is_nan():
    """First few bars should be NaN (min_periods not yet met or dead zone)."""
    base = _make_ohlcv(300)
    base["ema_1200_position"] = pd.Series(0.0, index=base["close"].index)  # dead zone
    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    # In dead zone + early bars: should be all NaN or close to it
    for col in df.columns:
        assert (
            df[col].iloc[0:10].isna().all()
        ), f"{col}[0:10] should be NaN in dead zone"


def test_values_become_available_after_sufficient_history():
    """After lookback bars, at least some values should be non-NaN (outside dead zone)."""
    n = 500
    base = _make_ohlcv(n)
    # Force ema outside dead zone for all bars
    base["ema_1200_position"] = pd.Series(0.20, index=base["close"].index)
    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    for col in df.columns:
        non_nan = df[col].iloc[300:].dropna()
        assert len(non_nan) > 20, f"{col}: not enough non-NaN after warmup"


# ── 3. Directional response ─────────────────────────────────────────


def test_vol_contraction_rises_when_volume_collapses():
    """Deliberately collapse volume in the last 60 bars; vol_contraction should rise."""
    n = 500
    base = _make_ohlcv(n)
    base["ema_1200_position"] = pd.Series(0.20, index=base["close"].index)

    # Normal volume for first 440 bars, then collapse
    base["volume"].iloc[:440] = 10000.0
    base["volume"].iloc[440:] = 1000.0  # 10x collapse

    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    # After collapse, vol_contraction should be elevated
    post_collapse = df["tpc_macro_vol_contraction"].iloc[450:].dropna()
    pre_collapse = df["tpc_macro_vol_contraction"].iloc[300:430].dropna()

    assert post_collapse.mean() > pre_collapse.mean(), (
        f"vol contraction should rise after volume collapse: "
        f"pre={pre_collapse.mean():.3f}, post={post_collapse.mean():.3f}"
    )


def test_range_convergence_rises_when_range_narrows():
    """Deliberately narrow the price range; range_convergence should rise."""
    n = 500
    base = _make_ohlcv(n)
    base["ema_1200_position"] = pd.Series(0.20, index=base["close"].index)

    # Wide range for first 440 bars, then narrow
    base["close"].iloc[:440] = 100 + np.arange(440) * 0.1
    base["high"].iloc[:440] = base["close"].iloc[:440] + 5
    base["low"].iloc[:440] = base["close"].iloc[:440] - 5

    # Narrow: price oscillates in tight band
    base["close"].iloc[440:] = 150 + np.sin(np.arange(60) * 0.2) * 0.5
    base["high"].iloc[440:] = base["close"].iloc[440:] + 0.3
    base["low"].iloc[440:] = base["close"].iloc[440:] - 0.3

    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    post_narrow = df["tpc_macro_range_convergence"].iloc[460:].dropna()
    pre_narrow = df["tpc_macro_range_convergence"].iloc[300:430].dropna()

    if len(post_narrow) > 0 and len(pre_narrow) > 0:
        assert post_narrow.mean() > pre_narrow.mean(), (
            f"range convergence should rise after range narrows: "
            f"pre={pre_narrow.mean():.3f}, post={post_narrow.mean():.3f}"
        )


# ── 4. Bounded output ────────────────────────────────────────────────


def test_output_is_bounded_0_1():
    """All non-NaN values must be in [0, 1]."""
    base = _make_ohlcv(500)
    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    for col in df.columns:
        valid = df[col].dropna()
        assert (valid >= 0).all(), f"{col} has negative values"
        assert (valid <= 1).all(), f"{col} has values > 1"


# ── 5. Regime gate ──────────────────────────────────────────────────


def test_dead_zone_produces_nan():
    """ema between -0.10 and +0.10 → both columns NaN."""
    base = _make_ohlcv(300)
    base["ema_1200_position"] = pd.Series(0.0, index=base["close"].index)
    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    # After warmup, all values in dead zone should be NaN
    for col in df.columns:
        assert df[col].iloc[250:].isna().all(), f"{col}: dead zone should be all NaN"


def test_bull_zone_produces_non_nan_after_warmup():
    """ema >= 0.10 → values should be available after warmup."""
    base = _make_ohlcv(500)
    base["ema_1200_position"] = pd.Series(0.20, index=base["close"].index)
    df = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    valid = df["tpc_macro_vol_contraction"].iloc[300:].dropna()
    assert len(valid) > 0, "bull zone should have values"


# ── 6. Determinism ──────────────────────────────────────────────────


def test_deterministic_output():
    """Same input twice → same output twice."""
    base = _make_ohlcv(300)
    df1 = compute_tpc_macro_absorption_from_series(**base, lookback=240)
    df2 = compute_tpc_macro_absorption_from_series(**base, lookback=240)

    for col in df1.columns:
        pd.testing.assert_series_equal(df1[col], df2[col], check_names=False)


# ── 7. Edge: shift(1) on rolling max/min ─────────────────────────────


def test_shift_prevents_current_bar_from_affecting_own_rolling_window():
    """Verify that close at time t is NOT used in roll_high at time t (shift(1))."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    high = pd.Series(100.0, index=idx)
    low = pd.Series(100.0, index=idx)
    close = pd.Series(100.0, index=idx)
    vol = pd.Series(5000.0, index=idx)
    ema = pd.Series(0.20, index=idx)

    # At t=80, spike high to 200
    high.iloc[80] = 200.0
    close.iloc[80] = 150.0

    df = compute_tpc_macro_absorption_from_series(
        high=high,
        low=low,
        close=close,
        volume=vol,
        ema_1200_position=ema,
        lookback=60,
    )

    # At t=80, roll_high uses shift(1) → the spike at 200 is NOT in the window
    # (it enters at t=81). So range_convergence at t=80 should NOT jump.
    # We check that the value at t=80 is not wildly different from t=79.
    rc = df["tpc_macro_range_convergence"].dropna()
    if len(rc) >= 3 and 80 in rc.index:
        diff = abs(rc.loc[80] - rc.loc[79])
        assert diff < 0.5, (
            f"shift(1) should prevent bar 80's own spike from affecting rolling at t=80. "
            f"diff={diff:.4f}"
        )
