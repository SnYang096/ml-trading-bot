import numpy as np
import pandas as pd

from time_series_model.pipeline.training.forward_selection import (
    analyze_timeframe,
    compute_info_efficiency,
    pick_plateau,
)


def _make_ar_series(length: int = 1500, phi: float = 0.7, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, size=length)
    values = np.zeros(length)
    for t in range(1, length):
        values[t] = phi * values[t - 1] + eps[t]
    base = pd.Series(values).cumsum()
    close = 100 + base
    close.index = pd.date_range("2024-01-01", periods=length, freq="15min")
    return close


def test_compute_info_efficiency_returns_series_with_positive_values():
    close = _make_ar_series()
    eff = compute_info_efficiency(close, max_forward=24)
    assert not eff.empty
    assert eff.index.min() == 1
    # Efficiency should be non-negative and finite
    assert (eff.values >= 0).all()
    assert np.isfinite(eff.values).all()


def test_pick_plateau_detects_first_drop():
    eff = pd.Series([0.5, 0.7, 0.71, 0.69, 0.65], index=[1, 2, 3, 4, 5])
    plateau = pick_plateau(eff)
    # First negative diff occurs at horizon 4 (0.69 < 0.71)
    assert plateau == 4


def test_analyze_timeframe_outputs_expected_keys():
    close = _make_ar_series()
    stats = analyze_timeframe(close, max_forward=12)
    assert set(stats.keys()) == {"plateau_forward", "efficiency_max", "efficiency_len"}
    assert stats["plateau_forward"] >= 1
    assert stats["efficiency_len"] >= stats["plateau_forward"]

