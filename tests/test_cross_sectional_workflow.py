import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import cross_sectional.workflow as cs_workflow


@contextmanager
def _argv(args):
    original = sys.argv[:]
    sys.argv = [original[0], *args]
    try:
        yield
    finally:
        sys.argv = original


@pytest.fixture
def synthetic_panel():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
    symbols = ["BTCUSDT", "ETHUSDT"]
    index = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])

    feature_a = np.linspace(0.1, 0.9, len(index))
    feature_b = np.linspace(1.5, 2.1, len(index))
    close = np.linspace(100, 105, len(index))
    volume = np.linspace(1_000_000, 1_500_000, len(index))
    future_return = np.linspace(0.01, 0.015, len(index))

    panel = pd.DataFrame(
        {
            "close": close,
            "volume": volume,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "future_return_12": future_return,
        },
        index=index,
    )
    return panel


def test_cross_sectional_workflow_main(monkeypatch, synthetic_panel, tmp_path):
    def fake_generate_panel(config):
        return synthetic_panel, "future_return_12"

    monkeypatch.setattr(cs_workflow, "generate_cross_sectional_panel", fake_generate_panel)

    args = [
        "--data-dir",
        "unused",
        "--symbols",
        "BTCUSDT,ETHUSDT",
        "--timeframe",
        "15T",
        "--horizon",
        "12",
        "--feature-type",
        "baseline",
        "--save-dir",
        str(tmp_path),
        "--winsorize-sigma",
        "2.5",
        "--zscore-clip",
        "2.5",
        "--liq-quantile",
        "0.2",
        "--de-corr-threshold",
        "0.95",
        "--long-only",
        "--regime-overlay",
    ]

    with _argv(args):
        cs_workflow.main()

    weights_path = tmp_path / "weights.parquet"
    assert weights_path.exists()
    weights = pd.read_parquet(weights_path)
    assert "weight" in weights.columns
    # Long-only overlay should keep weights non-negative and sum close to 1
    assert (weights["weight"] >= -1e-6).all()
    assert abs(weights["weight"].sum() - 1.0) < 1e-6


def test_overlay_regime_weights_scales_and_normalizes():
    weights = pd.Series({"BTC": 0.25, "ETH": -0.25})
    scaled = cs_workflow.overlay_regime_weights(weights, regime_state="TRENDING", trend_gain=1.5)
    # Gross leverage preserved
    assert pytest.approx(abs(scaled).sum()) == abs(weights).sum()
    # Signs should be preserved after overlay
    assert np.all(np.sign(scaled.values) == np.sign(weights.values))


def test_filter_by_liquidity_keeps_top_quantile():
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    symbols = ["BTC", "ETH", "SOL"]
    idx = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])
    values = np.arange(len(idx), dtype=float)
    panel = pd.DataFrame(
        {
            "factor": values,
            "dollar_volume": values / values.max(),
        },
        index=idx,
    )
    filtered = cs_workflow.filter_by_liquidity(panel, liq_col="dollar_volume", min_quantile=0.6)
    assert len(filtered) <= len(panel)
    grouped_threshold = panel.groupby(level=0)["dollar_volume"].quantile(0.6)
    matched_thresholds = grouped_threshold.reindex(filtered.index.get_level_values(0)).values
    assert (filtered["dollar_volume"].values >= matched_thresholds).all()

