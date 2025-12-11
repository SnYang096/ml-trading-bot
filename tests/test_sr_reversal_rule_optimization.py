"""
Tests for SR Reversal Rule-Based Parameter Optimization
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.sr_reversal_rule_optimization import (
    define_parameter_grid,
    sample_random_params,
    evaluate_rule_strategy,
    find_plateau_regions,
)


def test_define_parameter_grid():
    """Test parameter grid definition"""
    grid = define_parameter_grid()

    assert isinstance(grid, dict)
    assert "sr_strength_min" in grid
    assert "sqs_min" in grid
    assert "stop_loss_r" in grid
    assert "take_profit_r" in grid
    assert "max_holding_bars" in grid
    assert "use_vpin_filter" in grid

    # Check that all values are lists
    for key, values in grid.items():
        assert isinstance(values, list), f"{key} should be a list"
        assert len(values) > 0, f"{key} should have at least one value"


def test_sample_random_params():
    """Test random parameter sampling"""
    grid = define_parameter_grid()
    n_trials = 10

    samples = sample_random_params(grid, n_trials)

    assert len(samples) == n_trials

    # Check that each sample has required keys
    for sample in samples:
        assert "sr_strength_min" in sample
        assert "sqs_min" in sample
        assert "stop_loss_r" in sample
        assert "take_profit_r" in sample
        assert "max_holding_bars" in sample
        assert "use_vpin_filter" in sample

        # Check VPIN parameters
        if sample["use_vpin_filter"]:
            assert "min_vpin" in sample
            assert "max_vpin" in sample
            assert sample["min_vpin"] is not None
            assert sample["max_vpin"] is not None
            assert sample["min_vpin"] < sample["max_vpin"]
        else:
            assert sample.get("min_vpin") is None or sample["min_vpin"] is None
            assert sample.get("max_vpin") is None or sample["max_vpin"] is None


def test_evaluate_rule_strategy_basic():
    """Test basic rule strategy evaluation"""
    # Create mock data
    n_bars = 100
    df_features = pd.DataFrame(
        {
            "open": np.random.randn(n_bars).cumsum() + 100,
            "high": np.random.randn(n_bars).cumsum() + 101,
            "low": np.random.randn(n_bars).cumsum() + 99,
            "close": np.random.randn(n_bars).cumsum() + 100,
            "volume": np.random.rand(n_bars) * 1000,
            "sr_strength_max": np.random.rand(n_bars),
            "sqs_hal_high": np.random.rand(n_bars),
            "sqs_hal_low": np.random.rand(n_bars),
        }
    )

    # Calculate ATR
    high_low = df_features["high"] - df_features["low"]
    high_close = np.abs(df_features["high"] - df_features["close"].shift(1))
    low_close = np.abs(df_features["low"] - df_features["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = true_range.rolling(window=14, min_periods=1).mean()
    df_features["atr"] = atr_series

    # Test parameters
    params = {
        "sr_strength_min": 0.5,
        "sqs_min": 0.5,
        "touch_distance_atr": 1.0,
        "stop_loss_r": 1.0,
        "take_profit_r": 2.0,
        "max_holding_bars": 24,
        "use_vpin_filter": False,
        "min_vpin": None,
        "max_vpin": None,
    }

    # Evaluate strategy
    result = evaluate_rule_strategy(df_features, atr_series, params)

    # Check result structure
    assert isinstance(result, dict)
    assert "n_signals" in result
    assert "n_trades" in result
    assert "win_rate" in result
    assert "total_r" in result
    assert "avg_r" in result
    assert "sharpe_ratio" in result
    assert "profit_factor" in result
    assert "max_drawdown" in result

    # Check value types
    assert isinstance(result["n_signals"], (int, np.integer))
    assert isinstance(result["n_trades"], (int, np.integer))
    assert isinstance(result["win_rate"], (float, np.floating))
    assert isinstance(result["total_r"], (float, np.floating))
    assert isinstance(result["avg_r"], (float, np.floating))
    assert isinstance(result["sharpe_ratio"], (float, np.floating))
    assert isinstance(result["profit_factor"], (float, np.floating))
    assert isinstance(result["max_drawdown"], (float, np.floating))

    # Check value ranges
    assert result["n_signals"] >= 0
    assert result["n_trades"] >= 0
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["sharpe_ratio"] >= 0.0 or np.isnan(result["sharpe_ratio"])


def test_evaluate_rule_strategy_no_trades():
    """Test evaluation when no trades are generated"""
    # Create mock data with no signals
    n_bars = 100
    df_features = pd.DataFrame(
        {
            "open": np.random.randn(n_bars).cumsum() + 100,
            "high": np.random.randn(n_bars).cumsum() + 101,
            "low": np.random.randn(n_bars).cumsum() + 99,
            "close": np.random.randn(n_bars).cumsum() + 100,
            "volume": np.random.rand(n_bars) * 1000,
            "sr_strength_max": np.zeros(n_bars),  # No SR strength
            "sqs_hal_high": np.zeros(n_bars),
            "sqs_hal_low": np.zeros(n_bars),
        }
    )

    atr_series = pd.Series(np.ones(n_bars) * 1.0)
    df_features["atr"] = atr_series

    params = {
        "sr_strength_min": 0.9,  # Very high threshold
        "sqs_min": 0.9,
        "touch_distance_atr": 1.0,
        "stop_loss_r": 1.0,
        "take_profit_r": 2.0,
        "max_holding_bars": 24,
        "use_vpin_filter": False,
        "min_vpin": None,
        "max_vpin": None,
    }

    result = evaluate_rule_strategy(df_features, atr_series, params)

    # Should return zero trades
    assert result["n_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_r"] == 0.0
    assert result["avg_r"] == 0.0


def test_find_plateau_regions():
    """Test plateau region finding"""
    # Create mock results
    results_df = pd.DataFrame(
        {
            "sr_strength_min": [0.3, 0.4, 0.5, 0.6, 0.7] * 4,
            "sqs_min": [0.3] * 5 + [0.4] * 5 + [0.5] * 5 + [0.6] * 5,
            "stop_loss_r": [1.0] * 20,
            "take_profit_r": [2.0] * 20,
            "total_r": [
                10,
                20,
                30,
                40,
                50,
                15,
                25,
                35,
                45,
                55,
                5,
                15,
                25,
                35,
                45,
                0,
                10,
                20,
                30,
                40,
            ],
            "win_rate": [0.3, 0.4, 0.5, 0.6, 0.7] * 4,
            "sharpe_ratio": [0.5, 1.0, 1.5, 2.0, 2.5] * 4,
        }
    )

    plateau_df = find_plateau_regions(
        results_df, metric_col="total_r", threshold_percentile=0.8
    )

    assert isinstance(plateau_df, pd.DataFrame)
    assert len(plateau_df) > 0
    assert "parameter" in plateau_df.columns
    assert "most_common_value" in plateau_df.columns
    assert "frequency_in_high_performance" in plateau_df.columns
    assert "n_occurrences" in plateau_df.columns

    # Check that frequencies are valid
    for _, row in plateau_df.iterrows():
        assert 0.0 <= row["frequency_in_high_performance"] <= 1.0
        assert row["n_occurrences"] > 0


def test_find_plateau_regions_empty():
    """Test plateau finding with empty results"""
    results_df = pd.DataFrame(
        {
            "sr_strength_min": [],
            "total_r": [],
        }
    )

    plateau_df = find_plateau_regions(results_df, metric_col="total_r")

    assert isinstance(plateau_df, pd.DataFrame)
    assert len(plateau_df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
