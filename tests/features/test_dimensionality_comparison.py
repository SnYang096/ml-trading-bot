"""Test cases for config-driven dimensionality comparison.

Note: PROJECT_ROOT setup is handled in tests/conftest.py.
If you've installed the project via `pip install -e .`, you don't need sys.path setup.
"""

import tempfile
import shutil
from pathlib import Path
import textwrap

import pytest
import pandas as pd
import numpy as np

# PROJECT_ROOT is available from conftest.py, but we need it here for file paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.time_series_model.pipeline.dimensionality.dimensionality_comparison import (
    run_dim_compare,
    sanitize_features,
    train_lightgbm_model_simple,
    evaluate_model_simple,
)


def _write_yaml(path: Path, content: str) -> None:
    """Helper to write YAML file."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture
def tmp_strategy_config_dir(tmp_path):
    """Create a temporary strategy config directory for testing."""
    config_dir = tmp_path / "test_strategy"
    config_dir.mkdir()

    # Create features.yaml
    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            - atr
            - rsi
            - macd
          post_processors: []
        """,
    )

    # Create labels.yaml
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: future_return
        generator:
          module: tests.sample_module
          function: fake_label
          params:
            horizon: 24
        filters: []
        post_label_filters: []
        """,
    )

    # Create model.yaml (optional, but may be needed)
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    # Create evaluation.yaml (optional)
    _write_yaml(
        config_dir / "evaluation.yaml",
        """
        evaluation:
          metrics:
            - name: correlation
              type: correlation
        """,
    )

    return config_dir


@pytest.fixture
def sample_market_data(tmp_path):
    """Create sample market data in parquet format for testing."""
    data_dir = tmp_path / "data" / "parquet_data"
    data_dir.mkdir(parents=True)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000  # Enough for testing
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="15min")

    # Generate price data
    price_base = 50000
    returns = np.random.randn(n_samples) * 0.005
    prices = price_base * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
            "cvd": np.random.randn(n_samples).cumsum() * 1000,
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, n_samples),
            "_symbol": "BTCUSDT",
        },
        index=dates,
    )

    # Save to parquet
    symbol_dir = data_dir / "BTCUSDT"
    symbol_dir.mkdir()

    # Save as parquet file
    parquet_file = symbol_dir / "15T.parquet"
    df.to_parquet(parquet_file)

    return str(data_dir)


def test_sanitize_features():
    """Test sanitize_features function."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    X_sanitized = sanitize_features(X, clip_std=5.0)

    assert X_sanitized.shape == X.shape
    assert np.isfinite(X_sanitized).all()
    assert not np.isnan(X_sanitized).any()


def test_train_and_evaluate_model_simple():
    """Test train_lightgbm_model_simple and evaluate_model_simple functions."""
    np.random.seed(42)

    n_samples = 100
    n_features = 10

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples)
    X_val = np.random.randn(30, n_features)
    y_val = np.random.randn(30)
    X_test = np.random.randn(20, n_features)
    y_test = np.random.randn(20)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Train model
    model = train_lightgbm_model_simple(X_train, y_train, X_val, y_val, feature_names)

    assert model is not None

    # Evaluate model
    metrics = evaluate_model_simple(model, X_test, y_test)

    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert isinstance(metrics["r2"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["mae"], float)


def test_run_dim_compare_basic(tmp_strategy_config_dir, sample_market_data, tmp_path):
    """Test run_dim_compare with basic configuration."""
    # This is a more comprehensive test that may require actual data loading
    # For now, we'll skip if dependencies are not available
    try:
        from src.data_tools.data_utils import load_raw_data
        from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
    except ImportError:
        pytest.skip("Required dependencies not available for full test")

    # Test with minimal data requirements
    results, top_factors_path = run_dim_compare(
        config_dir=tmp_strategy_config_dir,
        symbol="BTCUSDT",
        data_path=sample_market_data,
        timeframe="15T",
        train_start=None,
        train_end=None,
    )

    # Verify results structure
    assert isinstance(results, dict)
    assert "strategy" in results
    assert "symbol" in results
    assert "data_info" in results
    assert "performance" in results
    assert "top_factors_path" in results

    # Verify top_factors_path exists
    top_factors_file = Path(top_factors_path)
    assert top_factors_file.exists()

    # Verify top_factors.json structure
    import json

    with open(top_factors_file, "r") as f:
        top_factors_data = json.load(f)

    assert "top_factors" in top_factors_data
    assert "count" in top_factors_data
    assert isinstance(top_factors_data["top_factors"], list)
    assert top_factors_data["count"] >= 0


def test_run_dim_compare_with_dates(tmp_strategy_config_dir, sample_market_data):
    """Test run_dim_compare with date range specified."""
    try:
        from src.data_tools.data_utils import load_raw_data
        from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
    except ImportError:
        pytest.skip("Required dependencies not available for full test")

    # Skip if required tick parquet data is not available
    tick_dir = Path(sample_market_data).parent / "parquet_data"
    if not tick_dir.exists() or not any(tick_dir.glob("*.parquet")):
        pytest.skip("Tick parquet data not available; skip run_dim_compare_with_dates.")

    results, top_factors_path = run_dim_compare(
        config_dir=tmp_strategy_config_dir,
        symbol="BTCUSDT",
        data_path=sample_market_data,
        timeframe="15T",
        train_start="2024-01-01",
        train_end="2024-12-31",
    )

    assert isinstance(results, dict)
    assert results["symbol"] == "BTCUSDT"
    assert Path(top_factors_path).exists()


@pytest.mark.skip(reason="Requires full data pipeline - run manually if needed")
def test_run_dim_compare_integration(tmp_strategy_config_dir):
    """Full integration test with real config structure.

    This test requires:
    - Actual strategy config directory
    - Real market data
    - All feature dependencies
    """
    # Use actual config if available
    actual_config_dir = PROJECT_ROOT / "config" / "strategies" / "sr_reversal"
    if not actual_config_dir.exists():
        pytest.skip("Actual strategy config not available")

    results, top_factors_path = run_dim_compare(
        config_dir=actual_config_dir,
        symbol="BTCUSDT",
        data_path=str(PROJECT_ROOT / "data" / "parquet_data"),
        timeframe="15T",
        train_start="2024-01-01",
        train_end="2024-12-31",
    )

    assert isinstance(results, dict)
    assert "data_info" in results
    assert "performance" in results
    assert Path(top_factors_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
