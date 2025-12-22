"""Pytest configuration and fixtures for integration tests."""

import sys
import tempfile
import shutil
from pathlib import Path
import textwrap
import json

import pytest
import pandas as pd
import numpy as np

# NOTE: Do not define `pytest_plugins` in non-top-level conftest files.
# Pytest treats it as global and errors (pytest>=8).

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark everything under tests/integration as `integration`.

    This keeps the workflow ergonomic:
    - Fast dev loop:  pytest -m "not integration"
    - Integration:    pytest -m integration
    """
    for item in items:
        item.add_marker(pytest.mark.integration)


def _write_yaml(path: Path, content: str) -> None:
    """Helper to write YAML file."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture(scope="session")
def integration_test_dir(tmp_path_factory):
    """Create a temporary directory for integration tests."""
    test_dir = tmp_path_factory.mktemp("integration_test")
    yield test_dir
    # Cleanup is handled by tmp_path_factory


@pytest.fixture(scope="session")
def integration_data_dir(integration_test_dir):
    """Create a data directory structure for integration tests."""
    data_dir = integration_test_dir / "data" / "parquet_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


@pytest.fixture(scope="session")
def integration_config_dir(integration_test_dir):
    """Create a strategy config directory for integration tests."""
    config_dir = integration_test_dir / "config" / "strategies" / "test_strategy"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture(scope="session")
def generate_market_data(integration_data_dir):
    """Generate realistic market data for integration tests."""

    def _generate_data(symbol: str, n_samples: int = 2000, timeframe: str = "15T"):
        """Generate market data and save to parquet format.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            n_samples: Number of samples to generate
            timeframe: Timeframe for data (e.g., "15T")

        Returns:
            Path to the generated parquet file
        """
        np.random.seed(42)

        # Generate realistic price data
        # Convert timeframe to pandas frequency (e.g., "15T" -> "15min")
        freq_map = {
            "15T": "15min",
            "5T": "5min",
            "60T": "60min",
            "240T": "240min",
            "1H": "1H",
        }
        pandas_freq = freq_map.get(timeframe, timeframe.replace("T", "min"))
        dates = pd.date_range("2024-01-01", periods=n_samples, freq=pandas_freq)
        price_base = 50000.0

        # Generate returns with some autocorrelation
        returns = np.random.randn(n_samples) * 0.005
        # Add some trend and mean reversion
        trend = np.linspace(0, 0.1, n_samples)
        returns = returns + trend / n_samples + np.random.randn(n_samples) * 0.002

        prices = price_base * (1 + returns).cumprod()

        # Generate OHLCV data
        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n_samples) * 0.0005),
                "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
                "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n_samples),
                "cvd": np.random.randn(n_samples).cumsum() * 1000,
                "taker_buy_ratio": np.random.uniform(0.3, 0.7, n_samples),
                "_symbol": symbol,
            },
            index=dates,
        )

        # Ensure OHLC relationships are correct
        df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
        df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999
        df["open"] = df["close"].shift(1).fillna(df["close"])

        # Save to parquet format
        # MarketDataLoader expects files like: {symbol}_*.parquet in the data root
        # Also support {symbol}/{timeframe}.parquet structure for load_raw_data
        data_path = Path(integration_data_dir)

        # Save in MarketDataLoader format: {symbol}_YYYYMMDD.parquet in root
        timestamp_str = df.index[0].strftime("%Y%m%d")
        loader_file = data_path / f"{symbol}_{timestamp_str}.parquet"
        df.to_parquet(loader_file)

        # Also save in nested format: {symbol}/{timeframe}.parquet (for compatibility)
        symbol_dir = data_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        nested_file = symbol_dir / f"{timeframe}.parquet"
        df.to_parquet(nested_file)

        print(f"✅ Generated {n_samples} samples for {symbol} at {timeframe}")
        print(f"   Saved to: {loader_file} (MarketDataLoader format)")
        print(f"   Saved to: {nested_file} (nested format)")

        return str(loader_file)

    return _generate_data


@pytest.fixture(scope="session")
def setup_strategy_config(integration_config_dir):
    """Setup a test strategy configuration."""

    # Create features.yaml
    _write_yaml(
        integration_config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            - atr
            - rsi
            - macd
            - sr_strength_max
            - sqs_hal_high
            - sqs_hal_low
          post_processors: []
        """,
    )

    # Create labels.yaml - use simple future return for integration tests
    _write_yaml(
        integration_config_dir / "labels.yaml",
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

    # Create model.yaml
    _write_yaml(
        integration_config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    # Create evaluation.yaml
    _write_yaml(
        integration_config_dir / "evaluation.yaml",
        """
        evaluation:
          metrics:
            - name: correlation
              type: correlation
        """,
    )

    print(f"✅ Strategy config created at: {integration_config_dir}")
    return integration_config_dir


@pytest.fixture(scope="session")
def integration_env(generate_market_data, setup_strategy_config, integration_data_dir):
    """Complete integration test environment.

    This fixture sets up:
    - Market data for testing
    - Strategy configuration
    - Returns a dict with all paths and configs
    """
    # Generate test data
    btc_data = generate_market_data("BTCUSDT", n_samples=2000, timeframe="15T")

    env = {
        "data_dir": integration_data_dir,
        "config_dir": setup_strategy_config,
        "symbol": "BTCUSDT",
        "timeframe": "15T",
        "btc_data_file": btc_data,
    }

    print("\n" + "=" * 70)
    print("Integration Test Environment Setup Complete")
    print("=" * 70)
    print(f"Data Directory: {integration_data_dir}")
    print(f"Config Directory: {setup_strategy_config}")
    print(f"Symbol: BTCUSDT")
    print(f"Timeframe: 15T")
    print("=" * 70 + "\n")

    return env
