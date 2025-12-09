"""Example integration test showing how to use the integration test environment.

This is a simple example test that demonstrates how to use the fixtures
provided in conftest.py.
"""

import pytest
from pathlib import Path


def test_integration_env_setup(integration_env):
    """Test that integration environment is properly set up."""
    assert "data_dir" in integration_env
    assert "config_dir" in integration_env
    assert "symbol" in integration_env
    assert "timeframe" in integration_env

    # Verify data directory exists
    data_dir = Path(integration_env["data_dir"])
    assert data_dir.exists(), "Data directory should exist"

    # Verify config directory exists
    config_dir = Path(integration_env["config_dir"])
    assert config_dir.exists(), "Config directory should exist"
    assert (config_dir / "features.yaml").exists(), "features.yaml should exist"
    assert (config_dir / "labels.yaml").exists(), "labels.yaml should exist"

    print(f"\n✅ Integration environment verified:")
    print(f"   Data dir: {integration_env['data_dir']}")
    print(f"   Config dir: {integration_env['config_dir']}")
    print(f"   Symbol: {integration_env['symbol']}")
    print(f"   Timeframe: {integration_env['timeframe']}")


def test_data_files_exist(integration_env):
    """Test that data files are generated."""
    data_dir = Path(integration_env["data_dir"])
    symbol = integration_env["symbol"]

    # Check for MarketDataLoader format files
    loader_files = list(data_dir.glob(f"{symbol}_*.parquet"))
    assert len(loader_files) > 0, f"Should have at least one {symbol}_*.parquet file"

    # Check for nested format files
    nested_dir = data_dir / symbol
    if nested_dir.exists():
        nested_files = list(nested_dir.glob("*.parquet"))
        assert len(nested_files) > 0, f"Should have parquet files in {nested_dir}"

    print(f"\n✅ Data files verified:")
    print(f"   Loader format files: {len(loader_files)}")
    if nested_dir.exists():
        print(f"   Nested format files: {len(list(nested_dir.glob('*.parquet')))}")
