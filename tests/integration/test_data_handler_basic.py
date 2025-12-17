"""
Basic integration test for DataHandler.

This test verifies that DataHandler provides a consistent interface
for loading OHLCV data, matching the behavior of the original load_raw_data.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_tools.data_handler import DataHandler
from src.data_tools.data_utils import load_raw_data

# Use real data path
REAL_DATA_PATH = "data/parquet_data"


@pytest.mark.integration
def test_data_handler_load_ohlcv_matches_load_raw_data(tmp_path):
    """
    Test that DataHandler.load_ohlcv produces the same output as load_raw_data.

    This ensures backward compatibility during migration.
    """
    sample_data_dir = REAL_DATA_PATH
    if not Path(sample_data_dir).exists():
        pytest.skip(f"Real data path not found: {sample_data_dir}")

    symbol = "BTCUSDT"
    timeframe = "240T"  # Use 4H for faster test

    # Load using original function
    df_original = load_raw_data(
        data_path=sample_data_dir,
        symbol=symbol,
        timeframe=timeframe,
    )

    # Load using DataHandler
    handler = DataHandler(data_path=sample_data_dir)
    df_handler = handler.load_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
    )

    # Compare basic properties
    assert len(df_handler) > 0, "DataHandler should return non-empty DataFrame"
    assert len(df_handler) == len(df_original), "Row count should match"
    assert set(df_handler.columns) == set(df_original.columns), "Columns should match"

    # Compare required base columns
    base_cols = handler.get_base_schema()
    for col in base_cols:
        assert col in df_handler.columns, f"Base column {col} should be present"

    # Compare index alignment
    assert isinstance(
        df_handler.index, pd.DatetimeIndex
    ), "Index should be DatetimeIndex"
    assert df_handler.index.is_monotonic_increasing, "Index should be sorted"

    # Compare a few key columns
    for col in ["open", "high", "low", "close", "volume", "_symbol"]:
        if col in df_original.columns and col in df_handler.columns:
            pd.testing.assert_series_equal(
                df_original[col].sort_index(),
                df_handler[col].sort_index(),
                check_names=False,
                rtol=1e-5,
            )


@pytest.mark.integration
def test_data_handler_base_schema(tmp_path):
    """Test that DataHandler provides correct base schema."""
    handler = DataHandler(data_path="dummy")

    base_cols = handler.get_base_schema()
    assert "open" in base_cols
    assert "high" in base_cols
    assert "low" in base_cols
    assert "close" in base_cols
    assert "volume" in base_cols
    assert "_symbol" in base_cols

    orderflow_cols = handler.get_orderflow_base_schema()
    assert "buy_qty" in orderflow_cols
    assert "cvd" in orderflow_cols
    assert "taker_buy_ratio" in orderflow_cols


@pytest.mark.integration
def test_data_handler_ensure_base_columns():
    """Test that ensure_base_columns adds missing columns."""
    handler = DataHandler(data_path="dummy")

    # Create a minimal DataFrame
    df = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [99, 100, 101],
            "close": [104, 105, 106],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    df_ensured = handler.ensure_base_columns(df)

    # Check that base columns are present
    assert "volume" in df_ensured.columns
    assert "_symbol" in df_ensured.columns
    assert df_ensured["volume"].dtype in [float, int]
    assert df_ensured["_symbol"].dtype == object
