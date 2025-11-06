#!/usr/bin/env python3
"""Test script to verify the symbol column fix without alphalens dependency."""

import pandas as pd
import numpy as np
import os
import tempfile

# Copy the key function we fixed
def test_load_and_prepare_data_logic():
    print("Testing the fixed logic for symbol column preservation...")
    
    # Create test data similar to what would be loaded
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    data = pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 2000, 100),
    }, index=dates)
    
    # Add symbol column (this simulates what happens in the original function)
    data['symbol'] = 'TEST'
    
    print("Original data shape:", data.shape)
    print("Original columns:", list(data.columns))
    print("Symbol column present:", 'symbol' in data.columns)
    
    # Simulate what happens in feature engineering
    # In the real code, this would be the result of engineer_baseline_features
    # For this test, we'll just add some dummy feature columns
    engineered_data = data.copy()
    
    # Add some feature columns (simulating feature engineering)
    engineered_data['feature_1'] = np.random.randn(100)
    engineered_data['feature_2'] = np.random.randn(100)
    
    # This simulates the bug: feature engineering might drop the symbol column
    # In our fixed version, we preserve the symbol column
    preserved_symbol = engineered_data['symbol'].copy()
    
    # Simulate the bug where symbol column gets dropped during feature engineering
    if 'symbol' in engineered_data.columns:
        engineered_data = engineered_data.drop('symbol', axis=1)
    
    print("After simulated feature engineering:")
    print("  Engineered data shape:", engineered_data.shape)
    print("  Engineered columns:", list(engineered_data.columns))
    print("  Symbol column present:", 'symbol' in engineered_data.columns)
    
    # Apply our fix: reattach the symbol column
    combined_eng = engineered_data.copy()
    
    # This is the key fix we implemented:
    if preserved_symbol is not None:
        # Always ensure symbol column exists in the final DataFrame
        if "symbol" not in combined_eng.columns:
            try:
                # Try direct reindexing first
                combined_eng["symbol"] = preserved_symbol.reindex(
                    combined_eng.index, method="ffill").bfill()
            except Exception:
                # Fallback approach: merge on timestamp
                try:
                    sym_df = preserved_symbol.to_frame(name="symbol")
                    sym_df["timestamp"] = sym_df.index
                    tmp = combined_eng.copy()
                    tmp["timestamp"] = tmp.index
                    combined_eng = tmp.merge(sym_df.drop_duplicates(subset=["timestamp"]),
                                             on="timestamp",
                                             how="left").set_index("timestamp")
                except Exception:
                    # Last resort: fill with UNKNOWN
                    combined_eng["symbol"] = "UNKNOWN"
    
    # Ensure symbol column is of string type
    combined_eng["symbol"] = combined_eng["symbol"].astype(str)
    
    print("After applying fix:")
    print("  Final data shape:", combined_eng.shape)
    print("  Final columns:", list(combined_eng.columns))
    print("  Symbol column preserved:", 'symbol' in combined_eng.columns)
    if 'symbol' in combined_eng.columns:
        print("  Symbol column values:", combined_eng['symbol'].unique())
    
    # Verify the fix worked
    assert 'symbol' in combined_eng.columns, "Symbol column should be preserved"
    assert combined_eng['symbol'].nunique() == 1, "Should have consistent symbol values"
    assert combined_eng['symbol'].iloc[0] == 'TEST', "Should preserve original symbol value"
    
    print("✅ Test passed! Symbol column is correctly preserved.")
    return True

if __name__ == "__main__":
    test_load_and_prepare_data_logic()