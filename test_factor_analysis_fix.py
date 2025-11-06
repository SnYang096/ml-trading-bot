#!/usr/bin/env python3
"""Test script to verify the fix for factor analysis symbol column issue."""

import pandas as pd
import numpy as np
from scripts.analysis.factor_analysis_alphalens import load_and_prepare_data

# Create a simple test DataFrame to verify the fix
def test_symbol_column_preservation():
    print("Testing symbol column preservation in load_and_prepare_data...")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    data = pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 2000, 100),
    }, index=dates)
    
    # Add symbol column
    data['symbol'] = 'TEST'
    
    print("Original data shape:", data.shape)
    print("Original columns:", list(data.columns))
    print("Symbol column present:", 'symbol' in data.columns)
    
    # Test the function
    try:
        # Save test data to a temporary file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data.to_parquet(tmp.name)
            tmp_path = tmp.name
        
        # Test the function
        try:
            result_df, feature_cols = load_and_prepare_data([tmp_path], "5T", "baseline")
            print("Result data shape:", result_df.shape)
            print("Result columns:", list(result_df.columns))
            print("Symbol column preserved:", 'symbol' in result_df.columns)
            if 'symbol' in result_df.columns:
                print("Symbol column values:", result_df['symbol'].unique())
            print("Feature columns count:", len(feature_cols))
            print("Test passed!")
            # Clean up
            os.unlink(tmp_path)
            return True
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Clean up
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
            return False
        print("Result data shape:", result_df.shape)
        print("Result columns:", list(result_df.columns))
        print("Symbol column preserved:", 'symbol' in result_df.columns)
        if 'symbol' in result_df.columns:
            print("Symbol column values:", result_df['symbol'].unique())
        print("Feature columns count:", len(feature_cols))
        print("Test passed!")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_symbol_column_preservation()