"""Identify what Column_245 actually represents by regenerating features."""

import os
import sys
import pandas as pd
import numpy as np

# Add common utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import (
    load_and_process_file,
    add_order_flow_features,
    engineer_features,
    get_feature_columns,
)


def main():
    """Identify Column_245 by reproducing the feature engineering."""

    # Load a small sample of data (just need to know feature names, not full data)
    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"
    sample_file = os.path.join(data_dir, "BTCUSDT-aggTrades-2024-10.zip")

    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        return

    print("🔍 Loading sample data to identify feature names...")
    print(f"   File: {sample_file}")

    # Load and process
    df = load_and_process_file(sample_file)
    if df is None:
        print("❌ Failed to load data")
        return

    # Take only first 1000 rows for speed
    df = df.head(1000)
    print(f"   ✓ Loaded {len(df)} rows")

    # Add order flow features
    print("\n📊 Adding order flow features...")
    try:
        df = add_order_flow_features(sample_file, df)
        print("   ✓ Order flow features added")
    except Exception as e:
        print(f"   ⚠️  Order flow failed: {e}")

    # Engineer features (this is where the magic happens)
    print("\n🔧 Engineering features...")
    print("   (This may take a minute...)")
    df, feature_engineer = engineer_features(df, None, fit=True)
    print("   ✓ Features engineered")

    # Get all feature columns (in the order they would be passed to the model)
    feature_cols = get_feature_columns(df)

    print(f"\n✅ Total features: {len(feature_cols)}")

    # Find Column_245
    if len(feature_cols) > 245:
        column_245_name = feature_cols[245]
        print(f"\n🎯 Column_245 corresponds to: {column_245_name}")
        print(f"\n" + "=" * 80)
        print(f"ANSWER: Column_245 = {column_245_name}")
        print("=" * 80)

        # Show some context
        print(f"\nNearby features:")
        for i in range(max(0, 243), min(len(feature_cols), 248)):
            print(f"   Column_{i}: {feature_cols[i]}")

        # Analyze the feature name
        print(f"\n📊 Feature Analysis:")
        fname = column_245_name.lower()

        if "wpt" in fname or "wavelet" in fname:
            print("   Type: Wavelet Packet Transform (WPT) feature")
            print("   Description: Frequency-domain decomposition of a signal")
        elif "hurst" in fname:
            print("   Type: Hurst Exponent feature")
            print(
                "   Description: Measures trend persistence (>0.5 trending, <0.5 mean-reverting)"
            )
        elif "hilbert" in fname:
            print("   Type: Hilbert Transform feature")
            print(
                "   Description: Analytic signal analysis (instantaneous amplitude/phase)"
            )
        elif "spectral" in fname:
            print("   Type: Spectral Analysis feature")
            print("   Description: Frequency spectrum characteristics")
        elif any(x in fname for x in ["cvd", "taker", "order_flow", "ofi"]):
            print("   Type: Order Flow feature")
            print("   Description: Market microstructure and order book dynamics")
        elif any(x in fname for x in ["rsi", "macd", "bb", "ema", "sma", "atr"]):
            print("   Type: Technical Indicator")
            print("   Description: Traditional technical analysis indicator")
        else:
            print("   Type: Derived/Advanced feature")
            print("   Description: Complex feature derived from multiple signals")

    else:
        print(f"❌ Only {len(feature_cols)} features found, Column_245 doesn't exist")
        print("\nAll feature names:")
        for i, col in enumerate(feature_cols):
            print(f"   Column_{i}: {col}")

    # Save feature mapping for future reference
    output_dir = "results/monthly_rolling_2025"
    os.makedirs(output_dir, exist_ok=True)

    feature_mapping = pd.DataFrame(
        {
            "column_index": [f"Column_{i}" for i in range(len(feature_cols))],
            "feature_name": feature_cols,
        }
    )

    mapping_path = os.path.join(output_dir, "feature_column_mapping.csv")
    feature_mapping.to_csv(mapping_path, index=False)
    print(f"\n💾 Feature mapping saved to: {mapping_path}")
    print("   (Use this to look up any Column_X in the future)")


if __name__ == "__main__":
    main()
