"""Test script to run the ML trading strategy with real data."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np

# Import with full paths to avoid import issues
from src.ml_trading.strategies.ml_strategy import MLTradingStrategy
from src.ml_trading.data.data_loader import MarketDataLoader
from src.ml_trading.data.feature_engineering import FeatureEngineer


def main():
    """Main function to test the strategy with real data."""
    print("🚀 Testing ML Trading Strategy with Real Data")
    print("=" * 50)

    # Specify the data file path
    data_file = "/home/yin/trading/rlbot/data/agg_data/BTCUSDT-aggTrades-2025-05-01.csv"

    # Check if file exists
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    print(f"✅ Data file found: {data_file}")

    # Initialize components
    print("\n1. Initializing components...")
    data_loader = MarketDataLoader(data_path=data_file)
    feature_engineer = FeatureEngineer()

    # Load and prepare data
    print("\n2. Loading market data...")
    try:
        raw_data = data_loader.load_data()
        print(f"   ✓ Loaded {len(raw_data)} data points")
        print(f"   ✓ Data range: {raw_data.index[0]} to {raw_data.index[-1]}")
        print(
            f"   ✓ Price range: {raw_data['close'].min():.2f} to {raw_data['close'].max():.2f}"
        )
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return

    # Show data sample
    print("\n3. Data sample:")
    print(raw_data.head())

    # Multi-timeframe analysis
    print("\n4. Resampling to multiple timeframes...")
    try:
        multi_tf_data = data_loader.get_multi_timeframe_data()
        print(f"   ✓ Created data for timeframes: {list(multi_tf_data.keys())}")
        for tf, data in multi_tf_data.items():
            print(f"     - {tf}: {len(data)} bars")
    except Exception as e:
        print(f"   ✗ Error resampling data: {e}")
        return

    # Feature engineering
    print("\n5. Engineering features...")
    try:
        engineered_data = feature_engineer.engineer_features(multi_tf_data)
        print(f"   ✓ Engineered features for all timeframes")
        for tf, data in engineered_data.items():
            print(f"     - {tf}: {data.shape[1]} features, {len(data)} rows")
    except Exception as e:
        print(f"   ✗ Error engineering features: {e}")
        return

    # Initialize and train strategy
    print("\n6. Training ML strategy...")
    try:
        strategy = MLTradingStrategy()
        metrics = strategy.train_strategy()
        print("   ✓ Strategy training completed")

        # Print training metrics
        print("\n   Training Metrics:")
        for stage, stage_metrics in metrics.items():
            print(f"     {stage.upper()}:")
            for timeframe, metrics in stage_metrics.items():
                print(f"       {timeframe}: {metrics}")
    except Exception as e:
        print(f"   ✗ Error training strategy: {e}")
        return

    # Generate signals (just for 5T timeframe to keep it simple)
    print("\n7. Generating trading signals (5T timeframe)...")
    try:
        # Get 5T data
        data_5t = engineered_data["5T"]

        # Check if we have enough data
        if len(data_5t) < 10:
            print("   ✗ Not enough data for signal generation")
            return

        # Generate stage 1 predictions
        stage1_model = strategy.pipeline.stage1_models["5T"]
        feature_columns = [
            col
            for col in data_5t.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X_5t = data_5t[feature_columns]

        # Remove rows with any NaN values (LightGBM requirement)
        X_5t_clean = X_5t.dropna()
        if X_5t_clean.empty:
            print("   ✗ No valid data for prediction after cleaning")
            return

        stage1_preds = stage1_model.predict(X_5t_clean)

        # Generate stage 2 predictions
        stage2_model = strategy.pipeline.stage2_models["5T"]
        stage2_preds = stage2_model.predict(X_5t_clean)

        print(f"   ✓ Generated {len(stage1_preds)} stage 1 predictions")
        print(f"   ✓ Generated {len(stage2_preds)} stage 2 predictions")
        print(
            f"   ✓ Stage 1 prediction range: {stage1_preds.min():.4f} to {stage1_preds.max():.4f}"
        )
        print(
            f"   ✓ Stage 2 prediction range: {stage2_preds.min():.6f} to {stage2_preds.max():.6f}"
        )

        # Create simple signals
        signals = pd.DataFrame(
            {
                "timestamp": X_5t_clean.index,
                "stage1_pred": stage1_preds,
                "stage2_pred": stage2_preds,
                "discrete_signal": 0,
            }
        )

        # Convert continuous signal to discrete (-1, 0, 1)
        signals.loc[stage1_preds > 0.6, "discrete_signal"] = 1  # Long
        signals.loc[stage1_preds < 0.4, "discrete_signal"] = -1  # Short

        print(f"\n   Signal distribution:")
        print(f"     Long signals (1): {len(signals[signals['discrete_signal'] == 1])}")
        print(
            f"     Short signals (-1): {len(signals[signals['discrete_signal'] == -1])}"
        )
        print(f"     Hold signals (0): {len(signals[signals['discrete_signal'] == 0])}")

        # Save signals to CSV for backtesting
        signals.to_csv("backtest_signals.csv", index=False)
        print(f"\n   ✓ Saved signals to backtest_signals.csv")

        # Show sample signals
        print(f"\n   Sample signals:")
        print(signals.head(10))

    except Exception as e:
        print(f"   ✗ Error generating signals: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n🎉 Test completed successfully!")
    print("\nNext steps:")
    print("1. Check backtest_signals.csv for the generated signals")
    print("2. Use these signals for backtesting with Nautilus Trader")
    print("3. Adjust the signal thresholds in the code as needed")


if __name__ == "__main__":
    main()
