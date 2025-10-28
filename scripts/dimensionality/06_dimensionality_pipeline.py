#!/usr/bin/env python3
"""
Advanced Dimensionality Reduction Pipeline Script
Implements Autoencoder + SHAP distillation for interpretable factor compression.

This script provides a complete pipeline for:
1. Loading and preprocessing factor data
2. Training Autoencoder for nonlinear compression
3. SHAP-based factor contribution distillation
4. Generating interpretable trading signals
5. Visualization and monitoring
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set matplotlib backend to avoid display issues
import matplotlib

matplotlib.use("Agg")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.models.interpretable_factor_engine import (
    InterpretableFactorEngine,
    create_sample_data,
)
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
)


def load_real_data(data_path: str, symbol: str = "ETH-USD") -> tuple:
    """
    Load real trading data and extract features.

    Args:
        data_path: Path to data directory
        symbol: Trading symbol

    Returns:
        X, y, factor_names
    """
    print(f"📂 Loading real data for {symbol}")

    try:
        # Initialize data loader
        loader = MarketDataLoader(data_path)

        # Load data
        df = loader.load_data()
        if df is not None and not df.empty:
            df = loader.resample_data("5T")

        if df is None or df.empty:
            print(f"❌ No data found for {symbol}. Using sample data instead.")
            return create_sample_data()

        print(f"✅ Loaded {len(df)} samples")

        # Initialize feature engineering
        comprehensive_engineer = ComprehensiveFeatureEngineer()

        # Generate features
        df_features = comprehensive_engineer.engineer_all_features(df, fit=True)

        # Select numeric columns as features
        feature_columns = df_features.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # Remove target columns and metadata
        exclude_columns = ["timestamp", "future_return", "signal", "expected_return"]
        feature_columns = [col for col in feature_columns if col not in exclude_columns]

        print(f"📊 Generated {len(feature_columns)} features")

        # Prepare data
        X = df_features[feature_columns].values

        # Create target (future return)
        if "future_return" in df_features.columns:
            y = df_features["future_return"].values
        else:
            # Create synthetic target based on price movement
            y = df_features["close"].pct_change(5).shift(-5).fillna(0).values

        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y, feature_columns

    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("📊 Using sample data instead")
        return create_sample_data()


def run_dimensionality_reduction_pipeline(args):
    """
    Run the complete dimensionality reduction pipeline.

    Args:
        args: Command line arguments
    """
    print("🚀 Starting Advanced Dimensionality Reduction Pipeline")
    print("=" * 70)
    print("Method: Autoencoder + SHAP Distillation")
    print("Purpose: Interpretable factor compression for trading signals")
    print("=" * 70)

    # Load data
    if args.use_real_data and args.data_path:
        X, y, factor_names = load_real_data(args.data_path, args.symbol)
    else:
        print("📊 Using sample data for demonstration")
        X, y, factor_names = create_sample_data(
            n_samples=args.n_samples, n_factors=args.n_factors
        )

    # Initialize engine
    engine = InterpretableFactorEngine(
        encoding_dim=args.encoding_dim,
        autoencoder_lr=args.learning_rate,
        autoencoder_epochs=args.epochs,
        dropout_rate=args.dropout_rate,
    )

    # Train pipeline
    print(f"\n🎯 Training on {X.shape[0]} samples with {X.shape[1]} factors")
    engine.fit(X, y, factor_names, top_k=args.top_k)

    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = engine.predict(X)

    # Generate interpretable signals
    signals = engine.get_interpretable_signal(X)

    print(f"📊 Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print(f"📈 Signals: mean={signals.mean():.4f}, std={signals.std():.4f}")

    # Save model if requested
    if args.save_model:
        model_path = (
            f"models/interpretable_factor_engine_{args.symbol.replace('-', '_')}.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        engine.save_model(model_path)
        print(f"💾 Model saved to: {model_path}")

    # Create visualizations
    if args.visualize:
        print(f"\n📊 Creating visualizations...")
        viz_path = f"reports/factor_contributions_{args.symbol.replace('-', '_')}.png"
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        engine.visualize_factor_contributions(save_path=viz_path, top_k=args.top_k)

    # Generate report
    if args.generate_report:
        generate_pipeline_report(engine, args, predictions, signals)

    print("\n" + "=" * 70)
    print("🎉 Dimensionality Reduction Pipeline Complete!")
    print("=" * 70)

    # Summary
    print(
        f"📊 Compression: {len(factor_names)} factors → {args.encoding_dim} dimensions"
    )
    print(f"🎯 Top factors: {len(engine.top_factors)} selected")
    print(f"🔍 Interpretability: SHAP distillation complete")

    if engine.top_factors is not None:
        print(f"\n🏆 Top 5 Driving Factors:")
        for i, (factor, weight) in enumerate(
            zip(engine.top_factors[:5], engine.factor_weights[:5])
        ):
            print(f"  {i+1}. {factor}: {weight:.3f}")


def generate_pipeline_report(engine, args, predictions, signals):
    """
    Generate a comprehensive pipeline report.

    Args:
        engine: Trained InterpretableFactorEngine
        args: Command line arguments
        predictions: Model predictions
        signals: Interpretable signals
    """
    print(f"\n📋 Generating pipeline report...")

    report_path = (
        f"reports/dimensionality_reduction_report_{args.symbol.replace('-', '_')}.txt"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("Advanced Dimensionality Reduction Pipeline Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Symbol: {args.symbol}\n")
        f.write(f"  Encoding Dimension: {args.encoding_dim}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Dropout Rate: {args.dropout_rate}\n")
        f.write(f"  Top K Factors: {args.top_k}\n\n")

        f.write(f"Results:\n")
        f.write(f"  Total Factors: {len(engine.factor_names)}\n")
        f.write(f"  Compressed Dimensions: {args.encoding_dim}\n")
        f.write(
            f"  Compression Ratio: {len(engine.factor_names) / args.encoding_dim:.1f}x\n"
        )
        f.write(f"  Selected Factors: {len(engine.top_factors)}\n\n")

        f.write(f"Performance:\n")
        f.write(f"  Predictions Mean: {predictions.mean():.4f}\n")
        f.write(f"  Predictions Std: {predictions.std():.4f}\n")
        f.write(f"  Signals Mean: {signals.mean():.4f}\n")
        f.write(f"  Signals Std: {signals.std():.4f}\n\n")

        f.write(f"Top Factors and Weights:\n")
        for i, (factor, weight) in enumerate(
            zip(engine.top_factors, engine.factor_weights)
        ):
            f.write(f"  {i+1:2d}. {factor:<30} {weight:.3f}\n")

    print(f"📋 Report saved to: {report_path}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced Dimensionality Reduction Pipeline using Autoencoder + SHAP Distillation"
    )

    # Data parameters
    parser.add_argument(
        "--data-path", type=str, default="data/agg_data", help="Path to data directory"
    )
    parser.add_argument("--symbol", type=str, default="ETH-USD", help="Trading symbol")
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real trading data instead of sample data",
    )

    # Model parameters
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=8,
        help="Dimension of compressed factor space",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for autoencoder",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for regularization",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top factors to select"
    )

    # Sample data parameters
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples for synthetic data",
    )
    parser.add_argument(
        "--n-factors", type=int, default=60, help="Number of factors for synthetic data"
    )

    # Output options
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate detailed report"
    )

    args = parser.parse_args()

    # Run pipeline
    run_dimensionality_reduction_pipeline(args)


if __name__ == "__main__":
    main()
