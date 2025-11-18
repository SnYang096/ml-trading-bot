"""Export feature importance with actual feature names from rolling training models."""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

from data_tools.rolling_data import (
    load_and_process_file,
    add_order_flow_features,
    engineer_features,
    create_labels,
    get_feature_columns,
)


def get_feature_names_from_training_data():
    """Get actual feature names by recreating the training process."""
    data_dir = os.environ.get("DATA_DIR", "data/parquet_data")

    print("🔍 Recreating feature names from training data...")

    # Load 2024 Q4 data (same as training)
    train_files = []
    for month in [10, 11, 12]:
        file_path = os.path.join(data_dir, f"BTCUSDT-aggTrades-2024-{month:02d}.zip")
        if os.path.exists(file_path):
            train_files.append(file_path)

    if not train_files:
        print("❌ No training files found")
        return None

    # Load and process one month to get feature names
    print(f"   Loading {os.path.basename(train_files[0])}...")
    df = load_and_process_file(train_files[0])

    if df is None:
        print("❌ Failed to load training data")
        return None

    # Add order flow features
    try:
        df = add_order_flow_features(train_files[0], df)
        print("   ✓ Order flow features added")
    except Exception as e:
        print(f"   ⚠️  Order flow failed: {e}")

    # Engineer features
    print("   Engineering features...")
    df, _ = engineer_features(df, None, fit=True)

    # Create labels
    df = create_labels(df)
    df = df.dropna()

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"   ✓ Found {len(feature_cols)} features")

    return feature_cols


def load_model_and_get_importance_with_names(model_path, feature_names):
    """Load a LightGBM model and get feature importance with actual names."""
    import lightgbm as lgb

    # Load model
    model = lgb.Booster(model_file=model_path)

    # Get feature importance
    importance = model.feature_importance(importance_type="gain")
    model_feature_names = model.feature_name()

    # Map model feature names to actual feature names
    actual_names = []
    for model_name in model_feature_names:
        # Extract column index from "Column_XXX"
        try:
            col_idx = int(model_name.replace("Column_", ""))
            if col_idx < len(feature_names):
                actual_names.append(feature_names[col_idx])
            else:
                actual_names.append(f"Unknown_{col_idx}")
        except:
            actual_names.append(model_name)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "feature": actual_names,
            "model_feature": model_feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    return df


def main():
    results_dir = "results/monthly_rolling_2025"

    print("🔍 Extracting feature importance with actual names...")

    # Get actual feature names
    feature_names = get_feature_names_from_training_data()
    if feature_names is None:
        print("❌ Failed to get feature names")
        return

    print(f"✓ Feature names loaded: {len(feature_names)} features")

    all_importance = {}

    # Process each month's model
    for month in ["01", "02", "03", "04", "05", "06"]:
        model_path = os.path.join(results_dir, f"model_2025-{month}.txt")

        if os.path.exists(model_path):
            print(f"\n📊 Processing 2025-{month}...")
            try:
                importance_df = load_model_and_get_importance_with_names(
                    model_path, feature_names
                )
                all_importance[f"2025-{month}"] = importance_df

                print(f"   ✓ {len(importance_df)} features processed")
                print(f"   Top 5 features:")
                for i, row in importance_df.head().iterrows():
                    print(f"     {row['feature']}: {row['importance']:.2f}")

            except Exception as e:
                print(f"   ❌ Error processing {month}: {e}")
        else:
            print(f"   ⚠️  Model file not found: {model_path}")

    # Calculate average importance across all months
    if all_importance:
        print(f"\n📈 Calculating average feature importance...")

        # Get all unique features
        all_features = set()
        for month_data in all_importance.values():
            all_features.update(month_data["feature"].tolist())

        # Calculate average importance for each feature
        avg_importance = []
        for feature in all_features:
            importances = []
            for month_data in all_importance.values():
                feature_row = month_data[month_data["feature"] == feature]
                if not feature_row.empty:
                    importances.append(feature_row["importance"].iloc[0])

            if importances:
                avg_importance.append(
                    {
                        "feature": feature,
                        "avg_importance": np.mean(importances),
                        "std_importance": np.std(importances),
                        "months_count": len(importances),
                    }
                )

        # Sort by average importance
        avg_df = pd.DataFrame(avg_importance).sort_values(
            "avg_importance", ascending=False
        )

        # Save results
        output_dir = "results/monthly_rolling_2025"
        os.makedirs(output_dir, exist_ok=True)

        # Save average importance with names
        avg_path = os.path.join(output_dir, "feature_importance_with_names.csv")
        avg_df.to_csv(avg_path, index=False)
        print(f"   💾 Average importance with names saved: {avg_path}")

        # Save monthly importance with names
        monthly_path = os.path.join(
            output_dir, "feature_importance_monthly_with_names.csv"
        )
        monthly_data = []
        for month, df in all_importance.items():
            df_copy = df.copy()
            df_copy["month"] = month
            monthly_data.append(df_copy)

        if monthly_data:
            monthly_df = pd.concat(monthly_data, ignore_index=True)
            monthly_df.to_csv(monthly_path, index=False)
            print(f"   💾 Monthly importance with names saved: {monthly_path}")

        # Print top 20 features with actual names
        print(f"\n🏆 Top 20 Features (Average Importance with Names):")
        print("-" * 100)
        for i, row in avg_df.head(20).iterrows():
            print(
                f"{row['feature']:<50} {row['avg_importance']:>8.2f} ± {row['std_importance']:>6.2f}"
            )

        # Feature categories analysis with actual names
        print(f"\n📊 Feature Categories Analysis (with Names):")
        categories = {
            "WPT": [f for f in avg_df["feature"] if "wpt" in f.lower()],
            "Hurst": [f for f in avg_df["feature"] if "hurst" in f.lower()],
            "Hilbert": [f for f in avg_df["feature"] if "hilbert" in f.lower()],
            "Spectral": [f for f in avg_df["feature"] if "spectral" in f.lower()],
            "OrderFlow": [
                f
                for f in avg_df["feature"]
                if any(x in f.lower() for x in ["cvd", "taker", "buy", "sell"])
            ],
            "Technical": [
                f
                for f in avg_df["feature"]
                if any(
                    x in f.lower()
                    for x in ["rsi", "macd", "bb", "ema", "sma", "atr", "stoch"]
                )
            ],
            "Volume": [f for f in avg_df["feature"] if "volume" in f.lower()],
            "Price": [
                f
                for f in avg_df["feature"]
                if any(x in f.lower() for x in ["close", "open", "high", "low"])
            ],
            "Derived": [
                f
                for f in avg_df["feature"]
                if any(
                    x in f.lower() for x in ["hl", "hc", "lc", "tr", "return", "change"]
                )
            ],
        }

        for category, features in categories.items():
            if features:
                category_importance = avg_df[avg_df["feature"].isin(features)][
                    "avg_importance"
                ].sum()
                print(
                    f"   {category:<12}: {len(features):>3} features, {category_importance:>8.2f} total importance"
                )
                # Show top 3 features in each category
                top_features = avg_df[avg_df["feature"].isin(features)].head(3)
                for _, feat in top_features.iterrows():
                    print(f"     - {feat['feature']}: {feat['avg_importance']:.2f}")

        print(f"\n✅ Feature importance analysis with names complete!")

    else:
        print("❌ No models found to analyze")


if __name__ == "__main__":
    main()
