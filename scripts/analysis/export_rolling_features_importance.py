"""Export feature importance from rolling training models."""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add common utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from training_utils import train_lightgbm_model


def load_model_and_get_importance(model_path):
    """Load a LightGBM model and get feature importance."""
    import lightgbm as lgb

    # Load model
    model = lgb.Booster(model_file=model_path)

    # Get feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()

    # Create DataFrame
    df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    return df


def main():
    results_dir = "results/monthly_rolling_2025"

    print("🔍 Extracting feature importance from rolling training models...")

    all_importance = {}

    # Process each month's model
    for month in ["01", "02", "03", "04", "05", "06"]:
        model_path = os.path.join(results_dir, f"model_2025-{month}.txt")

        if os.path.exists(model_path):
            print(f"\n📊 Processing 2025-{month}...")
            try:
                importance_df = load_model_and_get_importance(model_path)
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

        # Save average importance
        avg_path = os.path.join(output_dir, "feature_importance_average.csv")
        avg_df.to_csv(avg_path, index=False)
        print(f"   💾 Average importance saved: {avg_path}")

        # Save monthly importance
        monthly_path = os.path.join(output_dir, "feature_importance_monthly.csv")
        monthly_data = []
        for month, df in all_importance.items():
            df_copy = df.copy()
            df_copy["month"] = month
            monthly_data.append(df_copy)

        if monthly_data:
            monthly_df = pd.concat(monthly_data, ignore_index=True)
            monthly_df.to_csv(monthly_path, index=False)
            print(f"   💾 Monthly importance saved: {monthly_path}")

        # Print top 20 features
        print(f"\n🏆 Top 20 Features (Average Importance):")
        print("-" * 80)
        for i, row in avg_df.head(20).iterrows():
            print(
                f"{row['feature']:<40} {row['avg_importance']:>8.2f} ± {row['std_importance']:>6.2f}"
            )

        # Feature categories analysis
        print(f"\n📊 Feature Categories Analysis:")
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
                if any(x in f.lower() for x in ["rsi", "macd", "bb", "ema", "sma"])
            ],
            "Volume": [f for f in avg_df["feature"] if "volume" in f.lower()],
            "Price": [
                f
                for f in avg_df["feature"]
                if any(x in f.lower() for x in ["close", "open", "high", "low"])
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

        print(f"\n✅ Feature importance analysis complete!")

    else:
        print("❌ No models found to analyze")


if __name__ == "__main__":
    main()
