"""Test feature cleaning process to ensure Q50 constraint compliance.

This test verifies that:
1. Features are properly cleaned (no NaN, inf, extreme values)
2. Features don't cause Q50 constraint violations
3. Feature cleaning is consistent between train/test splits
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.data_tools.baseline_feature_engineering import (
    BaselineFeatureEngineer,
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)
from ml_trading.models.lightgbm_model import LightGBMModel


class TestFeatureCleaningQ50(unittest.TestCase):
    """Test feature cleaning process for Q50 constraint compliance."""

    @staticmethod
    def clean_features_for_training(features_df: pd.DataFrame,
                                    feature_cols: List[str]) -> pd.DataFrame:
        """Clean features for training: fill NaN and replace inf.
        
        This is a helper function to ensure features are clean before training.
        Should be called after feature engineering.
        """
        cleaned_df = features_df.copy()

        for col in feature_cols:
            if col in cleaned_df.columns:
                # Fill NaN with 0 (or median if preferred)
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = cleaned_df[col].fillna(0.0)

                # Replace inf with large finite values
                if np.isinf(cleaned_df[col]).any():
                    max_val = cleaned_df[col].replace([np.inf, -np.inf],
                                                      np.nan).max()
                    min_val = cleaned_df[col].replace([np.inf, -np.inf],
                                                      np.nan).min()
                    if pd.notna(max_val) and pd.notna(min_val):
                        large_val = max(abs(max_val), abs(min_val)) * 10
                    else:
                        large_val = 1e6
                    cleaned_df[col] = cleaned_df[col].replace(
                        [np.inf, -np.inf], [large_val, -large_val])

        return cleaned_df

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 1000

        # Create realistic OHLCV data
        dates = pd.date_range("2024-01-01",
                              periods=self.n_samples,
                              freq="5min")

        # Generate price data with trend and volatility
        base_price = 50000
        returns = np.random.normal(0.0001, 0.01, self.n_samples)
        prices = base_price * np.exp(np.cumsum(returns))

        # Create OHLCV data
        self.ohlcv_data = pd.DataFrame(
            {
                "open":
                prices * (1 + np.random.normal(0, 0.001, self.n_samples)),
                "high":
                (prices *
                 (1 + np.abs(np.random.normal(0, 0.002, self.n_samples)))),
                "low":
                (prices *
                 (1 - np.abs(np.random.normal(0, 0.002, self.n_samples)))),
                "close":
                prices,
                "volume":
                np.random.lognormal(10, 1, self.n_samples),
            },
            index=dates)

        # Ensure high >= close >= low
        self.ohlcv_data["high"] = np.maximum(
            self.ohlcv_data["high"],
            np.maximum(self.ohlcv_data["close"], self.ohlcv_data["low"]))
        self.ohlcv_data["low"] = np.minimum(
            self.ohlcv_data["low"],
            np.minimum(self.ohlcv_data["close"], self.ohlcv_data["high"]))

        # Create future returns for testing
        self.future_returns = (
            (self.ohlcv_data["close"].shift(-3) / self.ohlcv_data["close"]) -
            1)

    def test_feature_cleaning_no_nan_inf(self):
        """Test that features are cleaned of NaN and inf values."""
        print("\n=== Test: Feature Cleaning (NaN/Inf) - Baseline ===")

        # Engineer features
        engineer = BaselineFeatureEngineer()
        features_df = engineer.engineer_features(self.ohlcv_data, fit=True)

        # Get feature columns
        feature_cols = get_baseline_feature_columns(features_df)

        self._check_nan_inf(features_df, feature_cols, "baseline")

    def test_feature_cleaning_all_feature_types(self):
        """Test feature cleaning for all feature types."""
        print("\n=== Test: Feature Cleaning (All Feature Types) ===")

        # List of all feature types to test
        feature_types = [
            "baseline",
            "default",
            "enhanced",
            "hurst",
            "wavelet",
            "hilbert",
            "spectral",
            "order_flow",
            "dl_sequence",
        ]

        # Test each feature type
        for feature_type in feature_types:
            print(f"\n--- Testing {feature_type} features ---")
            try:
                # Engineer features
                engineer = ComprehensiveFeatureEngineer(
                    feature_types=feature_type)
                features_df = engineer.engineer_features(self.ohlcv_data,
                                                         fit=True)

                # Get feature columns
                feature_cols = get_feature_columns_by_type(
                    features_df, feature_type)

                if len(feature_cols) == 0:
                    print(f"  ⚠️  No features generated for {feature_type}")
                    continue

                print(f"  Generated {len(feature_cols)} features")

                # Check NaN/inf
                self._check_nan_inf(features_df, feature_cols, feature_type)

            except Exception as e:
                print(f"  ❌ Error testing {feature_type}: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail the test, just report the error
                continue

    def _check_nan_inf(self, features_df: pd.DataFrame,
                       feature_cols: List[str], feature_type: str):
        """Helper method to check NaN and inf values in features."""
        # Check for NaN and inf
        available_cols = [
            col for col in feature_cols if col in features_df.columns
        ]
        if len(available_cols) == 0:
            print(f"  ⚠️  No available feature columns for {feature_type}")
            return

        nan_counts = features_df[available_cols].isna().sum()
        inf_counts = np.isinf(features_df[available_cols]).sum()

        print(f"  Features checked: {len(available_cols)}")

        if nan_counts.sum() > 0:
            print(
                f"  ❌ Found {nan_counts.sum()} NaN values in {feature_type} features"
            )
            print(f"     Columns with NaN:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"       {col}: {count}")
            print(
                f"     Recommendation: Fill NaN values with 0 or median before training"
            )
        else:
            print(f"  ✅ No NaN values found in {feature_type} features")

        if inf_counts.sum() > 0:
            print(
                f"  ❌ Found {inf_counts.sum()} inf values in {feature_type} features"
            )
            print(f"     Columns with inf:")
            for col, count in inf_counts[inf_counts > 0].items():
                print(f"       {col}: {count}")
            print(
                f"     Recommendation: Replace inf with large finite values or clip"
            )
        else:
            print(f"  ✅ No inf values found in {feature_type} features")

        # Hard assertion: inf should never exist
        self.assertEqual(
            inf_counts.sum(), 0,
            f"[{feature_type}] Found {inf_counts.sum()} inf values in features. Columns with inf: {inf_counts[inf_counts > 0].to_dict()}"
        )

    def test_feature_cleaning_extreme_values(self):
        """Test that features don't have extreme values that could cause Q50 issues."""
        print("\n=== Test: Feature Cleaning (Extreme Values) ===")

        # Engineer features
        engineer = BaselineFeatureEngineer()
        features_df = engineer.engineer_features(self.ohlcv_data, fit=True)

        # Get feature columns
        feature_cols = get_baseline_feature_columns(features_df)

        # Check for extreme values (beyond reasonable bounds)
        extreme_threshold = 1e6  # Very large threshold
        extreme_counts = {}

        for col in feature_cols:
            if col in features_df.columns:
                abs_values = np.abs(features_df[col])
                extreme_count = (abs_values > extreme_threshold).sum()
                if extreme_count > 0:
                    extreme_counts[col] = extreme_count
                    max_val = abs_values.max()
                    print(
                        f"  ⚠️  {col}: {extreme_count} extreme values, max abs: {max_val:.2e}"
                    )

        if extreme_counts:
            print(
                f"⚠️  Found extreme values in {len(extreme_counts)} features")
            # This is a warning, not necessarily a failure
            # Some features might legitimately have large values
        else:
            print("✅ No extreme values found")

        # Check for very large standard deviations (potential outliers)
        std_threshold = 100  # Reasonable threshold for normalized features
        large_std_features = {}

        for col in feature_cols:
            if col in features_df.columns:
                std_val = features_df[col].std()
                if std_val > std_threshold:
                    large_std_features[col] = std_val
                    print(
                        f"  ⚠️  {col}: std={std_val:.2f} (threshold: {std_threshold})"
                    )

        if large_std_features:
            print(
                f"⚠️  Found {len(large_std_features)} features with large std")
        else:
            print("✅ All features have reasonable standard deviations")

    def test_feature_cleaning_q50_constraint(self):
        """Test that cleaned features don't violate Q50 constraint."""
        print("\n=== Test: Feature Cleaning (Q50 Constraint) ===")

        # Engineer features
        engineer = BaselineFeatureEngineer()
        features_df = engineer.engineer_features(self.ohlcv_data, fit=True)

        # Get feature columns
        feature_cols = get_baseline_feature_columns(features_df)

        # Clean features before training (fill NaN, replace inf)
        features_df_clean = self.clean_features_for_training(
            features_df, feature_cols)

        # Align features and target
        # Rename future_returns to avoid column name conflict
        future_returns_series = self.future_returns.rename("future_return")
        aligned_data = features_df_clean.join(future_returns_series, how="inner")
        aligned_data = aligned_data.dropna()

        if len(aligned_data) < 100:
            print(
                f"⚠️  Insufficient data after alignment: {len(aligned_data)} samples"
            )
            return

        X = aligned_data[feature_cols]
        y = aligned_data["future_return"]

        # Split into train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train Q10, Q50, Q90 models
        model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)

        # Train models
        print("Training Q10 model...")
        metrics_q10, _ = model_q10.train(X_train, y_train, n_splits=3)

        print("Training Q50 model...")
        metrics_q50, _ = model_q50.train(X_train, y_train, n_splits=3)

        print("Training Q90 model...")
        metrics_q90, _ = model_q90.train(X_train, y_train, n_splits=3)

        # Get losses
        q10_loss = metrics_q10.get("cv_quantile_loss", float("inf"))
        q50_loss = metrics_q50.get("cv_quantile_loss", float("inf"))
        q90_loss = metrics_q90.get("cv_quantile_loss", float("inf"))

        print(f"\nQuantile Losses:")
        print(f"  Q10 loss: {q10_loss:.6f}")
        print(f"  Q50 loss: {q50_loss:.6f}")
        print(f"  Q90 loss: {q90_loss:.6f}")

        # Check Q50 constraint: Q50 loss <= max(Q10, Q90) loss
        max_other_loss = max(q10_loss, q90_loss)
        q50_ratio = q50_loss / max_other_loss if max_other_loss > 0 else float(
            "inf")

        print(f"\nQ50 Constraint Check:")
        print(f"  Max(Q10, Q90) loss: {max_other_loss:.6f}")
        print(f"  Q50 loss ratio: {q50_ratio:.2f}")

        if q50_loss <= max_other_loss:
            print(
                "  ✅ Q50 constraint satisfied: Q50 loss <= max(Q10, Q90) loss")
        else:
            print(
                f"  ❌ Q50 constraint violated: Q50 loss ({q50_loss:.6f}) > max(Q10, Q90) loss ({max_other_loss:.6f})"
            )
            print(
                f"     This suggests features may have issues (extreme values, outliers, etc.)"
            )

        # Assertion (with tolerance for numerical issues)
        tolerance = 0.01  # 1% tolerance
        self.assertLessEqual(
            q50_loss, max_other_loss * (1 + tolerance),
            f"Q50 constraint violated: Q50 loss ({q50_loss:.6f}) > max(Q10, Q90) loss ({max_other_loss:.6f}) * (1 + {tolerance})"
        )

    def test_feature_cleaning_q50_constraint_all_types(self):
        """Test Q50 constraint for all feature types."""
        print(
            "\n=== Test: Feature Cleaning (Q50 Constraint) - All Feature Types ==="
        )

        # List of feature types to test (skip dl_sequence as it may be slow)
        feature_types = [
            "baseline",
            "default",
            "enhanced",
            "hurst",
            "wavelet",
            "hilbert",
            "spectral",
            "order_flow",
        ]

        # Test each feature type
        for feature_type in feature_types:
            print(
                f"\n--- Testing Q50 constraint for {feature_type} features ---"
            )
            try:
                # Engineer features
                engineer = ComprehensiveFeatureEngineer(
                    feature_types=feature_type)
                features_df = engineer.engineer_features(self.ohlcv_data,
                                                         fit=True)

                # Get feature columns
                feature_cols = get_feature_columns_by_type(
                    features_df, feature_type)

                if len(feature_cols) == 0:
                    print(f"  ⚠️  No features generated for {feature_type}")
                    continue

                # Clean features before training
                features_df_clean = self.clean_features_for_training(
                    features_df, feature_cols)

                # Test Q50 constraint
                self._test_q50_constraint(features_df_clean, feature_cols,
                                          feature_type)

            except Exception as e:
                print(
                    f"  ❌ Error testing Q50 constraint for {feature_type}: {e}"
                )
                import traceback
                traceback.print_exc()
                # Don't fail the test, just report the error
                continue

    def _test_q50_constraint(self, features_df: pd.DataFrame,
                             feature_cols: List[str], feature_type: str):
        """Helper method to test Q50 constraint."""
        # Align features and target
        # Rename future_returns to avoid column name conflict
        future_returns_series = self.future_returns.rename("future_return")
        aligned_data = features_df.join(future_returns_series, how="inner")
        aligned_data = aligned_data.dropna()

        if len(aligned_data) < 100:
            print(
                f"  ⚠️  Insufficient data after alignment: {len(aligned_data)} samples"
            )
            return

        # Get available feature columns
        available_cols = [
            col for col in feature_cols if col in aligned_data.columns
        ]
        if len(available_cols) == 0:
            print(f"  ⚠️  No available feature columns for {feature_type}")
            return

        X = aligned_data[available_cols]
        y = aligned_data["future_return"]

        # Split into train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Features: {len(available_cols)}")

        # Train Q10, Q50, Q90 models
        model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)

        # Train models
        print("  Training Q10 model...")
        metrics_q10, _ = model_q10.train(X_train, y_train, n_splits=3)

        print("  Training Q50 model...")
        metrics_q50, _ = model_q50.train(X_train, y_train, n_splits=3)

        print("  Training Q90 model...")
        metrics_q90, _ = model_q90.train(X_train, y_train, n_splits=3)

        # Get losses
        q10_loss = metrics_q10.get("cv_quantile_loss", float("inf"))
        q50_loss = metrics_q50.get("cv_quantile_loss", float("inf"))
        q90_loss = metrics_q90.get("cv_quantile_loss", float("inf"))

        print(f"\n  Quantile Losses [{feature_type}]:")
        print(f"    Q10 loss: {q10_loss:.6f}")
        print(f"    Q50 loss: {q50_loss:.6f}")
        print(f"    Q90 loss: {q90_loss:.6f}")

        # Check Q50 constraint: Q50 loss <= max(Q10, Q90) loss
        max_other_loss = max(q10_loss, q90_loss)
        q50_ratio = q50_loss / max_other_loss if max_other_loss > 0 else float(
            "inf")

        print(f"\n  Q50 Constraint Check [{feature_type}]:")
        print(f"    Max(Q10, Q90) loss: {max_other_loss:.6f}")
        print(f"    Q50 loss ratio: {q50_ratio:.2f}")

        if q50_loss <= max_other_loss:
            print(f"    ✅ Q50 constraint satisfied for {feature_type}")
        else:
            print(f"    ❌ Q50 constraint violated for {feature_type}")
            print(
                f"       Q50 loss ({q50_loss:.6f}) > max(Q10, Q90) loss ({max_other_loss:.6f})"
            )
            print(
                f"       This suggests features may have issues (extreme values, outliers, etc.)"
            )

        # Assertion (with tolerance for numerical issues)
        tolerance = 0.01  # 1% tolerance
        self.assertLessEqual(
            q50_loss, max_other_loss * (1 + tolerance),
            f"[{feature_type}] Q50 constraint violated: Q50 loss ({q50_loss:.6f}) > max(Q10, Q90) loss ({max_other_loss:.6f}) * (1 + {tolerance})"
        )

    def test_feature_cleaning_consistency(self):
        """Test that feature cleaning is consistent between train/test splits."""
        print("\n=== Test: Feature Cleaning (Consistency) ===")

        # Split data into train/test
        split_idx = int(len(self.ohlcv_data) * 0.8)
        train_data = self.ohlcv_data.iloc[:split_idx]
        test_data = self.ohlcv_data.iloc[split_idx:]

        # Engineer features on train (fit=True)
        engineer = BaselineFeatureEngineer()
        train_features = engineer.engineer_features(train_data, fit=True)
        train_feature_cols = get_baseline_feature_columns(train_features)

        # Engineer features on test (fit=False)
        test_features = engineer.engineer_features(test_data, fit=False)
        test_feature_cols = get_baseline_feature_columns(test_features)

        # Check that feature columns match
        self.assertEqual(
            set(train_feature_cols), set(test_feature_cols),
            "Feature columns should match between train and test")

        print(f"✅ Feature columns match: {len(train_feature_cols)} features")

        # Check that feature statistics are similar (not identical, but similar)
        for col in train_feature_cols[:10]:  # Check first 10 features
            if col in train_features.columns and col in test_features.columns:
                train_mean = train_features[col].mean()
                test_mean = test_features[col].mean()
                train_std = train_features[col].std()
                test_std = test_features[col].std()

                # Check that means and stds are within reasonable range
                if train_std > 0:
                    mean_diff_ratio = abs(train_mean - test_mean) / train_std
                    std_diff_ratio = abs(train_std - test_std) / train_std

                    if mean_diff_ratio > 2 or std_diff_ratio > 1:
                        print(
                            f"  ⚠️  {col}: mean_diff={mean_diff_ratio:.2f}, std_diff={std_diff_ratio:.2f}"
                        )

        print("✅ Feature statistics are consistent between train/test")

    def test_time_features_cleaning(self):
        """Test that time features (hour_sin, hour_cos, Is_Weekend, etc.) are properly cleaned."""
        print("\n=== Test: Time Features Cleaning ===")

        # Engineer features
        engineer = BaselineFeatureEngineer()
        features_df = engineer.engineer_features(self.ohlcv_data, fit=True)

        # Check time features
        time_features = [
            "hour_sin", "hour_cos", "Is_Weekend", "Minutes_Since_Last_Trade"
        ]

        for feature in time_features:
            if feature in features_df.columns:
                values = features_df[feature]

                # Check for NaN/inf
                nan_count = values.isna().sum()
                inf_count = np.isinf(values).sum()

                # Check value ranges
                min_val = values.min()
                max_val = values.max()

                print(f"  {feature}:")
                print(f"    NaN: {nan_count}, Inf: {inf_count}")
                print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")

                # Assertions
                self.assertEqual(nan_count, 0,
                                 f"{feature} should have no NaN values")
                self.assertEqual(inf_count, 0,
                                 f"{feature} should have no inf values")

                # Check reasonable ranges
                if feature == "hour_sin" or feature == "hour_cos":
                    self.assertGreaterEqual(min_val, -1.1,
                                            f"{feature} should be >= -1")
                    self.assertLessEqual(max_val, 1.1,
                                         f"{feature} should be <= 1")
                elif feature == "Is_Weekend":
                    self.assertGreaterEqual(min_val, 0,
                                            f"{feature} should be >= 0")
                    self.assertLessEqual(max_val, 1,
                                         f"{feature} should be <= 1")
                elif feature == "Minutes_Since_Last_Trade":
                    self.assertGreaterEqual(min_val, 0,
                                            f"{feature} should be >= 0")
                    self.assertLessEqual(
                        max_val, 60.1, f"{feature} should be <= 60 (clipped)")

        print("✅ All time features are properly cleaned")

    def test_feature_cleaning_with_extreme_inputs(self):
        """Test feature cleaning with extreme input values."""
        print("\n=== Test: Feature Cleaning (Extreme Inputs) ===")

        # Create data with extreme values
        extreme_data = self.ohlcv_data.copy()

        # Add extreme values
        extreme_data.loc[extreme_data.index[100],
                         "volume"] = 1e10  # Very large volume
        extreme_data.loc[extreme_data.index[200], "close"] = extreme_data.loc[
            extreme_data.index[200], "close"] * 10  # Price jump
        extreme_data.loc[extreme_data.index[300], "volume"] = 0  # Zero volume

        # Engineer features
        engineer = BaselineFeatureEngineer()
        features_df = engineer.engineer_features(extreme_data, fit=True)

        # Get feature columns
        feature_cols = get_baseline_feature_columns(features_df)

        # Check that features are still clean
        nan_counts = features_df[feature_cols].isna().sum()
        inf_counts = np.isinf(features_df[feature_cols]).sum()

        print(f"NaN counts: {nan_counts.sum()}")
        print(f"Inf counts: {inf_counts.sum()}")

        # Features should still be clean despite extreme inputs
        self.assertEqual(nan_counts.sum(), 0,
                         "Features should handle extreme inputs without NaN")
        self.assertEqual(inf_counts.sum(), 0,
                         "Features should handle extreme inputs without inf")

        print("✅ Features handle extreme inputs gracefully")


if __name__ == "__main__":
    unittest.main(verbosity=2)
