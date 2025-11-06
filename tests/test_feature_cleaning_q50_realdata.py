"""Real-data feature cleaning + Q50 constraint tests using parquet_data.

This test loads real parquet files, tests various methods to fix Q50 constraint violations,
and compares their effectiveness.
"""

import os
import glob
import unittest
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Ensure src on path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.data_tools.baseline_feature_engineering import (
    BaselineFeatureEngineer,
    get_baseline_feature_columns,
)
from ml_trading.models.lightgbm_model import LightGBMModel

PARQUET_DIR = "/home/yin/trading/ml_trading_bot/data/parquet_data"


class TestFeatureCleaningQ50RealData(unittest.TestCase):
    """Run feature cleaning and Q50 checks on real parquet data."""

    def _load_sample_frames(self, max_files: int = 2) -> list[pd.DataFrame]:
        """Load sample parquet files."""
        if not os.path.isdir(PARQUET_DIR):
            self.skipTest(f"Parquet dir not found: {PARQUET_DIR}")

        # Prefer recent months for BTC/ETH/SOL
        patterns = [
            os.path.join(PARQUET_DIR, "BTC-USD_2025-0*.parquet"),
            os.path.join(PARQUET_DIR, "ETH-USD_2025-0*.parquet"),
            os.path.join(PARQUET_DIR, "SOL-USD_2025-0*.parquet"),
            os.path.join(PARQUET_DIR, "BTC-USD_2024-1*.parquet"),
            os.path.join(PARQUET_DIR, "ETH-USD_2024-1*.parquet"),
            os.path.join(PARQUET_DIR, "SOL-USD_2024-1*.parquet"),
            os.path.join(PARQUET_DIR, "*.parquet"),
        ]

        files: list[str] = []
        for pat in patterns:
            cand = glob.glob(pat)
            for f in cand:
                if f not in files and os.path.isfile(f):
                    files.append(f)
            if len(files) >= max_files:
                break

        if not files:
            self.skipTest("No parquet files found under parquet_data")

        frames: list[pd.DataFrame] = []
        for f in files[:max_files]:
            df = pd.read_parquet(f)
            if "timestamp" in df.columns and df.index.name != "timestamp":
                df = df.set_index("timestamp")
            # Infer symbol from filename
            fname = os.path.basename(f)
            if "BTC-USD" in fname:
                df["symbol"] = "BTCUSDT"
            elif "ETH-USD" in fname:
                df["symbol"] = "ETHUSDT"
            elif "SOL-USD" in fname:
                df["symbol"] = "SOLUSDT"
            else:
                df["symbol"] = "UNKNOWN"
            frames.append(df)

        return frames

    def _clean_features(self, features_df: pd.DataFrame,
                        feature_cols: list[str]) -> pd.DataFrame:
        """Clean features: fill NaN and replace inf."""
        cleaned = features_df.copy()
        for col in feature_cols:
            if col in cleaned.columns:
                if cleaned[col].isna().any():
                    cleaned[col] = cleaned[col].fillna(0.0)
                if np.isinf(cleaned[col]).any():
                    finite_max = cleaned[col].replace([np.inf, -np.inf],
                                                      np.nan).abs().max()
                    finite_max = float(finite_max) if pd.notna(
                        finite_max) else 1e6
                    cleaned[col] = cleaned[col].replace(
                        [np.inf, -np.inf], [finite_max, -finite_max])
        return cleaned

    def _train_quantile_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            groups: Optional[np.ndarray],
            robust_params: Optional[Dict] = None,
            auto_tune: bool = False,
            tune_trials: int = 20) -> Tuple[float, float, float]:
        """Train Q10, Q50, Q90 models and return losses.
        
        Args:
            X_train: Training features
            y_train: Training target
            groups: Optional groups for GroupKFold
            robust_params: Optional robust parameters to use
            auto_tune: If True, automatically tune Q50 model parameters (default: False)
            tune_trials: Number of trials for auto-tuning if auto_tune=True (default: 20)
        """
        # Base params with longer training for Q50
        base_params = {
            "num_boost_round": 1000,
            "learning_rate": 0.01,
            "min_data_in_leaf": 50,
            "num_leaves": 31,
            "lambda_l2": 0.1,
        }
        if robust_params:
            base_params.update(robust_params)

        # For Q50, use more conservative params to prevent overfitting
        q50_params = base_params.copy()
        q50_params["num_boost_round"] = 2000
        q50_params["learning_rate"] = 0.005

        q10 = LightGBMModel(model_type="quantile",
                            quantile_alpha=0.1,
                            params=base_params)
        q50 = LightGBMModel(model_type="quantile",
                            quantile_alpha=0.5,
                            params=q50_params)
        q90 = LightGBMModel(model_type="quantile",
                            quantile_alpha=0.9,
                            params=base_params)

        # Auto-tune Q50 model if requested
        if auto_tune:
            print("  🔍 Auto-tuning Q50 model parameters...")
            m50, _ = q50.train(X_train,
                               y_train,
                               n_splits=2,
                               groups=groups,
                               auto_tune_params=True,
                               tune_trials=tune_trials)
        else:
            m50, _ = q50.train(X_train, y_train, n_splits=2, groups=groups)

        m10, _ = q10.train(X_train, y_train, n_splits=2, groups=groups)
        m90, _ = q90.train(X_train, y_train, n_splits=2, groups=groups)

        l10 = float(m10.get("cv_quantile_loss", np.inf))
        l50 = float(m50.get("cv_quantile_loss", np.inf))
        l90 = float(m90.get("cv_quantile_loss", np.inf))

        return l10, l50, l90

    def test_realdata_q50_fix_methods(self) -> None:
        """Test various methods to fix Q50 constraint violations."""
        print("\n" + "=" * 80)
        print("Q50 Constraint Fix Methods Comparison")
        print("=" * 80)

        # Load data
        frames = self._load_sample_frames(max_files=2)
        df = pd.concat(frames, axis=0).sort_index()
        if len(df) > 50000:
            df = df.iloc[-50000:]

        # Feature engineering
        engineer = BaselineFeatureEngineer()
        feat_df = engineer.engineer_features(df, fit=True)
        feature_cols = get_baseline_feature_columns(feat_df)
        feat_df = self._clean_features(feat_df, feature_cols)

        # Build future returns
        future_returns = (feat_df["close"].shift(-3) / feat_df["close"] -
                          1).rename("future_return")
        aligned = feat_df.join(future_returns, how="inner").dropna()

        if len(aligned) < 200:
            self.skipTest(f"Insufficient aligned samples: {len(aligned)}")

        X = aligned[feature_cols]
        y = aligned["future_return"]
        groups = aligned[
            "symbol"].values if "symbol" in aligned.columns else None

        # Print target statistics
        print(f"\nTarget (future_return) statistics:")
        print(f"  Samples: {len(y)}")
        print(f"  Mean: {y.mean():.6f}, Std: {y.std():.6f}")
        print(f"  Min: {y.min():.6f}, Max: {y.max():.6f}")
        print(
            f"  Q10: {y.quantile(0.1):.6f}, Q50: {y.quantile(0.5):.6f}, Q90: {y.quantile(0.9):.6f}"
        )
        print(f"  Skew: {y.skew():.2f}, Kurtosis: {y.kurt():.2f}")

        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

        results = {}

        # Method 1: Baseline (original)
        print("\n" + "-" * 80)
        print("Method 1: Baseline (original)")
        print("-" * 80)
        try:
            l10, l50, l90 = self._train_quantile_models(X_train,
                                                        y_train,
                                                        groups,
                                                        auto_tune=False)
            results["baseline"] = {
                "l10": l10,
                "l50": l50,
                "l90": l90,
                "q50_ratio":
                l50 / max(l10, l90) if max(l10, l90) > 0 else np.inf,
                "passed": l50 <= max(l10, l90) * 1.05
            }
            print(f"  Q10={l10:.6f}, Q50={l50:.6f}, Q90={l90:.6f}")
            print(f"  Q50/max(Q10,Q90)={results['baseline']['q50_ratio']:.2f}")
            print(
                f"  Status: {'✅ PASSED' if results['baseline']['passed'] else '❌ FAILED'}"
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["baseline"] = {"error": str(e)}

        # Method 2: Winsorize (clip extreme values) - more aggressive
        print("\n" + "-" * 80)
        print("Method 2: Winsorize (clip to [0.5%, 99.5%])")
        print("-" * 80)
        try:
            q_lo, q_hi = y_train.quantile(0.005), y_train.quantile(0.995)
            y_win = y_train.clip(lower=q_lo, upper=q_hi)
            clipped_count = ((y_train < q_lo) | (y_train > q_hi)).sum()
            print(
                f"  Clipped {clipped_count} extreme values ({clipped_count/len(y_train)*100:.1f}%)"
            )
            print(f"  Range: [{q_lo:.6f}, {q_hi:.6f}]")

            robust_params = {
                "num_boost_round": 1000,
                "learning_rate": 0.01,
                "min_data_in_leaf": 50,
                "num_leaves": 31,
                "lambda_l2": 0.1,
            }
            l10, l50, l90 = self._train_quantile_models(X_train,
                                                        y_win,
                                                        groups,
                                                        robust_params,
                                                        auto_tune=True,
                                                        tune_trials=15)
            results["winsorize"] = {
                "l10": l10,
                "l50": l50,
                "l90": l90,
                "q50_ratio":
                l50 / max(l10, l90) if max(l10, l90) > 0 else np.inf,
                "passed": l50 <= max(l10, l90) * 1.05
            }
            print(f"  Q10={l10:.6f}, Q50={l50:.6f}, Q90={l90:.6f}")
            print(
                f"  Q50/max(Q10,Q90)={results['winsorize']['q50_ratio']:.2f}")
            print(
                f"  Status: {'✅ PASSED' if results['winsorize']['passed'] else '❌ FAILED'}"
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["winsorize"] = {"error": str(e)}

        # Method 3: Log-return (log transformation)
        print("\n" + "-" * 80)
        print("Method 3: Log-return (log transformation)")
        print("-" * 80)
        try:
            # Use log returns instead of simple returns
            y_log = np.log(aligned["close"].shift(-3) / aligned["close"])
            y_log = y_log.replace([np.inf, -np.inf], np.nan).dropna()
            common_idx = y_log.index.intersection(X.index)
            X_log = X.loc[common_idx]
            y_log = y_log.loc[common_idx]

            if len(y_log) < 200:
                print(
                    f"  ⚠️  Insufficient samples after log transform: {len(y_log)}"
                )
                results["log_return"] = {"error": "Insufficient samples"}
            else:
                split_idx2 = int(len(X_log) * 0.8)
                X_log_train = X_log.iloc[:split_idx2]
                y_log_train = y_log.iloc[:split_idx2]
                groups_log = aligned.loc[
                    common_idx,
                    "symbol"].values if "symbol" in aligned.columns else None

                print(f"  Samples: {len(y_log_train)}")
                print(
                    f"  Range: [{y_log_train.min():.6f}, {y_log_train.max():.6f}]"
                )

                robust_params = {
                    "num_boost_round": 1000,
                    "learning_rate": 0.01,
                    "min_data_in_leaf": 50,
                    "num_leaves": 31,
                    "lambda_l2": 0.1,
                }
                l10, l50, l90 = self._train_quantile_models(X_log_train,
                                                            y_log_train,
                                                            groups_log,
                                                            robust_params,
                                                            auto_tune=True,
                                                            tune_trials=15)
                results["log_return"] = {
                    "l10": l10,
                    "l50": l50,
                    "l90": l90,
                    "q50_ratio":
                    l50 / max(l10, l90) if max(l10, l90) > 0 else np.inf,
                    "passed": l50 <= max(l10, l90) * 1.05
                }
                print(f"  Q10={l10:.6f}, Q50={l50:.6f}, Q90={l90:.6f}")
                print(
                    f"  Q50/max(Q10,Q90)={results['log_return']['q50_ratio']:.2f}"
                )
                print(
                    f"  Status: {'✅ PASSED' if results['log_return']['passed'] else '❌ FAILED'}"
                )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["log_return"] = {"error": str(e)}

        # Method 4: More aggressive winsorize (0.1%-99.9%)
        print("\n" + "-" * 80)
        print("Method 4: Aggressive Winsorize (clip to [0.1%, 99.9%])")
        print("-" * 80)
        try:
            q_lo, q_hi = y_train.quantile(0.001), y_train.quantile(0.999)
            y_win_agg = y_train.clip(lower=q_lo, upper=q_hi)
            clipped_count = ((y_train < q_lo) | (y_train > q_hi)).sum()
            print(
                f"  Clipped {clipped_count} extreme values ({clipped_count/len(y_train)*100:.1f}%)"
            )
            print(f"  Range: [{q_lo:.6f}, {q_hi:.6f}]")

            robust_params = {
                "num_boost_round": 2000,
                "learning_rate": 0.005,
                "min_data_in_leaf": 100,
                "num_leaves": 31,
                "lambda_l2": 0.1,
            }
            l10, l50, l90 = self._train_quantile_models(X_train,
                                                        y_win_agg,
                                                        groups,
                                                        robust_params,
                                                        auto_tune=True,
                                                        tune_trials=15)
            results["winsorize_aggressive"] = {
                "l10": l10,
                "l50": l50,
                "l90": l90,
                "q50_ratio":
                l50 / max(l10, l90) if max(l10, l90) > 0 else np.inf,
                "passed": l50 <= max(l10, l90) * 1.05
            }
            print(f"  Q10={l10:.6f}, Q50={l50:.6f}, Q90={l90:.6f}")
            print(
                f"  Q50/max(Q10,Q90)={results['winsorize_aggressive']['q50_ratio']:.2f}"
            )
            print(
                f"  Status: {'✅ PASSED' if results['winsorize_aggressive']['passed'] else '❌ FAILED'}"
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["winsorize_aggressive"] = {"error": str(e)}

        # Method 5: Filter extreme periods (remove outliers)
        print("\n" + "-" * 80)
        print("Method 5: Filter extreme periods (remove >3σ outliers)")
        print("-" * 80)
        try:
            # Remove extreme outliers (>3 standard deviations, more aggressive)
            y_mean = y_train.mean()
            y_std = y_train.std()
            threshold = 3 * y_std  # More aggressive: 3σ instead of 5σ
            mask = (y_train >= y_mean - threshold) & (y_train
                                                      <= y_mean + threshold)
            filtered_count = (~mask).sum()

            if filtered_count > 0:
                X_filtered = X_train[mask]
                y_filtered = y_train[mask]
                groups_filtered = groups[mask] if groups is not None else None

                print(
                    f"  Filtered {filtered_count} extreme samples ({filtered_count/len(y_train)*100:.1f}%)"
                )
                print(f"  Remaining samples: {len(y_filtered)}")

                if len(y_filtered) < 200:
                    print(
                        f"  ⚠️  Insufficient samples after filtering: {len(y_filtered)}"
                    )
                    results["filter_extreme"] = {
                        "error": "Insufficient samples"
                    }
                else:
                    robust_params = {
                        "num_boost_round": 1000,
                        "learning_rate": 0.01,
                        "min_data_in_leaf": 50,
                        "num_leaves": 31,
                        "lambda_l2": 0.1,
                    }
                    l10, l50, l90 = self._train_quantile_models(
                        X_filtered,
                        y_filtered,
                        groups_filtered,
                        robust_params,
                        auto_tune=True,
                        tune_trials=15)
                    results["filter_extreme"] = {
                        "l10":
                        l10,
                        "l50":
                        l50,
                        "l90":
                        l90,
                        "q50_ratio":
                        l50 / max(l10, l90) if max(l10, l90) > 0 else np.inf,
                        "passed":
                        l50 <= max(l10, l90) * 1.05
                    }
                    print(f"  Q10={l10:.6f}, Q50={l50:.6f}, Q90={l90:.6f}")
                    print(
                        f"  Q50/max(Q10,Q90)={results['filter_extreme']['q50_ratio']:.2f}"
                    )
                    print(
                        f"  Status: {'✅ PASSED' if results['filter_extreme']['passed'] else '❌ FAILED'}"
                    )
            else:
                print(f"  No extreme samples to filter")
                results["filter_extreme"] = {"error": "No extreme samples"}
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["filter_extreme"] = {"error": str(e)}

        # Summary
        print("\n" + "=" * 80)
        print("Summary: Q50 Constraint Results")
        print("=" * 80)
        for method, result in results.items():
            if "error" in result:
                print(f"  {method}: ❌ Error - {result['error']}")
            else:
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(
                    f"  {method}: {status} (Q50/max={result['q50_ratio']:.2f})"
                )

        # Find best method
        passed_methods = [
            m for m, r in results.items()
            if "error" not in r and r.get("passed", False)
        ]
        if passed_methods:
            print(
                f"\n✅ Methods that PASSED Q50 constraint: {', '.join(passed_methods)}"
            )
        else:
            print(f"\n❌ No methods passed Q50 constraint. All methods failed.")

        # Assert: at least one method should pass
        self.assertTrue(
            len(passed_methods) > 0,
            f"All methods failed Q50 constraint. Results: {results}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
