"""Detailed debugging tests for Q50 model issues."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from time_series_model.models.lightgbm_model import LightGBMModel


class TestQ50DebugDetailed(unittest.TestCase):
    """Detailed debugging tests for Q50 model issues."""

    def setUp(self):
        """Set up test fixtures with data that causes Q50 issues."""
        # Create data that specifically causes Q50 loss > Q10/Q90 loss
        np.random.seed(42)
        n_samples = 1000
        
        # Create data with the pattern that causes issues:
        # 1. Most values are small and clustered
        # 2. A few extreme outliers that heavily influence MAE (Q50 loss)
        # 3. Q10/Q90 can "ignore" these outliers more easily due to their asymmetric loss
        
        # Base data (95% of samples)
        base_data = np.random.normal(0, 0.008, int(n_samples * 0.95))
        
        # Extreme outliers (5% of samples) - these cause Q50 problems
        n_extreme = n_samples - len(base_data)
        extreme_positive = np.random.uniform(0.04, 0.08, n_extreme // 2)
        extreme_negative = np.random.uniform(-0.08, -0.04, n_extreme - n_extreme // 2)
        extreme_data = np.concatenate([extreme_positive, extreme_negative])
        
        self.y_return = pd.Series(np.concatenate([base_data, extreme_data]), name='future_return')
        
        # Create features with varying predictive power
        self.X_df = pd.DataFrame({
            'weak_predictor': np.random.normal(0, 1, n_samples),
            'strong_predictor': self.y_return + np.random.normal(0, 0.002, n_samples),
            'noise_feature_1': np.random.normal(0, 1, n_samples),
            'noise_feature_2': np.random.normal(0, 1, n_samples),
        })
        
    def test_residual_analysis(self):
        """Detailed residual analysis to understand Q50 issues."""
        # Train Q50 model
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        pred_q50 = model_q50.model.predict(small_X.values)
        
        # Calculate residuals
        residuals = small_y.values - pred_q50
        abs_residuals = np.abs(residuals)
        
        print("=== Detailed Residual Analysis ===")
        print(f"Mean residual: {np.mean(residuals):.6f}")
        print(f"Std residual: {np.std(residuals):.6f}")
        print(f"Median residual: {np.median(residuals):.6f}")
        print(f"Mean absolute residual: {np.mean(abs_residuals):.6f}")
        print(f"Median absolute residual: {np.median(abs_residuals):.6f}")
        
        # Percentile analysis
        percentiles = [5, 25, 50, 75, 95]
        residual_percentiles = np.percentile(abs_residuals, percentiles)
        
        print(f"\nAbsolute residual percentiles:")
        for p, value in zip(percentiles, residual_percentiles):
            print(f"  {p:2d}th percentile: {value:.6f}")
            
        # Identify extreme residuals
        q95 = np.percentile(abs_residuals, 95)
        extreme_mask = abs_residuals > q95
        normal_mask = ~extreme_mask
        
        print(f"\nExtreme residuals (> 95th percentile):")
        print(f"  Count: {np.sum(extreme_mask)} ({np.sum(extreme_mask)/len(abs_residuals)*100:.1f}%)")
        print(f"  Mean: {np.mean(abs_residuals[extreme_mask]):.6f}")
        print(f"  Normal residuals mean: {np.mean(abs_residuals[normal_mask]):.6f}")
        
        if np.mean(abs_residuals[normal_mask]) > 0:
            extreme_ratio = np.mean(abs_residuals[extreme_mask]) / np.mean(abs_residuals[normal_mask])
            print(f"  Extreme/Normal ratio: {extreme_ratio:.2f}")
            
        return residuals, abs_residuals
        
    def test_quantile_specific_behavior(self):
        """Test how different quantiles behave with extreme values."""
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        # Train all three quantile models
        models = {}
        predictions = {}
        losses = {}
        
        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            model = LightGBMModel(model_type="quantile", quantile_alpha=q)
            metrics, _ = model.train(small_X, small_y, n_splits=2)
            pred = model.model.predict(small_X.values)
            
            models[q] = model
            predictions[q] = pred
            losses[q] = metrics.get('cv_quantile_loss', 0)
            
        print("=== Quantile-Specific Behavior Analysis ===")
        for q in quantiles:
            print(f"Q{int(q*100)} loss: {losses[q]:.6f}")
            
        # Check quantile ordering
        q10_pred = predictions[0.1]
        q50_pred = predictions[0.5]
        q90_pred = predictions[0.9]
        
        # Verify quantile ordering
        order_violations = np.sum(~((q10_pred <= q50_pred) & (q50_pred <= q90_pred)))
        violation_pct = order_violations / len(q50_pred) * 100
        
        print(f"\nQuantile ordering violations: {order_violations} ({violation_pct:.1f}%)")
        
        # Analyze how each quantile handles extreme values
        y_true = small_y.values
        
        # Calculate residuals for each quantile
        residuals_q10 = np.abs(y_true - q10_pred)
        residuals_q50 = np.abs(y_true - q50_pred)
        residuals_q90 = np.abs(y_true - q90_pred)
        
        # Compare how each quantile handles extremes
        q90_residual_threshold = np.percentile(residuals_q50, 90)
        extreme_mask = residuals_q50 > q90_residual_threshold
        
        print(f"\nExtreme residual threshold (90th percentile): {q90_residual_threshold:.6f}")
        print(f"Extreme samples: {np.sum(extreme_mask)}")
        
        if np.sum(extreme_mask) > 0:
            mean_resid_q10_extreme = np.mean(residuals_q10[extreme_mask])
            mean_resid_q50_extreme = np.mean(residuals_q50[extreme_mask])
            mean_resid_q90_extreme = np.mean(residuals_q90[extreme_mask])
            
            mean_resid_q10_normal = np.mean(residuals_q10[~extreme_mask])
            mean_resid_q50_normal = np.mean(residuals_q50[~extreme_mask])
            mean_resid_q90_normal = np.mean(residuals_q90[~extreme_mask])
            
            print(f"\nExtreme samples residuals:")
            print(f"  Q10: {mean_resid_q10_extreme:.6f}")
            print(f"  Q50: {mean_resid_q50_extreme:.6f}")
            print(f"  Q90: {mean_resid_q90_extreme:.6f}")
            
            print(f"\nNormal samples residuals:")
            print(f"  Q10: {mean_resid_q10_normal:.6f}")
            print(f"  Q50: {mean_resid_q50_normal:.6f}")
            print(f"  Q90: {mean_resid_q90_normal:.6f}")
            
            # Check if Q50 is disproportionately affected
            if mean_resid_q50_normal > 0:
                extreme_ratio = mean_resid_q50_extreme / mean_resid_q50_normal
                print(f"\nQ50 extreme/normal residual ratio: {extreme_ratio:.2f}")
                
        return losses, predictions
        
    def test_model_prediction_distributions(self):
        """Analyze prediction distributions for each quantile model."""
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        # Train all three quantile models
        models = {}
        predictions = {}
        
        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            model = LightGBMModel(model_type="quantile", quantile_alpha=q)
            metrics, _ = model.train(small_X, small_y, n_splits=2)
            pred = model.model.predict(small_X.values)
            
            models[q] = model
            predictions[q] = pred
            
        print("=== Prediction Distribution Analysis ===")
        
        y_true = small_y.values
        
        for q in quantiles:
            pred = predictions[q]
            print(f"\nQ{int(q*100)} predictions:")
            print(f"  Mean: {np.mean(pred):.6f}")
            print(f"  Std: {np.std(pred):.6f}")
            print(f"  Min: {np.min(pred):.6f}")
            print(f"  Max: {np.max(pred):.6f}")
            print(f"  Median: {np.median(pred):.6f}")
            
        # Check prediction ranges
        q10_pred = predictions[0.1]
        q50_pred = predictions[0.5]
        q90_pred = predictions[0.9]
        
        pred_range_q10 = np.max(q10_pred) - np.min(q10_pred)
        pred_range_q50 = np.max(q50_pred) - np.min(q50_pred)
        pred_range_q90 = np.max(q90_pred) - np.min(q90_pred)
        
        true_range = np.max(y_true) - np.min(y_true)
        
        print(f"\nPrediction ranges:")
        print(f"  Q10 range: {pred_range_q10:.6f} ({pred_range_q10/true_range*100:.1f}% of true range)")
        print(f"  Q50 range: {pred_range_q50:.6f} ({pred_range_q50/true_range*100:.1f}% of true range)")
        print(f"  Q90 range: {pred_range_q90:.6f} ({pred_range_q90/true_range*100:.1f}% of true range)")
        print(f"  True range: {true_range:.6f}")
        
        # Check if predictions are too narrow
        if pred_range_q50/true_range < 0.3:
            print(f"\n⚠️  Q50 predictions are very narrow ({pred_range_q50/true_range*100:.1f}% of true range)")
            print(f"   This could contribute to Q50 loss issues")
            
        return predictions, y_true
        
    def test_extreme_value_handling_mechanisms(self):
        """Test different mechanisms for handling extreme values."""
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        print("=== Extreme Value Handling Mechanisms ===")
        
        # 1. Test with original data
        model_orig = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        metrics_orig, _ = model_orig.train(small_X, small_y, n_splits=2)
        loss_orig = metrics_orig.get('cv_quantile_loss', 0)
        
        print(f"Original Q50 loss: {loss_orig:.6f}")
        
        # 2. Test with Winsorized data
        def robust_winsorize(data, k=3.0):
            """Robust winsorize based on MAD"""
            if isinstance(data, pd.Series):
                data = data.values
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return data
            sigma = 1.4826 * mad
            lower = median - k * sigma
            upper = median + k * sigma
            return np.clip(data, lower, upper)
        
        y_winsorized = pd.Series(robust_winsorize(small_y, k=2.5), index=small_y.index)
        model_winsorized = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        metrics_winsorized, _ = model_winsorized.train(small_X, y_winsorized, n_splits=2)
        loss_winsorized = metrics_winsorized.get('cv_quantile_loss', 0)
        
        print(f"Winsorized Q50 loss: {loss_winsorized:.6f}")
        print(f"Improvement: {loss_orig - loss_winsorized:.6f}")
        
        # 3. Test with sample weighting
        # Get initial predictions to identify outliers
        pred_initial = model_orig.model.predict(small_X.values)
        residuals = np.abs(small_y.values - pred_initial)
        
        # Calculate weights
        residual_median = np.median(residuals)
        delta = 2.0 * residual_median if residual_median > 0 else 1.0
        sample_weights = np.where(residuals < delta, 1.0, delta / (residuals + 1e-8))
        sample_weights = sample_weights / np.mean(sample_weights)
        
        model_weighted = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        metrics_weighted, _ = model_weighted.train(small_X, small_y, n_splits=2, sample_weight=sample_weights)
        loss_weighted = metrics_weighted.get('cv_quantile_loss', 0)
        
        print(f"Weighted Q50 loss: {loss_weighted:.6f}")
        print(f"Improvement: {loss_orig - loss_weighted:.6f}")
        
        # 4. Test with increased regularization
        params_regularized = {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.02,
            "lambda_l1": 5.0,
            "lambda_l2": 5.0,
        }
        
        model_regularized = LightGBMModel(model_type="quantile", quantile_alpha=0.5, params=params_regularized)
        metrics_regularized, _ = model_regularized.train(small_X, small_y, n_splits=2)
        loss_regularized = metrics_regularized.get('cv_quantile_loss', 0)
        
        print(f"Regularized Q50 loss: {loss_regularized:.6f}")
        print(f"Improvement: {loss_orig - loss_regularized:.6f}")
        
        # Summary
        print(f"\nSummary of improvements:")
        print(f"  Winsorization: {loss_orig - loss_winsorized:.6f}")
        print(f"  Sample weighting: {loss_orig - loss_weighted:.6f}")
        print(f"  Regularization: {loss_orig - loss_regularized:.6f}")
        
        best_improvement = max(loss_orig - loss_winsorized, loss_orig - loss_weighted, loss_orig - loss_regularized)
        if best_improvement > 0:
            print(f"✅ Best improvement: {best_improvement:.6f}")
        else:
            print("⚠️  No improvement found with these methods")
            
        return {
            'original': loss_orig,
            'winsorized': loss_winsorized,
            'weighted': loss_weighted,
            'regularized': loss_regularized
        }
        
    def test_cross_quantile_loss_validation(self):
        """Validate cross-quantile loss relationships."""
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        # Train all three quantile models
        models = {}
        losses = {}
        
        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            model = LightGBMModel(model_type="quantile", quantile_alpha=q)
            metrics, _ = model.train(small_X, small_y, n_splits=2)
            models[q] = model
            losses[q] = metrics.get('cv_quantile_loss', 0)
            
        print("=== Cross-Quantile Loss Validation ===")
        
        q10_loss = losses[0.1]
        q50_loss = losses[0.5]
        q90_loss = losses[0.9]
        
        print(f"Q10 loss: {q10_loss:.6f}")
        print(f"Q50 loss: {q50_loss:.6f}")
        print(f"Q90 loss: {q90_loss:.6f}")
        
        # Validate fundamental property: Q50 loss should be <= max(Q10, Q90)
        max_other = max(q10_loss, q90_loss)
        q50_ratio = q50_loss / max_other if max_other > 0 else float('inf')
        
        print(f"\nQ50 loss ratio (Q50/max(Q10,Q90)): {q50_ratio:.2f}")
        
        if q50_loss <= max_other:
            print("✅ Quantile loss property satisfied")
        else:
            print("❌ Quantile loss property VIOLATED")
            print("   This indicates a fundamental problem with the Q50 model")
            
        # Additional checks
        if q50_loss > q10_loss:
            ratio_50_10 = q50_loss / q10_loss
            print(f"⚠️  Q50 loss is {ratio_50_10:.2f}x Q10 loss")
            
        if q50_loss > q90_loss:
            ratio_50_90 = q50_loss / q90_loss
            print(f"⚠️  Q50 loss is {ratio_50_90:.2f}x Q90 loss")
            
        # Check for severe violations
        if q50_ratio > 1.5:
            print("🚨 SEVERE VIOLATION: Q50 loss ratio > 1.5")
            print("   Model should be considered unusable")
        elif q50_ratio > 1.2:
            print("⚠️  MODERATE VIOLATION: Q50 loss ratio > 1.2")
            print("   Model may be unreliable")
        else:
            print("✅ Q50 loss ratio within acceptable range")
            
        return q50_ratio, losses


if __name__ == "__main__":
    # Run specific tests to debug Q50 issues
    test_suite = unittest.TestSuite()
    
    # Add tests in order of importance
    test_suite.addTest(TestQ50DebugDetailed('test_cross_quantile_loss_validation'))
    test_suite.addTest(TestQ50DebugDetailed('test_quantile_specific_behavior'))
    test_suite.addTest(TestQ50DebugDetailed('test_residual_analysis'))
    test_suite.addTest(TestQ50DebugDetailed('test_model_prediction_distributions'))
    test_suite.addTest(TestQ50DebugDetailed('test_extreme_value_handling_mechanisms'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun-len(result.failures)-len(result.errors))/result.testsRun*100:.1f}%" if result.testsRun > 0 else "No tests run")