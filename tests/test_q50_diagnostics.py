"""Tests for Q50 model diagnostics and analysis."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from time_series_model.models.lightgbm_model import LightGBMModel
from time_series_model.pipeline.training.train import _compute_direction_threshold


class TestQ50Diagnostics(unittest.TestCase):
    """Test cases for Q50 model diagnostics."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with known issues that could cause Q50 problems
        np.random.seed(42)  # For reproducible tests
        n_samples = 1000
        
        # Create base data with some normal distribution
        base_returns = np.random.normal(0, 0.02, n_samples)
        
        # Add some extreme outliers (similar to what might cause Q50 issues)
        outlier_indices = np.random.choice(n_samples, size=10, replace=False)
        base_returns[outlier_indices] = np.random.choice([-0.1, 0.1], size=10)
        
        self.y_return = pd.Series(base_returns, name='future_return')
        self.y_vol = pd.Series(np.abs(base_returns) + np.random.normal(0, 0.005, n_samples), name='future_volatility')
        
        # Create simple features
        self.X_df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
        })
        
    def test_q50_loss_ratio_calculation(self):
        """Test Q50 loss ratio calculation to verify the mathematical property."""
        # Train quantile models
        model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)
        
        # Train models with small dataset for speed
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        metrics_q10, _ = model_q10.train(small_X, small_y, n_splits=2)
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        metrics_q90, _ = model_q90.train(small_X, small_y, n_splits=2)
        
        # Extract losses
        q10_loss = metrics_q10.get('cv_quantile_loss', 0)
        q50_loss = metrics_q50.get('cv_quantile_loss', 0)
        q90_loss = metrics_q90.get('cv_quantile_loss', 0)
        
        # Calculate ratio
        max_other_loss = max(q10_loss, q90_loss) if (q10_loss > 0 and q90_loss > 0) else 1.0
        q50_loss_ratio = q50_loss / max_other_loss if max_other_loss > 0 else float('inf')
        
        print(f"Q10 loss: {q10_loss:.6f}")
        print(f"Q50 loss: {q50_loss:.6f}")
        print(f"Q90 loss: {q90_loss:.6f}")
        print(f"Q50 loss ratio: {q50_loss_ratio:.6f}")
        
        # This test might fail with randomly generated data, but it shows the calculation
        # In real scenarios, Q50 loss should be <= max(Q10, Q90) loss
        
    def test_extreme_value_impact_on_q50(self):
        """Test how extreme values affect Q50 model performance."""
        # Create data with extreme outliers
        n_samples = 1000
        normal_data = np.random.normal(0, 0.01, n_samples - 20)  # 98% normal data
        extreme_data = np.random.choice([-0.05, 0.05], size=20)  # 2% extreme values
        y_return_with_outliers = pd.Series(np.concatenate([normal_data, extreme_data]))
        
        # Create corresponding features
        X_with_outliers = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
        })
        
        # Train models
        model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)
        
        # Train with small dataset for speed
        small_X = X_with_outliers.head(100)
        small_y = y_return_with_outliers.head(100)
        
        metrics_q10, _ = model_q10.train(small_X, small_y, n_splits=2)
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        metrics_q90, _ = model_q90.train(small_X, small_y, n_splits=2)
        
        # Extract losses
        q10_loss = metrics_q10.get('cv_quantile_loss', 0)
        q50_loss = metrics_q50.get('cv_quantile_loss', 0)
        q90_loss = metrics_q90.get('cv_quantile_loss', 0)
        
        print(f"With outliers - Q10 loss: {q10_loss:.6f}")
        print(f"With outliers - Q50 loss: {q50_loss:.6f}")
        print(f"With outliers - Q90 loss: {q90_loss:.6f}")
        
        # Calculate residuals to check outlier impact
        pred_q50 = model_q50.model.predict(small_X.values)
        residuals = np.abs(small_y.values - pred_q50)
        
        # Check if extreme values contribute disproportionately to loss
        threshold = np.percentile(residuals, 90)
        extreme_residuals = residuals[residuals > threshold]
        normal_residuals = residuals[residuals <= threshold]
        
        print(f"Extreme residuals mean: {np.mean(extreme_residuals):.6f}")
        print(f"Normal residuals mean: {np.mean(normal_residuals):.6f}")
        
        if len(normal_residuals) > 0:
            outlier_ratio = np.mean(extreme_residuals) / np.mean(normal_residuals)
            print(f"Outlier loss ratio: {outlier_ratio:.2f}")
            
    def test_prediction_range_coverage(self):
        """Test Q50 prediction range coverage."""
        # Train Q50 model
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        
        # Get predictions
        pred_q50 = model_q50.model.predict(small_X.values)
        
        # Calculate range coverage
        pred_range = np.percentile(pred_q50, 99) - np.percentile(pred_q50, 1)
        true_range = np.percentile(small_y, 99) - np.percentile(small_y, 1)
        coverage = pred_range / true_range if true_range > 0 else 0.0
        
        print(f"Prediction range: {pred_range:.6f}")
        print(f"True range: {true_range:.6f}")
        print(f"Coverage: {coverage:.2%}")
        
        # This is a diagnostic test - coverage should ideally be around 1.0
        # Low coverage indicates the model is not capturing the full range of outcomes
        
    def test_pinball_loss_calculation(self):
        """Test manual pinball loss calculation."""
        # Simple test case
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.015, -0.01, 0.02, 0.0, 0.025])
        
        # Manual calculation for Q10 (tau=0.1)
        tau = 0.1
        residuals = y_true - y_pred
        pinball_loss_q10 = np.mean(np.where(residuals >= 0, tau * residuals, (1 - tau) * (-residuals)))
        
        # Manual calculation for Q50 (tau=0.5)
        tau = 0.5
        residuals = y_true - y_pred
        pinball_loss_q50 = np.mean(np.where(residuals >= 0, tau * residuals, (1 - tau) * (-residuals)))
        
        # Manual calculation for Q90 (tau=0.9)
        tau = 0.9
        residuals = y_true - y_pred
        pinball_loss_q90 = np.mean(np.where(residuals >= 0, tau * residuals, (1 - tau) * (-residuals)))
        
        print(f"Manual Q10 pinball loss: {pinball_loss_q10:.6f}")
        print(f"Manual Q50 pinball loss: {pinball_loss_q50:.6f}")
        print(f"Manual Q90 pinball loss: {pinball_loss_q90:.6f}")
        
        # Verify that Q50 loss should be <= max(Q10, Q90) in a well-trained model
        max_other = max(pinball_loss_q10, pinball_loss_q90)
        if max_other > 0:
            ratio = pinball_loss_q50 / max_other
            print(f"Q50 loss ratio: {ratio:.2f}")
            
    def test_direction_threshold_computation(self):
        """Test direction threshold computation methods."""
        # Create sample predictions with bias
        y_score = np.array([-0.001, -0.002, -0.0015, -0.0005, -0.001, -0.002, -0.001, -0.0008])
        y_true_dir = np.array([0, 0, 0, 1, 0, 0, 0, 1])  # Mostly negative class
        
        # Test different threshold methods
        threshold_zero = _compute_direction_threshold(y_score, y_true_dir, method="zero")
        threshold_median = _compute_direction_threshold(y_score, y_true_dir, method="median")
        threshold_f1 = _compute_direction_threshold(y_score, y_true_dir, method="f1_optimize")
        
        print(f"Fixed zero threshold: {threshold_zero:.6f}")
        print(f"Median threshold: {threshold_median:.6f}")
        print(f"F1-optimized threshold: {threshold_f1:.6f}")
        
        # With all negative predictions, F1 will be 0 with fixed threshold
        y_pred_fixed = (y_score > threshold_zero).astype(int)
        y_pred_opt = (y_score > threshold_f1).astype(int)
        
        from sklearn.metrics import f1_score
        f1_fixed = f1_score(y_true_dir, y_pred_fixed, zero_division=0)
        f1_opt = f1_score(y_true_dir, y_pred_opt, zero_division=0)
        
        print(f"F1 score with fixed threshold: {f1_fixed:.3f}")
        print(f"F1 score with optimized threshold: {f1_opt:.3f}")
        
    def test_model_parameter_sensitivity(self):
        """Test how model parameters affect Q50 performance."""
        # Train with default parameters
        model_default = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        metrics_default, _ = model_default.train(small_X, small_y, n_splits=2)
        default_loss = metrics_default.get('cv_quantile_loss', 0)
        
        print(f"Default parameters Q50 loss: {default_loss:.6f}")
        
        # Train with more regularization
        params_regularized = {
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "learning_rate": 0.02,
            "lambda_l1": 5.0,
            "lambda_l2": 5.0,
            "feature_fraction": 0.7,
        }
        
        model_regularized = LightGBMModel(
            model_type="quantile", 
            quantile_alpha=0.5,
            params=params_regularized
        )
        
        metrics_regularized, _ = model_regularized.train(small_X, small_y, n_splits=2)
        regularized_loss = metrics_regularized.get('cv_quantile_loss', 0)
        
        print(f"Regularized parameters Q50 loss: {regularized_loss:.6f}")
        print(f"Loss improvement: {default_loss - regularized_loss:.6f}")


if __name__ == "__main__":
    unittest.main()