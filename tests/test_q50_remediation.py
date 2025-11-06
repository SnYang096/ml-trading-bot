"""Tests for Q50 model remediation strategies."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.models.lightgbm_model import LightGBMModel


class TestQ50Remediation(unittest.TestCase):
    """Test cases for Q50 model remediation strategies."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data that simulates Q50 issues
        np.random.seed(42)
        n_samples = 500
        
        # Create data with issues that cause Q50 loss > Q10/Q90 loss
        # Most values are small, but with some extreme outliers
        base_data = np.random.normal(0, 0.01, n_samples - 20)  # Normal returns
        outliers = np.random.choice([-0.08, 0.08], size=20)   # Extreme outliers
        self.y_return = pd.Series(np.concatenate([base_data, outliers]), name='future_return')
        
        # Create simple features
        self.X_df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
        })
        
    def test_winsorization_effect(self):
        """Test the effect of Winsorization on extreme values."""
        # Original data statistics
        original_std = self.y_return.std()
        original_min = self.y_return.min()
        original_max = self.y_return.max()
        
        print(f"Original data - std: {original_std:.6f}, min: {original_min:.6f}, max: {original_max:.6f}")
        
        # Apply Winsorization (similar to what's done in train.py)
        def robust_winsorize(data, k=3.0):
            """Robust winsorize based on MAD"""
            if isinstance(data, pd.Series):
                data = data.values
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return data
            sigma = 1.4826 * mad  # Convert MAD to approximate std
            lower = median - k * sigma
            upper = median + k * sigma
            return np.clip(data, lower, upper)
        
        # Apply winsorization
        y_winsorized = pd.Series(robust_winsorize(self.y_return, k=2.5), index=self.y_return.index)
        
        # Winsorized data statistics
        winsorized_std = y_winsorized.std()
        winsorized_min = y_winsorized.min()
        winsorized_max = y_winsorized.max()
        
        print(f"Winsorized data - std: {winsorized_std:.6f}, min: {winsorized_min:.6f}, max: {winsorized_max:.6f}")
        
        # Count clipped values
        n_clipped = np.sum(np.abs(y_winsorized - self.y_return) > 1e-10)
        print(f"Number of clipped values: {n_clipped}")
        
        # Train models before and after winsorization
        model_before = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_after = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        
        # Use smaller dataset for speed
        small_X = self.X_df.head(100)
        small_y_original = self.y_return.head(100)
        small_y_winsorized = y_winsorized.head(100)
        
        metrics_before, _ = model_before.train(small_X, small_y_original, n_splits=2)
        metrics_after, _ = model_after.train(small_X, small_y_winsorized, n_splits=2)
        
        loss_before = metrics_before.get('cv_quantile_loss', 0)
        loss_after = metrics_after.get('cv_quantile_loss', 0)
        
        print(f"Q50 loss before winsorization: {loss_before:.6f}")
        print(f"Q50 loss after winsorization: {loss_after:.6f}")
        print(f"Loss improvement: {loss_before - loss_after:.6f}")
        
    def test_sample_weighting_effect(self):
        """Test the effect of sample weighting on Q50 training."""
        # First train a basic model to get residuals
        model_initial = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        metrics_initial, _ = model_initial.train(small_X, small_y, n_splits=2)
        
        # Get initial predictions and residuals
        pred_initial = model_initial.model.predict(small_X.values)
        residuals = small_y.values - pred_initial
        
        # Calculate robust weights using Huber-like weighting
        residual_median = np.median(np.abs(residuals))
        delta = 2.0 * residual_median  # Huber threshold
        
        # Weight: 1.0 for normal residuals, decreasing for extreme residuals
        sample_weights = np.where(
            np.abs(residuals) < delta, 1.0, delta / np.abs(residuals))
        # Normalize weights to have mean=1.0
        sample_weights = sample_weights / np.mean(sample_weights)
        
        print(f"Sample weights - min: {np.min(sample_weights):.4f}, max: {np.max(sample_weights):.4f}, mean: {np.mean(sample_weights):.4f}")
        
        n_low_weight = np.sum(sample_weights < 0.5)
        print(f"Low weight samples (<0.5): {n_low_weight} ({n_low_weight/len(sample_weights)*100:.1f}%)")
        
        # Train model with sample weights
        model_weighted = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        metrics_weighted, _ = model_weighted.train(
            small_X, small_y, n_splits=2, sample_weight=sample_weights)
        
        loss_initial = metrics_initial.get('cv_quantile_loss', 0)
        loss_weighted = metrics_weighted.get('cv_quantile_loss', 0)
        
        print(f"Q50 loss without weighting: {loss_initial:.6f}")
        print(f"Q50 loss with weighting: {loss_weighted:.6f}")
        print(f"Loss improvement: {loss_initial - loss_weighted:.6f}")
        
    def test_range_calibration_effect(self):
        """Test the effect of range calibration."""
        # Train Q50 model
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        pred_q50 = model_q50.model.predict(small_X.values)
        
        # Calculate original coverage
        pred_range_original = np.percentile(pred_q50, 99) - np.percentile(pred_q50, 1)
        true_range = np.percentile(small_y, 99) - np.percentile(small_y, 1)
        coverage_original = pred_range_original / true_range if true_range > 0 else 0.0
        
        print(f"Original prediction range: {pred_range_original:.6f}")
        print(f"True range: {true_range:.6f}")
        print(f"Original coverage: {coverage_original:.2%}")
        
        # Apply range calibration if coverage is too low
        if coverage_original < 0.5 and true_range > 0:
            # Use 1st-99th percentile range
            true_range_99 = np.percentile(small_y.values, 99) - np.percentile(small_y.values, 1)
            pred_range_99 = np.percentile(pred_q50, 99) - np.percentile(pred_q50, 1)
            
            scale_factor = true_range_99 / (pred_range_99 + 1e-8)
            # Limit scale factor to prevent over-amplification
            scale_factor = min(max(scale_factor, 1.0), 3.0)
            print(f"Scale factor: {scale_factor:.2f}")
            
            # Apply calibration: shift to match median, then scale
            pred_median = np.median(pred_q50)
            true_median = np.median(small_y.values)
            pred_q50_calibrated = (pred_q50 - pred_median) * scale_factor + true_median
            
            # Recalculate coverage after calibration
            pred_range_calibrated = np.percentile(pred_q50_calibrated, 99) - np.percentile(pred_q50_calibrated, 1)
            coverage_calibrated = pred_range_calibrated / true_range_99 if true_range_99 > 0 else 0.0
            
            print(f"Calibrated prediction range: {pred_range_calibrated:.6f}")
            print(f"Calibrated coverage: {coverage_calibrated:.2%}")
            print(f"Coverage improvement: {coverage_calibrated - coverage_original:.2%}")
            
            # Calculate pinball loss before and after calibration
            def pinball_loss(y_true, y_pred, tau=0.5):
                """Calculate pinball loss (quantile loss)"""
                resid = y_true - y_pred
                return np.mean(np.where(resid >= 0, tau * resid, (1 - tau) * (-resid)))
            
            loss_before = pinball_loss(small_y.values, pred_q50, 0.5)
            loss_after = pinball_loss(small_y.values, pred_q50_calibrated, 0.5)
            
            print(f"Pinball loss before calibration: {loss_before:.6f}")
            print(f"Pinball loss after calibration: {loss_after:.6f}")
            print(f"Loss improvement: {loss_before - loss_after:.6f}")
            
    def test_model_regularization_effect(self):
        """Test the effect of increased regularization on Q50 model."""
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        # Train with default parameters
        model_default = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        metrics_default, _ = model_default.train(small_X, small_y, n_splits=2)
        loss_default = metrics_default.get('cv_quantile_loss', 0)
        
        print(f"Default parameters Q50 loss: {loss_default:.6f}")
        
        # Train with more regularization (similar to what's done in auto-remediation)
        n_samples = len(small_y)
        if n_samples < 1000:
            min_data_in_leaf = 50
        elif n_samples < 5000:
            min_data_in_leaf = 100
        elif n_samples < 20000:
            min_data_in_leaf = 200
        else:
            min_data_in_leaf = 300
            
        params_regularized = {
            "num_leaves": 63,
            "min_data_in_leaf": min_data_in_leaf,
            "learning_rate": 0.02,
            "lambda_l1": 5.0,
            "lambda_l2": 5.0,
            "feature_fraction": 0.8,
        }
        
        model_regularized = LightGBMModel(
            model_type="quantile", 
            quantile_alpha=0.5,
            params=params_regularized
        )
        
        metrics_regularized, _ = model_regularized.train(small_X, small_y, n_splits=2)
        loss_regularized = metrics_regularized.get('cv_quantile_loss', 0)
        
        print(f"Regularized parameters Q50 loss: {loss_regularized:.6f}")
        print(f"Loss improvement: {loss_default - loss_regularized:.6f}")
        
        # Check if regularization helped
        if loss_regularized < loss_default:
            print("✅ Regularization improved Q50 loss")
        else:
            print("⚠️ Regularization did not improve Q50 loss")
            
    def test_quantile_loss_property_validation(self):
        """Test validation of quantile loss properties (Q50 <= max(Q10, Q90))."""
        # Train all three quantile models
        model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)
        
        small_X = self.X_df.head(100)
        small_y = self.y_return.head(100)
        
        metrics_q10, _ = model_q10.train(small_X, small_y, n_splits=2)
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        metrics_q90, _ = model_q90.train(small_X, small_y, n_splits=2)
        
        # Extract losses
        q10_loss = metrics_q10.get('cv_quantile_loss', 0)
        q50_loss = metrics_q50.get('cv_quantile_loss', 0)
        q90_loss = metrics_q90.get('cv_quantile_loss', 0)
        
        print(f"Q10 loss: {q10_loss:.6f}")
        print(f"Q50 loss: {q50_loss:.6f}")
        print(f"Q90 loss: {q90_loss:.6f}")
        
        # Check quantile loss property
        max_other_loss = max(q10_loss, q90_loss) if (q10_loss > 0 and q90_loss > 0) else 1.0
        q50_loss_ratio = q50_loss / max_other_loss if max_other_loss > 0 else float('inf')
        
        print(f"Q50 loss ratio: {q50_loss_ratio:.2f}")
        
        # Validate property
        if q50_loss <= max_other_loss:
            print("✅ Quantile loss property satisfied: Q50 loss <= max(Q10, Q90) loss")
        else:
            print("❌ Quantile loss property violated: Q50 loss > max(Q10, Q90) loss")
            print(f"   This indicates a problem with the Q50 model!")
            
        # Calculate loss ratios
        ratio_q10 = q50_loss / q10_loss if q10_loss > 0 else float('inf')
        ratio_q90 = q50_loss / q90_loss if q90_loss > 0 else float('inf')
        
        print(f"Q50/Q10 loss ratio: {ratio_q10:.2f}")
        print(f"Q50/Q90 loss ratio: {ratio_q90:.2f}")
        
        # Check for severe violations
        if q50_loss_ratio > 1.5:
            print("⚠️ Severe violation: Q50 loss ratio > 1.5")
        elif q50_loss_ratio > 1.2:
            print("⚠️ Moderate violation: Q50 loss ratio > 1.2")
        else:
            print("✅ Q50 loss ratio within acceptable range")


if __name__ == "__main__":
    unittest.main()