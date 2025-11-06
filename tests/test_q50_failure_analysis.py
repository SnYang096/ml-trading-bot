"""Analysis tests for Q50 failure patterns based on the training report."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.models.lightgbm_model import LightGBMModel


class TestQ50FailureAnalysis(unittest.TestCase):
    """Analysis tests for Q50 failure patterns."""

    def setUp(self):
        """Set up test fixtures based on the failure report."""
        # Create data that simulates the conditions leading to Q50 failure
        np.random.seed(42)
        n_samples = 1000
        
        # Based on the report, we see Q50 loss ratios around 1.7-1.9
        # This suggests the Q50 model is significantly worse than Q10/Q90
        # Let's create data that would cause this
        
        # Create data with characteristics that cause Q50 problems:
        # 1. Most values are small and centered around zero
        # 2. A few extreme outliers that heavily influence Q50 loss
        # 3. This makes Q50 loss much larger than Q10/Q90 losses
        
        base_data = np.random.normal(0, 0.005, n_samples - 50)  # Small returns
        # Add extreme outliers that will heavily impact Q50 (MAE)
        extreme_positive = np.random.uniform(0.03, 0.08, 25)
        extreme_negative = np.random.uniform(-0.08, -0.03, 25)
        extreme_data = np.concatenate([extreme_positive, extreme_negative])
        
        self.y_return = pd.Series(np.concatenate([base_data, extreme_data]), name='future_return')
        
        # Create features with some predictive power but also noise
        self.X_df = pd.DataFrame({
            'trend_feature': np.random.normal(0, 1, n_samples),
            'volatility_feature': np.random.exponential(1, n_samples),
            'momentum_feature': np.random.normal(0, 1, n_samples),
            'noise_feature_1': np.random.normal(0, 1, n_samples),
            'noise_feature_2': np.random.normal(0, 1, n_samples),
        })
        
    def test_replicate_q50_failure_conditions(self):
        """Test to replicate the conditions that cause Q50 failure."""
        # Train all three quantile models
        model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)
        
        # Use smaller dataset for speed but still representative
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        # Train models
        metrics_q10, _ = model_q10.train(small_X, small_y, n_splits=2)
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        metrics_q90, _ = model_q90.train(small_X, small_y, n_splits=2)
        
        # Extract losses
        q10_loss = metrics_q10.get('cv_quantile_loss', 0)
        q50_loss = metrics_q50.get('cv_quantile_loss', 0)
        q90_loss = metrics_q90.get('cv_quantile_loss', 0)
        
        print("=== Q50 Failure Replication Test ===")
        print(f"Q10 loss: {q10_loss:.6f}")
        print(f"Q50 loss: {q50_loss:.6f}")
        print(f"Q90 loss: {q90_loss:.6f}")
        
        # Calculate ratios
        max_other_loss = max(q10_loss, q90_loss) if (q10_loss > 0 and q90_loss > 0) else 1.0
        q50_loss_ratio = q50_loss / max_other_loss if max_other_loss > 0 else float('inf')
        
        print(f"Q50 loss ratio: {q50_loss_ratio:.2f}")
        
        # Analyze the distribution of residuals
        pred_q50 = model_q50.model.predict(small_X.values)
        residuals = np.abs(small_y.values - pred_q50)
        
        # Check outlier impact
        residual_90th = np.percentile(residuals, 90)
        residual_10th = np.percentile(residuals, 10)
        
        high_residuals = residuals[residuals > residual_90th]
        low_residuals = residuals[residuals < residual_10th]
        
        print(f"\nResidual analysis:")
        print(f"Mean residual: {np.mean(residuals):.6f}")
        print(f"Median residual: {np.median(residuals):.6f}")
        print(f"90th percentile residual: {residual_90th:.6f}")
        print(f"10th percentile residual: {residual_10th:.6f}")
        
        if len(low_residuals) > 0:
            outlier_impact_ratio = np.mean(high_residuals) / np.mean(low_residuals)
            print(f"High vs low residual ratio: {outlier_impact_ratio:.2f}")
        
        # Check if we've replicated the failure condition
        if q50_loss_ratio > 1.2:
            print(f"\n⚠️  Replicated Q50 failure: ratio {q50_loss_ratio:.2f} > 1.2")
            if q50_loss_ratio > 1.5:
                print(f"   Severe failure: ratio {q50_loss_ratio:.2f} > 1.5")
        else:
            print(f"\n✅ Q50 model is healthy: ratio {q50_loss_ratio:.2f} <= 1.2")
            
        return q50_loss_ratio
        
    def test_extreme_value_sensitivity(self):
        """Test how sensitive Q50 is to extreme values compared to Q10/Q90."""
        # Create datasets with varying levels of extreme values
        np.random.seed(42)
        n_samples = 500
        
        # Base data
        base_data = np.random.normal(0, 0.01, n_samples)
        
        # Test with different numbers of extreme values
        extreme_counts = [0, 5, 10, 20, 50]
        
        results = []
        
        for ext_count in extreme_counts:
            if ext_count > 0:
                # Add extreme values
                extreme_positive = np.random.uniform(0.05, 0.1, ext_count // 2)
                extreme_negative = np.random.uniform(-0.1, -0.05, ext_count - ext_count // 2)
                extreme_data = np.concatenate([extreme_positive, extreme_negative])
                
                y_data = pd.Series(np.concatenate([base_data[:n_samples-ext_count], extreme_data]), name='future_return')
            else:
                y_data = pd.Series(base_data, name='future_return')
                
            # Create features
            X_data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
            })
            
            # Train models
            model_q10 = LightGBMModel(model_type="quantile", quantile_alpha=0.1)
            model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
            model_q90 = LightGBMModel(model_type="quantile", quantile_alpha=0.9)
            
            # Use smaller dataset
            small_X = X_data.head(100)
            small_y = y_data.head(100)
            
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
            
            results.append({
                'extreme_count': ext_count,
                'q10_loss': q10_loss,
                'q50_loss': q50_loss,
                'q90_loss': q90_loss,
                'q50_loss_ratio': q50_loss_ratio
            })
            
            print(f"Extreme values: {ext_count:2d} | Q10: {q10_loss:.6f} | Q50: {q50_loss:.6f} | Q90: {q90_loss:.6f} | Ratio: {q50_loss_ratio:.2f}")
            
        # Analyze results
        print("\n=== Extreme Value Sensitivity Analysis ===")
        for result in results:
            status = "❌ FAIL" if result['q50_loss_ratio'] > 1.2 else "✅ OK"
            print(f"{status} | {result['extreme_count']:2d} extremes | Ratio: {result['q50_loss_ratio']:.2f}")
            
    def test_prediction_range_analysis(self):
        """Analyze prediction range coverage as mentioned in the failure report."""
        # Train Q50 model
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        pred_q50 = model_q50.model.predict(small_X.values)
        
        # Calculate prediction range coverage (similar to what's done in train.py)
        pred_range = np.percentile(pred_q50, 99) - np.percentile(pred_q50, 1)
        true_range = np.percentile(small_y, 99) - np.percentile(small_y, 1)
        coverage = pred_range / true_range if true_range > 0 else 0.0
        
        print("=== Prediction Range Analysis ===")
        print(f"Predicted range (1st-99th percentile): {pred_range:.6f}")
        print(f"True range (1st-99th percentile): {true_range:.6f}")
        print(f"Coverage ratio: {coverage:.2%}")
        
        # According to the report, coverage < 30% is problematic
        if coverage < 0.3:
            print(f"⚠️  Poor coverage: {coverage:.2%} < 30%")
            print(f"   This indicates the model is not capturing the full range of outcomes")
        elif coverage < 0.5:
            print(f"⚠️  Moderate coverage: {coverage:.2%} < 50%")
        else:
            print(f"✅ Good coverage: {coverage:.2%} >= 50%")
            
        return coverage
        
    def test_outlier_loss_contribution(self):
        """Analyze how much outliers contribute to the total Q50 loss."""
        # Train Q50 model
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        pred_q50 = model_q50.model.predict(small_X.values)
        
        # Calculate pinball loss manually for each quantile
        y_array = small_y.values
        
        # Pinball loss for Q50 (tau=0.5)
        pinball_q50 = np.mean(np.maximum(0.5 * (y_array - pred_q50), 0.5 * (pred_q50 - y_array)))
        
        # Check contribution of extreme values
        threshold = 0.01  # 1% threshold for "extreme" values
        mask_extreme = np.abs(y_array) > threshold
        mask_normal = ~mask_extreme
        
        if np.sum(mask_extreme) > 0:
            pinball_q50_extreme = np.mean(
                np.maximum(0.5 * (y_array[mask_extreme] - pred_q50[mask_extreme]),
                          0.5 * (pred_q50[mask_extreme] - y_array[mask_extreme])))
            pinball_q50_normal = np.mean(
                np.maximum(0.5 * (y_array[mask_normal] - pred_q50[mask_normal]),
                          0.5 * (pred_q50[mask_normal] - y_array[mask_normal])))
            
            print("=== Outlier Loss Contribution Analysis ===")
            print(f"Extreme values threshold: ±{threshold:.4f}")
            print(f"Extreme samples: {np.sum(mask_extreme)} ({np.sum(mask_extreme)/len(y_array)*100:.1f}%)")
            print(f"Normal samples: {np.sum(mask_normal)} ({np.sum(mask_normal)/len(y_array)*100:.1f}%)")
            print(f"Q50 loss (extreme values): {pinball_q50_extreme:.6f}")
            print(f"Q50 loss (normal values): {pinball_q50_normal:.6f}")
            
            if pinball_q50_normal > 0:
                outlier_contribution_ratio = pinball_q50_extreme / pinball_q50_normal
                print(f"Outlier contribution ratio: {outlier_contribution_ratio:.2f}")
                
                if outlier_contribution_ratio > 2.0:
                    print(f"⚠️  Extreme values contribute {outlier_contribution_ratio:.1f}x more to loss than normal values")
                    print(f"   This is likely causing the Q50 failure")
                elif outlier_contribution_ratio > 1.5:
                    print(f"⚠️  Extreme values contribute significantly more to loss")
                else:
                    print(f"✅ Extreme values do not disproportionately contribute to loss")
                    
            return pinball_q50_extreme, pinball_q50_normal
        else:
            print("No extreme values found with threshold ±0.01")
            return 0, pinball_q50
            
    def test_feature_importance_bias(self):
        """Test if feature importance is biased due to extreme values."""
        # Train Q50 model and examine feature importance
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        small_X = self.X_df.head(200)
        small_y = self.y_return.head(200)
        
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        
        # Get feature importance if available
        if hasattr(model_q50.model, 'feature_importance'):
            importance = model_q50.model.feature_importance()
            feature_names = small_X.columns
            
            print("=== Feature Importance Analysis ===")
            for i, (name, imp) in enumerate(zip(feature_names, importance)):
                print(f"{name}: {imp}")
                
            # Check if any feature dominates
            max_importance = np.max(importance)
            total_importance = np.sum(importance)
            
            if total_importance > 0:
                dominant_features = [(name, imp/total_importance) for name, imp in zip(feature_names, importance) if imp/total_importance > 0.5]
                
                if dominant_features:
                    print(f"⚠️  Dominant features (>50% importance): {dominant_features}")
                    print(f"   This could indicate overfitting to specific patterns")
                else:
                    print("✅ No single feature dominates the model")
                    
    def generate_failure_report(self):
        """Generate a comprehensive report of Q50 failure analysis."""
        print("\n" + "="*60)
        print("COMPREHENSIVE Q50 FAILURE ANALYSIS REPORT")
        print("="*60)
        
        # Run all tests and collect results
        failure_ratio = self.test_replicate_q50_failure_conditions()
        coverage = self.test_prediction_range_analysis()
        
        print(f"\nSUMMARY:")
        print(f"  Q50 Loss Ratio: {failure_ratio:.2f}")
        print(f"  Prediction Coverage: {coverage:.2%}")
        
        if failure_ratio > 1.2:
            print(f"  STATUS: ❌ Q50 MODEL FAILURE DETECTED")
            print(f"  RECOMMENDATION: Apply auto-remediation strategies")
        else:
            print(f"  STATUS: ✅ Q50 MODEL IS HEALTHY")
            
        print("\nRECOMMENDED ACTIONS:")
        if failure_ratio > 1.2:
            print("  1. Apply Winsorization to reduce extreme value impact")
            print("  2. Use sample weighting to downweight outlier influence")
            print("  3. Increase model regularization to prevent overfitting")
            print("  4. Consider range calibration if prediction coverage is low")
        print("  5. Validate quantile loss property (Q50 ≤ max(Q10,Q90))")
        print("  6. Monitor outlier contribution to total loss")
        
        return failure_ratio, coverage


if __name__ == "__main__":
    # Run the analysis
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestQ50FailureAnalysis)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
    
    # Generate detailed report
    analyzer = TestQ50FailureAnalysis()
    analyzer.setUp()
    analyzer.generate_failure_report()