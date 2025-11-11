"""Simulation tests using real-data patterns that cause Q50 failures."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from time_series_model.models.lightgbm_model import LightGBMModel


class TestQ50RealDataSimulation(unittest.TestCase):
    """Simulation tests using real-data patterns that cause Q50 failures."""

    def create_btc_like_data(self):
        """Create data that simulates BTCUSDT patterns."""
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate BTC-like behavior:
        # 1. Most of the time small fluctuations around trend
        # 2. Occasional large jumps (both directions)
        # 3. Volatility clustering
        
        # Create trend component
        trend = np.linspace(0, 0.5, n_samples)  # Upward trend
        
        # Create volatility clustering (periods of high/low volatility)
        volatility_regimes = np.random.choice([0.005, 0.02], size=n_samples, p=[0.7, 0.3])
        
        # Create base returns with volatility clustering
        base_returns = np.random.normal(0, volatility_regimes)
        
        # Add occasional large jumps (like BTC flash crashes/pumps)
        jump_indices = np.random.choice(n_samples, size=15, replace=False)
        jump_magnitudes = np.random.choice([-0.08, -0.05, 0.06, 0.1], size=15)
        base_returns[jump_indices] += jump_magnitudes
        
        # Combine trend and returns
        y_return = pd.Series(trend + base_returns, name='future_return')
        
        # Create features that have some predictive power
        X_df = pd.DataFrame({
            'rsi': 50 + np.cumsum(base_returns * 10) + np.random.normal(0, 5, n_samples),
            'macd': np.cumsum(base_returns * 5) + np.random.normal(0, 2, n_samples),
            'volatility': volatility_regimes * 100 + np.random.normal(0, 5, n_samples),
            'trend_strength': np.convolve(base_returns, np.ones(20)/20, mode='same') * 100,
            'volume_proxy': np.abs(base_returns) * 1000 + np.random.exponential(100, n_samples),
        })
        
        return X_df, y_return
        
    def create_multi_asset_data(self):
        """Create data that simulates multi-asset training issues."""
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate different assets with different characteristics
        assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Different volatility levels
        volatilities = {'BTCUSDT': 0.04, 'ETHUSDT': 0.05, 'SOLUSDT': 0.07}
        
        # Different trend strengths
        trends = {'BTCUSDT': 0.0002, 'ETHUSDT': 0.0001, 'SOLUSDT': 0.0003}
        
        data_frames = []
        
        for asset in assets:
            n_asset_samples = n_samples // len(assets) + np.random.randint(-50, 50)
            
            # Create asset-specific returns
            asset_returns = np.random.normal(trends[asset], volatilities[asset], n_asset_samples)
            
            # Add asset-specific jumps
            jump_indices = np.random.choice(n_asset_samples, size=max(1, n_asset_samples//100), replace=False)
            jump_magnitudes = np.random.choice([-0.1, -0.05, 0.05, 0.1], size=len(jump_indices))
            asset_returns[jump_indices] += jump_magnitudes
            
            df = pd.DataFrame({
                'future_return': asset_returns,
                'symbol': asset,
                'feature_1': np.random.normal(0, 1, n_asset_samples),
                'feature_2': np.random.normal(0, 1, n_asset_samples),
            })
            
            data_frames.append(df)
            
        # Combine all assets
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Separate features and target
        X_df = combined_df[['feature_1', 'feature_2']]
        y_return = combined_df['future_return']
        
        return X_df, y_return
        
    def test_btc_like_data_q50_behavior(self):
        """Test Q50 behavior with BTC-like data patterns."""
        print("=== BTC-like Data Q50 Behavior Test ===")
        
        X_df, y_return = self.create_btc_like_data()
        
        # Use smaller dataset for speed
        small_X = X_df.head(200)
        small_y = y_return.head(200)
        
        # Train all three quantile models
        models = {}
        losses = {}
        
        for q in [0.1, 0.5, 0.9]:
            model = LightGBMModel(model_type="quantile", quantile_alpha=q)
            metrics, _ = model.train(small_X, small_y, n_splits=2)
            models[q] = model
            losses[q] = metrics.get('cv_quantile_loss', 0)
            
        print(f"BTC-like data results:")
        print(f"  Q10 loss: {losses[0.1]:.6f}")
        print(f"  Q50 loss: {losses[0.5]:.6f}")
        print(f"  Q90 loss: {losses[0.9]:.6f}")
        
        # Check quantile loss property
        max_other = max(losses[0.1], losses[0.9])
        q50_ratio = losses[0.5] / max_other if max_other > 0 else float('inf')
        
        print(f"  Q50 loss ratio: {q50_ratio:.2f}")
        
        if q50_ratio > 1.2:
            print(f"  ❌ Q50 failure detected in BTC-like data")
        else:
            print(f"  ✅ Q50 healthy in BTC-like data")
            
        return q50_ratio, losses
        
    def test_multi_asset_q50_behavior(self):
        """Test Q50 behavior with multi-asset data patterns."""
        print("\n=== Multi-Asset Data Q50 Behavior Test ===")
        
        X_df, y_return = self.create_multi_asset_data()
        
        # Use smaller dataset for speed
        small_X = X_df.head(200)
        small_y = y_return.head(200)
        
        # Train all three quantile models
        models = {}
        losses = {}
        
        for q in [0.1, 0.5, 0.9]:
            model = LightGBMModel(model_type="quantile", quantile_alpha=q)
            metrics, _ = model.train(small_X, small_y, n_splits=2)
            models[q] = model
            losses[q] = metrics.get('cv_quantile_loss', 0)
            
        print(f"Multi-asset data results:")
        print(f"  Q10 loss: {losses[0.1]:.6f}")
        print(f"  Q50 loss: {losses[0.5]:.6f}")
        print(f"  Q90 loss: {losses[0.9]:.6f}")
        
        # Check quantile loss property
        max_other = max(losses[0.1], losses[0.9])
        q50_ratio = losses[0.5] / max_other if max_other > 0 else float('inf')
        
        print(f"  Q50 loss ratio: {q50_ratio:.2f}")
        
        if q50_ratio > 1.2:
            print(f"  ❌ Q50 failure detected in multi-asset data")
        else:
            print(f"  ✅ Q50 healthy in multi-asset data")
            
        return q50_ratio, losses
        
    def test_high_frequency_pattern_q50_behavior(self):
        """Test Q50 behavior with high-frequency data patterns."""
        print("\n=== High-Frequency Pattern Q50 Behavior Test ===")
        
        # Create high-frequency-like data with lots of noise
        np.random.seed(42)
        n_samples = 1000
        
        # High-frequency data characteristics:
        # 1. Very small mean moves
        # 2. High noise-to-signal ratio
        # 3. Frequent small jumps
        # 4. Occasional large outliers
        
        base_returns = np.random.normal(0.0001, 0.01, n_samples)  # Very small mean
        
        # Add microstructure noise
        noise = np.random.normal(0, 0.001, n_samples)
        base_returns += noise
        
        # Add frequent small jumps
        small_jump_indices = np.random.choice(n_samples, size=n_samples//20, replace=False)
        small_jumps = np.random.normal(0, 0.005, len(small_jump_indices))
        base_returns[small_jump_indices] += small_jumps
        
        # Add occasional large outliers
        large_outlier_indices = np.random.choice(n_samples, size=10, replace=False)
        large_outliers = np.random.choice([-0.05, -0.03, 0.03, 0.05], size=10)
        base_returns[large_outlier_indices] += large_outliers
        
        y_return = pd.Series(base_returns, name='future_return')
        
        # Create simple features
        X_df = pd.DataFrame({
            'momentum': np.convolve(base_returns, np.ones(10)/10, mode='same'),
            'volatility': np.convolve(np.abs(base_returns), np.ones(20)/20, mode='same'),
            'noise_feature_1': np.random.normal(0, 1, n_samples),
            'noise_feature_2': np.random.normal(0, 1, n_samples),
        })
        
        # Use smaller dataset for speed
        small_X = X_df.head(200)
        small_y = y_return.head(200)
        
        # Train all three quantile models
        models = {}
        losses = {}
        
        for q in [0.1, 0.5, 0.9]:
            model = LightGBMModel(model_type="quantile", quantile_alpha=q)
            metrics, _ = model.train(small_X, small_y, n_splits=2)
            models[q] = model
            losses[q] = metrics.get('cv_quantile_loss', 0)
            
        print(f"High-frequency data results:")
        print(f"  Q10 loss: {losses[0.1]:.6f}")
        print(f"  Q50 loss: {losses[0.5]:.6f}")
        print(f"  Q90 loss: {losses[0.9]:.6f}")
        
        # Check quantile loss property
        max_other = max(losses[0.1], losses[0.9])
        q50_ratio = losses[0.5] / max_other if max_other > 0 else float('inf')
        
        print(f"  Q50 loss ratio: {q50_ratio:.2f}")
        
        if q50_ratio > 1.2:
            print(f"  ❌ Q50 failure detected in high-frequency data")
        else:
            print(f"  ✅ Q50 healthy in high-frequency data")
            
        return q50_ratio, losses
        
    def test_extreme_outlier_impact(self):
        """Test the specific impact of extreme outliers on Q50."""
        print("\n=== Extreme Outlier Impact Test ===")
        
        # Create data specifically to test outlier impact
        np.random.seed(42)
        n_samples = 500
        
        # 95% normal data
        normal_data = np.random.normal(0, 0.01, int(n_samples * 0.95))
        
        # 5% extreme outliers
        n_extreme = n_samples - len(normal_data)
        extreme_positive = np.array([0.08, 0.1, 0.12])  # Large positive jumps
        extreme_negative = np.array([-0.08, -0.1, -0.12])  # Large negative jumps
        extreme_data = np.concatenate([extreme_positive, extreme_negative])
        
        # Pad to required size
        if len(extreme_data) < n_extreme:
            additional_extreme = np.random.choice(extreme_data, size=n_extreme - len(extreme_data))
            extreme_data = np.concatenate([extreme_data, additional_extreme])
        elif len(extreme_data) > n_extreme:
            extreme_data = extreme_data[:n_extreme]
            
        y_return = pd.Series(np.concatenate([normal_data, extreme_data]), name='future_return')
        
        # Create features
        X_df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
        })
        
        # Use smaller dataset for speed
        small_X = X_df.head(100)
        small_y = y_return.head(100)
        
        # Train Q50 model
        model_q50 = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
        metrics_q50, _ = model_q50.train(small_X, small_y, n_splits=2)
        q50_loss = metrics_q50.get('cv_quantile_loss', 0)
        
        # Get predictions and analyze outlier impact
        pred_q50 = model_q50.model.predict(small_X.values)
        residuals = np.abs(small_y.values - pred_q50)
        
        # Identify outliers in true values
        y_std = np.std(small_y.values)
        outlier_threshold = 3 * y_std
        outlier_mask = np.abs(small_y.values) > outlier_threshold
        
        print(f"Dataset statistics:")
        print(f"  Total samples: {len(small_y)}")
        print(f"  Std deviation: {y_std:.6f}")
        print(f"  Outlier threshold: {outlier_threshold:.6f}")
        print(f"  Outliers detected: {np.sum(outlier_mask)} ({np.sum(outlier_mask)/len(small_y)*100:.1f}%)")
        
        if np.sum(outlier_mask) > 0:
            outlier_residuals = residuals[outlier_mask]
            normal_residuals = residuals[~outlier_mask]
            
            print(f"\nResidual analysis:")
            print(f"  Outlier residuals mean: {np.mean(outlier_residuals):.6f}")
            print(f"  Normal residuals mean: {np.mean(normal_residuals):.6f}")
            
            if np.mean(normal_residuals) > 0:
                outlier_impact_ratio = np.mean(outlier_residuals) / np.mean(normal_residuals)
                print(f"  Outlier impact ratio: {outlier_impact_ratio:.2f}")
                
                if outlier_impact_ratio > 5.0:
                    print(f"  ⚠️  Extreme outliers have {outlier_impact_ratio:.1f}x impact on Q50 loss")
                    print(f"    This is likely causing Q50 failure")
                    
        print(f"\nQ50 loss: {q50_loss:.6f}")
        
        return q50_loss, np.sum(outlier_mask)
        
    def generate_comprehensive_failure_report(self):
        """Generate a comprehensive report of Q50 failure patterns."""
        print("\n" + "="*70)
        print("COMPREHENSIVE Q50 FAILURE ANALYSIS REPORT")
        print("="*70)
        
        # Test different data patterns
        btc_ratio, btc_losses = self.test_btc_like_data_q50_behavior()
        multi_ratio, multi_losses = self.test_multi_asset_q50_behavior()
        hf_ratio, hf_losses = self.test_high_frequency_pattern_q50_behavior()
        q50_loss, outlier_count = self.test_extreme_outlier_impact()
        
        print(f"\nSUMMARY OF Q50 BEHAVIOR ACROSS DIFFERENT DATA PATTERNS:")
        print(f"  BTC-like data Q50 ratio: {btc_ratio:.2f}")
        print(f"  Multi-asset data Q50 ratio: {multi_ratio:.2f}")
        print(f"  High-frequency data Q50 ratio: {hf_ratio:.2f}")
        print(f"  Extreme outliers: {outlier_count} detected")
        
        # Identify which patterns cause failures
        failure_patterns = []
        if btc_ratio > 1.2:
            failure_patterns.append("BTC-like data")
        if multi_ratio > 1.2:
            failure_patterns.append("Multi-asset data")
        if hf_ratio > 1.2:
            failure_patterns.append("High-frequency data")
            
        if failure_patterns:
            print(f"\n❌ FAILURE PATTERNS DETECTED:")
            for pattern in failure_patterns:
                print(f"  - {pattern}")
        else:
            print(f"\n✅ NO CLEAR FAILURE PATTERNS DETECTED")
            
        print(f"\nRECOMMENDED DIAGNOSTIC STEPS:")
        print(f"  1. Check for extreme outliers in training data")
        print(f"  2. Verify quantile loss property: Q50 ≤ max(Q10, Q90)")
        print(f"  3. Analyze residual distributions")
        print(f"  4. Test remediation strategies:")
        print(f"     - Winsorization")
        print(f"     - Sample weighting")
        print(f"     - Increased regularization")
        print(f"     - Range calibration")
        
        return {
            'btc_ratio': btc_ratio,
            'multi_ratio': multi_ratio,
            'hf_ratio': hf_ratio,
            'outlier_count': outlier_count,
            'failure_patterns': failure_patterns
        }


if __name__ == "__main__":
    # Run the comprehensive analysis
    simulator = TestQ50RealDataSimulation()
    simulator.generate_comprehensive_failure_report()