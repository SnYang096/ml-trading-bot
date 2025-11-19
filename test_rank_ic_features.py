#!/usr/bin/env python3
"""
Test script for Rank IC training features.
This script tests all newly implemented features with synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import all new modules
from time_series_model.pipeline.training.label_utils import (
    volatility_normalized_target,
    historical_quantile_label,
    tradable_mask,
    trend_strength_weight,
    compute_momentum,
    smooth_target,
)
from time_series_model.pipeline.training.rank_ic_utils import (
    compute_rank_ic,
    prediction_quantile,
    confidence_score,
    generate_trading_signals,
)
from time_series_model.pipeline.training.rank_ic_trainer import (
    prepare_rank_ic_labels,
    train_rank_ic_model,
    generate_ensemble_signals,
    evaluate_model_performance,
)
from time_series_model.pipeline.training.evaluation_utils import (
    analyze_quantile_distribution,
    compute_confidence_statistics,
    ensure_volatility_feature,
)

print("="*60)
print("🧪 Testing Rank IC Training Features")
print("="*60)

# Generate synthetic data
print("\n1️⃣  Generating synthetic data...")
np.random.seed(42)
n_samples = 1000
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

# Generate price data with trend and noise
price = 100 + np.cumsum(np.random.randn(n_samples) * 0.5) + np.arange(n_samples) * 0.01

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'close': price,
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.randn(n_samples),
})

print(f"   ✅ Generated {len(df)} samples")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# Test 1: Prepare Rank IC labels
print("\n2️⃣  Testing prepare_rank_ic_labels...")
try:
    df_labels = prepare_rank_ic_labels(
        df,
        price_col="close",
        date_col="date",
        hold_period=5,
        lookback_window=60,
        ensure_volatility=True,
    )
    
    print(f"   ✅ Labels prepared successfully")
    print(f"   Columns added: {set(df_labels.columns) - set(df.columns)}")
    print(f"   Valid samples: {df_labels['future_return'].notna().sum()}/{len(df_labels)}")
    print(f"   Tradable samples: {df_labels['tradable'].sum()}/{len(df_labels)}")
    
    # Check if volatility was computed
    if 'rolling_vol' in df_labels.columns:
        print(f"   ✅ Volatility feature exists: mean={df_labels['rolling_vol'].mean():.6f}")
    else:
        print(f"   ⚠️  Volatility feature missing")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Ensure volatility feature
print("\n3️⃣  Testing ensure_volatility_feature...")
try:
    df_test = df.copy()
    df_test = ensure_volatility_feature(
        df_test,
        price_col="close",
        volatility_col="test_vol",
        window=20,
    )
    
    if 'test_vol' in df_test.columns:
        print(f"   ✅ Volatility feature created: mean={df_test['test_vol'].mean():.6f}")
    else:
        print(f"   ❌ Volatility feature not created")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Quantile distribution analysis
print("\n4️⃣  Testing analyze_quantile_distribution...")
try:
    # Use labels from previous step
    quantile_stats = analyze_quantile_distribution(
        df_labels['return_quantile'],
        pred_quantile=None,
    )
    
    if 'return_quantile' in quantile_stats:
        rq = quantile_stats['return_quantile']
        print(f"   ✅ Quantile analysis complete")
        print(f"      Mean: {rq['mean']:.3f}, Std: {rq['std']:.3f}")
        print(f"      Uniformity: {rq['histogram']['uniformity_score']:.3f}")
    else:
        print(f"   ⚠️  No return_quantile stats")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Train Rank IC model (with smaller dataset for speed)
print("\n5️⃣  Testing train_rank_ic_model...")
try:
    # Filter to valid samples
    df_train = df_labels.dropna(subset=['volatility_normalized_target', 'future_return', 'return_quantile']).copy()
    
    if len(df_train) < 100:
        print(f"   ⚠️  Insufficient samples ({len(df_train)}), skipping training test")
    else:
        feature_cols = ['feature1', 'feature2', 'feature3', 'momentum', 'rolling_vol']
        
        # Ensure all feature columns exist
        missing_features = [f for f in feature_cols if f not in df_train.columns]
        if missing_features:
            print(f"   ⚠️  Missing features: {missing_features}, using available features only")
            feature_cols = [f for f in feature_cols if f in df_train.columns]
        
        if len(feature_cols) < 2:
            print(f"   ⚠️  Too few features ({len(feature_cols)}), skipping training")
        else:
            print(f"   Training with {len(df_train)} samples, {len(feature_cols)} features...")
            
            models, avg_ic, results_df = train_rank_ic_model(
                df_train,
                feature_cols=feature_cols,
                target_col="volatility_normalized_target",
                date_col="date",
                n_splits=3,  # Use fewer splits for speed
                use_gpu=False,  # Use CPU for testing
                filter_high_confidence=False,  # Don't filter for testing
            )
            
            print(f"   ✅ Training complete")
            print(f"      Average Rank IC: {avg_ic:.4f}")
            print(f"      Number of models: {len(models)}")
            print(f"      CV results:\n{results_df}")
            
            # Test 5: Generate ensemble signals
            print("\n6️⃣  Testing generate_ensemble_signals...")
            try:
                df_signals = generate_ensemble_signals(
                    df_train,
                    models=models,
                    feature_cols=feature_cols,
                    confidence_threshold=0.85,
                )
                
                print(f"   ✅ Signals generated")
                print(f"      Signal distribution:")
                print(f"         Long: {(df_signals['signal'] == 1).sum()}")
                print(f"         Short: {(df_signals['signal'] == -1).sum()}")
                print(f"         Hold: {(df_signals['signal'] == 0).sum()}")
                
                # Test 6: Evaluate model performance
                print("\n7️⃣  Testing evaluate_model_performance...")
                try:
                    eval_results = evaluate_model_performance(
                        df_signals,
                        signals=df_signals['signal'],
                        confidence_threshold=0.85,
                    )
                    
                    print(f"   ✅ Evaluation complete")
                    if 'quantile_distribution' in eval_results:
                        print(f"      Quantile distribution: ✅")
                    if 'confidence_statistics' in eval_results:
                        print(f"      Confidence statistics: ✅")
                        
                except Exception as e:
                    print(f"   ❌ Evaluation error: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"   ❌ Signal generation error: {e}")
                import traceback
                traceback.print_exc()
                
except Exception as e:
    print(f"   ❌ Training error: {e}")
    import traceback
    traceback.print_exc()

# Test individual utility functions
print("\n8️⃣  Testing individual utility functions...")

# Test compute_rank_ic
try:
    pred = np.random.randn(100)
    true_ret = pred + np.random.randn(100) * 0.1
    ic = compute_rank_ic(pred, true_ret)
    print(f"   ✅ compute_rank_ic: {ic:.4f}")
except Exception as e:
    print(f"   ❌ compute_rank_ic error: {e}")

# Test confidence_score
try:
    pred_quantile = pd.Series(np.random.rand(100))
    conf = confidence_score(pred_quantile)
    print(f"   ✅ confidence_score: mean={conf.mean():.3f}")
except Exception as e:
    print(f"   ❌ confidence_score error: {e}")

# Test smooth_target
try:
    returns = pd.Series(np.random.randn(100))
    smoothed = smooth_target(returns, method="moving_average", window=5)
    print(f"   ✅ smooth_target: mean={smoothed.mean():.3f}")
except Exception as e:
    print(f"   ❌ smooth_target error: {e}")

print("\n" + "="*60)
print("✅ All tests completed!")
print("="*60)

