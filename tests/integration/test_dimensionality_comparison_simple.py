"""Simple integration test for dimensionality comparison that can run without full data.

This test verifies the basic functionality without requiring large datasets.
Note: sys.path setup is handled in tests/conftest.py.
"""

import pytest
import numpy as np

# Import after sys.path setup (handled by conftest.py)
from src.time_series_model.pipeline.dimensionality.dimensionality_comparison import (
    sanitize_features,
    train_lightgbm_model_simple,
    evaluate_model_simple,
)


def test_sanitize_features_integration():
    """Test sanitize_features with realistic feature data (numpy array)."""
    np.random.seed(42)

    # Create realistic feature matrix (n_samples x n_features)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)

    # Add some extreme values to test clipping
    X[0, 0] = 100.0  # Extreme positive value
    X[1, 0] = -100.0  # Extreme negative value
    X[2, 0] = np.nan  # NaN value

    sanitized = sanitize_features(X, clip_std=5.0)

    # Should return numpy array of same shape
    assert isinstance(sanitized, np.ndarray)
    assert sanitized.shape == X.shape
    assert not np.any(np.isnan(sanitized))  # NaNs should be handled
    assert np.all(np.isfinite(sanitized))  # All values should be finite

    # Extreme values should be clipped
    assert np.abs(sanitized[0, 0]) < np.abs(X[0, 0])
    assert np.abs(sanitized[1, 0]) < np.abs(X[1, 0])

    print(f"✅ Sanitized {n_samples}x{n_features} feature matrix")
    print(f"   Original range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Sanitized range: [{sanitized.min():.2f}, {sanitized.max():.2f}]")


def test_train_and_evaluate_integration():
    """Test model training and evaluation with realistic data."""
    np.random.seed(42)

    # Create realistic feature data
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Split into train/val
    split_idx = int(n_samples * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Train model
    model = train_lightgbm_model_simple(X_train, y_train, X_val, y_val, feature_names)

    assert model is not None
    print(f"✅ Model trained with {len(X_train)} train, {len(X_val)} val samples")

    # Evaluate model
    metrics = evaluate_model_simple(model, X_val, y_val)

    # Check metrics (actual keys may vary based on implementation)
    assert "rmse" in metrics or "mse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    rmse = metrics.get("rmse", metrics.get("mse", 0) ** 0.5)
    assert rmse >= 0
    print(
        f"✅ Model evaluated: RMSE={rmse:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
