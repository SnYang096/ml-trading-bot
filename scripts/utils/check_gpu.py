"""Quick GPU check for LightGBM."""

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

print("=" * 60)
print("🔍 LightGBM GPU Test")
print("=" * 60)
print(f"LightGBM version: {lgb.__version__}")

# Create test data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Test GPU
print("\n🎮 Testing GPU device...")
params_gpu = {
    "device": "gpu",
    "verbosity": 1,  # Show more info
    "objective": "binary",
    "num_leaves": 31,
}

try:
    train_data = lgb.Dataset(X, y)
    print("\n[GPU Test Training...]")
    model = lgb.train(
        params_gpu,
        train_data,
        num_boost_round=10,
    )
    print("\n✅ GPU IS AVAILABLE AND WORKING!")
    print(f"   Model trained with {model.num_trees()} trees")

except Exception as e:
    print(f"\n❌ GPU NOT AVAILABLE: {e}")
    print("\n⚠️  Will fall back to CPU training")

# Test CPU for comparison
print("\n\n🖥️  Testing CPU device...")
params_cpu = {
    "device": "cpu",
    "verbosity": -1,
    "objective": "binary",
}

try:
    model_cpu = lgb.train(params_cpu, train_data, num_boost_round=10)
    print("✅ CPU training works")
except Exception as e:
    print(f"❌ CPU training failed: {e}")

print("\n" + "=" * 60)
