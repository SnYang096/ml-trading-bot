"""Integration helpers for deploying dimensionality-reduced models."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_trading.models.autoencoder import UnifiedAutoencoder
from ml_trading.pipeline.dimensionality.dimensionality_comparison import (
    create_enhanced_sample_data, )


class DimensionalityIntegrationEngine:
    """Wrapper combining trained autoencoder and LightGBM model."""

    def __init__(
        self,
        model_path: str,
        autoencoder_path: str,
        results_path: str,
    ) -> None:
        self.model_path = model_path
        self.autoencoder_path = autoencoder_path
        self.results_path = results_path

        with open(results_path, "r") as f:
            self.results = json.load(f)

        self.model = joblib.load(model_path)
        self.autoencoder = self._load_autoencoder(autoencoder_path)

        print("✅ DimensionalityIntegrationEngine initialized")
        print(
            f"📊 Compression ratio: {self.results['data_info']['compression_ratio']:.1f}x"
        )
        print(
            "📈 Performance: R² = "
            f"{self.results['performance']['compressed_features']['r2']:.4f}")

    def _load_autoencoder(self, autoencoder_path: str) -> UnifiedAutoencoder:
        input_dim = self.results["data_info"]["original_features_count"]
        encoding_dim = self.results["data_info"]["compressed_dimensions"]

        autoencoder = UnifiedAutoencoder(
            input_dim,
            encoding_dim,
            architecture="production",
        )
        autoencoder.load_state_dict(
            torch.load(autoencoder_path, map_location="cpu"))
        autoencoder.eval()
        return autoencoder

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            _, X_compressed = self.autoencoder(X_tensor)
            return X_compressed.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_compressed = self.transform_features(X)
        return self.model.predict(X_compressed)

    def get_feature_importance(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importance"):
            importance = self.model.feature_importance(importance_type="gain")
            return {
                f"compressed_feature_{i}": float(importance[i])
                for i in range(len(importance))
            }
        return {}

    def get_model_info(self) -> Dict[str, any]:
        return {
            "compression_ratio":
            self.results["data_info"]["compression_ratio"],
            "original_features":
            self.results["data_info"]["original_features_count"],
            "compressed_features":
            self.results["data_info"]["compressed_dimensions"],
            "performance":
            self.results["performance"]["compressed_features"],
            "training_info":
            self.results["training_info"],
        }


def _find_latest_production_result() -> str | None:
    result_dirs = glob.glob("results/production_dimensionality_*")
    if not result_dirs:
        return None
    return max(result_dirs, key=os.path.getctime)


def integrate_with_existing_training(
) -> Tuple[DimensionalityIntegrationEngine, Dict] | None:
    print("🔗 Integrating dimensionality reduction with existing training...")

    latest_dir = _find_latest_production_result()
    if not latest_dir:
        print(
            "❌ No production dimensionality results found! Run production training first."
        )
        return None

    print(f"📁 Using latest results from: {latest_dir}")

    model_path = os.path.join(latest_dir, "production_model.pkl")
    autoencoder_path = os.path.join(latest_dir, "production_autoencoder.pth")
    results_path = os.path.join(latest_dir, "production_results.json")

    integration_engine = DimensionalityIntegrationEngine(
        model_path,
        autoencoder_path,
        results_path,
    )

    print("\n📊 Loading new market data...")
    X_new, y_new, _ = load_new_market_data()

    print("\n🔮 Making predictions with compressed features...")
    predictions = integration_engine.predict(X_new)

    print("\n📊 Evaluating performance on new data...")
    mse = mean_squared_error(y_new, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_new, predictions)
    r2 = r2_score(y_new, predictions)

    print("📈 Performance on new data:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    integration_report = {
        "integration_timestamp": datetime.now().isoformat(),
        "model_info": integration_engine.get_model_info(),
        "new_data_performance": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        },
        "feature_importance": integration_engine.get_feature_importance(),
    }

    integration_dir = f"results/integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(integration_dir, exist_ok=True)

    with open(os.path.join(integration_dir, "integration_report.json"),
              "w") as f:
        json.dump(integration_report, f, indent=2, default=str)

    print(f"✅ Integration report saved to: {integration_dir}")

    return integration_engine, integration_report


def load_new_market_data() -> Tuple[np.ndarray, np.ndarray, list]:
    print("📊 Loading new market data...")

    np.random.seed(123)
    n_samples = 5000
    X, y, feature_names = create_enhanced_sample_data(n_samples=n_samples,
                                                      n_factors=100)
    return X, y, feature_names


def demonstrate_production_usage() -> None:
    print("🏭 Demonstrating production usage...")

    integration = integrate_with_existing_training()
    if integration is None:
        return

    integration_engine, _ = integration

    print("\n⚡ Simulating real-time predictions...")
    for i in range(5):
        X_realtime = np.random.randn(1, 100)
        prediction = integration_engine.predict(X_realtime)
        print(f"  Prediction {i+1}: {prediction[0]:.6f}")

    print("\n📊 Model information:")
    model_info = integration_engine.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("\n🎉 Production integration demonstration complete!")


def main() -> None:
    print("🚀 Dimensionality Reduction Integration Demo")
    print("=" * 60)

    try:
        demonstrate_production_usage()
        print("\n✅ Integration demo completed successfully!")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Integration demo failed: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
