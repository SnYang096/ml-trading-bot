"""
Advanced Interpretable Factor Engine using Autoencoder + SHAP Distillation
Based on the latest research in explainable AI for quantitative finance.

This module implements a state-of-the-art dimensionality reduction pipeline that combines:
1. Autoencoder for nonlinear factor compression
2. SHAP for interpretability distillation
3. LightGBM for prediction with explainability
4. Factor contribution analysis and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import joblib
import warnings

warnings.filterwarnings("ignore")


class Autoencoder(nn.Module):
    """
    Advanced Autoencoder for factor dimensionality reduction.

    Architecture:
    - Encoder: Input -> 32 -> 16 -> encoding_dim (bottleneck)
    - Decoder: encoding_dim -> 16 -> 32 -> Input
    - Uses ReLU activation and dropout for regularization
    """

    def __init__(
        self, input_dim: int, encoding_dim: int = 8, dropout_rate: float = 0.1
    ):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),  # Bottleneck
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x):
        """Extract compressed representations (embeddings)."""
        with torch.no_grad():
            return self.encoder(x)


class InterpretableFactorEngine:
    """
    Advanced Interpretable Factor Engine implementing Autoencoder + SHAP distillation.

    This engine provides:
    1. Nonlinear factor compression via Autoencoder
    2. Interpretable factor contributions via SHAP distillation
    3. Semantic factor combinations with weights
    4. Real-time factor monitoring and visualization
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        autoencoder_lr: float = 0.001,
        autoencoder_epochs: int = 100,
        dropout_rate: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Initialize the Interpretable Factor Engine.

        Args:
            encoding_dim: Dimension of compressed factor space
            autoencoder_lr: Learning rate for autoencoder training
            autoencoder_epochs: Number of training epochs
            dropout_rate: Dropout rate for regularization
            device: Device for PyTorch (auto-detect if None)
        """
        self.encoding_dim = encoding_dim
        self.autoencoder_lr = autoencoder_lr
        self.autoencoder_epochs = autoencoder_epochs
        self.dropout_rate = dropout_rate

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.autoencoder = None
        self.lgb_model = None
        self.shap_explainer = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # Storage for results
        self.factor_contributions = None
        self.top_factors = None
        self.factor_weights = None
        self.embeddings = None

        print(f"🚀 InterpretableFactorEngine initialized on {self.device}")

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray, factor_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and standardize data for training.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            factor_names: List of factor names

        Returns:
            Standardized X and y
        """
        print(f"📊 Preparing data: {X.shape[0]} samples, {X.shape[1]} factors")

        # Store factor names
        self.factor_names = factor_names

        # Standardize features
        X_scaled = self.scaler_X.fit_transform(X)

        # Standardize target
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        print(f"✅ Data standardized and ready")
        return X_scaled, y_scaled

    def train_autoencoder(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Train the autoencoder for factor compression.

        Args:
            X_scaled: Standardized feature matrix

        Returns:
            Compressed embeddings
        """
        print(
            f"🧠 Training Autoencoder: {X_scaled.shape[1]} -> {self.encoding_dim} dimensions"
        )

        # Initialize autoencoder
        self.autoencoder = Autoencoder(
            input_dim=X_scaled.shape[1],
            encoding_dim=self.encoding_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.autoencoder_lr)

        # Training loop
        self.autoencoder.train()
        for epoch in range(self.autoencoder_epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(
                    f"  Epoch {epoch+1:3d}/{self.autoencoder_epochs}: Loss = {avg_loss:.6f}"
                )

        # Extract embeddings
        self.autoencoder.eval()
        with torch.no_grad():
            _, embeddings_tensor = self.autoencoder(X_tensor)
            embeddings = embeddings_tensor.cpu().numpy()

        self.embeddings = embeddings
        print(f"✅ Autoencoder training complete. Embeddings shape: {embeddings.shape}")
        return embeddings

    def train_lightgbm(
        self, embeddings: np.ndarray, y_scaled: np.ndarray
    ) -> lgb.Booster:
        """
        Train LightGBM model on compressed embeddings.

        Args:
            embeddings: Compressed factor representations
            y_scaled: Standardized target variable

        Returns:
            Trained LightGBM model
        """
        print(f"🌲 Training LightGBM on {embeddings.shape[1]}-dimensional embeddings")

        # Split data (time series aware)
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, y_scaled, test_size=0.2, shuffle=False
        )

        # LightGBM parameters
        params = {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
        }

        # Training
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        self.lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_eval],
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

        print(f"✅ LightGBM training complete")
        return self.lgb_model

    def compute_shap_distillation(
        self, X_scaled: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute SHAP values and distill factor contributions.

        Args:
            X_scaled: Original standardized features
            embeddings: Compressed embeddings

        Returns:
            Factor contribution scores
        """
        print(f"🔍 Computing SHAP distillation for factor interpretability")

        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.lgb_model)
        shap_values = self.shap_explainer.shap_values(embeddings)

        print(f"  SHAP values shape: {shap_values.shape}")

        # Distill factor contributions
        factor_contributions = np.zeros(len(self.factor_names))

        for dim in range(self.encoding_dim):
            # Target: SHAP contribution for this embedding dimension
            target_shap = shap_values[:, dim]

            # Features: Original factors
            # Use corresponding validation set portion
            val_size = int(0.2 * len(X_scaled))
            X_val = X_scaled[-val_size:]
            target_shap_val = target_shap[-val_size:]

            # Train proxy model: original factors -> SHAP contribution
            proxy_model = Ridge(alpha=1.0).fit(X_val, target_shap_val)

            # Accumulate factor contributions
            factor_contributions += np.abs(proxy_model.coef_)

        self.factor_contributions = factor_contributions
        print(f"✅ SHAP distillation complete. Top 5 factors:")

        # Show top factors
        contrib_df = pd.DataFrame(
            {"factor": self.factor_names, "contribution": factor_contributions}
        ).sort_values("contribution", ascending=False)

        for i, (_, row) in enumerate(contrib_df.head(5).iterrows()):
            print(f"  {i+1}. {row['factor']}: {row['contribution']:.4f}")

        return factor_contributions

    def generate_semantic_factors(
        self, top_k: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generate semantic factor combinations with interpretable weights.

        Args:
            top_k: Number of top factors to select

        Returns:
            Top factor names and normalized weights
        """
        print(f"🧩 Generating semantic factor combinations (top {top_k})")

        # Create contribution dataframe
        contrib_df = pd.DataFrame(
            {"factor": self.factor_names, "contribution": self.factor_contributions}
        ).sort_values("contribution", ascending=False)

        # Select top factors
        top_factors = contrib_df.head(top_k)["factor"].values
        weights = contrib_df.head(top_k)["contribution"].values

        # Normalize weights
        weights_normalized = weights / weights.sum()

        self.top_factors = top_factors
        self.factor_weights = weights_normalized

        print(f"✅ Semantic factors generated:")
        for i, (factor, weight) in enumerate(zip(top_factors, weights_normalized)):
            print(f"  {i+1}. {factor}: {weight:.3f}")

        return top_factors, weights_normalized

    def fit(
        self, X: np.ndarray, y: np.ndarray, factor_names: List[str], top_k: int = 10
    ) -> "InterpretableFactorEngine":
        """
        Complete training pipeline.

        Args:
            X: Feature matrix
            y: Target variable
            factor_names: List of factor names
            top_k: Number of top factors to select

        Returns:
            Self (for method chaining)
        """
        print("🚀 Starting InterpretableFactorEngine training pipeline")
        print("=" * 60)

        # Step 1: Prepare data
        X_scaled, y_scaled = self.prepare_data(X, y, factor_names)

        # Step 2: Train autoencoder
        embeddings = self.train_autoencoder(X_scaled)

        # Step 3: Train LightGBM
        self.train_lightgbm(embeddings, y_scaled)

        # Step 4: Compute SHAP distillation
        self.compute_shap_distillation(X_scaled, embeddings)

        # Step 5: Generate semantic factors
        self.generate_semantic_factors(top_k)

        print("=" * 60)
        print("🎉 InterpretableFactorEngine training complete!")
        print(
            f"📊 Compressed {len(factor_names)} factors into {self.encoding_dim} dimensions"
        )
        print(
            f"🎯 Selected {len(self.top_factors)} top factors for interpretable trading"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained pipeline.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.autoencoder is None or self.lgb_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        # Standardize features
        X_scaled = self.scaler_X.transform(X)

        # Get embeddings
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            embeddings = self.autoencoder.encode(X_tensor).cpu().numpy()

        # Make predictions
        predictions = self.lgb_model.predict(embeddings)

        # Inverse transform predictions
        predictions_original = self.scaler_y.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

        return predictions_original

    def get_interpretable_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Generate interpretable trading signals using top factors.

        Args:
            X: Feature matrix

        Returns:
            Interpretable trading signals
        """
        if self.top_factors is None or self.factor_weights is None:
            raise ValueError("Semantic factors not generated. Call fit() first.")

        # Get top factors from input
        factor_indices = [
            self.factor_names.index(factor) for factor in self.top_factors
        ]
        top_factor_values = X[:, factor_indices]

        # Compute weighted signal
        signal = np.dot(top_factor_values, self.factor_weights)

        return signal

    def visualize_factor_contributions(
        self, save_path: Optional[str] = None, top_k: int = 15
    ) -> None:
        """
        Visualize factor contributions and importance.

        Args:
            save_path: Path to save the plot
            top_k: Number of top factors to show
        """
        if self.factor_contributions is None:
            raise ValueError("Factor contributions not computed. Call fit() first.")

        # Create contribution dataframe
        contrib_df = pd.DataFrame(
            {"factor": self.factor_names, "contribution": self.factor_contributions}
        ).sort_values("contribution", ascending=True)

        # Plot
        plt.figure(figsize=(12, 8))
        top_contrib = contrib_df.tail(top_k)

        bars = plt.barh(
            range(len(top_contrib)),
            top_contrib["contribution"],
            color="steelblue",
            alpha=0.7,
        )

        plt.yticks(range(len(top_contrib)), top_contrib["factor"])
        plt.xlabel("SHAP Contribution Strength (Distilled)")
        plt.title("Autoencoder + SHAP Distillation: Core Driving Factors")
        plt.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(
                width + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Factor contributions plot saved to: {save_path}")

        plt.show()

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and components.

        Args:
            filepath: Path to save the model
        """
        model_data = {
            "autoencoder_state": (
                self.autoencoder.state_dict() if self.autoencoder else None
            ),
            "lgb_model": self.lgb_model,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "factor_names": self.factor_names,
            "factor_contributions": self.factor_contributions,
            "top_factors": self.top_factors,
            "factor_weights": self.factor_weights,
            "encoding_dim": self.encoding_dim,
            "device": str(self.device),
        }

        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and components.

        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)

        # Load components
        self.scaler_X = model_data["scaler_X"]
        self.scaler_y = model_data["scaler_y"]
        self.factor_names = model_data["factor_names"]
        self.factor_contributions = model_data["factor_contributions"]
        self.top_factors = model_data["top_factors"]
        self.factor_weights = model_data["factor_weights"]
        self.encoding_dim = model_data["encoding_dim"]

        # Recreate autoencoder
        if model_data["autoencoder_state"] is not None:
            self.autoencoder = Autoencoder(
                input_dim=len(self.factor_names), encoding_dim=self.encoding_dim
            ).to(self.device)
            self.autoencoder.load_state_dict(model_data["autoencoder_state"])

        self.lgb_model = model_data["lgb_model"]

        print(f"📂 Model loaded from: {filepath}")
        print(f"🎯 Ready for inference with {len(self.top_factors)} semantic factors")


def create_sample_data(
    n_samples: int = 1000, n_factors: int = 60
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sample data for testing the InterpretableFactorEngine.

    Args:
        n_samples: Number of samples
        n_factors: Number of factors

    Returns:
        X, y, factor_names
    """
    np.random.seed(42)

    # Generate factor names
    factor_names = []
    categories = ["momentum", "volatility", "mean_reversion", "trend", "volume"]
    for i in range(n_factors):
        category = categories[i % len(categories)]
        factor_names.append(f"{category}_{i+1}")

    # Generate synthetic factors with some structure
    X = np.random.randn(n_samples, n_factors)

    # Create some meaningful relationships
    momentum_factors = [i for i, name in enumerate(factor_names) if "momentum" in name]
    volatility_factors = [
        i for i, name in enumerate(factor_names) if "volatility" in name
    ]

    # Create target with some signal
    y = (
        X[:, momentum_factors].mean(axis=1) * 0.3
        + X[:, volatility_factors].mean(axis=1) * -0.2
        + np.random.randn(n_samples) * 0.1
    )

    return X, y, factor_names


if __name__ == "__main__":
    # Example usage
    print("🚀 Testing InterpretableFactorEngine")

    # Create sample data
    X, y, factor_names = create_sample_data(n_samples=1000, n_factors=60)

    # Initialize and train engine
    engine = InterpretableFactorEngine(encoding_dim=8)
    engine.fit(X, y, factor_names, top_k=10)

    # Generate interpretable signals
    signals = engine.get_interpretable_signal(X)
    print(
        f"📊 Generated interpretable signals: mean={signals.mean():.3f}, std={signals.std():.3f}"
    )

    # Visualize results
    engine.visualize_factor_contributions()

    print("✅ Test complete!")
