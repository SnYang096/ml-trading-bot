"""Unified data loading helpers for dimensionality pipelines."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.features.loader.config_feature_engineer import ConfigFeatureEngineer
from data_tools.data_loader import MarketDataLoader
from time_series_model.utils.sample_data import create_sample_data


class UnifiedDataLoader:
    """Reusable data + feature loader for research and production workflows.

    Responsibilities:
    - Load raw market data (via MarketDataLoader)
    - Run comprehensive feature engineering
    - Provide matrix or DataFrame outputs for pipelines
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path
        self.feature_engineer: Optional[ConfigFeatureEngineer] = None

    def load_real_data(
        self,
        symbol: str = "ETH-USD",
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
        return_dataframe: bool = False,
    ) -> (
        Tuple[np.ndarray, np.ndarray, list]
        | Tuple[np.ndarray, np.ndarray, list, pd.DataFrame]
    ):
        try:
            print(f"📊 Loading real market data for {symbol}...")

            loader = MarketDataLoader(self.data_path)
            df = loader.load_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )

            if df is None or df.empty:
                print("⚠️ No real data found, generating sample data...")
                return self._generate_sample_data(return_dataframe=return_dataframe)

            df = loader.resample_data("5T")
            # Preserve timestamps as a column for downstream date filtering
            df = df.copy()
            df["timestamp"] = df.index

            self.feature_engineer = ConfigFeatureEngineer(strategy_name="sr_breakout")
            df_features = self.feature_engineer.engineer_all_features(df, fit=True)
            # Ensure timestamp survives after feature engineering
            if "timestamp" not in df_features.columns and "timestamp" in df.columns:
                df_features = df_features.copy()
                df_features["timestamp"] = df["timestamp"].values

            feature_cols = [
                col for col in df_features.columns if col not in ["timestamp", "close"]
            ]

            X = df_features[feature_cols].values
            y = df_features["close"].pct_change().shift(-1).dropna().values

            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            df_features = df_features.iloc[:min_len].copy()

            target_column = "target"
            df_features[target_column] = y

            print(f"✅ Real data loaded: {X.shape}, {y.shape}")

            if return_dataframe:
                return X, y, feature_cols, df_features, target_column

            return X, y, feature_cols

        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Error loading real data: {exc}")
            print("📊 Generating sample data...")
            return self._generate_sample_data(return_dataframe=return_dataframe)

    def _generate_sample_data(
        self,
        n_samples: int = 10000,
        n_factors: int = 100,
        return_dataframe: bool = False,
    ) -> (
        Tuple[np.ndarray, np.ndarray, list]
        | Tuple[np.ndarray, np.ndarray, list, pd.DataFrame, str]
    ):
        print(f"📊 Generating sample data: {n_samples} samples, {n_factors} features")

        if not return_dataframe:
            return create_sample_data(
                n_samples=n_samples,
                n_factors=n_factors,
                seed=42,
                return_dataframe=False,
            )

        X, y, factor_names, df = create_sample_data(
            n_samples=n_samples,
            n_factors=n_factors,
            seed=42,
            return_dataframe=True,
        )
        df = df.reset_index(drop=True)

        target_column = "target"
        if "synthetic_future_return" in df.columns:
            df[target_column] = df.pop("synthetic_future_return")
        else:
            df[target_column] = y

        return X, y, factor_names, df, target_column

    def load_quarterly_data(
        self,
        symbol: str = "ETH-USD",
        year: int = 2024,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, list]]:
        quarters = {
            f"{year}_Q1": {"start": f"{year}-01-01", "end": f"{year}-03-31"},
            f"{year}_Q2": {"start": f"{year}-04-01", "end": f"{year}-06-30"},
            f"{year}_Q3": {"start": f"{year}-07-01", "end": f"{year}-09-30"},
            f"{year}_Q4": {"start": f"{year}-10-01", "end": f"{year}-12-31"},
        }

        quarterly_data: Dict[str, Tuple[np.ndarray, np.ndarray, list]] = {}

        for quarter_name in quarters:
            print(f"📊 Loading {quarter_name} data...")
            X, y, feature_names = self._generate_sample_data(
                n_samples=2500,
                n_factors=100,
            )

            quarterly_data[quarter_name] = (X, y, feature_names)
            print(f"✅ {quarter_name} loaded: {X.shape}")

        return quarterly_data

    def create_time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]

        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def get_data_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
    ) -> Dict[str, Any]:
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": feature_names,
            "target_stats": {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
            },
            "feature_stats": {
                "mean": np.mean(X, axis=0),
                "std": np.std(X, axis=0),
            },
        }


__all__ = ["UnifiedFeatureDataLoader"]


# Backward-compatible alias for external imports
class UnifiedFeatureDataLoader(UnifiedDataLoader):
    pass
