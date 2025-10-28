"""Improved feature engineering module with normalization.

基于基础特征工程，添加了：
1. 特征归一化（StandardScaler/MinMaxScaler/RobustScaler）
2. 额外的衍生特征（动量、移动平均比率等）
3. Scaler保存和加载功能

基础指标计算复用 base_indicators 模块。
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle

from .base_indicators import add_basic_indicators


class ImprovedFeatureEngineer:
    """Improved feature engineer with normalization."""

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the improved feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers = {}  # Store scalers for each timeframe
        self.feature_stats = {}  # Store feature statistics

        # Choose scaler
        if scaler_type == "standard":
            self.scaler_class = StandardScaler
        elif scaler_type == "minmax":
            self.scaler_class = MinMaxScaler
        elif scaler_type == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标（基础 + 改进特征）."""
        if data.empty:
            return data

        # 1. 添加所有基础指标
        df = add_basic_indicators(data)

        if df.empty:
            return df

        # 2. 添加改进版特有的衍生特征

        # Price position within Bollinger Bands
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # RSI normalized position
        df["rsi_normalized"] = (df["rsi"] - 50) / 50  # Center around 0

        # MACD normalized
        df["macd_normalized"] = df["macd"] / df["close"]  # Relative to price

        # ATR normalized
        df["atr_normalized"] = df["atr"] / df["close"]  # Relative to price

        # Price momentum
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)
        df["momentum_20"] = df["close"].pct_change(20)

        # Moving average ratios
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"]
        df["sma_ratio_10_20"] = df["sma_10"] / df["sma_20"]

        # Fill NaN values
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        for col in feature_cols:
            df[col] = df[col].fillna(0)

        return df

    def normalize_features(
        self, data: pd.DataFrame, timeframe: str, fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using the specified scaler.

        Args:
            data: DataFrame with features
            timeframe: Timeframe identifier
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            DataFrame with normalized features
        """
        df = data.copy()

        # Get feature columns (exclude OHLCV)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        if not feature_cols:
            return df

        # Prepare feature matrix
        X = df[feature_cols].values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            # Fit scaler on training data
            scaler = self.scaler_class()
            X_scaled = scaler.fit_transform(X)
            self.scalers[timeframe] = scaler

            # Store feature statistics
            self.feature_stats[timeframe] = {
                "mean": np.mean(X, axis=0),
                "std": np.std(X, axis=0),
                "min": np.min(X, axis=0),
                "max": np.max(X, axis=0),
            }
        else:
            # Use existing scaler for prediction
            if timeframe not in self.scalers:
                raise ValueError(
                    f"No scaler found for timeframe {timeframe}. Call fit first."
                )
            scaler = self.scalers[timeframe]
            X_scaled = scaler.transform(X)

        # Update DataFrame with scaled features
        df_scaled = df.copy()
        for i, col in enumerate(feature_cols):
            df_scaled[col] = X_scaled[:, i]

        return df_scaled

    def engineer_features(
        self, multi_tf_data: Dict[str, pd.DataFrame], fit: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multi-timeframe data with normalization.

        Args:
            multi_tf_data: Dictionary mapping timeframe to DataFrame
            fit: Whether to fit scalers (True for training, False for prediction)

        Returns:
            Dictionary with engineered and normalized features for each timeframe
        """
        engineered_data = {}

        for timeframe, data in multi_tf_data.items():
            print(f"Engineering features for {timeframe}: {data.shape}")

            # Add technical indicators
            df_with_indicators = self.add_technical_indicators(data)
            print(f"Added indicators for {timeframe}: {df_with_indicators.shape}")

            # Normalize features
            df_normalized = self.normalize_features(
                df_with_indicators, timeframe, fit=fit
            )
            print(f"Normalized features for {timeframe}: {df_normalized.shape}")

            engineered_data[timeframe] = df_normalized

        return engineered_data

    def save_scalers(self, filepath: str):
        """Save fitted scalers to file."""
        scaler_data = {
            "scalers": self.scalers,
            "feature_stats": self.feature_stats,
            "scaler_type": self.scaler_type,
        }

        with open(filepath, "wb") as f:
            pickle.dump(scaler_data, f)

        print(f"Scalers saved to {filepath}")

    def load_scalers(self, filepath: str):
        """Load fitted scalers from file."""
        with open(filepath, "rb") as f:
            scaler_data = pickle.load(f)

        self.scalers = scaler_data["scalers"]
        self.feature_stats = scaler_data["feature_stats"]
        self.scaler_type = scaler_data["scaler_type"]

        print(f"Scalers loaded from {filepath}")

    def get_feature_importance_info(self, timeframe: str) -> Dict:
        """Get feature statistics for analysis."""
        if timeframe not in self.feature_stats:
            return {}

        stats = self.feature_stats[timeframe]
        return {
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "min": stats["min"].tolist(),
            "max": stats["max"].tolist(),
            "scaler_type": self.scaler_type,
        }
