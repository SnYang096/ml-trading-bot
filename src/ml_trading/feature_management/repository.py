"""Feature repository and management utilities for rolling workflows."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class FeatureRepository:
    """Persisted repository tracking feature usage and importances."""

    def __init__(self,
                 repo_path: str = "results/feature_repository.json") -> None:
        self.repo_path = repo_path
        self.all_features: Set[str] = set()
        self.feature_importances: Dict[str, float] = {}
        self.feature_frequency: Dict[str, int] = {}
        self.feature_history: List[Dict] = []
        self.last_update: Optional[str] = None
        self.load_repository()

    def load_repository(self) -> None:
        if os.path.exists(self.repo_path):
            try:
                with open(self.repo_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.all_features = set(data.get("all_features", []))
                self.feature_importances = data.get("feature_importances", {})
                self.feature_frequency = data.get("feature_frequency", {})
                self.feature_history = data.get("feature_history", [])
                self.last_update = data.get("last_update")
                print(
                    f"✓ Loaded feature repository: {len(self.all_features)} features"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Failed to load repository: {exc}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self) -> None:
        self.all_features = set()
        self.feature_importances = {}
        self.feature_frequency = {}
        self.feature_history = []
        self.last_update = None

    def save_repository(self) -> None:
        os.makedirs(os.path.dirname(self.repo_path), exist_ok=True)
        data = {
            "all_features": list(self.all_features),
            "feature_importances": self.feature_importances,
            "feature_frequency": self.feature_frequency,
            "feature_history": self.feature_history,
            "last_update": datetime.now().isoformat(),
            "version": "1.0",
        }
        with open(self.repo_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        print(f"✓ Saved feature repository: {self.repo_path}")

    def update(self, new_features: List[str], month: str) -> None:
        old_count = len(self.all_features)
        self.all_features.update(new_features)
        new_count = len(self.all_features)

        for feature in new_features:
            self.feature_frequency[feature] = self.feature_frequency.get(
                feature, 0) + 1
            self.feature_importances.setdefault(feature, 0.0)

        self.feature_history.append({
            "month": month,
            "new_features": new_features,
            "total_features": new_count,
            "timestamp": datetime.now().isoformat(),
        })

        print(
            f"   📊 Feature repository updated: +{new_count - old_count} new features"
        )
        print(f"   📊 Total features: {new_count}")

    def update_importances(
        self,
        model,
        current_features: List[str],
        month: str,
        *,
        decay_factor: float = 0.9,
    ) -> None:
        if hasattr(model, "feature_importance"):
            importances = model.feature_importance(importance_type="gain")
        else:
            importances = model.feature_importances_

        importance_dict = dict(zip(current_features, importances))

        for feature in self.feature_importances:
            self.feature_importances[feature] *= decay_factor

        alpha = 0.3
        for feature, value in importance_dict.items():
            if feature in self.feature_importances:
                self.feature_importances[feature] = (
                    (1 - alpha) * self.feature_importances[feature] +
                    alpha * value)
            else:
                self.feature_importances[feature] = value

        self.feature_history.append({
            "month": month,
            "action": "importance_update",
            "updated_features": len(importance_dict),
            "timestamp": datetime.now().isoformat(),
        })
        print(f"   📈 Updated importances for {len(importance_dict)} features")

    def select_features(
        self,
        *,
        threshold: float = 0.8,
        min_features: int = 50,
        max_features: int = 150,
    ) -> List[str]:
        if not self.feature_importances:
            return []

        sorted_features = sorted(
            self.all_features,
            key=lambda name: self.feature_importances.get(name, 0.0),
            reverse=True,
        )

        total_importance = sum(self.feature_importances.values())
        if total_importance == 0:
            return sorted_features[:min_features]

        cumulative = 0.0
        selected: List[str] = []
        for feature in sorted_features:
            importance = self.feature_importances.get(feature, 0.0)
            cumulative += importance / total_importance
            selected.append(feature)
            if cumulative >= threshold or len(selected) >= max_features:
                break

        if len(selected) < min_features:
            selected = sorted_features[:min_features]

        print(
            f"   🎯 Selected {len(selected)} features (threshold: {threshold:.1%})"
        )
        print(f"   🎯 Cumulative importance: {cumulative:.1%}")
        return selected

    def get_feature_stats(self) -> Dict[str, float]:
        if not self.feature_importances:
            return {}

        importances = list(self.feature_importances.values())
        return {
            "total_features":
            len(self.all_features),
            "avg_importance":
            float(np.mean(importances)),
            "std_importance":
            float(np.std(importances)),
            "max_importance":
            float(np.max(importances)),
            "min_importance":
            float(np.min(importances)),
            "zero_importance_count":
            int(sum(1 for val in importances if val == 0)),
            "high_importance_count":
            int(
                sum(1 for val in importances
                    if val > np.mean(importances) + np.std(importances))),
        }

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        if not self.feature_importances:
            return []
        return sorted(self.feature_importances.items(),
                      key=lambda item: item[1],
                      reverse=True)[:n]

    def analyze_feature_categories(self) -> Dict[str, Dict[str, float]]:
        categories = {
            "WPT":
            [name for name in self.all_features if "wpt" in name.lower()],
            "Hurst":
            [name for name in self.all_features if "hurst" in name.lower()],
            "Hilbert":
            [name for name in self.all_features if "hilbert" in name.lower()],
            "Spectral":
            [name for name in self.all_features if "spectral" in name.lower()],
            "OrderFlow": [
                name for name in self.all_features
                if any(keyword in name.lower()
                       for keyword in ["cvd", "taker", "buy", "sell", "ofi"])
            ],
            "Technical": [
                name for name in self.all_features if any(
                    keyword in name.lower()
                    for keyword in ["rsi", "macd", "bb", "ema", "sma", "atr"])
            ],
            "Volume":
            [name for name in self.all_features if "volume" in name.lower()],
            "Price": [
                name for name in self.all_features
                if any(keyword in name.lower()
                       for keyword in ["close", "open", "high", "low"])
            ],
            "Derived": [
                name for name in self.all_features
                if any(keyword in name.lower() for keyword in
                       ["hl", "hc", "lc", "tr", "return", "change"])
            ],
        }

        analysis: Dict[str, Dict[str, float]] = {}
        for category, features in categories.items():
            if not features:
                continue
            cat_importances = [
                self.feature_importances.get(name, 0.0) for name in features
            ]
            analysis[category] = {
                "count":
                len(features),
                "avg_importance":
                float(np.mean(cat_importances)),
                "total_importance":
                float(np.sum(cat_importances)),
                "top_feature":
                max(
                    features,
                    key=lambda name: self.feature_importances.get(name, 0.0),
                    default=None,
                ),
            }
        return analysis


class FeatureManager:
    """High-level manager that integrates repository, selection, and reporting."""

    def __init__(self,
                 repo_path: str = "results/feature_repository.json") -> None:
        self.repo = FeatureRepository(repo_path)
        self.current_features: List[str] = []
        self.selected_features: List[str] = []
        self.feature_engineer = None

    def initialize_features(
        self,
        df: pd.DataFrame,
        feature_engineer,
        *,
        month: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        print(f"\n🔧 Initializing features for {month}...")

        df_engineered, self.feature_engineer = feature_engineer(
            df, self.feature_engineer, fit=True)

        from ml_trading.data_tools.rolling_data import get_feature_columns

        self.current_features = get_feature_columns(df_engineered)
        self.repo.update(self.current_features, month)
        self.selected_features = self.current_features

        print(f"   ✓ Engineered {len(self.current_features)} features")
        print("   ✓ Using all features (first run)")
        return df_engineered, self.selected_features

    def update_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_engineer,
        *,
        month: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        print(f"\n🔄 Updating features for {month}...")

        train_df_engineered, self.feature_engineer = feature_engineer(
            train_df, self.feature_engineer, fit=True)
        test_df_engineered, _ = feature_engineer(test_df,
                                                 self.feature_engineer,
                                                 fit=False)

        from ml_trading.data_tools.rolling_data import get_feature_columns

        current_features = get_feature_columns(train_df_engineered)

        new_features = [
            f for f in current_features if f not in self.repo.all_features
        ]
        if new_features:
            print(
                f"   🆕 Found {len(new_features)} new features: {new_features[:5]}..."
            )

        self.repo.update(current_features, month)

        if len(self.repo.feature_history) > 1:
            self.selected_features = self.repo.select_features(
                threshold=0.8, min_features=50, max_features=150)
        else:
            self.selected_features = current_features

        print(f"   ✓ Selected {len(self.selected_features)} features")
        return train_df_engineered, test_df_engineered, self.selected_features

    def update_importances(self, model, month: str) -> None:
        if self.selected_features:
            self.repo.update_importances(model, self.selected_features, month)

    def get_feature_report(self) -> Dict[str, object]:
        return {
            "stats": self.repo.get_feature_stats(),
            "top_features": self.repo.get_top_features(20),
            "categories": self.repo.analyze_feature_categories(),
            "selected_count": len(self.selected_features),
            "total_count": len(self.current_features),
        }

    def save_state(self) -> None:
        self.repo.save_repository()

    def print_feature_summary(self) -> None:
        stats = self.repo.get_feature_stats()
        top_features = self.repo.get_top_features(10)
        categories = self.repo.analyze_feature_categories()

        print("\n📊 Feature Repository Summary:")
        print(f"   Total features: {stats.get('total_features', 0)}")
        print(f"   Selected features: {len(self.selected_features)}")
        print(f"   Avg importance: {stats.get('avg_importance', 0):.3f}")

        print("\n🏆 Top 10 Features:")
        for idx, (feature, importance) in enumerate(top_features, 1):
            print(f"   {idx:2d}. {feature:<40} {importance:>8.2f}")

        print("\n📈 Feature Categories:")
        for category, info in categories.items():
            if info["count"] > 0:
                print(f"   {category:<12}: {info['count']:>3} features, "
                      f"avg_imp: {info['avg_importance']:>6.3f}, "
                      f"total: {info['total_importance']:>8.2f}")


def create_feature_manager(
        repo_path: str = "results/feature_repository.json") -> FeatureManager:
    """Helper to match legacy API."""

    return FeatureManager(repo_path)


__all__ = [
    "FeatureRepository",
    "FeatureManager",
    "create_feature_manager",
]
