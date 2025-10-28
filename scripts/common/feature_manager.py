"""Feature Management System for Rolling Training.

特征管理系统，解决新特征无法纳入的问题
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class FeatureRepository:
    """特征存储库管理器"""

    def __init__(self, repo_path: str = "results/feature_repository.json"):
        self.repo_path = repo_path
        self.all_features: Set[str] = set()
        self.feature_importances: Dict[str, float] = {}
        self.feature_frequency: Dict[str, int] = {}
        self.feature_history: List[Dict] = []
        self.last_update: Optional[str] = None

        # 加载已存在的特征库
        self.load_repository()

    def load_repository(self):
        """加载特征库"""
        if os.path.exists(self.repo_path):
            try:
                with open(self.repo_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.all_features = set(data.get("all_features", []))
                    self.feature_importances = data.get("feature_importances", {})
                    self.feature_frequency = data.get("feature_frequency", {})
                    self.feature_history = data.get("feature_history", [])
                    self.last_update = data.get("last_update")
                    print(
                        f"✓ Loaded feature repository: {len(self.all_features)} features"
                    )
            except Exception as e:
                print(f"⚠️  Failed to load repository: {e}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self):
        """初始化空的特征库"""
        self.all_features = set()
        self.feature_importances = {}
        self.feature_frequency = {}
        self.feature_history = []
        self.last_update = None

    def save_repository(self):
        """保存特征库"""
        os.makedirs(os.path.dirname(self.repo_path), exist_ok=True)

        data = {
            "all_features": list(self.all_features),
            "feature_importances": self.feature_importances,
            "feature_frequency": self.feature_frequency,
            "feature_history": self.feature_history,
            "last_update": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(self.repo_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved feature repository: {self.repo_path}")

    def update(self, new_features: List[str], month: str):
        """添加新特征到存储库"""
        old_count = len(self.all_features)
        self.all_features.update(new_features)
        new_count = len(self.all_features)

        # 更新特征频率
        for f in new_features:
            self.feature_frequency[f] = self.feature_frequency.get(f, 0) + 1

        # 初始化新特征的重要性
        for f in new_features:
            if f not in self.feature_importances:
                self.feature_importances[f] = 0.0

        # 记录历史
        self.feature_history.append(
            {
                "month": month,
                "new_features": new_features,
                "total_features": len(self.all_features),
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(
            f"   📊 Feature repository updated: +{new_count - old_count} new features"
        )
        print(f"   📊 Total features: {new_count}")

    def update_importances(
        self, model, current_features: List[str], month: str, decay_factor: float = 0.9
    ):
        """根据模型更新特征重要性"""
        # 获取当前模型的特征重要性
        if hasattr(model, "feature_importance"):
            importances = model.feature_importance(importance_type="gain")
        else:
            importances = model.feature_importances_

        imp_dict = dict(zip(current_features, importances))

        # 应用衰减（避免早期特征永久主导）
        for f in self.feature_importances:
            self.feature_importances[f] *= decay_factor

        # 更新重要性（指数平滑）
        alpha = 0.3  # 学习率
        for f, v in imp_dict.items():
            if f in self.feature_importances:
                self.feature_importances[f] = (1 - alpha) * self.feature_importances[
                    f
                ] + alpha * v
            else:
                self.feature_importances[f] = v

        # 记录更新历史
        self.feature_history.append(
            {
                "month": month,
                "action": "importance_update",
                "updated_features": len(imp_dict),
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(f"   📈 Updated importances for {len(imp_dict)} features")

    def select_features(
        self, threshold: float = 0.8, min_features: int = 50, max_features: int = 150
    ) -> List[str]:
        """动态特征选择"""
        if not self.feature_importances:
            return []

        # 按重要性排序
        sorted_features = sorted(
            self.all_features,
            key=lambda x: self.feature_importances.get(x, 0),
            reverse=True,
        )

        # 累积重要性选择
        total_importance = sum(self.feature_importances.values())
        if total_importance == 0:
            return sorted_features[:min_features]

        cumulative = 0
        selected = []

        for f in sorted_features:
            if f in self.feature_importances:
                cumulative += self.feature_importances[f] / total_importance
                selected.append(f)

                # 达到阈值或最大特征数时停止
                if cumulative >= threshold or len(selected) >= max_features:
                    break

        # 确保最少特征数
        if len(selected) < min_features:
            selected = sorted_features[:min_features]

        print(f"   🎯 Selected {len(selected)} features (threshold: {threshold:.1%})")
        print(f"   🎯 Cumulative importance: {cumulative:.1%}")

        return selected

    def get_feature_stats(self) -> Dict:
        """获取特征统计信息"""
        if not self.feature_importances:
            return {}

        importances = list(self.feature_importances.values())

        return {
            "total_features": len(self.all_features),
            "avg_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "max_importance": np.max(importances),
            "min_importance": np.min(importances),
            "zero_importance_count": sum(1 for v in importances if v == 0),
            "high_importance_count": sum(
                1 for v in importances if v > np.mean(importances) + np.std(importances)
            ),
        }

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """获取Top N特征"""
        if not self.feature_importances:
            return []

        sorted_features = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_features[:n]

    def analyze_feature_categories(self) -> Dict[str, Dict]:
        """分析特征类别分布"""
        categories = {
            "WPT": [f for f in self.all_features if "wpt" in f.lower()],
            "Hurst": [f for f in self.all_features if "hurst" in f.lower()],
            "Hilbert": [f for f in self.all_features if "hilbert" in f.lower()],
            "Spectral": [f for f in self.all_features if "spectral" in f.lower()],
            "OrderFlow": [
                f
                for f in self.all_features
                if any(x in f.lower() for x in ["cvd", "taker", "buy", "sell", "ofi"])
            ],
            "Technical": [
                f
                for f in self.all_features
                if any(
                    x in f.lower() for x in ["rsi", "macd", "bb", "ema", "sma", "atr"]
                )
            ],
            "Volume": [f for f in self.all_features if "volume" in f.lower()],
            "Price": [
                f
                for f in self.all_features
                if any(x in f.lower() for x in ["close", "open", "high", "low"])
            ],
            "Derived": [
                f
                for f in self.all_features
                if any(
                    x in f.lower() for x in ["hl", "hc", "lc", "tr", "return", "change"]
                )
            ],
        }

        analysis = {}
        for category, features in categories.items():
            if features:
                cat_importances = [self.feature_importances.get(f, 0) for f in features]
                analysis[category] = {
                    "count": len(features),
                    "avg_importance": np.mean(cat_importances),
                    "total_importance": np.sum(cat_importances),
                    "top_feature": (
                        max(features, key=lambda x: self.feature_importances.get(x, 0))
                        if cat_importances
                        else None
                    ),
                }

        return analysis


class FeatureManager:
    """特征管理器 - 整合特征工程和特征选择"""

    def __init__(self, repo_path: str = "results/feature_repository.json"):
        self.repo = FeatureRepository(repo_path)
        self.current_features: List[str] = []
        self.selected_features: List[str] = []
        self.feature_engineer = None

    def initialize_features(
        self, df: pd.DataFrame, feature_engineer, month: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """初始化特征工程"""
        print(f"\n🔧 Initializing features for {month}...")

        # 完整特征工程（不筛选）
        df_engineered, self.feature_engineer = feature_engineer(
            df, self.feature_engineer, fit=True
        )

        # 获取所有特征
        from data_utils import get_feature_columns

        self.current_features = get_feature_columns(df_engineered)

        # 更新特征库
        self.repo.update(self.current_features, month)

        # 首次运行，使用所有特征
        self.selected_features = self.current_features

        print(f"   ✓ Engineered {len(self.current_features)} features")
        print(f"   ✓ Using all features (first run)")

        return df_engineered, self.selected_features

    def update_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_engineer,
        month: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """更新特征（滚动训练）"""
        print(f"\n🔄 Updating features for {month}...")

        # 完整特征工程
        train_df_engineered, self.feature_engineer = feature_engineer(
            train_df, self.feature_engineer, fit=True
        )
        test_df_engineered, _ = feature_engineer(
            test_df, self.feature_engineer, fit=False
        )

        # 获取当前特征
        from data_utils import get_feature_columns

        current_features = get_feature_columns(train_df_engineered)

        # 检查新特征
        new_features = [f for f in current_features if f not in self.repo.all_features]
        if new_features:
            print(
                f"   🆕 Found {len(new_features)} new features: {new_features[:5]}..."
            )

        # 更新特征库
        self.repo.update(current_features, month)

        # 动态特征选择
        if len(self.repo.feature_history) > 1:  # 非首次迭代
            self.selected_features = self.repo.select_features(
                threshold=0.8, min_features=50, max_features=150
            )
        else:
            self.selected_features = current_features

        print(f"   ✓ Selected {len(self.selected_features)} features")

        return train_df_engineered, test_df_engineered, self.selected_features

    def update_importances(self, model, month: str):
        """更新特征重要性"""
        if self.selected_features:
            self.repo.update_importances(model, self.selected_features, month)

    def get_feature_report(self) -> Dict:
        """生成特征报告"""
        stats = self.repo.get_feature_stats()
        top_features = self.repo.get_top_features(20)
        categories = self.repo.analyze_feature_categories()

        return {
            "stats": stats,
            "top_features": top_features,
            "categories": categories,
            "selected_count": len(self.selected_features),
            "total_count": len(self.current_features),
        }

    def save_state(self):
        """保存状态"""
        self.repo.save_repository()

    def print_feature_summary(self):
        """打印特征摘要"""
        stats = self.repo.get_feature_stats()
        top_features = self.repo.get_top_features(10)
        categories = self.repo.analyze_feature_categories()

        print(f"\n📊 Feature Repository Summary:")
        print(f"   Total features: {stats.get('total_features', 0)}")
        print(f"   Selected features: {len(self.selected_features)}")
        print(f"   Avg importance: {stats.get('avg_importance', 0):.3f}")

        print(f"\n🏆 Top 10 Features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feature:<40} {importance:>8.2f}")

        print(f"\n📈 Feature Categories:")
        for category, info in categories.items():
            if info["count"] > 0:
                print(
                    f"   {category:<12}: {info['count']:>3} features, "
                    f"avg_imp: {info['avg_importance']:>6.3f}, "
                    f"total: {info['total_importance']:>8.2f}"
                )


def create_feature_manager(
    repo_path: str = "results/feature_repository.json",
) -> FeatureManager:
    """创建特征管理器"""
    return FeatureManager(repo_path)
