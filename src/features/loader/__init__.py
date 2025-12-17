"""
特征加载器模块

提供基于配置文件的特征加载、并行计算和缓存功能
"""

from src.features.registry import (
    get_compute_func,
    get_feature_func,
    ensure_features_registered,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.features.loader.parallel_computer import (
    ParallelFeatureComputer,
    analyze_dependency_levels,
)

__all__ = [
    "get_compute_func",
    "get_feature_func",
    "ensure_features_registered",
    "StrategyFeatureLoader",
    "ParallelFeatureComputer",
    "analyze_dependency_levels",
]
