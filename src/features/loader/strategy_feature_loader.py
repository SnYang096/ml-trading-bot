"""
基于纯配置文件的特征加载器（支持并行计算和缓存）
"""

import yaml
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

from src.features.loader.parallel_computer import ParallelFeatureComputer


class StrategyFeatureLoader:
    """
    基于纯配置文件的特征加载器（支持并行计算和缓存）
    
    特点：
    1. 特征依赖关系在 YAML 中定义
    2. 策略特征集在 YAML 中定义
    3. 自动解析依赖，按需计算
    4. 支持并行计算和缓存
    """
    
    def __init__(
        self,
        feature_deps_path: str = "config/feature_dependencies.yaml",
        strategy_config_path: str = "config/strategy_features.yaml",
        cache_dir: Optional[str] = "cache/features",
        use_disk_cache: bool = True,
        use_memory_cache: bool = True,
        max_workers: Optional[int] = None,
        parallel_backend: str = "process",
    ):
        """
        初始化特征加载器
        
        Args:
            feature_deps_path: 特征依赖配置文件路径
            strategy_config_path: 策略配置文件路径
            cache_dir: 磁盘缓存目录
            use_disk_cache: 是否使用磁盘缓存
            use_memory_cache: 是否使用内存缓存
            max_workers: 最大并行数
            parallel_backend: 并行后端（process/thread）
        """
        self.feature_deps = self._load_yaml(feature_deps_path)
        self.strategy_config = self._load_yaml(strategy_config_path)
        
        # 创建并行计算器
        self.computer = ParallelFeatureComputer(
            cache_dir=cache_dir,
            use_disk_cache=use_disk_cache,
            use_memory_cache=use_memory_cache,
            max_workers=max_workers,
            parallel_backend=parallel_backend,
        )
    
    def _load_yaml(self, path: str) -> Dict:
        """加载 YAML 配置文件"""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def resolve_dependencies(self, requested_features: List[str]) -> List[str]:
        """
        解析特征依赖关系，返回计算顺序（拓扑排序）
        
        Args:
            requested_features: 请求的特征列表
        
        Returns:
            computation_order: 计算顺序
        """
        features = self.feature_deps.get("features", {})
        
        # 1. 收集所有需要的特征（包括依赖）
        all_needed = set(requested_features)
        queue = list(requested_features)
        
        while queue:
            feature = queue.pop(0)
            if feature in features:
                deps = features[feature].get("dependencies", [])
                for dep in deps:
                    if dep not in all_needed:
                        all_needed.add(dep)
                        queue.append(dep)
        
        # 2. 构建依赖图
        graph = {f: [] for f in all_needed}
        in_degree = {f: 0 for f in all_needed}
        
        for feature in all_needed:
            if feature in features:
                deps = features[feature].get("dependencies", [])
                for dep in deps:
                    if dep in all_needed:
                        graph[dep].append(feature)
                        in_degree[feature] += 1
        
        # 3. 拓扑排序
        queue = [f for f in all_needed if in_degree[f] == 0]
        result = []
        
        while queue:
            feature = queue.pop(0)
            result.append(feature)
            for neighbor in graph[feature]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 4. 检查循环依赖
        if len(result) != len(all_needed):
            raise ValueError("Circular dependency detected!")
        
        return result
    
    def load_strategy_features(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        fit: bool = True,
        requested_features_override: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        为指定策略加载特征（支持并行计算和缓存）
        
        Args:
            df: 输入 DataFrame
            strategy_name: 策略名称（sr_reversal, sr_breakout, compression_breakout, trend_following）
            fit: 是否拟合（研究阶段=True，实盘阶段=False）
        
        Returns:
            df_with_features: 包含计算特征的 DataFrame
        """
        if strategy_name not in self.strategy_config.get("strategies", {}):
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available strategies: {list(self.strategy_config.get('strategies', {}).keys())}"
            )
        
        strategy_config = self.strategy_config["strategies"][strategy_name]
        result_df = df.copy()
        
        requested_features = (
            requested_features_override
            if requested_features_override is not None
            else strategy_config.get("requested_features", [])
        )
        if not requested_features:
            print(f"   ⚠️  No requested features for strategy {strategy_name}")
            return result_df
        
        features = self.feature_deps.get("features", {})
        
        result_df = self.computer.compute_features_parallel(
            result_df,
            features,
            requested_features,
            fit=fit,
        )
        
        # 3. 只返回请求的特征列（以及它们的输出列）
        output_cols = []
        for feature_name in requested_features:
            if feature_name in features:
                feature_info = features[feature_name]
                output_cols.extend(
                    feature_info.get("output_columns", [feature_name])
                )
        
        # 保留原始列和计算的特征列
        all_cols = list(df.columns) + [
            c for c in output_cols if c in result_df.columns
        ]
        return result_df[all_cols]
    
    def load_features_from_requested(
        self,
        df: pd.DataFrame,
        requested_features: Optional[List[str]],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        直接根据请求的特征列表加载特征。
        
        Args:
            df: 输入 DataFrame
            requested_features: 特征列表
            fit: 是否拟合（研究阶段=True，实盘阶段=False）
        """
        result_df = df.copy()
        requested_features = requested_features or []
        if not requested_features:
            return result_df
        
        features = self.feature_deps.get("features", {})
        result_df = self.computer.compute_features_parallel(
            result_df,
            features,
            requested_features,
            fit=fit,
        )
        
        output_cols = []
        for feature_name in requested_features:
            if feature_name in features:
                feature_info = features[feature_name]
                output_cols.extend(feature_info.get("output_columns", [feature_name]))
        
        all_cols = list(df.columns) + [c for c in output_cols if c in result_df.columns]
        return result_df[all_cols]
    
    def get_strategy_features(self, strategy_name: str) -> List[str]:
        """
        获取策略的特征列表（包括依赖）
        
        Args:
            strategy_name: 策略名称
        
        Returns:
            feature_list: 特征列表（包括依赖）
        """
        if strategy_name not in self.strategy_config.get("strategies", {}):
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        requested_features = self.strategy_config["strategies"][strategy_name].get(
            "requested_features", []
        )
        return self.resolve_dependencies(requested_features)
    
    def clear_cache(self, memory: bool = True, disk: bool = False):
        """清除缓存"""
        self.computer.clear_cache(memory=memory, disk=disk)

