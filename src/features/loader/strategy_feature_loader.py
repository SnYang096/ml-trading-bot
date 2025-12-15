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
        strategy_config_path: Optional[str] = None,
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
            strategy_config_path: 策略配置文件路径（可选，如果文件不存在则不加载）
            cache_dir: 磁盘缓存目录
            use_disk_cache: 是否使用磁盘缓存
            use_memory_cache: 是否使用内存缓存
            max_workers: 最大并行数
            parallel_backend: 并行后端（process/thread）
        """
        self.feature_deps = self._load_yaml(feature_deps_path)
        # 可选加载策略配置（用于向后兼容）
        if strategy_config_path is not None:
            self.strategy_config = self._load_yaml_optional(strategy_config_path)
        else:
            self.strategy_config = {}
        
        # 创建并行计算器
        self.computer = ParallelFeatureComputer(
            cache_dir=cache_dir,
            use_disk_cache=use_disk_cache,
            use_memory_cache=use_memory_cache,
            max_workers=max_workers,
            parallel_backend=parallel_backend,
        )
    
    def _load_yaml(self, path: str) -> Dict:
        """加载 YAML 配置文件（必需）"""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_yaml_optional(self, path: str) -> Dict:
        """加载 YAML 配置文件（可选，文件不存在时返回空字典）"""
        path_obj = Path(path)
        if not path_obj.exists():
            return {}
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
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
        
        注意：此方法已废弃，推荐使用 load_features_from_requested() 配合目录管理方式。
        此方法仅用于向后兼容，需要 strategy_features.yaml 文件存在。
        
        Args:
            df: 输入 DataFrame
            strategy_name: 策略名称（sr_reversal, sr_breakout, compression_breakout, trend_following）
            fit: 是否拟合（研究阶段=True，实盘阶段=False）
            requested_features_override: 可选的特征列表覆盖
        
        Returns:
            df_with_features: 包含计算特征的 DataFrame
        """
        if not self.strategy_config or "strategies" not in self.strategy_config:
            raise ValueError(
                f"Strategy config not loaded. "
                f"Please use load_features_from_requested() with directory-based configs instead."
            )
        
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
        
        # 调试信息：检查需要 ticks 的特征是否配置了 ticks_loader_json
        features_need_ticks = ["vpin_features", "footprint_basic"]
        for feature_name in requested_features:
            if feature_name in features_need_ticks and feature_name in features:
                compute_params = features[feature_name].get("compute_params", {})
                if "ticks_loader_json" in compute_params:
                    print(f"     ✅ {feature_name} has ticks_loader_json in load_features_from_requested")
                else:
                    print(f"     ⚠️  {feature_name} does NOT have ticks_loader_json in load_features_from_requested")
                    print(f"     compute_params keys: {list(compute_params.keys())}")
                    print(f"     feature_info keys: {list(features[feature_name].keys())}")
        
        # 确保所有请求特征的 required_columns 都在 DataFrame 中
        # 收集所有需要的 required_columns
        all_required_columns = set()
        for feature_name in requested_features:
            if feature_name in features:
                feature_info = features[feature_name]
                required_columns = feature_info.get("required_columns", [])
                all_required_columns.update(required_columns)
        
        # 检查缺失的 required_columns 并尝试从原始 df 中获取
        missing_required = [col for col in all_required_columns if col not in result_df.columns]
        if missing_required:
            for col in missing_required:
                if col in df.columns:
                    # 确保索引对齐
                    try:
                        result_df[col] = df[col].reindex(result_df.index)
                    except Exception:
                        # 如果 reindex 失败，尝试直接赋值（假设索引相同）
                        if len(df) == len(result_df):
                            result_df[col] = df[col].values
                        else:
                            # 如果长度不匹配，尝试使用 loc
                            common_idx = result_df.index.intersection(df.index)
                            if len(common_idx) > 0:
                                result_df.loc[common_idx, col] = df.loc[common_idx, col]
        
        # Store original indices to filter out any new indices introduced during feature computation
        original_indices = set(df.index)
        
        result_df = self.computer.compute_features_parallel(
            result_df,
            features,
            requested_features,
            fit=fit,
        )
        
        # Filter out any indices that were not in the original input DataFrame
        # This prevents feature computation from introducing overlapping indices
        new_indices = set(result_df.index) - original_indices
        if new_indices:
            print(f"     ⚠️  Feature computation introduced {len(new_indices)} new indices, filtering them out")
            if len(new_indices) <= 10:
                print(f"        Examples of new indices: {sorted(list(new_indices))[:5]}")
            result_df = result_df.loc[result_df.index.isin(original_indices)]
        
        # 验证：确保输出索引与输入索引一致
        if not result_df.index.equals(df.index):
            # 尝试重新对齐索引
            result_df = result_df.reindex(df.index)
            print(f"     ℹ️  Reindexed output to match input index (may introduce NaN)")
        
        # 验证：检查数据类型
        for col in result_df.columns:
            try:
                # 检查 col 是否是单个列（Series）还是多个列（DataFrame）
                col_data = result_df[col]
                if isinstance(col_data, pd.DataFrame):
                    # 如果是 DataFrame，跳过（可能是多列特征）
                    continue
                elif isinstance(col_data, pd.Series):
                    # 如果是 Series，检查 dtype
                    if col_data.dtype == 'object':
                        # 检查是否有意外的 object 类型（可能是字符串或其他类型）
                        sample_values = col_data.dropna().head(5)
                        if len(sample_values) > 0:
                            first_val = sample_values.iloc[0]
                            if not isinstance(first_val, (int, float, bool, type(None))):
                                print(f"     ⚠️  Warning: Column '{col}' has unexpected dtype 'object' with sample value: {first_val}")
            except (KeyError, AttributeError, TypeError) as e:
                # 如果无法访问列，跳过
                continue
        
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
        
        注意：此方法已废弃，推荐使用目录管理方式。
        此方法仅用于向后兼容，需要 strategy_features.yaml 文件存在。
        
        Args:
            strategy_name: 策略名称
        
        Returns:
            feature_list: 特征列表（包括依赖）
        """
        if not self.strategy_config or "strategies" not in self.strategy_config:
            raise ValueError(
                f"Strategy config not loaded. "
                f"Please use directory-based configs instead."
            )
        
        if strategy_name not in self.strategy_config.get("strategies", {}):
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        requested_features = self.strategy_config["strategies"][strategy_name].get(
            "requested_features", []
        )
        return self.resolve_dependencies(requested_features)
    
    def clear_cache(self, memory: bool = True, disk: bool = False):
        """清除缓存"""
        self.computer.clear_cache(memory=memory, disk=disk)
