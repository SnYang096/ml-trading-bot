"""
并行特征计算器

支持：
1. 按依赖层级并行计算
2. 内存缓存
3. 磁盘缓存
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np

from src.features.loader.feature_function_mapping import get_compute_func


def analyze_dependency_levels(
    features: Dict, requested_features: List[str]
) -> Dict[int, List[str]]:
    """
    分析特征依赖层级
    
    Args:
        features: 特征配置字典
        requested_features: 请求的特征列表
    
    Returns:
        levels: {level: [feature_names]}
    """
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
    
    # 2. 计算每个特征的层级
    feature_levels = {}
    
    def get_level(feature_name: str) -> int:
        if feature_name in feature_levels:
            return feature_levels[feature_name]
        
        if feature_name not in features:
            return 0
        
        deps = features[feature_name].get("dependencies", [])
        if not deps:
            level = 0
        else:
            level = max([get_level(dep) for dep in deps]) + 1
        
        feature_levels[feature_name] = level
        return level
    
    # 3. 按层级分组
    levels = {}
    for feature in all_needed:
        level = get_level(feature)
        if level not in levels:
            levels[level] = []
        levels[level].append(feature)
    
    return levels


def _compute_single_feature_worker(
    feature_name: str,
    feature_info: Dict,
    df_bytes: bytes,
    fit: bool,
    cache_key: Optional[str],
    cache_dir: Optional[str],
) -> Tuple[str, bytes, Optional[str]]:
    """
    工作进程函数：计算单个特征
    
    Args:
        feature_name: 特征名
        feature_info: 特征配置信息
        df_bytes: DataFrame 的 pickle 字节
        fit: 是否拟合
        cache_key: 缓存键
        cache_dir: 缓存目录
    
    Returns:
        (feature_name, result_df_bytes, cache_key)
    """
    import pandas as pd
    from src.features.loader.feature_function_mapping import get_compute_func
    
    # 反序列化 DataFrame
    df = pickle.loads(df_bytes)
    
    # 检查磁盘缓存
    if cache_key and cache_dir:
        cache_file = Path(cache_dir) / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_df = pickle.load(f)
                    return (feature_name, pickle.dumps(cached_df), cache_key)
            except Exception:
                pass
    
    # 计算特征
    try:
        compute_func_name = feature_info["compute_func"]
        compute_func = get_compute_func(compute_func_name)
        compute_params = feature_info.get("compute_params", {})
        
        if compute_params:
            result_df = compute_func(df, **compute_params)
        else:
            result_df = compute_func(df)
        
        # 保存磁盘缓存
        if cache_key and cache_dir:
            cache_file = Path(cache_dir) / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result_df, f)
            except Exception:
                pass
        
        return (feature_name, pickle.dumps(result_df), cache_key)
    except Exception as e:
        print(f"     ❌ Error computing {feature_name}: {e}")
        return (feature_name, df_bytes, cache_key)  # 返回原始 DataFrame


class ParallelFeatureComputer:
    """
    并行特征计算器
    
    支持：
    1. 按依赖层级并行计算
    2. 内存缓存
    3. 磁盘缓存
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_disk_cache: bool = True,
        use_memory_cache: bool = True,
        max_workers: Optional[int] = None,
        parallel_backend: str = "process",  # "process", "thread"
    ):
        """
        Args:
            cache_dir: 磁盘缓存目录
            use_disk_cache: 是否使用磁盘缓存
            use_memory_cache: 是否使用内存缓存
            max_workers: 最大并行数（None 表示使用 CPU 核心数）
            parallel_backend: 并行后端（process/thread）
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_disk_cache = use_disk_cache
        self.use_memory_cache = use_memory_cache
        self.max_workers = max_workers or mp.cpu_count()
        self.parallel_backend = parallel_backend
        
        # 内存缓存
        self.memory_cache = {}
        
        # 并行执行器
        if parallel_backend == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        elif parallel_backend == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None
    
    def _get_cache_key(
        self, feature_name: str, df_hash: str, params: Dict
    ) -> str:
        """生成缓存键"""
        params_str = str(sorted(params.items()))
        key_str = f"{feature_name}_{df_hash}_{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_df_hash(self, df: pd.DataFrame, n_rows: int = 100) -> str:
        """生成 DataFrame 哈希（基于前 N 行）"""
        sample = df.head(n_rows)
        # 只使用数值列
        numeric_cols = sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return hashlib.md5(str(df.shape).encode()).hexdigest()
        sample_data = sample[numeric_cols].values.tobytes()
        return hashlib.md5(sample_data).hexdigest()
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[pd.DataFrame | pd.Series]:
        """从磁盘加载缓存（支持 DataFrame 和 Series）"""
        if not self.use_disk_cache or not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"     ⚠️  Error loading cache {cache_key}: {e}")
                return None
        return None
    
    def _save_to_disk_cache(self, cache_key: str, result: pd.DataFrame | pd.Series):
        """保存到磁盘缓存（支持 DataFrame 和 Series）"""
        if not self.use_disk_cache or not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"     ⚠️  Error saving cache {cache_key}: {e}")
    
    def compute_features_parallel(
        self,
        df: pd.DataFrame,
        features: Dict,
        requested_features: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        并行计算特征
        
        Args:
            df: 输入 DataFrame
            features: 特征配置字典
            requested_features: 请求的特征列表
            fit: 是否拟合
        
        Returns:
            df_with_features: 包含计算特征的 DataFrame
        """
        # 1. 分析依赖层级
        levels = analyze_dependency_levels(features, requested_features)
        
        print(
            f"   📊 Computing {len(requested_features)} features in {len(levels)} levels..."
        )
        
        result_df = df.copy()
        df_hash = self._get_df_hash(result_df)
        
        # 2. 按层级顺序计算（每层内并行）
        for level in sorted(levels.keys()):
            level_features = levels[level]
            print(
                f"   🔄 Level {level}: Computing {len(level_features)} features in parallel..."
            )
            
            # 提交并行任务
            futures = []
            for feature_name in level_features:
                if feature_name not in features:
                    print(
                        f"     ⚠️  Warning: Feature '{feature_name}' not found in dependencies config, skipping..."
                    )
                    continue
                
                # 检查内存缓存
                if self.use_memory_cache and feature_name in self.memory_cache:
                    print(f"     💾 Using memory cache for {feature_name}")
                    cached_result = self.memory_cache[feature_name]
                    # 合并结果（支持 Series 和 DataFrame）
                    if isinstance(cached_result, pd.DataFrame):
                        new_cols = [
                            c for c in cached_result.columns if c not in result_df.columns
                        ]
                        if new_cols:
                            result_df = pd.concat([result_df, cached_result[new_cols]], axis=1)
                    elif isinstance(cached_result, pd.Series):
                        if cached_result.name and cached_result.name not in result_df.columns:
                            result_df[cached_result.name] = cached_result
                        elif feature_name not in result_df.columns:
                            result_df[feature_name] = cached_result
                    continue
                
                feature_info = features[feature_name]
                compute_params = feature_info.get("compute_params", {})
                cache_key = (
                    self._get_cache_key(feature_name, df_hash, compute_params)
                    if self.use_disk_cache
                    else None
                )
                
                # 检查磁盘缓存
                if cache_key:
                    cached_result = self._load_from_disk_cache(cache_key)
                    if cached_result is not None:
                        print(f"     💾 Using disk cache for {feature_name}")
                        if self.use_memory_cache:
                            self.memory_cache[feature_name] = cached_result
                        # 合并结果（支持 Series 和 DataFrame）
                        if isinstance(cached_result, pd.DataFrame):
                            new_cols = [
                                c for c in cached_result.columns if c not in result_df.columns
                            ]
                            if new_cols:
                                result_df = pd.concat([result_df, cached_result[new_cols]], axis=1)
                        elif isinstance(cached_result, pd.Series):
                            if cached_result.name and cached_result.name not in result_df.columns:
                                result_df[cached_result.name] = cached_result
                            elif feature_name not in result_df.columns:
                                result_df[feature_name] = cached_result
                        continue
                
                # 提交任务
                if self.executor:
                    df_bytes = pickle.dumps(result_df)
                    future = self.executor.submit(
                        _compute_single_feature_worker,
                        feature_name,
                        feature_info,
                        df_bytes,
                        fit,
                        cache_key,
                        str(self.cache_dir) if self.cache_dir else None,
                    )
                    futures.append(future)
                else:
                    # 串行计算（fallback）
                    try:
                        compute_func_name = feature_info["compute_func"]
                        compute_func = get_compute_func(compute_func_name)
                        if compute_params:
                            feature_result = compute_func(result_df, **compute_params)
                        else:
                            feature_result = compute_func(result_df)
                        
                        # 合并结果
                        # 如果返回的是 DataFrame，合并新列
                        if isinstance(feature_result, pd.DataFrame):
                            new_cols = [
                                c for c in feature_result.columns if c not in result_df.columns
                            ]
                            if new_cols:
                                result_df = pd.concat([result_df, feature_result[new_cols]], axis=1)
                        # 如果返回的是 Series，添加到 DataFrame
                        elif isinstance(feature_result, pd.Series):
                            if feature_result.name and feature_result.name not in result_df.columns:
                                result_df[feature_result.name] = feature_result
                            elif feature_name not in result_df.columns:
                                result_df[feature_name] = feature_result
                        
                        # 保存缓存
                        if self.use_memory_cache:
                            self.memory_cache[feature_name] = feature_result
                        if cache_key:
                            self._save_to_disk_cache(cache_key, feature_result)
                        
                        print(f"     ✅ Computed {feature_name}")
                    except Exception as e:
                        print(f"     ❌ Error computing {feature_name}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    feature_name, result_df_bytes, cache_key = future.result()
                    feature_df = pickle.loads(result_df_bytes)
                    
                    # 合并结果
                    # 如果 feature_df 是 DataFrame，合并新列
                    if isinstance(feature_df, pd.DataFrame):
                        new_cols = [
                            c for c in feature_df.columns if c not in result_df.columns
                        ]
                        if new_cols:
                            result_df = pd.concat([result_df, feature_df[new_cols]], axis=1)
                    # 如果 feature_df 是 Series，添加到 DataFrame
                    elif isinstance(feature_df, pd.Series):
                        if feature_df.name and feature_df.name not in result_df.columns:
                            result_df[feature_df.name] = feature_df
                        elif feature_name not in result_df.columns:
                            result_df[feature_name] = feature_df
                    
                    # 保存内存缓存
                    if self.use_memory_cache:
                        self.memory_cache[feature_name] = feature_df
                    
                    print(f"     ✅ Computed {feature_name}")
                except Exception as e:
                    print(f"     ❌ Error in parallel computation: {e}")
                    import traceback
                    traceback.print_exc()
        
        return result_df
    
    def clear_cache(self, memory: bool = True, disk: bool = False):
        """清除缓存"""
        if memory:
            self.memory_cache.clear()
            print("   🗑️  Memory cache cleared")
        
        if disk and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("   🗑️  Disk cache cleared")
    
    def __del__(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)

