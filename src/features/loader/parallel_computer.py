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
from typing import Dict, List, Optional, Callable, Tuple, Any
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


def _build_call_args(
    feature_info: Dict, df: pd.DataFrame
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    根据特征配置构建 compute_func 所需的 args/kwargs.
    支持配置 column_mappings，将 DataFrame 指定列注入到函数参数。
    """
    compute_params = feature_info.get("compute_params", {}) or {}
    column_mappings = feature_info.get("column_mappings", {}) or {}
    call_kwargs = dict(compute_params)

    for param_name, source in column_mappings.items():
        if isinstance(source, str):
            col_name = source
            if col_name not in df.columns:
                raise KeyError(
                    f"Column '{col_name}' required for parameter '{param_name}' not found in DataFrame"
                )
            call_kwargs[param_name] = df[col_name]
        elif isinstance(source, list):
            missing = [col for col in source if col not in df.columns]
            if missing:
                raise KeyError(
                    f"Columns {missing} required for parameter '{param_name}' not found in DataFrame"
                )
            call_kwargs[param_name] = df[source]
        else:
            raise ValueError(
                f"Unsupported column mapping type for parameter '{param_name}': {type(source)}"
            )

    call_args: List[Any] = []
    if feature_info.get("pass_full_df", True):
        call_args.append(df)

    return call_args, call_kwargs


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
        call_args, call_kwargs = _build_call_args(feature_info, df)
        feature_result = compute_func(*call_args, **call_kwargs)
        
        # Handle different return types
        # If function returns a tuple (e.g., MACD returns (macd, signal, histogram)),
        # convert it to a DataFrame with columns from output_columns config
        if isinstance(feature_result, tuple):
            output_cols = feature_info.get("output_columns", [feature_name])
            if len(feature_result) == len(output_cols):
                # Create DataFrame from tuple
                result_df = pd.DataFrame({
                    col: series for col, series in zip(output_cols, feature_result)
                }, index=df.index)
            else:
                # Fallback: use feature_name with index
                result_df = pd.DataFrame({
                    f"{feature_name}_{i}": series for i, series in enumerate(feature_result)
                }, index=df.index)
        elif isinstance(feature_result, pd.DataFrame):
            result_df = feature_result
        elif isinstance(feature_result, pd.Series):
            # Convert Series to DataFrame
            output_cols = feature_info.get("output_columns", [feature_name])
            if len(output_cols) == 1:
                result_df = pd.DataFrame({output_cols[0]: feature_result}, index=df.index)
            else:
                result_df = pd.DataFrame({feature_name: feature_result}, index=df.index)
        else:
            # Fallback: try to convert to Series/DataFrame
            result_df = pd.DataFrame({feature_name: feature_result}, index=df.index)
        
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
        import traceback
        traceback.print_exc()
        # 如果是依赖缺失错误，应该抛出异常而不是继续
        if isinstance(e, (ValueError, KeyError)) and ("not found" in str(e) or "Required" in str(e)):
            raise  # 重新抛出依赖缺失错误
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
        self, feature_name: str, df_hash: str, params: Dict, feature_info: Optional[Dict] = None
    ) -> str:
        """
        生成缓存键
        
        Args:
            feature_name: 特征名
            df_hash: DataFrame 哈希
            params: 计算参数
            feature_info: 特征配置信息（用于包含 output_columns 等信息）
        """
        params_str = str(sorted(params.items()))
        
        # 包含 output_columns 信息，确保缓存键在配置改变时也会改变
        output_cols_str = ""
        if feature_info:
            output_cols = feature_info.get("output_columns", [feature_name])
            output_cols_str = str(sorted(output_cols))
        
        # 包含代码版本（处理 Tuple 的逻辑版本）
        # 当处理逻辑改变时，这个版本号应该更新
        code_version = "v2"  # v2: 支持 Tuple 返回值转换为 DataFrame
        
        key_str = f"{feature_name}_{df_hash}_{params_str}_{output_cols_str}_{code_version}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_df_hash(self, df: pd.DataFrame, n_rows: int = 100) -> str:
        """生成 DataFrame 哈希（基于前 N 行 + 时间范围/行数等元信息）"""
        if df.empty:
            base_hash = hashlib.md5(str(df.shape).encode()).hexdigest()
            start_meta = "EMPTY"
            end_meta = "EMPTY"
        else:
            sample = df.head(n_rows)
            numeric_cols = sample.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                base_hash = hashlib.md5(str(df.shape).encode()).hexdigest()
            else:
                sample_data = sample[numeric_cols].values.tobytes()
                base_hash = hashlib.md5(sample_data).hexdigest()
            try:
                start_meta = str(df.index[0])
                end_meta = str(df.index[-1])
            except Exception:
                start_meta = "NO_INDEX_START"
                end_meta = "NO_INDEX_END"
        meta_str = f"{base_hash}|rows={len(df)}|start={start_meta}|end={end_meta}"
        return hashlib.md5(meta_str.encode()).hexdigest()
    
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
        # 0. 预处理：如果用户请求了 output_columns（如 macd_signal），自动找到对应的父特征
        actual_requested = []
        output_col_to_feature = {}  # Map output column to parent feature
        
        # Build reverse mapping: output_columns -> feature_name
        for feature_name, feature_info in features.items():
            output_cols = feature_info.get("output_columns", [feature_name])
            for output_col in output_cols:
                output_col_to_feature[output_col] = feature_name
        
        # Resolve requested features
        for requested in requested_features:
            if requested in features:
                # Direct feature name
                actual_requested.append(requested)
            elif requested in output_col_to_feature:
                # Output column name, use parent feature
                parent_feature = output_col_to_feature[requested]
                if parent_feature not in actual_requested:
                    actual_requested.append(parent_feature)
                    print(f"     ℹ️  '{requested}' is an output column of '{parent_feature}', computing parent feature instead")
            else:
                # Not found, will be handled later
                actual_requested.append(requested)
        
        # Remove duplicates while preserving order
        actual_requested = list(dict.fromkeys(actual_requested))
        
        # 1. 分析依赖层级
        levels = analyze_dependency_levels(features, actual_requested)
        
        print(
            f"   📊 Computing {len(actual_requested)} features in {len(levels)} levels..."
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
                    self._get_cache_key(feature_name, df_hash, compute_params, feature_info)
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
                
                run_sequential = feature_info.get("run_sequential", False) or not self.executor

                # 提交任务
                if not run_sequential:
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
                    print(f"     🔸 Running {feature_name} sequentially", flush=True)
                    # 串行计算（或被标记为顺序执行）
                    try:
                        compute_func_name = feature_info["compute_func"]
                        compute_func = get_compute_func(compute_func_name)
                        call_args, call_kwargs = _build_call_args(feature_info, result_df)
                        feature_result = compute_func(*call_args, **call_kwargs)
                        
                        # Handle different return types
                        # If function returns a tuple (e.g., MACD), convert to DataFrame
                        if isinstance(feature_result, tuple):
                            output_cols = feature_info.get("output_columns", [feature_name])
                            if len(feature_result) == len(output_cols):
                                # Create DataFrame from tuple and merge columns
                                feature_df = pd.DataFrame({
                                    col: series for col, series in zip(output_cols, feature_result)
                                }, index=result_df.index)
                                new_cols = [c for c in feature_df.columns if c not in result_df.columns]
                                if new_cols:
                                    result_df = pd.concat([result_df, feature_df[new_cols]], axis=1)
                            else:
                                # Fallback: add with indexed names
                                for i, series in enumerate(feature_result):
                                    col_name = output_cols[i] if i < len(output_cols) else f"{feature_name}_{i}"
                                    if col_name not in result_df.columns:
                                        result_df[col_name] = series
                        # 如果返回的是 DataFrame，合并新列
                        elif isinstance(feature_result, pd.DataFrame):
                            new_cols = [
                                c for c in feature_result.columns if c not in result_df.columns
                            ]
                            if new_cols:
                                result_df = pd.concat([result_df, feature_result[new_cols]], axis=1)
                        # 如果返回的是 Series，添加到 DataFrame
                        elif isinstance(feature_result, pd.Series):
                            output_cols = feature_info.get("output_columns", [feature_name])
                            col_name = output_cols[0] if output_cols else (feature_result.name or feature_name)
                            if col_name not in result_df.columns:
                                result_df[col_name] = feature_result
                        
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
            self.executor.shutdown(wait=True)  # ensure executor cleaned up
