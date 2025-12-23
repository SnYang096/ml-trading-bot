"""
特征计算器（顺序 + 缓存）

当前实现**强制顺序执行**特征计算，并依赖：
- 内存缓存（同一 DataFrame 签名内复用）
- 磁盘缓存 / monthly 缓存（跨运行复用，支持按月增量）

说明：
- 文件名已从 `parallel_computer.py` 重命名为 `feature_computer.py`（旧并行实现已移除，现为顺序执行）。
- 为了减少复杂度与大 DataFrame 序列化/复制风险，已移除多进程/多线程执行路径。
"""

import multiprocessing as mp
from functools import lru_cache
import hashlib
import os
import pickle
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
import pandas as pd
import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.features.registry import get_compute_func


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
        feat_name = queue.pop(0)
        if feat_name not in features:
            continue
        deps = features[feat_name].get("dependencies", [])
        for dep in deps:
            if dep not in all_needed:
                all_needed.add(dep)
                queue.append(dep)

    # 2. 计算层级
    levels: Dict[int, List[str]] = {}
    resolved = set()

    def get_level(feat_name: str) -> int:
        if feat_name in resolved:
            return -1  # 已解析，避免循环依赖
        if feat_name not in features:
            return 0
        deps = features[feat_name].get("dependencies", [])
        if not deps:
            return 0
        return max([get_level(dep) for dep in deps], default=0) + 1

    for feat_name in all_needed:
        level = get_level(feat_name)
        if level not in levels:
            levels[level] = []
        levels[level].append(feat_name)
        resolved.add(feat_name)

    return levels


def _build_call_args(
    feature_info: Dict, df: pd.DataFrame, feature_name: str
) -> Tuple[tuple, dict]:
    """
    构建特征计算函数的调用参数

    Args:
        feature_info: 特征配置信息
        df: 输入 DataFrame
        feature_name: 特征名称

    Returns:
        (args, kwargs): 调用参数元组和关键字参数字典
    """
    compute_params = feature_info.get("compute_params", {})
    required_columns = feature_info.get("required_columns", [])

    # 构建位置参数（按顺序）
    args = []
    for param_name in feature_info.get("positional_params", []):
        if param_name == "df":
            args.append(df)
        elif param_name in df.columns:
            args.append(df[param_name])
        elif param_name in compute_params:
            args.append(compute_params[param_name])
        else:
            raise KeyError(
                f"Column '{param_name}' required for parameter '{param_name}' not found in DataFrame"
            )

    # 构建关键字参数
    kwargs = {}
    for param_name, param_value in compute_params.items():
        if param_name not in feature_info.get("positional_params", []):
            if isinstance(param_value, str) and param_value in df.columns:
                kwargs[param_name] = df[param_name]
            else:
                kwargs[param_name] = param_value

    return (tuple(args), kwargs)


def _get_monthly_cache_key(
    feature_name: str,
    month_key: str,
    df_sig: str,
    cache_version: str,
    fit: bool,
    compute_params: Dict,
) -> str:
    """
    生成月度缓存键

    Args:
        feature_name: 特征名称
        month_key: 月份键（如 "2024-01"）
        df_sig: DataFrame 签名（用于区分不同的数据集）
        cache_version: 缓存版本
        fit: 是否拟合
        compute_params: 计算参数

    Returns:
        cache_key: 缓存键字符串
    """
    # 构建参数签名（用于区分不同的参数配置）
    params_sig = hashlib.md5(
        str(sorted(compute_params.items())).encode()
    ).hexdigest()[:8]

    # 构建缓存键
    key_parts = [
        feature_name,
        month_key,
        df_sig[:16],  # 使用前16个字符
        cache_version,
        "fit" if fit else "predict",
        params_sig,
    ]
    cache_key = "_".join(key_parts)
    return cache_key


def _save_monthly_cache(
    cache_dir: Path, cache_key: str, result: pd.DataFrame | pd.Series
) -> None:
    """保存月度缓存"""
    cache_file = cache_dir / f"{cache_key}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)


def _load_monthly_cache(cache_dir: Path, cache_key: str) -> Optional[pd.DataFrame | pd.Series]:
    """加载月度缓存"""
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def _compute_single_feature_worker_monthly(
    feature_name: str,
    feature_info: Dict,
    df_bytes: bytes,
    fit: bool,
    monthly_cache_dir: Optional[str],
    ticks_loader_json: Optional[str] = None,
) -> Tuple[str, bytes]:
    """
    Legacy parallel worker (DEPRECATED).

    This function is kept for backward compatibility but is no longer used
    in the current sequential implementation.
    """
    # This function is deprecated and should not be called
    raise NotImplementedError(
        "Parallel feature computation has been removed. Use sequential computation instead."
    )


class FeatureComputer:
    """
    特征计算器（顺序 + 缓存）

    当前实现**强制顺序执行**特征计算，并依赖：
    - 内存缓存（同一 DataFrame 签名内复用）
    - 磁盘缓存 / monthly 缓存（跨运行复用，支持按月增量）

    说明：
    - 为了减少复杂度与大 DataFrame 序列化/复制风险，已移除多进程/多线程执行路径。
    - 性能主要依赖磁盘/月度缓存，内存缓存作为补充加速。
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_disk_cache: bool = True,
        use_memory_cache: bool = True,
        max_workers: Optional[int] = None,
        parallel_backend: str = "process",  # deprecated (sequential-only)
        use_monthly_cache: bool = True,  # 是否使用按月缓存
        enable_parallel: bool = False,  # deprecated (sequential-only)
    ):
        """
        Args:
            cache_dir: 磁盘缓存目录
            use_disk_cache: 是否使用磁盘缓存
            use_memory_cache: 是否使用内存缓存
            max_workers: 最大并行数（None 表示使用 CPU 核心数）
            parallel_backend: 并行后端（process/thread）
            use_monthly_cache: 是否使用按月缓存（增量计算）
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_disk_cache = use_disk_cache
        self.use_memory_cache = use_memory_cache
        self.use_monthly_cache = use_monthly_cache

        # Simplified design: always run sequentially at the feature level.
        # Performance is achieved via disk/monthly caches rather than process/thread pools.
        self.enable_parallel = False
        self.max_workers = 1
        self.parallel_backend = "sequential"
        self.executor = None
        self._print_memory_info()
        print("   🔧 Feature-level parallelism disabled. Running sequentially.")

        # 内存缓存：使用 (df_signature, feature_name) 作为键，而不是全局清空
        # 这样即使 DataFrame 切换，相同 DataFrame 签名的特征结果仍可复用
        self.memory_cache = {}  # key: (df_sig_tuple, feature_name), value: cached result

        # Debug stats (last run). Callers can drain via drain_debug_stats().
        # - index_mismatch: feature_name -> {"extra": int, "missing": int}
        self._debug_stats: Dict[str, Any] = {"index_mismatch": {}}

        # 缓存版本控制（用于失效旧缓存）
        # 当特征计算逻辑改变时，更新此版本号以失效旧缓存
        self.cache_version = "v6"

        # 月度缓存目录
        if self.use_monthly_cache and self.cache_dir:
            self.monthly_cache_dir = self.cache_dir / "monthly"
            self.monthly_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.monthly_cache_dir = None

        print(f"   🔧 Workers: {self.max_workers}, Backend: {self.parallel_backend}")

    def _print_memory_info(self) -> None:
        """打印内存信息"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            system_mem = psutil.virtual_memory()
            print(
                f"   💾 Memory: {system_mem.available / 1024**3:.1f}GB available, "
                f"{mem_info.rss / 1024**3:.1f}GB used, "
                f"{system_mem.total / 1024**3:.1f}GB total "
                f"({system_mem.percent:.1f}% used)"
            )

    def drain_debug_stats(self) -> Dict[str, Any]:
        """Return and reset debug stats from the last compute_features_parallel run."""
        stats = self._debug_stats.copy()
        self._debug_stats = {"index_mismatch": {}}
        return stats

    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """计算 DataFrame 的哈希值（用于缓存键）"""
        h = hashlib.md5()
        h.update(str(df.index).encode())
        h.update(str(df.shape).encode())
        h.update(str(df.columns.tolist()).encode())
        # 采样一些数据值以确保不同数据集的哈希不同
        if len(df) > 0:
            sample_size = min(100, len(df))
            sample_indices = df.index[:: max(1, len(df) // sample_size)][:sample_size]
            sample_data = df.loc[sample_indices]
            h.update(str(sample_data.values).encode())
        return h.hexdigest()

    def _get_df_signature(self, df: pd.DataFrame) -> str:
        """
        生成 DataFrame 的轻量级签名（用于月度缓存键）

        这个签名应该足够轻量，但又能区分不同的数据集。
        包括：索引范围、长度、列哈希、采样数据值。

        Args:
            df: 输入 DataFrame

        Returns:
            df_sig: DataFrame 签名字符串
        """
        h = hashlib.md5()
        # 索引范围
        if len(df) > 0:
            h.update(str(df.index.min()).encode())
            h.update(str(df.index.max()).encode())
        # 长度
        h.update(str(len(df)).encode())
        # 列哈希
        h.update(str(sorted(df.columns)).encode())
        # 采样数据值（只采样 close 和 volume，如果存在）
        if len(df) > 0:
            sample_size = min(50, len(df))
            sample_indices = df.index[:: max(1, len(df) // sample_size)][:sample_size]
            for col in ["close", "volume"]:
                if col in df.columns:
                    sample_values = df.loc[sample_indices, col].values
                    h.update(str(sample_values).encode())
        return h.hexdigest()

    def _align_to_base_index(
        self, feature_name: str, result: pd.DataFrame | pd.Series, base_index: pd.Index
    ) -> pd.DataFrame | pd.Series:
        """
        将特征结果对齐到基础索引

        Args:
            feature_name: 特征名称（用于调试）
            result: 特征计算结果
            base_index: 基础索引

        Returns:
            aligned_result: 对齐后的结果
        """
        if isinstance(result, pd.Series):
            if not result.index.equals(base_index):
                # 记录索引不匹配统计
                extra = len(result.index.difference(base_index))
                missing = len(base_index.difference(result.index))
                if feature_name not in self._debug_stats["index_mismatch"]:
                    self._debug_stats["index_mismatch"][feature_name] = {
                        "extra": 0,
                        "missing": 0,
                    }
                self._debug_stats["index_mismatch"][feature_name]["extra"] += extra
                self._debug_stats["index_mismatch"][feature_name]["missing"] += missing

                # 对齐到基础索引
                result = result.reindex(base_index)
            return result
        elif isinstance(result, pd.DataFrame):
            if not result.index.equals(base_index):
                # 记录索引不匹配统计
                extra = len(result.index.difference(base_index))
                missing = len(base_index.difference(result.index))
                if feature_name not in self._debug_stats["index_mismatch"]:
                    self._debug_stats["index_mismatch"][feature_name] = {
                        "extra": 0,
                        "missing": 0,
                    }
                self._debug_stats["index_mismatch"][feature_name]["extra"] += extra
                self._debug_stats["index_mismatch"][feature_name]["missing"] += missing

                # 对齐到基础索引
                result = result.reindex(base_index)
            return result
        else:
            return result

    def _validate_cache_quality(
        self,
        cached_result: pd.DataFrame | pd.Series,
        feature_name: str,
        cache_type: str = "unknown",
    ) -> None:
        """
        验证缓存数据质量

        Args:
            cached_result: 缓存的结果
            feature_name: 特征名称
            cache_type: 缓存类型（memory/disk/monthly）
        """
        if isinstance(cached_result, (pd.Series, pd.DataFrame)):
            if cached_result.empty:
                print(
                    f"     ⚠️  {cache_type} cache for {feature_name} is empty, recomputing."
                )
            elif isinstance(cached_result, pd.DataFrame):
                nan_ratio = cached_result.isna().sum().sum() / (
                    len(cached_result) * len(cached_result.columns)
                )
                if nan_ratio > 0.5:
                    print(
                        f"     ⚠️  {cache_type} cache for {feature_name} has high NaN ratio "
                        f"({nan_ratio:.1%}), but using it anyway."
                    )

    def _compute_and_cache_monthly(
        self,
        feature_name: str,
        df: pd.DataFrame,
        compute_params: Dict,
        feature_info: Dict,
        compute_func: Callable,
    ) -> pd.DataFrame | pd.Series:
        """
        按月计算并缓存特征（支持增量计算）

        Args:
            feature_name: 特征名称
            df: 输入 DataFrame
            compute_params: 计算参数
            feature_info: 特征配置信息
            compute_func: 计算函数

        Returns:
            result: 特征计算结果
        """
        if not self.use_monthly_cache or self.monthly_cache_dir is None:
            # 如果不使用月度缓存，直接计算
            call_args, call_kwargs = _build_call_args(feature_info, df, feature_name)
            result = compute_func(*call_args, **call_kwargs)
            return result

        # 获取 DataFrame 签名（用于区分不同的数据集）
        df_sig = self._get_df_signature(df)

        # 按月分组
        monthly_groups = df.groupby(pd.Grouper(freq="M"))
        monthly_results: Dict[str, pd.DataFrame | pd.Series] = {}

        # 统计缓存命中情况
        cached_months = 0
        computed_months = 0

        for month_key, df_month in monthly_groups:
            if df_month.empty:
                continue

            month_str = month_key.strftime("%Y-%m")

            # 生成月度缓存键
            monthly_cache_key = _get_monthly_cache_key(
                feature_name=feature_name,
                month_key=month_str,
                df_sig=df_sig,
                cache_version=self.cache_version,
                fit=True,  # 月度缓存不区分 fit/predict
                compute_params=compute_params,
            )

            # 尝试加载缓存
            cached_result = _load_monthly_cache(
                self.monthly_cache_dir, monthly_cache_key
            )

            if cached_result is not None:
                # 验证缓存质量
                self._validate_cache_quality(
                    cached_result, feature_name, cache_type="monthly"
                )
                monthly_results[month_str] = cached_result
                cached_months += 1
            else:
                # 计算特征
                print(f"       🔸 Computing {feature_name} for {month_str}...")
                call_args, call_kwargs = _build_call_args(
                    feature_info, df_month, feature_name
                )
                month_result = compute_func(*call_args, **call_kwargs)

                # 处理计算结果的重复列名
                if (
                    isinstance(month_result, pd.DataFrame)
                    and month_result.columns.duplicated().any()
                ):
                    month_result = month_result.loc[
                        :, ~month_result.columns.duplicated()
                    ]

                # 根本性解决方案：只提取 output_columns 中定义的列
                output_cols = feature_info.get("output_columns", [feature_name])
                if not output_cols:
                    output_cols = [feature_name]

                if isinstance(month_result, pd.DataFrame):
                    # 只保留 output_columns 中定义的列
                    result_cols = [
                        col for col in output_cols if col in month_result.columns
                    ]
                    if result_cols:
                        month_result = month_result[result_cols].copy()
                    else:
                        month_result = pd.DataFrame(index=month_result.index)
                elif isinstance(month_result, pd.Series):
                    series_name = (
                        month_result.name if month_result.name else feature_name
                    )
                    if series_name not in output_cols:
                        # 如果不在 output_columns 中，返回空 DataFrame
                        month_result = pd.DataFrame(index=month_result.index)

                # 保存缓存（只保存 output_columns）
                _save_monthly_cache(self.monthly_cache_dir, monthly_cache_key, month_result)

                monthly_results[month_str] = month_result
                computed_months += 1

            # 打印每个月份的结果信息
            if isinstance(monthly_results[month_str], pd.DataFrame):
                month_result = monthly_results[month_str]
                print(
                    f"       📊 Month {month_str}: {len(month_result)} rows, {len(month_result.columns)} columns: {list(month_result.columns)[:5]}..."
                )
                if month_result.columns.duplicated().any():
                    dup_cols = month_result.columns[
                        month_result.columns.duplicated()
                    ].tolist()
                    print(
                        f"       ⚠️  Month {month_str} has duplicate columns: {dup_cols}"
                    )
            elif isinstance(monthly_results[month_str], pd.Series):
                month_result = monthly_results[month_str]
                print(
                    f"       📊 Month {month_str}: {len(month_result)} rows, Series name: {month_result.name}"
                )

        # 获取特征的输出列（根本性解决方案：只返回 output_columns 中定义的列）
        output_cols = feature_info.get("output_columns", [feature_name])
        if not output_cols:
            output_cols = [feature_name]

        # 合并所有月份的结果
        print(f"       🔄 Merging {len(monthly_results)} monthly results...")
        try:
            # 处理不同的返回类型
            if isinstance(list(monthly_results.values())[0], tuple):
                # 如果是tuple，需要分别合并每个元素
                combined_results = []
                for i in range(len(output_cols)):
                    combined_series = pd.concat(
                        [r[i] for r in monthly_results.values()], axis=0
                    ).sort_index()
                    combined_results.append(combined_series)
                result_df = pd.DataFrame(
                    {col: series for col, series in zip(output_cols, combined_results)}
                )
            elif isinstance(list(monthly_results.values())[0], pd.DataFrame):
                # 根本性解决方案：只保留 output_columns 中定义的列
                all_columns = set(output_cols)  # 只使用 output_columns，不包含其他列

                # 确保所有 DataFrame 都有相同的列（缺失的列填充 NaN）
                aligned_results = []
                for month_key, month_result in monthly_results.items():
                    # 处理重复列名：如果有重复列，保留第一个
                    if (
                        isinstance(month_result, pd.DataFrame)
                        and month_result.columns.duplicated().any()
                    ):
                        dup_cols = month_result.columns[
                            month_result.columns.duplicated()
                        ].tolist()
                        print(
                            f"       ⚠️  Month {month_key} has duplicate columns before dedup: {dup_cols}"
                        )
                        month_result = month_result.loc[
                            :, ~month_result.columns.duplicated()
                        ]
                        print(
                            f"       ✅ Month {month_key} after dedup: {len(month_result.columns)} columns"
                        )
                        # 更新字典中的值
                        monthly_results[month_key] = month_result

                    # 只提取 output_columns 中定义的列
                    if isinstance(month_result, pd.DataFrame):
                        result_cols = [
                            col for col in output_cols if col in month_result.columns
                        ]
                        if result_cols:
                            # 确保列顺序一致
                            month_result = month_result[result_cols].copy()
                        else:
                            month_result = pd.DataFrame(index=month_result.index)
                    aligned_results.append(month_result)

                # 合并所有月份的结果
                result_df = pd.concat(aligned_results, axis=0).sort_index()

                # 确保只保留 output_columns 中定义的列
                result_cols = [
                    col for col in output_cols if col in result_df.columns
                ]
                if result_cols:
                    result_df = result_df[result_cols].copy()
                else:
                    result_df = pd.DataFrame(index=result_df.index)
            elif isinstance(list(monthly_results.values())[0], pd.Series):
                # 合并 Series
                combined_series = pd.concat(
                    list(monthly_results.values()), axis=0
                ).sort_index()
                # 如果 output_columns 中只有一个列，且 Series name 匹配，则返回 Series
                if len(output_cols) == 1 and combined_series.name == output_cols[0]:
                    result_df = combined_series
                else:
                    # 否则转换为 DataFrame
                    result_df = pd.DataFrame({output_cols[0]: combined_series})
            else:
                raise ValueError(
                    f"Unexpected result type: {type(list(monthly_results.values())[0])}"
                )
        except Exception as e:
            print(f"       ❌ Error merging monthly results: {e}")
            raise

        # 打印缓存统计
        if cached_months > 0 or computed_months > 0:
            print(
                f"       💾 Used {cached_months} cached months, computed {computed_months} new months"
            )

        return result_df

    def compute_features_parallel(
        self,
        df: pd.DataFrame,
        features: Dict,
        requested_features: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        计算特征（顺序执行 + 缓存）

        Args:
            df: 输入 DataFrame
            features: 特征配置字典
            requested_features: 请求的特征列表
            fit: 是否拟合

        Returns:
            result_df: 包含计算特征的 DataFrame
        """
        result_df = df.copy()
        base_index = result_df.index
        df_hash = self._get_df_hash(result_df)

        # Memory cache key: use (df_signature, feature_name) instead of global clearing.
        # This allows cross-variant reuse when the same DataFrame signature is encountered.
        # We still track the current signature for logging/debugging purposes.
        current_df_sig = None
        if self.use_memory_cache:
            try:
                current_df_sig = (
                    df_hash,
                    len(result_df),
                    str(result_df.index.min()) if len(result_df) else "EMPTY",
                    str(result_df.index.max()) if len(result_df) else "EMPTY",
                )
                # Store current signature for potential future use (e.g., cache size limits)
                self._current_df_sig = current_df_sig
            except Exception:
                # If signature calculation fails, disable memory cache for this call
                current_df_sig = None

        # 1.5. 确保所有请求特征的 required_columns 都在 DataFrame 中
        # 收集所有需要的 required_columns
        all_required_columns = set()
        for feature_name in requested_features:
            if feature_name in features:
                feature_info = features[feature_name]
                required_columns = feature_info.get("required_columns", [])
                all_required_columns.update(required_columns)

        # 检查缺失的 required_columns
        missing_required = [
            col for col in all_required_columns if col not in result_df.columns
        ]
        if missing_required:
            # 检查哪些是基础数据列（可能在原始df中），哪些是特征输出列（会通过依赖关系计算）
            base_data_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            base_missing = [c for c in missing_required if c in base_data_cols]
            feature_missing = [c for c in missing_required if c not in base_data_cols]

            if base_missing:
                raise ValueError(
                    f"Missing required base data columns: {base_missing}. "
                    f"Please ensure these columns exist in the input DataFrame."
                )
            # feature_missing 会在依赖计算时自动解决，这里只警告
            if feature_missing:
                print(
                    f"   ⚠️  Missing feature columns (will be computed via dependencies): {feature_missing[:10]}..."
                )

        # 2. 分析依赖层级
        levels = analyze_dependency_levels(features, requested_features)

        # 3. 按层级顺序计算特征
        computed_features = set()
        for level in sorted(levels.keys()):
            level_features = levels[level]
            print(
                f"   📊 Memory usage before level {level}: Process={psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f}GB, "
                f"System={psutil.virtual_memory().used / 1024**3:.1f}GB/{psutil.virtual_memory().total / 1024**3:.1f}GB "
                f"({psutil.virtual_memory().percent:.1f}%)"
            )

            for feature_name in level_features:
                if feature_name in computed_features:
                    continue

                feature_info = features[feature_name]
                compute_func = get_compute_func(feature_name)

                if compute_func is None:
                    print(f"     ⚠️  Skipping {feature_name}: compute function not found")
                    continue

                # 检查内存缓存：使用 (df_signature, feature_name) 作为键
                cache_key = None
                cached_result = None
                if self.use_memory_cache and current_df_sig is not None:
                    cache_key = (current_df_sig, feature_name)
                    if cache_key in self.memory_cache:
                        print(
                            f"     💾 Using memory cache for {feature_name} "
                            f"(df_sig={current_df_sig[0][:8]}...)"
                        )
                        cached_result = self.memory_cache[cache_key]

                        # SAFETY: cached results must match current index; otherwise it will
                        # explode the index (train/test/volatility mix) and can cause huge memory use.
                        try:
                            if isinstance(cached_result, (pd.Series, pd.DataFrame)):
                                if not cached_result.index.equals(result_df.index):
                                    print(
                                        f"     ⚠️  Memory cache index mismatch for {feature_name} "
                                        f"(cached={len(cached_result)}, current={len(result_df)}), recomputing."
                                    )
                                    cached_result = None
                        except Exception:
                            # Be conservative: if we cannot validate, don't use cache
                            cached_result = None

                        if cached_result is not None:
                            cached_result = self._align_to_base_index(
                                feature_name, cached_result, base_index
                            )
                            # 验证cache数据质量
                            self._validate_cache_quality(
                                cached_result, feature_name, cache_type="memory"
                            )
                            # 合并结果（支持 Series 和 DataFrame）
                            if isinstance(cached_result, pd.DataFrame):
                                new_cols = [
                                    c
                                    for c in cached_result.columns
                                    if c not in result_df.columns
                                ]
                                existing_cols = [
                                    c
                                    for c in cached_result.columns
                                    if c in result_df.columns
                                ]

                                # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                                if existing_cols:
                                    # 直接跳过，不合并（节省内存和时间）
                                    pass

                                # 添加新列
                                if new_cols:
                                    # Ensure both DataFrames are aligned to base_index before concat
                                    # This prevents index expansion when indices don't match
                                    cached_aligned = cached_result[new_cols].reindex(
                                        base_index
                                    )
                                    result_df = pd.concat(
                                        [result_df, cached_aligned], axis=1
                                    )
                            elif isinstance(cached_result, pd.Series):
                                if cached_result.name not in result_df.columns:
                                    cached_aligned = cached_result.reindex(base_index)
                                    result_df[cached_result.name] = cached_aligned

                            computed_features.add(feature_name)
                            continue

                # 如果没有缓存，计算特征
                compute_params = feature_info.get("compute_params", {})
                print(f"     ▶️ {feature_name}: start (level {level})")

                # 检查是否使用月度缓存
                if self.use_monthly_cache and self.monthly_cache_dir:
                    # 使用月度缓存计算
                    print(f"     🔸 Running {feature_name} sequentially (monthly)")
                    # 检查内存是否足够
                    if self.max_workers == 1 and PSUTIL_AVAILABLE:
                        available_gb = (
                            psutil.virtual_memory().available / 1024**3
                        )
                        if len(df) < 10000 or available_gb > 10:
                            print(
                                f"     ⚡ Using full-data computation (memory: {available_gb:.1f}GB available, data: {len(df)} rows)"
                            )
                            # 直接计算全量数据（不使用月度缓存）
                            call_args, call_kwargs = _build_call_args(
                                feature_info, df, feature_name
                            )
                            combined_result = compute_func(*call_args, **call_kwargs)
                        else:
                            # 使用月度缓存
                            combined_result = self._compute_and_cache_monthly(
                                feature_name,
                                df,
                                compute_params,
                                feature_info,
                                compute_func,
                            )
                    else:
                        # 使用月度缓存
                        combined_result = self._compute_and_cache_monthly(
                            feature_name,
                            df,
                            compute_params,
                            feature_info,
                            compute_func,
                        )

                    # 验证合并后的cache数据质量
                    self._validate_cache_quality(
                        combined_result, feature_name, cache_type="monthly"
                    )
                    combined_result = self._align_to_base_index(
                        feature_name, combined_result, base_index
                    )
                    # Store in memory cache with (df_signature, feature_name) key
                    if self.use_memory_cache and current_df_sig is not None:
                        cache_key = (current_df_sig, feature_name)
                        self.memory_cache[cache_key] = combined_result
                    # 合并到result_df
                    if isinstance(combined_result, pd.DataFrame):
                        # 处理重复列名：如果有重复列，保留第一个
                        if combined_result.columns.duplicated().any():
                            combined_result = combined_result.loc[
                                :, ~combined_result.columns.duplicated()
                            ]

                        new_cols = [
                            c
                            for c in combined_result.columns
                            if c not in result_df.columns
                        ]
                        existing_cols = [
                            c
                            for c in combined_result.columns
                            if c in result_df.columns
                        ]

                        # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                        if existing_cols:
                            # 直接跳过，不合并（节省内存和时间）
                            pass

                        # 添加新列
                        if new_cols:
                            # Ensure both DataFrames are aligned to base_index before concat
                            # This prevents index expansion when indices don't match
                            combined_aligned = combined_result[new_cols].reindex(
                                base_index
                            )
                            result_df = pd.concat([result_df, combined_aligned], axis=1)
                    elif isinstance(combined_result, pd.Series):
                        if combined_result.name not in result_df.columns:
                            combined_aligned = combined_result.reindex(base_index)
                            result_df[combined_result.name] = combined_aligned
                else:
                    # 不使用月度缓存，直接计算
                    call_args, call_kwargs = _build_call_args(
                        feature_info, df, feature_name
                    )
                    computed_result = compute_func(*call_args, **call_kwargs)

                    # 对齐到基础索引
                    computed_result = self._align_to_base_index(
                        feature_name, computed_result, base_index
                    )

                    # 验证新计算的特征质量
                    self._validate_cache_quality(
                        computed_result,
                        feature_name,
                        cache_type="computed",
                    )

                    # 保存内存缓存
                    if self.use_memory_cache and current_df_sig is not None:
                        cache_key = (current_df_sig, feature_name)
                        self.memory_cache[cache_key] = computed_result

                    # 合并到result_df
                    if isinstance(computed_result, pd.DataFrame):
                        new_cols = [
                            c
                            for c in computed_result.columns
                            if c not in result_df.columns
                        ]
                        if new_cols:
                            computed_aligned = computed_result[new_cols].reindex(
                                base_index
                            )
                            result_df = pd.concat([result_df, computed_aligned], axis=1)
                    elif isinstance(computed_result, pd.Series):
                        if computed_result.name not in result_df.columns:
                            computed_aligned = computed_result.reindex(base_index)
                            result_df[computed_result.name] = computed_aligned

                print(f"     ✅ Computed {feature_name}")
                computed_features.add(feature_name)

            print(
                f"   📊 Memory usage after level {level}: Process={psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f}GB, "
                f"System={psutil.virtual_memory().used / 1024**3:.1f}GB/{psutil.virtual_memory().total / 1024**3:.1f}GB "
                f"({psutil.virtual_memory().percent:.1f}%)"
            )

        # 4. 清理内存
        if self.use_memory_cache:
            # 可选：限制内存缓存大小（避免内存爆炸）
            # 这里暂时不限制，因为键是基于 (df_signature, feature_name) 的，通常不会太多
            pass

        return result_df

    def clear_cache(self, memory: bool = True, disk: bool = False) -> None:
        """
        清除缓存

        Args:
            memory: 是否清除内存缓存
            disk: 是否清除磁盘缓存
        """
        if memory:
            self.memory_cache.clear()
            print("   🗑️  Memory cache cleared")
        if disk and self.cache_dir:
            import shutil

            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.monthly_cache_dir:
                self.monthly_cache_dir.mkdir(parents=True, exist_ok=True)
            print("   🗑️  Disk cache cleared")


__all__ = ["FeatureComputer", "analyze_dependency_levels"]
