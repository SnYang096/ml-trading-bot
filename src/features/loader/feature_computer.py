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
import json
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
    compute_params = feature_info.get("compute_params", {}) or {}
    column_mappings = feature_info.get("column_mappings", {}) or {}

    # 获取 compute_func 以检查函数签名
    from src.features.registry import get_compute_func
    compute_func_name = feature_info.get("compute_func", feature_name)
    func_sig = None
    try:
        compute_func = get_compute_func(compute_func_name)
        import inspect
        func_sig = inspect.signature(compute_func)
        first_param = list(func_sig.parameters.values())[0] if func_sig.parameters else None
        # 如果第一个参数是 'df' 且没有显式配置 positional_params，自动添加
        auto_df_positional = (
            first_param is not None
            and first_param.name == "df"
            and first_param.kind != inspect.Parameter.KEYWORD_ONLY
            and "positional_params" not in feature_info
        )
    except Exception:
        auto_df_positional = False
        compute_func = None

    # 构建位置参数（按顺序）
    args = []
    positional_params = feature_info.get("positional_params", [])
    if auto_df_positional:
        # 自动检测：如果函数第一个参数是 df，且没有显式配置，自动添加
        args.append(df)
    else:
        for param_name in positional_params:
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

    # 构建关键字参数（来自 compute_params）
    # 排除非函数参数的配置项（如 ticks_dir, lookback_minutes，这些只用于自动生成 ticks_loader_json）
    excluded_params = {"ticks_dir", "lookback_minutes"}
    kwargs: Dict[str, Any] = {}
    for param_name, param_value in compute_params.items():
        if param_name in excluded_params:
            # 跳过这些配置项，它们不是函数参数
            continue
        if param_name not in feature_info.get("positional_params", []):
            # 对于 compute_params，我们保持保守策略：大多数字符串参数是枚举/配置，
            # 而不是列名（如 price_col="close"），真正的列映射由 column_mappings 处理。
            # 只有当参数名本身就是列名时，才从 DataFrame 提取对应列。
            if isinstance(param_value, str) and param_name in df.columns:
                kwargs[param_name] = df[param_name]
            else:
                kwargs[param_name] = param_value

    # 自动生成 ticks_loader_json（如果函数需要 ticks_loader_json/ticks，且未显式提供）
    try:
        needs_ticks_loader = False
        if func_sig is not None:
            needs_ticks_loader = (
                "ticks_loader_json" in func_sig.parameters
                or "ticks" in func_sig.parameters
            )
        if needs_ticks_loader:
            tlj = compute_params.get("ticks_loader_json")
            if tlj is None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                # 尝试自动生成：调用 list_tick_files 获取 tick 文件列表，再序列化
                from src.data_tools.tick_loader import (
                    list_tick_files,
                    serialize_tick_loader_params,
                )

                ticks_dir = compute_params.get("ticks_dir", "data/parquet_data")
                lookback_minutes = compute_params.get("lookback_minutes", 60)
                start_ts = str(df.index.min())
                end_ts = str(df.index.max())

                symbol = None
                if "_symbol" in df.columns:
                    sym_series = df["_symbol"].dropna()
                    if len(sym_series) > 0:
                        symbol = str(sym_series.iloc[0])
                elif "symbol" in df.columns:
                    sym_series = df["symbol"].dropna()
                    if len(sym_series) > 0:
                        symbol = str(sym_series.iloc[0])

                try:
                    tick_files = list_tick_files(
                        symbol=symbol or "",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        ticks_dir=ticks_dir,
                        lookback_minutes=lookback_minutes,
                    )
                    if tick_files:
                        tlj_obj = {
                            "symbol": symbol or "",
                            "tick_files": tick_files,
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "lookback_minutes": lookback_minutes,
                        }
                        tlj = serialize_tick_loader_params(tlj_obj)
                except Exception:
                    tlj = None

            if tlj is not None:
                kwargs["ticks_loader_json"] = tlj
    except Exception:
        pass

    # 处理 column_mappings：将 DataFrame 指定列注入到函数参数
    for param_name, source in column_mappings.items():
        if isinstance(source, str):
            col_name = source
            if col_name not in df.columns:
                raise KeyError(
                    f"Column '{col_name}' required for parameter '{param_name}' not "
                    f"found in DataFrame when building call args for feature "
                    f"'{feature_name}'"
                )
            kwargs[param_name] = df[col_name]
        elif isinstance(source, list):
            missing = [col for col in source if col not in df.columns]
            if missing:
                raise KeyError(
                    f"Columns {missing} required for parameter '{param_name}' not "
                    f"found in DataFrame when building call args for feature "
                    f"'{feature_name}'"
                )
            # 对于多列映射，按列名子 DataFrame 传入
            kwargs[param_name] = df[source]
        else:
            raise ValueError(
                f"Unsupported column mapping type for parameter '{param_name}': "
                f"{type(source)}"
            )

    return (tuple(args), kwargs)


def _get_monthly_cache_key(
    feature_name: str,
    month_key: str,
    cache_version: str,
    fit: bool,
    compute_params: Dict,
    *,
    data_id: str = "unknown",
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
    # Build a stable param signature.
    # IMPORTANT: strip volatile runtime params (e.g. ticks_loader_json) that would
    # destroy cache hit-rate across processes/runs for the same month.
    volatile_keys = {
        "ticks_loader_json",
        "ticks_dir",
        "lookback_minutes",
    }
    stable_items = []
    try:
        for k, v in sorted((compute_params or {}).items()):
            if k in volatile_keys:
                continue
            stable_items.append((k, v))
    except Exception:
        stable_items = []
    params_sig = hashlib.md5(str(stable_items).encode()).hexdigest()[:8]

    # 构建缓存键
    key_parts = [
        feature_name,
        month_key,
        data_id,
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
        # - cache_hits: {"memory": int, "monthly": int, "memory_features": List[str], "monthly_features": List[str]}
        self._debug_stats: Dict[str, Any] = {
            "index_mismatch": {},
            "cache_hits": {
                "memory": 0,
                "monthly": 0,
                "memory_features": [],
                "monthly_features": [],
            },
        }

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
        self._debug_stats = {
            "index_mismatch": {},
            "cache_hits": {
                "memory": 0,
                "monthly": 0,
                "memory_features": [],
                "monthly_features": [],
            },
        }
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

        # Stable dataset id for cache keys.
        # Goal: cache should hit across runs even if caller passes a larger warmup window,
        # as long as the per-month slice is identical for the same symbol/timeframe.
        sym = None
        try:
            if "_symbol" in df.columns:
                uniq = pd.Series(df["_symbol"]).dropna().astype(str).unique().tolist()
                if len(uniq) == 1:
                    sym = uniq[0]
                elif len(uniq) > 1:
                    sym = "multi_" + hashlib.md5(
                        ("|".join(sorted(uniq))).encode()
                    ).hexdigest()[:8]
            if sym is None and "symbol" in df.columns:
                uniq = pd.Series(df["symbol"]).dropna().astype(str).unique().tolist()
                if len(uniq) == 1:
                    sym = uniq[0]
                elif len(uniq) > 1:
                    sym = "multi_" + hashlib.md5(
                        ("|".join(sorted(uniq))).encode()
                    ).hexdigest()[:8]
        except Exception:
            sym = None
        data_id = f"sym={sym or 'unknown'}"

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
                cache_version=self.cache_version,
                fit=True,  # 月度缓存不区分 fit/predict
                compute_params=compute_params,
                data_id=data_id,
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
                # 记录月度缓存命中（每个特征只记录一次）
                if feature_name not in self._debug_stats["cache_hits"]["monthly_features"]:
                    self._debug_stats["cache_hits"]["monthly_features"].append(feature_name)
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
                f"       💾 [monthly] Used {cached_months} cached months, computed {computed_months} new months"
            )
            # 更新月度缓存命中总数（按月份数累加）
            self._debug_stats["cache_hits"]["monthly"] += cached_months

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

            # 在同一层内做一个简单的拓扑顺序：优先计算依赖已完成的特征
            pending = list(level_features)
            safeguard = len(pending) * 2  # 防止意外死循环
            while pending and safeguard > 0:
                feature_name = pending.pop(0)
                safeguard -= 1

                if feature_name in computed_features:
                    continue

                # 如果依赖未满足，放到队尾，等待依赖先计算
                deps = features.get(feature_name, {}).get("dependencies", [])
                if any(dep not in computed_features for dep in deps):
                    pending.append(feature_name)
                    continue

                feature_info = features[feature_name]
                # 优先使用 feature_dependencies.yaml 中显式配置的 compute_func，
                # 否则退回到以 feature_name 作为注册名（兼容旧配置）。
                compute_func_name = feature_info.get("compute_func", feature_name)
                compute_func = get_compute_func(compute_func_name)

                if compute_func is None:
                    print(
                        f"     ⚠️  Skipping {feature_name}: compute function "
                        f"'{compute_func_name}' not found in registry"
                    )
                    continue

                # 检查内存缓存：使用 (df_signature, feature_name) 作为键
                cache_key = None
                cached_result = None
                if self.use_memory_cache and current_df_sig is not None:
                    cache_key = (current_df_sig, feature_name)
                    if cache_key in self.memory_cache:
                        print(
                            f"     💾 [memory] Using memory cache for {feature_name} "
                            f"(df_sig={current_df_sig[0][:8]}...)"
                        )
                        # 记录内存缓存命中
                        self._debug_stats["cache_hits"]["memory"] += 1
                        if feature_name not in self._debug_stats["cache_hits"]["memory_features"]:
                            self._debug_stats["cache_hits"]["memory_features"].append(feature_name)
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
                    # Always use monthly cache path (even for small DataFrames).
                    # Rationale: the goal of monthly caching is cross-run reuse; bypassing it
                    # destroys cache hit-rate in multi-seed / multi-step workflows.
                    combined_result = self._compute_and_cache_monthly(
                        feature_name,
                        result_df,
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
                        feature_info, result_df, feature_name
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
