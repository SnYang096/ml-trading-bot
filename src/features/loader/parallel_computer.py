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
    feature_info: Dict, df: pd.DataFrame, ticks_loader_json: Optional[str] = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    根据特征配置构建 compute_func 所需的 args/kwargs.
    支持配置 column_mappings，将 DataFrame 指定列注入到函数参数。
    支持从 ticks_loader_json 加载 ticks 数据。
    """
    compute_params = feature_info.get("compute_params", {}) or {}
    column_mappings = feature_info.get("column_mappings", {}) or {}
    
    # 如果 ticks_loader_json 参数为 None，尝试从 compute_params 获取
    if ticks_loader_json is None:
        ticks_loader_json = compute_params.get("ticks_loader_json")
    
    # 复制 compute_params，但排除 ticks_loader_json（它只是配置，不是函数参数）
    call_kwargs = {k: v for k, v in compute_params.items() if k != "ticks_loader_json"}

    # 处理 ticks_loader_json：如果函数需要 ticks 或 ticks_loader_json 参数
    import inspect
    from src.features.loader.feature_function_mapping import get_compute_func
    compute_func_name = feature_info.get("compute_func")
    if compute_func_name:
        compute_func = get_compute_func(compute_func_name)
        func_sig = inspect.signature(compute_func)
        
        # 如果函数同时接受 ticks 和 ticks_loader_json，优先传递 ticks_loader_json
        # 因为某些函数（如 extract_order_flow_features）可以自己处理 ticks_loader_json
        has_ticks_param = "ticks" in func_sig.parameters
        has_ticks_loader_json_param = "ticks_loader_json" in func_sig.parameters
        
        if has_ticks_loader_json_param:
            # 函数支持 ticks_loader_json，直接传递（优先）
            if ticks_loader_json:
                call_kwargs["ticks_loader_json"] = ticks_loader_json
            # 注意：如果 ticks_loader_json 是 None，不传递（让函数抛出错误，便于调试）
        elif has_ticks_param and ticks_loader_json:
            # 函数只支持 ticks 参数，需要加载 ticks 数据
            from src.data_tools.tick_loader import deserialize_tick_loader_params, load_tick_data
            try:
                tick_params = deserialize_tick_loader_params(ticks_loader_json)
                # 根据 df 的时间范围加载 ticks
                if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                    start_ts = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
                    end_ts = df.index.max().strftime("%Y-%m-%d %H:%M:%S")
                    # 从 tick_params 中获取 ticks_dir
                    ticks_dir = tick_params.get("ticks_dir")
                    if not ticks_dir:
                        # 尝试从 tick_files 推断
                        tick_files = tick_params.get("tick_files", [])
                        if tick_files:
                            from pathlib import Path
                            ticks_dir = str(Path(tick_files[0]).parent)
                        else:
                            ticks_dir = "data/parquet_data"
                    
                    ticks = load_tick_data(
                        symbol=tick_params["symbol"],
                        start_ts=start_ts,
                        end_ts=end_ts,
                        ticks_dir=ticks_dir,
                        lookback_minutes=tick_params.get("lookback_minutes", 60),
                    )
                    if ticks is not None and len(ticks) > 0:
                        call_kwargs["ticks"] = ticks
                    else:
                        print(f"     ⚠️  No ticks loaded for {compute_func_name} (time range: {start_ts} to {end_ts})")
                        # 如果加载失败，但函数支持 ticks_loader_json，传递 ticks_loader_json 作为 fallback
                        if has_ticks_loader_json_param:
                            call_kwargs["ticks_loader_json"] = ticks_loader_json
            except Exception as e:
                print(f"     ⚠️  Failed to load ticks for {compute_func_name}: {e}")
                import traceback
                traceback.print_exc()
                # 如果加载失败，但函数支持 ticks_loader_json，传递 ticks_loader_json 作为 fallback
                if has_ticks_loader_json_param:
                    call_kwargs["ticks_loader_json"] = ticks_loader_json

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


def _compute_single_feature_worker_monthly(
    feature_name: str,
    feature_info: Dict,
    df_bytes: bytes,
    fit: bool,
    monthly_cache_dir: Optional[str],
    ticks_loader_json: Optional[str] = None,
) -> Tuple[str, bytes]:
    """
    工作进程函数：按月计算单个特征
    
    Args:
        feature_name: 特征名
        feature_info: 特征配置信息
        df_bytes: DataFrame 的 pickle 字节
        fit: 是否拟合
        monthly_cache_dir: 按月缓存目录
    
    Returns:
        (feature_name, result_df_bytes)
    """
    import pandas as pd
    from src.features.loader.feature_function_mapping import get_compute_func
    from pathlib import Path
    import hashlib
    
    # 反序列化 DataFrame
    df = pickle.loads(df_bytes)
    
    # 按月份拆分
    def _split_df_by_month(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按月份拆分DataFrame"""
        if df.empty or not hasattr(df.index, 'to_period'):
            return {"all": df}
        monthly_dfs = {}
        try:
            for period, group in df.groupby(df.index.to_period('M')):
                month_key = str(period)
                monthly_dfs[month_key] = group
        except Exception:
            return {"all": df}
        return monthly_dfs
    
    def _get_monthly_cache_key(feature_name: str, month_key: str, params: Dict, feature_info: Dict) -> str:
        """生成按月缓存的键（模块级函数，用于 worker 进程）"""
        params_str = str(sorted(params.items()))
        output_cols_str = ""
        if feature_info:
            output_cols = feature_info.get("output_columns", [feature_name])
            output_cols_str = str(sorted(output_cols))
        # v5: 改进错误处理和流程验证，添加索引对齐检查
        # 注意：这是模块级函数，无法访问实例的 cache_version，所以使用硬编码版本
        code_version = "v5"
        key_str = f"{feature_name}_monthly_{month_key}_{params_str}_{output_cols_str}_{code_version}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_monthly_cache(cache_dir: Path, cache_key: str) -> Optional[pd.DataFrame | pd.Series]:
        """从按月缓存加载"""
        if not cache_dir:
            return None
        cache_file = cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None
    
    def _save_monthly_cache(cache_dir: Path, cache_key: str, result: pd.DataFrame | pd.Series):
        """保存到按月缓存"""
        if not cache_dir:
            return
        cache_file = cache_dir / f"{cache_key}.pkl"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass
    
    # 按月计算
    monthly_dfs = _split_df_by_month(df)
    compute_params = feature_info.get("compute_params", {})
    compute_func_name = feature_info["compute_func"]
    compute_func = get_compute_func(compute_func_name)
    cache_dir = Path(monthly_cache_dir) if monthly_cache_dir else None
    
    # 检查函数是否支持 monthly_cache_dir 参数，如果支持则自动注入
    import inspect
    func_sig = inspect.signature(compute_func)
    supports_monthly_cache = "monthly_cache_dir" in func_sig.parameters
    
    monthly_results = {}
    for month_key, month_df in monthly_dfs.items():
        if month_key == "all":
            # 无法按月拆分，直接计算
            call_args, call_kwargs = _build_call_args(feature_info, month_df, ticks_loader_json)
            # 如果函数支持 monthly_cache_dir，自动注入
            if supports_monthly_cache and monthly_cache_dir:
                call_kwargs["monthly_cache_dir"] = monthly_cache_dir
            month_result = compute_func(*call_args, **call_kwargs)
        else:
            # 检查缓存
            monthly_cache_key = _get_monthly_cache_key(feature_name, month_key, compute_params, feature_info)
            cached_result = _load_monthly_cache(cache_dir, monthly_cache_key)
            
            if cached_result is not None:
                month_result = cached_result
                # 处理从缓存加载的数据中的重复列名
                if isinstance(month_result, pd.DataFrame) and month_result.columns.duplicated().any():
                    month_result = month_result.loc[:, ~month_result.columns.duplicated()]
                
                # 根本性解决方案：只提取 output_columns 中定义的列
                output_cols = feature_info.get("output_columns", [feature_name])
                if not output_cols:
                    output_cols = [feature_name]
                
                if isinstance(month_result, pd.DataFrame):
                    # 只保留 output_columns 中定义的列
                    result_cols = [col for col in output_cols if col in month_result.columns]
                    if result_cols:
                        month_result = month_result[result_cols].copy()
                    else:
                        month_result = pd.DataFrame(index=month_result.index)
                elif isinstance(month_result, pd.Series):
                    series_name = month_result.name if month_result.name else feature_name
                    if series_name not in output_cols:
                        # 如果不在 output_columns 中，返回空 DataFrame
                        month_result = pd.DataFrame(index=month_result.index)
            else:
                # 计算该月份
                call_args, call_kwargs = _build_call_args(feature_info, month_df, ticks_loader_json)
                # 如果函数支持 monthly_cache_dir，自动注入
                if supports_monthly_cache and monthly_cache_dir:
                    call_kwargs["monthly_cache_dir"] = monthly_cache_dir
                month_result = compute_func(*call_args, **call_kwargs)
                # 处理计算结果的重复列名
                if isinstance(month_result, pd.DataFrame) and month_result.columns.duplicated().any():
                    month_result = month_result.loc[:, ~month_result.columns.duplicated()]
                
                # 根本性解决方案：只提取 output_columns 中定义的列
                output_cols = feature_info.get("output_columns", [feature_name])
                if not output_cols:
                    output_cols = [feature_name]
                
                if isinstance(month_result, pd.DataFrame):
                    # 只保留 output_columns 中定义的列
                    result_cols = [col for col in output_cols if col in month_result.columns]
                    if result_cols:
                        month_result = month_result[result_cols].copy()
                    else:
                        month_result = pd.DataFrame(index=month_result.index)
                elif isinstance(month_result, pd.Series):
                    series_name = month_result.name if month_result.name else feature_name
                    if series_name not in output_cols:
                        # 如果不在 output_columns 中，返回空 DataFrame
                        month_result = pd.DataFrame(index=month_result.index)
                
                # 保存缓存（只保存 output_columns）
                _save_monthly_cache(cache_dir, monthly_cache_key, month_result)
        
        monthly_results[month_key] = month_result
        # 打印每个月份的结果信息
        if isinstance(month_result, pd.DataFrame):
            print(f"       📊 Month {month_key}: {len(month_result)} rows, {len(month_result.columns)} columns: {list(month_result.columns)[:5]}...")
            if month_result.columns.duplicated().any():
                dup_cols = month_result.columns[month_result.columns.duplicated()].tolist()
                print(f"       ⚠️  Month {month_key} has duplicate columns: {dup_cols}")
        elif isinstance(month_result, pd.Series):
            print(f"       📊 Month {month_key}: {len(month_result)} rows, Series name: {month_result.name}")
    
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
                combined_series = pd.concat([r[i] for r in monthly_results.values()], axis=0).sort_index()
                combined_results.append(combined_series)
            result_df = pd.DataFrame({
                col: series for col, series in zip(output_cols, combined_results)
            })
        elif isinstance(list(monthly_results.values())[0], pd.DataFrame):
            # 根本性解决方案：只保留 output_columns 中定义的列
            all_columns = set(output_cols)  # 只使用 output_columns，不包含其他列
            
            # 确保所有 DataFrame 都有相同的列（缺失的列填充 NaN）
            aligned_results = []
            for month_key, month_result in monthly_results.items():
                # 处理重复列名：如果有重复列，保留第一个
                if isinstance(month_result, pd.DataFrame) and month_result.columns.duplicated().any():
                    dup_cols = month_result.columns[month_result.columns.duplicated()].tolist()
                    print(f"       ⚠️  Month {month_key} has duplicate columns before dedup: {dup_cols}")
                    month_result = month_result.loc[:, ~month_result.columns.duplicated()]
                    print(f"       ✅ Month {month_key} after dedup: {len(month_result.columns)} columns")
                    # 更新字典中的值
                    monthly_results[month_key] = month_result
                
                # 只提取 output_columns 中定义的列
                if isinstance(month_result, pd.DataFrame):
                    result_cols = [col for col in output_cols if col in month_result.columns]
                    if result_cols:
                        month_result_filtered = month_result[result_cols].copy()
                    else:
                        month_result_filtered = pd.DataFrame(index=month_result.index)
                    print(f"       🔄 Aligning month {month_key}: {len(month_result.columns)} -> {len(all_columns)} columns (only output_columns)")
                    aligned_df = month_result_filtered.reindex(columns=sorted(all_columns))
                    print(f"       ✅ Month {month_key} aligned: {len(aligned_df)} rows, {len(aligned_df.columns)} columns")
                else:
                    aligned_df = month_result
                aligned_results.append(aligned_df)
            
            print(f"       🔗 Concatenating {len(aligned_results)} aligned DataFrames...")
            result_df = pd.concat(aligned_results, axis=0).sort_index()
            print(f"       ✅ Merged result: {len(result_df)} rows, {len(result_df.columns)} columns (only output_columns)")
        elif isinstance(list(monthly_results.values())[0], pd.Series):
            result_df = pd.concat(monthly_results.values(), axis=0).sort_index()
            series_name = result_df.name if hasattr(result_df, 'name') and result_df.name else feature_name
            # 只保留 output_columns 中定义的列
            if series_name in output_cols:
                if len(output_cols) == 1:
                    result_df = pd.DataFrame({output_cols[0]: result_df})
                else:
                    result_df = pd.DataFrame({feature_name: result_df})
            else:
                # 如果不在 output_columns 中，返回空 DataFrame
                result_df = pd.DataFrame(index=result_df.index)
        else:
            # Fallback
            result_df = pd.concat([pd.DataFrame({feature_name: r}) if isinstance(r, pd.Series) else r 
                                  for r in monthly_results.values()], axis=0).sort_index()
    except Exception as e:
        # 打印详细的错误信息
        print(f"       ❌ Error merging monthly results for {feature_name}: {e}")
        print(f"       📊 Monthly results info:")
        for month_key, month_result in monthly_results.items():
            if isinstance(month_result, pd.DataFrame):
                print(f"          Month {month_key}: shape={month_result.shape}, columns={list(month_result.columns)[:10]}...")
                if month_result.columns.duplicated().any():
                    dup_cols = month_result.columns[month_result.columns.duplicated()].tolist()
                    print(f"          Month {month_key} duplicate columns: {dup_cols}")
            elif isinstance(month_result, pd.Series):
                print(f"          Month {month_key}: Series, length={len(month_result)}, name={month_result.name}")
        import traceback
        traceback.print_exc()
        # 如果合并失败，尝试直接计算
        # 注意：这里无法获取 ticks_loader_json，可能需要从 compute_params 中获取
        ticks_loader_json = compute_params.get("ticks_loader_json")
        call_args, call_kwargs = _build_call_args(feature_info, df, ticks_loader_json)
        feature_result = compute_func(*call_args, **call_kwargs)
        if isinstance(feature_result, tuple):
            output_cols = feature_info.get("output_columns", [feature_name])
            result_df = pd.DataFrame({
                col: series for col, series in zip(output_cols, feature_result)
            }, index=df.index)
        elif isinstance(feature_result, pd.DataFrame):
            result_df = feature_result
        elif isinstance(feature_result, pd.Series):
            output_cols = feature_info.get("output_columns", [feature_name])
            result_df = pd.DataFrame({output_cols[0] if output_cols else feature_name: feature_result}, index=df.index)
        else:
            result_df = pd.DataFrame({feature_name: feature_result}, index=df.index)
    
    return (feature_name, pickle.dumps(result_df))


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
        use_monthly_cache: bool = True,  # 是否使用按月缓存
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
        
        # 智能计算并行进程数（基于可用内存）
        # 检查环境变量，允许强制单进程模式
        force_single_process = os.environ.get("FORCE_SINGLE_PROCESS", "").lower() in ("1", "true", "yes")
        
        if max_workers is None:
            if force_single_process:
                print("   🔧 Force single-process mode (FORCE_SINGLE_PROCESS=1)")
                self.max_workers = 1
            else:
                # 先打印内存信息，再计算最优进程数
                self._print_memory_info()
                self.max_workers = self._calculate_optimal_workers()
        else:
            self.max_workers = max_workers
            # 即使指定了 max_workers，也打印内存信息用于调试
            self._print_memory_info()
            print(f"   🔧 Using {self.max_workers} worker(s) (manually specified)")
        
        self.parallel_backend = parallel_backend
        
        # 内存缓存
        self.memory_cache = {}
        
        # 缓存版本控制（用于失效旧缓存）
        # 当特征计算逻辑改变时，更新此版本号以失效旧缓存
        # v5: 改进错误处理和流程验证，添加索引对齐检查
        self.cache_version = "v5"
        
        # 在初始化完成后再次打印内存信息（确保可见）
        print("=" * 60)
        print("🔧 ParallelFeatureComputer Initialized")
        self._print_memory_info()
        print(f"   🔧 Workers: {self.max_workers}, Backend: {self.parallel_backend}")
        print("=" * 60)
        
        # 按月缓存目录
        if self.use_monthly_cache and self.cache_dir:
            self.monthly_cache_dir = self.cache_dir / "monthly"
            self.monthly_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查版本号变化，自动清理旧缓存
            self._check_and_cleanup_old_cache()
        else:
            self.monthly_cache_dir = None
        
        # 并行执行器
        if parallel_backend == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        elif parallel_backend == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """生成 DataFrame 哈希（仅用于调试，不再用于缓存键）"""
        if df.empty:
            return "EMPTY"
        try:
            start_meta = str(df.index[0])
            end_meta = str(df.index[-1])
            return f"{start_meta}_{end_meta}"
        except Exception:
            return "NO_INDEX"
    
    def _validate_cache_quality(
        self,
        data: pd.DataFrame | pd.Series,
        feature_name: str,
        cache_type: str = "cache",
        warn_threshold_nan: float = 0.5,
        warn_threshold_inf: float = 0.1,
    ) -> Dict[str, Any]:
        """
        验证cache数据的质量
        
        Args:
            data: 要验证的数据（DataFrame或Series）
            feature_name: 特征名称
            cache_type: cache类型（"memory" 或 "monthly"）
            warn_threshold_nan: NaN值占比警告阈值（默认0.5=50%）
            warn_threshold_inf: inf值占比警告阈值（默认0.1=10%）
        
        Returns:
            质量检查结果字典
        """
        result = {
            "feature_name": feature_name,
            "cache_type": cache_type,
            "total_values": 0,
            "nan_count": 0,
            "nan_pct": 0.0,
            "inf_count": 0,
            "inf_pct": 0.0,
            "has_issues": False,
            "warnings": [],
        }
        
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame(name=feature_name)
            
            if data.empty:
                result["warnings"].append("⚠️  Cache is empty")
                result["has_issues"] = True
                return result
            
            # 只检查数值类型的列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                # 没有数值列，跳过验证（可能是纯object类型的数据）
                return result
            
            # 只对数值列进行统计
            numeric_data = data[numeric_cols]
            total_values = numeric_data.size
            
            if total_values == 0:
                result["warnings"].append("⚠️  Cache has no numeric values")
                result["has_issues"] = True
                return result
            
            # 统计NaN和inf（只对数值类型）
            nan_count = numeric_data.isna().sum().sum()
            inf_count = 0
            
            # 检查inf值（需要转换为numpy数组）
            for col in numeric_cols:
                col_values = numeric_data[col].values
                # 只对数值类型使用isinf
                try:
                    inf_mask = np.isinf(col_values)
                    inf_count += np.sum(inf_mask)
                except (TypeError, ValueError):
                    # 如果无法检查inf（非数值类型），跳过
                    pass
            
            nan_pct = (nan_count / total_values) * 100
            inf_pct = (inf_count / total_values) * 100
            
            result["total_values"] = total_values
            result["nan_count"] = int(nan_count)
            result["nan_pct"] = nan_pct
            result["inf_count"] = int(inf_count)
            result["inf_pct"] = inf_pct
            
            # 检查阈值
            if nan_pct > warn_threshold_nan * 100:
                msg = f"⚠️  {feature_name} ({cache_type} cache): {nan_pct:.1f}% NaN values (threshold: {warn_threshold_nan*100:.0f}%)"
                result["warnings"].append(msg)
                result["has_issues"] = True
                print(f"     {msg}")
            
            if inf_pct > warn_threshold_inf * 100:
                msg = f"⚠️  {feature_name} ({cache_type} cache): {inf_pct:.1f}% inf values (threshold: {warn_threshold_inf*100:.0f}%)"
                result["warnings"].append(msg)
                result["has_issues"] = True
                print(f"     {msg}")
            
        except Exception as e:
            result["warnings"].append(f"⚠️  Error validating cache quality: {e}")
            result["has_issues"] = True
            print(f"     ⚠️  {feature_name} ({cache_type} cache): Validation error: {e}")
        
        return result
    
    def _split_df_by_month(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按月份拆分DataFrame"""
        if df.empty or not hasattr(df.index, 'to_period'):
            return {"all": df}
        
        monthly_dfs = {}
        try:
            # 尝试按月份分组
            for period, group in df.groupby(df.index.to_period('M')):
                month_key = str(period)
                monthly_dfs[month_key] = group
        except Exception:
            # 如果无法按月份分组，返回整个DataFrame
            return {"all": df}
        
        return monthly_dfs
    
    def _get_monthly_cache_key(
        self, feature_name: str, month_key: str, params: Dict, feature_info: Optional[Dict] = None
    ) -> str:
        """生成按月缓存的键"""
        params_str = str(sorted(params.items()))
        output_cols_str = ""
        if feature_info:
            output_cols = feature_info.get("output_columns", [feature_name])
            output_cols_str = str(sorted(output_cols))
        # 使用实例的缓存版本（而不是硬编码）
        # v5: 改进错误处理和流程验证，添加索引对齐检查
        code_version = getattr(self, 'cache_version', 'v5')
        key_str = f"{feature_name}_monthly_{month_key}_{params_str}_{output_cols_str}_{code_version}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_monthly_cache(self, cache_key: str) -> Optional[pd.DataFrame | pd.Series]:
        """从按月缓存加载"""
        if not self.monthly_cache_dir:
            return None
        cache_file = self.monthly_cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                return None
        return None
    
    def _save_monthly_cache(self, cache_key: str, result: pd.DataFrame | pd.Series):
        """保存到按月缓存"""
        if not self.monthly_cache_dir:
            return
        cache_file = self.monthly_cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            pass
    
    def _try_monthly_cache(
        self,
        feature_name: str,
        df: pd.DataFrame,
        compute_params: Dict,
        feature_info: Dict,
    ) -> Optional[Dict[str, pd.DataFrame | pd.Series]]:
        """
        尝试使用按月缓存
        
        Returns:
            Dict[month_key, result] 如果所有月份都有缓存，否则None
        """
        if df.empty or not self.monthly_cache_dir:
            return None
        
        # 按月份拆分
        monthly_dfs = self._split_df_by_month(df)
        if len(monthly_dfs) <= 1:
            # 无法按月拆分或只有一个月，不使用按月缓存
            return None
        
        # 检查每个月的缓存
        monthly_results = {}
        missing_months = []
        
        for month_key, month_df in monthly_dfs.items():
            if month_key == "all":
                # 无法按月拆分，不使用按月缓存
                return None
            
            monthly_cache_key = self._get_monthly_cache_key(
                feature_name, month_key, compute_params, feature_info
            )
            cached_result = self._load_monthly_cache(monthly_cache_key)
            
            if cached_result is not None:
                monthly_results[month_key] = cached_result
            else:
                missing_months.append(month_key)
        
        if missing_months:
            # 有月份缺失缓存，返回None（将使用全量计算）
            return None
        
        # 所有月份都有缓存
        return monthly_results
    
    def _compute_and_cache_monthly(
        self,
        feature_name: str,
        df: pd.DataFrame,
        compute_params: Dict,
        feature_info: Dict,
        compute_func: Callable,
    ) -> pd.DataFrame | pd.Series:
        """
        按月计算特征并缓存
        
        Returns:
            合并后的特征结果
        """
        if df.empty:
            return df
        
        # 获取 ticks_loader_json
        ticks_loader_json = compute_params.get("ticks_loader_json")
        if not ticks_loader_json and feature_name in ["vpin_features", "footprint_basic"]:
            print(f"     ⚠️  Warning: {feature_name} needs ticks_loader_json but it's not in compute_params")
            print(f"     compute_params keys: {list(compute_params.keys())}")
        
        # 优化：单进程模式下，如果数据量不大，直接全量计算（避免按月拆分的开销）
        use_monthly_split = True
        if self.max_workers == 1 and PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024 ** 3)
                # 如果可用内存 > 20GB 且数据量 < 100万行，直接全量计算
                if available_gb > 20 and len(df) < 1_000_000:
                    use_monthly_split = False
                    print(f"     ⚡ Using full-data computation (memory: {available_gb:.1f}GB available, data: {len(df)} rows)")
            except Exception:
                pass
        
        # 按月份拆分（如果启用）
        if use_monthly_split:
            monthly_dfs = self._split_df_by_month(df)
            if len(monthly_dfs) <= 1:
                # 无法按月拆分，使用全量计算
                call_args, call_kwargs = _build_call_args(feature_info, df, ticks_loader_json)
                return compute_func(*call_args, **call_kwargs)
        else:
            # 不使用按月拆分，直接全量计算
            monthly_dfs = {"all": df}
        
        # 按月计算
        monthly_results = {}
        for month_key, month_df in monthly_dfs.items():
            if month_key == "all":
                # 无法按月拆分或禁用按月拆分，使用全量计算
                call_args, call_kwargs = _build_call_args(feature_info, df, ticks_loader_json)
                return compute_func(*call_args, **call_kwargs)
            
            # 检查缓存
            monthly_cache_key = self._get_monthly_cache_key(
                feature_name, month_key, compute_params, feature_info
            )
            cached_result = self._load_monthly_cache(monthly_cache_key)
            
            if cached_result is not None:
                monthly_results[month_key] = cached_result
            else:
                # 计算该月份
                call_args, call_kwargs = _build_call_args(feature_info, month_df, ticks_loader_json)
                month_result = compute_func(*call_args, **call_kwargs)
                monthly_results[month_key] = month_result
                # 保存缓存
                self._save_monthly_cache(monthly_cache_key, month_result)
        
        # 合并所有月份的结果
        # 处理不同的返回类型（tuple, DataFrame, Series）
        if not monthly_results:
            return df
        
        first_result = list(monthly_results.values())[0]
        if isinstance(first_result, tuple):
            # 如果是tuple，需要分别合并每个元素
            output_cols = feature_info.get("output_columns", [feature_name])
            combined_results = []
            for i in range(len(output_cols)):
                combined_series = pd.concat([r[i] for r in monthly_results.values()], axis=0).sort_index()
                combined_results.append(combined_series)
            combined_result = pd.DataFrame({
                col: series for col, series in zip(output_cols, combined_results)
            })
        elif isinstance(first_result, pd.DataFrame):
            # 根本性解决方案：只使用 output_columns 中定义的列
            output_cols = feature_info.get("output_columns", [feature_name])
            if not output_cols:
                output_cols = [feature_name]
            all_columns = set(output_cols)  # 只使用 output_columns
            
            # 确保所有 DataFrame 都有相同的列（缺失的列填充 NaN）
            aligned_results = []
            for month_key, month_result in monthly_results.items():
                # 处理重复列名：如果有重复列，保留第一个
                if isinstance(month_result, pd.DataFrame) and month_result.columns.duplicated().any():
                    month_result = month_result.loc[:, ~month_result.columns.duplicated()]
                    # 更新字典中的值
                    monthly_results[month_key] = month_result
                
                # 只提取 output_columns 中定义的列
                if isinstance(month_result, pd.DataFrame):
                    result_cols = [col for col in output_cols if col in month_result.columns]
                    if result_cols:
                        month_result_filtered = month_result[result_cols].copy()
                    else:
                        month_result_filtered = pd.DataFrame(index=month_result.index)
                    aligned_df = month_result_filtered.reindex(columns=sorted(all_columns))
                else:
                    aligned_df = month_result
                aligned_results.append(aligned_df)
            
            combined_result = pd.concat(aligned_results, axis=0).sort_index()
            
            # 最终确保只包含 output_columns
            final_cols = [col for col in output_cols if col in combined_result.columns]
            if len(final_cols) != len(combined_result.columns):
                combined_result = combined_result[final_cols]
        elif isinstance(first_result, pd.Series):
            combined_result = pd.concat(monthly_results.values(), axis=0).sort_index()
            output_cols = feature_info.get("output_columns", [feature_name])
            if len(output_cols) == 1:
                combined_result = pd.DataFrame({output_cols[0]: combined_result})
            else:
                combined_result = pd.DataFrame({feature_name: combined_result})
        else:
            # Fallback: 尝试转换为 DataFrame
            combined_result = pd.concat([pd.Series(r) if not isinstance(r, (pd.Series, pd.DataFrame)) else r 
                                        for r in monthly_results.values()], axis=0).sort_index()
        
        return combined_result
    
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
        
        # 1.5. 确保所有请求特征的 required_columns 都在 DataFrame 中
        # 收集所有需要的 required_columns
        all_required_columns = set()
        for feature_name in actual_requested:
            if feature_name in features:
                feature_info = features[feature_name]
                required_columns = feature_info.get("required_columns", [])
                all_required_columns.update(required_columns)
        
        # 检查缺失的 required_columns
        missing_required = [col for col in all_required_columns if col not in result_df.columns]
        if missing_required:
            # 检查哪些是基础数据列（可能在原始df中），哪些是特征输出列（会通过依赖关系计算）
            base_data_cols = ["open", "high", "low", "close", "volume", "timestamp", "datetime", "date", "symbol", "_symbol"]
            feature_output_cols = set()
            # 收集所有特征的输出列，包括依赖特征的输出列
            for feature_name, feature_info in features.items():
                output_cols = feature_info.get("output_columns", [feature_name])
                feature_output_cols.update(output_cols)
                # 也检查依赖特征的输出列
                deps = feature_info.get("dependencies", [])
                for dep in deps:
                    if dep in features:
                        dep_output_cols = features[dep].get("output_columns", [dep])
                        feature_output_cols.update(dep_output_cols)
            
            missing_base = [col for col in missing_required if col in base_data_cols]
            missing_features = [col for col in missing_required if col in feature_output_cols and col not in base_data_cols]
            missing_unknown = [col for col in missing_required if col not in base_data_cols and col not in feature_output_cols]
            
            # 尝试从原始输入 df 中获取基础数据列
            for col in missing_required:
                if col in df.columns:
                    result_df[col] = df[col]
                elif col in missing_base:
                    # 只对真正缺失的基础数据列发出警告
                    print(f"   ⚠️  Warning: Required base column '{col}' not found in input DataFrame")
                elif col in missing_features:
                    # 特征输出列会通过依赖关系自动计算，不需要警告
                    pass
                elif col in missing_unknown:
                    # 未知列，可能是配置错误或需要手动计算
                    print(f"   ⚠️  Warning: Required column '{col}' not found and not in feature outputs. Check dependencies.")
        
        # 2. 按层级顺序计算（每层内并行）
        for level in sorted(levels.keys()):
            level_features = levels[level]
            print(
                f"   🔄 Level {level}: Computing {len(level_features)} features in parallel..."
            )
            
            # 记录内存使用情况（在每层开始前）
            self._log_memory_usage(f"before level {level}")
            
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
                    # 验证cache数据质量
                    self._validate_cache_quality(cached_result, feature_name, cache_type="memory")
                    # 合并结果（支持 Series 和 DataFrame）
                    if isinstance(cached_result, pd.DataFrame):
                        new_cols = [
                            c for c in cached_result.columns if c not in result_df.columns
                        ]
                        existing_cols = [c for c in cached_result.columns if c in result_df.columns]
                        
                        # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                        if existing_cols:
                            # 直接跳过，不合并（节省内存和时间）
                            pass
                        
                        # 添加新列
                        if new_cols:
                            result_df = pd.concat([result_df, cached_result[new_cols]], axis=1)
                            # 释放临时对象
                            del cached_result
                            import gc
                            gc.collect()
                    elif isinstance(cached_result, pd.Series):
                        if cached_result.name and cached_result.name not in result_df.columns:
                            result_df[cached_result.name] = cached_result
                        elif feature_name not in result_df.columns:
                            result_df[feature_name] = cached_result
                    continue
                
                feature_info = features[feature_name]
                compute_params = feature_info.get("compute_params", {})
                
                # 调试信息：检查 ticks_loader_json
                if feature_name in ["vpin_features", "footprint_basic"]:
                    if "ticks_loader_json" not in compute_params:
                        print(f"     ⚠️  Warning: {feature_name} compute_params does not contain ticks_loader_json")
                        print(f"     compute_params keys: {list(compute_params.keys())}")
                        print(f"     feature_info keys: {list(feature_info.keys())}")
                
                # 使用按月缓存（如果特征支持）
                use_monthly = self.use_monthly_cache and not feature_info.get("no_monthly_cache", False)
                monthly_results = None
                if use_monthly:
                    monthly_results = self._try_monthly_cache(
                        feature_name, result_df, compute_params, feature_info
                    )
                
                if monthly_results is not None:
                    # 按月缓存成功，合并结果
                    print(f"     💾 Using monthly cache for {feature_name} ({len(monthly_results)} months)")
                    if self.use_memory_cache:
                        # 合并所有月份的结果（处理 tuple, DataFrame, Series）
                        first_result = list(monthly_results.values())[0]
                        if isinstance(first_result, tuple):
                            # 如果是tuple，需要分别合并每个元素
                            output_cols = feature_info.get("output_columns", [feature_name])
                            combined_results = []
                            for i in range(len(output_cols)):
                                combined_series = pd.concat([r[i] for r in monthly_results.values()], axis=0).sort_index()
                                combined_results.append(combined_series)
                            combined_result = pd.DataFrame({
                                col: series for col, series in zip(output_cols, combined_results)
                            })
                        elif isinstance(first_result, pd.DataFrame):
                            # 确保所有月份的 DataFrame 都有相同的列（使用 outer join）
                            # 这样可以处理某些月份可能缺少某些列的情况
                            all_columns = set()
                            for month_result in monthly_results.values():
                                all_columns.update(month_result.columns)
                            
                            # 确保所有 DataFrame 都有相同的列（缺失的列填充 NaN）
                            aligned_results = []
                            for month_key, month_result in monthly_results.items():
                                # 处理重复列名：如果有重复列，保留第一个
                                if isinstance(month_result, pd.DataFrame) and month_result.columns.duplicated().any():
                                    month_result = month_result.loc[:, ~month_result.columns.duplicated()]
                                    # 更新字典中的值
                                    monthly_results[month_key] = month_result
                                
                                # 使用 reindex 确保列对齐，缺失的列自动填充 NaN
                                if isinstance(month_result, pd.DataFrame):
                                    aligned_df = month_result.reindex(columns=sorted(all_columns))
                                else:
                                    aligned_df = month_result
                                aligned_results.append(aligned_df)
                            
                            combined_result = pd.concat(aligned_results, axis=0).sort_index()
                        elif isinstance(first_result, pd.Series):
                            combined_result = pd.concat(monthly_results.values(), axis=0).sort_index()
                            output_cols = feature_info.get("output_columns", [feature_name])
                            if len(output_cols) == 1:
                                combined_result = pd.DataFrame({output_cols[0]: combined_result})
                            else:
                                combined_result = pd.DataFrame({feature_name: combined_result})
                        else:
                            # Fallback: 尝试直接 concat
                            combined_result = pd.concat([pd.Series(r) if not isinstance(r, (pd.Series, pd.DataFrame)) else r 
                                                        for r in monthly_results.values()], axis=0).sort_index()
                        
                        # 验证合并后的cache数据质量
                        self._validate_cache_quality(combined_result, feature_name, cache_type="monthly")
                        self.memory_cache[feature_name] = combined_result
                        # 合并到result_df
                        if isinstance(combined_result, pd.DataFrame):
                            # 处理重复列名：如果有重复列，保留第一个
                            if combined_result.columns.duplicated().any():
                                combined_result = combined_result.loc[:, ~combined_result.columns.duplicated()]
                            
                            new_cols = [
                                c for c in combined_result.columns if c not in result_df.columns
                            ]
                            existing_cols = [c for c in combined_result.columns if c in result_df.columns]
                            
                            # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                            if existing_cols:
                                print(f"        ⏭️  Skipping {len(existing_cols)} existing columns (dropping duplicates): {existing_cols[:10]}...")
                                # 直接跳过，不合并（节省内存和时间）
                            
                            # 添加新列
                            if new_cols:
                                result_df = pd.concat([result_df, combined_result[new_cols]], axis=1)
                                # 释放临时对象
                                del combined_result
                                import gc
                                gc.collect()
                        elif isinstance(combined_result, pd.Series):
                            if combined_result.name and combined_result.name not in result_df.columns:
                                result_df[combined_result.name] = combined_result
                            elif feature_name not in result_df.columns:
                                result_df[feature_name] = combined_result
                    continue
                
                run_sequential = feature_info.get("run_sequential", False) or not self.executor

                # 提交任务
                if not run_sequential:
                    # 并行计算：按月计算并缓存
                    print(f"     🔸 Computing {feature_name} in parallel (monthly)...", flush=True)
                    df_bytes = pickle.dumps(result_df)
                    # 从 compute_params 中获取 ticks_loader_json
                    ticks_loader_json = compute_params.get("ticks_loader_json")
                    future = self.executor.submit(
                        _compute_single_feature_worker_monthly,
                        feature_name,
                        feature_info,
                        df_bytes,
                        fit,
                        str(self.monthly_cache_dir) if self.monthly_cache_dir else None,
                        ticks_loader_json,
                    )
                    futures.append(future)
                else:
                    print(f"     🔸 Running {feature_name} sequentially (monthly)", flush=True)
                    # 串行计算（或被标记为顺序执行）：按月计算并缓存
                    try:
                        compute_func_name = feature_info["compute_func"]
                        compute_func = get_compute_func(compute_func_name)
                        
                        # 使用按月计算（如果特征支持）
                        use_monthly = self.use_monthly_cache and not feature_info.get("no_monthly_cache", False)
                        if use_monthly:
                            feature_result = self._compute_and_cache_monthly(
                                feature_name, result_df, compute_params, feature_info, compute_func
                            )
                        else:
                            # 不支持按月缓存的特征，直接计算
                            ticks_loader_json = compute_params.get("ticks_loader_json")
                            call_args, call_kwargs = _build_call_args(feature_info, result_df, ticks_loader_json)
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
                                # 处理重复列名：如果有重复列，保留第一个
                                if feature_df.columns.duplicated().any():
                                    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
                                
                                new_cols = [c for c in feature_df.columns if c not in result_df.columns]
                                existing_cols = [c for c in feature_df.columns if c in result_df.columns]
                                
                                # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                                if existing_cols:
                                    # 直接跳过，不合并（节省内存和时间）
                                    pass
                                
                                # 添加新列
                                if new_cols:
                                    result_df = pd.concat([result_df, feature_df[new_cols]], axis=1)
                                    # 释放临时对象
                                    del feature_df
                                    import gc
                                    gc.collect()
                            else:
                                # Fallback: add with indexed names
                                for i, series in enumerate(feature_result):
                                    col_name = output_cols[i] if i < len(output_cols) else f"{feature_name}_{i}"
                                    if col_name not in result_df.columns:
                                        result_df[col_name] = series
                        # 如果返回的是 DataFrame，合并新列
                        elif isinstance(feature_result, pd.DataFrame):
                            # 处理重复列名：如果有重复列，保留第一个
                            if feature_result.columns.duplicated().any():
                                feature_result = feature_result.loc[:, ~feature_result.columns.duplicated()]
                            
                            new_cols = [
                                c for c in feature_result.columns if c not in result_df.columns
                            ]
                            existing_cols = [c for c in feature_result.columns if c in result_df.columns]
                            
                            # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                            if existing_cols:
                                # 直接跳过，不合并（节省内存和时间）
                                pass
                            
                            # 添加新列
                            if new_cols:
                                result_df = pd.concat([result_df, feature_result[new_cols]], axis=1)
                                # 释放临时对象
                                del feature_result
                                import gc
                                gc.collect()
                        # 如果返回的是 Series，添加到 DataFrame
                        elif isinstance(feature_result, pd.Series):
                            output_cols = feature_info.get("output_columns", [feature_name])
                            col_name = output_cols[0] if output_cols else (feature_result.name or feature_name)
                            if col_name not in result_df.columns:
                                result_df[col_name] = feature_result
                        
                        # 验证新计算的特征质量
                        self._validate_cache_quality(feature_result, feature_name, cache_type="computed")
                        
                        # 保存内存缓存
                        if self.use_memory_cache:
                            self.memory_cache[feature_name] = feature_result
                        
                        print(f"     ✅ Computed {feature_name}")
                    except Exception as e:
                        print(f"     ❌ Error computing {feature_name}: {e}")
                        import traceback
                        # 提供更详细的错误信息
                        error_type = type(e).__name__
                        error_msg = str(e)
                        print(f"        Error type: {error_type}")
                        print(f"        Error message: {error_msg}")
                        # 打印完整的堆栈跟踪
                        print(f"        Full traceback:")
                        traceback.print_exc()
                        # 如果是特征计算相关的错误，提供诊断信息
                        if "ticks_loader_json" in error_msg.lower() or "tick" in error_msg.lower():
                            print(f"        💡 Tip: This might be related to tick data loading. "
                                  f"Check if ticks_loader_json is properly configured.")
                        elif "required_columns" in error_msg.lower() or "column" in error_msg.lower():
                            print(f"        💡 Tip: This might be related to missing columns. "
                                  f"Check if all required_columns are present in the input DataFrame.")
            
            # 等待所有任务完成（分批处理，避免内存峰值）
            completed_count = 0
            total_futures = len(futures)
            if total_futures > 0:
                print(f"     ⏳ Waiting for {total_futures} feature(s) to complete...", flush=True)
            
            for future in as_completed(futures):
                try:
                    feature_name, result_df_bytes = future.result()
                    completed_count += 1
                    print(f"     📦 Received result for {feature_name} ({completed_count}/{total_futures})...", flush=True)
                    feature_df = pickle.loads(result_df_bytes)
                    
                    # 打印 feature_df 的详细信息
                    if isinstance(feature_df, pd.DataFrame):
                        print(f"        📊 feature_df shape: {feature_df.shape}, columns: {len(feature_df.columns)}")
                        print(f"        📋 feature_df columns: {list(feature_df.columns)[:15]}...")
                        if feature_df.columns.duplicated().any():
                            dup_cols = feature_df.columns[feature_df.columns.duplicated()].tolist()
                            print(f"        ⚠️  feature_df has duplicate columns: {dup_cols}")
                    elif isinstance(feature_df, pd.Series):
                        print(f"        📊 feature_df is Series, length: {len(feature_df)}, name: {feature_df.name}")
                    
                    # 每完成 5 个特征，检查一次内存并清理
                    if completed_count % 5 == 0:
                        self._cleanup_memory()
                    
                    # 合并结果
                    # 如果 feature_df 是 DataFrame，合并新列
                    if isinstance(feature_df, pd.DataFrame):
                        # 处理重复列名：如果有重复列，保留第一个
                        if feature_df.columns.duplicated().any():
                            dup_cols = feature_df.columns[feature_df.columns.duplicated()].tolist()
                            print(f"        🔧 Removing duplicate columns from feature_df: {dup_cols}")
                            feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
                        
                        print(f"        📋 result_df columns before merge: {len(result_df.columns)}")
                        new_cols = [
                            c for c in feature_df.columns if c not in result_df.columns
                        ]
                        existing_cols = [c for c in feature_df.columns if c in result_df.columns]
                        
                        # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                        if existing_cols:
                            print(f"        ⏭️  Skipping {len(existing_cols)} existing columns (dropping duplicates): {existing_cols[:10]}...")
                            # 直接跳过，不合并（节省内存和时间）
                        
                        # 添加新列
                        if new_cols:
                            print(f"        ➕ Adding {len(new_cols)} new columns: {new_cols[:10]}...")
                            result_df = pd.concat([result_df, feature_df[new_cols]], axis=1)
                            print(f"        ✅ result_df columns after merge: {len(result_df.columns)}")
                            # 释放临时对象
                            del feature_df
                            import gc
                            gc.collect()
                        elif existing_cols:
                            print(f"        ✅ Merged existing columns using combine_first")
                        else:
                            print(f"        ℹ️  No new columns to add")
                    # 如果 feature_df 是 Series，添加到 DataFrame
                    elif isinstance(feature_df, pd.Series):
                        if feature_df.name and feature_df.name not in result_df.columns:
                            result_df[feature_df.name] = feature_df
                        elif feature_name not in result_df.columns:
                            result_df[feature_name] = feature_df
                    
                    # 验证并行计算的特征质量
                    self._validate_cache_quality(feature_df, feature_name, cache_type="computed")
                    
                    # 保存内存缓存
                    if self.use_memory_cache:
                        self.memory_cache[feature_name] = feature_df
                    
                    print(f"     ✅ Computed {feature_name}")
                except Exception as e:
                    print(f"     ❌ Error in parallel computation: {e}")
                    import traceback
                    print(f"        Error type: {type(e).__name__}")
                    print(f"        Error message: {str(e)}")
                    print(f"        Feature: {feature_name}")
                    # 打印 feature_df 的信息（如果可用）
                    try:
                        if 'feature_df' in locals():
                            if isinstance(feature_df, pd.DataFrame):
                                print(f"        feature_df shape: {feature_df.shape}")
                                print(f"        feature_df columns: {list(feature_df.columns)[:20]}...")
                                if feature_df.columns.duplicated().any():
                                    dup_cols = feature_df.columns[feature_df.columns.duplicated()].tolist()
                                    print(f"        feature_df duplicate columns: {dup_cols}")
                            elif isinstance(feature_df, pd.Series):
                                print(f"        feature_df is Series, length: {len(feature_df)}, name: {feature_df.name}")
                    except Exception:
                        pass
                    print(f"        Full traceback:")
                    traceback.print_exc()
            
            # 每层完成后清理内存
            self._cleanup_memory()
            self._log_memory_usage(f"after level {level}")

        return result_df
    
    def _print_memory_info(self):
        """打印内存信息（用于调试和监控）"""
        if not PSUTIL_AVAILABLE:
            print("   ⚠️  psutil not available, cannot show memory info")
            return
        
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            percent = mem.percent
            
            print(f"   💾 Memory: {available_gb:.1f}GB available, {used_gb:.1f}GB used, {total_gb:.1f}GB total ({percent:.1f}% used)")
        except Exception as e:
            print(f"   ⚠️  Failed to get memory info: {e}")
    
    def _calculate_optimal_workers(self) -> int:
        """
        基于可用内存智能计算最优并行进程数
        
        策略：
        - 单进程模式：如果内存充足（>50GB），使用单进程模式，分配30GB内存
        - 多进程模式：每个进程估计需要 4GB 内存
        - 保留至少 20% 的系统内存
        - 不超过 CPU 核心数
        """
        cpu_count = mp.cpu_count()
        
        if not PSUTIL_AVAILABLE:
            # 如果没有 psutil，使用保守策略：使用 CPU 核心数的一半
            return max(1, cpu_count // 2)
        
        try:
            # 获取可用内存（GB）
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            
            # 单进程模式：如果总内存 >= 40GB，使用单进程模式
            # 单进程可以使用更多内存（~30GB），避免进程间通信开销，通常更快
            # 降低阈值到 40GB，因为 Docker 容器可能限制了可见内存
            if total_gb >= 40:
                print(f"   🔧 Using single-process mode (1 worker)")
                print(f"   💡 Single process can use ~30GB memory, avoiding inter-process overhead")
                print(f"   💡 This is often faster for large datasets due to better cache locality")
                return 1
            
            # 多进程模式：每个进程估计需要 4GB 内存
            memory_per_worker_gb = 4.0
            
            # 保留至少 20% 的系统内存
            reserved_gb = total_gb * 0.2
            usable_gb = available_gb - reserved_gb
            
            # 计算基于内存的进程数
            memory_based_workers = max(1, int(usable_gb / memory_per_worker_gb))
            
            # 取 CPU 核心数和内存限制的较小值
            optimal_workers = min(cpu_count, memory_based_workers)
            
            # 至少保留 1 个进程，最多不超过 CPU 核心数
            optimal_workers = max(1, min(optimal_workers, cpu_count))
            
            print(f"   🔧 Optimal workers: {optimal_workers} (CPU: {cpu_count}, Memory-based: {memory_based_workers})")
            
            return optimal_workers
        except Exception as e:
            # 如果获取内存信息失败，使用保守策略
            print(f"   ⚠️  Warning: Failed to calculate optimal workers: {e}, using {cpu_count // 2}")
            return max(1, cpu_count // 2)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况（GB）"""
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            mem = psutil.virtual_memory()
            process = psutil.Process()
            process_mem = process.memory_info()
            
            return {
                "total_gb": mem.total / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "used_gb": mem.used / (1024 ** 3),
                "percent": mem.percent,
                "process_rss_gb": process_mem.rss / (1024 ** 3),  # Resident Set Size
                "process_vms_gb": process_mem.vms / (1024 ** 3),  # Virtual Memory Size
            }
        except Exception:
            return {}
    
    def _log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        if not PSUTIL_AVAILABLE:
            return
        
        mem_info = self._get_memory_usage()
        if mem_info:
            print(f"   📊 Memory usage {context}: "
                  f"Process={mem_info.get('process_rss_gb', 0):.2f}GB, "
                  f"System={mem_info.get('used_gb', 0):.1f}GB/{mem_info.get('total_gb', 0):.1f}GB "
                  f"({mem_info.get('percent', 0):.1f}%)")
    
    def _cleanup_memory(self):
        """清理内存（强制垃圾回收）"""
        gc.collect()
        if PSUTIL_AVAILABLE:
            # 尝试清理系统缓存（如果可能）
            try:
                # 触发 Python 的垃圾回收
                collected = gc.collect()
                if collected > 0:
                    print(f"   🧹 Garbage collected {collected} objects")
            except Exception:
                pass

    def clear_cache(self, memory: bool = True, disk: bool = False, old_versions: bool = True):
        """
        清除缓存
        
        Args:
            memory: 是否清除内存缓存
        """
        if memory:
            self.memory_cache.clear()
            print("   🗑️  Memory cache cleared")
        
        if disk and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("   🗑️  Disk cache cleared")
        
        # 自动清理旧版本的按月缓存
        if old_versions and self.monthly_cache_dir:
            self._cleanup_old_version_cache()
    
    def _check_and_cleanup_old_cache(self):
        """
        检查版本号变化，自动清理旧版本的按月缓存文件
        
        工作原理：
        1. 检查版本标记文件（.cache_version）中记录的版本号
        2. 如果版本号改变了（比如从 v4 改为 v5），自动删除所有旧缓存文件
        3. 更新版本标记文件为当前版本
        
        这样，当你更新代码中的 cache_version 时，旧缓存会自动被清理，无需手动删除。
        """
        if not self.monthly_cache_dir or not self.monthly_cache_dir.exists():
            return
        
        current_version = getattr(self, 'cache_version', 'v5')
        version_marker = self.monthly_cache_dir / ".cache_version"
        
        last_version = None
        if version_marker.exists():
            try:
                last_version = version_marker.read_text().strip()
            except Exception:
                pass
        
        # 如果版本号改变了，清理所有旧缓存
        if last_version and last_version != current_version:
            print(f"   🔄 Cache version changed from {last_version} to {current_version}, cleaning old caches...")
            deleted_count = 0
            total_size = 0
            
            for cache_file in self.monthly_cache_dir.glob("*.pkl"):
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    deleted_count += 1
                    total_size += file_size
                except Exception:
                    pass
            
            if deleted_count > 0:
                print(f"   🗑️  Deleted {deleted_count} old cache files ({total_size / 1024 / 1024:.2f} MB)")
            else:
                print(f"   ℹ️  No old cache files to clean")
        
        # 更新版本标记文件（首次运行或版本改变后）
        if not last_version or last_version != current_version:
            try:
                version_marker.write_text(current_version)
            except Exception:
                pass
    
    def _cleanup_old_version_cache(self):
        """
        清理旧版本的按月缓存文件（手动调用）
        
        这个方法在 clear_cache(old_versions=True) 时被调用
        """
        self._check_and_cleanup_old_cache()
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor') and self.executor:
            try:
                self.executor.shutdown(wait=True)  # ensure executor cleaned up
            except Exception:
                pass  # 忽略清理时的错误
