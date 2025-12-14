# Ticks 数据自动加载机制详解

## 背景

某些特征计算函数需要 `ticks` 参数（如 `compute_footprint_features`），但在并行计算时，这些参数需要自动传递。之前的问题是：

1. `compute_footprint_features(df, ticks, ...)` 需要 `ticks` 参数
2. 并行计算器不知道如何获取 `ticks` 数据
3. 导致 `TypeError: missing 1 required positional argument: 'ticks'`

## 解决方案：自动检测和加载

### 1. `ticks_loader_json` 是什么？

`ticks_loader_json` 是一个 JSON 字符串，包含了加载 ticks 数据所需的所有参数：

```python
tick_params = {
    "symbol": "BTCUSDT",
    "tick_files": ["/path/to/BTCUSDT_2025-01.parquet", ...],
    "start_ts": "2025-01-01 00:00:00",
    "end_ts": "2025-07-31 23:59:59",
    "lookback_minutes": 60,
}
ticks_loader_json = serialize_tick_loader_params(tick_params)  # 转为 JSON 字符串
```

这个 JSON 字符串被存储在特征配置的 `compute_params` 中：

```yaml
# config/feature_dependencies.yaml
features:
  vpin_features:
    compute_params:
      ticks_loader_json: "{\"symbol\":\"BTCUSDT\",\"tick_files\":[...],...}"
```

### 2. `_build_call_args` 的自动检测机制

`_build_call_args` 函数现在会自动：

1. **检测函数签名**：使用 `inspect.signature` 检查函数是否需要 `ticks` 参数
2. **加载 ticks 数据**：如果需要且 `ticks_loader_json` 存在，自动加载
3. **注入到 kwargs**：将加载的 ticks 数据添加到 `call_kwargs["ticks"]`

### 3. 详细流程

```python
def _build_call_args(
    feature_info: Dict, 
    df: pd.DataFrame, 
    ticks_loader_json: Optional[str] = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    构建函数调用参数
    
    步骤：
    1. 获取特征配置中的 compute_params 和 column_mappings
    2. 检查函数是否需要 ticks 参数
    3. 如果需要，从 ticks_loader_json 加载 ticks 数据
    4. 构建 call_args 和 call_kwargs
    """
    compute_params = feature_info.get("compute_params", {}) or {}
    call_kwargs = dict(compute_params)  # 先复制所有 compute_params
    
    # === 自动检测和加载 ticks ===
    compute_func_name = feature_info.get("compute_func")
    if compute_func_name:
        compute_func = get_compute_func(compute_func_name)
        func_sig = inspect.signature(compute_func)
        
        # 检查函数是否需要 ticks 参数
        if "ticks" in func_sig.parameters and ticks_loader_json:
            # 从 ticks_loader_json 反序列化参数
            tick_params = deserialize_tick_loader_params(ticks_loader_json)
            
            # 根据 df 的时间范围加载 ticks
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                start_ts = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
                end_ts = df.index.max().strftime("%Y-%m-%d %H:%M:%S")
                
                # 获取 ticks_dir（从 tick_files 推断或使用默认值）
                ticks_dir = tick_params.get("ticks_dir")
                if not ticks_dir:
                    tick_files = tick_params.get("tick_files", [])
                    if tick_files:
                        ticks_dir = str(Path(tick_files[0]).parent)
                    else:
                        ticks_dir = "data/parquet_data"
                
                # 加载 ticks 数据
                ticks = load_tick_data(
                    symbol=tick_params["symbol"],
                    start_ts=start_ts,
                    end_ts=end_ts,
                    ticks_dir=ticks_dir,
                    lookback_minutes=tick_params.get("lookback_minutes", 60),
                )
                
                # 注入到 call_kwargs
                if ticks is not None and len(ticks) > 0:
                    call_kwargs["ticks"] = ticks
    
    # ... 处理 column_mappings ...
    
    call_args = []
    if feature_info.get("pass_full_df", True):
        call_args.append(df)
    
    return call_args, call_kwargs
```

### 4. 示例：Footprint 特征

#### 之前（失败）

```python
# 并行计算器调用
compute_footprint_features(df)  # ❌ 缺少 ticks 参数
# TypeError: missing 1 required positional argument: 'ticks'
```

#### 现在（自动）

```python
# 1. _build_call_args 检测到 compute_footprint_features 需要 ticks 参数
func_sig = inspect.signature(compute_footprint_features)
# func_sig.parameters = {'df': ..., 'ticks': ..., ...}

# 2. 从 ticks_loader_json 加载 ticks
ticks_loader_json = compute_params.get("ticks_loader_json")
tick_params = deserialize_tick_loader_params(ticks_loader_json)
ticks = load_tick_data(...)  # 根据 df 的时间范围加载

# 3. 自动注入到 call_kwargs
call_kwargs["ticks"] = ticks

# 4. 调用函数
compute_footprint_features(df, ticks=ticks, ...)  # ✅ 成功
```

### 5. 时间范围匹配

**关键点**：`_build_call_args` 会根据 `df` 的时间范围加载 ticks，而不是加载所有 ticks：

```python
# 如果 df 是 2025-01 的数据
df.index.min() = Timestamp('2025-01-01 00:00:00')
df.index.max() = Timestamp('2025-01-31 23:59:59')

# 只加载这个时间范围的 ticks
ticks = load_tick_data(
    symbol="BTCUSDT",
    start_ts="2025-01-01 00:00:00",
    end_ts="2025-01-31 23:59:59",
    ...
)
```

这样可以：
- **节省内存**：只加载需要的 ticks 数据
- **提高效率**：减少数据加载时间
- **支持按月计算**：每个月的计算只加载当月的 ticks

### 6. 在并行计算中的传递

```python
# 在 compute_features_parallel 中
compute_params = feature_info.get("compute_params", {})
ticks_loader_json = compute_params.get("ticks_loader_json")  # 从配置中获取

# 传递给 worker
future = executor.submit(
    _compute_single_feature_worker_monthly,
    feature_name,
    feature_info,
    df_bytes,
    fit,
    monthly_cache_dir,
    ticks_loader_json,  # ← 传递给 worker
)

# 在 worker 中
def _compute_single_feature_worker_monthly(
    feature_name: str,
    feature_info: Dict,
    df_bytes: bytes,
    fit: bool,
    monthly_cache_dir: Optional[str],
    ticks_loader_json: Optional[str] = None,  # ← 接收
):
    df = pickle.loads(df_bytes)
    
    # 按月计算
    for month_key, month_df in monthly_dfs.items():
        # 调用 _build_call_args，自动加载该月的 ticks
        call_args, call_kwargs = _build_call_args(
            feature_info, 
            month_df,  # ← 只包含该月的数据
            ticks_loader_json
        )
        # call_kwargs["ticks"] 现在包含该月的 ticks 数据
        month_result = compute_func(*call_args, **call_kwargs)
```

### 7. 优势

1. **自动化**：不需要手动在每个特征函数中处理 ticks 加载
2. **按需加载**：只加载当前计算需要的 ticks 数据
3. **内存高效**：按月计算时，每个 worker 只加载当月的 ticks
4. **透明**：特征函数不需要知道 ticks 是从哪里来的

### 8. 配置示例

```yaml
# config/feature_dependencies.yaml
features:
  footprint_basic:
    compute_func: compute_footprint_features
    compute_params:
      # ticks_loader_json 会在运行时自动注入（通过 _maybe_configure_vpin_ticks）
      price_bin_method: "fd"
      value_area_pct: 0.7
    output_columns:
      - fp_poc
      - fp_vah
      - fp_val
      - ...
```

### 9. 错误处理

如果加载失败，会打印警告但不中断计算：

```python
try:
    ticks = load_tick_data(...)
    if ticks is not None and len(ticks) > 0:
        call_kwargs["ticks"] = ticks
    else:
        print(f"⚠️  No ticks loaded for {compute_func_name}")
except Exception as e:
    print(f"⚠️  Failed to load ticks for {compute_func_name}: {e}")
    # 不抛出异常，让函数自己处理缺少 ticks 的情况
```

## 总结

`_build_call_args` 的自动检测机制：

1. **检测**：使用 `inspect.signature` 检查函数是否需要 `ticks` 参数
2. **加载**：如果需要，从 `ticks_loader_json` 反序列化参数并加载 ticks
3. **匹配**：根据 `df` 的时间范围加载对应的 ticks 数据
4. **注入**：自动将 ticks 添加到 `call_kwargs["ticks"]`

这样，特征函数只需要声明需要 `ticks` 参数，不需要关心如何获取它。

