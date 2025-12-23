# 函数式 vs 配置文件式特征工程对比

## 当前两种方式

### 1. 配置文件方式（推荐）

**流程：**
```
requested_features (YAML) 
  → StrategyFeatureLoader.load_features_from_requested()
    → FeatureComputer.compute_features_parallel()
      → 自动解析依赖 → 按层级分组 → 每层并行计算 → 缓存
  → post_processors (可选，如 build_sr_reversal_features)
```

**特点：**
- ✅ 按需加载：只计算 `requested_features` 及其依赖
- ✅ 并行计算：同层特征并行，不同层顺序执行
- ✅ 缓存支持：内存缓存 + 磁盘缓存
- ✅ 依赖管理：自动解析依赖关系
- ✅ 配置驱动：特征定义在 YAML 中，易于修改

**示例配置：**
```yaml
# config/strategies/sr_reversal/features.yaml
feature_pipeline:
  requested_features:
    - sr_strength_max
    - sqs_hal_high
    - sqs_hal_low
    - rsi
  post_processors:
    - module: src.time_series_model.strategies.sr_reversal.features
      function: build_sr_reversal_features
      params: {}
```

### 2. 函数式方式（当前 build_sr_reversal_features）

**流程：**
```
build_sr_reversal_features(df)
  → extract_wpt_features()  # 顺序执行
  → extract_hilbert_features()  # 顺序执行
  → extract_hurst_features()  # 顺序执行
  → extract_liquidity_features()  # 顺序执行
  → extract_order_flow_features()  # 顺序执行
  → extract_interaction_features()  # 顺序执行
```

**问题：**
- ❌ 总是计算所有特征，无法按需加载
- ❌ 顺序执行，没有并行
- ❌ 没有缓存
- ❌ 硬编码，难以修改

## 改进方案：让函数式也支持并行和按需加载

### 方案 A：将特征定义移到配置文件（推荐）

**步骤：**

1. **在 `feature_dependencies.yaml` 中定义所有特征：**
```yaml
features:
  wpt_price_features:
    module: enhanced
    compute_func: extract_wpt_features
    dependencies: []
    required_columns: ["close", "high", "low", "volume"]
    output_columns: ["wpt_price_trend", "wpt_price_fluctuation", ...]
    category: wpt
    pass_full_df: true
  
  hilbert_features:
    module: enhanced
    compute_func: extract_hilbert_features
    dependencies: ["wpt_price_features"]  # 依赖 WPT
    required_columns: ["wpt_price_fluctuation", "wpt_cvd_fluctuation"]
    output_columns: ["hilbert_phase_diff", ...]
    category: hilbert
    pass_full_df: true
  
  vpin_features:
    module: enhanced
    compute_func: extract_order_flow_features
    dependencies: []
    required_columns: ["open", "close", "high", "low", "volume"]
    output_columns: ["vpin", "vpin_ma5", "vpin_ma10", ...]
    category: order_flow
    pass_full_df: true
  
  interaction_features:
    module: enhanced
    compute_func: extract_interaction_features
    dependencies: ["vpin_features", "wpt_price_features"]  # 依赖多个特征
    required_columns: ["vpin", "compression_energy", ...]
    output_columns: ["vpin_x_compression_rank", ...]
    category: interaction
    pass_full_df: true
```

2. **简化 `build_sr_reversal_features`，只做组合和衍生：**
```python
def build_sr_reversal_features(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    cvd_col: Optional[str] = None,
    tbr_col: Optional[str] = None,
    atr_col: str = "atr",
) -> pd.DataFrame:
    """
    构建 SR 反转策略的专属特征集（组合和衍生特征）
    
    注意：基础特征（WPT, Hilbert, VPIN等）应该通过配置文件加载
    此函数只做：
    1. 特征组合（如 SR 强度组合）
    2. 衍生特征（如 SR 距离归一化）
    3. shift(1) 避免未来数据
    """
    df = df.copy()
    
    # 只做组合和衍生特征（基础特征已在配置文件中计算）
    # 1. SR 强度组合
    if "sqs" in df.columns:
        df["sr_strength_combined"] = df["sqs"].fillna(0.0)
    
    # 2. SR 距离归一化
    if "dist_to_nearest_sr" in df.columns and atr_col in df.columns:
        df["sr_distance_normalized"] = (
            df["dist_to_nearest_sr"] / df[atr_col].replace(0, np.nan)
        ).fillna(0.0)
    
    # 3. ZigZag 距离
    if "zz_high_value" in df.columns and price_col in df.columns:
        df["dist_to_zz_high"] = (df[price_col] - df["zz_high_value"]).abs()
        if atr_col in df.columns:
            df["dist_to_zz_high_atr"] = (
                df["dist_to_zz_high"] / df[atr_col].replace(0, np.nan)
            ).fillna(0.0)
    
    # 4. 确保所有特征 shift(1)
    feature_cols = [
        col for col in df.columns
        if col.startswith(("wpt_", "hilbert_", "hurst_", "vpvr_", "vpin", "_x_", "_rank"))
        and col not in df.columns  # 避免重复 shift
    ]
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)
    
    return df
```

3. **在 `features.yaml` 中请求需要的特征：**
```yaml
feature_pipeline:
  requested_features:
    - sr_strength_max
    - sqs_hal_high
    - sqs_hal_low
    - rsi
    - wpt_price_features  # 新增
    - hilbert_features    # 新增
    - vpin_features       # 新增
    - interaction_features # 新增
  post_processors:
    - module: src.time_series_model.strategies.sr_reversal.features
      function: build_sr_reversal_features
      params: {}
```

**优势：**
- ✅ 自动按需加载（只计算 requested_features）
- ✅ 自动并行计算（同层并行）
- ✅ 自动缓存（内存+磁盘）
- ✅ 自动依赖解析

### 方案 B：让 post_processor 也能利用 FeatureComputer

如果不想改配置文件，可以让 `build_sr_reversal_features` 内部也使用并行计算：

```python
def build_sr_reversal_features(
    df: pd.DataFrame,
    feature_loader: Optional[StrategyFeatureLoader] = None,
    price_col: str = "close",
    ...
) -> pd.DataFrame:
    """
    构建 SR 反转策略的专属特征集（支持并行计算）
    """
    df = df.copy()
    
    # 如果提供了 feature_loader，使用并行计算
    if feature_loader is not None:
        # 定义需要计算的特征
        requested = [
            "wpt_price_features",
            "hilbert_features",
            "hurst_features",
            "vpvr_features",
            "vpin_features",
            "interaction_features",
        ]
        
        # 并行计算基础特征
        df = feature_loader.load_features_from_requested(
            df, requested_features=requested, fit=True
        )
    else:
        # 回退到顺序执行（向后兼容）
        df = extract_wpt_features(df, ...)
        df = extract_hilbert_features(df, ...)
        # ...
    
    # 组合和衍生特征（顺序执行，因为依赖基础特征）
    # ...
    
    return df
```

**但这种方式需要：**
- 在 `feature_dependencies.yaml` 中定义这些特征
- 修改 post_processor 的调用方式，传入 feature_loader

## 推荐方案

**推荐使用方案 A**，因为：
1. 更符合现有架构设计
2. 配置驱动，易于维护
3. 自动获得并行和缓存能力
4. 可以灵活选择需要的特征

**实施步骤：**
1. 在 `feature_dependencies.yaml` 中添加所有特征定义
2. 简化 `build_sr_reversal_features`，只做组合和衍生
3. 在 `features.yaml` 的 `requested_features` 中列出需要的特征

