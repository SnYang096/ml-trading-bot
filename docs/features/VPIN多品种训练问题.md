# VPIN 多品种训练问题与解决方案

## 问题描述

当同时训练多个品种（如 BTC 和 ETH）时，VPIN 的 `bucket_volume` 会根据每个品种的成交量自适应计算，导致：

1. **BTC 和 ETH 的 bucket_volume 不同**
   - BTC 可能：`bucket_volume = 100 BTC`
   - ETH 可能：`bucket_volume = 5000 ETH`
   
2. **VPIN 值不可比较**
   - 即使两个品种的订单流不平衡程度相同，VPIN 值也会因为 bucket_volume 不同而不同
   - 模型无法学习到跨品种的通用模式

3. **特征分布不一致**
   - 不同品种的 VPIN 特征分布差异很大
   - 模型可能过度拟合到品种特定的 bucket_volume，而不是真正的订单流模式

## 当前实现

### 自适应 bucket_volume 计算

当前实现（`_estimate_bucket_volume_from_cache`）：
```python
# 计算典型小时成交量（使用分位数）
hourly_volumes = df["volume"].resample("1H").sum()
typical_hourly_vol = hourly_volumes.rolling(window=lookback_hours, min_periods=1).quantile(quantile)
bucket_volume = typical_hourly_vol.iloc[-1]
```

**问题：**
- 每个品种独立计算，导致 bucket_volume 差异很大
- BTC 的 bucket_volume 可能是 100 BTC
- ETH 的 bucket_volume 可能是 5000 ETH
- 两者不可比较

## 解决方案

### 方案 1: 使用名义价值（USD）作为 bucket_volume ✅ 推荐

**核心思想：** 使用 USD 价值而不是数量作为 bucket_volume，使得不同品种的 VPIN 值可比较。

**实现：**
```python
def compute_vpin_with_nominal_bucket_volume(
    ticks: pd.DataFrame,
    bucket_volume_usd: float = 1000000.0,  # 固定 USD 价值
    n_buckets: int = 50,
) -> pd.DataFrame:
    """
    使用名义价值（USD）作为 bucket_volume 计算 VPIN
    
    Args:
        ticks: DataFrame with tick data (price, volume, side)
        bucket_volume_usd: Bucket volume in USD (固定值，所有品种相同)
        n_buckets: Number of buckets for rolling average
    
    Returns:
        DataFrame with VPIN values
    """
    # 计算每个 tick 的 USD 价值
    ticks = ticks.copy()
    ticks["value_usd"] = ticks["price"] * ticks["volume"]
    
    # 按 USD 价值切片
    buckets = []
    current_buy_usd = 0.0
    current_sell_usd = 0.0
    filled_value_usd = 0.0
    
    for _, row in ticks.iterrows():
        value_usd = row["value_usd"]
        side = row["side"]
        remaining = value_usd
        
        while remaining > 0:
            space_left = bucket_volume_usd - filled_value_usd
            trade_value = min(remaining, space_left)
            
            if side == 1:
                current_buy_usd += trade_value
            else:
                current_sell_usd += trade_value
            
            filled_value_usd += trade_value
            remaining -= trade_value
            
            if filled_value_usd >= bucket_volume_usd - 1e-9:
                imbalance = abs(current_buy_usd - current_sell_usd) / bucket_volume_usd
                buckets.append((row["timestamp"], imbalance))
                current_buy_usd = 0.0
                current_sell_usd = 0.0
                filled_value_usd = 0.0
    
    # 计算滚动平均
    vpin_series = pd.Series([b[1] for b in buckets], index=[b[0] for b in buckets])
    vpin_series = vpin_series.rolling(window=n_buckets, min_periods=1).mean()
    
    return vpin_series
```

**优点：**
- ✅ 所有品种使用相同的 bucket_volume（USD），VPIN 值可比较
- ✅ 模型可以学习跨品种的通用订单流模式
- ✅ 实现简单，只需修改计算逻辑

**缺点：**
- ⚠️ 需要价格数据（但通常都有）
- ⚠️ 需要确定合适的 USD bucket_volume（如 100 万 USD）

### 方案 2: 标准化 VPIN 值

**核心思想：** 对每个品种的 VPIN 值进行标准化（Z-score 或分位数标准化）。

**实现：**
```python
def normalize_vpin_by_symbol(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    按品种标准化 VPIN 值
    
    Args:
        df: DataFrame with VPIN values and symbol column
        vpin_col: VPIN column name
        symbol_col: Symbol column name
    
    Returns:
        DataFrame with normalized VPIN
    """
    df = df.copy()
    
    # 按品种计算均值和标准差
    stats = df.groupby(symbol_col)[vpin_col].agg(["mean", "std"])
    
    # 标准化
    def normalize(group):
        symbol = group[symbol_col].iloc[0]
        mean = stats.loc[symbol, "mean"]
        std = stats.loc[symbol, "std"]
        if std > 0:
            return (group[vpin_col] - mean) / std
        else:
            return group[vpin_col] - mean
    
    df[f"{vpin_col}_normalized"] = df.groupby(symbol_col).apply(normalize).values
    
    return df
```

**优点：**
- ✅ 不需要修改 VPIN 计算逻辑
- ✅ 可以保留原始 VPIN 值（用于单品种分析）

**缺点：**
- ⚠️ 需要按品种分组，可能增加计算复杂度
- ⚠️ 标准化可能丢失一些信息

### 方案 3: 使用相对 bucket_volume

**核心思想：** 使用相对于该品种典型成交量的 bucket_volume（如 1% 的日均成交量）。

**实现：**
```python
def compute_vpin_with_relative_bucket_volume(
    ticks: pd.DataFrame,
    relative_ratio: float = 0.01,  # 1% 的日均成交量
    n_buckets: int = 50,
) -> pd.DataFrame:
    """
    使用相对 bucket_volume 计算 VPIN
    
    Args:
        ticks: DataFrame with tick data
        relative_ratio: Bucket volume 相对于日均成交量的比例
        n_buckets: Number of buckets for rolling average
    """
    # 计算日均成交量
    daily_volume = ticks["volume"].resample("1D").sum()
    typical_daily_volume = daily_volume.rolling(window=7, min_periods=1).median()
    
    # 相对 bucket_volume
    bucket_volume = typical_daily_volume.iloc[-1] * relative_ratio
    
    # 使用标准 VPIN 计算逻辑
    # ...
```

**优点：**
- ✅ 保持品种间的相对一致性
- ✅ 自适应不同品种的成交量水平

**缺点：**
- ⚠️ 仍然可能因为品种差异导致 VPIN 值不可比较
- ⚠️ 需要确定合适的相对比例

## 推荐方案

**推荐使用方案 1（名义价值 bucket_volume）**，原因：

1. **最直接有效**：所有品种使用相同的 USD bucket_volume，VPIN 值天然可比较
2. **符合业务逻辑**：订单流不平衡应该用 USD 价值衡量，而不是数量
3. **实现简单**：只需修改 `compute_vpin_from_cached_ticks` 和 `compute_vpin_from_ticks` 函数

## 实施步骤

### 步骤 1: 修改 VPIN 计算函数

在 `src/data_tools/tick_loader.py` 和 `src/features/time_series/utils_order_flow_features.py` 中：

1. 添加 `bucket_volume_usd` 参数（可选，默认 None）
2. 如果提供了 `bucket_volume_usd`，使用 USD 价值计算
3. 否则，使用原有的自适应逻辑（向后兼容）

### 步骤 2: 更新配置文件

在 `config/feature_dependencies.yaml` 中：
```yaml
vpin_features:
  compute_params:
    vpin_n_buckets: 50
    vpin_adaptive: true
    vpin_bucket_volume_usd: 1000000.0  # 新增：固定 USD bucket_volume
```

### 步骤 3: 测试验证

1. 单品种测试：确保新逻辑与旧逻辑结果一致（当 bucket_volume_usd 未设置时）
2. 多品种测试：验证 BTC 和 ETH 的 VPIN 值分布是否更一致
3. 模型训练：对比使用新/旧 VPIN 特征的模型性能

## 注意事项

1. **缓存键更新**：如果修改了 bucket_volume 计算方式，需要更新缓存键生成逻辑
2. **向后兼容**：保持原有自适应逻辑作为默认行为
3. **USD bucket_volume 选择**：
   - 太小：桶太多，噪声大
   - 太大：桶太少，信号滞后
   - 推荐：根据主要品种的成交量选择（如 100 万 USD）

## 相关文件

- `src/data_tools/tick_loader.py` - VPIN 计算和缓存
- `src/features/time_series/utils_order_flow_features.py` - VPIN 特征提取
- `config/feature_dependencies.yaml` - VPIN 特征配置

