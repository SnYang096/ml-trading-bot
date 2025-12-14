# Inf 值修复总结

## 问题分析

从训练输出中发现多个特征产生 `inf/-inf` 值，导致大量样本被删除：
- `sr_strength_max`: 706 个 inf
- `hurst_price_rolling`: 298 个 inf
- `hurst_cvd_rolling`: 298 个 inf
- `rsi`: 70 个 inf
- `trade_cluster_*_zscore`: 5 个 inf

## 根本原因修复

### 1. **sr_strength_max** - 价格趋势计算除零

**位置**: `baseline_features.py` 第 812 行

**问题**:
```python
price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
```
如果 `recent_prices.iloc[0]` 为 0 或非常接近 0，会产生 `inf`。

**修复**:
```python
start_price = recent_prices.iloc[0]
if abs(start_price) < 1e-10:
    # 如果起始价格接近 0，使用 pct_change 方法
    price_trend = recent_prices.pct_change().mean()
else:
    price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / start_price
# 检查结果是否有效
if not np.isfinite(price_trend):
    price_trend = 0.0
```

### 2. **Hurst 特征** - 输入数据有效性检查

**位置**: `utils_hurst_features.py` 第 80 行

**问题**:
- 如果输入数据包含 `inf`，`np.var(x)` 可能产生 `inf`
- 如果 `seg_y` 包含 `inf`，去趋势计算会产生 `inf`

**修复**:
```python
# 检查输入数据是否有效
if not np.all(np.isfinite(seg_y)):
    continue  # 跳过包含 inf/NaN 的段

x_var = np.var(x)
if x_var < 1e-12 or not np.isfinite(x_var):
    continue  # 跳过方差过小或无效的段

slope = (np.mean(x * seg_y) - np.mean(x) * np.mean(seg_y)) / x_var
# 检查 slope 是否有效
if not np.isfinite(slope):
    continue  # 跳过产生 inf 的段

# 检查 detrended 是否有效
if not np.all(np.isfinite(detrended)):
    continue  # 跳过包含 inf/NaN 的去趋势结果
```

### 3. **RSI 特征** - 输入数据清理

**位置**: `baseline_features.py` 第 72 行

**问题**:
- 如果价格序列包含 `inf` 或全为 0，`talib.RSI` 可能产生 `inf`

**修复**:
```python
# 检查输入数据：如果包含 inf/NaN 或全为 0，可能导致 RSI 计算异常
if series.isna().all() or (series == 0).all():
    return pd.Series(np.nan, index=series.index)

# 清理输入数据中的 inf 值，避免传递给 talib
series_clean = series.replace([np.inf, -np.inf], np.nan)

# 如果清理后数据不足，返回 NaN
if series_clean.notna().sum() < period + 1:
    return pd.Series(np.nan, index=series.index)

values = talib.RSI(series_clean.values, timeperiod=period)
rsi_series = pd.Series(values, index=series.index)
# 清理输出中的 inf 值
rsi_series = rsi_series.replace([np.inf, -np.inf], np.nan)
```

### 4. **Trade Clustering Z-score** - 输入数据清理和 rolling_std 检查

**位置**: `utils_order_flow_features.py` 第 1670-1724 行

**问题**:
- 如果输入数据包含 `inf`，`rolling_std` 可能产生 `inf`
- 即使有 `TOL` 保护，如果 `rolling_std` 本身是 `inf`，zscore 仍会是 `inf`

**修复**:
```python
# 先清理 inf 值，避免 rolling_std 产生 inf
entropy_clean = df["trade_cluster_directional_entropy"].replace([np.inf, -np.inf], np.nan)
rolling_mean = entropy_clean.rolling(window=w, min_periods=1).mean()
rolling_std = entropy_clean.rolling(window=w, min_periods=1).std()
# 检查 rolling_std 是否包含 inf（可能由输入数据中的 inf 导致）
if (~np.isfinite(rolling_std)).any():
    rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)
z = (entropy_clean - rolling_mean) / (rolling_std + TOL)
df[f"trade_cluster_directional_entropy_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
```

### 5. **Trade Clustering 比率特征** - 输入数据清理

**位置**: `utils_order_flow_features.py` 第 1633-1643 行

**问题**:
- 如果 `trade_cluster_max_buy_run` 或 `trade_cluster_max_sell_run` 包含 `inf`，比率计算会产生 `inf`

**修复**:
```python
# 检查输入数据是否包含 inf/NaN
max_buy_clean = df["trade_cluster_max_buy_run"].replace([np.inf, -np.inf], np.nan)
max_sell_clean = df["trade_cluster_max_sell_run"].replace([np.inf, -np.inf], np.nan)
df["trade_cluster_buy_sell_max_ratio"] = (
    max_buy_clean / (max_sell_clean + TOL)
).replace([np.inf, -np.inf], np.nan)
```

### 6. **sr_strength_max** - 输入数据检查和调试信息

**位置**: `baseline_features.py` 第 3741 行

**修复**:
```python
# 计算 max/sum（pandas 的 max/sum 会自动忽略 NaN，但如果输入有 inf 会产生 inf）
strength_df = data[strength_columns].copy()
# 检查并记录哪些列包含 inf
inf_cols = []
for col in strength_columns:
    if col in strength_df.columns:
        inf_mask = ~np.isfinite(strength_df[col])
        if inf_mask.any():
            inf_count = inf_mask.sum()
            print(f"   ⚠️  {col}: {inf_count} inf values (min={strength_df[col].min()}, max={strength_df[col].max()})")
            inf_cols.append(col)
            # 将 inf 替换为 NaN
            strength_df[col] = strength_df[col].replace([np.inf, -np.inf], np.nan)
data["sr_strength_max"] = strength_df.max(axis=1)
data["sr_strength_sum"] = strength_df.sum(axis=1)
```

## 测试集 Trade Clustering 特征全为 NaN 的调试

### 验证脚本

创建了 `scripts/check_july_tick_data.py` 用于验证 7 月 tick 数据：

```bash
python scripts/check_july_tick_data.py --symbol BTCUSDT --data-path data/parquet_data
```

**验证结果**:
- ✅ 7 月数据存在：`BTCUSDT_2025-07.parquet`
- ✅ 数据量：4,573,836 条
- ✅ 时间范围：2025-07-01 00:00:00 到 2025-07-31 23:59:59
- ✅ 数据质量：无缺失值，side 分布正常

### 可能原因

如果测试集 Trade Clustering 特征全为 NaN，可能原因：

1. **时间对齐问题**：Trade Clustering 事件的时间戳与测试集 K 线时间不匹配
   - 添加了详细的调试信息，打印对齐前后的时间范围和对齐结果

2. **计算逻辑问题**：在计算 Trade Clustering 时，可能没有正确处理 7 月数据
   - 添加了数据统计打印，显示计算的事件数量和时间范围

3. **对齐逻辑问题**：在将 Trade Clustering 事件对齐到 K 线时，可能出现了问题
   - 添加了每个特征的对齐统计，显示有多少 K 线获得了有效值

### 调试信息

在 `extract_trade_clustering_features` 中添加了以下调试信息：

1. **原始数据统计**:
   - Trade Clustering 事件数量
   - 时间范围
   - 有效列数量

2. **对齐统计**:
   - Cluster 事件数量 vs K 线数量
   - 时间范围对比
   - 每个特征的对齐结果（有多少 K 线获得了有效值）

3. **训练流程中的时间范围打印**:
   - 测试集时间范围（用于验证 tick 数据可用性）

## 修复效果

修复后，这些特征应该：
1. **不再产生 inf 值**：所有除法操作都有除零保护
2. **输入数据清理**：在计算前清理 inf 值，防止传播
3. **输出数据清理**：在返回前再次清理，确保安全
4. **详细的调试信息**：帮助定位问题根源

## 下一步

1. 重新运行训练，观察 inf 值是否减少
2. 查看调试信息，确认 Trade Clustering 对齐是否正常
3. 如果仍有问题，根据调试信息进一步分析

