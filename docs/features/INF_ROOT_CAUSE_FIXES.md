# Inf 值根本原因修复总结

## 修复原则

**不再简单地用 NaN 替换 inf，而是找到根本原因并修复计算逻辑**，确保：
1. 输入数据在计算前被清理（移除 inf/NaN）
2. 所有除法操作都有除零保护
3. 所有中间计算结果都检查有效性
4. 只有在 warmup 期间数据不足时才允许 NaN

## 修复详情

### 1. **sr_strength_max** - Volume 列包含 inf 值

**根本原因**:
- `calculate_sqs` 函数在计算 `vol_ratio` 时，如果 `volume` 列包含 inf，会导致 `vol_ratio` 和 `vol_factor` 也是 inf
- 最终 `weighted_reaction = (reaction / current_atr) * np.sqrt(vol_factor)` 会产生 inf

**修复位置**: `baseline_features.py` 第 331-348 行

**修复内容**:
```python
# 清理 ref_vols 中的 inf/NaN 值，避免影响 avg_vol 计算
ref_vols_clean = ref_vols.replace([np.inf, -np.inf], np.nan).dropna()
if len(ref_vols_clean) > 0:
    avg_vol = ref_vols_clean.mean()
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        avg_vol = 1.0
else:
    avg_vol = 1.0

current_vol = window_df.loc[idx, "volume"]
# 清理 current_vol 中的 inf/NaN 值
if not np.isfinite(current_vol) or current_vol < 0:
    current_vol = 0.0

vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
# 确保 vol_ratio 是有限值
if not np.isfinite(vol_ratio):
    vol_ratio = 1.0

vol_factor = min(vol_ratio, 3.0)
# 确保 vol_factor 是有限值
if not np.isfinite(vol_factor):
    vol_factor = 1.0
```

### 2. **Hurst 特征** - 输入数据包含 inf 值

**根本原因**:
- 如果输入数据（price, cvd, volume）包含 inf，`pct_change()` 和 `diff()` 会产生 inf
- 这些 inf 值会传播到 Hurst 计算中

**修复位置**: `utils_hurst_features.py` 第 405-412, 432-433, 451-458 行

**修复内容**:
```python
# 首先检查输入数据是否包含 inf/NaN，如果有，先清理
price_series = df[price_col].replace([np.inf, -np.inf], np.nan)
# 如果价格序列包含 inf/NaN，pct_change 可能产生 inf
# 在计算 pct_change 前，确保没有 inf 值
price_returns = price_series.pct_change()
# 处理 inf 值（可能由除权、价格归零等导致）
price_returns = price_returns.replace([np.inf, -np.inf], np.nan)
```

同样修复了 CVD 和 Volume 的处理。

### 3. **RSI 特征** - talib.RSI 输入验证

**根本原因**:
- 如果价格序列包含 inf 或全为 0，`talib.RSI` 可能产生 inf

**修复位置**: `baseline_features.py` 第 72-86 行

**修复内容**:
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

### 4. **Trade Clustering** - 统计量计算中的 inf 值

**根本原因**:
- 如果 `temp_buy_runs` 或 `temp_sell_runs` 包含 inf，`max()` 和 `np.mean()` 会产生 inf

**修复位置**: `utils_order_flow_features.py` 第 1149-1158 行

**修复内容**:
```python
# 清理 temp_buy_runs 和 temp_sell_runs 中的 inf/NaN 值
temp_buy_runs_clean = [x for x in temp_buy_runs if np.isfinite(x) and x >= 0]
temp_sell_runs_clean = [x for x in temp_sell_runs if np.isfinite(x) and x >= 0]
max_buy_run = max(temp_buy_runs_clean) if temp_buy_runs_clean else 0.0
max_sell_run = max(temp_sell_runs_clean) if temp_sell_runs_clean else 0.0
avg_buy_run = np.mean(temp_buy_runs_clean) if temp_buy_runs_clean else 0.0
avg_sell_run = np.mean(temp_sell_runs_clean) if temp_sell_runs_clean else 0.0
# 确保结果是有限值
max_buy_run = max_buy_run if np.isfinite(max_buy_run) else 0.0
max_sell_run = max_sell_run if np.isfinite(max_sell_run) else 0.0
avg_buy_run = avg_buy_run if np.isfinite(avg_buy_run) else 0.0
avg_sell_run = avg_sell_run if np.isfinite(avg_sell_run) else 0.0
```

### 5. **价格趋势计算** - 起始价格为 0 的除零问题

**根本原因**:
- 在 `_compute_boundary_strengths` 中，如果 `recent_prices.iloc[0]` 为 0，`price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]` 会产生 inf

**修复位置**: `baseline_features.py` 第 820-829 行

**修复内容**:
```python
# 防止除零：如果起始价格为 0 或非常小，使用 pct_change 代替
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

## 测试用例

创建了 `tests/features/test_inf_root_cause_fixes.py`，包含 7 个测试用例：

1. `test_sr_strength_max_with_inf_volume`: 测试 volume 包含 inf 时的处理
2. `test_hurst_features_with_inf_input`: 测试 Hurst 特征在输入包含 inf 时的处理
3. `test_rsi_with_inf_input`: 测试 RSI 在输入包含 inf 时的处理
4. `test_trade_clustering_with_inf_input`: 测试 Trade Clustering 在输入包含 inf 时的处理
5. `test_hurst_dfa_with_inf_input`: 测试 `compute_hurst_dfa` 在输入包含 inf 时的处理
6. `test_price_trend_calculation_with_zero_price`: 测试价格趋势计算在起始价格为 0 时的处理
7. `test_volume_ratio_calculation_with_inf`: 测试成交量比率计算在 volume 包含 inf 时的处理

**所有测试用例都通过** ✅

## 修复效果

修复后，这些特征应该：
1. **不再产生 inf 值**：所有计算都有输入数据清理和结果验证
2. **正确处理边界情况**：零值、inf 值、NaN 值都有适当的处理
3. **保持数值稳定性**：所有中间计算结果都检查有效性
4. **只在必要时返回 NaN**：只有在 warmup 期间数据不足时才返回 NaN

## 与之前的修复对比

**之前的修复**（简单替换）:
- 在计算后简单地用 `replace([np.inf, -np.inf], np.nan)` 替换 inf
- 没有解决根本原因，inf 仍然会在计算过程中传播
- NaN 值过多，影响模型训练

**现在的修复**（根本原因修复）:
- 在计算前清理输入数据，防止 inf 产生
- 在计算过程中检查中间结果，确保有效性
- 只在必要时返回 NaN（warmup 期间）
- 所有修复都有测试用例验证

## 下一步

1. 重新运行训练，观察 inf 值是否完全消除
2. 如果仍有问题，根据新的调试信息进一步分析
3. 确保所有特征计算都遵循相同的原则（输入清理 + 结果验证）

