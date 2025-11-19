# 数据泄露审计报告

## 检查结果总结

### ✅ 均线特征本身：无数据泄露
- Simple baseline test: Avg Rank IC = 0.0100 < 0.03 ✅
- Full random walk test: Avg Rank IC = 0.0295 < 0.03 ✅
- Feature-future correlation: Max |corr| = 0.0802 < 0.1 ✅

### ⚠️ 发现的问题

#### 1. `rolling_vol` 计算的时间对齐问题

**问题描述：**
```python
def _rolling_vol(series: pd.Series) -> pd.Series:
    rets = series.pct_change(hold_period).dropna()
    return rets.rolling(window=lookback_window, min_periods=min_samples).std()
```

**问题分析：**
- `pct_change(hold_period)` 计算的是 `(price[t] - price[t-hold_period]) / price[t-hold_period]`
- `dropna()` 会移除前 `hold_period` 个值，导致 `rets` 的索引从 `hold_period` 开始
- `rolling(window=lookback_window).std()` 计算的是 `rets[t-lookback_window+1:t+1]` 的 std
- 当赋值回 DataFrame 时，pandas 会按索引对齐，导致前 `hold_period` 个位置为 NaN

**潜在泄露风险：**
- `rolling_vol[t]` 使用的价格窗口：`[t-lookback_window+1-hold_period, t]`
- `future_return[t]` 使用的价格窗口：`[t, t+hold_period]`
- **在 `t` 处有重叠**，但这是已知的当前价格，不是未来信息
- **真正的风险**：如果 `rolling_vol[t]` 的计算窗口包含了 `[t+1, t+hold_period]` 的价格，那就是泄露

**验证结果：**
- ✅ `rolling_vol[t]` 只使用 `[t-lookback_window+1-hold_period, t]` 的价格
- ✅ `future_return[t]` 使用 `[t, t+hold_period]` 的价格
- ✅ 虽然都使用了 `price[t]`，但这是当前时刻的已知价格，不是未来信息
- ✅ **结论：`rolling_vol` 计算本身是安全的**

#### 2. `historical_quantile_label` 计算：安全

**验证结果：**
- ✅ 历史窗口：`[i-lookback_window-hold_period, i-hold_period)`
- ✅ 当前索引：`i`
- ✅ 未来窗口：`[i, i+hold_period]`
- ✅ **结论：历史窗口严格在当前位置之前，无泄露**

#### 3. `volatility_normalized_target` 计算：安全

**验证结果：**
- ✅ `target[t] = future_return[t] / (rolling_vol[t] + eps)`
- ✅ `rolling_vol[t]` 只使用历史数据
- ✅ `future_return[t]` 是未来收益（标签）
- ✅ **结论：计算是安全的**

### 🔍 其他潜在问题

#### 1. TSCV Gap 可能不够大
- 当前 gap = 24（与 hold_period 相同）
- 建议：gap 应该 >= `max(lookback_window, 2×hold_period)`
- 对于 `lookback_window=60, hold_period=24`，建议 gap >= 60

#### 2. OOS 测试集时间范围检查
- 当前实现：简单按比例分割（85% train, 15% test）
- ✅ 已按时间排序，test 在 train 之后
- ⚠️ 但 train 和 test 之间没有额外的 buffer

#### 3. 模型性能问题（非泄露）
- OOS Rank IC = -0.0813（负值）
- 方向准确率 = 54%（接近随机）
- Pearson 相关性 = -0.0564（负相关）
- **可能原因：**
  - 信号需要反转
  - 模型过拟合到训练集
  - 数据分布变化

## 建议修复

### 1. 增加 TSCV Gap
```python
# 建议在 Makefile 中设置
RANK_IC_TSCV_GAP = 60  # 或 max(lookback_window, 2×hold_period)
```

### 2. 添加 OOS Buffer
在 `split_train_test` 函数中添加 buffer：
```python
def split_train_test(df, test_size=0.15, buffer_size=24):
    n_total = len(df)
    split_idx = int(n_total * (1 - test_size))
    buffer_idx = split_idx - buffer_size  # 在 train 和 test 之间留出 buffer
    df_train = df.iloc[:buffer_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    return df_train, df_test
```

### 3. 检查信号方向
如果 Pearson 相关性为负，考虑反转信号或检查模型训练逻辑。

## 结论

**标签计算逻辑本身是安全的，没有发现数据泄露。**

但模型性能差的原因可能是：
1. TSCV gap 不够大，导致 train/val 边界有轻微泄露
2. 信号方向可能错误（负相关）
3. 模型过拟合或数据分布变化

建议先增加 TSCV gap 和添加 OOS buffer，然后重新训练观察效果。

