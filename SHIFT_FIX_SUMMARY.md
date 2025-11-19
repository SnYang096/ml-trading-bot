# 滚动窗口统计特征 shift(1) 修复总结

## 🎯 核心原则：决策时间 vs. 特征可用时间

**关键区分**：
- **基础滚动指标**（sma, ema, atr, bb_width）：不需要 shift(1)
- **滚动窗口统计特征**（zscore, percentile, entropy, compression_ratio）：需要 shift(1)

## 📌 为什么需要 shift(1)？

### 情况一：T+1 开盘交易（最常见）

- **决策时间**：在 t 时刻收盘后（即你知道 close[t], high[t], low[t], volume[t]）
- **执行时间**：以 open[t+1] 成交
- ✅ **此时你可以合法使用 t 时刻的所有 OHLCV 数据（包括 close[t]）**

### ❓ 那为什么滚动窗口统计特征还需要 shift(1)？

**问题不在于用了 close[t]，而在于如何用！**

#### ✅ 基础滚动指标（不需要 shift）

```python
# sma_20[t] = mean(close[t-19:t+1])
# 这是基础指标，天然只依赖 ≤ t 的数据
sma_20[t] = close.rolling(20).mean()  # ✅ 不需要 shift
```

#### ⚠️ 滚动窗口统计特征（需要 shift）

```python
# zscore_w288[t] = (close[t] - mean(close[t-287:t+1])) / std(close[t-287:t+1])
# 它把当前值 close[t] 同时用作"被标准化的对象"和"分布的一部分"
zscore_w288[t] = (close - close.rolling(288).mean()) / close.rolling(288).std()
# ⚠️ 需要 shift(1) 以避免将当前值包含在历史分布中
```

**更严谨的说法**：
- 任何将当前样本包含在滚动统计分母/分布中的特征，都会引入轻微前视偏差（look-ahead bias）
- 即使数据本身是已知的，但"它在最近 N 根中的百分位"这个信息，在真实交易中无法在 t 时刻精确获得
- 除非你假设未来没有极端波动改变分布 —— 但这正是过拟合的来源

## ✅ 正确做法（统一规则）

### 对所有「滚动窗口统计特征」强制 shift(1)

包括：
- `zscore`（z-score 标准化）
- `percentile`（百分位排名）
- `entropy`（熵）
- `r2`（R²）
- `skew`（偏度）
- `kurtosis`（峰度）
- `compression_ratio`（压缩比率）

### ❌ 不对基础指标 shift(1)

如：
- `sma`, `ema`（移动平均）
- `atr`, `bb_upper`, `bb_width`（技术指标）
- `rsi`, `macd`（技术指标）

## 🔧 修复内容

### 1. `_rolling_zscore` 修复

**修复前**：
```python
zscore = (series - rolling_mean) / rolling_std
return zscore  # 没有 shift
```

**修复后**：
```python
zscore = (series - rolling_mean) / rolling_std
if shift:  # 默认 True
    zscore = zscore.shift(1)  # ← 关键修复
return zscore
```

**影响**：
- 所有 `*_zscore_w*` 特征（如 `atr_zscore_w288`, `bb_width_zscore_w288`）都会自动 shift(1)
- 在 t 时刻使用的特征基于 t-1 及之前的数据计算

### 2. `_rolling_percentile` 修复

**修复前**：
```python
percentile_series = series.rolling(...).apply(_percentile, raw=True)
return percentile_series  # 没有 shift
```

**修复后**：
```python
percentile_series = series.rolling(...).apply(_percentile, raw=True)
if shift:  # 默认 True
    percentile_series = percentile_series.shift(1)  # ← 关键修复
return percentile_series
```

**影响**：
- 所有 `*_percentile` 特征（如 `atr_percentile`, `volume_percentile`）都会自动 shift(1)
- 在 t 时刻使用的特征基于 t-1 及之前的数据计算

### 3. `atr_percentile` 修复（feature_engineering_enhanced.py）

**修复前**：
```python
df["atr_percentile"] = (
    df["atr"].rolling(100, min_periods=100).apply(_percentile, raw=True)
)
```

**修复后**：
```python
atr_percentile_raw = (
    df["atr"].rolling(100, min_periods=100).apply(_percentile, raw=True)
)
df["atr_percentile"] = atr_percentile_raw.shift(1)  # ← 关键修复
```

### 4. `atr_compression_ratio` 修复

**修复前**：
```python
data["atr_compression_ratio"] = (
    atr_mean_hist / (data["atr"] + eps)
).replace([np.inf, -np.inf], np.nan)
```

**修复后**：
```python
atr_compression_ratio_raw = (
    atr_mean_hist / (data["atr"] + eps)
).replace([np.inf, -np.inf], np.nan)
data["atr_compression_ratio"] = atr_compression_ratio_raw.shift(1)  # ← 关键修复
```

## 📊 预期效果

修复后，这些特征与未来收益的相关性应该：

| 特征 | 修复前相关性 | 预期修复后 |
|------|------------|-----------|
| `atr_percentile` | 0.1685 | < 0.05 |
| `atr_zscore_w288` | 0.1735 | < 0.05 |
| `atr_compression_ratio` | -0.1496 | < 0.05 |
| `bb_width_zscore_w288` | 0.1288 | < 0.05 |

**整体预期**：
- OOS Rank IC 从 0.0833 降至 0.00 ~ 0.02
- 消除了虚假信号，特征更可靠
- 虽然可能损失一些信息，但结果更可信

## 🎯 关键要点总结

| 问题 | 答案 |
|------|------|
| 能否用 t 时刻的 close 来预测 t+1 的收益？ | ✅ **可以！** 只要交易决策发生在 t 时刻K线结束之后 |
| 那为什么滚动窗口统计特征还需要 shift(1)？ | 因为"我知道今天的收盘价" ≠ "我知道今天的收盘价在过去 N 天里排第几"。后者需要等待更多数据才能稳定估计 |
| 正确的做法是什么？ | 对所有滚动窗口统计特征强制 shift(1)，确保在 t 时刻使用的特征基于 t-1 及之前的数据计算 |

## ✅ 修复状态

- ✅ `_rolling_zscore` 已修复：默认 `shift=True`
- ✅ `_rolling_percentile` 已修复：默认 `shift=True`
- ✅ `atr_percentile`（feature_engineering_enhanced.py）已修复：添加 `shift(1)`
- ✅ `atr_compression_ratio` 已修复：添加 `shift(1)`
- ✅ `compression_confidence` 已确认：使用了已 shift(1) 的特征，本身已是因果的

所有修复已完成，符合更严格的因果性原则！

