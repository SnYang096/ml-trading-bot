# 特征统计可靠性修复报告

## 问题分析

根据数据泄露检测结果，以下特征与未来收益高度相关（>0.1）：

1. `atr_zscore_w288`: 0.1882
2. `atr_percentile`: 0.1771
3. `atr_compression_ratio`: -0.1671
4. `atr_zscore_w500`: 0.1599
5. `bb_width_zscore_w288`: 0.1337
6. `bb_width_zscore_w500`: 0.1317

## 关键澄清：这不是数据泄漏问题！

### ✅ 正确理解

1. **代码检查结果**：所有特征都使用了 `.rolling()` 方法，**没有使用全局统计量**
   - ✅ 没有 `df['col'].mean()`（全局均值）
   - ✅ 没有 `df['col'].rank(pct=True)`（全局排名）
   - ✅ 没有 `df['col'].quantile()`（全局分位数）
   - ✅ 只使用了 `df['col'].rolling(window).mean()`（滚动均值）

2. **rolling() 的行为**：
   - `rolling(window=288)` **只使用历史和当前数据**，绝不包含未来
   - 即使 `min_periods=1`，也不会引入未来信息
   - **这不是数据泄漏问题，而是统计可靠性问题**

### ❌ 之前的误解

| 说法 | 正确吗？ | 说明 |
|------|---------|------|
| "min_periods 小会导致统计量不准" | ✅ 正确 | 小样本估计方差大 |
| "这会间接包含未来信息" | ❌ **错误** | rolling 永远只用历史，不包含未来 |
| "提高 min_periods 能防泄漏" | ❌ **错误** | 泄漏来自全局统计，不是 min_periods |
| "提高 min_periods 能降低虚假相关" | ✅ 正确 | 因为剔除了高噪声样本，但本质是降噪，不是去泄漏 |

## 真正的问题：统计可靠性

### 问题根源

`min_periods` 过小会导致早期统计量不稳定（小样本噪声）：

```python
# 示例：window=288, min_periods=28
# 第 28 行：均值 = 前 28 个点的平均，标准差 = 基于 28 个点
# 第 288 行：均值 = 前 288 个点的平均

# 前 259 行（28~287）的 z-score 是基于不完整窗口计算的
# 统计量噪声大、不可靠，可能偶然与 future_return 对齐
```

### 为什么提高 min_periods 能降低相关性？

**不是因为它"去除了泄漏"**，而是因为：

1. **减少了高噪声区域的样本数量**
   - 前 100~200 个点的 z-score 噪声极大（小样本估计）
   - 这些噪声可能偶然与 future_return 高相关（过拟合）

2. **如果设 min_periods=144（window=288 的一半）**
   - 直接丢弃了前 143 行
   - 剩下的样本统计更稳，偶然相关性下降 → correlation 降低

3. **这是一种"删数据降噪"，而非"修复泄漏"**

## 修复方案

### 1. `_rolling_zscore` 修复

**修复前**：
```python
min_periods = max(10, window // 10)  # 对于 window=288，min_periods=28
```

**修复后**：
```python
min_periods = window  # 默认：必须满窗才输出，最稳健
```

**影响**：
- 对于 `window=288`，`min_periods` 从 28 增加到 288
- 对于 `window=500`，`min_periods` 从 50 增加到 500
- **前 window-1 行全为 NaN，被 drop，但统计更稳健**

### 2. `_rolling_percentile` 修复

**修复前**：
```python
min_periods = 1  # 太小，早期数据不稳定
```

**修复后**：
```python
min_periods = window  # 默认：必须满窗才输出，最稳健
```

### 3. `atr_percentile` 修复（feature_engineering_enhanced.py）

**修复前**：
```python
df["atr_percentile"] = (
    df["atr"].rolling(100, min_periods=20).apply(pct_rank, raw=False)
)
```

**修复后**：
```python
df["atr_percentile"] = (
    df["atr"].rolling(100, min_periods=100).apply(_percentile, raw=True)
)
```

### 4. `atr_compression_ratio` 修复

**修复前**：
```python
atr_mean_hist = (
    data["atr"].rolling(self.percentile_window, min_periods=1).mean()
)
```

**修复后**：
```python
atr_mean_hist = (
    data["atr"].rolling(self.percentile_window, min_periods=self.percentile_window).mean()
)
```

## 新增功能：质量标记（可选）

`_rolling_zscore` 现在支持返回质量分数：

```python
zscore, quality = BaselineFeatureEngineer._rolling_zscore(
    series, window=288, return_quality=True
)
# quality: 0~1，1表示使用了完整窗口
```

**使用场景**：
- **严格模式**：只使用 `quality == 1.0` 的样本
- **宽松模式 + 样本加权**：质量越高，权重越大

## 预期效果

修复后，这些特征与未来收益的相关性应该：
- 从 0.15-0.19 降低到 < 0.05
- OOS Rank IC 可能从 0.0833 降低到更合理的值（0.03-0.05）
- 交易表现应该更稳定，因为消除了虚假信号

**但请注意**：
- 这不是因为"修复了泄漏"，而是因为"剔除了高噪声样本"
- 真正的数据泄漏来自全局统计量（如 `df['col'].mean()`），代码中已确认没有使用

## 真正防泄漏的关键

✅ **永远不要在特征计算中使用**：
- `df['col'].mean()`（全局均值）
- `df['col'].rank(pct=True)`（全局排名）
- `df['col'].quantile()`（全局分位数）

✅ **只使用**：
- `.rolling(w).xxx()`（滚动窗口）
- `.ewm().xxx()`（指数加权移动）

## 总结

| 问题类型 | 是否涉及未来信息？ | 是否属于泄漏？ | 修复方法 |
|---------|------------------|--------------|---------|
| min_periods 太小 | ❌ 否（只用历史） | ❌ 不是泄漏 | 提高 min_periods，剔除高噪声样本 |
| 用全样本 mean() 做标准化 | ✅ 是（用了未来） | ✅ 是泄漏 | 改用 rolling().mean() |

当前代码**没有使用全局统计量**，所以**不存在数据泄漏**。修复的是**统计可靠性问题**，通过提高 `min_periods` 来剔除高噪声样本，降低虚假相关性。
