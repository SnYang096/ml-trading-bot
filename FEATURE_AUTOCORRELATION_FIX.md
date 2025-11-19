# 特征自相关偏差修复报告

## 🔍 核心发现：自相关偏差（Auto-correlation Bias）

### 问题根源

`_rolling_percentile` 的实现包含了**当前值**，导致自相关偏差：

```python
# ❌ 错误实现（修复前）
def _percentile(x: np.ndarray) -> float:
    last = x[-1]  # 当前值
    arr = x[np.isfinite(x)]  # 包含当前值！
    return (arr <= last).sum() / float(len(arr))  # 当前值自己也算进去了
```

**问题分析**：
- `x` 是 rolling window 的完整窗口（包含当前值）
- `(arr <= last).sum()` 包含了当前值自己
- 所以 `percentile = (number of values ≤ current) / total`
- **这会导致：当前值越大，percentile 越高 → 与 future_return 正相关！**

### 为什么会产生虚假相关？

**举例说明**：
```
窗口 = [1, 2, 3, 4] → 当前=4，percentile = 4/4 = 1.0
窗口 = [4, 3, 2, 1] → 当前=1，percentile = 1/4 = 0.25
```

**真实市场中的情况**：
- 当前价格高 → percentile 高
- 如果价格有动量效应（当前高 → 未来可能更高）
- 就会出现：`atr_percentile` 与 `future_return` 正相关

**但这不是未来信息，而是自相关偏差（auto-correlation bias）**：
- 它让特征与当前价格强相关
- 而当前价格又与短期 future_return 相关（动量效应）
- 从而产生虚假信号

### 为什么 Random Walk Test 通过，但真实数据仍有高相关？

| 测试类型 | 结果 | 原因 |
|---------|------|------|
| Random Walk | Avg IC = 0.0146 ✅ | 随机游走无自相关，`_percentile` 包含当前值也不会产生系统性偏差 |
| 真实市场数据 | `atr_percentile` corr = 0.1685 ⚠️ | 真实价格有动量效应（当前高 → 未来可能更高），导致"包含当前的 percentile"与 future_return 正相关 |

## ✅ 修复方案

### 1. 修复 `_rolling_percentile`（关键！）

**修复前**：
```python
def _percentile(x: np.ndarray) -> float:
    last = x[-1]
    arr = x[np.isfinite(x)]  # 包含当前值
    return (arr <= last).sum() / float(len(arr))
```

**修复后**：
```python
def _percentile(x: np.ndarray) -> float:
    current = x[-1]  # 当前值
    history = x[:-1]  # ← 关键修复：只用历史，排除当前值！
    history = history[np.isfinite(history)]
    return (history <= current).sum() / float(len(history))
```

**修复位置**：
- `src/data_tools/baseline_features.py` 第 1156-1171 行
- `src/data_tools/feature_engineering_enhanced.py` 第 832-841 行

### 2. 强制所有 `_rolling_zscore` 使用 `min_periods=window`

**修复前**：
```python
result[zscore_col] = BaselineFeatureEngineer._rolling_zscore(
    result[base_col], window=window  # 使用默认 min_periods
)
```

**修复后**：
```python
result[zscore_col] = BaselineFeatureEngineer._rolling_zscore(
    result[base_col], window=window, min_periods=window  # 强制满窗才输出
)
```

**修复位置**：
- `src/data_tools/baseline_features.py` 第 966-968 行

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
- Sharpe 仍低（≈0），但结果可信
- 消除了虚假信号，特征更可靠

## 🎯 问题类型总结

| 问题 | 类型 | 是否泄漏？ | 修复方式 |
|------|------|-----------|---------|
| `_rolling_percentile` 包含当前值 | **特征设计偏差** | ❌ 不是传统泄漏 | 改为只用 `x[:-1]` |
| `_rolling_zscore` 早期噪声大 | 统计不稳定 | ❌ 不是泄漏 | 设置 `min_periods=window` |
| ATR 特征与 future_return 高相关 | 结果 | - | 上述修复后应消失 |

## 💡 关键洞察

**这不是传统意义上的"数据泄漏"，而是"特征设计偏差"，但在量化中同样致命！**

- ✅ 传统泄漏：使用未来数据（如 `df['col'].mean()`）
- ⚠️ 设计偏差：特征与当前价格强相关，而当前价格又与未来收益相关（动量效应）

两者都会导致虚假信号，都需要修复！

## ✅ 修复完成

所有修复已完成：
1. ✅ `_rolling_percentile` 改为只使用历史数据（`x[:-1]`）
2. ✅ 所有 `_rolling_zscore` 调用强制使用 `min_periods=window`
3. ✅ 修复了 `baseline_features.py` 和 `feature_engineering_enhanced.py` 中的实现

下一步：重新运行训练，验证修复效果！

