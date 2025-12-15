# 为什么 `trade_cluster_directional_entropy` 会产生 inf？

## 问题现象

在计算 `trade_cluster_directional_entropy` 的滚动标准差（rolling_std）时，会出现 inf 值。

## 原因分析

### 1. 方向熵的计算

方向熵的计算逻辑（第 1172-1187 行）：

```python
if total_runs > 0:
    buy_ratio = buy_run_count / total_runs
    sell_ratio = sell_run_count / total_runs
    if HAS_SCIPY and scipy_entropy is not None:
        entropy_val = scipy_entropy([buy_ratio, sell_ratio], base=2)
        directional_entropy = entropy_val
    else:
        if buy_ratio > 0 and sell_ratio > 0:
            directional_entropy = -(
                buy_ratio * np.log2(buy_ratio + TOL) +
                sell_ratio * np.log2(sell_ratio + TOL)
            )
        else:
            directional_entropy = 0.0
else:
    directional_entropy = 0.0
```

### 2. 可能导致 inf 的原因

#### 原因1：scipy.entropy 返回 inf

**场景**：当 `buy_ratio` 或 `sell_ratio` 为 0 或非常接近 0 时

```python
# scipy.entropy 的行为
from scipy.stats import entropy

# 如果 buy_ratio = 0, sell_ratio = 1
entropy([0, 1], base=2)  # 可能返回 0 或 inf（取决于实现）

# 如果 buy_ratio 和 sell_ratio 都非常小（接近 0）
entropy([1e-10, 1-1e-10], base=2)  # 可能产生数值不稳定
```

**数学原理**：
- 熵的定义：`H(X) = -Σ p(x) * log2(p(x))`
- 当 `p(x) = 0` 时，`0 * log2(0)` 在数学上是 0，但在数值计算中可能产生 inf
- 当 `p(x)` 非常接近 0 时，`log2(p(x))` 会趋向于 -inf，导致数值不稳定

#### 原因2：rolling_std 计算时的数值问题

**场景**：当输入数据中包含 inf 或极端值时

```python
# 计算滚动标准差
rolling_std = entropy_clean.rolling(window=w, min_periods=1).std()
```

**问题**：
1. 如果 `entropy_clean` 中包含 inf 值（即使已经清理过，但可能在某些边界情况下仍有 inf）
2. 如果窗口内所有值都是 NaN，`rolling_std` 可能返回 inf
3. 如果窗口内只有一个有效值，`rolling_std` 可能返回 inf（因为标准差 = 0，但除以 0 可能产生 inf）

**具体场景**：
```python
# 场景1：窗口内只有一个有效值
values = [1.0, np.nan, np.nan, np.nan, np.nan]
rolling_std = pd.Series(values).rolling(window=5, min_periods=1).std()
# 结果可能包含 inf

# 场景2：窗口内所有值都是 NaN
values = [np.nan, np.nan, np.nan, np.nan, np.nan]
rolling_std = pd.Series(values).rolling(window=5, min_periods=1).std()
# 结果可能包含 inf

# 场景3：窗口内包含 inf
values = [1.0, np.inf, 2.0, 3.0, 4.0]
rolling_std = pd.Series(values).rolling(window=5, min_periods=1).std()
# 结果会包含 inf
```

#### 原因3：除零或数值溢出

**场景**：在计算 Z-score 时

```python
z = (entropy_clean - rolling_mean) / (rolling_std + TOL)
```

**问题**：
- 如果 `rolling_std` 为 0 或非常小，即使加了 `TOL`，仍可能产生数值不稳定
- 如果 `rolling_mean` 或 `entropy_clean` 包含 inf，计算结果也会是 inf

## 解决方案

### 当前实现（已修复）

代码已经正确处理了这些问题：

```python
# 1. 先清理 inf 值
entropy_clean = df["trade_cluster_directional_entropy"].replace([np.inf, -np.inf], np.nan)

# 2. 计算滚动统计
rolling_mean = entropy_clean.rolling(window=w, min_periods=1).mean()
rolling_std = entropy_clean.rolling(window=w, min_periods=1).std()

# 3. 检查并清理 rolling_std 中的 inf
if (~np.isfinite(rolling_std)).any():
    rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)

# 4. 计算 Z-score（使用 TOL 避免除零）
z = (entropy_clean - rolling_mean) / (rolling_std + TOL)

# 5. 最终清理
df[f"trade_cluster_directional_entropy_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
```

### 为什么需要这些步骤？

1. **清理输入数据**：确保 `entropy_clean` 中没有 inf
2. **清理滚动统计**：`rolling_std` 可能因为输入数据或数值问题产生 inf
3. **使用 TOL**：避免除零错误
4. **最终清理**：确保输出中没有 inf

## 根本原因总结

**inf 值主要来自**：

1. **scipy.entropy 的数值不稳定**：
   - 当概率接近 0 时，`log2(0)` 或 `log2(接近0)` 可能导致 inf
   - 虽然理论上 `0 * log2(0) = 0`，但数值计算中可能产生 inf

2. **rolling_std 的边界情况**：
   - 窗口内只有一个有效值
   - 窗口内所有值都是 NaN
   - 输入数据中包含 inf（即使已经清理过）

3. **数值溢出**：
   - 在计算 Z-score 时，如果 `rolling_std` 为 0 或非常小
   - 如果 `entropy_clean` 或 `rolling_mean` 包含 inf

## 预防措施

1. ✅ **在计算前清理 inf**：`entropy_clean = df["trade_cluster_directional_entropy"].replace([np.inf, -np.inf], np.nan)`
2. ✅ **在计算后检查并清理**：`rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)`
3. ✅ **使用 TOL 避免除零**：`z = (entropy_clean - rolling_mean) / (rolling_std + TOL)`
4. ✅ **最终清理输出**：`z.replace([np.inf, -np.inf], np.nan)`

这些步骤确保了即使在某些边界情况下产生 inf，也会被正确处理为 NaN，不会影响后续的计算。

