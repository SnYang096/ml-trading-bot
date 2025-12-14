# EVT excesses 统计量说明

## 什么是 excesses？

在 EVT（极值理论）中，**excesses（超额值）**是指超过某个阈值的数据点。

### 左尾（暴跌风险）的 excesses

```python
# 1. 计算阈值（例如：10% 分位数）
u_left = np.quantile(window_returns, 0.1)  # 例如：-0.02（-2%）

# 2. 找出所有低于阈值的收益率（极端下跌）
extreme_losses = window_returns[window_returns < u_left]
# 例如：[-0.03, -0.025, -0.04, -0.035, ...]

# 3. 计算 excesses（超额损失，转换为正数）
excesses_left = u_left - extreme_losses
# 例如：u_left = -0.02, extreme_losses = -0.03
# excesses_left = -0.02 - (-0.03) = 0.01（1%的超额损失）
```

**含义**：
- `excesses_left` 表示"超过阈值的损失幅度"
- 例如：如果阈值是 -2%，实际损失是 -3%，那么 excess 是 1%（即超出阈值的部分）

### 右尾（泡沫风险）的 excesses

```python
# 1. 计算阈值（例如：90% 分位数）
u_right = np.quantile(window_returns, 0.9)  # 例如：0.02（+2%）

# 2. 找出所有高于阈值的收益率（极端上涨）
extreme_gains = window_returns[window_returns > u_right]
# 例如：[0.03, 0.025, 0.04, 0.035, ...]

# 3. 计算 excesses（超额收益）
excesses_right = extreme_gains - u_right
# 例如：u_right = 0.02, extreme_gains = 0.03
# excesses_right = 0.03 - 0.02 = 0.01（1%的超额收益）
```

## 实际 excesses 的统计量

当 GPD（广义帕累托分布）拟合失败或数据不足时，我们使用**基于实际 excesses 的统计量**作为保守估计。

### 1. 经验分位数（Empirical Quantile）

```python
# 左尾：使用 excesses 的 99% 分位数作为保守的 VaR 估计
empirical_var = u_left - np.quantile(excesses_left, 0.99)
```

**含义**：
- `np.quantile(excesses_left, 0.99)` 表示 excesses 中 99% 的数据都小于这个值
- 例如：如果 excesses = [0.01, 0.015, 0.02, 0.025, 0.03, ...]
  - `np.quantile(excesses_left, 0.99)` 可能是 0.028
  - 这意味着 99% 的 excesses 都小于 0.028（即 2.8% 的超额损失）

**为什么用 99% 分位数？**
- 对应 99% VaR（Value at Risk）
- 表示"99% 的情况下，超额损失不会超过这个值"
- 这是一个保守的估计

### 2. 均值（Mean）

```python
# 当 excesses 太少（只有1个）时，使用均值
empirical_var = u_left - np.mean(excesses_left)
```

**含义**：
- 当 excesses 只有1个数据点时，无法计算分位数
- 使用均值作为简单的估计
- 例如：如果 excesses = [0.02]，那么 `np.mean(excesses_left) = 0.02`

### 3. 实际代码示例

#### 左尾（暴跌风险）

```python
# 情况1：excesses 不足（少于 min_excesses=10）
if len(excesses_left) > 0:
    # 使用经验分位数作为保守估计
    empirical_var = u_left - np.quantile(excesses_left, 0.99) if len(excesses_left) > 1 else u_left - np.mean(excesses_left)
    var_99_left[df_idx] = min(empirical_var, -0.001)  # 确保是负值且很小
    es_99_left[df_idx] = var_99_left[df_idx] * 1.1  # ES 比 VaR 更差（更负）
```

**示例计算**：
```python
# 假设：
u_left = -0.02  # 阈值：-2%
excesses_left = [0.01, 0.015, 0.02, 0.025, 0.03]  # 5个 excesses（不足10个）

# 计算：
quantile_99 = np.quantile(excesses_left, 0.99)  # 约 0.03（99%分位数）
empirical_var = -0.02 - 0.03 = -0.05  # -5% 的 VaR
var_99_left = min(-0.05, -0.001) = -0.05  # 使用计算值
es_99_left = -0.05 * 1.1 = -0.055  # ES 比 VaR 更差
```

#### 右尾（泡沫风险）

```python
# 情况2：拟合失败
except Exception:
    if len(excesses_right) > 0:
        # 使用经验分位数作为保守估计
        empirical_var = u_right + np.quantile(excesses_right, 0.99) if len(excesses_right) > 1 else u_right + np.mean(excesses_right)
        var_99_right[df_idx] = max(empirical_var, 0.001)  # 确保是正值且很小
        es_99_right[df_idx] = var_99_right[df_idx] * 1.1  # ES 比 VaR 更好（更正）
```

**示例计算**：
```python
# 假设：
u_right = 0.02  # 阈值：+2%
excesses_right = [0.01, 0.015, 0.02, 0.025, 0.03]  # 5个 excesses

# 计算：
quantile_99 = np.quantile(excesses_right, 0.99)  # 约 0.03（99%分位数）
empirical_var = 0.02 + 0.03 = 0.05  # +5% 的 VaR
var_99_right = max(0.05, 0.001) = 0.05  # 使用计算值
es_99_right = 0.05 * 1.1 = 0.055  # ES 比 VaR 更好
```

## 为什么使用这些统计量？

### 1. 基于实际数据
- 不是任意填充，而是基于实际观察到的极端事件
- 例如：如果实际 excesses 是 [0.01, 0.015, 0.02]，那么估计值会反映这些实际数据

### 2. 保守估计
- 使用 99% 分位数，表示"99% 的情况下不会超过这个值"
- 对于风险管理，保守估计更安全

### 3. 简单有效
- 当 GPD 拟合失败时，使用简单的统计量作为替代
- 比完全缺失（NaN）更有信息量

## 对比：GPD 拟合 vs 经验统计量

### GPD 拟合（理想情况）

```python
# 当 excesses >= 10 时，使用 GPD 拟合
xi_l, loc, sigma_l = genpareto.fit(excesses_left, floc=0)
# 使用理论公式计算 VaR 和 ES
var_99_left = u_left - (sigma_l / xi_l) * ((p_level / tail_prob_left) ** (-xi_l) - 1)
```

**优点**：
- 基于极值理论，更科学
- 可以外推到更极端的概率（如 99.9% VaR）

**缺点**：
- 需要足够的 excesses（至少 10 个）
- 拟合可能失败

### 经验统计量（备用方案）

```python
# 当 excesses < 10 或拟合失败时，使用经验统计量
empirical_var = u_left - np.quantile(excesses_left, 0.99)
```

**优点**：
- 简单直接，基于实际数据
- 不需要拟合，不会失败
- 即使数据少也能计算

**缺点**：
- 不能外推到更极端的概率
- 可能不够精确

## 总结

**实际 excesses 的统计量**是指：
1. **经验分位数**：`np.quantile(excesses, 0.99)` - 99% 分位数
2. **均值**：`np.mean(excesses)` - 当数据太少时使用

这些统计量基于**实际观察到的极端事件**（excesses），而不是任意填充的默认值。

**关键点**：
- ✅ 基于实际数据：值来自实际观察到的 excesses
- ✅ 保守估计：使用 99% 分位数，表示"99% 的情况下不会超过"
- ✅ 简单有效：当 GPD 拟合失败时，提供合理的替代估计

