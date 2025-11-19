# 特征时间对齐（Feature Timing Alignment）核心原则

## 📌 核心原则：决策点决定一切

**用于生成 t 时刻交易信号的所有信息，必须在 t 时刻决策点之前已知。**

## 🎯 关键问题：能否使用 close[t]？

### 答案：**不一定所有 t 时刻的 close 都不能用，但必须严格遵循决策点原则**

### 决策场景分析

| 决策场景 | 决策时间点 | 可用的最新价格 | 特征计算方式 | 能否使用 close[t]? |
|---------|-----------|--------------|-------------|------------------|
| **收盘价交易者** (Close-to-Close) | t 时刻收盘后 | close[t] | 基于 ..., close[t-1], close[t] | ✅ **可以** |
| **开盘价交易者** (Open-to-Close) | t+1 时刻开盘前 | close[t] | 基于 ..., close[t-1], close[t] | ✅ **可以** |
| **盘中交易者** (Intraday) | t 时刻盘中 (e.g., 14:00) | price[t, 14:00] | 基于截至 14:00 的数据 | ❌ **不能用 close[t]** |

### 💡 关键洞察

- `close[t]` 在 t 根K线走完时才确定
- 如果你的交易逻辑是在 t 根K线结束后才执行（比如收盘价下单，或在下一根K线开盘时下单）
- 那么 `close[t]` 就是历史数据，**完全可以使用**
- **绝大多数日频/小时频回测都属于这种场景**

## 🔍 回到具体问题：_rolling_percentile

### 假设：收盘价交易者或开盘价交易者（最常见的情况）

- **决策点**：在 t 时刻K线结束后
- **目标**：预测 `future_return[t+1]`（即从 t+1 开始的收益）
- **可用数据**：所有截至 t 时刻的数据，包括 `close[t]`

✅ **在这种情况下，使用 close[t] 计算特征是完全合法且正确的！**

### ❓ 那为什么 _rolling_percentile 还是有问题？

**问题不在于用了 close[t]，而在于如何用！**

#### ✅ 正确用法：计算 close[t] 相对于过去N天（不含今天）的位置

```python
# 正确：在 t 时刻，用 [t-N, t-1] 的历史来评估 close[t] 的强弱
percentile[t] = rank(close[t-N], ..., close[t-1]) 中 <= close[t] 的比例
```

这个 `percentile[t]` 是一个基于历史对当前状态的评估，完全合法。

#### ❌ 错误用法：计算 close[t] 相对于过去N-1天 + 今天的位置

```python
# 错误：在 t 时刻，把 close[t] 自己也放进历史窗口里去排名
percentile[t] = rank(close[t-N+1], ..., close[t-1], close[t]) 中 <= close[t] 的比例
```

这相当于问："我自己在我自己里面排第几？" 答案永远是100%（如果无重复）。这放大了当前值的影响，制造了偏差。

## 🧩 总结：何时能用 close[t]？

| 问题 | 答案 |
|------|------|
| 我能不能用 t 时刻的 close 来预测 t+1 的收益？ | ✅ **可以！** 只要你的交易决策发生在 t 时刻K线结束之后。 |
| 那 _rolling_percentile 为什么错了？ | 因为它把 close[t] 同时当作了"被评估的对象"和"评估的标尺"，导致自我参照偏差。 |
| 正确的做法是什么？ | 将 close[t] 作为"新来的考生"，用 [t-N, t-1] 这群"老考生"的成绩作为"分数线"来给它打分。而不是把它和老考生混在一起重新排名。 |

## 🛠 代码层面的修正

### ❌ 错误：窗口包含当前值 x[-1]

```python
window_with_current = x  # [x0, x1, ..., x_{t-1}, x_t]
percentile = (window_with_current <= x_t).sum() / len(window_with_current)
```

### ✅ 正确：窗口只包含历史值

```python
history_only = x[:-1]  # [x0, x1, ..., x_{t-1}]
current_value = x[-1]  # x_t
percentile = (history_only <= current_value).sum() / len(history_only)
```

### 这样修正后的优势

- ✅ 你依然使用了 close[t]（作为 current_value）
- ✅ 但评估它的基准（history_only）完全是历史数据
- ✅ 既利用了最新的价格信息，又避免了自我参照偏差
- ✅ **这才是真实、可交易、无偏差的动量信号**

## 📊 实现细节

### 当前实现（已修复）

```python
def _percentile(x: np.ndarray) -> float:
    """
    计算当前值在历史窗口中的百分位排名（严格因果，无自我参照偏差）
    
    【核心原则：特征时间对齐】
    - 决策点：在 t 时刻K线结束后做决策，预测 future_return[t+1]
    - 可用数据：所有截至 t 时刻的数据，包括 close[t]（这是历史数据）
    - 正确用法：计算 close[t] 相对于过去N天（不含今天）的位置
    - 错误用法：把 close[t] 自己也放进历史窗口里去排名（自我参照偏差）
    
    【实现说明】
    - current = x[-1]：当前值（如 close[t]），作为"新来的考生"
    - history = x[:-1]：历史窗口（如 [t-N, t-1]），作为"老考生的成绩分数线"
    - percentile = (history <= current).sum() / len(history)
      表示：当前值在历史中的相对位置，完全基于历史评估当前状态
    """
    if len(x) < 2 or not np.isfinite(x[-1]):
        return np.nan
    current = x[-1]  # 当前值（如 close[t]），作为"新来的考生"
    history = x[:-1]  # ← 关键：只用历史（如 [t-N, t-1]），作为"老考生的成绩分数线"
    history = history[np.isfinite(history)]
    if len(history) == 0:
        return np.nan
    # 当前值在历史中的分位：(历史中 ≤ 当前值的数量) / 历史总数量
    # 这表示：当前值相对于历史的位置，完全基于历史评估当前状态
    return (history <= current).sum() / float(len(history))
```

## 🎯 关键要点

1. **决策点决定一切**：在 t 时刻K线结束后做决策，可以使用 close[t]
2. **正确用法**：用历史窗口评估当前值，而不是把当前值放进历史窗口
3. **避免自我参照偏差**：不要问"我自己在我自己里面排第几？"
4. **真实可交易信号**：基于历史评估当前状态，这才是真实、可交易、无偏差的动量信号

## ✅ 修复状态

- ✅ `_rolling_percentile` 已修复：只使用历史数据（x[:-1]）评估当前值（x[-1]）
- ✅ 所有 `_rolling_zscore` 调用强制使用 `min_periods=window`
- ✅ 代码注释已更新，明确说明特征时间对齐原则

所有修复已完成，符合特征时间对齐的核心原则！

