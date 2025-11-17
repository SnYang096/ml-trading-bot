# strategies 目录是否可以删除的分析

## 📋 分析结果

**结论：不能全部删除，但可以部分删除**

---

## 🔍 文件使用情况

### 1. `ClassificationStrategyHandler` - ✅ **必须保留**

**被使用**:
- `auto_rolling_update.py` (第 498 行) - 用于生成信号

**代码位置**:
```python
# auto_rolling_update.py
strategy_handler = ClassificationStrategyHandler(
    pipeline_stub,
    signal_strength_threshold=0.0,
    confidence_threshold=0.0,
    base_position_size=1.0,
    classification_threshold=0.5,
)
signals_df = strategy_handler.generate_signals(X_test_df, test_slice, timeframe="default")
```

**结论**: ✅ **不能删除**，rolling 流程中需要使用

---

### 2. `BaseStrategyHandler` - ✅ **必须保留**

**被使用**:
- `ClassificationStrategyHandler` 继承自它
- `QuantileStrategyHandler` 继承自它

**结论**: ✅ **不能删除**，是基类

---

### 3. `MLTradingStrategy` - ⚠️ **可以标记为 deprecated**

**被使用**:
- `vectorbot.py` (第 18, 93 行) - 用于加载 legacy 格式的模型

**代码位置**:
```python
# vectorbot.py
from time_series_model.strategies.ml_strategy import MLTradingStrategy

# Legacy format: single pickle file
self.strategy: MLTradingStrategy = model_data["strategy"]
```

**结论**: ⚠️ **可以标记为 deprecated**，但需要保留用于向后兼容（加载旧的 .pkl 文件）

---

### 4. `QuantileStrategyHandler` - ❌ **可以删除**

**被使用**:
- 只被 `MLTradingStrategy` 使用（第 162 行）
- 没有其他地方使用

**代码位置**:
```python
# ml_strategy.py
self.strategy_handler = QuantileStrategyHandler(
    pipeline=self.pipeline,
    ...
)
```

**结论**: ❌ **可以删除**，因为：
- 只被 `MLTradingStrategy` 使用
- `MLTradingStrategy` 已经 deprecated
- rolling 不使用 quantile 模式（只使用 classification 模式）

---

## 📊 使用情况总结

| 文件 | 被使用位置 | 是否可以删除 | 建议 |
|------|-----------|------------|------|
| `ClassificationStrategyHandler` | `auto_rolling_update.py` | ❌ 不能 | ✅ 保留 |
| `BaseStrategyHandler` | 基类 | ❌ 不能 | ✅ 保留 |
| `MLTradingStrategy` | `vectorbot.py` (legacy) | ⚠️ 可以标记 deprecated | ⚠️ 保留但标记 |
| `QuantileStrategyHandler` | 只被 `MLTradingStrategy` 使用 | ✅ 可以 | ❌ 删除 |

---

## 🎯 关键发现

### rolling.py 不使用 ClassificationStrategyHandler

**发现**:
- `rolling.py` **不使用** `ClassificationStrategyHandler`
- 它自己生成信号（第 1176-1187 行）：
  ```python
  risk_adjusted_signal = (
      (2 * y_prob - 1) *
      (y_pred_return / np.maximum(y_pred_vol, 1e-6)))
  signals_df = pd.DataFrame({
      "signal_strength": risk_adjusted_signal,
      "class_proba": y_prob,
      "return_pred": y_pred_return,
      "vol_pred": y_pred_vol,
  })
  ```

**只有 `auto_rolling_update.py` 使用 `ClassificationStrategyHandler`**

---

## 💡 建议方案

### 方案 1: 部分删除（推荐）

**删除**:
- `QuantileStrategyHandler` - 只被 deprecated 的 `MLTradingStrategy` 使用

**保留但标记为 deprecated**:
- `MLTradingStrategy` - 用于向后兼容（vectorbot 加载 legacy 格式）

**保留**:
- `ClassificationStrategyHandler` - 被 `auto_rolling_update.py` 使用
- `BaseStrategyHandler` - 基类

---

### 方案 2: 统一信号生成逻辑

**问题**:
- `rolling.py` 自己生成信号（简单逻辑）
- `auto_rolling_update.py` 使用 `ClassificationStrategyHandler`（复杂逻辑）
- 两者逻辑不一致

**建议**:
- 统一使用 `ClassificationStrategyHandler` 生成信号
- 或者将信号生成逻辑提取到独立模块

---

## 🔧 实施建议

### 步骤 1: 删除 QuantileStrategyHandler

```bash
# 删除文件
rm src/time_series_model/strategies/quantile_strategy_handler.py

# 更新 ml_strategy.py（如果保留的话）
# 移除 QuantileStrategyHandler 的导入和使用
```

### 步骤 2: 标记 MLTradingStrategy 为 deprecated

```python
# ml_strategy.py
class MLTradingStrategy:
    """
    DEPRECATED: This class is deprecated.
    Use 'make rolling' instead, which uses QuantTradingModel pipelines.
    
    This class is kept for backward compatibility only (loading legacy .pkl files).
    """
    ...
```

### 步骤 3: 统一信号生成（可选）

- 让 `rolling.py` 也使用 `ClassificationStrategyHandler`
- 或者将信号生成逻辑提取到独立模块

---

## 📝 总结

### 可以删除
- ✅ `QuantileStrategyHandler` - 只被 deprecated 的 `MLTradingStrategy` 使用

### 必须保留
- ✅ `ClassificationStrategyHandler` - 被 `auto_rolling_update.py` 使用
- ✅ `BaseStrategyHandler` - 基类

### 可以标记为 deprecated
- ⚠️ `MLTradingStrategy` - 用于向后兼容（vectorbot 加载 legacy 格式）

### 建议
1. **删除** `QuantileStrategyHandler`
2. **标记** `MLTradingStrategy` 为 deprecated
3. **保留** `ClassificationStrategyHandler` 和 `BaseStrategyHandler`
4. **考虑** 统一信号生成逻辑（让 rolling.py 也使用 ClassificationStrategyHandler）

