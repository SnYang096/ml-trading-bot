# strategies 目录删除建议

## 📋 分析结果

**结论：不能全部删除，但可以部分删除**

---

## 🔍 详细分析

### 使用情况

| 文件 | 被使用位置 | 是否必需 | 建议 |
|------|-----------|---------|------|
| `ClassificationStrategyHandler` | `auto_rolling_update.py` | ⚠️ 看情况 | 如果 `auto_rolling_update.py` 不使用，可以删除 |
| `BaseStrategyHandler` | 基类 | ✅ 必需 | 保留（如果保留 handler） |
| `MLTradingStrategy` | `vectorbot.py` (legacy) | ⚠️ 向后兼容 | 标记为 deprecated |
| `QuantileStrategyHandler` | 只被 `MLTradingStrategy` 使用 | ❌ 不需要 | **可以删除** |

---

## 🎯 关键发现

### 1. `rolling.py` 不使用 `ClassificationStrategyHandler`

**发现**:
- `rolling.py` **自己生成信号**（第 1176-1187 行）
- 不使用 `ClassificationStrategyHandler`
- 信号生成逻辑简单：`risk_adjusted_signal = (2 * y_prob - 1) * (y_pred_return / vol_pred)`

### 2. 只有 `auto_rolling_update.py` 使用 `ClassificationStrategyHandler`

**代码位置**:
```python
# auto_rolling_update.py (第 498 行)
strategy_handler = ClassificationStrategyHandler(...)
signals_df = strategy_handler.generate_signals(...)
```

**问题**: `auto_rolling_update.py` 是否还在使用？

**检查结果**:
- ❌ Makefile 中没有 `auto-rolling-update` 目标
- ✅ README.md 中有提到，但可能是旧的文档
- ⚠️ 需要确认是否仍在使用

---

## 💡 建议方案

### 方案 1: 如果只使用 `make rolling`（推荐）

**可以删除**:
- ✅ `QuantileStrategyHandler` - 只被 deprecated 的 `MLTradingStrategy` 使用
- ✅ `MLTradingStrategy` - 如果不需要向后兼容 legacy 格式
- ✅ `ClassificationStrategyHandler` - 如果 `auto_rolling_update.py` 不使用
- ✅ `BaseStrategyHandler` - 如果没有 handler 了

**保留**:
- ⚠️ `MLTradingStrategy` - 如果 `vectorbot.py` 需要加载 legacy 格式（向后兼容）

---

### 方案 2: 统一信号生成逻辑（更彻底）

**问题**:
- `rolling.py` 自己生成信号（简单逻辑）
- `auto_rolling_update.py` 使用 `ClassificationStrategyHandler`（复杂逻辑）
- 两者逻辑不一致

**建议**:
1. **统一使用 `ClassificationStrategyHandler`**:
   - 让 `rolling.py` 也使用 `ClassificationStrategyHandler` 生成信号
   - 这样可以统一信号生成逻辑

2. **或者提取信号生成逻辑**:
   - 将信号生成逻辑提取到独立模块
   - `rolling.py` 和 `auto_rolling_update.py` 都使用这个模块

---

## 🔧 实施步骤

### 步骤 1: 删除 `QuantileStrategyHandler`

```bash
# 删除文件
rm src/time_series_model/strategies/quantile_strategy_handler.py

# 更新 ml_strategy.py（如果保留的话）
# 移除 QuantileStrategyHandler 的导入
```

### 步骤 2: 检查 `auto_rolling_update.py` 是否仍在使用

```bash
# 检查 Makefile
grep -r "auto-rolling-update" Makefile

# 检查是否有其他脚本调用
grep -r "auto_rolling_update" .
```

### 步骤 3: 根据检查结果决定

**如果 `auto_rolling_update.py` 不使用**:
- ✅ 删除 `ClassificationStrategyHandler`
- ✅ 删除 `BaseStrategyHandler`
- ⚠️ 保留 `MLTradingStrategy`（用于向后兼容）

**如果 `auto_rolling_update.py` 仍在使用**:
- ✅ 保留 `ClassificationStrategyHandler`
- ✅ 保留 `BaseStrategyHandler`
- ⚠️ 保留 `MLTradingStrategy`（用于向后兼容）

### 步骤 4: 统一信号生成（可选但推荐）

- 让 `rolling.py` 也使用 `ClassificationStrategyHandler`
- 或者提取信号生成逻辑到独立模块

---

## 📝 我的建议

### 推荐方案：部分删除 + 统一信号生成

1. **删除** `QuantileStrategyHandler` ✅
2. **标记** `MLTradingStrategy` 为 deprecated ⚠️
3. **统一信号生成**：
   - 让 `rolling.py` 也使用 `ClassificationStrategyHandler`
   - 这样可以统一逻辑，减少重复代码
4. **保留** `ClassificationStrategyHandler` 和 `BaseStrategyHandler`（如果统一信号生成）

**理由**:
- `rolling.py` 和 `auto_rolling_update.py` 应该使用相同的信号生成逻辑
- 统一后可以减少代码重复
- `ClassificationStrategyHandler` 的逻辑更完善（考虑置信度、收益方向等）

---

## ✅ 总结

### 可以删除
- ✅ `QuantileStrategyHandler` - 只被 deprecated 的 `MLTradingStrategy` 使用

### 需要确认
- ⚠️ `ClassificationStrategyHandler` - 取决于 `auto_rolling_update.py` 是否仍在使用
- ⚠️ `BaseStrategyHandler` - 取决于是否保留 handler

### 建议保留但标记为 deprecated
- ⚠️ `MLTradingStrategy` - 用于向后兼容（vectorbot 加载 legacy 格式）

### 推荐行动
1. **立即删除** `QuantileStrategyHandler`
2. **检查** `auto_rolling_update.py` 是否仍在使用
3. **统一信号生成**：让 `rolling.py` 也使用 `ClassificationStrategyHandler`
4. **标记** `MLTradingStrategy` 为 deprecated

