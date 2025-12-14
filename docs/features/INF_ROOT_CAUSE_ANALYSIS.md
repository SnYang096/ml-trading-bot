# Inf 值根本原因分析

## 问题现象

从训练输出（`@bash (339-433)`）看，在完整训练流程（2025-01-01 到 2025-07-31）中，出现了大量 inf 值：

- `sr_strength_max`: 706个inf，样本在 `2025-01-03`（1月初）
- `hurst_price_rolling`: 298个inf，样本在 `2025-01-03`（1月初）
- `hurst_cvd_rolling`: 298个inf，样本在 `2025-01-03`（1月初）
- `rsi`: 70个inf，样本在 `2025-02-01`（2月初）
- `trade_cluster_*_zscore_*`: 5个inf，样本在每月1日

## 关键发现

### 1. 测试结果 vs 训练输出

**集成测试结果**（`tests/features/test_inf_root_cause_integration.py`）：
- 在单独加载1-2月数据时，**没有发现任何inf值**
- 所有测试都通过（PASSED）

**训练流程输出**：
- 在完整训练流程（1-7月）中，出现了大量inf值
- inf值都出现在**数据集的早期**（1-2月），而不是后期（6-7月）

### 2. 问题定位

通过诊断脚本（`scripts/diagnose_early_data_inf.py`）发现：
- 在单独加载1-2月数据时，特征计算是正常的（没有inf）
- 但在完整训练流程中，却出现了inf值

**可能的原因**：
1. **特征依赖关系**：某些特征可能依赖于其他特征，而这些依赖在完整流程中可能有问题
2. **数据对齐问题**：在完整流程中，不同特征的计算顺序可能导致数据对齐问题
3. **缓存问题**：训练流程使用了缓存，可能缓存的数据有问题
4. **边界条件**：早期数据的某些计算（如rolling mean/std）可能因为数据不足产生inf

### 3. 根本原因

**`calculate_sqs` 函数中的问题**：

在 `src/features/time_series/baseline_features.py` 的 `calculate_sqs` 函数中：

```python
# 使用窗口内最后一个 ATR（即最新可用ATR）
current_atr = df["atr"].iloc[-1]
if current_atr <= 0:  # ❌ 只检查了 <= 0，没有检查 NaN 或 inf
    return 0.0

# 后续除法操作
weighted_reaction = (reaction / current_atr) * np.sqrt(vol_factor)  # ❌ 如果 current_atr 是 inf，会产生 inf
reactions.append(reaction / current_atr)  # ❌ 如果 current_atr 是 inf，会产生 inf
```

**问题**：
1. 只检查了 `current_atr <= 0`，但没有检查 `NaN` 或 `inf`
2. 在除法操作前，没有确保 `current_atr` 是有限正数
3. 在添加 `reactions` 前，没有检查计算结果是否为有限值

## 修复方案

### 1. 修复 `calculate_sqs` 函数

```python
# 使用窗口内最后一个 ATR（即最新可用ATR）
current_atr = df["atr"].iloc[-1]
# ✅ 检查 ATR 是否有效（必须是有限的正数）
if not np.isfinite(current_atr) or current_atr <= 0:
    return 0.0

# ✅ 在除法操作前，再次确认 current_atr 是有限正数
if np.isfinite(current_atr) and current_atr > 0:
    weighted_reaction = (reaction / current_atr) * np.sqrt(vol_factor)
    # ✅ 检查计算结果是否为有限值
    if np.isfinite(weighted_reaction):
        reactions.append(weighted_reaction)
```

### 2. 其他修复

- 移除了不必要的调试打印（`vol_ratio is inf/NaN`），避免输出过多信息
- 确保所有除法操作都有适当的检查

## 验证

1. **集成测试**：`tests/features/test_inf_root_cause_integration.py` 已通过
2. **诊断脚本**：`scripts/diagnose_early_data_inf.py` 显示在单独加载1-2月数据时没有inf值
3. **需要重新运行训练流程**：验证修复是否有效

## 下一步

1. 重新运行训练流程，验证修复是否有效
2. 如果仍有inf值，需要进一步检查：
   - 特征计算顺序
   - 缓存机制
   - 数据对齐问题

