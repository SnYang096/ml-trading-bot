# RSI "Inf" 值问题根本原因分析

## 问题发现

通过完整的训练流程追踪测试，发现了问题的根本原因：

### 训练日志显示
```
⚠️  Train before drop_inf_rows: found inf/-inf in 10 columns (top): {'rsi': 70, ...}
```

### 实际情况
测试显示这 70 个值**不是 inf，而是 NaN**！

## 根本原因

`_debug_inf` 函数使用了 `np.isfinite()` 来检查 inf 值：

```python
inf_mask = ~np.isfinite(df[feature_cols])
```

**问题**：`np.isfinite()` 对以下值都返回 `False`：
- `inf`
- `-inf`
- `NaN`

所以这个函数会把 **NaN 也误报为 inf**！

## 证据

测试输出显示：
```
步骤 7: drop_inf_rows 之前（这是训练日志显示的位置）
   使用 _debug_inf 逻辑检查:
      RSI Inf 数量: 70
      ⚠️  发现 70 个 inf 值！
      前 10 个位置:
         2025-02-01 00:00:00: nan (type: float64)
            ℹ️  是 NaN
```

## RSI NaN 的原因

RSI 需要至少 14 个周期才能计算，所以：
- 前 14 个数据点：RSI = NaN（正常）
- 数据不足的窗口：RSI = NaN（正常）

这 70 个 NaN 是**正常的**，不是数据质量问题！

## 修复方案

已修复 `_debug_inf` 函数，使用 `np.isinf()` 而不是 `~np.isfinite()`：

```python
# 修复前（错误）
inf_mask = ~np.isfinite(df[feature_cols])  # 会误报 NaN 为 inf

# 修复后（正确）
inf_mask = np.isinf(df[feature_cols])  # 只检查真正的 inf/-inf
```

## 结论

1. **RSI 没有 inf 值问题**：所有测试都显示 RSI 计算正常，没有真正的 inf 值
2. **训练日志的误报**：70 个 "inf" 实际上是 70 个 NaN，这是正常的
3. **修复完成**：`_debug_inf` 函数已修复，现在会正确区分 inf 和 NaN

## 建议

1. ✅ 问题已解决：RSI 特征计算正常，无需修复
2. ✅ 日志已修复：现在会正确区分 inf 和 NaN
3. ℹ️  NaN 是正常的：RSI 前 14 个周期为 NaN 是预期行为

