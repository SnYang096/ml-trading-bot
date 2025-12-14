# SR Strength Max 全部为 NaN 问题根本原因分析

## 问题描述

训练日志显示：
- 训练集中 `sr_strength_max` 有 706 个 inf（实际可能是 NaN）
- 全部为 NaN，说明计算过程中某个步骤失败了

## 根本原因

### 1. **代码结构错误（已修复）**

`compute_sr_strength_max` 函数被 `compute_footprint_features` 函数打断，导致关键代码未执行：

**问题代码结构**：
```python
def compute_sr_strength_max(...):
    # 1. 获取边界定义
    boundaries = ...
    if not boundaries:
        return result
    # ❌ 函数在这里就结束了，下面的代码永远不会执行！

def compute_footprint_features(...):
    # 这是另一个函数
    ...

# ❌ 这些代码永远不会执行（死代码）
# 2. 计算边界强度
boundary_strengths = ...
# 3. 找到最大强度
result["sr_strength_max"] = ...
```

**修复后**：
```python
def compute_sr_strength_max(...):
    # 1. 获取边界定义
    boundaries = ...
    if not boundaries:
        return result
    
    # 2. 计算边界强度 ✅ 现在会执行
    boundary_strengths = ...
    
    # 3. 找到最大强度 ✅ 现在会执行
    result["sr_strength_max"] = ...
    return result
```

### 2. **ATR 列缺失**

边界强度计算需要 `atr` 列，但在特征计算流程中可能不存在：
- `_compute_boundary_strengths` 函数在第 784 行检查：`if "atr" not in data.columns or not boundaries: return {}`
- 如果 ATR 不存在，边界强度计算会返回空字典
- 导致 `sr_strength_max` 被设置为 0.0（而不是 NaN）

## 测试结果

修复后测试显示：
- ✅ `sr_strength_max` 现在有有效值
- ✅ 范围: [0.5000, 4.3700]
- ✅ 均值: 3.1233
- ✅ 有 1076 个有效值（全部有效）
- ✅ 边界强度计算成功（9 个边界强度序列）

## 修复内容

1. ✅ **修复了代码结构错误**：将 `compute_sr_strength_max` 函数的完整逻辑移回函数内部
2. ✅ **移除了死代码**：删除了 `compute_footprint_features` 函数后面的死代码

## 为什么训练日志显示全部为 NaN？

**已修复**：代码结构错误已修复，`sr_strength_max` 现在可以正常计算。

但训练日志显示全部为 NaN，可能的原因：
1. **ATR 列在训练流程中不存在**：需要确保 ATR 在计算 `sr_strength_max` 之前已经计算
   - 测试显示：当 ATR 不存在时，边界强度计算会返回空字典
   - 导致 `sr_strength_max` 被设置为 0.0（而不是 NaN）
2. **边界定义获取失败**：如果 `hal_high`、`hal_low`、`poc` 等列不存在或全部为 NaN，边界定义会为空
   - 测试显示：边界列有 160 个 NaN（前 160 个周期），这是正常的（需要窗口计算）
3. **特征计算顺序问题**：`sr_strength_max` 可能在其他必需特征计算之前被调用
   - 依赖关系：`sr_strength_max` 需要 `atr`、`hal_high`、`hal_low`、`poc`
   - `hal_high` 和 `hal_low` 由 `sqs_hal_high` 和 `sqs_hal_low` 计算时创建
   - 但 `poc` 可能没有被计算（如果只计算了 `hal_high` 或 `hal_low`）

## 测试结果（修复后）

修复代码结构错误后，测试显示：
- ✅ `sr_strength_max` 现在有有效值
- ✅ 范围: [0.5000, 4.3700]
- ✅ 均值: 3.1233
- ✅ 有 1076 个有效值（全部有效）
- ✅ 边界强度计算成功（9 个边界强度序列）

## 建议

1. ✅ **代码已修复**：`compute_sr_strength_max` 函数现在会正确执行
2. ⚠️ **检查特征依赖**：确保 ATR 和边界特征（hal_high, hal_low, poc）在 `sr_strength_max` 之前计算
3. ⚠️ **检查特征计算顺序**：在 `feature_dependencies.yaml` 中确认依赖关系正确

## 测试文件

- `tests/integration/test_sr_strength_max_nan.py` - 完整的诊断测试

