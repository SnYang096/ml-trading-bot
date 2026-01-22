# ET策略2024年数据测试状态报告

**检查时间**: 2026-01-22  
**目的**: 使用2024年完整tick数据和包含订单流特征的FeatureStore测试ET策略

---

## 执行摘要

### 当前状态

1. **数据完整性**: ✅ BTCUSDT 2024年数据完整（12个月份）
2. **FeatureStore**: ⏳ v3 layer正在构建中（包含完整订单流特征）
3. **Logs文件**: ⚠️ 没有2024年logs文件，但有2025年logs文件可用
4. **Gate检查**: ⚠️ 已运行，但发现特征合并问题

---

## 详细状态

### 任务1: FeatureStore可用性检查 ✅

**检查结果**:
- **v3 layer**: 不存在（正在构建中）
  - FeatureStore构建进程正在运行
  - 预期包含: vpin + volume_profile
  
- **v2 layer**: 存在
  - ✅ 包含vpin特征（30个相关列）
  - ❌ 缺少volume_profile特征
  - 数据可用: 2025-05-01 到 2025-10-31

**结论**: v2 layer可以用于验证vpin特征，但缺少volume_profile，无法完整测试ET策略。

### 任务2: 准备2024年logs文件 ✅

**检查结果**:
- ❌ 没有找到包含BTCUSDT 2024年数据的logs文件
- ✅ 找到2025年logs文件: `logs_3action_with_et_regime_v3.parquet`
  - 时间范围: 2025-05-01 到 2025-10-31
  - 总行数: 2930
  - BTCUSDT行数: 1099
  - BTCUSDT ET_REGIME样本: 2

**结论**: 可以使用2025年数据先验证流程，但最终需要使用2024年数据完整测试。

### 任务3-4: Gate检查 ⚠️

**执行情况**:
- ✅ 已运行gate检查命令
- ⚠️ 发现gated文件中缺少vpin特征

**问题诊断**:
1. **手动测试确认**: vpin特征可以正确从FeatureStore读取和合并
   - FeatureStore v2 layer包含30个vpin相关列
   - 手动merge测试成功: 1094/1099行成功合并vpin特征
   
2. **脚本问题**: `apply_archetype_gate.py`可能没有正确处理timestamp在index的情况
   - FeatureStore返回的DataFrame的timestamp在index中，不在列中
   - 脚本的`_read_feature_store_range`函数应该处理这种情况，但可能在某些情况下失败

**Gate检查结果**（使用v2 layer + 2025年数据）:
- ET_REGIME样本总数: 2
- 通过gate的ET样本: 2
- 平均ret_mean: -0.038210
- 胜率: 0.0%
- Sharpe: -11.225
- ⚠️ gated文件中没有vpin和volume_profile特征

---

## 问题分析

### 特征合并问题

**现象**: Gate检查后，gated文件中缺少vpin特征，尽管FeatureStore中包含这些特征。

**可能原因**:
1. `apply_archetype_gate.py`的`_read_feature_store_range`函数在某些情况下没有正确处理timestamp在index的情况
2. 时间戳格式不匹配导致merge失败
3. 特征合并逻辑有bug

**验证**: 手动测试确认vpin特征可以正确合并，说明问题在脚本的特定执行路径。

### 数据可用性

**2024年数据**:
- ✅ BTCUSDT: 完整（12个月份）
- ❌ 其他symbols: 不完整（但zip文件存在，可转换）

**2025年数据**:
- ✅ 有logs文件可用
- ✅ FeatureStore v2 layer有数据

---

## 下一步行动

### 选项1: 等待v3 layer完成（推荐）

**优点**:
- 包含完整订单流特征（vpin + volume_profile）
- 可以完整测试ET策略的所有gate rules和evidence rules

**步骤**:
1. 等待v3 layer构建完成
2. 检查v3 layer是否包含完整特征
3. 使用2024年数据生成logs文件（如果需要）
4. 重新运行regime分类和gate检查
5. 分析ET表现

### 选项2: 修复特征合并问题并使用v2 layer

**优点**:
- 可以立即验证vpin特征是否工作
- 不需要等待v3 layer完成

**缺点**:
- 缺少volume_profile特征，无法完整测试ET策略
- 需要修复`apply_archetype_gate.py`脚本

**步骤**:
1. 修复`apply_archetype_gate.py`的timestamp处理问题
2. 使用v2 layer和2025年数据验证vpin特征
3. 等v3完成后，再用完整特征测试

### 选项3: 生成2024年logs文件

**如果v3 layer完成后仍没有2024年logs文件**:
1. 需要模型文件（model.pt）
2. 运行predict生成preds（2024年数据）
3. 运行build-logs-3action生成logs

---

## 关键发现

1. **FeatureStore v2 layer有vpin特征**: 30个相关列，数据完整
2. **特征可以手动合并**: 手动测试确认vpin特征可以正确合并到logs文件
3. **脚本可能有bug**: `apply_archetype_gate.py`在实际执行时可能没有正确处理特征合并
4. **v3 layer正在构建**: 包含完整订单流特征，完成后可以完整测试ET策略

---

## 建议

**优先方案**: 等待v3 layer完成，然后使用2024年数据完整测试ET策略。

**临时方案**: 如果急需验证vpin特征，可以：
1. 修复`apply_archetype_gate.py`的timestamp处理问题
2. 使用v2 layer和2025年数据验证vpin特征是否工作
3. 但注意：缺少volume_profile，无法完整测试ET策略

---

## 相关文件

- `data/parquet_data/BTCUSDT_2024-*.parquet` - BTCUSDT 2024年tick数据（12个文件）
- `results/e2e_kpi/logs_3action_with_et_regime_v3.parquet` - 2025年logs文件
- `scripts/apply_archetype_gate.py` - Gate检查脚本（可能需要修复timestamp处理）
- `docs/experiments/EXP_ET_2024_DATA_AVAILABILITY_2026_01.md` - 数据完整性报告
