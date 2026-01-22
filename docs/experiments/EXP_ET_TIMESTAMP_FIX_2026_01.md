# ET策略测试：timestamp处理问题修复报告

**修复时间**: 2026-01-22  
**目的**: 修复`apply_archetype_gate.py`中导致FeatureStore特征无法正确合并到gated文件的问题

---

## 问题描述

### 现象

运行`apply_archetype_gate.py`后，生成的gated文件中缺少从FeatureStore读取的特征（如vpin、volume_profile等），尽管：
1. FeatureStore中确实包含这些特征
2. 手动测试确认特征可以正确读取和合并

### 根本原因

在`apply_archetype_gate.py`的第660行，代码使用了：
```python
out = logs_df.copy()
```

而不是使用已经合并了FeatureStore特征的`merged` DataFrame。这导致所有从FeatureStore合并的特征在最终输出时丢失。

---

## 修复方案

### 修复内容

**文件**: `scripts/apply_archetype_gate.py`

**修改位置**: 第660行

**修改前**:
```python
else:
    # Original logic: single archetype per row
    out = logs_df.copy()
    out["gate_ok"] = gate_ok
    out["gate_decision"] = gate_decision
    out["gate_reasons"] = gate_reasons
    out["gate_archetype"] = gate_arch
```

**修改后**:
```python
else:
    # Original logic: single archetype per row
    # Use merged DataFrame (which includes features from FeatureStore) instead of logs_df
    out = merged.copy()
    out["gate_ok"] = gate_ok
    out["gate_decision"] = gate_decision
    out["gate_reasons"] = gate_reasons
    out["gate_archetype"] = gate_arch
```

### 额外改进

同时改进了timestamp处理逻辑（第318-332行），确保正确处理timestamp在index中的情况。

---

## 验证结果

### 测试配置

- **输入文件**: `results/e2e_kpi/logs_3action_with_et_regime_v3.parquet`
- **FeatureStore layer**: `nnmh_highcap6_240T_2024_202510_v2`
- **输出文件**: `results/e2e_kpi/logs_3action_2025_v2_gated_final.parquet`

### 验证结果

✅ **修复成功**:
- VPIN相关列: 30个
- vpin特征覆盖率: 99.8% (2925/2930)
- ET样本的vpin特征: 2/2 非空

### 特征列表

修复后，gated文件包含以下vpin相关特征：
- `vpin`
- `vpin_signed_imbalance`
- `vpin_last`
- `vpin_max`
- `vpin_min`
- `vpin_std`
- `vpin_count`
- `vpin_skewness`
- `vpin_trend`
- 以及其他20个vpin相关特征

---

## 影响

### 正面影响

1. ✅ FeatureStore特征现在可以正确合并到gated文件
2. ✅ Gate rules和evidence rules现在可以正确使用vpin、volume_profile等特征
3. ✅ ET策略测试可以正常进行

### 注意事项

- 修复后，gated文件会包含更多列（从FeatureStore合并的特征）
- 文件大小可能会增加
- 需要确保FeatureStore layer包含所需的特征

---

## 下一步

1. ✅ 修复已完成并验证
2. ⏳ 等待v3 layer完成（包含完整订单流特征：vpin + volume_profile）
3. ⏳ 使用2024年数据完整测试ET策略

---

## 相关文件

- `scripts/apply_archetype_gate.py` - 已修复的脚本
- `results/e2e_kpi/logs_3action_2025_v2_gated_final.parquet` - 修复后的测试输出
- `docs/experiments/EXP_ET_2024_TESTING_STATUS_2026_01.md` - 测试状态报告
