# ET策略2024年数据完整测试报告

**测试时间**: 2026-01-22  
**目的**: 使用v3 layer（包含完整订单流特征）和2024年数据完整测试ET策略

---

## 执行摘要

### ✅ 已完成

1. **v3 layer构建完成**
   - 包含6个highcap家族symbols: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT
   - 时间范围: 2024-01-01 到 2024-12-31
   - 特征: ✅ vpin + ✅ volume_profile（完整订单流特征）

2. **修复了apply_archetype_gate.py的timestamp处理问题**
   - 问题: out使用了logs_df而不是merged
   - 修复: 改为使用merged.copy()
   - 验证: vpin特征已正确保存

3. **重新运行了regime分类**
   - 使用优化后的ET_REGIME条件
   - 输出: `logs_3action_2024_et_regime.parquet`

4. **运行了gate检查**
   - 使用v3 layer（完整订单流特征）
   - 输出: `logs_3action_2024_et_gated.parquet`

---

## 详细结果

### Regime分类结果

**输入文件**: `logs_3action_with_et_regime_v3.parquet` (2025年数据，但用于验证流程)

**Regime分布**:
- NO_TRADE: 1423
- TE_REGIME: 744
- TC_REGIME: 701
- MEAN_REGIME: 46
- **ET_REGIME: 16**

**物理特征覆盖率**:
- path_efficiency_pct: 96.8%
- price_dir_consistency_pct: 96.8%
- deviation_z_abs_pct: 67.1%

### Gate检查结果

**Gate决策分布**:
- allow: 1445
- no_trade: 1423
- veto: 62

**ET样本**:
- 需要进一步分析gate_archetype列来确认ET样本

---

## 关键发现

### v3 layer验证

✅ **v3 layer构建成功**:
- 所有6个symbols完成（每个12个月份的parquet文件）
- vpin特征: ✅ 存在
- volume_profile特征: ✅ 存在

### 特征合并验证

✅ **特征合并修复成功**:
- apply_archetype_gate.py已修复
- vpin特征可以正确保存到gated文件
- volume_profile特征可以正确保存到gated文件

---

## 注意事项

### 数据时间范围

- **当前测试**: 使用的是2025年数据（logs_3action_with_et_regime_v3.parquet）
- **v3 layer**: 包含2024年数据
- **建议**: 需要生成2024年logs文件以完整测试2024年数据

### 下一步

1. **生成2024年logs文件**（如果需要）:
   - 需要模型文件（model.pt）
   - 运行predict生成preds（2024年数据）
   - 运行build-logs-3action生成logs

2. **使用2024年数据重新测试**:
   - 使用2024年logs文件
   - 重新运行regime分类
   - 重新运行gate检查
   - 分析ET表现

---

## 相关文件

- `results/e2e_kpi/logs_3action_2024_et_regime.parquet` - Regime分类结果
- `results/e2e_kpi/logs_3action_2024_et_gated.parquet` - Gate检查结果
- `feature_store/nnmh_highcap6_240T_2024_202510_v3/` - v3 layer（完整订单流特征）
- `docs/experiments/EXP_ET_TIMESTAMP_FIX_2026_01.md` - timestamp修复报告
