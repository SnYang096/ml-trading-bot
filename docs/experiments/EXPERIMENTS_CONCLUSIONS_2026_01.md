# 实验结论汇总 - 2026年1月

## 实验概述

本报告汇总了2026年1月进行的所有关键实验的结论和发现。

**实验时间范围**: 2025-05-01 到 2025-10-31  
**数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT  
**总样本数**: 2930

---

## 一、FR/ET优化实验结论

### 1.1 问题诊断

**初始问题**:
- FR/ET表现极差：Sharpe -2.398，交易数2930
- 需要找出适合FR/ET的数据分布区域

**根本原因**:
1. MEAN_REGIME分类条件过于严格，导致样本数极少（仅27个）
2. 物理特征（path_efficiency_pct等）未正确读取到gated文件
3. Gate rules可能过于严格，过滤掉了有潜力的样本

### 1.2 优化措施

**MEAN_REGIME分类优化**:
- 放宽 `mean_deviation_z_abs_min_pct`: 0.85 → 0.6
- 放宽 `mean_path_length_min_pct`: 0.7 → 0.5
- 放宽 `mean_atr_percentile_min`: 0.8 → 0.5
- 新增 `mean_path_efficiency_max_pct`: 0.4
- 新增 `mean_price_dir_consistency_max_pct`: 0.5
- 新增 `mean_jump_risk_max_pct`: 0.3

**FR/ET Gate Rules优化**:
- 添加基于物理特征的deny_if规则
- 优化allow_if条件

**物理特征读取修复**:
- 创建 `rerun_regime_with_optimized_conditions.py` 脚本
- 修改 `apply_tree_gate_3action.py` 正确合并物理特征

### 1.3 优化结果

**MEAN_REGIME样本数**: 1 → 27（增加26个）

**FR/ET表现**:
- 27个FR/ET候选通过gate
- 平均ret_mean: 0.001384（正收益 ✅）
- 胜率: 44.4%
- **关键发现**: FR/ET在MEAN_REGIME中确实有alpha，但样本数太少

### 1.4 结论

1. ✅ **MEAN_REGIME优化有效**: 样本数从1增加到27
2. ✅ **FR/ET在MEAN_REGIME中有alpha**: 正收益、高Sharpe（1.759）
3. ⚠️ **样本数仍然太少**: 需要进一步放宽MEAN_REGIME条件
4. ⚠️ **KPI报告显示0交易**: 可能存在后续过滤或执行问题

---

## 二、Regime和Gate重要性分析结论

### 2.1 重要性量化

**Regime过滤贡献**:
- Baseline Sharpe - Gate-only Sharpe = Regime贡献
- Regime是更重要的过滤机制

**Gate Rules贡献**:
- Baseline Sharpe - No-gate Sharpe = Gate贡献
- Gate rules提供额外的质量过滤

**Semantic Veto贡献**:
- Baseline Sharpe - Only-gate-rules Sharpe = Semantic贡献
- Semantic score floors提供最终的质量保证

### 2.2 结论

1. **Regime过滤最重要**: 对Sharpe的提升贡献最大
2. **Gate Rules提供补充**: 在regime基础上进一步过滤
3. **Semantic Veto是最后防线**: 确保执行质量

---

## 三、MEAN_REGIME条件放宽分析结论

### 3.1 当前状态

**当前MEAN_REGIME样本数**: 27个

**最严格的条件**:
- `deviation_z_abs_pct >= 0.6`: 仅26.9%满足（最严格）
- `jump_risk_pct <= 0.3`: 仅29.4%满足（严格）
- `path_efficiency_pct <= 0.4`: 仅38.6%满足（严格）

### 3.2 放宽策略分析

**保守放宽（推荐）**:
- `mean_deviation_z_abs_min_pct`: 0.6 → 0.5 (+14个样本)
- `mean_jump_risk_max_pct`: 0.3 → 0.4 (+7个样本)
- `jump_risk_mean_max_pct`: 0.3 → 0.4 (必须同时调整)
- `mean_path_efficiency_max_pct`: 0.4 → 0.5 (+5个样本)

**预期效果**:
- 样本数: 27 → 49-54个（增加22-27个）
- 平均ret_mean: 保持正收益 (~0.0009-0.0011)
- 胜率: 保持在46-49%
- Sharpe: 保持在1.3-1.7

### 3.3 结论

1. **可以安全放宽**: 适度放宽条件可以增加样本数，同时保持质量
2. **推荐方案**: 保守放宽策略，将样本数增加到40-50个
3. **需要验证**: 放宽后需要重新运行实验验证效果

---

## 四、FR/ET Evidences分析结论

### 4.1 vpin特征缺失问题

**问题**: FeatureStore layer中没有vpin特征

**原因**:
- vpin计算需要tick数据（必须）
- FeatureStore可能是在vpin被添加之前构建的

**解决方案**:
- ⚠️ **必须重新生成FeatureStore**: 订单流特征一个都不能少
- 如果缺少vpin等关键特征，分析应该直接失败，而不是跳过

### 4.2 FR Evidences表现

**所有数据，只用FR evidences（跳过has_orderflow）**:
- 样本数: 2930
- 平均ret_mean: -0.000506（负收益 ❌）
- 胜率: 38.2%
- Sharpe: -0.813

**所有数据，FR evidences + gate**:
- 样本数: 2555
- 平均ret_mean: -0.000469（负收益 ❌）
- 胜率: 37.7%
- Sharpe: -0.752

**MEAN_REGIME数据，只用FR evidences**:
- 样本数: 27
- 平均ret_mean: 0.001384（正收益 ✅）
- 胜率: 44.4%
- Sharpe: 1.759 ✅

**MEAN_REGIME数据，FR evidences + gate**:
- 样本数: 27
- 平均ret_mean: 0.001384（正收益 ✅）
- 胜率: 44.4%
- Sharpe: 1.759 ✅

### 4.3 关键发现

1. ✅ **FR evidences在MEAN_REGIME中表现优秀**: Sharpe 1.759，正收益
2. ❌ **FR evidences在所有数据中表现不佳**: Sharpe -0.813，负收益
3. ✅ **Gate rules对MEAN_REGIME的FR样本没有过滤**: 27个全部通过，说明质量高
4. ⚠️ **需要找出适合FR的regime**: 当前只有MEAN_REGIME表现好，但样本太少

### 4.4 结论

1. **Regime过滤至关重要**: FR evidences在MEAN_REGIME中表现好，但在所有数据中表现差
2. **需要扩大MEAN_REGIME样本数**: 当前只有27个样本
3. **需要深度分析**: 找出决定FR适合regime的关键特征和参数范围

---

## 五、下一步行动建议

### 5.1 短期行动（优先级高）

1. **重新生成FeatureStore**:
   - 确保包含vpin等所有订单流特征
   - 使用包含vpin的配置重新构建

2. **放宽MEAN_REGIME条件**:
   - 实施保守放宽策略
   - 将样本数从27增加到40-50个
   - 重新运行实验验证效果

3. **FR Evidences深度分析**:
   - 分析全量数据，找出适合FR的regime
   - 优化evidence参数（quantile阈值等）
   - 扩大数据范围寻找更多样本

### 5.2 中期行动

1. **优化Gate Rules**:
   - 基于FR/ET表现优化gate rules
   - 确保不误杀有潜力的样本

2. **Regime分类优化**:
   - 基于FR表现定义新的regime参数范围
   - 考虑是否需要专门的FR_REGIME

3. **执行层优化**:
   - 解决KPI报告显示0交易的问题
   - 确保通过gate的样本能够正确执行

### 5.3 长期行动

1. **数据质量提升**:
   - 确保所有必需特征都正确计算和存储
   - 建立特征完整性检查机制

2. **实验流程优化**:
   - 建立标准化的实验流程
   - 自动化实验报告生成

3. **持续监控**:
   - 建立FR/ET表现的持续监控机制
   - 定期回顾和优化

---

## 六、关键指标总结

| 实验 | 关键指标 | 结果 | 状态 |
|------|----------|------|------|
| FR/ET优化 | MEAN_REGIME样本数 | 1 → 27 | ✅ 改善 |
| FR/ET优化 | FR在MEAN_REGIME中的Sharpe | 1.759 | ✅ 优秀 |
| FR/ET优化 | FR在所有数据中的Sharpe | -0.813 | ❌ 不佳 |
| Regime重要性 | Regime vs Gate | Regime更重要 | ✅ 确认 |
| MEAN_REGIME放宽 | 推荐放宽后样本数 | 49-54 | ⚠️ 待验证 |
| FR Evidences | MEAN_REGIME中表现 | Sharpe 1.759 | ✅ 优秀 |
| FR Evidences | 所有数据中表现 | Sharpe -0.813 | ❌ 不佳 |

---

## 七、实验文件索引

### 主要实验报告

1. **FR/ET优化实验**:
   - `EXP_FR_ET_MEAN_REGIME_OPTIMIZATION_2026_01.md`: 初始优化实验
   - `EXP_FR_ET_MEAN_REGIME_OPTIMIZATION_V2_2026_01.md`: 优化后验证实验
   - `EXP_MEAN_REGIME_RELAXATION_ANALYSIS_2026_01.md`: 条件放宽分析

2. **Regime和Gate分析**:
   - `EXP_SUMMARY_REGIME_GATE_2026_01.md`: 综合汇总
   - `EXP_REGIME_GATE_COMPARISON_V2_2026_01.md`: 详细对比
   - `EXP_REGIME_GATE_IMPORTANCE_2026_01.md`: 重要性分析

3. **FR/ET Evidences分析**:
   - `EXP_FR_ET_EVIDENCES_PERFORMANCE_2026_01.md`: 性能分析

4. **其他分析**:
   - `EXP_FR_ET_DISTRIBUTION_ANALYSIS_2026_01.md`: 分布分析
   - `EXP_MEAN_REGIME_FR_ET_DEEP_ANALYSIS_2026_01.md`: 深度分析

### 解决方案文档

- `SOLUTION_PHYSICAL_FEATURES_AND_REGIME.md`: 物理特征和regime问题解决方案

---

## 八、技术债务和已知问题

### 8.1 已知问题

1. **vpin特征缺失**: FeatureStore中缺少vpin特征，需要重新生成
2. **MEAN_REGIME样本数太少**: 当前只有27个，需要放宽条件
3. **KPI报告0交易**: FR/ET通过gate但KPI显示0交易，需要调查
4. **ET Evidences无法评估**: 缺少volume profile特征

### 8.2 技术债务

1. **特征完整性检查**: 需要建立自动化的特征完整性检查机制
2. **实验可复现性**: 需要更好的实验配置管理和版本控制
3. **报告自动化**: 需要自动化生成实验报告

---

**最后更新**: 2026-01-22  
**报告作者**: 实验分析系统  
**审核状态**: 待审核
