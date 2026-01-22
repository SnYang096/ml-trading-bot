# 实验状态总结 - 2026年1月

## 已完成的工作

### 1. 实验结论保存 ✅
- ✅ 创建了综合实验结论文档: `EXPERIMENTS_CONCLUSIONS_2026_01.md`
- ✅ 更新了所有现有实验报告，添加了结论部分
- ✅ 所有关键发现和结论都已文档化

### 2. 订单流特征严格验证 ✅
- ✅ 修改了所有分析脚本，严格要求订单流特征完整
- ✅ 移除了 `--vpin-missing-strategy` 参数
- ✅ 如果缺少vpin等关键特征，脚本会直接报错退出

### 3. FR Evidences深度分析准备 ✅
- ✅ 创建了FR Evidences深度分析脚本: `analyze_fr_evidences_regime_optimization.py`
- ✅ 创建了分析报告模板: `EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_2026_01.md`
- ✅ 创建了下一步行动计划: `NEXT_STEPS_VPIN_FEATURE_2026_01.md`

## 当前阻塞问题

### vpin特征缺失

**问题**: FeatureStore中缺少vpin特征，导致所有依赖vpin的分析无法运行

**影响**:
1. FR Evidences深度分析无法运行
2. FR/ET Evidences性能分析无法运行
3. 所有依赖`has_orderflow` evidence的分析无法运行

**检查结果**:
- ❌ 现有FeatureStore layers (`nnmh_highcap6_240T_2024_202510`, `nnmh_highcap6_240T_2024_202510_ma_adx_cvd_vwap_v1`) 都缺少vpin
- ✅ 配置文件中已包含vpin (`config/nnmultihead/live_feature_plan.yaml`)
- ✅ 配置正确，问题在于FeatureStore构建时未包含vpin

**解决方案**: 必须重新生成FeatureStore，使用包含vpin的配置

## 下一步行动

### 优先级1: 重新生成FeatureStore

**目标**: 生成包含所有订单流特征（vpin, cvd_change_5, cvd_change_5_normalized）的FeatureStore

**步骤**:
1. 确认tick数据可用（vpin计算需要tick数据）
2. 使用 `config/nnmultihead/live_feature_plan.yaml` 重新构建FeatureStore
3. 验证vpin特征已正确生成
4. 更新分析脚本使用新的FeatureStore layer

**详细步骤**: 参见 `NEXT_STEPS_VPIN_FEATURE_2026_01.md`

### 优先级2: 运行FR Evidences深度分析

**目标**: 找出适合FR的regime和evidence参数范围

**前提**: FeatureStore包含vpin特征

**分析内容**:
1. 不同regime下FR evidences的表现
2. Evidence参数优化（quantile阈值）
3. 适合FR的regime特征范围
4. 数据范围扩展分析

### 优先级3: 根据分析结果优化

**目标**: 基于FR Evidences深度分析结果进行优化

**可能的方向**:
1. 调整regime分类参数
2. 优化evidence参数
3. 定义新的regime范围
4. 放宽MEAN_REGIME条件（参考 `EXP_MEAN_REGIME_RELAXATION_ANALYSIS_2026_01.md`）

## 关键发现总结

### FR/ET优化
- ✅ MEAN_REGIME样本数从1增加到27
- ✅ FR/ET在MEAN_REGIME中有alpha（Sharpe 1.759，正收益）
- ⚠️ 样本数仍然太少，需要进一步放宽条件

### Regime和Gate重要性
- ✅ Regime过滤最重要（对Sharpe提升贡献最大）
- ✅ Gate Rules提供补充过滤
- ✅ Semantic Veto是最后防线

### MEAN_REGIME放宽分析
- ✅ 可以安全放宽条件，将样本数从27增加到49-54个
- ✅ 推荐保守放宽策略，保持质量的同时增加样本数

### FR Evidences表现
- ✅ FR evidences在MEAN_REGIME中表现优秀（Sharpe 1.759）
- ❌ FR evidences在所有数据中表现不佳（Sharpe -0.813）
- ⚠️ 需要找出适合FR的regime（当前只有MEAN_REGIME表现好）

## 文件索引

### 实验报告
- `EXPERIMENTS_CONCLUSIONS_2026_01.md`: 综合实验结论
- `EXP_FR_ET_MEAN_REGIME_OPTIMIZATION_V2_2026_01.md`: FR/ET优化实验
- `EXP_MEAN_REGIME_RELAXATION_ANALYSIS_2026_01.md`: MEAN_REGIME放宽分析
- `EXP_FR_ET_EVIDENCES_PERFORMANCE_2026_01.md`: FR/ET Evidences性能分析
- `EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_2026_01.md`: FR Evidences深度分析（待运行）

### 行动计划
- `NEXT_STEPS_VPIN_FEATURE_2026_01.md`: vpin特征修复计划
- `STATUS_SUMMARY_2026_01.md`: 本文件（状态总结）

### 分析脚本
- `scripts/analyze_fr_evidences_regime_optimization.py`: FR Evidences深度分析
- `scripts/analyze_fr_et_evidences_performance.py`: FR/ET Evidences性能分析（已更新，严格要求特征完整）

---

**最后更新**: 2026-01-22  
**当前状态**: 等待重新生成FeatureStore（包含vpin特征）
