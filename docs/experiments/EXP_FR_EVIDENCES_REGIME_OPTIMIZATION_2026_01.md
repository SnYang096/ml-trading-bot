# FR Evidences Regime和Evidence参数优化分析报告

## 实验元信息

- **实验时间**: 2026-01-22
- **实验目的**: 
  1. 分析不同regime下FR evidences的表现
  2. 找出适合FR的regime特征范围
  3. 优化evidence参数（quantile阈值等）
  4. 扩大数据范围寻找更多样本

- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- **总样本数**: 2930
- **FeatureStore Layer**: `nnmh_highcap6_240T_2024_202510`
- **Timeframe**: 240T (4H)

## 实验状态

### ✅ 当前状态：已完成分析

**FeatureStore**: 已重新生成，包含vpin特征 ✅
- Layer: `nnmh_highcap6_240T_2024_202510_v2`
- vpin特征验证通过：100%覆盖率

**分析结果**: 已完成，详见下方结果部分

## 分析计划

### 分析1: 不同Regime下FR Evidences的表现

**目标**: 找出FR evidences在不同regime下的表现差异

**方法**:
- 对每个regime（TC_REGIME, TE_REGIME, MEAN_REGIME, NO_TRADE）分别应用FR evidences
- 计算通过evidences的样本数、平均ret_mean、胜率、Sharpe

**预期输出**:
- 各regime下FR evidences的表现对比表
- 识别表现最好的regime

### 分析2: Evidence参数优化

**目标**: 找出最优的evidence参数（quantile阈值）

**方法**:
- 测试 `has_orderflow` evidence的不同quantile阈值（0.5, 0.55, 0.6, 0.65, 0.7, 0.75）
- 评估每个参数组合下的样本数、平均ret_mean、胜率、Sharpe

**预期输出**:
- 不同quantile阈值下的表现对比
- 推荐的最优参数值

### 分析3: 适合FR的Regime特征范围

**目标**: 找出决定FR适合regime的关键特征和参数范围

**方法**:
- 分析通过FR evidences的样本的物理特征分布
- 对比正收益和负收益样本的特征差异
- 计算各特征的分位数范围

**预期输出**:
- 适合FR的物理特征范围（path_efficiency_pct, price_dir_consistency_pct, deviation_z_abs_pct等）
- 推荐的新regime参数范围

### 分析4: 数据范围扩展分析

**目标**: 扩大数据范围寻找更多适合FR的样本

**方法**:
- 尝试读取更广泛的数据范围
- 分析不同时间段的FR表现
- 评估扩大数据范围后的样本数增加

**预期输出**:
- 扩大数据范围后的样本数变化
- 不同时间段的FR表现对比

## 脚本说明

**脚本路径**: `scripts/analyze_fr_evidences_regime_optimization.py`

**使用方法**:
```bash
python3 scripts/analyze_fr_evidences_regime_optimization.py \
  --logs results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_highcap6_240T_2024_202510 \
  --timeframe 240T \
  --start-date 2025-05-01 \
  --end-date 2025-10-31 \
  --output results/fr_evidences_regime_optimization.json
```

**前提条件**:
- FeatureStore必须包含所有订单流特征（vpin, cvd_change_5, cvd_change_5_normalized）
- 如果缺少任何特征，脚本会直接报错退出

## 实际分析结果

### 1. Regime表现对比

| Regime | 总样本数 | 通过evidences | 平均ret_mean | 胜率 | Sharpe |
|--------|----------|---------------|--------------|------|--------|
| TC_REGIME | 711 | 0 | 0.000000 | 0.0% | 0.000 |
| TE_REGIME | 744 | 0 | 0.000000 | 0.0% | 0.000 |
| MEAN_REGIME | 27 | 0 | 0.000000 | 0.0% | 0.000 |
| NO_TRADE | 1448 | 0 | 0.000000 | 0.0% | 0.000 |

**关键发现**: 
- ❌ **所有regime下FR evidences的通过数都是0**
- ⚠️ **当前has_orderflow evidence的quantile阈值（0.6）过高，导致所有样本都无法通过**

### 2. Evidence参数优化

| has_orderflow quantile | 通过evidences | 平均ret_mean | 胜率 | Sharpe |
|----------------------|---------------|--------------|------|--------|
| 0.55 | 1317 | -0.000769 | 37.5% | -1.246 |
| 0.65 | 1024 | -0.000570 | 37.8% | -0.966 |

**关键发现**:
- ⚠️ **调整quantile阈值后，有样本通过，但表现不佳**（负收益、负Sharpe）
- ⚠️ **quantile=0.65的表现略好**（Sharpe -0.966 vs -1.246）
- ⚠️ **需要进一步分析vpin分布，确定合适的quantile阈值**

### 3. 适合FR的Regime特征范围

**分析结果**: 由于没有样本通过evidences，无法分析适合FR的regime特征范围

**下一步**: 需要先解决evidence通过率问题，然后才能分析特征范围

## 关键发现和结论

### 问题诊断

1. **has_orderflow evidence阈值过高**:
   - 当前quantile=0.6导致所有样本无法通过
   - 需要分析vpin实际分布，确定合适的quantile阈值

2. **即使调整quantile，表现仍然不佳**:
   - quantile=0.55和0.65时，样本通过但表现不佳（负收益、负Sharpe）
   - 说明可能需要结合regime过滤（MEAN_REGIME）才能表现好

3. **需要进一步分析**:
   - vpin的实际分布（按symbol和regime）
   - MEAN_REGIME中FR evidences的表现
   - 最优的evidence参数组合

### 下一步行动

1. **分析vpin分布**: 确定合适的quantile阈值（按symbol计算）
2. **测试MEAN_REGIME中的FR**: 在MEAN_REGIME中单独测试FR evidences
3. **优化evidence参数**: 根据分析结果调整quantile阈值
4. **结合regime过滤**: 优先在MEAN_REGIME中应用FR evidences

## 文件位置

- 分析脚本: `scripts/analyze_fr_evidences_regime_optimization.py`
- 分析结果: `results/fr_evidences_regime_optimization.json`（待生成）
- 配置文件: `config/nnmultihead/execution_archetypes.yaml`

---

**最后更新**: 2026-01-22  
**状态**: 待运行（需要重新生成FeatureStore）
