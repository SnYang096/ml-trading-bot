# 实验汇总报告

## 实验元信息

- **实验时间**: 2026-01-22 03:00:00
- **实验目的**: 汇总所有regime和gate实验的KPI对比结果
- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT
- **总样本数**: 2930
- **FeatureStore Layer**: `nnmh_highcap6_240T_2024_202510`
- **Timeframe**: 240T (4H)

---

# 实验测试汇总报告（修复后）

生成时间: 2026-01-22T01:46:38.796485

**注意**: 使用修复后的regime分类代码（基于path_efficiency和price_dir_consistency等物理特征）

## 整体KPI对比

| 配置 | 描述 | Sharpe | 交易数 | 胜率 | 平均收益 |
|------|------|--------|--------|------|----------|
| baseline | 有Regime + 有Gate Veto + 有Semantic Veto (基准) | 4.657 | 660 | 36.4% | 0.000000 |
| only_gate_rules | 有Regime + 有Gate Veto (仅Gate Rules，无Semantic Veto) | 4.400 | 715 | 36.5% | 0.000000 |
| no_regime_filter | 无Regime + 有Gate Veto + 有Semantic Veto | 0.542 | 1808 | 38.9% | 0.000000 |
| no_gate_veto | 有Regime + 无Gate Veto + 有Semantic Veto | 1.925 | 1350 | 35.7% | 0.000000 |
| no_semantic_veto | 有Regime + 有Gate Veto + 无Semantic Veto | 4.400 | 715 | 36.5% | 0.000000 |
| no_regime_no_veto | 无Regime + 无Gate Veto + 有Semantic Veto | -0.044 | 11292 | 37.9% | 0.000000 |
| all_veto_off | 无Regime + 无Gate Veto + 无Semantic Veto (全部关闭，并行开仓) | -0.045 | 11720 | 37.7% | 0.000000 |

## 按Archetype的KPI对比

### ET

| 配置 | Sharpe | 交易数 | 胜率 | 平均收益 |
|------|--------|--------|------|----------|
| no_regime_no_veto | -2.454 | 2823 | 38.5% | 0.000000 |
| all_veto_off | -2.398 | 2930 | 38.2% | 0.000000 |

### FR

| 配置 | Sharpe | 交易数 | 胜率 | 平均收益 |
|------|--------|--------|------|----------|
| baseline | 0.000 | 0 | 0.0% | 0.000000 |
| only_gate_rules | 0.000 | 0 | 0.0% | 0.000000 |
| no_regime_filter | -1.641 | 1004 | 41.3% | 0.000000 |
| no_gate_veto | 0.000 | 1 | 100.0% | 0.000000 |
| no_semantic_veto | 0.000 | 0 | 0.0% | 0.000000 |
| no_regime_no_veto | -2.454 | 2823 | 38.5% | 0.000000 |
| all_veto_off | -2.398 | 2930 | 38.2% | 0.000000 |

### TC

| 配置 | Sharpe | 交易数 | 胜率 | 平均收益 |
|------|--------|--------|------|----------|
| baseline | 4.657 | 660 | 36.4% | 0.000000 |
| only_gate_rules | 4.400 | 715 | 36.5% | 0.000000 |
| no_regime_filter | 3.749 | 804 | 35.8% | 0.000000 |
| no_gate_veto | 0.368 | 871 | 34.1% | 0.000000 |
| no_semantic_veto | 4.400 | 715 | 36.5% | 0.000000 |
| no_regime_no_veto | 2.600 | 2823 | 37.3% | 0.000000 |
| all_veto_off | 2.533 | 2930 | 37.2% | 0.000000 |

### TE

| 配置 | Sharpe | 交易数 | 胜率 | 平均收益 |
|------|--------|--------|------|----------|
| baseline | 0.000 | 0 | 0.0% | 0.000000 |
| only_gate_rules | 0.000 | 0 | 0.0% | 0.000000 |
| no_gate_veto | 4.454 | 478 | 38.5% | 0.000000 |
| no_semantic_veto | 0.000 | 0 | 0.0% | 0.000000 |
| no_regime_no_veto | 2.600 | 2823 | 37.3% | 0.000000 |
| all_veto_off | 2.533 | 2930 | 37.2% | 0.000000 |

## 关键发现

### 1. 并行开仓效果（修复后）
- 并行开仓配置（all_veto_off）:
  - Sharpe: -0.045
  - 交易数: 11720
  - 包含的Archetype: ET, FR, TC, TE

### 2. FR/ET表现（修复后）
- FR: Sharpe -2.398, 交易数 2930
- ET: Sharpe -2.398, 交易数 2930

### 3. Regime分类修复
- ✅ Regime分类已修复为完全基于价格轨迹物理特征
- ✅ 不再依赖pred_dir_prob等模型预测
- ✅ 使用path_efficiency和price_dir_consistency等物理特征

