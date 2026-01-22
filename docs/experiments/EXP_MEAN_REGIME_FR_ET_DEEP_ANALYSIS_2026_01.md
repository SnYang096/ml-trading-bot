# MEAN_REGIME和FR/ET深度分析报告

## 实验元信息

- **实验时间**: 2026-01-22 03:00:20
- **实验目的**: 深入分析MEAN_REGIME分类问题和FR/ET被拒绝的根本原因
- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- **总样本数**: 2930

## 1. MEAN_REGIME分类问题分析

### Regime分布

| Regime | 样本数 |
|--------|--------|
| NO_TRADE | 1473 |
| TE_REGIME | 812 |
| TC_REGIME | 644 |
| MEAN_REGIME | 1 |

**需要重新运行regime分类**: True
**有物理特征**: False

## 2. 物理特征读取问题分析

| 特征 | 存在 | 覆盖率 |
|------|------|--------|
| path_efficiency_pct | ❌ | N/A |
| price_dir_consistency_pct | ❌ | N/A |
| deviation_z_abs_pct | ❌ | N/A |
| path_efficiency | ❌ | N/A |
| price_dir_consistency | ❌ | N/A |
| deviation_z_abs | ❌ | N/A |
| path_length_pct | ❌ | N/A |
| jump_risk_pct | ❌ | N/A |
| atr_percentile | ❌ | N/A |

**问题**: 物理特征可能没有从FeatureStore正确读取到gated文件

## 3. FR/ET被拒绝原因分析

**FR/ET候选数**: 1
**被拒绝数**: 1

## 4. MEAN_REGIME Alpha分析

**MEAN_REGIME样本数**: 1

### MEAN_REGIME收益统计

- 平均ret_mean: 0.003686
- 中位数ret_mean: 0.003686
- 胜率: 100.0%
