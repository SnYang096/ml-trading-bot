# 平坦高原优化工作流程

## 概述

使用平坦高原方法优化gate规则的阈值，找到在多个分桶维度下都表现稳定的阈值区间。

## 优化目标

- **目标函数**: Robustness Score = min Sharpe across buckets
- **分桶维度**: Archetype (TC/TE/FR/ET) × Vol (low/mid/high)
- **约束条件**: 
  - `trade_rate(θ) ≥ R_min` (默认0.5%)
  - `coverage_per_bucket ≥ N_min` (默认10 trades)

## 优化顺序

按重要性顺序优化，每一类冻结后再动下一类：

1. **结构存在类** (path_efficiency, consistency)
2. **稳定性 veto** (jump_risk)
3. **极端 veto** (deviation_z)
4. **CVD regime判断** (cvd percentile)

## 工作流程

### 优化单个规则

```bash
mlbot optimize gate-plateau \
  --archetype TrendContinuationTC \
  --rule-name tc_not_tc_regime_path_efficiency_too_low \
  --gated-logs results/baseline_smoke_test/logs_baseline.parquet \
  --raw-logs results/e2e_kpi/logs_3action_2024_2025.parquet \
  --output results/optimization/tc_path_efficiency_optimization.json \
  --min-trade-rate 0.005 \
  --min-trades-per-bucket 10 \
  --min-sharpe-threshold 0.5 \
  --threshold-step 0.05
```

### 批量优化所有规则

```bash
mlbot optimize gate-plateau-all \
  --gated-logs results/baseline_smoke_test/logs_baseline.parquet \
  --raw-logs results/e2e_kpi/logs_3action_2024_2025.parquet \
  --output-dir results/optimization/all_archetypes \
  --min-trade-rate 0.005 \
  --min-trades-per-bucket 10
```

## 输出格式

优化结果JSON格式：

```json
{
  "archetype": "TrendContinuationTC",
  "rule_name": "tc_not_tc_regime_path_efficiency_too_low",
  "current_threshold": 0.6,
  "plateau_start": 0.55,
  "plateau_end": 0.65,
  "recommended_threshold": 0.6,
  "robustness_score": 0.8,
  "trade_rate": 0.45,
  "min_coverage": 15,
  "worst_bucket": "TC_low",
  "bucket_sharpes": {
    "TC": {
      "TC_low": 0.8,
      "TC_mid": 1.2,
      "TC_high": 1.5
    }
  }
}
```

## 参数说明

- `--min-trade-rate`: 最小交易率阈值（默认0.5%）
- `--min-trades-per-bucket`: 每个分桶的最小交易数（默认10）
- `--min-sharpe-threshold`: 平台高原的最低Sharpe要求（默认0.5）
- `--threshold-step`: 阈值扫描步长（默认0.05）

## 使用建议

1. **逐步优化**: 按优化顺序，一类一类地优化
2. **验证结果**: 优化后重新运行基线测试，验证改进
3. **记录变更**: 将优化结果记录到配置文件中
