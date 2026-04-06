# 基线测试工作流程

## 概述

基线测试用于建立各archetype的性能基准，作为后续优化的参考点。

## 数据范围

- **时间范围**: 2024-01-01 ~ 2025-12-31
- **FeatureStore layer**: `nnmh_highcap6_240T_2024_202510_v2` 或 `nnmh_highcap6_240T_2024_202510_ma_adx_cvd_vwap_v1`
- **前置条件**: 需要先通过`mlbot nnmultihead build-execution-logs`生成logs文件

## 工作流程

### 方法1: 使用mlbot命令（推荐）

```bash
# Step 1: 运行gate检查
mlbot gate apply-archetype \
  --logs results/e2e_kpi/logs_3action_2024_2025.parquet \
  --out results/baseline_smoke_test/logs_baseline.parquet \
  --features-store-layer nnmh_highcap6_240T_2024_202510_v2 \
  --features-store-root feature_store

# Step 2: 生成KPI报告
mlbot diagnose e2e-kpi \
  --logs results/baseline_smoke_test/logs_baseline.parquet \
  --output-md results/baseline_smoke_test/baseline_kpi.md \
  --output-json results/baseline_smoke_test/baseline_kpi.json

# Step 3: 生成archetype详细报告
mlbot analyze archetype-performance \
  --logs results/baseline_smoke_test/logs_baseline.parquet \
  --output results/baseline_smoke_test/archetype_performance.md
```

### 方法2: 使用一键脚本

```bash
python scripts/run_baseline_smoke_test.py \
  --logs results/e2e_kpi/logs_3action_2024_2025.parquet \
  --output-dir results/baseline_smoke_test \
  --features-store-layer nnmh_highcap6_240T_2024_202510_v2 \
  --features-store-root feature_store
```

## 输出文件

- `results/baseline_smoke_test/logs_baseline.parquet` - 应用gate后的logs文件
- `results/baseline_smoke_test/baseline_kpi.md` - KPI报告（Markdown）
- `results/baseline_smoke_test/baseline_kpi.json` - KPI报告（JSON）
- `results/baseline_smoke_test/archetype_performance.md` - Archetype性能详细报告

## 报告内容

### 整体KPI
- Sharpe比率
- 交易数
- 胜率
- 平均收益

### 按Archetype
- TC (TrendContinuation)
- TE (TrendExpansion)
- FR (FailureReversion)
- ET (ExhaustionTurn)

每个archetype的详细指标：
- 交易数
- Sharpe比率
- 胜率
- 平均收益
- 总收益

### 多Archetype统计
- 同时触发的archetype组合
- 各组合的出现频率

### CVD判断效果
- 正CVD vs 负CVD的Sharpe差异
- CVD percentile规则的影响

## 使用建议

1. **定期运行**: 在每次架构变更后重新运行基线测试
2. **版本控制**: 将基线报告保存到版本控制系统，便于对比
3. **对比分析**: 使用基线报告与优化后的结果进行对比
