# HighCap6压缩优化指南

## 概述

本指南说明如何使用压缩模式优化Gate规则，从全松阈值开始逐步收紧，压缩过度交易。

## 核心概念

### 压缩模式（Compression Mode）

压缩模式的目标是：
- **压缩过度交易**：从全松阈值（~100% trade_rate）开始，逐步收紧
- **保持规则有效性**：在压缩过程中，保持robustness_score不显著下降
- **找到最优压缩点**：使用compression_efficiency指标选择最优阈值

### 压缩效率指标

```
compression_efficiency = robustness_score / trade_rate
```

含义：每单位trade_rate获得的robustness_score。越高越好，表示用更少的交易获得更高的稳健性。

## 使用方法

### 1. 重新走完整Pipeline流程

确保包含所有6个token（BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT）的完整数据：

```bash
# 方式1: 使用自动化脚本
bash scripts/run_highcap6_full_pipeline.sh

# 方式2: 手动执行（参考 docs/workflow/PIPELINE_WORKFLOW.md）
```

**输出**: `results/pipeline_highcap6_2024_full_*/logs_execution.parquet`

### 2. 运行TC压缩优化

```bash
python scripts/run_tc_compression_optimization.py \
    --raw-logs results/pipeline_highcap6_2024_full_*/logs_execution.parquet \
    --output-dir results/tc_compression_optimization \
    --compression-target-trade-rate 0.02 \
    --compression-min-robustness 0.5 \
    --compression-step 0.01 \
    --global-trade-budget 0.02 \
    --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
    --feature-store-root feature_store \
    --timeframe 240T \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

**参数说明**:
- `--compression-target-trade-rate`: 目标trade_rate（例如0.02表示压缩到2%）
- `--compression-min-robustness`: 压缩过程中最低robustness_score要求
- `--compression-step`: 收紧步长
- `--global-trade-budget`: 全局trade_rate生存约束

**输出**: `results/tc_compression_optimization/tc_optimization_compression.json`

### 3. 直接使用Hard-Gate脚本（高级用法）

```bash
python scripts/optimize_gate_plateau_hard_gate.py \
    --gated-logs results/pipeline_highcap6_2024_full_*/logs_execution_gated.parquet \
    --raw-logs results/pipeline_highcap6_2024_full_*/logs_execution.parquet \
    --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
    --output results/tc_optimization_compression.json \
    --compression-mode \
    --compression-target-trade-rate 0.02 \
    --compression-min-robustness 0.5 \
    --compression-step 0.01 \
    --global-trade-budget 0.02 \
    --archetype-filter TC \
    --multi-objective-strategy max_compression_efficiency \
    --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
    --feature-store-root feature_store \
    --timeframe 240T \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### 4. 按Archetype顺序优化

优化顺序：TC → TE → FR → ET

```bash
# 步骤1: 优化TC
python scripts/optimize_gate_plateau_hard_gate.py \
    --raw-logs results/pipeline_highcap6_2024_full_*/logs_execution.parquet \
    --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
    --output results/tc_optimization.json \
    --compression-mode \
    --archetype-filter TC \
    --archetype-order TC \
    ...

# 步骤2: 验证TC结果
# 如果TC正常，继续优化TE
python scripts/optimize_gate_plateau_hard_gate.py \
    --raw-logs results/pipeline_highcap6_2024_full_*/logs_execution.parquet \
    --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
    --output results/te_optimization.json \
    --compression-mode \
    --archetype-filter TE \
    --archetype-order TE \
    ...

# 步骤3: 依次优化FR和ET
```

### 5. 生成对比报告

```bash
python scripts/compare_gate_optimization_experiments.py \
    --results-file results/gate_optimization_experiments/all_experiments_results.json \
    --output-dir results/gate_optimization_experiments
```

## 关键参数

### 压缩模式参数

- `--compression-mode`: 启用压缩模式
- `--compression-target-trade-rate`: 目标trade_rate（例如0.02）
- `--compression-min-robustness`: 最低robustness_score要求（例如0.5）
- `--compression-step`: 收紧步长（例如0.01）

### Archetype过滤参数

- `--archetype-filter`: 只优化指定archetype（例如 `TC` 或 `TC,TE,FR,ET`）
- `--archetype-order`: 指定优化顺序（例如 `TC,TE,FR,ET`）

### 多目标优化策略

- `max_compression_efficiency`: 最大压缩效率（推荐用于压缩模式）
- `max_robustness`: 最大稳健性
- `max_trade_rate`: 最大交易率
- `balanced`: 平衡策略
- `pareto_midpoint`: Pareto前沿中点

## 验证标准

### TC优化验证

- **trade_rate**: 从全松（~100%）压缩到目标（例如2%）
- **robustness_score**: 不低于最低要求（例如0.5）
- **E2E Sharpe**: 不低于baseline或显著提升
- **Max Drawdown**: 不显著增加

### 最终验证

- 所有archetype优化完成后，整体trade_rate在合理范围（例如1-5%）
- 整体robustness_score保持较高水平（例如>0.6）
- E2E KPI显著优于baseline

## 注意事项

1. **全松阈值**: 确保`config/nnmultihead/execution_archetypes.yaml`中所有阈值都是最松的（已在之前完成）

2. **FeatureStore**: 确保FeatureStore包含所有gate规则需要的特征（包括反身性特征）

3. **数据完整性**: 确保Pipeline日志包含所有6个token的完整数据

4. **优化顺序**: 建议按archetype顺序优化（TC → TE → FR → ET），每个archetype优化完成后验证效果

5. **压缩目标**: 根据实际需求调整`compression-target-trade-rate`，不要过度压缩导致交易机会过少

## 相关文档

- [Hard-Gate System指南](./HARD_GATE_SYSTEM.md)
- [多目标优化指南](./MULTI_OBJECTIVE_GATE_OPTIMIZATION.md)
- [Pipeline工作流文档](../workflow/PIPELINE_WORKFLOW.md)
