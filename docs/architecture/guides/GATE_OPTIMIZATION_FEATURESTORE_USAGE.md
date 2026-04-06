# Gate优化脚本FeatureStore使用指南

## 概述

Gate优化脚本现在支持从FeatureStore动态加载特征，解决了logs文件缺少gate规则所需特征的问题。

## 当前机制

1. **logs_execution.parquet** 只包含基本列：
   - `symbol`, `timestamp`, `ret_mean`, `ret_trend`, `open`, `high`, `low`, `close`, `atr` 等

2. **Gate规则需要的特征**在FeatureStore中：
   - `path_efficiency_pct`, `jump_risk_pct`, `cvd_change_5_pct` 等
   - 这些特征需要从FeatureStore加载

3. **优化脚本的工作流程**：
   - 读取logs文件
   - 提取gate规则所需的所有特征（从execution_archetypes.yaml）
   - 检查logs文件是否包含这些特征
   - 如果缺失，从FeatureStore加载（如果提供了`--feature-store-layer`参数）
   - 合并特征到logs DataFrame
   - 如果FeatureStore中也缺失特征，提示用户重新构建FeatureStore

## 使用方法

### Hard-Gate System优化

```bash
python scripts/optimize_gate_plateau_hard_gate.py \
    --gated-logs results/pipeline_<run_id>/logs_execution_gated.parquet \
    --raw-logs results/pipeline_<run_id>/logs_execution.parquet \
    --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
    --output results/gate_optimization_hard_gate.json \
    --feature-store-root feature_store \
    --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
    --timeframe 240T \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --min-trade-rate 0.001 \
    --min-trades-per-bucket 3 \
    --min-sharpe-threshold 0.05 \
    --threshold-step 0.05
```

### 渐进式优化

```bash
python scripts/optimize_gate_plateau_progressive.py \
    --gated-logs results/pipeline_<run_id>/logs_execution_gated.parquet \
    --raw-logs results/pipeline_<run_id>/logs_execution.parquet \
    --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
    --output results/gate_optimization_progressive.json \
    --feature-store-root feature_store \
    --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
    --timeframe 240T \
    --target-trades 200 \
    --tighten-step 0.05
```

### 对比实验

```bash
python scripts/compare_gate_optimization_methods.py \
    --gated-logs results/pipeline_<run_id>/logs_execution_gated.parquet \
    --raw-logs results/pipeline_<run_id>/logs_execution.parquet \
    --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
    --output-dir results/gate_optimization_comparison \
    --feature-store-root feature_store \
    --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
    --timeframe 240T
```

## 参数说明

### FeatureStore相关参数

- `--feature-store-root`: FeatureStore根目录（默认: `feature_store`）
- `--feature-store-layer`: FeatureStore layer名称（**必需**，如果logs文件缺少特征）
- `--timeframe`: 时间框架（默认: `240T`）
- `--start-date`: 开始日期（可选，如果不提供则从logs文件推断）
- `--end-date`: 结束日期（可选，如果不提供则从logs文件推断）

### 优化参数

- `--min-trade-rate`: 最小交易率（默认: 0.001）
- `--min-trades-per-bucket`: 每桶最少交易数（默认: 5）
- `--min-sharpe-threshold`: 平台高原的最低Sharpe要求（默认: 0.1）
- `--threshold-step`: 阈值扫描步长（默认: 0.05）

## 工作流程

1. **读取logs文件** → 检查包含哪些列
2. **提取所需特征** → 从execution_archetypes.yaml提取所有gate规则使用的特征
3. **检查缺失特征** → 对比logs文件和所需特征
4. **从FeatureStore加载** → 如果提供了`--feature-store-layer`，从FeatureStore加载缺失的特征
5. **合并特征** → 将FeatureStore的特征merge到logs DataFrame
6. **再次检查** → 如果仍有缺失，提示用户重新构建FeatureStore

## 如果特征缺失

如果FeatureStore中也缺少某些特征，脚本会：
1. 打印缺失的特征列表
2. 提示用户重新构建FeatureStore
3. 退出并返回错误码

**解决方案**：
```bash
# 重新构建FeatureStore，包含所需特征
mlbot nnmultihead build-feature-store \
    --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
    --symbols BTCUSDT,ETHUSDT \
    --timeframe 240T \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --feature-store-root feature_store \
    --layer nnmh_highcap6_240T_2024_with_reflexivity \
    --no-docker
```

## 注意事项

1. **FeatureStore必须存在**：如果提供了`--feature-store-layer`，FeatureStore必须存在且包含所需特征
2. **时间框架匹配**：`--timeframe`必须与logs文件的时间框架匹配
3. **特征合并**：如果logs文件中已有某些特征，优先使用logs文件中的值
4. **性能**：从FeatureStore加载特征可能需要一些时间，特别是对于大量数据

## 与apply_archetype_gate.py的一致性

优化脚本复用了`apply_archetype_gate.py`中的`_read_feature_store_range`函数，确保：
- 特征加载逻辑一致
- timestamp处理方式一致
- 特征合并方式一致

这样可以确保优化脚本使用的特征与gate应用脚本使用的特征完全一致。
