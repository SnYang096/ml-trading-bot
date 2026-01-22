# 解决方案：物理特征读取和MEAN_REGIME分类问题

## 问题总结

1. **MEAN_REGIME分类问题**: 原始logs中的regime是在优化前分类的，需要重新运行regime分类以应用优化后的条件
2. **物理特征读取问题**: `baseline_gated.parquet`中没有`path_efficiency_pct`等物理特征列，导致gate rules无法正确判断

## 解决方案

### 1. 修复物理特征读取 (`apply_tree_gate_3action.py`)

**修改内容**:
- 修改了`apply_tree_gate_3action.py`，让它从`physics_regime`文件merge物理特征
- 现在会merge以下物理特征：
  - `path_efficiency_pct`
  - `price_dir_consistency_pct`
  - `deviation_z_abs_pct`
  - `path_length_pct`
  - `jump_risk_pct`
  - `atr_percentile`
  - 以及优化后的`regime`分类

**使用方法**:
```bash
python scripts/apply_tree_gate_3action.py \
  --logs logs_3action.parquet \
  --out gated.parquet \
  --physics-regime physics_regime_optimized.parquet \
  --features-store-root feature_store \
  --features-store-layer nnmh_highcap6_240T_2024_202510 \
  ...
```

### 2. 重新运行Regime分类 (`rerun_regime_with_optimized_conditions.py`)

**新脚本**: `scripts/rerun_regime_with_optimized_conditions.py`

这个脚本会：
1. 从原始logs读取数据
2. 使用优化后的`PhysicsRegimeConfig`重新运行regime分类
3. 输出包含物理特征和优化后regime的parquet文件

**使用方法**:
```bash
python scripts/rerun_regime_with_optimized_conditions.py \
  --logs results/e2e_kpi/logs_3action_with_new_regime.parquet \
  --output results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --feature-store-root feature_store \
  --layer nnmh_highcap6_240T_2024_202510 \
  --timeframe 240T
```

**输出**:
- 包含优化后regime分类的parquet文件
- 包含所有物理特征（path_efficiency_pct等）
- 可以用于后续的gate应用

### 3. 完整工作流程

#### 步骤1: 重新运行Regime分类
```bash
python scripts/rerun_regime_with_optimized_conditions.py \
  --logs results/e2e_kpi/logs_3action_with_new_regime.parquet \
  --output results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --feature-store-root feature_store \
  --layer nnmh_highcap6_240T_2024_202510 \
  --timeframe 240T
```

#### 步骤2: 使用优化后的regime文件运行实验
```bash
python scripts/experiment_regime_gate.py \
  --logs results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --physics-regime results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --output-dir results/experiments_optimized \
  --features-store-root feature_store \
  --features-store-layer nnmh_highcap6_240T_2024_202510 \
  --symbols BTCUSDT,ETHUSDT,ADAUSDT,BNBUSDT,SOLUSDT \
  --timeframe 240T \
  --start-date 2025-05-01 \
  --end-date 2025-10-31 \
  --auto-compute-semantic-floors
```

**关键点**:
- `--physics-regime`参数指向优化后的regime文件
- 这样`apply_tree_gate_3action.py`会从该文件merge物理特征和优化后的regime

## 优化后的MEAN_REGIME条件

以下条件已经在`src/time_series_model/rule/regime.py`中优化：

1. **mean_deviation_z_abs_min_pct**: 0.85 → **0.6** (放宽)
2. **mean_path_length_min_pct**: 0.7 → **0.5** (放宽)
3. **mean_atr_percentile_min**: 0.8 → **0.5** (放宽)
4. **mean_path_efficiency_max_pct**: **0.4** (新增，低效率路径)
5. **mean_price_dir_consistency_max_pct**: **0.5** (新增，不稳定方向)
6. **mean_jump_risk_max_pct**: **0.3** (新增，低跳空风险)

## 验证

运行后检查：
1. `gated.parquet`文件中应该包含物理特征列
2. MEAN_REGIME样本数应该增加（从1个增加到更多）
3. FR/ET的gate rules应该能够正确判断（因为有了物理特征）

## 文件位置

- 修改的脚本: `scripts/apply_tree_gate_3action.py`
- 新脚本: `scripts/rerun_regime_with_optimized_conditions.py`
- 优化后的regime配置: `src/time_series_model/rule/regime.py` (PhysicsRegimeConfig)
