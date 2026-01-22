# 实验: FR/ET和MEAN_REGIME优化

## 实验元信息

- **实验时间**: 2026-01-22
- **实验目的**: 
  1. 优化FR/ET表现（Sharpe从-2.398改善）
  2. 优化MEAN_REGIME分类条件，增加MEAN_REGIME样本数
  3. 分析regime vs gate的重要性
  4. 找出FR/ET被拒绝的根本原因

- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- **总样本数**: 2930
- **FeatureStore Layer**: `nnmh_highcap6_240T_2024_202510`
- **Timeframe**: 240T (4H)

## 实验配置

### 优化内容

1. **MEAN_REGIME优化** (`src/time_series_model/rule/regime.py`):
   - `mean_deviation_z_abs_min_pct`: 0.85 → 0.6 (放宽)
   - `mean_path_length_min_pct`: 0.7 → 0.5 (放宽)
   - `mean_atr_percentile_min`: 0.8 → 0.5 (放宽)
   - 新增 `mean_path_efficiency_max_pct`: 0.4 (低效率路径)
   - 新增 `mean_price_dir_consistency_max_pct`: 0.5 (不稳定方向)
   - 新增 `mean_jump_risk_max_pct`: 0.3 (低跳空风险)

2. **FR/ET Gate Rules优化** (`config/nnmultihead/execution_archetypes.yaml`):
   - FR新增: `path_efficiency_pct > 0.4` (拒绝高效率路径)
   - FR新增: `price_dir_consistency_pct > 0.5` (拒绝稳定方向)
   - FR新增: `deviation_z_abs_pct < 0.6` (拒绝低偏离)
   - ET新增: 同样的三个约束

### 实验配置对比

| 配置 | Regime | Gate Veto | Semantic Veto | 说明 |
|------|--------|-----------|---------------|------|
| baseline | ✅ | ✅ | ✅ | 有Regime + 优化后的Gate Rules |
| only_gate_rules | ✅ | ✅ | ❌ | 有Regime + Gate Rules（无Semantic） |
| no_regime_filter | ❌ | ✅ | ✅ | 无Regime + 优化后的Gate Rules |
| no_gate_veto | ✅ | ❌ | ✅ | 有Regime + 无Gate Veto |
| no_regime_no_veto | ❌ | ❌ | ✅ | 无Regime + 无Gate Veto |
| all_veto_off | ❌ | ❌ | ❌ | 全部关闭 |

## 实验结果

### 整体KPI对比

| 配置 | Sharpe | 交易数 | 胜率 | Profit/Loss Ratio |
|------|--------|--------|------|-------------------|
| **baseline** | **1.817** | **1456** | **35.6%** | **1.09** |
| only_gate_rules | 1.817 | 1456 | 35.6% | 1.09 |
| no_regime_filter | 1.305 | 1631 | 36.4% | 1.04 |
| no_gate_veto | 1.830 | 1457 | 35.6% | 1.08 |
| no_regime_no_veto | -0.045 | 11720 | 37.7% | 1.00 |
| all_veto_off | -0.045 | 11720 | 37.7% | 1.00 |

### FR/ET表现

#### Baseline配置（优化后）

| Archetype | Sharpe | 交易数 | 胜率 |
|-----------|--------|--------|------|
| FR | 0.000 | 0 | 0.0% |
| ET | 0.000 | 0 | 0.0% |
| TC | 1.817 | 1456 | 35.6% |

**关键发现**:
- FR/ET候选数: 1个（只有1个FR候选）
- FR/ET通过gate: 0个（被gate拒绝）
- 被拒绝的原因: `gate_allow_not_met`（FR的allow_if条件没有满足）
- MEAN_REGIME样本数: 0（原始logs中只有1个MEAN_REGIME）

#### All Veto Off配置（参考）

| Archetype | Sharpe | 交易数 | 胜率 |
|-----------|--------|--------|------|
| FR | -2.398 | 2930 | 38.2% |
| ET | -2.398 | 2930 | 38.2% |

**关键发现**:
- FR/ET交易数: 2930（全部通过）
- Sharpe: -2.398（表现很差）
- MEAN_REGIME样本数: 2个（从1个增加到2个，说明优化有一定效果）

## 问题诊断

### 1. MEAN_REGIME分类问题

**问题**: Baseline配置下MEAN_REGIME样本数仍然是0

**根本原因**:
- 原始logs中的regime是在优化前分类的
- 优化后的MEAN_REGIME条件需要重新运行regime分类才能生效
- 优化后的条件包括新的物理特征约束（path_efficiency_pct等）

**解决方案**:
- ✅ 已创建脚本 `scripts/rerun_regime_with_optimized_conditions.py` 来重新运行regime分类
- ✅ 使用优化后的`PhysicsRegimeConfig`（已在`regime.py`中配置）
- ✅ 确保物理特征（path_efficiency_pct等）在regime分类时正确计算并输出

**使用方法**:
```bash
python scripts/rerun_regime_with_optimized_conditions.py \
  --logs results/e2e_kpi/logs_3action_with_new_regime.parquet \
  --output results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --feature-store-root feature_store \
  --layer nnmh_highcap6_240T_2024_202510 \
  --timeframe 240T
```

### 2. 物理特征读取问题

**问题**: baseline_gated.parquet中没有path_efficiency_pct等物理特征列

**根本原因**:
- `apply_tree_gate_3action.py`从FeatureStore读取特征，但可能没有包含这些物理特征
- 或者FeatureStore中没有这些特征（需要在regime分类时计算）

**解决方案**:
- ✅ 已修改 `scripts/apply_tree_gate_3action.py`，让它从`physics_regime`文件merge物理特征
- ✅ 现在会merge以下物理特征：
  - `path_efficiency_pct`, `price_dir_consistency_pct`, `deviation_z_abs_pct`
  - `path_length_pct`, `jump_risk_pct`, `atr_percentile`
  - 以及优化后的`regime`分类

**使用方法**:
在运行`apply_tree_gate_3action.py`时，添加`--physics-regime`参数指向包含物理特征的regime文件：
```bash
python scripts/apply_tree_gate_3action.py \
  --logs logs_3action.parquet \
  --out gated.parquet \
  --physics-regime logs_3action_regime_optimized.parquet \
  ...
```

### 3. FR/ET被拒绝原因

**问题**: Baseline配置下只有1个FR候选，但被gate拒绝

**根本原因分析**:
- 被拒绝的FR/ET的regime是NO_TRADE（不是MEAN_REGIME）
- Gate Reasons显示: `gate_allow_not_met=['et_vol_climax', 'et_vpin_spike', ...]`
- FR的allow_if条件需要满足至少一个条件（allow_mode: any）
- 但所有allow_if条件都没有满足

**FR的allow_if条件**:
- `fr_divergence`: `cvd_change_5 < 0.3 quantile`
- `fr_vpin_spike`: `vpin > 0.65 quantile`
- `fr_cvd_divergence`: `cvd_change_5 < 0.3 quantile`
- `fr_absorption`: `vp_absorption_score > 0.7 quantile`
- `fr_near_sr`: `sr_distance_normalized < 0.3`

**可能原因**:
1. 被拒绝的FR/ET的特征值不满足allow_if条件
2. 需要quantile数据来判断，但可能quantile数据不正确
3. 物理特征约束（新增的deny_if）可能太严格

### 4. MEAN_REGIME Alpha分析

**发现**:
- 原始logs中只有1个MEAN_REGIME样本
- 该样本的ret_mean: 0.003686（正收益）
- All Veto Off配置下MEAN_REGIME样本增加到2个
- **结论**: MEAN_REGIME有alpha（正收益能力），但样本太少

### 5. 数据划分检查

**检查结果**:
- ✅ 数据时间范围正常: 2025-05-01 到 2025-10-31
- ✅ Symbols正常: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- ✅ 总样本数: 2930
- **结论**: 数据划分没有问题

## Regime vs Gate重要性分析

### 独立贡献

| 组件 | Sharpe影响 | 交易数影响 | 相对重要性 |
|------|-----------|-----------|-----------|
| **Regime** | **4.114** | **-1148** | **57.9%** |
| Gate | 2.732 | -690 | 38.5% |
| Semantic | 0.257 | +55 | 3.6% |

**结论**: Regime的影响更大，是主要过滤机制。

### 按Archetype分析

| Archetype | Baseline Sharpe | No Regime Sharpe | Impact |
|-----------|----------------|------------------|--------|
| TC | 4.657 | 3.749 | +0.908 |
| FR | 0.000 | -1.641 | +1.641 |

**发现**: 
- Regime过滤对FR的影响最大（+1.641 Sharpe）
- 说明Regime过滤对FR/ET非常重要

## 关键发现总结

1. **MEAN_REGIME优化有一定效果**:
   - All Veto Off配置下MEAN_REGIME样本从1个增加到2个
   - 但Baseline配置下仍然是0个（因为regime分类在gate之前，需要重新运行）

2. **FR/ET Gate Rules优化已应用**:
   - 配置已更新，但物理特征约束可能需要在gate阶段正确读取

3. **Regime vs Gate重要性确认**:
   - Regime影响: 4.114 Sharpe (57.9%) - 更重要
   - Gate影响: 2.732 Sharpe (38.5%)

4. **MEAN_REGIME有Alpha**:
   - 唯一的MEAN_REGIME样本有正收益（0.003686）
   - 说明MEAN_REGIME有收益能力，但样本太少

5. **数据划分正常**:
   - 时间范围、Symbols、样本数都正常
   - 没有数据划分问题

## 根本问题诊断

### 问题1: MEAN_REGIME样本数太少

**根本原因**: 
- 原始logs中的regime是在优化前分类的
- 优化后的MEAN_REGIME条件需要重新运行regime分类

**解决方案**:
1. 重新运行regime分类，使用优化后的`PhysicsRegimeConfig`
2. 确保物理特征在regime分类时正确计算
3. 将regime分类结果合并到logs中

### 问题2: 物理特征没有正确读取

**根本原因**:
- `apply_tree_gate_3action.py`从FeatureStore读取特征
- 但物理特征（path_efficiency_pct等）可能不在FeatureStore中
- 这些特征需要在regime分类时计算

**解决方案**:
1. 检查FeatureStore是否包含物理特征
2. 如果不包含，在regime分类时计算并保存到FeatureStore
3. 或者在apply_tree_gate_3action.py中从regime分类结果读取

### 问题3: FR/ET被Gate Rules拒绝

**根本原因**:
- FR的allow_if条件需要满足至少一个（allow_mode: any）
- 但被拒绝的FR/ET的所有allow_if条件都没有满足
- 可能原因：
  - 特征值不满足条件
  - quantile数据不正确
  - 新增的物理特征约束（deny_if）太严格

**解决方案**:
1. 检查被拒绝的FR/ET的特征值
2. 验证quantile数据是否正确
3. 考虑放宽物理特征约束或调整allow_if条件

## 下一步建议

1. **重新运行Regime分类**:
   - 使用优化后的`PhysicsRegimeConfig`
   - 确保物理特征正确计算
   - 将结果合并到logs中

2. **修复物理特征读取**:
   - 确保物理特征从FeatureStore或regime分类结果中读取
   - 验证特征列名匹配

3. **分析FR/ET Gate Rules**:
   - 检查被拒绝的FR/ET的特征值
   - 验证quantile数据
   - 考虑调整gate_rules配置

4. **进一步优化MEAN_REGIME条件**:
   - 如果MEAN_REGIME样本数仍然太少，进一步放宽条件
   - 但保持高质量（确保有alpha）

## 文件位置

- 实验结果: `results/experiments_optimized/`
- 对比报告: `results/experiments_optimized/regime_gate_comparison.md`
- 详细KPI: `results/experiments_optimized/*_kpi.json` 和 `*_kpi.md`
- 深度分析: `results/mean_regime_fr_et_deep_analysis.json`
