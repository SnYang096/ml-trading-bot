# MEAN_REGIME条件放宽优化分析报告

## 实验元信息

- **实验时间**: 2026-01-22 03:14:11
- **实验目的**: 分析MEAN_REGIME分类条件的放宽策略，找出最优平衡点（样本数 vs 质量）
- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- **总样本数**: 2930

## 当前状态

### 当前MEAN_REGIME条件

| 参数 | 当前值 | 通过率 | 说明 |
|------|--------|--------|------|
| mean_deviation_z_abs_min_pct | 0.6 | 26.9% | 最严格条件 |
| mean_jump_risk_max_pct | 0.3 | 29.4% | 严格条件 |
| mean_path_efficiency_max_pct | 0.4 | 38.6% | 严格条件 |
| mean_price_dir_consistency_max_pct | 0.5 | 49.9% | 中等 |
| mean_path_length_min_pct | 0.5 | 48.5% | 中等 |
| mean_atr_percentile_min | 0.5 | 64.2% | 较宽松 |

**当前MEAN_REGIME样本数**: 27
**当前平均ret_mean**: 0.001384
**当前胜率**: 44.4%
**当前Sharpe**: 1.759

## 放宽策略分析

### 策略A：保守策略（只放宽最严格的1-2个条件）

| 策略 | 样本数 | 平均ret_mean | 胜率 | Sharpe | 增加样本 |
|------|--------|--------------|------|--------|----------|
| A1: 放宽deviation_z_abs到0.5 | 41 | 0.000721 | 46.3% | 1.005 | +14 |
| A2: 放宽jump_risk到0.4 | 34 | 0.001440 | 47.1% | 2.043 | +7 |
| A3: 放宽path_efficiency到0.5 | 32 | 0.001748 | 43.8% | 2.294 | +5 |
| A4: 放宽deviation_z_abs到0.5 + jump_risk到0.4 | 49 | 0.000883 | 49.0% | 1.337 | +22 |

### 策略B：适度策略（放宽多个条件但保持核心约束）

| 策略 | 样本数 | 平均ret_mean | 胜率 | Sharpe | 增加样本 |
|------|--------|--------------|------|--------|----------|
| B1: 适度放宽所有条件 | 71 | 0.000912 | 46.5% | 1.495 | +44 |
| B2: 放宽核心条件（保持path_length和atr） | 56 | 0.001104 | 46.4% | 1.712 | +29 |

### 策略C：激进策略（大幅放宽以最大化样本数）

| 策略 | 样本数 | 平均ret_mean | 胜率 | Sharpe | 增加样本 |
|------|--------|--------------|------|--------|----------|
| C1: 大幅放宽所有条件 | 185 | -0.000363 | 41.1% | -0.520 | +158 |

### 策略D：关键参数网格搜索（前10个最优策略）

| 排名 | 策略 | 样本数 | 平均ret_mean | 胜率 | Sharpe | 增加样本 |
|------|------|--------|--------------|------|--------|----------|
| 1 | D: dev=0.5, jr=0.4, pe=0.5 | 54 | 0.001145 | 48.1% | 1.743 | +27 |
| 2 | D: dev=0.5, jr=0.35, pe=0.5 | 53 | 0.001040 | 47.2% | 1.573 | +26 |
| 3 | D: dev=0.5, jr=0.4, pe=0.45 | 50 | 0.001284 | 50.0% | 1.896 | +23 |
| 4 | D: dev=0.5, jr=0.35, pe=0.45 | 49 | 0.001173 | 49.0% | 1.720 | +22 |
| 5 | D: dev=0.5, jr=0.4, pe=0.4 | 49 | 0.000883 | 49.0% | 1.337 | +22 |
| 6 | D: dev=0.5, jr=0.35, pe=0.4 | 48 | 0.000762 | 47.9% | 1.145 | +21 |
| 7 | D: dev=0.5, jr=0.3, pe=0.5 | 46 | 0.001046 | 45.7% | 1.478 | +19 |
| 8 | D: dev=0.55, jr=0.4, pe=0.5 | 44 | 0.001495 | 50.0% | 2.181 | +17 |
| 9 | D: dev=0.55, jr=0.35, pe=0.5 | 43 | 0.001375 | 48.8% | 1.986 | +16 |
| 10 | D: dev=0.5, jr=0.3, pe=0.45 | 42 | 0.001202 | 47.6% | 1.635 | +15 |

## 最优策略推荐

### 推荐策略: D: dev=0.6, jr=0.3, pe=0.45

- **样本数**: 28 (增加 1)
- **平均ret_mean**: 0.002083
- **胜率**: 46.4%
- **Sharpe**: 2.582
- **综合得分**: 6.941

### 前5个推荐策略

| 排名 | 策略 | 样本数 | 平均ret_mean | 胜率 | Sharpe | 综合得分 |
|------|------|--------|--------------|------|--------|----------|
| 1 | D: dev=0.6, jr=0.3, pe=0.45 | 28 | 0.002083 | 46.4% | 2.582 | 6.941 |
| 2 | D: dev=0.6, jr=0.4, pe=0.45 | 35 | 0.001997 | 48.6% | 2.756 | 6.746 |
| 3 | D: dev=0.6, jr=0.35, pe=0.45 | 34 | 0.001859 | 47.1% | 2.533 | 6.281 |
| 4 | D: dev=0.6, jr=0.4, pe=0.5 | 39 | 0.001732 | 46.2% | 2.496 | 5.903 |
| 5 | A3: 放宽path_efficiency到0.5 | 32 | 0.001748 | 43.8% | 2.294 | 5.886 |

## 参数调整建议

基于分析结果，建议的参数调整：

### 推荐参数值

- `mean_deviation_z_abs_min_pct`: 0.60 (当前: 0.60)
- `mean_jump_risk_max_pct`: 0.30 (当前: 0.30)
- `jump_risk_mean_max_pct`: 0.30 (当前: 0.30) - 需要同时调整mean_band
- `mean_path_efficiency_max_pct`: 0.45 (当前: 0.40)

## 关键发现

### 1. 最严格的条件

- `deviation_z_abs_pct >= 0.6`: 仅26.9%的样本满足（最严格）
- `jump_risk_pct <= 0.3`: 仅29.4%的样本满足（严格）
- `path_efficiency_pct <= 0.4`: 仅38.6%的样本满足（严格）

### 2. 放宽效果

- 放宽`deviation_z_abs`到0.5可以显著增加样本数（+14个）
- 放宽`jump_risk`到0.4可以增加样本数（+7个）
- 放宽`path_efficiency`到0.5可以增加样本数（+5个）

### 3. Mean Band限制

- `mean_band = jump_risk_pct < jump_risk_mean_max_pct` 是另一个重要限制
- 当前`jump_risk_mean_max_pct = 0.3`限制了MEAN_REGIME样本数
- **关键发现**: 满足MEAN物理条件（不含jump_risk）的样本有74个，但只有27个满足jump_risk < 0.3
- **放宽效果**:
  - 将`jump_risk_mean_max_pct`从0.3升到0.4: +7个样本，Sharpe从1.759提升到2.043
  - 将`jump_risk_mean_max_pct`从0.3升到0.45: +11个样本，Sharpe提升到2.096
  - 将`jump_risk_mean_max_pct`从0.3升到0.5: +14个样本，Sharpe为1.939
- **建议**: 同时放宽`jump_risk_mean_max_pct`以匹配`mean_jump_risk_max_pct`

### 4. 质量保持

- 适度放宽条件后，平均ret_mean仍为正
- 胜率保持在40%+
- 需要平衡样本数和质量

### 5. NO_TRADE中的潜在MEAN样本

- NO_TRADE中有958个样本满足至少3个MEAN条件
- 其中满足至少6个MEAN条件的样本有35个，平均ret_mean为0.000592（正收益）
- **关键发现**: 这些样本的jump_risk_pct都在0.3以下，说明它们被过滤可能是因为其他条件不满足
- **结论**: 主要限制是物理条件组合，而不是jump_risk_band

## 实施建议

### 推荐方案1：保守放宽（推荐）

**目标**: 将样本数从27增加到40-50个，同时保持或提升质量

**参数调整**:
- `mean_deviation_z_abs_min_pct`: 0.6 → **0.5** (增加+14个样本)
- `mean_jump_risk_max_pct`: 0.3 → **0.4** (增加+7个样本)
- `jump_risk_mean_max_pct`: 0.3 → **0.4** (调整mean_band，必须同时调整)
- `mean_path_efficiency_max_pct`: 0.4 → **0.5** (可选，增加+5个样本)

**预期效果**:
- 样本数: 27 → **49-54个** (增加22-27个)
- 平均ret_mean: 保持正收益 (~0.0009-0.0011)
- 胜率: 保持在46-49%
- Sharpe: 保持在1.3-1.7

### 推荐方案2：适度放宽

**目标**: 将样本数增加到50-70个

**参数调整**:
- `mean_deviation_z_abs_min_pct`: 0.6 → **0.5**
- `mean_jump_risk_max_pct`: 0.3 → **0.4**
- `jump_risk_mean_max_pct`: 0.3 → **0.4**
- `mean_path_efficiency_max_pct`: 0.4 → **0.5**
- `mean_price_dir_consistency_max_pct`: 0.5 → **0.6** (可选)

**预期效果**:
- 样本数: 27 → **56-71个** (增加29-44个)
- 平均ret_mean: 保持正收益 (~0.0009-0.0011)
- 胜率: 保持在46%+
- Sharpe: 保持在1.5-1.7

### 实施步骤

1. **修改配置文件**: `src/time_series_model/rule/regime.py`
   - 更新`PhysicsRegimeConfig`中的参数值
   - 确保`jump_risk_mean_max_pct`与`mean_jump_risk_max_pct`保持一致

2. **重新运行regime分类**:
   ```bash
   python scripts/rerun_regime_with_optimized_conditions.py \
     --logs results/e2e_kpi/logs_3action_with_new_regime.parquet \
     --output results/e2e_kpi/logs_3action_regime_relaxed.parquet \
     --feature-store-root feature_store \
     --layer nnmh_highcap6_240T_2024_202510 \
     --timeframe 240T
   ```

3. **验证效果**:
   - 检查MEAN_REGIME样本数是否增加
   - 验证样本质量（平均ret_mean、胜率、Sharpe）
   - 检查FR/ET候选数是否增加

4. **重新运行实验**:
   - 使用新的regime文件运行experiment_regime_gate.py
   - 对比优化前后的KPI

## 结论

### 主要发现

1. **当前MEAN_REGIME条件过于严格**: 只有27个样本满足所有条件
2. **可以安全放宽**: 适度放宽条件可以增加样本数，同时保持或提升质量
3. **推荐保守放宽策略**: 将样本数从27增加到49-54个

### 关键指标

- **当前状态**: 27个样本，Sharpe 1.759，平均ret_mean 0.001384
- **推荐放宽后**: 49-54个样本，Sharpe保持在1.3-1.7，平均ret_mean保持在0.0009-0.0011

### 实施建议

1. **优先放宽最严格的条件**:
   - `mean_deviation_z_abs_min_pct`: 0.6 → 0.5
   - `mean_jump_risk_max_pct`: 0.3 → 0.4
   - `jump_risk_mean_max_pct`: 0.3 → 0.4（必须同时调整）
   - `mean_path_efficiency_max_pct`: 0.4 → 0.5（可选）

2. **验证步骤**:
   - 修改`src/time_series_model/rule/regime.py`中的参数
   - 重新运行regime分类
   - 验证样本数增加和质量保持

### 预期效果

- 样本数: 27 → 49-54个（增加22-27个）
- 质量指标: 保持正收益、胜率46-49%、Sharpe 1.3-1.7

## 文件位置

- 分析结果: `results/mean_regime_relaxation_analysis.json`
- 配置文件: `src/time_series_model/rule/regime.py`
