# ET独立Regime实施方案

**实验时间**: 2026-01-22  
**实验目的**: 为ET创建独立Regime（ET_REGIME），解决MEAN_REGIME不适合ET的问题

---

## 实施总结

### 已完成的工作

1. ✅ **实现ET_REGIME分类逻辑**：
   - 在`src/time_series_model/rule/regime.py`中添加了ET_REGIME
   - 定义了ET_REGIME的分类条件（中等跳风险、高波动率、中等路径效率等）
   - 确保与其他Regime互斥

2. ✅ **调整ET配置**：
   - 降低`has_orderflow`的vpin quantile要求（0.55 → 0.5）
   - 从required_evidence中移除`has_volume_profile`（特征不可用）
   - 放宽`et_mean_adx_too_high`阈值（25 → 30）
   - 降低`et_vpin_spike`的quantile要求（0.65 → 0.5）

3. ✅ **更新gate脚本**：
   - 在`scripts/apply_tree_gate_3action.py`中添加ET_REGIME支持
   - 设置ET_REGIME优先选择ExhaustionTurnET archetype

4. ✅ **更新live config**：
   - 在`config/nnmultihead/live/meta_router_live_config.yaml`中添加ET_REGIME配置

---

## ET_REGIME定义

### 分类条件

ET_REGIME用于识别趋势末期（Exhaustion Turn）的市场状态，具有以下特征：

1. **中等跳风险**（jump_risk_pct: 0.3-0.6）
   - 既不是太低（否则是TC），也不是太高（否则是TE或NO_TRADE）
   - 表示市场处于中等风险状态

2. **高波动率**（atr_percentile >= 0.8）
   - 波动率高潮是ET的典型特征
   - 表示市场处于极端状态

3. **中等路径效率**（path_efficiency_pct: 0.4-0.6）
   - 既不是太高（否则是TC），也不是太低（否则是MEAN）
   - 表示趋势开始变得低效

4. **足够路径长度**（path_length_pct >= 0.5）
   - 确保有足够的趋势形成

### 与其他Regime的互斥性

- **TC_REGIME**: 低跳风险（0.3-0.6），高路径效率，稳定趋势
- **TE_REGIME**: 高跳风险（0.6-0.9），高波动率，趋势扩张
- **MEAN_REGIME**: 低跳风险（< 0.4），低路径效率，均值回归
- **ET_REGIME**: 中等跳风险（0.3-0.6），中等路径效率，高波动率，趋势末期

**互斥逻辑**：
- 首先按跳风险划分：NO_TRADE (>= 0.9) > TE (0.6-0.9) > TC/ET/MEAN (< 0.6)
- 在低跳风险区域，按路径效率和波动率划分：
  - TC: 高路径效率 + 低波动率
  - ET: 中等路径效率 + 高波动率 (>= 0.8)
  - MEAN: 低路径效率 + 低波动率

---

## 配置变更

### 1. Regime分类配置 (`src/time_series_model/rule/regime.py`)

**新增参数**：
```python
# ET Regime constraints
et_adx_min: float = 20.0  # Minimum ADX for trend strength
et_adx_max: float = 30.0  # Maximum ADX (not too strong)
et_atr_percentile_min: float = 0.8  # High volatility
et_path_efficiency_min_pct: float = 0.4  # Minimum path efficiency
et_path_efficiency_max_pct: float = 0.6  # Maximum path efficiency
et_jump_risk_min_pct: float = 0.3  # Minimum jump risk
et_jump_risk_max_pct: float = 0.6  # Maximum jump risk
```

**分类逻辑**：
```python
et_band = (
    ~np.isnan(jump_risk_pct)
    & (jump_risk_pct >= cfg.et_jump_risk_min_pct)
    & (jump_risk_pct < cfg.et_jump_risk_max_pct)
)
et_physical_ok = (
    (~np.isnan(atr_percentile) & (atr_percentile >= cfg.et_atr_percentile_min))
    & (path_efficiency_pct >= cfg.et_path_efficiency_min_pct)
    & (path_efficiency_pct <= cfg.et_path_efficiency_max_pct)
    & (path_length_pct >= 0.5)
)
et_mask = et_physical_ok & et_band & (regime == "NO_TRADE") & (~hard_veto)
regime[et_mask] = "ET_REGIME"
```

### 2. ET Archetype配置 (`config/nnmultihead/execution_archetypes.yaml`)

**变更**：
1. **required_evidence**: 从`[has_orderflow, has_volume_profile]`改为`[has_orderflow]`
   - 移除了`has_volume_profile`（特征不可用）

2. **has_orderflow quantile**: 从0.55降低到0.5
   - 适应ET_REGIME中中等订单流活动的要求

3. **et_vpin_spike quantile**: 从0.65降低到0.5
   - 与has_orderflow保持一致

4. **et_mean_adx_too_high threshold**: 从25放宽到30
   - ET_REGIME允许ADX范围20-30

### 3. Gate脚本更新 (`scripts/apply_tree_gate_3action.py`)

**变更**：
1. 添加ET_REGIME的archetype优先级逻辑
2. 更新regime映射：ET_REGIME映射到MEAN（用于archetype选择）

### 4. Live配置更新 (`config/nnmultihead/live/meta_router_live_config.yaml`)

**变更**：
```yaml
enabled_archetypes:
  TREND:
    - TrendContinuationTC
    - TrendExpansionTE
  MEAN:
    - FailureReversionFR
  ET:  # NEW: ET_REGIME for ExhaustionTurnET
    - ExhaustionTurnET
  NO_TRADE: []
```

---

## 预期效果

### 1. ET样本数量

- **之前**: 在MEAN_REGIME中，0个ET样本通过evidence rules
- **预期**: 在ET_REGIME中，应该有更多ET样本能够通过gate和evidence rules

### 2. ET表现

- **之前**: ET在MEAN_REGIME中无法执行（evidence rules失败）
- **预期**: ET在ET_REGIME中应该能够正常执行，并表现出更好的alpha

### 3. Regime分布

- **之前**: 所有样本被分类为TC/TE/MEAN/NO_TRADE
- **预期**: 部分样本（趋势末期、高波动率、中等跳风险）被分类为ET_REGIME

---

## 下一步验证

1. **运行regime分类**：
   ```bash
   python3 scripts/rerun_regime_with_optimized_conditions.py \
     --logs results/e2e_kpi/logs_3action.parquet \
     --output results/e2e_kpi/logs_3action_regime_with_et.parquet
   ```

2. **检查ET_REGIME样本数**：
   - 统计ET_REGIME的样本数量
   - 分析ET_REGIME样本的特征分布

3. **运行gate检查**：
   ```bash
   python3 scripts/apply_tree_gate_3action.py \
     --logs results/e2e_kpi/logs_3action_regime_with_et.parquet \
     --out results/e2e_kpi/logs_3action_et_regime_gated.parquet
   ```

4. **分析ET表现**：
   - 检查ET样本通过gate和evidence rules的数量
   - 分析ET样本的实际表现（ret_mean, Sharpe等）

---

## 相关文件

- `src/time_series_model/rule/regime.py` - ET_REGIME分类逻辑
- `config/nnmultihead/execution_archetypes.yaml` - ET配置
- `scripts/apply_tree_gate_3action.py` - Gate脚本
- `config/nnmultihead/live/meta_router_live_config.yaml` - Live配置
- `docs/experiments/EXP_ET_RULES_SEMANTIC_ANALYSIS_2026_01.md` - 语义分析报告

---

## 注意事项

1. **ET_REGIME是新的regime类型**，需要确保所有相关脚本都支持它
2. **ADX特征可能不可用**，ET_REGIME分类不依赖ADX，而是使用其他物理特征
3. **Volume Profile特征不可用**，已从required_evidence中移除
4. **需要验证ET_REGIME的实际效果**，根据结果可能需要调整分类条件
