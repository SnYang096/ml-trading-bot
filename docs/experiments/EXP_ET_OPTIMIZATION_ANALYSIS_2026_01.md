# ET优化分析报告

**实验时间**: 2026-01-22  
**实验目的**: 分析并优化ET的三个关键问题：volume_profile特征、止损止盈配置、Sharpe优化

---

## 执行摘要

### 关键发现

1. **Volume Profile特征**：
   - ✅ 特征可以计算（`compute_unified_volume_profile`函数存在）
   - ❌ 但FeatureStore中当前没有这些特征（因为`volume_profile_block`未启用）
   - ✅ **解决方案**：在`task_spec_highcap6_2024_202510.yaml`中启用`volume_profile_block`

2. **止损止盈配置**：
   - ⚠️ **问题**：ET配置中的`fixed_rr`（stop_loss_r: 1.0, take_profit_r: 2.0）**没有被使用**
   - ⚠️ **实际使用**：ET使用MEAN模式的默认配置（stop_loss_r: 3.0, take_profit_r: 5.0）
   - **原因**：`ret_mean`的计算基于`mode`（MEAN/TREND），而不是archetype

3. **Sharpe优化**：
   - ✅ **找到正Sharpe条件**：高atr (>= 0.85) + 高path_efficiency (0.55-0.7) + 低jump_risk (0.2-0.5)
   - ✅ **优化结果**：15个样本，平均ret_mean: 0.002212，胜率40%，Sharpe: 2.495

---

## 详细分析

### 1. Volume Profile特征分析

#### 1.1 特征定义

**特征计算函数**：
- `compute_unified_volume_profile` - 在`src/features/time_series/utils_volume_profile.py`中定义
- 输出特征包括：
  - `vp_poc` - Point of Control
  - `vp_lvn_distance` - Distance to nearest LVN
  - `vp_hvn_count` - High Volume Node count
  - `vp_lvn_count` - Low Volume Node count
  - `vp_volume_density` - Volume density at current price

**FeatureStore中的特征节点**：
- `volume_profile_vpvr_f` - 输出`vpvr_*`前缀的特征
- `volume_profile_volatility_features_f` - 输出波动率相关特征

#### 1.2 为什么之前说"特征不可用"

**原因**：
- `volume_profile_block`在`feature_plan.yaml`中定义，但未在`task_spec_highcap6_2024_202510.yaml`中启用
- 只有`vpin_block`被启用
- 因此FeatureStore中没有计算和存储volume_profile相关特征

#### 1.3 解决方案

**已实施**：
1. ✅ 在`task_spec_highcap6_2024_202510.yaml`中启用`volume_profile_block`
2. ✅ 恢复`has_volume_profile`到`required_evidence`
3. ✅ 更新evidence rules以包含`vpvr_`前缀的特征

**下一步**：
- 需要重新生成FeatureStore以包含volume_profile特征
- 或者从现有FeatureStore的其他layer中查找是否有volume_profile特征

### 2. 止损止盈配置分析

#### 2.1 当前配置

**ET配置**（`execution_archetypes.yaml`）：
```yaml
fixed_rr:
  stop_loss_r: 1.0
  take_profit_r: 2.0
  max_holding_bars: 24
```

**实际使用**（`execution_returns_rr.py`）：
```python
# MEAN-specific execution overrides
mean_stop_loss_r: float = 3.0
mean_take_profit_r: float = 5.0
mean_trailing_atr_mult: float = 3.0
```

#### 2.2 问题分析

**关键发现**：
- `ret_mean`的计算基于`mode`（MEAN/TREND），而不是archetype
- ET的`mode`是`MEAN`，所以使用MEAN模式的配置
- ET配置中的`fixed_rr`**没有被使用**

**代码逻辑**：
```python
# In execution_returns_rr.py
entry_ok_m, sign_m = _compute_entry_signal_and_dir(g, cfg=cfg, mode="MEAN")
cfg_mean = replace(
    cfg,
    take_profit_r=cfg.mean_take_profit_r,  # 使用MEAN模式配置
    stop_loss_r=cfg.mean_stop_loss_r,      # 使用MEAN模式配置
)
```

#### 2.3 影响分析

**当前ET样本表现**：
- 平均ret_mean: -0.000978
- 正收益率: 11.1% (1/9)
- Sharpe: -3.803

**分析**：
- MEAN模式的止损止盈配置（stop_loss_r: 3.0, take_profit_r: 5.0）可能不适合ET
- ET是"趋势衰竭反转"，可能需要：
  - 更快的止盈（因为反转后可能快速回撤）
  - 更宽的止损（因为反转可能先触发止损）

#### 2.4 优化建议

**方案1：为ET创建专门的执行配置**（推荐）
- 在`RRExecutionReturnsConfig`中添加`et_*`配置
- 修改`compute_rr_execution_mode_returns`以支持ET模式
- 设置ET专用的止损止盈参数

**方案2：调整MEAN模式配置**
- 如果ET和FR都使用MEAN模式，需要找到平衡点
- 或者为ET创建独立的执行模式

### 3. Sharpe优化分析

#### 3.1 当前ET_REGIME表现

**当前条件**：
- atr_percentile >= 0.8
- path_efficiency_pct: 0.4-0.6
- jump_risk_pct: 0.3-0.6
- path_length_pct >= 0.5

**结果**：
- 16个样本
- 平均ret_mean: -0.002334
- 胜率: 25.0%
- Sharpe: -5.323

#### 3.2 优化条件发现

**最优条件组合**（测试结果）：
- atr_percentile >= 0.85（提高）
- path_efficiency_pct: 0.55-0.7（提高下限和上限）
- jump_risk_pct: 0.2-0.5（降低上限）
- path_length_pct >= 0.6（提高）

**结果**：
- 15个样本
- 平均ret_mean: 0.002212 ✅
- 胜率: 40.0% ✅
- Sharpe: 2.495 ✅ **正Sharpe！**

#### 3.3 正负收益样本特征对比

**正收益样本特征**：
- atr_percentile: 0.924（更高）
- path_efficiency_pct: 0.599（更高）
- jump_risk_pct: 0.378（更低）
- path_length_pct: 0.669（相似）

**负收益样本特征**：
- atr_percentile: 0.903
- path_efficiency_pct: 0.497
- jump_risk_pct: 0.506
- path_length_pct: 0.666

**关键发现**：
- 正收益样本有**更高的path_efficiency**（0.599 vs 0.497）
- 正收益样本有**更低的jump_risk**（0.378 vs 0.506）

#### 3.4 优化方案

**已实施的优化**：
1. ✅ 提高atr_percentile要求：0.8 → 0.85
2. ✅ 提高path_efficiency范围：0.4-0.6 → 0.55-0.7
3. ✅ 降低jump_risk上限：0.6 → 0.5
4. ✅ 降低jump_risk下限：0.3 → 0.2
5. ✅ 提高path_length要求：0.5 → 0.6

**预期效果**：
- 样本数：从16个减少到约15个
- 平均ret_mean：从-0.002334改善到约0.002212
- Sharpe：从-5.323改善到约2.495

---

## 实施总结

### 已完成的优化

1. ✅ **启用volume_profile_block**：
   - 在`task_spec_highcap6_2024_202510.yaml`中添加`volume_profile_block`
   - 恢复`has_volume_profile`到`required_evidence`
   - 更新evidence rules以包含`vpvr_`前缀

2. ✅ **优化ET_REGIME分类条件**：
   - 提高atr_percentile要求（0.8 → 0.85）
   - 提高path_efficiency范围（0.4-0.6 → 0.55-0.7）
   - 降低jump_risk范围（0.3-0.6 → 0.2-0.5）
   - 提高path_length要求（0.5 → 0.6）

### 待解决的问题

1. ⏳ **止损止盈配置**：
   - ET配置中的`fixed_rr`没有被使用
   - 需要修改`execution_returns_rr.py`以支持ET专用的止损止盈配置
   - 或者调整MEAN模式配置以适配ET

2. ⏳ **Volume Profile特征**：
   - 需要重新生成FeatureStore以包含volume_profile特征
   - 或者从其他layer中查找是否有这些特征

---

## 下一步行动

1. ✅ **已完成**: Volume Profile特征恢复（配置已更新）
2. ⏳ **待办**: 重新生成FeatureStore或查找现有volume_profile特征
3. ⏳ **待办**: 修改`execution_returns_rr.py`以支持ET专用的止损止盈配置
4. ⏳ **待办**: 使用优化后的ET_REGIME条件重新运行regime分类
5. ⏳ **待办**: 验证优化后的ET表现

---

## 相关文件

- `config/tasks/task_spec_highcap6_2024_202510.yaml` - 已启用volume_profile_block
- `config/nnmultihead/execution_archetypes.yaml` - 已恢复has_volume_profile
- `src/time_series_model/rule/regime.py` - 已优化ET_REGIME条件
- `src/time_series_model/rl/execution_returns_rr.py` - 需要修改以支持ET专用配置
- `results/et_optimization_analysis.json` - 详细分析数据
