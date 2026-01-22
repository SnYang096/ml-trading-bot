# ET优化实施报告

**实验时间**: 2026-01-22  
**实验目的**: 实施ET优化：改进auto-detect、移除手动配置、修改止损止盈、准备FeatureStore重建

---

## 实施总结

### ✅ 已完成的工作

1. **改进auto-detect功能**：
   - ✅ 修改`extract_required_features_from_execution_archetypes`函数，处理`any_key_contains`规则
   - ✅ 提取`any_key_contains`中的模式字符串（如`vpvr_`, `volume_profile`, `vp_`）
   - ✅ 更新`feature_to_block_patterns`映射，包含`vpvr`模式

2. **移除手动添加的blocks**：
   - ✅ 从`task_spec_highcap6_2024_202510.yaml`中移除手动添加的`volume_profile_block`
   - ✅ 添加注释说明blocks由auto-detect自动处理
   - ✅ 验证auto-detect能够正确检测到`volume_profile_block`和`vpin_block`

3. **修改止损止盈配置支持ET**：
   - ✅ 在`RRExecutionReturnsConfig`中添加ET专用配置：
     - `et_use_time_exit: bool = True`
     - `et_use_trailing_stop: bool = True`
     - `et_trailing_atr_mult: float = 2.0`
     - `et_take_profit_r: float = 1.5` (从ET config的2.0优化为1.5，更快止盈)
     - `et_stop_loss_r: float = 1.5` (从ET config的1.0优化为1.5，更宽止损)
     - `et_use_breakeven_stop: bool = False`
   - ✅ 修改`compute_rr_execution_mode_returns`函数，支持根据archetype选择ET配置
   - ✅ 如果archetype包含'ET'，使用ET专用配置；否则使用标准MEAN配置

### ⏳ 待完成的工作

1. **重新生成FeatureStore**：
   - ⏳ 运行`mlbot nnmultihead build-feature-store`命令
   - ⏳ 使用更新后的TaskSpec（依赖auto-detect）
   - ⏳ 验证FeatureStore中包含volume_profile特征

2. **验证优化效果**：
   - ⏳ 使用优化后的ET_REGIME条件重新运行regime分类
   - ⏳ 运行gate检查
   - ⏳ 分析ET样本的表现，验证Sharpe是否改善

---

## 详细实施内容

### 1. Auto-detect功能改进

**文件**: `src/cli/auto_detect_compute_requirements.py`

**修改内容**：
```python
# 在extract_required_features_from_execution_archetypes函数中
# 添加对any_key_contains规则的处理
if rule.get('kind') == 'any_key_contains' and 'any_key_contains' in rule:
    patterns = rule['any_key_contains']
    if isinstance(patterns, list):
        for pattern in patterns:
            required_features.add(str(pattern))
    elif isinstance(patterns, str):
        required_features.add(str(patterns))
```

**效果**：
- ✅ 现在可以检测到`has_volume_profile` evidence规则需要的特征模式
- ✅ `vpvr_`, `volume_profile`, `vp_`等模式被正确提取
- ✅ 自动映射到`volume_profile_block`

### 2. 移除手动配置

**文件**: `config/tasks/task_spec_highcap6_2024_202510.yaml`

**修改前**：
```yaml
feature_plan_overrides:
  optional_blocks_enabled:
    - vpin_block
    - volume_profile_block  # Enable volume_profile features for ET archetype
```

**修改后**：
```yaml
feature_plan_overrides:
  optional_blocks_enabled:
    # Note: volume_profile_block and vpin_block are auto-detected from execution_archetypes.yaml
    # Only add blocks here if they are NOT used in gate/evidence rules (user-defined blocks)
```

**验证结果**：
```
✅ 自动推导的required blocks: ['volume_profile_block', 'vpin_block']
✅ 所有必需的blocks都被自动检测到！
   - volume_profile_block: ✅
   - vpin_block: ✅
```

### 3. ET专用止损止盈配置

**文件**: `src/time_series_model/rl/execution_returns_rr.py`

**添加的配置**：
```python
# ET-specific execution overrides (Exhaustion Turn: trend late stage reversal)
# ET requires faster take-profit and wider stop-loss due to reversal nature
et_use_time_exit: bool = True
et_use_trailing_stop: bool = True
et_trailing_atr_mult: float = 2.0  # Tighter trailing stop for ET
et_take_profit_r: float = 1.5  # Faster take-profit (from ET config: 2.0, but optimized for reversal)
et_stop_loss_r: float = 1.5  # Wider stop-loss (from ET config: 1.0, but optimized for reversal)
et_use_breakeven_stop: bool = False
```

**修改的逻辑**：
```python
# 在compute_rr_execution_mode_returns函数中
# 检查是否有ET archetype
use_et_config = False
if archetype_col and archetype_col in g.columns:
    et_mask = g[archetype_col].astype(str).str.contains('ET', case=False, na=False)
    use_et_config = et_mask.any()

if use_et_config:
    # 使用ET专用配置
    cfg_mean = replace(cfg, ...et_*配置...)
else:
    # 使用标准MEAN配置
    cfg_mean = replace(cfg, ...mean_*配置...)
```

**注意**：
- 目前`build_execution_logs.py`在计算ret_mean时，还没有archetype信息
- 需要在gate检查之后，根据archetype重新计算ret_mean，或者修改`build_execution_logs.py`以传递archetype信息
- 这是一个后续优化点

---

## 下一步行动

1. **重新生成FeatureStore**：
   ```bash
   mlbot nnmultihead build-feature-store \
     --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
     --data-path /path/to/data \
     --output-layer nnmh_highcap6_240T_2024_202510_v3
   ```

2. **验证FeatureStore**：
   - 检查新layer中是否包含`vpvr_*`特征
   - 验证特征的非空率

3. **重新运行regime分类**：
   ```bash
   python scripts/rerun_regime_with_optimized_conditions.py \
     --logs results/e2e_kpi/logs_3action.parquet \
     --output results/e2e_kpi/logs_3action_et_optimized.parquet \
     --feature-store-root feature_store \
     --layer nnmh_highcap6_240T_2024_202510_v3 \
     --timeframe 240T
   ```

4. **运行gate检查**：
   ```bash
   python scripts/apply_tree_gate_3action.py \
     --logs results/e2e_kpi/logs_3action_et_optimized.parquet \
     --out results/e2e_kpi/logs_3action_et_optimized_gated.parquet \
     --features-store-root feature_store \
     --features-store-layer nnmh_highcap6_240T_2024_202510_v3 \
     --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
     --live-config config/nnmultihead/live/meta_router_live_config.yaml \
     --physics-regime results/e2e_kpi/logs_3action_et_optimized.parquet
   ```

5. **分析结果**：
   - 检查ET_REGIME样本数
   - 分析ET样本的ret_mean和Sharpe
   - 验证是否达到预期改善（Sharpe从-3.803改善到约2.495）

---

## 相关文件

- `src/cli/auto_detect_compute_requirements.py` - 改进的auto-detect功能
- `config/tasks/task_spec_highcap6_2024_202510.yaml` - 移除手动配置
- `src/time_series_model/rl/execution_returns_rr.py` - ET专用配置
- `src/time_series_model/rule/regime.py` - 已优化的ET_REGIME条件
- `config/nnmultihead/execution_archetypes.yaml` - 已恢复has_volume_profile

---

## 注意事项

1. **Archetype信息传递**：
   - 目前`build_execution_logs.py`在计算ret_mean时，还没有archetype信息
   - 需要在gate检查之后，根据archetype重新计算ret_mean
   - 或者修改`build_execution_logs.py`以传递archetype信息（如果可用）

2. **FeatureStore重建**：
   - 需要确保tick数据可用（volume_profile特征可能需要tick数据）
   - 重建可能需要较长时间

3. **向后兼容**：
   - ET专用配置是新增的，不影响现有的MEAN模式配置
   - 如果archetype信息不可用，自动回退到标准MEAN配置
