# ET优化最终状态报告

**实验时间**: 2026-01-22  
**实验目的**: 完成ET优化的所有任务

---

## 执行摘要

### ✅ 已完成的工作

1. **脚本重命名**：
   - ✅ 重命名`apply_tree_gate_3action.py` → `apply_archetype_gate.py`
   - ✅ 更新了主要引用文件（`experiment_regime_gate.py`, `src/cli/main.py`等）
   - ✅ 更新了脚本文档说明

2. **ET为什么需要Volume Profile - 数据说明**：
   - ✅ 创建了分析脚本`analyze_et_volume_profile_effectiveness.py`
   - ✅ 生成了详细分析报告`EXP_ET_VOLUME_PROFILE_EFFECTIVENESS_2026_01.md`
   - ✅ **核心发现**：
     - `et_near_lvn`规则是ET的**核心allow规则**之一
     - 需要`vpvr_lvn_distance`特征来判断价格是否接近LVN
     - `has_volume_profile`是**required_evidence**，必须满足
     - Volume Profile用于识别流动性节点（LVN/HVN/POC），检测趋势末期特征，验证反转信号

3. **Auto-detect功能**：
   - ✅ 改进了auto-detect以处理`any_key_contains`规则
   - ✅ 验证了auto-detect能够正确检测到`volume_profile_block`和`vpin_block`
   - ✅ 移除了手动配置，完全依赖auto-detect

4. **ET专用配置**：
   - ✅ 添加了ET专用止损止盈配置
   - ✅ 在`apply_archetype_gate.py`中添加了ET专用ret_mean重新计算逻辑

5. **ET_REGIME条件优化**：
   - ✅ 优化了ET_REGIME分类条件（提高atr、path_efficiency，降低jump_risk）

### ⏳ 进行中的工作

1. **FeatureStore重建**：
   - ⚠️ **问题**：vpin需要tick数据，但当前数据路径中没有tick文件
   - ✅ **确认**：volume_profile特征不需要tick数据（只需要OHLCV）
   - ⏳ **状态**：重建命令已运行，但vpin计算失败

### 📋 待完成的工作

1. **解决vpin的tick数据问题**：
   - 选项1：提供tick数据路径
   - 选项2：暂时跳过vpin，只计算volume_profile（如果vpin已存在于其他layer）
   - 选项3：使用现有的vpin数据（如果其他layer已有）

2. **完成FeatureStore重建**：
   - 确保volume_profile_vpvr_f被正确计算
   - 验证vpvr_*特征（特别是vpvr_lvn_distance）存在

3. **重新运行Regime分类**：
   - 使用优化后的ET_REGIME条件
   - 验证ET_REGIME样本数（预期约15个）

4. **运行Gate检查**：
   - 使用新的FeatureStore layer
   - 验证has_volume_profile evidence
   - 验证et_near_lvn规则

5. **分析结果**：
   - 验证Sharpe改善（从-3.803到约2.495）
   - 分析volume profile特征的有效性

---

## 详细分析

### 1. 脚本重命名

**重命名原因**：
- "3action"指的是NO_TRADE/MEAN/TREND三种交易模式
- 但名称不够直观，用户不清楚含义
- 脚本的主要功能是根据regime和archetype应用gate规则

**新名称**：`apply_archetype_gate.py`
- 更准确地描述了脚本的功能
- 清晰表明是根据archetype应用gate规则

**更新的文件**：
- `scripts/apply_archetype_gate.py` - 新文件（已创建）
- `scripts/experiment_regime_gate.py` - 已更新引用
- `src/cli/main.py` - 已更新引用（2处）
- `scripts/analyze_regime_as_gate_veto.py` - 已更新引用

**注意**：
- 旧文件`scripts/apply_tree_gate_3action.py`仍然存在（保留用于向后兼容）
- 文档中的引用可能需要后续更新

### 2. ET为什么需要Volume Profile - 数据说明

#### 2.1 Gate Rules中的使用

**`et_near_lvn`规则**：
```yaml
- name: et_near_lvn
  kind: value_lt
  key: vpvr_lvn_distance
  threshold: 0.2
  on_missing: false
```

**作用**：
- 检测价格是否接近低成交量节点（LVN）
- LVN是流动性真空，价格接近LVN时容易快速穿越
- 这是ET的**核心allow规则**之一（需要满足至少一个allow规则）

**为什么重要**：
- ET是"趋势衰竭反转"，需要在趋势末期识别反转点
- LVN是流动性真空，价格一旦接近，容易快速穿越，触发反转
- 如果没有volume profile，这个规则无法工作

#### 2.2 Evidence Rules中的使用

**`has_volume_profile` Evidence**：
```yaml
- name: has_volume_profile
  kind: any_key_contains
  any_key_contains: ["vpvr_", "volume_profile", "vp_"]
```

**作用**：
- 确保有volume profile数据可用
- 这是**required_evidence**，必须满足才能执行ET交易

#### 2.3 Volume Profile的语义价值

**识别流动性节点**：
- LVN (Low Volume Node)：流动性真空，价格容易快速穿越
- HVN (High Volume Node)：高成交量区域，可能成为支撑/阻力
- POC (Point of Control)：价值中枢，成交量最大的价格

**检测趋势末期特征**：
- 成交量集中在特定价格区间（POC附近）
- 价格偏离价值中枢（POC），可能回归
- 存在多个LVN，表示流动性分散，趋势可能衰竭

**验证反转信号**：
- 价格接近LVN时，反转概率增加（`et_near_lvn`规则）
- 成交量分布显示趋势末期的特征

### 3. FeatureStore重建状态

**当前问题**：
- vpin需要tick数据，但当前数据路径中没有tick文件
- 错误：`ValueError: VPIN calculation requires tick data`

**解决方案选项**：

**选项1：提供tick数据**（推荐）
- 如果tick数据在其他路径，需要指定正确的路径
- 或者先准备tick数据

**选项2：使用现有vpin数据**
- 如果其他FeatureStore layer已有vpin数据
- 可以从现有layer复制vpin特征

**选项3：暂时跳过vpin**
- 如果vpin不是ET的required_evidence（但实际上vpin是required的）
- 或者先只计算volume_profile，vpin后续补充

**Volume Profile状态**：
- ✅ `volume_profile_vpvr_f`在`volume_profile_block`中
- ✅ 只需要OHLCV数据，不需要tick数据
- ✅ 会输出`vpvr_*`特征（包括`vpvr_lvn_distance`）

---

## 下一步行动

### 立即行动

1. **解决vpin的tick数据问题**：
   - 检查是否有tick数据在其他路径
   - 或者使用现有的vpin数据（如果其他layer已有）
   - 或者暂时跳过vpin，只计算volume_profile（不推荐，因为vpin是required）

2. **完成FeatureStore重建**：
   - 确保volume_profile_vpvr_f被正确计算
   - 验证vpvr_*特征存在

### 后续验证

3. **重新运行Regime分类**：
   ```bash
   python scripts/rerun_regime_with_optimized_conditions.py \
     --logs results/e2e_kpi/logs_3action.parquet \
     --output results/e2e_kpi/logs_3action_et_optimized.parquet \
     --feature-store-root feature_store \
     --layer nnmh_highcap6_240T_2024_202510_v3 \
     --timeframe 240T
   ```

4. **运行Gate检查**：
   ```bash
   python scripts/apply_archetype_gate.py \
     --logs results/e2e_kpi/logs_3action_et_optimized.parquet \
     --out results/e2e_kpi/logs_3action_et_optimized_gated.parquet \
     --features-store-root feature_store \
     --features-store-layer nnmh_highcap6_240T_2024_202510_v3 \
     --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
     --live-config config/nnmultihead/live/meta_router_live_config.yaml \
     --physics-regime results/e2e_kpi/logs_3action_et_optimized.parquet
   ```

5. **分析结果**：
   - 验证ET_REGIME样本数
   - 验证Sharpe改善
   - 验证volume profile特征的有效性

---

## 相关文件

- `scripts/apply_archetype_gate.py` - 重命名后的脚本
- `scripts/analyze_et_volume_profile_effectiveness.py` - Volume Profile有效性分析
- `docs/experiments/EXP_ET_VOLUME_PROFILE_EFFECTIVENESS_2026_01.md` - 详细分析报告
- `config/tasks/task_spec_highcap6_2024_202510.yaml` - TaskSpec配置（auto-detect启用）
- `src/time_series_model/rule/regime.py` - 优化的ET_REGIME条件
- `src/time_series_model/rl/execution_returns_rr.py` - ET专用配置

---

## 关键发现总结

1. **ET为什么需要Volume Profile**：
   - `et_near_lvn`规则是ET的核心allow规则之一
   - 需要`vpvr_lvn_distance`特征来判断价格是否接近LVN
   - `has_volume_profile`是required_evidence
   - Volume Profile用于识别流动性节点，检测趋势末期特征，验证反转信号

2. **Auto-detect功能**：
   - 能够自动检测到volume_profile_block和vpin_block
   - 不需要手动配置

3. **FeatureStore重建**：
   - volume_profile特征不需要tick数据（只需要OHLCV）
   - vpin需要tick数据，需要解决这个问题
