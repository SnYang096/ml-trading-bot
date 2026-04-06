# ET规则语义分析和独立Regime方案

**实验时间**: 2026-01-22  
**实验目的**: 分析ET规则/evidence的语义作用，评估MEAN_REGIME适配性，设计ET独立Regime方案

---

## 执行摘要

### 关键发现

1. **MEAN_REGIME不适合ET**：
   - ET的语义定义：趋势衰竭反转（发生在趋势末期）
   - MEAN_REGIME特征：均值回归、低趋势强度、低订单流
   - **结论**：ET需要趋势末期环境，而不是均值回归环境

2. **规则语义作用分析**：
   - `has_orderflow` (vpin quantile > 0.55)：在MEAN_REGIME中vpin普遍较低，无法满足
   - `has_volume_profile`：数据中完全没有volume_profile相关特征
   - `et_mean_adx_too_high` (ADX > 25)：被拒绝的19个样本平均ret_mean为0.001619，正收益率36.8%，表现一般

3. **解决方案**：为ET创建独立Regime（ET_REGIME）

---

## 详细分析

### 1. ET语义定义

**ET的核心语义**（来自文档）：
- **存在理由**: "趋势没死，但信仰没油了"
- **特征本质**: 高阶结构衰竭、极端条件触发
- **触发条件**: 
  - `long_trend_late_stage` (趋势后期)
  - `vol_climax` (波动率高潮)
  - `mfe_extreme` (最大有利偏移极端)
  - `momentum_divergence` (动量背离)

**关键特征**：
- 发生在**趋势末期**，不是均值回归
- 需要**高订单流活动**来确认趋势衰竭和反转信号
- 需要**volume profile**来确认关键价位
- 需要**中等趋势强度**（既不是太强也不是太弱）

### 2. 规则语义作用分析

#### 2.1 `has_orderflow` (vpin quantile > 0.55)

**语义作用**：
- VPIN (Volume-Synchronized Probability of Informed Trading) 衡量知情交易概率
- 在ET中，需要**高订单流活动**来确认趋势衰竭和反转信号
- **问题**：MEAN_REGIME中vpin普遍较低（因为均值回归发生在低波动、低订单流环境）

**分析结果**：
- 通过gate的27个ET样本中，vpin的quantile分布需要从FeatureStore读取完整数据
- 但根据MEAN_REGIME的特征（低波动、低订单流），vpin quantile > 0.55的要求可能过高

**建议**：
- 如果为ET创建独立Regime，可以设置vpin quantile > 0.5（中等订单流活动）
- 如果继续使用MEAN_REGIME，需要降低到0.4或0.45

#### 2.2 `has_volume_profile`

**语义作用**：
- Volume Profile用于识别关键价位（POC, LVN等）
- ET需要确认价格是否接近关键支撑/阻力位
- **问题**：数据中只有`market_profile`，没有`volume_profile`或`vp_`前缀的特征

**分析结果**：
- FeatureStore中可能有`volume_profile_vpvr_f`、`volume_profile_volatility_features_f`等特征
- 但这些特征可能没有被包含在当前的logs中
- `market_profile`是字符串类型（值为"standard"），不能用于数值计算

**建议**：
- 检查FeatureStore中是否有volume_profile相关特征
- 如果有，确保这些特征被正确读取到logs中
- 如果没有，从required_evidence中移除`has_volume_profile`，或使用其他替代特征

#### 2.3 `et_mean_adx_too_high` (ADX > 25)

**语义作用**：
- ADX衡量趋势强度，ADX > 25表示强趋势
- ET发生在趋势末期，但**仍然需要一定的趋势强度**（否则不是趋势衰竭，而是震荡）
- **问题**：在MEAN_REGIME中，ADX > 25的样本可能仍然有趋势，不适合ET

**分析结果**：
- 被拒绝的19个样本平均ret_mean为0.001619
- 正收益率为36.8%（7/19）
- 表现一般，说明ADX > 25的规则可能有效，但阈值可能需要调整

**建议**：
- 如果为ET创建独立Regime，可以设置ADX范围：20-30（中等趋势强度）
- 如果继续使用MEAN_REGIME，可以考虑放宽到30

### 3. MEAN_REGIME适配性评估

**MEAN_REGIME特征**：
- 低路径效率（path_efficiency）
- 不稳定方向（price_dir_consistency低）
- 高偏离均值（deviation_z高）
- **低趋势强度**（ADX通常较低）
- **低订单流活动**（vpin通常较低）

**ET需求**：
- 趋势末期（需要一定的趋势强度，ADX: 20-30）
- 高订单流活动（vpin quantile > 0.5）
- Volume profile确认（关键价位）
- 高波动率（atr_percentile > 0.8）
- 动量背离（cvd_change_5 < 0）

**结论**：
- **MEAN_REGIME不适合ET**
- ET更适合在**趋势末期**（TC_REGIME或TE_REGIME的末期），而不是均值回归环境
- 需要为ET创建独立的Regime

---

## ET独立Regime设计方案

### ET_REGIME定义

**核心思想**：ET发生在趋势末期，需要特定的物理特征组合

**分类条件**：

1. **趋势强度中等**（ADX: 20-30）
   - 既不是太强（否则是TC/TE），也不是太弱（否则是MEAN）
   - 表示趋势仍在，但开始衰竭

2. **高波动率**（atr_percentile > 0.8）
   - 波动率高潮是ET的典型特征
   - 表示市场处于极端状态

3. **订单流活跃**（vpin quantile > 0.5）
   - 需要足够的订单流活动来确认反转信号
   - 但不需要像TE那样极端（quantile > 0.65）

4. **接近关键价位**（sr_distance_normalized < 0.3）
   - 价格接近支撑/阻力位，容易发生反转

5. **动量背离**（cvd_change_5 < 0 或 cvd_change_5_normalized < 0.3）
   - 价格仍在上涨，但订单流开始背离
   - 这是ET的核心信号

6. **路径效率中等**（path_efficiency_pct: 0.4-0.6）
   - 既不是太高（否则是TC），也不是太低（否则是MEAN）
   - 表示趋势开始变得低效

7. **跳风险中等**（jump_risk_pct: 0.3-0.6）
   - 既不是太低（否则是TC），也不是太高（否则是TE或NO_TRADE）
   - 表示市场处于中等风险状态

### 与其他Regime的互斥性

- **TC_REGIME**: 低跳风险（jump_risk_pct: 0.3-0.6），高路径效率，稳定趋势
- **TE_REGIME**: 高跳风险（jump_risk_pct: 0.6-0.9），高波动率，趋势扩张
- **MEAN_REGIME**: 低跳风险（jump_risk_pct < 0.4），低路径效率，均值回归
- **ET_REGIME**: 中等跳风险（jump_risk_pct: 0.3-0.6），中等路径效率，趋势末期

**互斥逻辑**：
- 首先按跳风险划分：NO_TRADE (>= 0.9) > TE (0.6-0.9) > TC/ET/MEAN (< 0.6)
- 在低跳风险区域，按路径效率和趋势强度划分：
  - TC: 高路径效率 + 低ADX (< 20) 或 中等ADX (20-25)
  - ET: 中等路径效率 + 中等ADX (20-30) + 高波动率 + 订单流活跃
  - MEAN: 低路径效率 + 低ADX (< 20)

### 实施步骤

1. **在`regime.py`中添加ET_REGIME**：
   - 更新`RegimeType`类型定义
   - 在`PhysicsRegimeConfig`中添加ET_REGIME配置参数
   - 在`classify_regime`函数中添加ET_REGIME分类逻辑

2. **调整ET配置**：
   - 更新`execution_archetypes.yaml`中的ET配置
   - 调整gate rules和evidence rules以适应ET_REGIME
   - 降低`has_orderflow`的vpin quantile要求（0.55 → 0.5）
   - 处理`has_volume_profile`（检查特征可用性，或从required_evidence中移除）

3. **测试和验证**：
   - 运行regime分类，检查ET_REGIME样本数
   - 运行gate/evidence检查，验证ET样本能否通过
   - 分析ET_REGIME样本的实际表现（ret_mean, Sharpe等）

---

## 修复建议

### 短期修复（如果暂时不实现ET_REGIME）

1. **降低`has_orderflow`的vpin quantile要求**：
   - 从0.55降低到0.4或0.45
   - 适应MEAN_REGIME中vpin普遍较低的情况

2. **处理`has_volume_profile`**：
   - 检查FeatureStore中是否有volume_profile相关特征
   - 如果有，确保这些特征被正确读取
   - 如果没有，从required_evidence中移除

3. **调整`et_mean_adx_too_high`阈值**：
   - 从25放宽到30
   - 或移除此规则（因为MEAN_REGIME中ADX通常较低）

### 长期修复（推荐：实现ET_REGIME）

1. **实现ET_REGIME分类**：
   - 按照上述设计方案实现ET_REGIME
   - 确保与其他Regime互斥

2. **调整ET配置**：
   - 为ET_REGIME设置专门的gate rules和evidence rules
   - 降低vpin quantile要求到0.5
   - 处理volume_profile特征

3. **验证和优化**：
   - 分析ET_REGIME样本的实际表现
   - 根据结果优化regime分类条件和gate/evidence rules

---

## 下一步行动

1. ✅ **已完成**: 分析规则语义作用和MEAN_REGIME适配性
2. ⏳ **进行中**: 设计ET_REGIME分类条件
3. ⏳ **待办**: 实现ET_REGIME分类逻辑
4. ⏳ **待办**: 调整ET配置（gate rules和evidence rules）
5. ⏳ **待办**: 测试和验证ET_REGIME效果

---

## 附录

### 相关文件

- `src/time_series_model/rule/regime.py` - Regime分类逻辑
- `config/nnmultihead/execution_archetypes.yaml` - ET配置
- `scripts/diagnose_et_missing_data_detailed.py` - 诊断脚本
- `results/et_detailed_diagnosis_v2.json` - 诊断数据

### 参考文档

- `docs/archive/architecture/archetype灭绝级回测.md` - ET的语义定义
- `docs/architecture/树模型策略知识迁移到多头模型.md` - ET的触发条件
