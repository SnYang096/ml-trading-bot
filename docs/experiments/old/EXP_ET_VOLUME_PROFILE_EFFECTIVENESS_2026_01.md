# ET为什么需要Volume Profile - 数据分析报告

**实验时间**: 2026-01-22  
**实验目的**: 用数据说明ET为什么需要volume profile特征

---

## 执行摘要

### 关键发现

1. **ET Gate Rules中使用Volume Profile**：
   - ✅ `et_near_lvn`规则：检测价格是否接近LVN（低成交量节点）
   - ✅ `et_vpin_spike`规则：检测订单流峰值（虽然vpin不是volume profile，但相关）

2. **ET Evidence Rules中使用Volume Profile**：
   - ✅ `has_volume_profile`：确保有volume profile数据可用

3. **Volume Profile对ET的价值**：
   - 识别流动性节点（LVN/HVN/POC）
   - 检测趋势末期特征
   - 验证反转信号

---

## 详细分析

### 1. ET Gate Rules中Volume Profile的使用

#### 1.1 `et_near_lvn`规则

**配置**：
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
- 适合ET反转策略：价格接近LVN时，反转概率增加

**为什么重要**：
- ET是"趋势衰竭反转"，需要在趋势末期识别反转点
- LVN是流动性真空，价格一旦接近，容易快速穿越，触发反转
- 这个规则是ET的**核心allow规则**之一（需要满足至少一个allow规则）

#### 1.2 `et_vpin_spike`规则

**配置**：
```yaml
- name: et_vpin_spike
  kind: quantile_gt
  key: vpin
  quantile: 0.5
  on_missing: false
```

**作用**：
- 检测订单流峰值（虽然vpin不是volume profile，但相关）
- 趋势末期通常伴随订单流异常

### 2. ET Evidence Rules中Volume Profile的使用

#### 2.1 `has_volume_profile` Evidence

**配置**：
```yaml
- name: has_volume_profile
  kind: any_key_contains
  any_key_contains: ["vpvr_", "volume_profile", "vp_"]
```

**作用**：
- 确保有volume profile数据可用
- 这是**required_evidence**，必须满足才能执行ET交易

**为什么重要**：
- ET需要volume profile特征来识别关键价位和流动性节点
- 如果没有volume profile数据，ET无法正确识别LVN，无法执行

### 3. Volume Profile对ET的语义价值

#### 3.1 识别流动性节点

**LVN (Low Volume Node)**：
- 流动性真空，成交量低的价格区间
- 价格接近LVN时容易快速穿越
- ET需要检测价格是否接近LVN（`et_near_lvn`规则）

**HVN (High Volume Node)**：
- 高成交量区域，可能成为支撑/阻力
- 趋势末期，成交量可能集中在特定价格区间

**POC (Point of Control)**：
- 价值中枢，成交量最大的价格
- 价格偏离POC时，可能回归到价值中枢

#### 3.2 检测趋势末期特征

**成交量分布**：
- 趋势末期，成交量可能集中在特定价格区间（POC附近）
- 存在多个LVN，表示流动性分散，趋势可能衰竭

**价格偏离**：
- 价格偏离价值中枢（POC），可能回归
- 这是ET反转策略的核心逻辑

#### 3.3 验证反转信号

**价格接近LVN**：
- 价格接近LVN时，反转概率增加（`et_near_lvn`规则）
- 这是ET的**核心allow规则**之一

**成交量分布**：
- 成交量分布显示趋势末期的特征
- 价格偏离POC，可能回归到价值中枢

---

## 数据验证

### 当前状态

**ET样本数**：9个（来自之前的gate检查）

**Volume Profile特征**：
- ⚠️ 当前数据中**没有volume profile特征**
- 需要重建FeatureStore以包含这些特征

### 预期效果

**重建FeatureStore后**：
- ET样本将包含`vpvr_lvn_distance`等特征
- `et_near_lvn`规则可以正常工作
- `has_volume_profile` evidence可以通过

**预期改善**：
- ET样本可以通过`et_near_lvn`规则（如果价格接近LVN）
- 更准确地识别ET反转机会
- 提高ET策略的成功率

---

## 结论

### ET为什么需要Volume Profile

1. **核心Gate Rule依赖**：
   - `et_near_lvn`规则是ET的**核心allow规则**之一
   - 需要`vpvr_lvn_distance`特征来判断价格是否接近LVN
   - 如果没有volume profile，这个规则无法工作

2. **Required Evidence**：
   - `has_volume_profile`是**required_evidence**
   - 必须满足才能执行ET交易
   - 确保有volume profile数据可用

3. **语义匹配**：
   - ET是"趋势衰竭反转"
   - 需要识别流动性节点（LVN）来检测反转点
   - Volume Profile提供了这些关键信息

### 数据支持

- ✅ ET配置中明确使用`vpvr_lvn_distance`特征
- ✅ `et_near_lvn`是ET的allow规则之一
- ✅ `has_volume_profile`是required_evidence
- ⏳ 需要重建FeatureStore后验证实际效果

---

## 相关文件

- `config/nnmultihead/execution_archetypes.yaml` - ET配置
- `scripts/analyze_et_volume_profile_effectiveness.py` - 分析脚本
- `results/et_volume_profile_analysis.json` - 分析结果

---

## 下一步

1. ⏳ 重建FeatureStore包含volume_profile特征
2. ⏳ 重新运行gate检查，验证`et_near_lvn`规则
3. ⏳ 分析有volume profile时ET的表现
