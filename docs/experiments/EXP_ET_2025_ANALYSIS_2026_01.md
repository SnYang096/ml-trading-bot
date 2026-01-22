# ET分析报告（2025年数据）

**实验时间**: 2026-01-22  
**实验目的**: 使用2025年数据分析ET（Exhaustion Turn）策略表现

---

## 执行摘要

### 关键发现

1. **ET样本数**：
   - ET_REGIME样本总数：9个
   - 通过gate的ET样本：9个（100%通过率）

2. **表现分析**：
   - ❌ **平均ret_mean**: -0.009604（负数）
   - ❌ **胜率**: 0.0%（所有样本都亏损）
   - ❌ **Sharpe**: -6.032（非常差）

3. **问题诊断**：
   - 所有9个样本的ret_mean都是负数
   - 胜率为0%，说明ET策略在当前条件下完全不适用
   - 可能原因：
     1. ET_REGIME分类条件不合适
     2. Gate rules太宽松，让不好的样本通过了
     3. 止损止盈配置不合适
     4. 2025年数据中ET策略本身不适用

---

## 详细分析

### 1. 样本统计

**ET_REGIME样本总数**: 9个  
**通过gate的ET样本**: 9个（100%通过率）

**注意**: 100%通过率可能表示gate rules太宽松，或者ET样本质量本身有问题。

### 2. 表现分析

| 指标 | 值 |
|------|-----|
| 平均ret_mean | -0.009604 |
| 中位数ret_mean | 0.000000 |
| 胜率 | 0.0% |
| Sharpe | -6.032 |

**结论**: ET策略在2025年数据中表现极差，所有样本都亏损。

### 3. 特征检查

**Volume Profile特征**:
- 需要检查是否有`vpvr_lvn_distance`等特征
- 如果特征缺失，`et_near_lvn`规则无法正常工作

**VPIN特征**:
- 需要检查是否有`vpin`特征
- 如果特征缺失，`has_orderflow` evidence无法通过

### 4. 问题诊断

#### 4.1 ET_REGIME分类条件

当前ET_REGIME条件（来自`src/time_series_model/rule/regime.py`）：
- `et_atr_percentile_min`: 0.85
- `et_path_efficiency_min_pct`: 0.55
- `et_path_efficiency_max_pct`: 0.7
- `et_jump_risk_min_pct`: 0.2
- `et_jump_risk_max_pct`: 0.5
- `et_path_length_min_pct`: 0.6

**可能问题**: 这些条件可能不适合2025年的市场环境。

#### 4.2 Gate Rules

**当前gate rules**（来自`execution_archetypes.yaml`）：
- `deny_if`: 多个deny规则
- `allow_if`: 需要满足至少一个allow规则
- `default_action`: deny

**可能问题**: 100%通过率说明gate rules可能太宽松，或者allow规则太容易满足。

#### 4.3 止损止盈配置

ET的止损止盈配置（来自`execution_archetypes.yaml`）：
- `stop_loss_r`: 1.0
- `take_profit_r`: 2.0
- `max_holding_bars`: 24

**可能问题**: 这些配置可能不适合2025年的市场波动。

#### 4.4 数据适用性

**2025年数据特点**:
- 时间范围：2025-05-01 到 2025-10-31
- 可能市场环境与ET策略设计时的环境不同

**可能问题**: ET策略可能不适用于2025年的市场环境。

---

## 下一步建议

### 1. 检查特征可用性

**优先级**: 高

- 检查`vpvr_lvn_distance`特征是否存在
- 检查`vpin`特征是否存在
- 如果特征缺失，需要等待FeatureStore重建完成

### 2. 分析ET样本的物理特征分布

**优先级**: 高

- 分析9个ET样本的物理特征值（path_efficiency, jump_risk, atr_percentile等）
- 与之前分析中表现好的ET样本对比
- 找出差异

### 3. 优化ET_REGIME分类条件

**优先级**: 中

- 基于2025年数据重新优化ET_REGIME条件
- 可能需要更严格的条件来筛选更好的样本

### 4. 优化Gate Rules

**优先级**: 中

- 分析为什么100%的样本都通过了gate
- 可能需要更严格的allow规则
- 或者需要更严格的deny规则

### 5. 优化止损止盈配置

**优先级**: 低

- 测试不同的止损止盈配置
- 但考虑到胜率为0%，可能不是主要问题

---

## 相关文件

- `results/e2e_kpi/logs_3action_et_2025_gated.parquet` - Gate检查后的logs
- `results/e2e_kpi/logs_3action_with_et_regime_v3.parquet` - 原始logs（包含ET_REGIME）
- `config/nnmultihead/execution_archetypes.yaml` - ET配置
- `src/time_series_model/rule/regime.py` - ET_REGIME分类条件

---

## 结论

ET策略在2025年数据中表现极差，所有样本都亏损。需要进一步分析：
1. 特征是否可用（volume_profile, vpin）
2. ET样本的物理特征分布
3. ET_REGIME分类条件是否合适
4. Gate rules是否太宽松

建议等待FeatureStore重建完成后，使用包含volume_profile和vpin特征的数据重新分析。
