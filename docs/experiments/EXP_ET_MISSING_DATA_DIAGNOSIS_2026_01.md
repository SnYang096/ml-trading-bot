# ET数据缺失深度诊断报告

**实验时间**: 2026-01-22  
**实验目的**: 全面诊断ET（ExhaustionTurn）没有数据的原因  
**数据范围**: MEAN_REGIME样本（46个）

---

## 执行摘要

### 关键发现

1. **Gate Rules不是主要问题**：
   - ✅ 27/46 (58.7%) 样本通过了gate rules
   - ❌ 19/46 (41.3%) 样本被拒绝，主要原因是 `et_mean_adx_too_high` 规则

2. **Evidence Rules是主要阻塞**：
   - ❌ **0/27 (0%)** 样本通过了evidence rules
   - 所有通过gate的样本都被evidence rules拒绝
   - `has_orderflow`: 0/27 (0.0%) - **主要阻塞**
   - `has_volume_profile`: 0/27 (0.0%) - **主要阻塞**

3. **特征可用性**：
   - ✅ 关键特征 `vpin`, `cvd_change_5`, `atr_percentile` 等都有
   - ❌ `vpvr_lvn_distance`, `trade_quality`, `volume_ratio`, `mean_score` 100%缺失

4. **FR vs ET竞争**：
   - 27个样本两者都通过了gate
   - 但都没有通过evidence（都被阻塞）

---

## 详细分析

### 1. Gate Rules分析

#### 总体统计
- 总样本数: 46
- 通过gate: 27 (58.7%)
- 被拒绝: 19 (41.3%)
- default_action拒绝: 0

#### deny_if规则统计

| 规则名称 | 触发次数 | 触发率 | 说明 |
|---------|---------|--------|------|
| `et_mean_adx_too_high` | 19 | 41.3% | **最严格规则** - 拒绝ADX > 25的样本 |
| `et_volume_too_low` | 0 | 0.0% | 未触发 |
| `et_bb_width_too_low` | 0 | 0.0% | 未触发 |
| `et_mean_sr_too_far` | 0 | 0.0% | 未触发 |
| `et_sqs_too_low` | 0 | 0.0% | 未触发 |
| `et_quality_too_low` | 0 | 0.0% | 未触发 |
| `et_score_too_low` | 0 | 0.0% | 未触发 |
| `et_path_efficiency_too_high` | 0 | 0.0% | 未触发 |
| `et_price_dir_consistency_too_high` | 0 | 0.0% | 未触发 |
| `et_deviation_too_low` | 0 | 0.0% | 未触发 |

**发现**: `et_mean_adx_too_high` 是唯一触发的deny_if规则，拒绝了19个样本。

#### allow_if规则统计

| 规则名称 | 满足次数 | 满足率 | 说明 |
|---------|---------|--------|------|
| `et_near_sr` | 27 | 58.7% | **唯一满足的规则** - 所有通过gate的样本都满足此规则 |
| `et_vol_climax` | 0 | 0.0% | 未满足 |
| `et_vpin_spike` | 0 | 0.0% | 未满足 |
| `et_cvd_divergence` | 0 | 0.0% | 未满足 |
| `et_momentum_decay` | 0 | 0.0% | 未满足 |
| `et_near_lvn` | 0 | 0.0% | 未满足 |

**发现**: 所有通过gate的样本都只满足了 `et_near_sr` 规则（allow_mode: any，所以满足一个即可）。

---

### 2. Evidence Rules分析

#### 总体统计
- 总样本数（通过gate后）: 27
- 通过evidence: **0 (0.0%)**
- 被拒绝: **27 (100.0%)**

#### Required Evidence统计

| Evidence名称 | 通过次数 | 失败次数 | 通过率 | 说明 |
|------------|---------|---------|--------|------|
| `has_orderflow` | 0 | 27 | 0.0% | **主要阻塞** - 需要vpin quantile > 0.55 |
| `has_volume_profile` | 0 | 27 | 0.0% | **主要阻塞** - 需要包含volume_profile相关特征 |

#### 所有Evidence统计

| Evidence名称 | 通过次数 | 失败次数 | 通过率 | 说明 |
|------------|---------|---------|--------|------|
| `has_orderflow` | 0 | 27 | 0.0% | vpin quantile > 0.55 |
| `has_volume_profile` | 0 | 27 | 0.0% | 需要包含volume_profile相关特征 |
| `has_momentum_decay` | 0 | 27 | 0.0% | 需要包含momentum相关特征 |
| `has_vol_climax` | 27 | 0 | 100.0% | ✅ 所有样本都满足 |

**关键发现**:
1. `has_orderflow` 失败率100% - 说明vpin的quantile阈值0.55可能太高，或者vpin值本身在MEAN_REGIME中较低
2. `has_volume_profile` 失败率100% - 说明数据中可能没有volume_profile相关的特征列

---

### 3. FR vs ET竞争分析

| 情况 | 样本数 | 说明 |
|------|--------|------|
| 两者都通过gate | 27 | FR和ET都通过了gate rules |
| 只有ET通过gate | 0 | 没有样本只有ET通过 |
| 只有FR通过gate | 0 | 没有样本只有FR通过 |
| 两者都不通过gate | 19 | 两者都被拒绝 |

#### Score分布比较

**ET Score** (通过gate的27个样本):
- 平均: 0.2716
- 中位数: 0.2845
- 标准差: 0.0697

**FR Score** (通过gate的27个样本):
- 平均: 0.2716
- 中位数: 0.2845
- 标准差: 0.0697

**发现**: ET和FR的score完全相同（因为使用的是同一个mean_score），说明在MEAN_REGIME中，两者在gate层面是平等的。

---

### 4. 特征可用性分析

#### Gate Rules需要的特征 (14个)

| 特征名称 | 可用数 | 缺失数 | 缺失率 | 说明 |
|---------|--------|--------|--------|------|
| `vpin` | 46 | 0 | 0.0% | ✅ 可用 |
| `cvd_change_5` | 46 | 0 | 0.0% | ✅ 可用 |
| `atr_percentile` | 46 | 0 | 0.0% | ✅ 可用 |
| `bb_width_normalized` | 46 | 0 | 0.0% | ✅ 可用 |
| `sr_distance_normalized` | 46 | 0 | 0.0% | ✅ 可用 |
| `sqs` | 46 | 0 | 0.0% | ✅ 可用 |
| `adx` | 46 | 0 | 0.0% | ✅ 可用 |
| `deviation_z_abs_pct` | 46 | 0 | 0.0% | ✅ 可用 |
| `path_efficiency_pct` | 46 | 0 | 0.0% | ✅ 可用 |
| `price_dir_consistency_pct` | 46 | 0 | 0.0% | ✅ 可用 |
| `vpvr_lvn_distance` | 0 | 46 | 100.0% | ❌ 完全缺失 |
| `trade_quality` | 0 | 46 | 100.0% | ❌ 完全缺失 |
| `volume_ratio` | 0 | 46 | 100.0% | ❌ 完全缺失 |
| `mean_score` | 0 | 46 | 100.0% | ❌ 完全缺失 |

**发现**: 
- 大部分gate rules需要的特征都可用
- 但 `vpvr_lvn_distance`, `trade_quality`, `volume_ratio`, `mean_score` 完全缺失
- 这些缺失的特征可能影响某些gate rules的评估，但由于allow_mode是any，只要满足一个allow_if即可

#### Evidence Rules需要的特征

| 特征名称 | 可用数 | 缺失数 | 缺失率 | 说明 |
|---------|--------|--------|--------|------|
| `vpin` | 46 | 0 | 0.0% | ✅ 可用（用于has_orderflow） |
| `atr_percentile` | 46 | 0 | 0.0% | ✅ 可用（用于has_vol_climax） |
| `bb_width_normalized` | 46 | 0 | 0.0% | ✅ 可用（用于has_vol_climax） |
| volume_profile相关 | 0 | 46 | 100.0% | ❌ 完全缺失（用于has_volume_profile） |

**发现**: `has_volume_profile` evidence失败是因为数据中完全没有volume_profile相关的特征列。

---

## 根本原因分析

### 问题1: `has_orderflow` Evidence失败（0/27）

**原因**: `has_orderflow` evidence要求 `vpin` 的quantile > 0.55，但所有27个通过gate的样本的vpin值都低于这个阈值。

**可能的原因**:
1. MEAN_REGIME中的vpin值普遍较低（因为mean reversion通常发生在低波动、低订单流的环境中）
2. quantile阈值0.55可能对MEAN_REGIME来说太高了
3. 需要检查这27个样本的实际vpin值和它们的quantile分布

**建议**:
- 检查这27个样本的vpin实际值和quantile分布
- 考虑为MEAN_REGIME中的ET单独设置更低的vpin quantile阈值（例如0.4或0.45）
- 或者将 `has_orderflow` 从required_evidence中移除，改为optional

### 问题2: `has_volume_profile` Evidence失败（0/27）

**原因**: `has_volume_profile` evidence使用 `any_key_contains` 规则，需要数据中包含 `volume_profile` 或 `vp_` 相关的特征列，但数据中完全没有这些特征。

**可能的原因**:
1. FeatureStore中没有计算volume_profile相关的特征
2. 这些特征没有被包含在logs中
3. 特征名称不匹配（例如实际特征名是 `volume_profile_*` 而不是 `vp_*`）

**建议**:
- 检查FeatureStore中是否有volume_profile相关的特征
- 如果有，确保这些特征被正确读取到logs中
- 如果没有，考虑：
  1. 从required_evidence中移除 `has_volume_profile`
  2. 或者实现volume_profile特征的计算

### 问题3: Gate Rules中的 `et_mean_adx_too_high` 规则

**原因**: 19个样本因为ADX > 25被拒绝。

**分析**: 这个规则的目的是过滤掉趋势太强的样本（因为ET是exhaustion turn，应该在趋势末期），但在MEAN_REGIME中，如果ADX > 25，说明仍然有较强的趋势，可能不适合ET。

**建议**:
- 检查这19个被拒绝的样本的实际表现（ret_mean）
- 如果这些样本的ret_mean是负的，说明规则有效
- 如果这些样本的ret_mean是正的，说明规则可能太严格，可以考虑将阈值从25提高到30或35

---

## 优化建议

### 短期优化（立即实施）

1. **放宽 `has_orderflow` evidence要求**:
   - 将quantile阈值从0.55降低到0.4或0.45
   - 或者将 `has_orderflow` 从required_evidence中移除，改为optional

2. **处理 `has_volume_profile` evidence**:
   - 检查FeatureStore中是否有volume_profile相关特征
   - 如果没有，从required_evidence中移除 `has_volume_profile`
   - 如果有，确保这些特征被正确读取

3. **检查 `et_mean_adx_too_high` 规则**:
   - 分析被拒绝的19个样本的实际表现
   - 如果表现不佳，保持规则
   - 如果表现良好，考虑放宽阈值

### 中期优化（需要进一步分析）

1. **分析vpin在MEAN_REGIME中的分布**:
   - 检查所有MEAN_REGIME样本的vpin值和quantile分布
   - 确定适合MEAN_REGIME的vpin quantile阈值

2. **优化allow_if规则**:
   - 目前只有 `et_near_sr` 规则满足
   - 考虑增加更多适合MEAN_REGIME的allow_if规则
   - 或者降低现有allow_if规则的阈值

3. **特征工程**:
   - 如果volume_profile特征确实需要，实现其计算
   - 或者找到替代特征来满足 `has_volume_profile` evidence

### 长期优化（架构层面）

1. **Regime-specific Evidence Rules**:
   - 考虑为不同regime设置不同的evidence rules
   - 例如，MEAN_REGIME中的ET可以使用更宽松的evidence要求

2. **特征可用性检查**:
   - 在gate/evidence评估之前，先检查所需特征的可用性
   - 如果特征缺失，提供明确的错误信息或fallback策略

---

## 下一步行动

1. ✅ **已完成**: 创建详细诊断脚本并运行分析
2. ⏳ **进行中**: 检查27个通过gate的样本的vpin实际值和quantile分布
3. ⏳ **待办**: 检查FeatureStore中是否有volume_profile相关特征
4. ⏳ **待办**: 分析被 `et_mean_adx_too_high` 拒绝的19个样本的实际表现
5. ⏳ **待办**: 根据分析结果优化evidence rules和gate rules

---

## 附录

### 诊断脚本

- 脚本路径: `scripts/diagnose_et_missing_data_detailed.py`
- 使用方法:
```bash
python3 scripts/diagnose_et_missing_data_detailed.py \
  --logs results/e2e_kpi/logs_3action_regime_optimized_v2.parquet \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_highcap6_240T_2024_202510_v2 \
  --timeframe 240T \
  --output-json results/et_detailed_diagnosis_v2.json \
  --output-md results/et_detailed_diagnosis_v2.md
```

### 相关文件

- 诊断报告: `results/et_detailed_diagnosis_v2.md`
- 详细数据: `results/et_detailed_diagnosis_v2.json`
- ET配置: `config/nnmultihead/execution_archetypes.yaml`
