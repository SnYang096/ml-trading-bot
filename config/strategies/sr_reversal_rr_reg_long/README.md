# SR Reversal 策略特征说明

## 📋 特征发现流程

本策略的特征通过 **feature-group-search** 自动搜索得到，流程如下：

1. **Pool B 生成**：通过 `mlbot analyze factor-eval` 从大量候选特征中筛选出 IC/IR 表现好的特征
2. **特征组合搜索**：使用 `pipeline_sh_beam_sffs` 算法在 Pool B + Semantic Groups 中搜索最佳组合
3. **最终选择**：基于 `Sharpe_mean` 目标，从 Stage C 结果中选出最佳特征集

**训练结果**：
- Tag: `20260108_best_abc`
- Stage: `C` (最高优先级)
- Objective: `Sharpe_mean`
- 搜索算法: `pipeline_sh_beam_sffs`
- 结果文件: `results/feature_group_search/sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260108_best_abc_C/`
- **训练日期范围**: `2023-01-01` → `2025-12-31`
- **训练日期**: `2026-01-08` (feature-group-search 运行日期)
- **CV_mean**: `0.0648` (交叉验证均值，baseline)
- **Sharpe_mean**: `0.93 ± 0.44` (最终选中特征组合，5个 seeds)
  - Baseline: `0.25`
  - 最终结果: `0.93` (改进: +273.2% ⭐)
  - Return%_mean: `6.55%`
  - Trades_mean: `36.8`

---

## 🎯 最终特征列表（3个）

| 特征节点 | 输出列 | 重要性 | 语义说明 |
|---------|--------|------|---------|
| `poc_hal_features_close_f` | `poc`, `hal_high`, `hal_low`, `hal_mid` | ⭐⭐⭐⭐⭐ | **SR 结构核心**：POC/HAL 归一化距离（ATR 倍数），用于判断价格到支撑阻力的距离 |
| `atr_f` | `atr` | ⭐⭐⭐⭐⭐ | **基础必需**：平均真实波幅，用于标签生成、止损止盈、特征归一化 |
| `volume_profile_volatility_features_f` | `vp_width_ratio`, `vp_poc_deviation`, `vp_skewness`, `vp_entropy`, `vp_lv_ratio`, `vp_hv_ratio` | ⭐⭐⭐⭐ | **波动率预测**：从 Volume Profile 提取的 6 个波动率特征，Sharpe +18% |

---

## 🔍 核心特征详解

### 1. `poc_hal_features_close_f` ⭐⭐⭐⭐⭐

**计算逻辑**：
- POC (Point of Control): 成交量最大的价格水平
- HAL (High Activity Level): 高活跃度价格区间（上下边界）
- 归一化：`(level - close) / ATR`，范围通常 [-3, 3] ATR

**输出列**：
- `poc`: 当前价格到 POC 的归一化距离
- `hal_high`: 到 HAL 上边界的距离（阻力位）
- `hal_low`: 到 HAL 下边界的距离（支撑位）
- `hal_mid`: 到 HAL 中位的距离

**规则形成**：
- **反转信号**：价格接近 `hal_high`（如 < 0.5 ATR）→ 可能反转做空
- **反转信号**：价格接近 `hal_low`（如 < 0.5 ATR）→ 可能反转做多
- **距离判断**：`dist_to_nearest_sr` 由这些列计算，用于标签生成

**重要性**：
- 标签生成器**必需**（输出 `dist_to_nearest_sr`）
- SR 反转策略的**核心结构特征**
- 跨资产可比（ATR 归一化）

---

### 2. `atr_f` ⭐⭐⭐⭐⭐

**计算逻辑**：
- Average True Range (ATR)，默认 14 周期
- 衡量价格波动幅度

**规则形成**：
- **止损止盈**：`stop_loss = entry_price ± 2 * ATR`
- **仓位大小**：波动率大时减小仓位
- **特征归一化**：其他特征用 ATR 归一化，实现跨资产可比

**重要性**：
- 标签生成器**必需**（计算 R/R 比率）
- 回测系统**必需**（止损止盈计算）
- 所有策略的**基础设施**

---

### 3. `volume_profile_volatility_features_f` ⭐⭐⭐⭐

**计算逻辑**：
从 WPT 降噪后的 Volume Profile 提取 6 个波动率预测特征：

| 输出列 | 语义 | 规则形成 |
|--------|------|---------|
| `vp_width_ratio` | Value Area 宽度 / 全范围 | 越小 → 市场共识强 → 压缩后可能爆发 |
| `vp_poc_deviation` | 当前价格 vs POC 的标准化偏离 | 绝对值大 → 远离价值中枢 → 回归动力强或趋势加速 |
| `vp_skewness` | 成交量分布偏度 | 绝对值大 → 趋势强 → 波动可能持续 |
| `vp_entropy` | 信息熵（不确定性） | 越大 → 成交分散 → 多空博弈激烈 → 波动↑ |
| `vp_lv_ratio` | 低成交量区域比例（LVN） | 越大 → 存在真空带 → 价格进入会快速穿越 |
| `vp_hv_ratio` | 高成交量区域比例（HVN） | 越大 → 支撑/阻力密集 → 波动可能受限 |

**规则形成**：
- **波动率预测**：结合这些特征判断未来波动率变化
- **反转时机**：`vp_entropy` 高 + 价格接近 SR → 反转概率高
- **突破确认**：`vp_poc_deviation` 大 + `vp_skewness` 高 → 趋势可能持续

**重要性**：
- **Sharpe +18%**（feature-group-search 结果）
- 从 Pool B 中**唯一被选中**的特征组
- 提供波动率 regime 信息，帮助判断反转时机

---

## 📊 特征选择过程

### Base Features（基础必需，不参与搜索）
- `poc_hal_features_close_f` - 标签生成必需
- `atr_f` - 标签生成和回测必需

### Selected Groups（搜索选中）
- `poolb__volume_profile_volatility_features_f` - 从 Pool B 中选出

### 被移除的负面特征
- `trend_r2_50_f` - ❌ Sharpe -1.504，严重负面

---

## 🔗 相关文档

- 特征发现流程: `docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- 最终结果汇总: `docs/strategies/树模型策略结论TREE_STRATEGY_FINAL_FEATURES_CN.md`
- Volume Profile 特征: `docs/features/Volume_Profile_波动率特征说明.md`
- SR 距离特征: `docs/features/SR_DISTANCE_FEATURES.md`

---

## 📜 特征使用规则

以下规则从训练好的树模型中提取（使用 RuleFit），展示了特征如何组合形成交易信号：

| 规则条件 | 系数 | 支持度 | 说明 |
|---------|------|--------|------|
| `cvd_medium ≤ 8999.94189 **AND** cvd_medium > -49748.96094 **AND** vp_width_ratio > 0.56304` | 0.1029 | 7.69% | **正向信号**（系数越大，信号越强） |
| `vp_poc_deviation ≤ -0.02497 **AND** vp_entropy ≤ 0.96382 **AND** cvd_normalized > -0.057` | 0.0944 | 12.80% | **正向信号**（系数越大，信号越强） |
| `cvd_medium > -75436.22266 **AND** vp_hv_ratio > 0.09545 **AND** cvd_normalized ≤ -0.01097` | -0.0903 | 19.85% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium > 11971.29834 **AND** vp_poc_deviation ≤ 0.69374 **AND** cvd_normalized ≤ 0.02997` | 0.0782 | 7.77% | **正向信号**（系数越大，信号越强） |
| `cvd_medium ≤ 4543.83887 **AND** vp_width_ratio ≤ 0.61717` | -0.0778 | 80.00% | **负向信号**（系数绝对值越大，抑制越强） |
| `hal_mid > -6.51279 **AND** cvd_change_1 > -2870.44604` | 0.0760 | 89.79% | **正向信号**（系数越大，信号越强） |
| `vp_poc_deviation > 0.02234 **AND** cvd_long ≤ -34349.6875` | -0.0683 | 6.68% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium ≤ 11674.89111 **AND** cvd_medium > -615.996 **AND** vp_skewness ≤ 0.92526` | -0.0641 | 13.25% | **负向信号**（系数绝对值越大，抑制越强） |
| `vp_poc_deviation ≤ 0.6681 **AND** cvd_long > -34890.44531` | 0.0640 | 57.75% | **正向信号**（系数越大，信号越强） |
| `cvd_change_1 ≤ 8066.396 **AND** cvd_normalized > -0.02076` | 0.0581 | 62.21% | **正向信号**（系数越大，信号越强） |
| `cvd_medium > -67646.67969 **AND** vp_width_ratio ≤ 0.61844 **AND** cvd_long ≤ -34225.73633` | -0.0481 | 37.04% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium > -66983.67578 **AND** cvd_long ≤ -48113.58984` | -0.0458 | 24.20% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium > -19027.23242 **AND** vp_width_ratio ≤ 0.63105 **AND** cvd_long ≤ -3499.96057` | -0.0455 | 39.51% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium ≤ -23007.91016 **AND** cvd_long > -48047.7168` | 0.0432 | 18.54% | **正向信号**（系数越大，信号越强） |
| `cvd_long > -35087.99219 **AND** poc > -8.19994` | 0.0354 | 59.81% | **正向信号**（系数越大，信号越强） |
| `vp_poc_deviation > 0.24886` | -0.0290 | 14.75% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_short ≤ -6198.58154 **AND** vp_hv_ratio ≤ 0.37413 **AND** cvd_change_5 > 684.54749` | 0.0279 | 7.99% | **正向信号**（系数越大，信号越强） |
| `cvd_short ≤ 6376.29663 **AND** cvd_medium > -25854.91406 **AND** cvd_long ≤ -34969.45312` | -0.0233 | 12.80% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 ≤ 6625.59351 **AND** cvd_change_1 > -3104.927` | 0.0226 | 91.97% | **正向信号**（系数越大，信号越强） |
| `cvd_medium ≤ -20465.5 **AND** cvd_long ≤ -47972.68164 **AND** cvd_long > -67054.47266` | -0.0216 | 5.70% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long ≤ -36000.5 **AND** cvd_long > -93934.32031` | -0.0160 | 29.16% | **负向信号**（系数绝对值越大，抑制越强） |
| `vp_width_ratio > 0.49425 **AND** hal_low > -2.14054` | 0.0115 | 27.02% | **正向信号**（系数越大，信号越强） |
| `cvd_change_20 ≤ -15376.45996` | 0.0114 | 20.98% | **正向信号**（系数越大，信号越强） |
| `cvd_change_20 > -15376.45996` | -0.0073 | 79.02% | **负向信号**（系数绝对值越大，抑制越强） |

**说明**：
- **系数**：规则对预测的贡献度，绝对值越大影响越大
- **支持度**：规则在训练数据中的覆盖比例（满足条件的样本占比）
- **规则条件**：多个特征条件的组合，满足所有条件时触发

**模型来源**：`results/rules_export/tree_best4/sr_reversal_rr_reg_long__20260108_best_abc__C__full/sr_reversal_rr_reg_long__imodels_rules/rules_regression.md`

> 💡 **提示**：这些规则是从树模型中提取的简化版本，实际模型可能包含更复杂的非线性组合。
