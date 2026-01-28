# SR Breakout 策略特征说明

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
- 结果文件: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic_20260108_best_abc_C/`
- **训练日期范围**: `2023-01-01` → `2025-12-31`
- **训练日期**: `2026-01-08` (feature-group-search 运行日期)
- **CV_mean**: `0.0379` (交叉验证均值，baseline)
- **Sharpe_mean**: `0.82 ± 0.93` (最终选中特征组合，5个 seeds)
  - Baseline: `0.75`
  - 最终结果: `0.82` (改进: +8.5%)
  - Return%_mean: `7.49%`
  - Trades_mean: `101.0`

---

## 🎯 最终特征列表（3个）

| 特征节点 | 输出列 | 重要性 | 语义说明 |
|---------|--------|------|---------|
| `atr_f` | `atr` | ⭐⭐⭐⭐⭐ | **基础必需**：平均真实波幅，用于标签生成、止损止盈、特征归一化 |
| `poc_hal_features_close_f` | `poc`, `hal_high`, `hal_low`, `hal_mid` | ⭐⭐⭐⭐⭐ | **SR 结构核心**：POC/HAL 归一化距离（ATR 倍数），用于判断价格到支撑阻力的距离 |
| `hal_low` | `hal_low` | ⭐⭐⭐⭐ | **突破方向判断**：HAL 下边界（支撑位），用于判断向下突破 |

---

## 🔍 核心特征详解

### 1. `atr_f` ⭐⭐⭐⭐⭐

**计算逻辑**：
- Average True Range (ATR)，默认 14 周期
- 衡量价格波动幅度

**规则形成**：
- **突破阈值**：`signal_threshold_atr` = 1.5 * ATR（判断是否有效突破）
- **止损止盈**：`stop_loss = entry_price ± 2 * ATR`
- **特征归一化**：其他特征用 ATR 归一化，实现跨资产可比

**重要性**：
- 标签生成器**必需**（计算 `signal_threshold_atr`）
- 回测系统**必需**（止损止盈计算）
- 所有策略的**基础设施**

---

### 2. `poc_hal_features_close_f` ⭐⭐⭐⭐⭐

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
- **突破判断**：价格突破 `hal_high` 或 `hal_low` → 可能形成趋势
- **距离判断**：`dist_to_nearest_sr` 由这些列计算，用于标签生成
- **突破质量**：结合 `hal_low` 判断突破是否有效

**重要性**：
- 标签生成器**必需**（输出 `dist_to_nearest_sr`）
- SR 突破策略的**核心结构特征**
- 跨资产可比（ATR 归一化）

---

### 3. `hal_low` ⭐⭐⭐⭐

**计算逻辑**：
- HAL 下边界（支撑位）的归一化距离
- 从 `poc_hal_features_close_f` 输出列中单独提取

**规则形成**：
- **向下突破**：价格跌破 `hal_low`（如 < -1.5 ATR）→ 向下突破信号
- **突破确认**：结合 `atr_f` 判断突破幅度是否足够（> 1.5 ATR）
- **反转风险**：如果突破后快速回抽 → 可能是假突破

**重要性**：
- 从 Pool B 中**唯一被选中**的列级特征
- 提供**突破方向判断**的关键信息
- 与 `poc_hal_features_close_f` 配合，形成完整的 SR 突破逻辑

**Invert Features（需要取反）**：
- `hal_mid` - 在某些情况下需要取反
- `trade_cluster_compression_score` - 压缩场景需要取反
- `trade_cluster_exhaustion_scene_score` - 衰竭场景需要取反

---

## 📊 特征选择过程

### Base Features（基础必需，不参与搜索）
- `atr_f` - 标签生成必需
- `poc_hal_features_close_f` - SR 结构判断必需

### Selected Groups（搜索选中）
- `poolb__poc_hal_features_f__hal_low` - 从 Pool B 中选出的列级特征

---

## 🔗 相关文档

- 特征发现流程: `docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- 最终结果汇总: `docs/strategies/树模型策略结论TREE_STRATEGY_FINAL_FEATURES_CN.md`
- SR 距离特征: `docs/features/SR_DISTANCE_FEATURES.md`

---

## 📜 特征使用规则

以下规则从训练好的树模型中提取（使用 RuleFit），展示了特征如何组合形成交易信号：

| 规则条件 | 系数 | 支持度 | 说明 |
|---------|------|--------|------|
| `cvd_long ≤ -10915.60059 **AND** cvd_long > -16221.68994` | 0.1519 | 6.38% | **正向信号**（系数越大，信号越强） |
| `cvd_medium > -23675.44336 **AND** cvd_long ≤ 21475.84668 **AND** cvd_change_5 > 3452.54102` | 0.1242 | 10.21% | **正向信号**（系数越大，信号越强） |
| `cvd_long > -111727.97656 **AND** cvd_change_5 > -10850.48584 **AND** cvd_normalized > -0.04937` | 0.0899 | 76.62% | **正向信号**（系数越大，信号越强） |
| `cvd_normalized > 0.04696` | 0.0818 | 12.08% | **正向信号**（系数越大，信号越强） |
| `cvd_short > 1008.9425 **AND** cvd_normalized ≤ -0.0469 **AND** hal_low > -2.2704` | 0.0749 | 3.56% | **正向信号**（系数越大，信号越强） |
| `cvd_medium > -7836.81958 **AND** cvd_change_5 ≤ -1782.76099` | 0.0646 | 12.16% | **正向信号**（系数越大，信号越强） |
| `cvd_long > -10915.60059 **AND** cvd_change_1 > -3940.22009` | -0.0570 | 26.15% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_5 > -9455.51709 **AND** cvd_normalized ≤ 0.0096` | -0.0541 | 55.95% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long ≤ 1330.38049 **AND** cvd_long > -2660.05054` | -0.0525 | 4.47% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_short ≤ -18058.25684 **AND** hal_high ≤ 1.31493` | 0.0426 | 15.01% | **正向信号**（系数越大，信号越强） |
| `cvd_short > -6753.76367 **AND** cvd_medium ≤ 24045.5127 **AND** cvd_normalized ≤ -0.05233` | 0.0369 | 6.60% | **正向信号**（系数越大，信号越强） |
| `cvd_short > 3367.30847` | 0.0329 | 26.64% | **正向信号**（系数越大，信号越强） |
| `cvd_long ≤ -77841.08594` | -0.0323 | 13.73% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium > 24045.5127` | -0.0298 | 5.18% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_5 ≤ -1257.42004 **AND** cvd_normalized ≤ 0.10538 **AND** hal_high ≤ 1.11781` | 0.0263 | 41.69% | **正向信号**（系数越大，信号越强） |
| `cvd_long ≤ -39967.81055 **AND** cvd_normalized > -0.0692` | 0.0254 | 29.72% | **正向信号**（系数越大，信号越强） |
| `cvd_long > -77144.05859 **AND** cvd_change_5 ≤ -9455.51709 **AND** cvd_normalized ≤ -0.00379` | 0.0235 | 5.10% | **正向信号**（系数越大，信号越强） |
| `cvd_change_5 ≤ 3965.74792` | -0.0235 | 86.75% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 > 1798.14252 **AND** cvd_normalized > 0.03102` | 0.0204 | 9.08% | **正向信号**（系数越大，信号越强） |
| `cvd_short ≤ 11876.25195 **AND** cvd_change_20 > -46515.81641` | -0.0089 | 89.01% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_5 > 3965.74792` | 0.0075 | 13.25% | **正向信号**（系数越大，信号越强） |
| `poc ≤ 0.4897 **AND** hal_high > 1.16959` | -0.0070 | 3.71% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 ≤ 3081.47546 **AND** cvd_change_20 > -13770.68164` | -0.0040 | 70.81% | **负向信号**（系数绝对值越大，抑制越强） |

**说明**：
- **系数**：规则对预测的贡献度，绝对值越大影响越大
- **支持度**：规则在训练数据中的覆盖比例（满足条件的样本占比）
- **规则条件**：多个特征条件的组合，满足所有条件时触发

**模型来源**：`results/rules_export/tree_best4/sr_breakout__20260108_best_abc__C__full/sr_breakout__imodels_rules/rules_regression.md`

> 💡 **提示**：这些规则是从树模型中提取的简化版本，实际模型可能包含更复杂的非线性组合。
