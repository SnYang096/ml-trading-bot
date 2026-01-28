# Compression Breakout 策略特征说明

## 📋 特征发现流程

本策略的特征通过 **feature-group-search** 自动搜索得到，流程如下：

1. **Pool B 生成**：通过 `mlbot analyze factor-eval` 从大量候选特征中筛选出 IC/IR 表现好的特征
2. **特征组合搜索**：使用 `pipeline_sh_beam_sffs` 算法在 Pool B + Semantic Groups 中搜索最佳组合
3. **最终选择**：基于 `CV_mean` 目标，从 Stage B 结果中选出最佳特征集

**训练结果**：
- Tag: `20260108_best_abc`
- Stage: `B`
- Objective: `CV_mean` (Coefficient of Variation，变异系数)
- 搜索算法: `pipeline_sh_beam_sffs`
- 结果文件: `results/feature_group_search/compression_breakout_pipeline_poolb_semantic_20260108_best_abc_B/`
- **训练日期范围**: `2023-01-01` → `2025-12-31`
- **训练日期**: `2026-01-08` (feature-group-search 运行日期)
- **CV_mean**: `0.8503` (交叉验证均值，baseline)
- **Sharpe_mean**: `-1.07 ± 0.83` (最终选中特征组合，3个 seeds)
  - Baseline: `-1.05`
  - 最终结果: `-1.07` (改进: -1.9%)
  - ⚠️ **注意**: 在 3 年长周期（2023-2025）上表现较差，可能受市场环境影响

---

## 🎯 最终特征列表（5个）

| 特征节点 | 输出列 | 重要性 | 语义说明 |
|---------|--------|------|---------|
| `compression_duration_f` | `compression_duration` | ⭐⭐⭐⭐⭐ | **压缩持续时间**：基于 ATR percentile，连续低波动 bar 数 |
| `atr_f` | `atr` | ⭐⭐⭐⭐⭐ | **基础必需**：平均真实波幅，用于标签生成、止损止盈 |
| `volume_ratio_f` | `volume_ratio` | ⭐⭐⭐⭐ | **成交量比率**：当前成交量 vs 历史均值，判断突破是否放量 |
| `liquidity_void_f` | `liquidity_void_*` (6列) | ⭐⭐⭐⭐ | **流动性真空**：检测低成交量区域，价格进入会快速穿越 |
| `trend_r2_20_f` | `trend_r2_20` | ⭐⭐⭐ | **趋势强度**：20 周期价格回归 R²，判断趋势是否存在 |

---

## 🔍 核心特征详解

### 1. `compression_duration_f` ⭐⭐⭐⭐⭐

**计算逻辑**：
- 基于 ATR percentile，统计连续低波动 bar 的数量
- 压缩阈值：ATR < 20% 分位数
- 归一化到 [0, 1]

**规则形成**：
- **压缩时间越长** → 突破概率越高
- **压缩后爆发**：`compression_duration > 0.6` → 可能即将突破
- **假突破过滤**：压缩时间太短（< 0.3）→ 可能是假突破

**重要性**：
- 标签生成器**必需**（压缩突破策略的核心）
- 从 `kline_core` 语义组中选出
- 提供**压缩强度**的关键信息

---

### 2. `atr_f` ⭐⭐⭐⭐⭐

**计算逻辑**：
- Average True Range (ATR)，默认 14 周期
- 衡量价格波动幅度

**规则形成**：
- **压缩判断**：ATR 低 → 市场压缩
- **突破确认**：突破时 ATR 上升 → 真突破
- **止损止盈**：`stop_loss = entry_price ± 2 * ATR`

**重要性**：
- 标签生成器**必需**（计算压缩持续时间）
- 回测系统**必需**（止损止盈计算）

---

### 3. `volume_ratio_f` ⭐⭐⭐⭐

**计算逻辑**：
- 当前成交量 / 历史平均成交量
- 归一化到合理范围

**规则形成**：
- **突破确认**：`volume_ratio > 1.5` → 放量突破，更可靠
- **假突破过滤**：`volume_ratio < 0.8` → 缩量突破，可能是假突破
- **压缩确认**：`volume_ratio < 0.6` → 缩量压缩，压缩质量高

**重要性**：
- 从 `kline_core` 语义组中选出
- 提供**突破质量**的关键信息
- 与 `compression_duration` 配合，判断压缩是否成熟

---

### 4. `liquidity_void_f` ⭐⭐⭐⭐

**计算逻辑**：
检测低成交量区域（Liquidity Void），输出 6 个特征：

| 输出列 | 语义 | 规则形成 |
|--------|------|---------|
| `liquidity_void_detected` | 是否检测到流动性真空 | 1.0 = 检测到，0.0 = 未检测到 |
| `liquidity_void_speed` | 价格穿越速度 | 越大 → 穿越越快，突破越强 |
| `liquidity_void_price_impact` | 价格冲击 | 越大 → 冲击越大，突破越有效 |
| `liquidity_void_retracement` | 回抽幅度 | 越小 → 回抽越小，突破越可靠 |
| `liquidity_void_false_breakout_risk` | 假突破风险 | 越小 → 风险越低 |
| `liquidity_void_volume_ratio` | 成交量比率 | 越小 → 真空带越明显 |

**规则形成**：
- **突破确认**：`liquidity_void_detected = 1` + `speed > 2.0` + `false_breakout_risk < 0.3` → 真突破
- **假突破过滤**：`false_breakout_risk > 0.5` → 可能是假突破
- **突破质量**：`price_impact` 大 + `retracement` 小 → 突破质量高

**重要性**：
- 从 Pool B 中选出
- 提供**突破有效性**的关键信息
- 帮助区分真突破和假突破

---

### 5. `trend_r2_20_f` ⭐⭐⭐

**计算逻辑**：
- 20 周期价格回归的 R² 值
- 范围 [0, 1]，越大表示趋势越强

**规则形成**：
- **趋势确认**：`trend_r2_20 > 0.7` → 强趋势，突破可能持续
- **震荡过滤**：`trend_r2_20 < 0.3` → 震荡市场，突破可能失败
- **方向判断**：结合价格方向，判断突破方向是否与趋势一致

**重要性**：
- 从 Pool B 中选出
- 提供**市场状态**信息（趋势 vs 震荡）
- 帮助判断突破是否可持续

---

## 📊 特征选择过程

### Base Features（基础必需，不参与搜索）
- `compression_duration_f` - 标签生成必需
- `atr_f` - 标签生成和回测必需

### Selected Groups（搜索选中）
- `kline_core__volume_ratio_f__volume_ratio_f` - K线核心特征
- `poolb__liquidity_void_f` - 流动性真空特征
- `poolb__trend_r2_20_f` - 趋势强度特征

### Invert Features（需要取反的输出列）
以下列在某些情况下需要取反（见配置文件）：
- `dtw_bull_flag_dist_w20`, `dtw_bull_flag_dist_w50` - DTW 模式距离
- `hurst_cvd_rolling`, `hurst_price_rolling` - Hurst 指数
- `wpt_price_energy_high_ratio`, `wpt_price_energy_mid_ratio` - WPT 能量比
- `trade_cluster_*` 系列 - 交易聚集特征

---

## 🔗 相关文档

- 特征发现流程: `docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- 最终结果汇总: `docs/strategies/树模型策略结论TREE_STRATEGY_FINAL_FEATURES_CN.md`
- 流动性真空: `docs/features/liquidity_void_price_impact_guide.md`

---

## 📜 特征使用规则

以下规则从训练好的树模型中提取（使用 RuleFit），展示了特征如何组合形成交易信号：

| 规则条件 | 系数 | 支持度 | 说明 |
|---------|------|--------|------|
| `cvd_change_5 > 4805.8501 **AND** volume_ratio > 1.87706` | 0.1443 | 2.96% | **正向信号**（系数越大，信号越强） |
| `cvd_short > 270.74957 **AND** cvd_change_1 > 3313.67749` | 0.1056 | 3.25% | **正向信号**（系数越大，信号越强） |
| `liquidity_void_retracement ≤ 0.03782 **AND** cvd_change_5 ≤ -848.44751 **AND** volume_ratio > 1.65399` | -0.0762 | 8.14% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_5 > 1236.80804 **AND** volume_ratio > 0.98018` | 0.0477 | 12.67% | **正向信号**（系数越大，信号越强） |
| `cvd_change_5 > 6132.74146 **AND** volume_ratio > 1.1774` | 0.0470 | 4.61% | **正向信号**（系数越大，信号越强） |
| `cvd_change_20 ≤ -7214.31641` | -0.0382 | 37.76% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 ≤ 1013.05399 **AND** volume_ratio > 2.00096` | -0.0270 | 7.20% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 ≤ -4757.24194 **AND** cvd_normalized ≤ -0.04527` | -0.0225 | 2.47% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_5 ≤ 3467.99097 **AND** volume_ratio > 1.07804` | -0.0193 | 25.79% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long ≤ -1624.883` | -0.0147 | 83.30% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_short ≤ -2261.68097 **AND** cvd_change_1 ≤ -4123.4375` | -0.0121 | 3.58% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_short > -11659.47998 **AND** cvd_change_1 > 863.28 **AND** trend_r2_20 ≤ 0.60629` | 0.0112 | 11.27% | **正向信号**（系数越大，信号越强） |
| `cvd_change_5 ≤ 1149.36853` | -0.0084 | 70.71% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 ≤ -1836.40698 **AND** cvd_change_5 ≤ -4901.70728` | -0.0079 | 7.28% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_1 > 2157.96509` | 0.0060 | 8.72% | **正向信号**（系数越大，信号越强） |
| `cvd_change_1 > 933.44901 **AND** volume_ratio > 1.05656` | 0.0037 | 10.53% | **正向信号**（系数越大，信号越强） |
| `cvd_change_1 ≤ 2157.96509` | -0.0016 | 91.28% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_change_20 > -7214.31641` | 0.0011 | 62.24% | **正向信号**（系数越大，信号越强） |
| `cvd_change_1 ≤ 2177.32141 **AND** cvd_change_20 ≤ -1710.88004 **AND** volume_ratio > 2.49885` | -0.0011 | 3.17% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long > -1624.883` | 0.0004 | 16.70% | **正向信号**（系数越大，信号越强） |

**说明**：
- **系数**：规则对预测的贡献度，绝对值越大影响越大
- **支持度**：规则在训练数据中的覆盖比例（满足条件的样本占比）
- **规则条件**：多个特征条件的组合，满足所有条件时触发

**模型来源**：`results/rules_export/tree_best4/compression_breakout__20260108_best_abc__B__lite/compression_breakout__imodels_rules/rules_regression.md`

> 💡 **提示**：这些规则是从树模型中提取的简化版本，实际模型可能包含更复杂的非线性组合。
