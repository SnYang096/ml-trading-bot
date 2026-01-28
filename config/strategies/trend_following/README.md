# Trend Following 策略特征说明

## 📋 特征发现流程

本策略的特征通过 **feature-group-search** 自动搜索得到，流程如下：

1. **Pool B 生成**：通过 `mlbot analyze factor-eval` 从大量候选特征中筛选出 IC/IR 表现好的特征
2. **特征组合搜索**：使用 `pipeline_sh_beam_sffs` 算法在 Pool B + Semantic Groups 中搜索最佳组合
3. **最终选择**：基于 `Sharpe_mean` 目标，从 Stage C 结果中选出最佳特征集

**训练结果**：
- Tag: `20260110_tf_fast_blacklist`
- Stage: `C` (最高优先级)
- Objective: `Sharpe_mean`
- 搜索算法: `pipeline_sh_beam_sffs`
- 结果文件: `results/feature_group_search/trend_following_pipeline_poolb_semantic_20260110_tf_fast_blacklist_C/`
- **训练日期范围**: `2023-01-01` → `2025-12-31`
- **训练日期**: `2026-01-10` (feature-group-search 运行日期)
- **CV_mean**: `0.0956` (交叉验证均值，baseline)
- **Sharpe_mean**: `-0.07` (baseline，最终结果待确认)
  - ⚠️ **注意**: 在 3 年长周期（2023-2025）上表现较差，可能受市场环境影响

---

## 🎯 最终特征列表（10个）

| 特征节点 | 输出列 | 重要性 | 语义说明 |
|---------|--------|------|---------|
| `atr_f` | `atr` | ⭐⭐⭐⭐⭐ | **基础必需**：平均真实波幅 |
| `macd_f` | `macd`, `macd_signal`, `macd_hist` | ⭐⭐⭐⭐ | **趋势动量**：MACD 指标，判断趋势方向和强度 |
| `rsi_f` | `rsi` | ⭐⭐⭐⭐ | **超买超卖**：RSI 指标，判断趋势是否过度 |
| `trend_r2_20_f` | `trend_r2_20` | ⭐⭐⭐⭐ | **趋势强度**：20 周期价格回归 R² |
| `bb_width_f` | `bb_width` | ⭐⭐⭐ | **波动率**：布林带宽度，判断市场波动 |
| `wick_ratios_f` | `wick_ratios_*` | ⭐⭐⭐ | **K线形态**：上下影线比率，判断多空力量 |
| `volume_ratio_f` | `volume_ratio` | ⭐⭐⭐ | **成交量**：当前成交量 vs 历史均值 |
| `trade_cluster_scene_semantic_scores_f` | `trade_cluster_*_score` (4列) | ⭐⭐⭐⭐ | **交易聚集场景**：压缩/点火/吸收/衰竭场景评分 |
| `wpt_cvd_fluctuation_f` | `wpt_cvd_fluctuation` | ⭐⭐⭐ | **资金流波动**：WPT 去趋势后的 CVD 波动 |
| `funding_scene_semantic_scores_f` | `funding_*_score` (4列) | ⭐⭐⭐ | **资金费率场景**：压缩/点火/吸收/衰竭场景评分 |

---

## 🔍 核心特征详解

### 1. `atr_f` ⭐⭐⭐⭐⭐

**计算逻辑**：
- Average True Range (ATR)，默认 14 周期
- 衡量价格波动幅度

**规则形成**：
- **止损止盈**：`stop_loss = entry_price ± 2 * ATR`
- **仓位大小**：波动率大时减小仓位
- **特征归一化**：其他特征用 ATR 归一化

**重要性**：
- 标签生成器**必需**
- 回测系统**必需**
- 所有策略的**基础设施**

---

### 2. `macd_f` ⭐⭐⭐⭐

**计算逻辑**：
- MACD (Moving Average Convergence Divergence)
- 输出：`macd`, `macd_signal`, `macd_hist`

**规则形成**：
- **趋势方向**：`macd_hist > 0` → 上升趋势
- **趋势强度**：`macd_hist` 绝对值大 → 趋势强
- **趋势确认**：`macd` 上穿 `macd_signal` → 买入信号

**重要性**：
- 从 `kline_core` 语义组中选出
- 提供**趋势动量**的关键信息
- 经典趋势跟踪指标

---

### 3. `rsi_f` ⭐⭐⭐⭐

**计算逻辑**：
- RSI (Relative Strength Index)，默认 14 周期
- 范围 [0, 100]

**规则形成**：
- **超买超卖**：`rsi > 70` → 超买，可能回调
- **趋势确认**：`rsi` 在 50 附近 → 趋势健康
- **反转信号**：`rsi` 从超买区回落 → 可能反转

**重要性**：
- 从 `kline_core` 语义组中选出
- 提供**趋势健康度**信息
- 帮助过滤过度延伸的趋势

---

### 4. `trend_r2_20_f` ⭐⭐⭐⭐

**计算逻辑**：
- 20 周期价格回归的 R² 值
- 范围 [0, 1]，越大表示趋势越强

**规则形成**：
- **趋势确认**：`trend_r2_20 > 0.7` → 强趋势，适合趋势跟踪
- **震荡过滤**：`trend_r2_20 < 0.3` → 震荡市场，不适合趋势跟踪
- **趋势持续性**：R² 高 → 趋势可能持续

**重要性**：
- 从 `kline_core` 语义组中选出
- 提供**趋势强度**的关键信息
- 帮助判断是否适合趋势跟踪策略

---

### 5. `trade_cluster_scene_semantic_scores_f` ⭐⭐⭐⭐

**计算逻辑**：
基于交易聚集特征，输出 4 个场景语义评分：

| 输出列 | 语义 | 规则形成 |
|--------|------|---------|
| `trade_cluster_compression_score` | 压缩场景 | 交易聚集 + 压缩 regime → 压缩场景 |
| `trade_cluster_ignition_score` | 点火场景 | 交易聚集 + 高流量 + 低假突破风险 → 点火场景 |
| `trade_cluster_absorption_scene_score` | 吸收场景 | 交易聚集 + 低回抽 + 趋势 regime → 吸收场景 |
| `trade_cluster_exhaustion_scene_score` | 衰竭场景 | 交易聚集 + 高回抽 + 趋势结束 → 衰竭场景 |

**规则形成**：
- **趋势确认**：`ignition_score` 高 → 趋势启动，适合跟随
- **趋势持续**：`absorption_score` 高 → 趋势持续，适合持有
- **趋势结束**：`exhaustion_score` 高 → 趋势可能结束，考虑退出

**重要性**：
- 从 `trade_cluster_scene` 语义组中选出
- 提供**市场场景**的高级语义信息
- 帮助判断趋势的**生命周期阶段**

---

### 6. `funding_scene_semantic_scores_f` ⭐⭐⭐

**计算逻辑**：
基于资金费率（Funding Rate），输出 4 个场景语义评分：

| 输出列 | 语义 | 规则形成 |
|--------|------|---------|
| `funding_compression_score` | 压缩场景 | 资金费率压力 + 压缩 regime → 压缩场景 |
| `funding_ignition_score` | 点火场景 | 资金费率压力 + 趋势 regime → 点火场景 |
| `funding_absorption_score` | 吸收场景 | 资金费率压力 + 压缩 + 趋势 → 拥挤延续 |
| `funding_exhaustion_scene_score` | 衰竭场景 | 资金费率压力 + 无趋势 → 衰竭场景 |

**规则形成**：
- **趋势确认**：`ignition_score` 高 → 资金费率支持趋势，适合跟随
- **拥挤警告**：`absorption_score` 高 → 市场拥挤，趋势可能反转
- **趋势结束**：`exhaustion_score` 高 → 资金费率不支持趋势，考虑退出

**重要性**：
- 从 `funding_scene` 语义组中选出
- 提供**市场情绪**信息（资金费率反映市场情绪）
- 帮助判断趋势的**可持续性**

---

### 7. `wpt_cvd_fluctuation_f` ⭐⭐⭐

**计算逻辑**：
- WPT (Wavelet Packet Transform) 去趋势后的 CVD 波动
- 识别资金流的波动模式

**规则形成**：
- **趋势确认**：CVD 波动小 → 资金流稳定，趋势可能持续
- **趋势结束**：CVD 波动大 → 资金流不稳定，趋势可能结束

**重要性**：
- 从 Pool B 中选出
- 提供**资金流稳定性**信息
- 帮助判断趋势的**质量**

---

## 📊 特征选择过程

### Base Features（基础必需，不参与搜索）
- `atr_f` - 标签生成必需

### Selected Groups（搜索选中）
- `kline_core` - K线核心特征（macd, rsi, trend_r2_20, bb_width, wick_ratios, volume_ratio）
- `trade_cluster_scene` - 交易聚集场景语义
- `poolb__wpt_cvd_fluctuation_f` - WPT CVD 波动
- `funding_scene` - 资金费率场景语义
- `poolb__atr_f` - ATR 特征

### Feature Blacklist（被排除的特征）
- `order_flow_all_features_f` - 订单流全特征（计算成本高）
- `footprint_basic_f` - 基础 Footprint（计算成本高）
- `dtw_features_reversal_f`, `dtw_features_trend_f` - DTW 特征（计算成本高）
- `spectrum_features_f` - 频谱特征（计算成本高）

---

## 🔗 相关文档

- 特征发现流程: `docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- 最终结果汇总: `docs/strategies/树模型策略结论TREE_STRATEGY_FINAL_FEATURES_CN.md`
- 语义特征: `docs/strategies/SEMANTIC_FEATURES_4_SCENARIOS.md`

---

## 📜 特征使用规则

以下规则从训练好的树模型中提取（使用 RuleFit），展示了特征如何组合形成交易信号：

| 规则条件 | 系数 | 支持度 | 说明 |
|---------|------|--------|------|
| `bb_width_normalized ≤ 4.85935 **AND** cvd_long ≤ -34895.86719 **AND** cvd_long > -64485.69531` | -0.0800 | 14.40% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium ≤ -6969.48047 **AND** rsi > 34.23576 **AND** cvd_long ≤ -45804.36914` | -0.0734 | 20.39% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium ≤ -4292.82739 **AND** cvd_long > -22683.5293 **AND** macd_signal ≤ 0.41226` | 0.0567 | 14.58% | **正向信号**（系数越大，信号越强） |
| `cvd_long ≤ -5701.96338 **AND** wpt_cvd_fluctuation > 8.74941` | -0.0560 | 15.77% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long ≤ -13344.65381 **AND** macd > 0.15551 **AND** macd_histogram ≤ 0.34473` | -0.0518 | 21.69% | **负向信号**（系数绝对值越大，抑制越强） |
| `bb_width_normalized > 2.70411 **AND** bb_position > 0.23145 **AND** macd_histogram ≤ 0.04866` | -0.0508 | 29.30% | **负向信号**（系数绝对值越大，抑制越强） |
| `rsi > 57.26204 **AND** bb_width_normalized ≤ 6.8121` | -0.0475 | 27.15% | **负向信号**（系数绝对值越大，抑制越强） |
| `funding_exhaustion_scene_score > 0.00204 **AND** macd_signal ≤ 0.42512` | 0.0460 | 52.61% | **正向信号**（系数越大，信号越强） |
| `cvd_medium ≤ -8285.74658 **AND** rsi ≤ 50.00967 **AND** funding_ignition_score > 0.00042` | 0.0353 | 26.65% | **正向信号**（系数越大，信号越强） |
| `bb_position ≤ 0.72625 **AND** wpt_cvd_fluctuation > -256.13433 **AND** macd_histogram ≤ 0.37125` | 0.0298 | 52.18% | **正向信号**（系数越大，信号越强） |
| `bb_width_normalized ≤ 7.56461 **AND** funding_absorption_score > 0.08091 **AND** macd > -0.9681` | -0.0296 | 42.99% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long > -2602.15308 **AND** macd_signal > 0.15446` | 0.0289 | 10.25% | **正向信号**（系数越大，信号越强） |
| `bb_position ≤ 0.46124 **AND** volume_ratio > 0.65458 **AND** cvd_normalized ≤ 0.03181` | 0.0199 | 21.69% | **正向信号**（系数越大，信号越强） |
| `cvd_short ≤ 10990.15918 **AND** rsi > 55.41221 **AND** funding_absorption_score > 0.06898` | -0.0166 | 16.41% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_medium ≤ -9895.55273 **AND** macd ≤ 0.15551 **AND** macd_histogram ≤ 0.34473` | 0.0163 | 38.35% | **正向信号**（系数越大，信号越强） |
| `cvd_medium ≤ -6969.48047 **AND** cvd_long > -45804.36914` | 0.0136 | 35.63% | **正向信号**（系数越大，信号越强） |
| `cvd_medium ≤ -6597.92505 **AND** cvd_long ≤ -35290.82617 **AND** macd_signal ≤ 0.17797` | 0.0126 | 24.79% | **正向信号**（系数越大，信号越强） |
| `cvd_long ≤ -29002.00684 **AND** macd_signal > 0.04227 **AND** macd_histogram ≤ 0.24511` | -0.0121 | 13.56% | **负向信号**（系数绝对值越大，抑制越强） |
| `rsi > 39.15442 **AND** cvd_long ≤ -25036.38867 **AND** cvd_long > -65066.31836` | -0.0083 | 25.18% | **负向信号**（系数绝对值越大，抑制越强） |
| `wpt_cvd_fluctuation > -895.32214 **AND** cvd_change_5 ≤ -1939.16302 **AND** macd_signal ≤ 0.15446` | 0.0079 | 19.47% | **正向信号**（系数越大，信号越强） |
| `bb_width_normalized ≤ 6.81965 **AND** funding_compression_score ≤ 0.14839` | -0.0072 | 52.89% | **负向信号**（系数绝对值越大，抑制越强） |
| `cvd_long ≤ -28700.27539 **AND** macd > -0.92225` | -0.0054 | 42.75% | **负向信号**（系数绝对值越大，抑制越强） |
| `funding_absorption_score > 0.00036 **AND** macd_signal ≤ 0.46216` | 0.0043 | 66.58% | **正向信号**（系数越大，信号越强） |
| `bb_position > 0.11578 **AND** macd_histogram > 0.25591` | 0.0042 | 9.65% | **正向信号**（系数越大，信号越强） |
| `bb_position > 0.25907 **AND** cvd_long > -27744.94043 **AND** funding_absorption_score ≤ 0.08791` | 0.0019 | 20.88% | **正向信号**（系数越大，信号越强） |

**说明**：
- **系数**：规则对预测的贡献度，绝对值越大影响越大
- **支持度**：规则在训练数据中的覆盖比例（满足条件的样本占比）
- **规则条件**：多个特征条件的组合，满足所有条件时触发

**模型来源**：`results/rules_export/tree_best4/trend_following__20260110_tf_fast_blacklist__C__full/trend_following__imodels_rules/rules_regression.md`

> 💡 **提示**：这些规则是从树模型中提取的简化版本，实际模型可能包含更复杂的非线性组合。
