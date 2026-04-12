# 各策略最佳特征列推荐

> **生成时间**: 2026-01-01
> **数据来源**: feature-group-search 结果 + 列级别 importance 分析
> **方法**: Greedy Forward Selection + LightGBM Gain Importance

> ✅ **统一入口**：该文档的“列级别建议”已被纳入统一索引，便于和最新 rerun 的“节点/组级推荐 YAML”放在同一处查看：
> `docs/architecture/树模型策略report/FEATURE_SELECTION_REPORTS.md`

---

## 📊 总览

| 策略                 | Baseline Sharpe | 最佳 Sharpe | 提升  | 关键特征           |
| -------------------- | --------------- | ----------- | ----- | ------------------ |
| SR Reversal          | 1.529           | 2.088       | +36%  | sqs_hal_high, vp_* |
| Compression Breakout | 1.577           | 1.817       | +15%  | sr_strength_max    |
| SR Breakout          | 0.759           | 1.658       | +118% | turnover_over_mcap |
| Trend Following      | -0.633          | -0.133      | +79%  | fp_ignition        |

---

## 1. SR Reversal (反转策略)

### ✅ 推荐的特征列

#### Tier 1: 核心特征（必加）

| 特征列         | Importance | 来源函数       | 说明                  |
| -------------- | ---------- | -------------- | --------------------- |
| `cvd_medium`   | 259.75     | baseline       | CVD 中期变化          |
| `cvd_short`    | 234.57     | baseline       | CVD 短期变化          |
| `cvd_long`     | 233.30     | baseline       | CVD 长期变化          |
| `cvd_change_5` | 188.92     | baseline       | CVD 5周期变化率       |
| `sqs_hal_high` | 12.83      | sqs_hal_high_f | SR 质量分数（关键！） |

#### Tier 2: Volume Profile 特征（推荐加）

| 特征列             | Importance | 来源函数                             | 说明           |
| ------------------ | ---------- | ------------------------------------ | -------------- |
| `vp_poc_deviation` | 62.49      | volume_profile_volatility_features_f | POC 偏离度     |
| `vp_entropy`       | 49.38      | volume_profile_volatility_features_f | 成交量分布熵   |
| `vp_width_ratio`   | 47.51      | volume_profile_volatility_features_f | 价值区宽度比   |
| `vp_skewness`      | 45.87      | volume_profile_volatility_features_f | 成交量分布偏度 |

#### Tier 3: DL Sequence 特征（可选）

| 特征列       | Importance | 来源函数               | 说明             |
| ------------ | ---------- | ---------------------- | ---------------- |
| `dl_seq_f40` | 74.04      | dl_sequence_features_f | 深度学习序列特征 |
| `dl_seq_f34` | 20.78      | dl_sequence_features_f | 深度学习序列特征 |

### ❌ 应该排除的特征列

| 特征列             | Importance | 问题                      | 建议                   |
| ------------------ | ---------- | ------------------------- | ---------------------- |
| `trend_r2_50`      | 225.57     | Sharpe -1.504（严重负面） | 排除或 invert          |
| `dtw_min_dist_w15` | 62.76      | DTW 整组负面              | 考虑用语义版 dtw_scene |
| `dtw_random_30_*`  | 42.67      | 随机模板无意义            | 排除                   |

### 📝 推荐的 features.yaml 配置

```yaml
feature_pipeline:
  requested_features:
    # Baseline（必需）
    - cvd_features_f
    # Tier 1: SR 质量
    - sqs_hal_high_f
    # Tier 2: Volume Profile
    - volume_profile_volatility_features_f
    # 可选：DL Sequence
    # - dl_sequence_features_f
  exclude_columns:  # 待实现
    - trend_r2_50
```

---

## 2. Compression Breakout (压缩突破策略)

### ✅ 推荐的特征列

#### Tier 1: 核心特征

| 特征列            | Importance | 来源函数          | 说明                   |
| ----------------- | ---------- | ----------------- | ---------------------- |
| `cvd_change_5`    | 825.99     | baseline          | CVD 变化率（最重要！） |
| `cvd_normalized`  | 524.00     | baseline          | 归一化 CVD             |
| `sr_strength_max` | TBD        | sr_strength_max_f | SR 强度最大值          |

#### Tier 2: 语义场景特征

| 特征列                                 | Sharpe 提升 | 来源函数            | 说明         |
| -------------------------------------- | ----------- | ------------------- | ------------ |
| `trade_cluster_exhaustion_scene_score` | +0.176      | trade_cluster_scene | 衰竭场景     |
| `wpt_ignition_score`                   | +0.119      | wpt_scene           | 点火场景     |
| `funding_ignition_score`               | +0.112      | funding_scene       | 资金费率点火 |

### ❌ 应该排除的特征列

| 特征列                   | Sharpe | 问题     | 建议 |
| ------------------------ | ------ | -------- | ---- |
| `dtw_*_trend_*`          | -0.932 | 严重负面 | 排除 |
| `ad_line_f`              | -0.139 | 负面     | 排除 |
| `wick_scene__exhaustion` | 0.954  | 拉低     | 排除 |

---

## 3. SR Breakout (突破策略)

### ✅ 推荐的特征列

#### Tier 1: 核心特征

| 特征列                | Sharpe 提升 | 来源函数                          | 说明                    |
| --------------------- | ----------- | --------------------------------- | ----------------------- |
| `turnover_over_mcap`  | +0.898      | market_cap_normalized_orderflow_f | 换手率/市值（最重要！） |
| `vpin_ignition_score` | +0.618      | vpin_scene                        | VPIN 点火场景           |

### ❌ 应该排除的特征列

| 特征列           | Sharpe | 问题     | 建议                    |
| ---------------- | ------ | -------- | ----------------------- |
| `wpt_scene__*`   | -0.484 | 全部负面 | 排除整组                |
| `market_cap_usd` | -0.484 | 负面     | 只用 turnover_over_mcap |

---

## 4. Trend Following (趋势跟踪策略)

### ✅ 推荐的特征列

| 特征列               | Sharpe 提升 | 来源函数                          | 说明                       |
| -------------------- | ----------- | --------------------------------- | -------------------------- |
| `fp_ignition_score`  | +0.501      | fp_imbalance_scene                | Footprint 点火（最重要！） |
| `turnover_over_mcap` | +0.421      | market_cap_normalized_orderflow_f | 换手率/市值                |

### ⚠️ 注意事项

- Baseline Sharpe 为负（-0.633），说明当前配置不适合趋势跟踪
- 需要调整标签/回测参数

---

## 📊 特征列重要性热图

```
                          SR_Rev  Comp  Break  Trend
cvd_medium                 ████    ███    ██     █
cvd_change_5               ████    ████   ██     █
sqs_hal_high               ██      █      █      █
vp_poc_deviation           ███     █      █      █
turnover_over_mcap         █       █      ████   ███
fp_ignition_score          █       ██     █      ████
vpin_ignition_score        ██      ██     ███    █
```

---

## 🔧 后续工作

1. **实现 `exclude_columns`**：在 features.yaml 中支持排除特定列
2. **验证 invert 效果**：测试负面特征取反后是否变正面
3. **列级别搜索**：对重要函数做更细粒度的列级别验证
4. **DTW 语义化**：用 `dtw_scene_semantic_scores_f` 替代原始 DTW

---

## 📁 数据来源

| 策略                 | 结果目录                                                     | 结果数 |
| -------------------- | ------------------------------------------------------------ | ------ |
| SR Reversal          | `results/feature_group_search/sr_reversal_expanded`          | 414    |
| Compression Breakout | `results/feature_group_search/compression_breakout_expanded` | 267    |
| SR Breakout          | `results/feature_group_search/sr_breakout_v4`                | 93     |
| Trend Following      | `results/feature_group_search/trend_following_v4`            | 93     |

---

