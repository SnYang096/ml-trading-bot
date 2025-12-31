# Pool B 与语义特征交叉情况报告

## 执行时间
生成时间: 2025-01-XX

## 四个策略的 Pool B 输出情况

### 统计汇总

| 策略 | Pool B 状态 | 语义 groups 状态 | Pool B 特征数 | 语义节点数 | 语义输出列数 |
|------|------------|-----------------|--------------|-----------|-------------|
| `sr_reversal_rr_reg_long` | ✅ 已生成 | ✅ 存在 | 24 | 12 | 45 |
| `sr_breakout` | ❌ 未生成 | ✅ 存在 | - | 9 | 37 |
| `compression_breakout` | ❌ 未生成 | ✅ 存在 | - | 9 | 37 |
| `trend_following` | ❌ 未生成 | ✅ 存在 | - | 9 | 37 |

**总结**:
- 已生成 Pool B 的策略: **1/4** (25%)
- 有语义 groups 的策略: **4/4** (100%)

---

## SR Reversal 详细交叉分析

### Pool B 特征列表（24 个）

1. `aroon_f`
2. `dist_to_zz_high_atr_f`
3. `dl_sequence_features_f`
4. `dtw_features_compression_f`
5. `evt_features_f`
6. `extended_volatility_features_f`
7. `garch_features_f`
8. `hilbert_phase_f`
9. `hurst_volume_f`
10. `liquidity_void_f` ⭐ **与语义 groups 交叉**
11. `spectrum_features_compression_breakout_f`
12. `spectrum_features_sr_breakout_f`
13. `spectrum_features_sr_reversal_f`
14. `spectrum_features_trend_following_f`
15. `trade_cluster_base_aligned_features_f`
16. `trade_cluster_block_features_f`
17. `trade_cluster_derived_features_f`
18. `trade_cluster_entropy_features_f`
19. `trade_cluster_entropy_ma_change_features_f`
20. `trade_cluster_entropy_zscore_features_f`
21. `trade_cluster_imbalance_ma_features_f`
22. `trade_cluster_imbalance_zscore_features_f`
23. `trade_cluster_max_buy_run_ma_features_f`
24. `trade_cluster_max_buy_run_zscore_features_f`

### 语义 groups 特征列表（12 个节点）

1. `liquidity_void_f` ⭐ **与 Pool B 交叉**
2. `compression_score_f`
3. `compression_energy_f`
4. `liquidity_void_scene_semantic_scores_f`
5. `vpin_scene_semantic_scores_f`
6. `trade_cluster_scene_semantic_scores_f`
7. `wpt_scene_semantic_scores_f`
8. `volume_profile_scene_semantic_scores_f`
9. `wick_scene_semantic_scores_f`
10. `fp_imbalance_scene_semantic_scores_f`
11. `market_cap_normalized_orderflow_f`
12. `funding_scene_semantic_scores_f`

### 交叉分析

#### 节点级别交叉

- **交叉特征数**: 1 个
  - `liquidity_void_f` ✅

- **Pool B 独有特征数**: 23 个
  - 主要是：DTW、EVT、GARCH、Hilbert、Hurst、Spectrum、Trade Cluster 的原始特征节点
  - **这些特征在语义 groups 中没有，说明 Pool B 发现了未被语义化的有效特征**

- **语义 groups 独有特征数**: 11 个
  - 主要是：各种 scene 语义特征（vpin_scene, wpt_scene, trade_cluster_scene 等）
  - **这些是经过语义化的特征，Pool B 中没有，说明语义化是有效的**

#### 输出列级别交叉

- **Pool B 覆盖语义特征输出列**: 0.0% (0/45)
  - Pool B 中的特征都是节点名，不是输出列名
  - 因此输出列级别的交叉为 0

---

## 关键发现

### 1. Pool B 和语义 groups 互补性强

- **Pool B 发现的特征**：
  - 主要是原始特征节点（DTW、EVT、GARCH、Hilbert、Hurst、Spectrum、Trade Cluster）
  - 这些特征还没有被语义化
  - 说明 Pool B 确实发现了未被语义化的有效特征

- **语义 groups 包含的特征**：
  - 主要是经过语义化的 scene 特征（compression/ignition/absorption/exhaustion）
  - 这些特征已经经过人工筛选和语义化
  - 说明语义化是有效的

### 2. 交叉很少，说明两者互补

- 节点级别交叉只有 1 个（`liquidity_void_f`）
- 这说明 Pool B 和语义 groups 覆盖的特征类型不同
- **建议：同时使用 Pool B 和语义 groups 进行 feature-group-search**

### 3. 其他三个策略还没有 Pool B

- `sr_breakout`、`compression_breakout`、`trend_following` 都还没有生成 Pool B
- 建议为这些策略也运行 `factor-eval` 生成 Pool B

---

## 建议

### 1. 为其他策略生成 Pool B

```bash
# SR Breakout
mlbot analyze factor-eval \
  -c config/strategies/sr_breakout/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/sr_breakout/pool_b \
  --export-yaml results/pools/sr_breakout/pool_b/features_pool_b.yaml \
  --remove-correlated --filter-by-best-lag --no-docker

# Compression Breakout
mlbot analyze factor-eval \
  -c config/strategies/compression_breakout/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/compression_breakout/pool_b \
  --export-yaml results/pools/compression_breakout/pool_b/features_pool_b.yaml \
  --remove-correlated --filter-by-best-lag --no-docker

# Trend Following
mlbot analyze factor-eval \
  -c config/strategies/trend_following/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/trend_following/pool_b \
  --export-yaml results/pools/trend_following/pool_b/features_pool_b.yaml \
  --remove-correlated --filter-by-best-lag --no-docker
```

### 2. 同时使用 Pool B 和语义 groups

对于 SR Reversal（已有 Pool B），建议运行：

```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_rr_reg_long \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --max-steps 6 \
  --writeback-yaml config/strategies/sr_reversal_rr_reg_long/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_reversal_poolb_semantic \
  --no-docker
```

### 3. 分析其他策略的交叉情况

等其他策略的 Pool B 生成后，运行：

```bash
python scripts/analyze_poolb_semantic_overlap.py
```

---

## 工具

- `scripts/analyze_poolb_semantic_overlap.py`: 分析 Pool B 与语义特征的交叉情况

