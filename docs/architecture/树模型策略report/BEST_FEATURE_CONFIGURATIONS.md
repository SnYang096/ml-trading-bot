# 各策略最佳特征配置汇总

本文档汇总了通过 `feature-group-search` 工具找到的各策略最佳特征配置。

**更新时间**：2024-12-30

> ⚠️ **维护状态**：本文件已被更“统一/最新”的入口文档替代（避免多处维护导致口径不一致）。
> 请优先阅读：`docs/architecture/树模型策略report/FEATURE_SELECTION_REPORTS.md`

---

## 1. SR Reversal (sr_reversal_rr_reg_long)

### 最佳配置
- **运行名称**：`sr_reversal_rr_reg_long_demo`
- **Base Features**：空（从零开始构建特征集）
- **Baseline Sharpe**：N/A（此运行从空特征集开始，无 baseline 对比）
- **最终选择特征组**：
  - `vpin_scene`（VPIN 场景语义特征）
  - `kline_core`（K 线核心特征：MACD, RSI, SMA, ATR, Trend R2, BB Width, Wick Ratios）
- **改进过程**：
  - **Step 1（添加 `vpin_scene`）**：Sharpe **1.5186**
    - Mean: 1.5186, Std: 0.4693, Min: 1.1868, Max: 1.8505
    - Return%: 12.46%, DD%: 5.09%, Trades: 47.5
  - **Step 2（添加 `kline_core`）**：Sharpe **1.8474**
    - Mean: 1.8474, Std: 0.6342, Min: 1.3990, Max: 2.2959
    - Return%: 17.96%, DD%: 6.54%, Trades: 33.0
- **最终 Sharpe（Step 2）**：**1.8474**
  - Mean: 1.8474
  - Std: 0.6342
  - Min: 1.3990
  - Max: 2.2959
- **停止原因**：N/A（可能是手动停止或达到最大步数）

### 说明
- SR Reversal 策略的最佳特征配置已通过实验确定，主要包含 VPIN 场景语义特征和 K 线核心特征。
- 从空特征集开始，逐步添加特征组，最终 Sharpe 达到 1.85，表现优秀。
- **注意**：此运行从空特征集开始（`base_features: []`），因此没有 baseline 对比。如果需要 baseline 对比，应该先运行一个包含基础特征的 baseline 配置。

---

## 2. SR Breakout (sr_breakout)

### 最佳配置（多符号）
- **运行名称**：`sr_breakout_best_combo_multisymbol_v3`
- **Baseline Sharpe**：**-0.6965** ⚠️
  - Mean: -0.6965
  - Std: 1.1524
  - Min: -2.5297
  - Max: 0.5060
- **最终选择特征组**：
  - `wpt_scene`（WPT 场景语义特征）
  - `market_cap_norm`（市值归一化订单流）
- **最终 Sharpe（Step 2）**：**1.1142**
  - Mean: 1.1142
  - Std: 1.2048
  - Min: -1.0169
  - Max: 1.8430
- **停止原因**：`no_improvement`（后续特征组未带来改进）

### 问题分析
⚠️ **严重问题**：Baseline Sharpe 为负（-0.6965），说明策略在多符号场景下表现极差。

**可能原因**：
1. **标签生成问题**：多符号场景下标签生成可能不正确（已修复，但可能此运行使用的是旧版本）
2. **特征计算问题**：多符号场景下特征计算可能未正确隔离（symbol 隔离问题）
3. **回测配置问题**：多符号回测配置可能不正确（`freq`、`use_rr_exit`、`use_signal_direction` 等）
4. **数据问题**：多符号数据可能不完整或有问题

**改进情况**：
- Step 1（添加 `wpt_scene`）：Sharpe 从 -0.6965 提升到 0.0088（轻微改进）
  - Mean: 0.0088, Std: 0.8211, Min: -1.0732, Max: 1.1319
- Step 2（添加 `market_cap_norm`）：Sharpe 提升到 1.1142（显著改进）
  - Mean: 1.1142, Std: 1.2048, Min: -1.0169, Max: 1.8430

**建议**：
1. 检查多符号标签生成逻辑（确保每个 symbol 独立计算）
2. 检查多符号特征计算（确保 symbol 隔离）
3. 检查回测配置（确保 `freq`、`use_rr_exit`、`use_signal_direction` 正确配置）
4. 重新运行 feature-group-search，使用修复后的代码

### 快速运行结果（v3_quick）
- **运行名称**：`sr_breakout_best_combo_multisymbol_v3_quick`（2024-12-30 03:02）
- **Baseline Sharpe**：**-0.2143**
  - Mean: -0.2143, Std: 0.1348, Min: -0.3096, Max: -0.1190
  - Return%: -60.17%, Trades: 121, DD%: 116.82%
- **最终选择特征组**：`volume_profile_scene`
- **最终 Sharpe（Step 1）**：**0.8987** ✅
  - Mean: 0.8987, Std: 0.3072, Min: 0.6815, Max: 1.1159
  - Return%: -5.65%, Trades: 124, DD%: 93.27%
  - **提升**：Sharpe 从 -0.21 提升到 0.90（+1.11），Return% 改善 54.52%
- **候选组排名**：
  1. `volume_profile_scene`: 0.90 ✅（已选择）
  2. `trade_cluster_scene`: 0.46
  3. `funding_scene`: 0.45
  4. `fp_scene`: 0.34
  5. `market_cap_norm`: 0.26
- **停止原因**：`max_steps_reached`
- **耗时**：15.1 分钟（2 seeds）

### 失败运行结果（v2）
- **运行名称**：`sr_breakout_best_combo_multisymbol_v2`（2024-12-30 01:45）
- **Baseline Sharpe**：**NaN**（0 trades）❌
  - Return%: 0.00%, Trades: 0
- **结果**：未选择任何特征组
- **停止原因**：`no_valid_candidates`（baseline 无交易，无法评估候选组）
- **耗时**：13.0 分钟（5 seeds）
- **问题**：可能是回测配置问题或数据问题，导致 baseline 无法生成交易

---

## 3. Compression Breakout (compression_breakout)

### 最佳配置（多符号）
- **运行名称**：`compression_breakout_best_combo_multisymbol_v1`
- **Baseline Sharpe**：**-0.7364** ⚠️
  - Mean: -0.7364, Std: 1.1894, Min: -1.9257, Max: 0.6790
- **最终选择特征组**：
  - `volume_profile_scene`（成交量分布场景语义特征）
  - `vpin_scene`（VPIN 场景语义特征）
- **最终 Sharpe（Step 2）**：**0.4189**
  - Mean: 0.4189, Std: 0.7063, Min: -0.4886, Max: 1.2121
- **停止原因**：`no_improvement`

### 问题分析
⚠️ **问题**：Baseline Sharpe 为负（-0.7364），说明策略在多符号场景下表现不佳。

**可能原因**：与 SR Breakout 类似，可能是多符号场景下的标签生成、特征计算或回测配置问题。

### 单符号运行结果
- **运行名称**：`compression_breakout_best_combo_v5`
- **Baseline Sharpe**：-1.4063
  - Mean: -1.4063, Std: 1.5181, Min: -3.4393, Max: 0.4221
- **最终选择特征组**：
  - `vpin_scene`
  - `wpt_scene`
- **最终 Sharpe（Step 2）**：**0.6291**
  - Mean: 0.6291, Std: 1.3274, Min: -1.3593, Max: 1.8860
- **停止原因**：`no_improvement`

---

## 4. Trend Following (trend_following)

### 最佳配置
- **运行名称**：`trend_following_best_combo_v5`
- **Baseline Sharpe**：**1.3728** ✅
  - Mean: 1.3728, Std: 2.0108, Min: -1.3794, Max: 3.7845
- **最终选择特征组**：**无**（Baseline 已经很好）
- **最终 Sharpe**：1.3728（与 Baseline 相同，无改进）
- **停止原因**：`no_improvement`（所有候选特征组都未带来改进）

### 说明
- Trend Following 策略的 Baseline 表现已经很好（Sharpe 1.37），所有候选特征组都未带来显著改进。
- 这说明 Baseline 特征集已经足够好，或者候选特征组不适合 Trend Following 策略。

### 快速运行结果
- **运行名称**：`trend_following_best_combo_quick3`
- **Baseline Sharpe**：1.1707
  - Mean: 0.4238（注意：这里 baseline 的 mean 和 score 不一致，可能是数据问题）
- **最终选择特征组**：`liquidity_void_scene`
- **最终 Sharpe（Step 1）**：**1.1707**
  - Mean: 1.1707（与 Baseline 相同，无改进）
- **停止原因**：`no_improvement`

---

## 总结

### 策略表现排名（按最终 Sharpe）
1. **SR Reversal**：最终 Sharpe **1.85**（最佳，添加特征后）
2. **Trend Following**：Baseline Sharpe **1.37**（Baseline 已很好，无需改进）
3. **SR Breakout（多符号 v3）**：最终 Sharpe **1.11**（添加 `wpt_scene` + `market_cap_norm`，从 -0.70 改进）
4. **SR Breakout（多符号 v3_quick）**：最终 Sharpe **0.90**（添加 `volume_profile_scene`，从 -0.21 改进）
5. **Compression Breakout（单符号）**：最终 Sharpe **0.63**（添加特征后，从 -1.41 改进）
6. **Compression Breakout（多符号）**：最终 Sharpe **0.42**（添加特征后，从 -0.74 改进）

### 关键发现
1. **Trend Following 策略表现最好**：Baseline 已经很好，不需要额外特征。
2. **SR Breakout 多符号场景改进显著**：
   - **v3_quick**：添加 `volume_profile_scene` 后，Sharpe 从 -0.21 提升到 0.90（+1.11）
   - **v3**：添加 `wpt_scene` + `market_cap_norm` 后，Sharpe 从 -0.70 提升到 1.11（+1.81）
   - 说明语义特征在多符号场景下非常有效
3. **最佳特征组（SR Breakout 多符号）**：
   - `volume_profile_scene`：Sharpe 0.90（v3_quick 验证）
   - `wpt_scene` + `market_cap_norm`：Sharpe 1.11（v3 验证）
   - `trade_cluster_scene`：Sharpe 0.46（候选）
   - `funding_scene`：Sharpe 0.45（候选）
4. **Compression Breakout 在多符号场景下表现差**：Baseline Sharpe 为负（-0.74），需要检查：
   - 标签生成逻辑（多符号隔离）
   - 特征计算逻辑（多符号隔离）
   - 回测配置（`freq`、`use_rr_exit`、`use_signal_direction`）

### 下一步行动
1. **修复 SR Breakout 和 Compression Breakout 的多符号问题**：
   - 检查标签生成逻辑
   - 检查特征计算逻辑
   - 检查回测配置
   - 重新运行 feature-group-search
2. **验证 Trend Following 的 Baseline 稳定性**：
   - 运行多符号确认
   - 运行 rolling 训练验证稳定性
3. **继续优化 SR Breakout**：
   - 尝试其他特征组组合
   - 调整策略参数

---

## 附录：特征组说明

### 语义特征组
- **`wpt_scene`**：WPT（小波包变换）场景语义特征
- **`vpin_scene`**：VPIN（订单流不平衡）场景语义特征
- **`volume_profile_scene`**：成交量分布场景语义特征
- **`trade_cluster_scene`**：交易聚类场景语义特征
- **`liquidity_void_scene`**：流动性真空场景语义特征
- **`fp_scene`**：Footprint 场景语义特征
- **`wick_scene`**：Wick 比率场景语义特征
- **`funding_scene`**：资金费率场景语义特征

### 其他特征组
- **`market_cap_norm`**：市值归一化订单流特征
- **`kline_core`**：K 线核心特征

---

## 工作流选择：Pool B vs 语义特征组

### 关键发现（通过实际分析）

**重要结论**：语义 groups 覆盖率很低（1.8%），确实可能遗漏重要特征！

- **全量特征数**: 650
- **语义 groups 特征节点数**: 12
- **语义 groups 实际输出列数**: 45 列
- **覆盖率（按节点）**: 1.8%
- **未覆盖特征数**: 648

**语义 groups 实际输出列详情**：
- `liquidity_void_f`: 6 列（liquidity_void_detected, speed, volume_ratio, price_impact, retracement, false_breakout_risk）
- `compression_score_f`: 1 列（compression_score）
- `compression_energy_f`: 1 列（compression_energy）
- `liquidity_void_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `vpin_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `trade_cluster_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `wpt_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `volume_profile_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `wick_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `fp_imbalance_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）
- `market_cap_normalized_orderflow_f`: 5 列（market_cap_usd, dollar_volume_over_mcap, turnover_over_mcap, net_buy_usd_over_mcap, abs_net_buy_usd_over_mcap）
- `funding_scene_semantic_scores_f`: 4 列（compression/ignition/absorption/exhaustion scores）

**主要遗漏的特征类型**：
- DTW 模式匹配: 175 个特征，0% 覆盖
- Trade Cluster: 67 个特征，0% 覆盖
- K线技术指标: 63 个特征，0% 覆盖
- VPIN 相关: 53 个特征，0% 覆盖
- WPT 小波: 22 个特征，0% 覆盖

**语义特征内部可能的冲突**：
- `liquidity_void` vs `liquidity_void_scene`: 同源但不同语义，需要测试是否冲突

### 关系说明

**语义特征组**和**Pool B（factor-eval 输出）**应该**同时使用**，而不是二选一：

- **语义特征组（Semantic Features）**：
  - **人类可理解的特征因子**，必须由人类根据市场逻辑设计和维护
  - 预定义的、经过语义化的特征组（如 `vpin_scene`, `wpt_scene` 等）
  - 已经过人工筛选和语义化，相对稳定
  - 当前最佳配置（`vpin_scene` + `kline_core`）就是来自语义 groups
  - **但覆盖率很低（1.8%）**，可能遗漏重要特征
  - **从 Pool B 深度加工而来**：语义特征通常是从 Pool B 中的原始特征（如 VPIN、TradeCluster）经过语义化转换得到的

- **Pool B**：
  - **数学/数值特征的"海选池"**，包含大量原始特征
  - 数学特征：DTW、EVT、GARCH、Hilbert、Hurst、频谱特征等
  - TA-Lib 特征：MACD、RSI、BBands、ATR 等技术指标
  - 数值特征：各种统计量、相关性、波动率等
  - 从大量原始特征中通过 IC/IR 筛选出的候选特征
  - **用于发现未被语义化的遗漏特征**（适合"海选"）
  - **必须使用**：因为语义 groups 覆盖率太低
  - **语义特征的"原料"**：语义特征是从 Pool B 中原始特征的深度加工

### 推荐工作流（两阶段）

**阶段 1: Pool B 过滤（发现遗漏特征）**

```bash
# 1. 运行 factor-eval 生成 Pool B
mlbot analyze factor-eval \
  -c config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/sr_reversal_rr_reg_long/pool_b \
  --export-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --remove-correlated --filter-by-best-lag --no-docker

# 2. 分析 Pool B 中是否有语义 groups 未覆盖的重要特征
python scripts/analyze_semantic_vs_all_features.py \
  --strategy sr_reversal_rr_reg_long \
  --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
  --all-features results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --output-dir results/feature_analysis

# 3. 检测语义特征内部可能的冲突
python scripts/detect_semantic_conflicts.py \
  --strategy sr_reversal_rr_reg_long \
  --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
  --test-combinations
```

**阶段 2: 合并搜索（同时使用 Pool B 和语义 groups）**

```bash
# 4. 运行 feature-group-search，同时使用语义 groups 和 Pool B
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_rr_reg_long \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --max-steps 6 \
  --writeback-yaml config/strategies/sr_reversal_rr_reg_long/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_reversal_best_combo \
  --no-docker
```

### 工具说明

1. **`scripts/analyze_semantic_vs_all_features.py`**: 分析语义特征 vs 全量特征的覆盖情况
2. **`scripts/detect_semantic_conflicts.py`**: 检测语义特征内部可能的冲突

详细文档：`docs/architecture/strategies/SEMANTIC_VS_POOLB_WORKFLOW.md`

---

## 参考文档

- **实验协议**：
  - `docs/architecture/strategies/SR_REVERSAL_EXPERIMENT_PROTOCOL.md`
  - `docs/architecture/strategies/SR_BREAKOUT_EXPERIMENT_PROTOCOL.md`
- **架构文档**：
  - `docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- **工具文档**：
  - `mlbot diagnose feature-group-search --help`


  - `mlbot diagnose feature-group-search --help`

