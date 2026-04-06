# 四场景语义化特征设计（Compression / Ignition / Absorption / Exhaustion）

本文档把“原始统计量特征”转化为**交易机制语义（semantic scores）**，并对齐到四类策略：

- **SR 反转**：更依赖 **Exhaustion/Reversal**
- **SR 突破**：更依赖 **Ignition/Breakout**
- **压缩突破**：更依赖 **Compression + Ignition**
- **趋势**：更依赖 **Absorption/Continuation**

> 语义化（Semanticization）定义：把底层原始特征（例如 VPIN、trade_cluster、volume_profile、WPT、wick 等）映射成少数几个“机制一致、可解释”的分数（0~1），让模型学的是“市场行为含义”，而不是互相混杂的统计量。

---

## 1. 四场景定义（机制口径）

- **Compression（压缩蓄势）**：波动/成交量收缩，结构紧致，能量积累。
- **Ignition/Breakout（点火突破）**：压缩后第一次有效释放（扩张 + 方向确认）。
- **Absorption/Continuation（吸收延续）**：强订单流/成交强度伴随明显位移（顺势延续或突破后继续）。
- **Exhaustion/Reversal（衰竭反转/失败突破）**：强成交/强订单流但“位移不随之扩大”（effort without progress）或出现“穿越后快速回撤/反包”。

---

## 2. 本仓库已有的“可语义化原料”（不引入 L2）

### 2.1 Order flow（ticks）
- **VPIN block**：`vpin_derived_features_f`（含 zscore/spike/momentum 等）
- **TradeCluster raw**：`trade_cluster_block_features_f`（已验证对反转负面）
- **TradeCluster semantic（已落地）**：`trade_cluster_semantic_scores_f`
  - `trade_cluster_flow_intensity`
  - `trade_cluster_exhaustion_score`
  - `trade_cluster_absorption_score`

### 2.2 Liquidity void（不需要 L2）
- `liquidity_void_f`：`liquidity_void_detected/speed/volume_ratio/price_impact/retracement/false_breakout_risk`

### 2.3 Volume profile / VPVR（不需要 ticks）
- `volume_profile_vpvr_f`：VPVR/HVN/LVN 统计
- `volume_profile_volatility_features_f`：`vp_entropy/vp_skewness/vp_poc_deviation/vp_lv_ratio/vp_hv_ratio` 等

### 2.4 WPT 能量（不需要 ticks）
- `wpt_volume_energy_f`：`wpt_breakout_confidence`, `wpt_false_breakout_risk` 等

### 2.5 Candle / rejection proxy（不需要 ticks）
- `wick_ratios_f`（长影线/拒绝代理）
- `price_range_symmetry_f`（价格形态对称性）

> 限价单墙（orderbook depth）若无 L2 数据无法直接测“挂单堆积”，但可以用 SR+VPVR+wick+WPT+liquidity_void 作为 proxy。

---

## 3. 四场景语义分数（建议实现形态）

下面给出**可实现的分数定义**（建议都 clip 到 0~1）。这些分数可以作为：
- 训练侧特征（给 ML）
- 或规则侧 gating（例如只在某分数>阈值时允许某类策略入场）

### 3.1 Compression Score（压缩蓄势）
目标：识别“结构紧致 + 波动/量收缩”状态。

建议原料：
- `bb_width_f` / `bb_width_ratio_f`（若可用）
- `atr_ratio_f`（若可用）
- `wpt_vper_*` / `wpt_multi_scale_consistency`（来自 `wpt_volume_energy_f`）
- `vp_entropy`（低 entropy 更像共识集中）

建议公式（示意）：
- `compression = f1(low_bb_width) * f2(low_atr_ratio) * f3(high_wpt_consistency)`

### 3.2 Ignition Score（点火突破）
目标：压缩后第一次释放，最好伴随“真实突破”的概率高。

建议原料：
- `wpt_breakout_confidence`（直接就是 ignition proxy）
- `wpt_false_breakout_risk`（用于抑制假突破）
- `volume_profile`：`vp_width_ratio`（VA 宽度变化）
- `roc/mom`（kline 动量）

建议公式（示意）：
- `ignition = clip(wpt_breakout_confidence - 0.7*wpt_false_breakout_risk)`

#### 3.2.1 已落地：VPIN / Footprint 的“点火/压缩/吸收/衰竭”四场景语义分数

本仓库已经提供两组**多场景语义映射**（都输出 0..1 分数），用于把 raw orderflow 特征对齐到四个“路径故事”，避免不同策略互相污染：

- `vpin_scene_semantic_scores_f`（ticks heavy）
  - `vpin_compression_score`：高 stress + 低位移 + 高 compression（压力积蓄）
  - `vpin_ignition_score`：高 stress + 高位移（可选 volume spike gate）
  - `vpin_absorption_score`：高 stress + 低位移（SR proximity 加权）
  - `vpin_exhaustion_scene_score`：absorption × (1 - trend_r2_20)（趋势衰竭/结束权重）

- `fp_imbalance_scene_semantic_scores_f`（ticks heavy）
  - `fp_imbalance_compression_score`
  - `fp_imbalance_ignition_score`
  - `fp_imbalance_absorption_score`
  - `fp_imbalance_exhaustion_scene_score`

> 设计要点：同一个 raw 信号（VPIN/imbalance），在反转/突破/趋势/压缩突破的“含义”不同；通过语义化拆成四个版本，让 Router/策略只消费匹配场景的语义版本。

### 3.3 Absorption/Continuation Score（吸收延续）
目标：强订单流 + 大位移（顺势延续/突破后继续）。

建议原料：
- `trade_cluster_absorption_score`（已实现）
- `vpin_signed_imbalance_zscore_*`（方向性）
- `vp_poc_deviation`（价格远离价值区且持续）

建议公式（示意）：
- `absorption = max(trade_cluster_absorption_score, g(vpin_directional_pressure) * g(price_displacement))`

### 3.4 Exhaustion/Reversal Score（衰竭反转/失败突破）
目标：强订单流/强成交，但位移变小（effort without progress），或“穿越后快速回撤”。

建议原料：
- `trade_cluster_exhaustion_score`（已实现）
- `liquidity_void_false_breakout_risk`（“穿越后回撤” proxy）
- `wick_ratios_f`（拒绝/反包影线）
- `vp_hv_ratio`/`vp_entropy`（高成交量节点“卡住”）

建议公式（示意）：
- `exhaustion = max(trade_cluster_exhaustion_score, liquidity_void_false_breakout_risk) * g(wick_rejection)`

---

## 4. 与四类策略的对应关系（怎么用）

| 策略类型 | 主要依赖语义 | 典型 gating |
|---|---|---|
| **SR 反转** | Exhaustion/Reversal | `exhaustion > t1` 且 `compression` 不极弱 |
| **SR 突破** | Ignition/Breakout + Absorption | `ignition > t2` 且 `absorption > t3` |
| **压缩突破** | Compression + Ignition | `compression > t4` 且 `ignition > t5` |
| **趋势** | Absorption/Continuation | `absorption > t6` 且 `exhaustion` 不高 |

> 注意：这里的 gating 不一定要做成硬规则，也可以作为模型输入，让模型学习 “哪些语义在某一策略目标下更重要”。

---

## 5. 已验证结论（当前进展）

在 BTCUSDT / 240T / 2023-01-01~2025-10-31 / test_size=0.3 / seeds=1..5：

- raw TradeCluster（`trade_cluster_block_features_f`）对 SR 反转 **显著负面**
- TradeCluster 语义化（`trade_cluster_semantic_scores_f`）可把信息拆成：
  - exhaustion（反转友好）
  - absorption（突破友好）
  并在反转任务上**转正且提升 Sharpe**

---

## 6. 下一步实现建议（按“只改一个因素”原则）

1) **把 VPIN 语义化**：分离 stress/toxicity 与 directional pressure（避免 VPIN 高=一概当机会）
2) **把 LiquidityVoid 语义化**：分离 sweep/void 与 failure/fakeout
3) **把 VolumeProfile 语义化**：输出 wall_strength / acceptance / rejection 等少数分数
4) 每新增一个语义分数，都做 seeds=1..5（并在 ETH 复核），避免单次 run 假阳性。


---

## 7. 本仓库已落地的“语义化特征”清单（可直接在 `features.yaml` 使用）

> 下面清单以 `config/feature_dependencies.yaml` 为准；实现代码主要在：
> - `src/features/time_series/utils_interaction_features.py`
> - `src/features/time_series/utils_order_flow_features.py`（TradeCluster semantic 原料）

### 7.1 语义化（semantic scores，偏 SR 反转口径）

| feature node | 输出列（output_columns） | 计算函数（compute_func） | 说明 |
|---|---|---|---|
| `trade_cluster_semantic_scores_f` | `trade_cluster_flow_intensity`, `trade_cluster_exhaustion_score`, `trade_cluster_absorption_score` | `compute_trade_cluster_semantic_scores_from_series` | TradeCluster raw → Exhaustion/Absorption（把“放量”从异义统计量变成机制语义） |
| `vpin_semantic_scores_f` | `vpin_stress_score`, `vpin_directional_pressure`, `vpin_exhaustion_score` | `compute_vpin_semantic_scores_from_series` | VPIN zscore → stress/pressure/exhaustion（含位移/ATR 与 SR proximity 权重） |
| `tbr_imbalance_semantic_scores_f` | `imbalance_ratio`, `imbalance_exhaustion_score` | `compute_tbr_imbalance_semantic_scores_from_series` | 用 `taker_buy_ratio` 近似不平衡，并生成 exhaustion 口径分数（bar-level proxy） |
| `exhaustion_at_liquidity_void_f` | `exhaustion_at_liquidity_void` | `compute_exhaustion_at_liquidity_void_from_series` | “逻辑加乘”特征：TradeCluster exhaustion × liquidity_void（可选再乘 fakeout risk） |

### 7.2 四场景语义化（scene semantic scores：Compression / Ignition / Absorption / Exhaustion）

| feature node | 输出列（output_columns） | 计算函数（compute_func） | 说明 |
|---|---|---|---|
| `vpin_scene_semantic_scores_f` | `vpin_compression_score`, `vpin_ignition_score`, `vpin_absorption_score`, `vpin_exhaustion_scene_score` | `compute_vpin_scene_semantic_scores_from_series` | VPIN 的四场景拆分（避免 raw VPIN 在不同策略里互相污染） |
| `fp_imbalance_scene_semantic_scores_f` | `fp_imbalance_compression_score`, `fp_imbalance_ignition_score`, `fp_imbalance_absorption_score`, `fp_imbalance_exhaustion_scene_score` | `compute_fp_imbalance_scene_semantic_scores_from_series` | Footprint imbalance 的四场景拆分 |
| `trade_cluster_scene_semantic_scores_f` | `trade_cluster_compression_score`, `trade_cluster_ignition_score`, `trade_cluster_absorption_scene_score`, `trade_cluster_exhaustion_scene_score` | `compute_trade_cluster_scene_semantic_scores_from_series` | 基于 `trade_cluster_semantic_scores_f` 再做四场景（叠加 compression/volume/trend context） |
| `liquidity_void_scene_semantic_scores_f` | `liquidity_void_compression_score`, `liquidity_void_ignition_score`, `liquidity_void_absorption_score`, `liquidity_void_exhaustion_scene_score` | `compute_liquidity_void_scene_semantic_scores_from_series` | liquidity void 的四场景拆分（含 WPT risk / compression / trend context） |
| `wpt_scene_semantic_scores_f` | `wpt_compression_score`, `wpt_ignition_score`, `wpt_absorption_score`, `wpt_exhaustion_score` | `compute_wpt_scene_semantic_scores_from_series` | WPT 能量/假突破风险的四场景拆分 |
| `volume_profile_scene_semantic_scores_f` | `vp_compression_score`, `vp_ignition_score`, `vp_absorption_score`, `vp_exhaustion_score` | `compute_volume_profile_scene_semantic_scores_from_series` | Volume Profile（VPVR/volatility features）的四场景拆分 |
| `wick_scene_semantic_scores_f` | `wick_compression_score`, `wick_ignition_score`, `wick_absorption_score`, `wick_exhaustion_score` | `compute_wick_scene_semantic_scores_from_series` | Wick/rejection 的四场景拆分（更适合做 gating / 反转确认） |


