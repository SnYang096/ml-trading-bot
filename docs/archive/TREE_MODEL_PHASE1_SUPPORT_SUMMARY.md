# 第一期树模型支持总结（语义特征 + 代码位置）

## 来源文档
- `docs/architecture/树模型策略知识迁移到多头模型.md`
- `docs/strategies/树模型策略区分机制.md`
- `docs/architecture/6种对称策略的启发式规则.md`

## 使用原则（从树模型到 Gate）
- 树模型规则不是“entry 触发器”，是语义证据 → Gate / Bias / Risk Modulator。
- 阈值不可照抄，应转换为**分位数**或**跨 symbol 归一化**指标。
- 同一特征在不同策略（sr_reversal / sr_breakout / compression_breakout / trend_following）中可能语义相反。

---

## Phase1 语义特征索引（含代码位置）

> 目的：当语义类似但名字不同，快速定位是否重复/替代。

| 语义特征 | 输出列（常见） | 主要用途 | Feature Node / 计算函数 | 代码位置 |
|---|---|---|---|---|
| wick_scene | `wick_exhaustion_score` 等 | FBF / LSR / AER | `wick_scene_semantic_scores_f` / `compute_wick_scene_semantic_scores_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_interaction_features.py` |
| fp_scene (footprint imbalance) | `fp_imbalance_*` | BPC / ME / LSR / AER | `fp_imbalance_scene_semantic_scores_f` / `compute_fp_imbalance_scene_semantic_scores_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_interaction_features.py` |
| fp_scene (exhaustion) | `fp_imbalance_exhaustion_score` | FR / AER proxy | `fp_imbalance_exhaustion_f` / `compute_fp_imbalance_exhaustion_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_interaction_features.py` |
| vpin_scene | `vpin`, `vpin_change_pct` 等 | BPC / ME / FBF | `vpin_*` family | `config/feature_dependencies.yaml` + `src/features/time_series/utils_order_flow_features.py` |
| trade_cluster_scene | `trade_cluster_*` | BPC / LSR / AER | `trade_cluster_block_features_f` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_order_flow_features.py` |
| liquidity_void_scene | `liquidity_void_*` | FBF / LSR | `liquidity_void_f` / `compute_liquidity_void_features_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_liquidity_features.py` |
| wpt_scene | `wpt_*_score` | ME / AER | `wpt_scene_semantic_scores_f` / `compute_wpt_scene_semantic_scores_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_liquidity_features.py` |
| funding_exhaustion | `funding_exhaustion_scene_score` | AER | `funding_scene_semantic_scores_f` / `compute_funding_scene_semantic_scores_from_df` | `config/feature_dependencies.yaml` + `src/features/time_series/funding_rate_features.py` |
| sr_distance (normalized) | `sr_distance_normalized` | FBF / LSR | `sr_distance_normalized_f` / `compute_sr_distance_normalized_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/utils_interaction_features.py` |
| path_efficiency | `path_efficiency_pct` | BPC / ME / AER | `path_efficiency_pct_f` / `compute_path_efficiency_pct_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| dir_consistency | `price_dir_consistency_pct` | BPC / HTF | `price_dir_consistency_pct_f` / `compute_price_dir_consistency_pct_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| jump_risk | `jump_risk_pct` | BPC / ME / LSR / AER | `jump_risk_pct_f` / `compute_jump_risk_pct_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| atr_percentile | `atr_percentile` | ME / AER | `atr_percentile_f` / `compute_atr_percentile_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| bb_width_percentile | `bb_width_normalized_pct` | ME | `bb_width_normalized_pct_f` / `compute_bb_width_normalized_pct_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| volume_ratio_percentile | `volume_ratio_pct` | ME / LSR | `volume_ratio_pct_f` / `compute_volume_ratio_pct_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| cvd_change | `cvd_change_5` / `cvd_change_5_pct` | BPC / LSR / FBF | `cvd_change_features_f` / `compute_cvd_change_5_pct_from_series` | `config/feature_dependencies.yaml` + `src/features/time_series/baseline_features.py` |
| reflexivity | `shd_pct`, `ofci_pct` | 全局安全过滤 | `shd_pct_f` / `ofci_pct_f` | `config/feature_dependencies.yaml` + `src/features/time_series/reflexivity_features.py` |

---

## 6 Archetype 语义与 Phase1 树模型支持

> 每个 archetype：  
> ① 旧语义（来自 `6种对称策略的启发式规则.md`）  
> ② Phase1 树规则例子（来自 `树模型策略知识迁移到多头模型.md` / `树模型策略区分机制.md`）  
> ③ 备注（阈值示例只能作为参考）

### ① Breakout → Pullback → Continuation (BPC)
**旧语义（核心）**  
- `fp_scene`（pullback 期间反向单未成主导）  
- `trade_cluster_scene`（低 aggressiveness pullback + 新大单 cluster）  
- `vpin_scene`（趋势确认）

**Phase1 树规则示例（来自 trend_following）**  
- `bb_width_normalized <= 4.86 AND cvd_long <= -34895`  
  → 语义：趋势已死，应 veto TC

**备注**  
树模型更多提供“否决”而非“触发”语义（Gate > Bias）。

---

### ② Momentum Expansion (ME)
**旧语义（核心）**  
- `trade_cluster_semantic`（cluster 连续、间距缩短）  
- `fp_scene`（单边 imbalance 连续）  
- `wpt_scene`（能量跨 bar 持续）  
- `liquidity_void_scene`（辅助）

**Phase1 树规则示例（来自 sr_breakout / compression_breakout）**  
- `cvd_change_5 > 4805 AND volume_ratio > 1.87`  
  → 语义：订单流 + 放量确认

---

### ③ HTF Bias + LTF Entry (HTF)
**旧语义（核心）**  
- `fp_scene`（LTF entry 与 HTF 方向一致）  
- `wick_scene`（反向 wick 被立即吸收）  
- `trade_cluster_scene`（反向失败 cluster）

**Phase1 树规则支持**  
无显式阈值，主要是“执行过滤”语义（非信号源）。

---

### ④ Failed Breakout Fade (FBF)
**旧语义（核心）**  
- `wick_scene_semantic_scores_f`（长 wick + 高成交量）  
- `fp_scene`（aggressor 多但价格不走）  
- `vpin_scene`（突破时 VPIN 快速上升）  
- `trade_cluster_scene`（极值区 cluster 但无 follow-through）

**Phase1 树规则示例（来自 sr_reversal）**  
- `liquidity_void_retracement <= 0.037 AND cvd_change_5 <= -848`  
  → 语义：假突破 + 订单流背离

---

### ⑤ Liquidity Sweep → Rejection (LSR)
**旧语义（核心）**  
- `liquidity_void_scene`（sweep 后快速回流）  
- `fp_scene`（单边 aggressor → 反向 imbalance）  
- `trade_cluster_scene`（大单 cluster 失败）  
- `wick_scene`（stop-hunt wick）

**Phase1 树规则支持**  
未给出明确阈值，但核心是“流动性陷阱 + 反向吸收”。

---

### ⑥ Auction Exhaustion Reversal (AER)
**旧语义（核心）**  
- `trade_cluster_semantic`（cluster 变大但推进变小）  
- `wpt_scene`（能量峰值后衰减）  
- `fp_scene`（imbalance 仍在但价格不走）

**Phase1 树规则示例**  
- `funding_exhaustion_scene_score > threshold AND macd_signal weak`  
  → 语义：趋势末期衰竭

---

## 备注：阈值的正确处理方式
- 以上阈值仅为 Phase1 树模型中的例子，不可直接落到 Gate。  
- 推荐映射为：`value → quantile` 或 `value → pct`（rolling percentile）。  
- 对跨 symbol 特征，用 `*_pct` 比 `value` 更稳定。

