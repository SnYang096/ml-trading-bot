# 当前 Gate 特征 vs Phase1 树模型语义（6 Archetype 对照表）

## 说明
- **Aligned**：语义与旧文档一致，且特征名/输出列一致或高度一致  
- **Proxy**：语义接近但特征名称/计算逻辑不同（可能是替代/重复）  
- **Missing**：旧语义核心，但当前 Gate 未覆盖  
- **Extra**：当前 Gate 新增但旧语义未强调

数据来源：
- 旧语义：`docs/architecture/6种对称策略的启发式规则.md`
- 当前 Gate：`config/nnmultihead/execution_archetypes.yaml`
- Phase1 树模型解释：`docs/architecture/树模型策略知识迁移到多头模型.md`

---

## ① Breakout → Pullback → Continuation (BPC)

| 旧语义特征 | 当前 Gate 特征 | 状态 | 备注 |
|---|---|---|---|
| fp_scene | vpin | Proxy | fp_scene 语义是 footprint imbalance，vpin 仅是粗代理 |
| trade_cluster_scene | cvd_change_5_pct | Proxy | 当前用订单流变化 proxy cluster |
| vpin_scene | vpin | Aligned | 名称不同但语义接近 |
| wick_scene（无用） | - | Missing | 旧文档标记为“用处不大” |
| - | path_efficiency_pct | Extra | 结构强度过滤 |
| - | price_dir_consistency_pct | Extra | 结构方向过滤 |
| - | jump_risk_pct / atr_percentile | Extra | regime 排除 |
| - | shd_pct / ofci_pct | Extra | 反身性风险过滤 |

---

## ② HTF Bias + LTF Entry (HTF)

| 旧语义特征 | 当前 Gate 特征 | 状态 | 备注 |
|---|---|---|---|
| fp_scene（LTF） | vpin | Proxy | vpin 不是 footprint imbalance |
| wick_scene（LTF） | cvd_change_5_pct | Proxy | cvd_change_5_pct 仅弱 proxy |
| trade_cluster_scene | - | Missing | 未覆盖 |
| - | path_efficiency_pct | Extra | HTF 结构过滤 |
| - | price_dir_consistency_pct | Extra | HTF 方向过滤 |
| - | jump_risk_pct | Extra | regime 排除 |
| - | shd_pct / ofci_pct | Extra | 反身性风险过滤 |

---

## ③ Momentum Expansion (ME)

| 旧语义特征 | 当前 Gate 特征 | 状态 | 备注 |
|---|---|---|---|
| trade_cluster_semantic | vpin | Proxy | vpin ≠ cluster 语义 |
| fp_scene | vpin | Proxy | 仍是粗代理 |
| wpt_scene | volume_ratio_pct | Proxy | 成交量放大是弱 proxy |
| liquidity_void_scene | - | Missing | 未覆盖 |
| - | atr_percentile | Extra | 波动扩张过滤 |
| - | bb_width_normalized_pct | Extra | 区间扩张过滤 |
| - | jump_risk_pct | Extra | regime 排除 |
| - | shd_pct / ofci_pct | Extra | 反身性风险过滤 |

---

## ④ Failed Breakout Fade (FBF)

| 旧语义特征 | 当前 Gate 特征 | 状态 | 备注 |
|---|---|---|---|
| wick_scene_semantic_scores | wick_exhaustion_score | Proxy | 名称不同但语义相近 |
| fp_scene（aggressive but stuck） | - | Missing | 核心语义缺失 |
| vpin_scene（VPIN spike） | - | Missing | 核心语义缺失 |
| trade_cluster_scene | - | Missing | 核心语义缺失 |
| liquidity_void_retracement | liquidity_void_false_breakout_risk | Proxy | 语义接近但指标不同 |
| sr_distance_normalized | sr_distance_normalized | Aligned | 一致 |
| - | range_ratio_5bar | Proxy | 仅弱代理 |
| - | cvd_change_5 | Proxy | 订单流背离 proxy |
| - | shd_pct / ofci_pct | Extra | 反身性风险过滤 |
| - | jump_risk_pct | Extra | regime 排除 |

---

## ⑤ Liquidity Sweep → Rejection (LSR)

| 旧语义特征 | 当前 Gate 特征 | 状态 | 备注 |
|---|---|---|---|
| liquidity_void_scene | range_ratio_5bar | Proxy | 语义差距较大 |
| fp_scene（opposite imbalance） | cvd_change_5_pct | Proxy | 弱 proxy |
| trade_cluster_scene | volume_ratio_pct | Proxy | 与 cluster 语义不等价 |
| wick_scene | - | Missing | 未覆盖 |
| - | sr_distance_normalized | Extra | 结构位置过滤 |
| - | jump_risk_pct | Extra | regime 排除 |
| - | shd_pct / ofci_pct | Extra | 反身性风险过滤 |

---

## ⑥ Auction Exhaustion Reversal (AER)

| 旧语义特征 | 当前 Gate 特征 | 状态 | 备注 |
|---|---|---|---|
| trade_cluster_semantic | vpin | Proxy | 语义不等价 |
| wpt_scene | - | Missing | 核心语义缺失 |
| fp_scene | - | Missing | 核心语义缺失 |
| funding_exhaustion_scene_score | - | Missing | 核心语义缺失 |
| - | atr_percentile | Extra | 波动极值过滤 |
| - | path_efficiency_pct | Extra | 结构效率过滤 |
| - | path_length_pct | Extra | 路径长度过滤 |
| - | jump_risk_pct | Extra | regime 排除 |
| - | shd_pct / ofci_pct | Extra | 反身性风险过滤 |

