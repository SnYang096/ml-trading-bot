# Orderflow 语义化映射模板库（可复用）

目的：把“raw 统计量特征”变成“机制一致的语义分数（semantic scores）”，降低异义性冲突，提升可解释性与跨 regime 稳健性。

适用对象：树模型 / NN（尤其是样本不足时），以及四类策略：
- SR 反转（Exhaustion/Reversal）
- SR 突破（Ignition/Breakout）
- 压缩突破（Compression + Ignition）
- 趋势（Absorption/Continuation）

---

## 1) 模板：TradeCluster → Exhaustion / Absorption（已落地）

### Raw 痛点
`trade_cluster_*` 同时携带：
- 趋势延续语义（突破/趋势里是正）
- 反转失败语义（反转里是负）

### 语义化输出
特征节点：`trade_cluster_semantic_scores_f`
- `trade_cluster_flow_intensity`：流强（活动 + 单边性）
- `trade_cluster_exhaustion_score`：流强高但位移小（effort without progress）
- `trade_cluster_absorption_score`：流强高且位移大（顺势吸收/延续）

实现位置：
- `src/features/time_series/utils_order_flow_features.py`
- `config/feature_dependencies.yaml`

---

## 2) 模板：LiquidityVoid → Sweep / Failure（已具备原料）

### 原料（已存在）
`liquidity_void_f` 输出：
- `liquidity_void_detected`
- `liquidity_void_speed`
- `liquidity_void_volume_ratio`
- `liquidity_void_price_impact`
- `liquidity_void_retracement`
- `liquidity_void_false_breakout_risk`

### 建议语义化
- **sweep_void_score**（真空/扫除强度）：`detected × speed × price_impact`
- **failure_score**（失败/反包概率）：`false_breakout_risk`（或 `detected × retracement`）

---

## 3) 模板：VPIN → Stress / Directional Pressure（建议落地）

### Raw 痛点
VPIN 高既可能是：
- “有毒流动性/高冲击风险”（不一定可交易）
也可能是：
- “方向性订单流”（可交易）

### 建议语义化拆分
使用 `vpin_zscore_features_f` + `vpin_signed_zscore_features_f` 的输出：
- **vpin_stress_score**：`clip(|vpin_zscore_50|) → 0..1`
- **vpin_directional_pressure**：`clip(vpin_signed_imbalance_zscore_50) → -1..1`（或分正负两列）
- **vpin_shock_score**：用 spike_flag / volatility（突发 vs 持续）

然后再按四场景组合：
- breakout/趋势：pressure 同向 + shock/持续
- 反转：stress 高 + price progress 变弱（类似 exhaustion）

实现状态（已落地第一版）：
- `vpin_semantic_scores_f`（输出三列）：
  - `vpin_stress_score`
  - `vpin_directional_pressure`
  - `vpin_exhaustion_score = vpin_stress_score × (1 - disp_norm) × sr_weight`
    - `disp_norm = clip( (|high-low|/ATR) / disp_atr_threshold, 0..1 )`
    - `sr_weight = 1 - clip( dist_atr / sr_prox_atr, 0..1 )`
    - `dist_atr = |dist_to_nearest_sr| * close / ATR`


---

## 4) 模板：CVD → Divergence（建议落地）

### Raw 痛点
高 CVD 既可能是趋势，也可能是出货/吸收。

### 语义化方向
做成“价格响应 vs 资金流响应”的背离：
- **bullish_divergence_score**：价格创新低，但 CVD 不创新低（卖压衰竭）
- **bearish_divergence_score**：价格创新高，但 CVD 不创新高（买压衰竭）

实现可不需要 ticks（DataHandler 常带 `cvd`）。

---

## 4.5) 模板：Imbalance（bar-level）→ Exhaustion（已落地一版）

当没有 L2 depth / stacked imbalance 时，我们用 bar-level 的 `taker_buy_ratio` 作为简化不平衡代理：
- `imbalance_ratio = (taker_buy_ratio - 0.5) * 2`（归一化到 [-1, 1]）
- `imbalance_exhaustion_score = |imbalance_ratio| × (1 - disp_norm) × compression_gate`
  - `compression_gate = clip(compression_score, 0..1)`
  - `disp_norm` 同上（ATR 归一化位移）

实现状态（已落地第一版）：
- `tbr_imbalance_semantic_scores_f`（输出两列）：
  - `imbalance_ratio`
  - `imbalance_exhaustion_score`

---

## 5) 模板：组合语义（Interaction）

原则：只做“机制一致”的加乘，避免再引入异义。

示例（已落地原型）：
- **Exhaustion_at_Liquidity_Void**：
  `trade_cluster_exhaustion_score × liquidity_void_detected × liquidity_void_false_breakout_risk`
  （特征节点：`exhaustion_at_liquidity_void_f`）

用途：
- SR 反转：在 “真空 + 衰竭” 出现时加强信号

---

## 6) 实验建议（如何验证模板有效）

1) 每次只新增一个语义节点（或一组很小的语义节点）
2) 固定窗口/timeframe/test-size，跑 seeds=1..5
3) 至少在 BTC + ETH 复核一次（防止单品种偶然）


