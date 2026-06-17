# DECISION — chop_grid Prefilter Fix

## 假设

| ID  | 假设                                          | Phase A 证据                                        | 决策                            |
| --- | --------------------------------------------- | --------------------------------------------------- | ------------------------------- |
| H0  | P0 fix 后回测结果应比修复前更保守（入场更少） | ✅ A0 完成 — 4 segment 全盈利，maxDD ≤9.21%，无 halt | confirmed — P0 fix 不影响盈利性 |
| H1  | 多 Box 共识过滤可减少假信号、提高胜率         | A1 实验                                             | pending                         |
| H2  | box_compression_score 可捕捉高胜率压缩形态    | A2 实验                                             | pending                         |
| H3  | Wall 静态信号可增强入场时机                   | A3 实验                                             | pending                         |

## A0 基线结果

**P0 fix 后首次完整回测** — backtest_multileg_timeline --no-trend，compound sizing。
5 symbols (BTC/ETH/SOL/BNB/XRP)，constitution: gross≤27000, daily≤600, dd≤20%, sym≤6。

| Segment              | 期间            | 总回报   | MaxDD  | 入场次数 | 拒绝次数 | Fee/Notional  |
| -------------------- | --------------- | -------- | ------ | -------- | -------- | ------------- |
| bear_2022            | 2022-01→2023-11 | +10,962% | -9.21% | 32,967   | 36       | $116K / $291M |
| bull_2023_2024       | 2023-06→2025-01 | +2,952%  | -3.12% | 28,805   | 0        | $29K / $74M   |
| recent_range_to_bear | 2025-01→2026-05 | +1,493%  | -3.98% | 25,285   | 6        | $19K / $48M   |
| recent_6m_oos        | 2025-12→2026-05 | +90.25%  | -3.98% | 8,945    | 0        | $1.2K / $3M   |

**Key observations:**
- P0 fix 后 prefilter 正确工作：bear_2022 有 36 笔拒绝（修复前为 0），确认 box_pos_60 真实过滤
- MaxDD 全部 ≤9.21%，远低于 20% 硬限制
- 所有 segment 无 halt
- 最弱 segment (recent_6m_oos) 仍稳定盈利 +90%/6mo
- 没有 Sharpe 计算（timeline backtest 未输出 Sharpe，需后续补充）

## Promote

- [ ] 三条杠（LAYER_PROMOTION_CRITERIA §1）
- [ ] `monitor_bundle/smoke_report.json` OK
- [ ] `mlbot research promote-baseline` 完成

---

## Phase 1 IC/Label Scan 结果

**数据**: feature_store + forward_rr (long方向, 50bar horizon), 94,332 bars, 5 symbols。
**filter**: `bpc_semantic_chop >= 0.52` → 67,110 samples, base_success=50.37%。

### A1: Box Macro IC (ic-decay)

| Feature | IC@1 | IC@5 | IC@20 | 方向 | 结论 |
|---|---|---|---|---|---|
| box_stability_240 | **-0.040** | -0.037 | -0.025 | NEG | ⚠️ 高稳定性反而预测更差 RR |
| box_pos_240 | **+0.043** | +0.033 | +0.008 | POS | 弱正 — 高位略好 |
| box_width_pct_240 | +0.003 | -0.001 | -0.017 | — | noise |
| box_stability_480 | +0.009 | +0.008 | +0.006 | POS | 弱正 |
| box_pos_480 | **+0.027** | +0.022 | +0.006 | POS | 中等正 |
| box_width_pct_480 | **-0.036** | -0.036 | -0.036 | NEG | 强负 — 宽盒子预测更差 RR |

### A1: Box Stability Quartile 分析 (chop only)

| 稳定性分位 | 范围 | mean RR | pos_rate | 含义 |
|---|---|---|---|---|
| Q1 (低) | 0.038-0.314 | **+0.60** | **53.8%** | ✅ 最好 |
| Q2 | 0.314-0.573 | +0.53 | 51.5% | ✅ |
| Q3 | 0.573-0.856 | -0.08 | 47.1% | ❌ |
| Q4 (高) | 0.856-1.00 | **-0.46** | **48.2%** | ❌ 最差 |

**核心发现**: box_stability_240 与 forward_rr **强负相关**。越"稳定"的盒子，后续收益越差。
原因分析：高稳定性 = 价格在窄幅震荡 → 动能耗尽 → 即将突破（对 grid 不利）。

### A1: Plateau 结论

- box_stability_240 plateau: 所有阈值 succ_hit < base，无 lift，确认负向
- box_pos_240 plateau: 阈值 0.35-0.40 succ_hit ≈ base (|z|<1)，无显著 lift

### A2: Compression Score

- IC@1 = -0.015 (弱负)
- plateau: compression_score ≥ 1.0 仅 562 samples, succ_hit = 40% (远低于 base 50.4%)
- compression_score 分布：p50=0.46, p90=0.73, ≥1.0 极稀有 (0.8%)

### Phase 1 结论 — 变体计划

| 变体 | 假设 | 修改 | 依据 |
|---|---|---|---|
| A1 | 低稳定性盒子更适合 grid | box_stability_240 < 0.4 过滤 | Q1 mean_rr=+0.60 vs Q4=-0.46 |
| A1b | 大尺度 box_pos 可辅助 | box_pos_480 ∈ [0.3, 0.7] | IC@1=+0.027 |
| A2 | compression_score 无价值 | 不使用 | IC 弱负，高值稀有且胜率低 |
| A3 | Wall 待 build | 待定 | 需先 build wall features |

**⚠️ 重要**: A0 的 box_stability_240 未用作过滤条件（当前只有 box_pos_60 ∈ [0.4,0.6]），
所以 A0 结果不受此发现影响。但 A1 变体可以通过加入 stability 方向过滤来提升。
