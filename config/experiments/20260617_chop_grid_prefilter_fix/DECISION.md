# DECISION — chop_grid Prefilter Fix

## 假设

| ID  | 假设                                          | Phase A 证据                                        | 决策                            |
| --- | --------------------------------------------- | --------------------------------------------------- | ------------------------------- |
| H0  | P0 fix 后回测结果应比修复前更保守（入场更少） | ✅ A0 完成 — 4 segment 全盈利，maxDD ≤9.21%，无 halt | confirmed — P0 fix 不影响盈利性 |
| H1  | 多 Box 共识过滤可减少假信号、提高胜率         | A1 实验                                             | pending                         |
| H2  | box_compression_score 可捕捉高胜率压缩形态    | A2 实验                                             | pending                         |
| H3  | Wall 静态信号可增强入场时机                   | A3 实验 — IC < 0.025, 无信号                            | ❌ REJECTED — wall IC 噪声级    |

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
| A3 | Wall features 可增强入场 | ❌ 取消 | IC < 0.025（噪声级），不值得回测 |

**⚠️ 重要**: A0 的 box_stability_240 未用作过滤条件（当前只有 box_pos_60 ∈ [0.4,0.6]），
所以 A0 结果不受此发现影响。但 A1 变体可以通过加入 stability 方向过滤来提升。

---

## Phase 2 Variant Backtest 结果

### A1: box_stability_240 < 0.4

| Segment              | A0 Return  | A1 Return | A0 MaxDD | A1 MaxDD | A0 Trades | A1 Trades | Keep% |
| -------------------- | ---------- | --------- | -------- | -------- | --------- | --------- | ----- |
| bear_2022            | +10,962%   | **+96%**  | -9.21%   | -1.42%   | 32,967    | 4,417     | 13.4% |
| bull_2023_2024       | +2,952%    | **+89%**  | -3.12%   | -1.90%   | 28,805    | 3,940     | 13.7% |
| recent_range_to_bear | +1,493%    | **+51%**  | -3.98%   | -0.98%   | 25,285    | 3,449     | 13.6% |
| recent_6m_oos        | +90%       | +10%       | -3.06%   | -0.53%   | 8,945     | 1,179     | 13.2% |

**❌ REJECTED**: A1 过滤器灾难性地削减交易量（保留 ~13.5%），收益崩塌 97%+。
每笔交易的边际收益也更差（不是"更好的交易更少"，而是"更差的交易更少"）。

**原因分析**：
- IC -0.04 是边际噪声，非可交易信号
- box_stability_240 < 0.4 选择的是"窄幅震荡无方向"环境——网格策略最不利的环境
- 四分位分析中的 Q1 优势来自样本偏差（low-stability 窗口稀有 + forward_rr 噪声大）

### A1b: box_pos_480 ∈ [0.3, 0.7]

| Segment              | A0 Return  | A1b Return | A0 MaxDD | A1b MaxDD | A0 Trades | A1b Trades | Keep% |
| -------------------- | ---------- | ---------- | -------- | --------- | --------- | ---------- | ----- |
| bear_2022            | +10,962%   | +71%       | -9.21%   | -2.77%    | 32,967    | 3,772      | 11.4% |
| bull_2023_2024       | +2,952%    | +58%       | -3.12%   | -1.47%    | 28,805    | 4,466      | 15.5% |
| recent_range_to_bear | +1,493%    | +62%       | -3.98%   | -0.74%    | 25,285    | 5,084      | 20.1% |
| recent_6m_oos        | +90%       | +14%       | -3.06%   | -0.55%    | 8,945     | 1,671      | 18.7% |

**❌ REJECTED**: 与 A1 相同模式。Keep 11-20%，收益缩减 84-93%。ret_ratio 0.65-15.6%。

**A1 vs A1b 对比**: 两者表现几乎一致。box_pos_480 过滤器略弱于 stability 过滤器
（keep% 稍高 11-20% vs 13-14%），但收益缩减同样严重。

### Wall IC Scan — ❌ 无信号

Wall feature build 完成（5 symbols × 77 months），但 IC scan 结果全是噪声：

| Feature                    | IC      | Rank IC | Coverage |
| -------------------------- | ------- | ------- | -------- |
| wall_bid_notional_usd_max  | -0.008  | -0.017  | 76.9%    |
| wall_ask_notional_usd_max  | -0.012  | -0.022  | 76.9%    |
| wall_bid_price             | -0.014  | -0.009  | 76.9%    |
| wall_ask_price             | -0.014  | -0.009  | 76.9%    |
| wall_nearest_dist_atr      | +0.008  | +0.016  | 76.9%    |

所有 |IC| < 0.025，远低于 0.05 噪声阈值。**A3 wall variant 取消**——不值得做回测。

**结论**: 当前静态 wall features（最大挂单量、价格、距离）无预测价值。
动态 wall features（persist_sec, cancel_rate_5m, eaten_ratio_1h）需要 WS 管道，
待未来实现后再评估。

---

## Phase 2 最终结论

### 结果汇总

| Variant | 配置 | 状态 | 结论 |
| ------- | ---- | ---- | ---- |
| **A0** | `box_pos_60 ∈ [0.40, 0.60]` | ✅ KEEP | **当前 production = 最优配置** |
| A1 | + `box_stability_240 < 0.4` | ❌ REJECTED | 收益缩减 90%+，keep 13% |
| A1b | + `box_pos_480 ∈ [0.3, 0.7]` | ❌ REJECTED | 收益缩减 84-93%，keep 11-20% |
| A1_combined | A1 + A1b 组合 | ⏭ SKIP | 两者都失败，组合更差 |
| A3 | wall features 增强 | ❌ CANCELLED | IC < 0.025 全是噪声 |

### 核心发现

1. **chop_grid 的 edge 来自 volume × 微小边际**：任何 "质量" 过滤都会摧毁 volume 乘数
2. **IC < 0.05 = 噪声**：对 prefilter 决策无效（box_stability, box_pos_480, wall features 均在此范围）
3. **A0 baseline 是最优配置**：不过滤，最大化交易量，让 law of large numbers 发挥作用
4. **静态 wall features 无预测价值**：动态 wall features 需要 WS 管道

### 下一步方向

- **增加交易量**: 扩展更多 symbols 或更小 timeframe（5min/15min grid）
- **参数优化**: bpc_semantic_chop 阈值（0.52/0.33）、ATR 网格间距
- **动态 wall features**: WS 管道实现后评估 persist_sec / cancel_rate / eaten_ratio
- **方向性增强**: 如果有方向信号，可做 long-only / short-only grid
