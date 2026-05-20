# SRB 单旋钮 Ablation 结果（20260418）

> 目的：针对用户 4/17 反馈（二期 SR 难识别、趋势段 trailing 过紧、加仓收益有限），
> 把之前一次性合入的四项改动**一个一个**验证。
>
> 方法：fast-replay — 复用 `20260417_163432` rolling 的每月 `strategies_calibrated/srb/`，
> 仅替换 `execution.yaml` 内的目标字段，其他层（feature store、model、gate、calibration）
> 与 baseline 完全一致，从而把变化全部归因到 `execution.yaml` 的那一个字段。
>
> 对齐验证：用 `baseline` 配置 replay 2023-09 → 5 笔 / −10.823 R，与 rolling 原始 5 笔 / −10.82 R 完全一致（误差 < 1e-3）。

## Baseline run
`results/srb/research_roll.features_on/_rolling_sim/20260417_163432`（合并态，包含两阶段反手 + fake_break_reverse 修复）。

## 实验矩阵

| tag  | 改动（其他全部与 baseline 相同）                                                               |
|------|-------------------------------------------------------------------------------------------|
| exp1 | `sr_feature_injection.swing_lookback_wide_bars: 96` — 只注入宽窗 SR 特征，**不进决策链**          |
| exp2 | `trailing.expand_with_primary_atr: true` — trailing 带宽 = max(入场ATR, 当前主周期ATR)            |
| exp3 | `trailing.activation_r: 7.0 / trail_r: 6.0`（默认从 6.0/5.0 放宽）                              |
| exp4 | `srb_add_position_policy.allow_regime_buckets` 新增 `low_adx_high_er`                         |

## 结果（16 个月全程回放 / ablation CSV: `results/srb/diag/ablation_fast_20260418/COMPARE.md`）

| tag | n | total_R | mean_R | win_rate | sl (n / mean_R) | trailing_sl (n / mean_R) | add (n / mean_R) | reverse (n / mean_R) |
|---|---|---|---|---|---|---|---|---|
| baseline | 192 | +161.24 | +0.840 | 47.9% | 134 / −0.11 | 52 / +1.58 | 77 / +1.46 | 23 / +1.39 |
| exp1 | 192 | +161.24 | +0.840 | 47.9% | 134 / −0.11 | 52 / +1.58 | 77 / +1.46 | 23 / +1.39 |
| **exp2** | **162** | **+671.69** | **+4.146** | **52.5%** | 107 / +0.51 | 45 / +1.52 | 68 / **+6.89** | 19 / **+9.74** |
| exp3 | 155 | +138.00 | +0.890 | 43.2% | 114 / −0.15 | 35 / +1.75 | 61 / +1.73 | 23 / +1.24 |
| exp4 | 228 | +150.54 | +0.660 | 49.6% | 171 / −0.08 | 50 / +1.69 | 121 / +0.77 | 22 / +1.61 |

### Δ vs baseline

| tag | Δn | ΔR | Δwin | Δtr mean_R | Δtr count | Δadd mean_R | Δadd count |
|---|---|---|---|---|---|---|---|
| exp1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **exp2** | **−30** | **+510.45** | **+4.6pp** | −0.060 | −7 | **+5.436** | −9 |
| exp3 | −37 | −23.24 | −4.7pp | +0.172 | −17 | +0.277 | −16 |
| exp4 | +36 | −10.70 | +1.7pp | +0.116 | −2 | −0.689 | +44 |

### EOB 拆分（开仓持有到 2024-12-31 强平）

| tag | closed (exclude EOB) n / ΣR / mean_R / win | EOB only n / ΣR |
|---|---|---|
| baseline | 186 / +67.83 / +0.365 / 46.8% | 6 / +93.41 |
| **exp2** | **152 / +122.88 / +0.808 / 50.0%** | **10 / +548.82** |

**结论**：exp2 即使只算"真正平仓"的部分也明显优于 baseline（ΣR +81%、mean_R 2.2×、win +3.2pp），不只是被"2024 年末强平的大单"拉高。

## 逐个解读

### exp1（宽窗 SR 注入）—— 0 影响
宽窗 SR 特征已注入 `funnel`，但当前 SRB 的 prefilter / gate / entry_filter / fake_break_reverse 都**没有**消费 `srb_sr_support_wide / srb_sr_resistance_wide`。所以结果和 baseline 位位相等 —— 也就是说：**"注入特征"本身是安全的、无副作用的**，可以长期保留作为"诊断埋点"；要用在决策上需要单独设计。

数据回答你的问题：宽窗 SR 到底是什么、有没有用？
- 在同一个 `20260417_163432` rolling 的 192 笔交易上（`wide_sr_and_trailing_diag.csv`）：
  - 入场到**窄窗**（20 条 2H bar）SR 反向位的中位数距离 = 4.36%
  - 入场到**宽窗**（96 条 2H bar ≈ 8 天）SR 反向位的中位数距离 = 9.81%（2.25× 窄窗）
  - 87% 的交易里宽窗 SR 至少比窄窗 SR 远 5% 以上
- 说明：窄窗 SR 只能看到最近 ~40 小时的结构；8 天宽窗识别的是真正的"二期"大级别支撑阻力。
- 但我们**还没把它接进决策**，所以目前它是被动指标。下一步是否接入，见"建议"。

### exp2（trailing 自适应 ATR）—— **显著改进**
`trailing_stop_distance = trail_r × max(atr_at_entry, current_primary_tf_atr)`。

- 为什么有效（用 `wide_sr_and_trailing_diag.md`）：在被 trailing_sl 打掉的 52 笔交易里，**出场 bar 的 ATR 是入场 ATR 的 1.51×（中位数）**，且出场后沿原方向继续走的中位数 MFE = **2.24 ATR**；53.8% 的 trailing_sl 出场后 ≥ 2 ATR。也就是说**一半以上的 trailing_sl 是"洗盘"，不是真反转**。
- 效果：trailing 带宽随波动扩大后：
  - 大趋势被继续持有（XRP/ETH LONG 从 2024-04 ~12 月的持续多头，baseline 早就 trailing_sl 了）→ EOB 强平时 ΣR +548（baseline 93）。
  - 即使"洗盘"场景不出现（小区间行情），adaptive 也不比 baseline 紧，因为是取 max。所以 mean_R 在"真正平仓"的子集里仍是 **+0.808 vs baseline +0.365**，没有负作用。
  - 反手 / 加仓的 mean_R 同步大幅抬升（+9.74 / +6.89 vs +1.39 / +1.46），因为它们开在趋势段，趋势被更多地兑现。

**结论：可上线。** 这是四项改动里唯一一个"用数据强支持、跨场景无明显损失、单独生效"的开关。

### exp3（默认放宽 activation_r=7 / trail_r=6）—— 不推荐单独上
- 交易数下降 37（192 → 155），trailing_sl 减少 17 笔，说明确实筛掉了一部分。
- 但 ΣR 小幅下降 23 R。mean_R 略好 (+0.84 → +0.89) 来自剔除低质量 trailing；可惜筛掉的里面也有真正的 trailing 获利。
- 这是"死参数"方案，exp2 的"活参数"（按 ATR 变化）显然更优。建议在 exp2 上线后**不再叠加 exp3**。

### exp4（加仓新增 low_adx_high_er bucket）—— 单独上线为负
- 加仓笔数 +44（77 → 121），新增了大量在 ADX 未起来但 ER 已抬升的阶段的加仓。
- 但这些加仓的 mean_R 只有 +0.77（baseline 加仓 mean_R +1.46）—— 新增的加仓质量明显低。
- 总体 ΣR −10.70 R，win rate 略升（+1.7pp）。
- 解读：low_adx_high_er 阶段还没真正进趋势，加仓容易被洗。如果和 exp2（adaptive ATR trailing）组合，可能这些"早加"的仓位能被 adaptive trailing 救回来 —— 但那是 exp2+exp4 的组合实验，**目前单独上 exp4 是净负收益**。

## 下一步建议

### 1. 立即合入 exp2（adaptive ATR trailing）
改动已经在代码里（`position_logic.py` + `event_backtest.py` + `position_tracker.py` + `generic_live_strategy.py`），只需把 `config/strategies/srb/archetypes/execution.yaml` 的 `trailing.expand_with_primary_atr: true` 保留。

**单一开关**，**显著改进**，**有数据解释**（1.51× ATR 扩张 + 53.8% 洗盘率）。

### 2. exp3（默认再放宽）回退
把 `activation_r / trail_r` 回到 **6.0 / 5.0**（baseline 值）。exp2 已经动态扩展，默认值不必再压一层保守。

### 3. exp4 先不合入
在 exp2 合入后做 exp2+exp4 组合实验，观察 adaptive trailing 能否救回 low_adx_high_er 加仓。

### 4. 宽窗 SR：保留被动注入，计划把它接进决策
proposals（排优先级）：
- **a)** 把 `srb_sr_support_wide / srb_sr_resistance_wide` 加进 `fake_break_reverse` 的 `true_sr_level` 选择：如果窄窗 SR 和近期入场价太近（可能是小级别假突破的噪声），fallback 到宽窗 SR。
- **b)** 给 `srb_sr_support_wide / srb_sr_resistance_wide` 加一条 `prefilter` 规则：禁止在离宽窗 SR 反向位过近（例如 < 2 ATR）的位置开仓 —— 可以用之前诊断报告里"87% 交易到宽窗 SR 超 5%"做阈值校准。

这两条都需要再跑一次 ablation，会单独立项。

### 5. 固化 replay 框架
修复后的 `scripts/srb_diag/replay_event_backtest.py` 现在可以在 ~80 min（4 路并行）给出单旋钮 ablation 全程结果，比 rolling 全跑 4.5h 每个快 ~4×。后续所有 SRB（以及其他 archetype）execution 层调参，都可以先用它验证。

## 文件产物
- 汇总对比：`results/srb/diag/ablation_fast_20260418/COMPARE.md`
- 每实验每月 json/csv：`results/srb/diag/ablation_fast_20260418/{exp1..exp4}/month_YYYY-MM/`
- 实验执行脚本：`scripts/srb_diag/replay_event_backtest.py`（已修好 2 处 regex bug）
- 汇总脚本：`scripts/srb_diag/summarize_ablation.py`
- 宽窗 SR + trailing 诊断：`scripts/srb_diag/wide_sr_and_trailing_diag.py` → `results/srb/diag/wide_sr_and_trailing/`
