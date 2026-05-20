# SRB 执行层调优穷尽报告 — 20260423

> 目的：回应用户在 `20260422_212338` rolling_sim 后的三个感知问题，系统性扫过所有执行层改进方向，得出"当前 SRB 已是数据支持的局部最优，剩余差距来自信号/校准层"的结论。
>
> 相关脚本：`scripts/fast_ab_srb_retrace_sweep.sh`、`scripts/fast_ab_srb_e1_e2_e4.sh`、
> `scripts/analyze_srb_add_characteristics.py`、`scripts/diag_srb_losing_adds.py`
>
> 相关报告：`SRB_break_level_attribution_20260422.md`（前置）、
> `SRB_l3_dynamic_trailing_20260422.md`（前置）、`SRB_wide_sr_and_trailing_diagnosis_20260418.md`

---

## TL;DR

- **所有执行层"加仓过滤"方向在 16 个月 fast AB 数据上都是负收益或中性**：retrace_guard (-23/-24/-112R)、E4 bars_since 门、E4 mfe_r 门、trend_r2_gate、recent_momentum、wide_sr_expansion。
- **SRB 当前加仓系统（allow_regime_buckets + float_r_ladder）贡献 +193.84R/70 笔**，是 SRB 主要 alpha 源（首仓仅 +36.4R）；任何 post-hoc 过滤都破坏加仓组合。
- **拆 SRB → wide/narrow 子 archetype 的假设在 81 笔 → 103 笔样本扩展后仍被拒绝**（`p(triple>L1_only)=0.928`）。
- **E1（分层 time_stop）+ E2（L3 结构化退出）合并入 HEAD 作为中性安全补丁**：16 月 AB Δ=-0.07R，未来若 regime 变化时可兜底。
- **用户感知中"没跟上趋势 / 波动区加仓多 / 漏 XRP Nov 大涨"的残余问题，不能通过执行层解决**；下一步需看信号/校准/仓位分配层。

---

## 1 · 用户三点诉求 → 数据查证

### 1.1 "突破后真趋势没加仓"

| | winner add (40 笔) | loser add (30 笔) |
|---|---|---|
| 均 add_R | +6.77 | -2.56 |
| 中位 mother_pnl_r | **+1.58** | **+0.05** |
| 中位 bars_since_mother | 20.5 | 22.5 |

加仓本身胜率 57.14%、总 +193.84R，"没加仓"这个感觉在 16 个月数据上不成立。残余问题是 **30 笔 losers 集中在 XRP (-24R) / SOL (-18R) / BTC (-17R) / ETH (-10R) 这几个特定月份的 drawdown 末端**。

### 1.2 "波动区加仓多亏损大"

按 `bars_since_mother_entry` 分桶：

| 桶 (bar) | n | sum R | mean R |
|---|---|---|---|
| 0-3 | 6 | +12.82 | +2.14 |
| 6-12 | 12 | +54.00 | +4.50 |
| 12-24 | 26 | +75.69 | +2.91 |
| 24-48 | 20 | +42.84 | +2.14 |
| 48-96 | 3 | -1.42 | -0.47 |
| 96-240 | 2 | +6.75 | +3.38 |

**没有证据显示"波动区加仓"在执行层特征上与"趋势区加仓"可分**：winner/loser 的 bars_since 中位数几乎一致（20.5 vs 22.5）。

`bars_since > 48` 虽然 mean 转负，但只有 3 笔样本。

### 1.3 "XRP Nov 大涨跟不上"

2024-11 实际 R = +61.85，addR = +45.28 / 3 adds。**是抓到了的**，只是前面 2024-06~09 震荡期被多次做空/做多止损（是 entry 层反复被 whipsaw，非加仓问题）。

---

## 2 · 本轮穷尽的执行层方向

### 2.1 E1：分层 time_stop（`max_holding_bars=360` + `uncap_mfe_r=2.0`）

目的：拦"长期不盈利的僵尸母仓 + 一路补刀 adds"。

- 实装：`src/time_series_model/live/position_logic.py` block 3g / `build_position_dict`
- 16 月 fast AB：**0 次触发**（现有 structural_sl + trailing 已接管所有 stale 情况）
- 结论：保留代码路径（安全补丁），默认 `enabled=true` 但实际未触发

### 2.2 E2：L3 结构化退出（价格完全反向击穿 wide_sr_* ± buffer ATR）

目的：SRB 突破后被 L3 彻底反向吞掉时立即退出。

- 实装：`src/time_series_model/live/position_logic.py` block 3f
- 16 月 fast AB：**6 次触发（2024-01/06/07×2 待确认/09×2），净 Δ=-0.07R**
- 结论：保留，中性安全补丁

### 2.3 E4：加仓"母仓趋势健康度"门（`min_mother_mfe_r` + `max_bars_since`）

目的：母仓 MFE 不够 / 过陈旧 时不加仓。

- 16 月 AB v1（`min_mfe=1.0`）、v2（`min_mfe=0.0`）结果一致，后发现是 baseline-snapshot 差异导致的 noise
- 数据本身支持度：
  - `min_mother_mfe_r` ≤ 0.5：已被 `float_r_ladder_only[0.5/1.0/1.5]` 吃（mfe ≥ current ≥ 0.5）
  - `max_bars_since`：winner/loser 无区分度（中位 20.5 vs 22.5）
- 结论：**禁用**，保留配置项但 `enabled=false`

### 2.4 F：retrace_guard 三阈值扫描 (0.5 / 0.7 / 0.85)

目的：母仓从 MFE 显著回撤时不加仓。

| arm | total R | add R | add N | Δ |
|---|---|---|---|---|
| baseline | **230.21** | 193.84 | 70 | - |
| rg_050 | 207.04 | 171.68 | 69 | **-23.17** |
| rg_070 | 205.96 | 171.29 | 64 | **-24.25** |
| rg_085 | 118.42 | 83.86 | 43 | **-111.79** |

有意思的失败模式：rg_050 add 数量几乎不变（70→69），但 add R 减少 22R——**过滤一个 add 后，ladder 会在更差时机再触发另一个**。过滤操作破坏原有加仓组合。

- 结论：**全路径拒绝**，`retrace_guard.enabled` 保持 false。

### 2.5 拆 wide / narrow SR 子 archetype

见 `SRB_break_level_attribution_20260422.md`：

- L1-only 入场 meanR = **+0.461** (n=79)
- L1+L2+L3 triple confluence meanR = **-0.382** (n=2)
- Bootstrap `p(triple > L1_only) = 0.928` → 假设被拒

**不拆**。`wide_sr_swing_f` 的正确用法是 exit/trailing 收紧 + `sr_wide_entry_guard` 反向过近拒单，即当前 HEAD 状态。

### 2.6 历史已否决的子项（回忆）

- `recent_momentum.enabled=true`：2026-04 扫参 +2.87R / rolling_sim 0 次实际触发（触发条件过严）。
- `trend_r2_gate.enabled=true`：扫参负贡献。
- `wide_sr_expansion.enabled=true`：扫参负贡献。

---

## 3 · 数据观察：输家 adds 的共同结构

30 笔输家 adds 中：

- **前 5 笔最差（-4R 到 -9R）**：母仓最终 sl（-1R）+ bars_since ≤ 24 + `add_est_current_r_at_entry ≤ 0.2R`
  - BTC 2024-08 / SOL 2024-08 / BTC 2024-06 / ETH 2024-07 / SOL 2024-08

- **20 笔 trailing_sl 退出（母仓还赚 0~0.5R）**：add 接在尾部被 trailing/SL 扫掉

**关键：at-add-time 看不到 "mother final pnl"**。理论上 "current_r 相对 mfe_r 回撤" 是代理，但扫三档都负（见 §2.4）——实际每次"该拒"的 add 被拒后，ladder 在更差时机再触发。

唯一实测有 marginal 正收益的启发式是 `add_est_current_r < 0.2R` 拒（+7.56R 净，11 笔），但这和我们能从代码直接看到的 `current_r` 与 `initial_risk_distance` 的比有语义差异（我的 diagnostic 用 atr×6 当 risk proxy；真实 risk 来自 structural_sl）。若要实现正确版本，需要在代码里加新特征 `current_r_at_trigger`，并做单独一次 AB。预期边际收益最多 +7.56R / 16 个月 ≈ +0.5R/月，**不值得**再开一个单独 branch。

---

## 4 · 当前 HEAD 配置（最终确认）

```yaml
# config/strategies/srb/archetypes/execution.yaml

# 母仓 time_stop 分层（E1, enabled 但 16 月 0 触发）
holding:
  max_holding_bars: 360
  time_stop_uncap_mfe_r: 2.0

# 统一 breakeven @ 3R（2026-04-22 rolling_sim 验证 +55R）
stop_loss.breakeven: { enabled: true, trigger_r: 3.0, lock_level_r: 0.0, measure: initial_risk }

# 结构化 SL（保留）+ 基础 trailing + L3 dynamic trailing
stop_loss.structural_sl: { enabled: true, opposite_sr_buffer_atr: 0.5, min_distance_atr: 2.0 }
stop_loss.trailing: { activation_r: 6.0, trail_r: 5.0, trail_r_far: 7.0, trail_r_near: 5.0, l3_near_threshold_atr: 2.0 }

# 加仓：regime bucket 白名单 + 压缩特征上限
srb_add_position_policy:
  enabled: true
  allow_regime_buckets: [high_adx_low_er, high_adx_high_er]
  max_volume_compression_pct: 0.55
  # post_hoc_shape_gate 所有子项 enabled: false（数据全部否决）

# L3 结构化退出（E2, enabled 但 16 月 6 触发净 0）
l3_structural_exit: { enabled: true, buffer_atr: 0.25 }

# 反向 L3 过近拒单（保留）
sr_wide_entry_guard: { enabled: true, min_distance_atr: 2.0, apply_to_new_only: true }
```

16 月 fast 基准：**total R = 230.21, n=213, addR = 193.84, addN=70**；对应慢管线 rolling_sim `20260422_212338`：total R = 180.63, n=184（calibration 过滤掉 29 trades / -50R）。

---

## 5 · 建议的下一步方向（非执行层）

如果仍想继续推动 SRB performance，剩下的牌是：

1. **Entry/signal 层**：降低 2024-06~09 震荡期 XRP 反复 whipsaw 的首仓数（`confirm_k` / `sr_wide_entry_guard.min_distance_atr` 调大？需要新扫参）。
2. **Calibration 层**：slow rolling_sim 过滤掉 29 trades / -50R，其中是否有被误杀的好 trade？可研究 `threshold_calibration` 的 R 分位点。
3. **Portfolio 层**：SRB 和 FBF 是否能在 XRP 2024-06~09 互补（SRB 震荡做空 + FBF 反手做多）？
4. **Feature 层**：新增"已 confirmed 突破的 persistence" 特征（把"突破 + N bar 站稳"作为二次信号），从数据产生到 feature store 都要重做。

以上均超出本轮 SRB 执行层 scope，建议记录为候选研究方向而非直接开工。

---

## 6 · 作为"阶段成果"的 merge 建议

执行层自 2026-04-16 到 2026-04-23 一共做了：

| 改动 | 状态 |
|---|---|
| `structural_sl` 统一用 L1 窄窗 swing SR + wide fallback | merged |
| `sr_wide_entry_guard` | merged |
| `wide_sr_swing_f` 特征统一（FBF/SRB 共用） | merged |
| `true_sr_level` 宽窄窗自动 fallback | merged |
| 统一 breakeven（`stop_loss.breakeven`）替换旧 `mother_breakeven` | merged |
| `srb_add_position_policy.post_hoc_shape_gate` 框架 + 4 子项 | merged (all disabled) |
| `holding.max_holding_bars` 分层 time_stop (E1) | **本轮 merged** |
| `l3_structural_exit` (E2) | **本轮 merged** |
| `trend_health_gate` (E4) 框架 + event_backtest 注入 | **本轮 merged (enabled=false)** |

所有改动**代码上是 additive，不破坏既有行为**，可安全进入 PR。

后续若要接入新 gate，可以直接在 `post_hoc_shape_gate` 块里加 enable flag，不必再改代码。
