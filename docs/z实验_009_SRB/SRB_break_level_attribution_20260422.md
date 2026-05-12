# SRB 破位级别归因诊断 — 20260422

> 目的：回答核心疑问——**"L3 破位的 SRB trade 真的不同吗？是否值得拆成独立子 archetype？"**
>
> 脚本：`scripts/analyze_srb_break_levels.py`
>
> 数据：`results/srb/research_roll.features_on/_rolling_sim/20260421_222624/`（rolling_sim 运行中，当前覆盖 2023-09 → 2024-09）
>
> Feature store：`feature_store/features_srb_120T_5643a66b47`（含新 `wide_sr_upper_px` / `wide_sr_lower_px`）

---

## TL;DR

- **不拆 L3 子 archetype**。按当前 13 个月数据（81 笔首单）：
  - `triple-confluence (L1+L2+L3)` meanR = **−0.38**（n=2），**显著差于** `L1-only` meanR = **+0.46**（n=56）
  - Bootstrap 单尾 `p(triple > L1-only) = 0.928`，即"triple 更强"这个假设完全没有数据支持
- 全 trade（含 add_position）看起来 triple-confluence 好，但拆开后**完全是 add_position 加仓杠杆复利**的功劳，不是入场 edge。
- `wide_sr_swing_f` 目前的用法（structural SL fallback 锚点 + `sr_wide_entry_guard` 拒单）就是**正确 ceiling**；不要升级为"选股信号"。
- 2024-10/11/12 跑完后复测一次，样本会接近翻倍。

---

## 方法

- **L1 confluence**（全员）：SRB 每笔 trade 都已通过 20-bar 窄窗 swing 突破确认。
- **L2 confluence**（入场 bar）：`|dist_to_nearest_sr| × entry_price / atr ≤ 1.0`（~160-bar POC 距离 ≤ 1 ATR）。
- **L3 confluence**（入场 bar）：`wide_sr_dist_atr ≤ 1.5` 且 `wide_sr_side` 对齐突破方向（LONG→side=+1；SHORT→side=−1）。

四组：`L1_only` / `L1+L2` / `L1+L3` / `L1+L2+L3`。

Bootstrap 2,000 样 CI95（meanR），5,000 样单尾 p（组 A meanR > 组 B meanR 的 bootstrap 概率）。

---

## 首单（剔除 add_position + reverse）n=81

| 组 | n | totalR | **meanR** | median | win | bars_held (min) | CI95(meanR) |
|---|---|---|---|---|---|---|---|
| **L1_only** | **56** | +25.84 | **+0.461** | −1.01 | 0.393 | 15,669 | [−0.17, +1.16] |
| L1+L2 | 14 | −1.68 | −0.120 | −0.50 | 0.500 | 12,536 | [−0.65, +0.58] |
| L1+L3 | 9 | −0.37 | −0.041 | −1.01 | 0.333 | 10,121 | [−1.01, +1.03] |
| L1+L2+L3 | 2 | −0.76 | −0.382 | −0.38 | 0.500 | 41,260 | [−1.01, +0.24] |

Bootstrap 单尾 p：

- `p(L1+L2+L3 > L1_only) = 0.928`
- `p(L1+L2   > L1_only) = 0.894`
- `p(L1+L3   > L1_only) = 0.805`

**三个 confluence 组没有一个显著好于 L1-only，趋势反而更差。**

---

## 全 trade（含 add_position）n=147 — 对比视角

| 组 | n | meanR | win | size× | **add%** |
|---|---|---|---|---|---|
| L1_only | 77 | +1.432 | 0.43 | 0.88 | 0.27 |
| L1+L2 | 36 | +0.696 | 0.39 | 0.74 | 0.61 |
| L1+L3 | 25 | +1.690 | 0.44 | 0.72 | 0.64 |
| L1+L2+L3 | 9 | +2.285 | 0.33 | 0.63 | **0.78** |

Triple-confluence meanR 高全来自"活下来 + 加仓多 + 复利"，不是入场信号的 alpha。

---

## 阈值扫描（首单）

**`wide_sr_dist_atr`**：

| threshold | near n | near meanR | near win | far n | far meanR | far win |
|---|---|---|---|---|---|---|
| 0.25 | 2 | +2.048 | 1.000 | 79 | +0.240 | 0.392 |
| 0.50 | 6 | +0.010 | 0.333 | 75 | +0.306 | 0.413 |
| 1.00 | 9 | −0.041 | 0.333 | 72 | +0.325 | 0.417 |
| 1.50 | 11 | −0.103 | 0.364 | 70 | +0.345 | 0.414 |
| 2.00 | 15 | +0.361 | 0.467 | 66 | +0.267 | 0.394 |
| 3.00 | 23 | +0.590 | 0.435 | 58 | +0.163 | 0.397 |

→ 仅 `< 0.25 ATR` 时 meanR 飙到 +2.05（n=2，样本不够），其他阈值与 far 几乎无差。

**`narrow_dist_atr`（L2 POC 距离）**：首单完全二峰分布——16 笔贴 L2（≤0.25 ATR，meanR=−0.15） + 60 笔远离 L2（>3 ATR，meanR=+0.29）；**贴 L2 反而是轻微负信号**。

---

## 结论与后续

1. **不拆 L3 子 archetype**；"L3 破位 ≠ L1 破位"的假设在当前数据上被拒绝。
2. 当前 `wide_sr_swing_f` 的两个落地位置是上限：
   - `true_sr_level.wide_fallback_atr`：窄窗太近时 structural SL 换锚
   - `sr_wide_entry_guard.min_distance_atr`：反向 L3 边界太近时拒单
3. **潜在增益点（数据指向）**：不是"入场筛 L3"，而是"**用 L3 做 exit/trailing 收紧**"——当 wide_sr 接近时把 trailing 乘数收紧，把 triple-confluence 长期复利的尾部风险锁住。这是下一个实验方向。
4. **待办**：rolling_sim 跑完 2024-10/11/12 后复测：

```bash
python scripts/analyze_srb_break_levels.py \
  --run-dir results/srb/research_roll.features_on/_rolling_sim/20260421_222624 \
  --feature-store feature_store/features_srb_120T_5643a66b47 \
  --out reports/srb_break_level_attribution_firstentry_v2.json \
  --filter-add-reverse
```

若复测 `p(triple > L1_only)` 仍 > 0.3，彻底搁置 L3 子 archetype 方案。

---

## 产物文件

- `reports/srb_break_level_attribution.json`（全 trade）
- `reports/srb_break_level_attribution_firstentry.json`（首单）
- `reports/srb_break_level_attribution_firstentry_trades.parquet`（enriched 明细）
- `scripts/analyze_srb_break_levels.py`（可复用；`--l2-atr-threshold` / `--l3-atr-threshold` 可调）

---

## V2 复测 — rolling_sim 完整 16 个月（20260422）

Rolling_sim 跑完，`stitched_summary.json`：**16 个月，187 笔 trade，totalR = +205.41**。
Feature store `5643a66b47` 写到 2024-11（2024-12 只在 event_backtest inline 计算、未落盘），7 笔 2024-12 首单 enrich 失败被丢；最终 103 笔首单参与分组。

### 首单 n=106（有效 103）

| 组 | n | totalR | **meanR** | win | CI95(meanR) |
|---|---|---|---|---|---|
| **L1_only** | **79** | +31.60 | **+0.400** | 0.443 | [−0.07, +0.94] |
| L1+L2 | 14 | −1.68 | −0.120 | 0.500 | [−0.65, +0.58] |
| L1+L3 | 11 | +1.57 | +0.143 | 0.455 | [−0.65, +0.99] |
| L1+L2+L3 | 2 | −0.76 | −0.382 | 0.500 | [−1.01, +0.24] |

Bootstrap 单尾 p：

- `p(L1+L2+L3 > L1_only) = 0.931`
- `p(L1+L2   > L1_only) = 0.897`
- `p(L1+L3   > L1_only) = 0.693`

→ **样本从 81 扩到 103 结论未变，三级 confluence 组仍没有一个显著优于 L1-only，triple confluence 反而偏负。**

### 全 trade n=187

| 组 | n | meanR | win | add% |
|---|---|---|---|---|
| L1_only | 113 | +1.031 | 0.49 | 0.30 |
| L1+L2 | 36 | +0.696 | 0.39 | 0.61 |
| L1+L3 | 29 | +1.494 | 0.45 | 0.62 |
| L1+L2+L3 | 9 | +2.285 | 0.33 | 0.78 |

`p(triple > L1_only, all-trades) = 0.303` — 仍不显著。high meanR 仍主要来自 `add%=0.78` 的加仓复利。

### `wide_sr_dist_atr` 扫描（全 trade）

| threshold | near n | near meanR | far n | far meanR |
|---|---|---|---|---|
| 1.00 | 31 | +1.525 | 153 | +1.042 |
| 2.00 | **52** | **+2.074** | 132 | +0.749 |
| 3.00 | 71 | +1.498 | 113 | +0.888 |

→ 贴近 L3 (< 2 ATR) 的 trade **总收益更好，但 addpct 也更高**，仍是"活下来被加仓"而非入场 edge。

### 最终 verdict

1. **L3 子 archetype 彻底搁置**：13 个月到 16 个月的样本扩充没有改变结论。
2. `wide_sr_swing_f` 保留当前两种用法（structural SL 锚点 + entry guard），这是正确的 ceiling。
3. 下一步研究方向（数据真正指向）：
   - **L3 exit/trailing**：价距离反向 L3 边界收敛时（`wide_sr_dist_atr` 快速下降）主动收紧 trailing；这能把 triple-confluence 的长尾下行风险锁住。
   - **加仓管理**：`add%` 与长期 PnL 强相关，但首单信号与 group 无关 → 把资源压在"add_position 的 trigger 条件和 size scaling"，而不是入场选股。
4. 复测产物：
   - `reports/srb_break_level_attribution_v2.json`（首单）
   - `reports/srb_break_level_attribution_v2_alltrades.json`（全 trade）
   - `reports/srb_break_level_attribution_v2_trades.parquet`
   - `reports/srb_break_level_attribution_v2_alltrades_trades.parquet`
