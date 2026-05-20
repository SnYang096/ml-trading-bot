# SRB：宽窗 SR 与 trailing 动态化 — 数据诊断与验证计划

- 日期：2026-04-18
- Baseline run：`results/srb/research_roll.features_on/_rolling_sim/20260417_163432`（修完 ReverseIntent bug 后的 16-month rolling，192 笔 / +161.24 R / 52 trailing_sl / 23 reverse）
- 诊断脚本：`scripts/srb_diag/wide_sr_and_trailing_diag.py`
- 诊断产物：`results/srb/diag/wide_sr_trailing_20260418/`
- Ablation 驱动：`scripts/srb_diag/run_ablation.sh`（full rolling_sim，每 exp ~4.5h）

---

## 1. 用户反馈还原

> "是有加仓，但似乎没有改进多少，二期一些明显的支撑阻力位置无法识别到，我给你一些图，你看看；而且有的时候抓到趋势了，却很快 trailing stop 了，能否帮忙改进"

两个核心抱怨：
1. **SR 识别太窄**：二期（宽幅整理）的明显结构位，当前 20 根 2H lookback 识别不到。
2. **Trailing 偏紧**：抓到趋势但被 trailing 洗出。

---

## 2. 宽窗 SR（wide SR）是什么 & 有用吗

**定义**：
- 窄窗（现有 `srb_sr_support / srb_sr_resistance`）= 最近 **20 根 2H** bar 的 min(low) / max(high)，约 40 小时。
- 宽窗（新增 `srb_sr_support_wide / srb_sr_resistance_wide`）= 最近 **96 根 2H** bar 的 min(low) / max(high)，约 **8 天**。

**数据**（192 笔 SRB 交易，192/192 样本 narrow+wide 都可用）：

| 指标 | Narrow (20 bar) | Wide (96 bar) |
|---|---|---|
| 入场 → 同向 SR 距离 中位数 | 4.36% | **9.81%** |
| p90 | 9.34% | **19.43%** |
| wide 比 narrow 远 ≥5% | — | **167/192 (87.0%)** |

**结论**：宽窗 SR 是一个和窄窗结构上不同的信号（只有 13% 情况下重合，说明多数时候窄窗只是局部 micro-range，宽窗才接住"二期"整理带）。

**但是**：当前代码只做到把它注入为 feature（passive），尚未进决策链——想真正改善业绩必须后续再加一轮"用宽窗作 `fake_break_reverse.true_sr_level` 的取数源"（Exp 6，待用户拍板后再做）。

---

## 3. Trailing 应该跟 ATR 动态变化吗

样本：baseline 中 52 笔 `trailing_sl` 交易。

| 指标 | 中位 | p75 | p90 |
|---|---|---|---|
| `ATR(exit 那根 bar) / ATR(at_entry)` | **1.51** | 2.01 | 2.46 |
| 退出后 10 根 bar 原方向 MFE（以入场 ATR 为单位） | **2.24** | 3.83 | 7.08 |

- 59.6% (31/52) 的 trailing_sl 发生在 ATR 已扩张到入场 ATR 的 **≥1.2×** 时。  
  → 用 `atr_at_entry * trail_r` 就是给一个加速趋势留 ~50% 偏紧的跟踪带。
- 53.8% (28/52) 的 trailing_sl 之后 10 根 bar 原方向继续走 ≥2 ATR。  
  → 你说的"抓到趋势后很快 trailing 出局" **定量上确实存在**、且过半属于洗出。

按 symbol 拆分：

| symbol | n | mean_R | median 后续 MFE (ATR) | median ATR(exit)/ATR(entry) |
|---|---|---|---|---|
| BNBUSDT | 13 | +1.48 | **4.01** | 1.57 |
| SOLUSDT | 10 | +1.36 | 2.30 | 1.27 |
| ADAUSDT | 6  | +3.04 | 1.91 | 1.84 |
| BTCUSDT | 12 | +1.90 | 1.75 | 1.46 |
| ETHUSDT | 4  | +0.86 | 1.91 | 1.18 |
| XRPUSDT | 7  | +0.66 | 1.03 | 1.53 |

BNB 洗得最严重（median 4 ATR 后续延续）。

**结论**：`trail_base = max(atr_at_entry, 当前 primary ATR)` 是数据支撑的；默认 `activation_r=6.0 / trail_r=5.0` 是否再放宽 **数据上尚不必**，因 adaptive 已覆盖 ~50% 的偏紧。

---

## 4. 目前代码里糅在一起的 5 处改动

| # | 改动 | 文件 | 现状 |
|---|---|---|---|
| a | 注入 wide SR 特征（96 bar） | `config/strategies/srb/archetypes/execution.yaml` / `srb_regime.py` / `generic_live_strategy.py` | passive 注入（未进决策链） |
| b | trailing 扩带 `expand_with_primary_atr=true` | `execution.yaml` / `position_logic.py` / `event_backtest.py` / `position_tracker.py` | 已开 |
| c | trailing `activation_r 6.0 → 7.0` | `execution.yaml` | 已放宽 |
| d | trailing `trail_r 5.0 → 6.0` | `execution.yaml` | 已放宽 |
| e | 加仓允许 `low_adx_high_er` bucket | `execution.yaml` | 已开 |

> 另外 fake_break_reverse 已是两阶段确认 + bug 修完；单元测试均通过。

---

## 5. 逐项验证计划

基线：`20260417_163432`（192 笔 / +161.24 R / 52 trailing_sl / 23 reverse）。  
每个 exp 相对基线只改一处（复用 `scripts/srb_diag/run_ablation.sh`）：

| Exp | 改动 | 预期方向 | Go 条件 |
|---|---|---|---|
| **Exp 1** wide SR only | +a | ≈ baseline（passive 不改决策） | Δtotal_R ∈ ±2R，n_trades ± 2 |
| **Exp 2** adaptive ATR | +b | trailing_sl 笔数 ↓ 或其 mean_R ↑；post-exit 同向 MFE 中位 ↓ | Δtotal_R ≥ +5R 且 trailing_sl mean_R ↑≥ +0.3R |
| **Exp 3** wider defaults | +c+d | 观察"放宽但不追波动"的独立贡献 | Δtotal_R 方向 vs Exp 2 对比 |
| **Exp 4** + bucket | +e | 加仓笔数 ↑、加仓 mean_R 不崩 | 加仓 R 贡献 ≥ +3R 且 win_rate 降幅 ≤ 3pp |
| **Exp 5** 汇总 | +b+c+d+e | 叠加 | total_R 大于最好单 exp |
| **Exp 6**（待商） | 用 wide SR 作 `fake_break_reverse.true_sr_level` | reverse 命中位更靠谱 | reverse 笔数 & mean_R 都 ≥ baseline |

**执行代价**：一次 full rolling_sim ≈ 4.5h，4 个 exp 串行 ≈ 18h。  
（快速 replay 框架写过一版但和 rolling_sim 对不上 `9 vs 5` 笔，已废弃，见 `scripts/srb_diag/replay_event_backtest.py`；要用需先修预热/state 对齐。）

---

## 6. 目前待用户决定

1. 什么时候跑、跑几个 exp？  
   - A. 先只整理诊断不跑  
   - B. 今晚只跑 Exp 2（adaptive ATR，最硬数据依据，4.5h）  
   - C. 排 4 个 exp（需先等 BPC 流水线结束才能不抢 CPU，~18h）  
   - D. 先修 replay 框架（1-2h 工程量），之后每个 exp 只需几分钟

2. 宽窗 SR 要不要进决策链（Exp 6）？  
   - 仅 passive，本轮不改决策  
   - 或：`fake_break_reverse.true_sr_level` 改用 wide-first

在你回来拍板前，我不主动启动任何回测，也不会再占 CPU。
