# Dual Add / Chop Grid 执行压力测试与止损机制复盘（2026-05-08）

## 结论摘要

本轮验证后，`dual_add_trend` 不应再用默认小 TP 参数直接实盘。默认参数在 `1min` 下仍然盈利，但在 BTC `100ms aggTrades` + 高成本压力下会失效，说明它对执行质量、点差、滑点和网络延迟敏感。

更可接受的候选方向是：

- `initial_hedge=false`：趋势单开，不再初始多空双开；
- `take_profit.mode=basket`：整个 basket 达到净盈利目标后一起退出；
- 放大 TP / 加仓间距，例如 `step_atr_mult=1.0`、`tp_atr_mult=0.75`、`tp_pct=0.0015`；
- 段级最大亏损可从 `1%` 放宽到 `2%` 做候选，但必须保留账户级灾难止损。

`chop_grid` 相对不依赖毫秒级 TP，但收益边际小，主要风险变成 regime exit 滞后和单边突破库存累积。

## 当前 dual_add 止损如何工作

当前有两层退出/保护逻辑。

### 1. Research simulator 的段级 MTM 止损

`scripts/diagnose_dual_add_trend.py` 在每个 segment 内计算：

- 已实现 PnL；
- 所有开放腿按当前 close 标记的浮动 PnL；
- 除以 `capital_units=max(2, max_gross_exposure)` 得到 `mtm_per_capital`。

如果：

```text
mtm_per_capital <= -max_loss_per_segment
```

则本段标记为 `risk_stop`，循环停止，剩余库存按当前 close 强制退出。

这不是“只止损加仓腿”，而是 segment 级组合止损。但它确实和执行频率强相关：`1min/100ms` 会更早、更频繁地看到段内波动，因此比 `2h` 更容易触发或更接近真实触发。

### 2. Live engine 的 exchange-side 灾难止损

`src/time_series_model/live/dual_add_trend_live_engine.py` 对每条成交腿都会生成 protection order：

- basket 模式下不挂 per-leg TP；
- 但仍挂 per-leg `stop_loss`；
- stop 距离大致为：

```text
max(tp_distance * 4, atr * max(step_atr_mult, 1.0))
```

这层更像交易所侧灾难保险，目的是防止进程宕机、断网、极端跳水时完全裸奔。它不是策略语义上的主退出。

### 3. Regime exit

live engine 里，如果：

```text
trend_confidence < exit_trend_below
or semantic_chop > max_hold_chop
```

会 `_exit_all(..., reason="regime_exit")`，即趋势语义失效或进入 chop 后退出全部库存。

这更接近最初设计中的“趋势段结束就退出”。

## 是否应改成 chop/regime 语义止损

方向上是合理的：`dual_add_trend` 的主退出应更语义化，不应让一个很近的 per-leg stop 决定策略表现。建议把退出分层：

1. **主止盈**：basket TP，整个库存一起止盈。
2. **主止损/退出**：regime exit，即趋势置信度跌破阈值或 semantic chop 进入不适合趋势的区域。
3. **风险阀**：segment-level MTM stop，但不宜过紧。可把 `max_loss_per_segment` 作为 profile 参数，例如 `0.01 / 0.02 / 0.03`。
4. **灾难保护**：live engine per-leg exchange stop 保留，但应足够远，只处理断网、宕机、极端行情，不参与日常策略逻辑。
5. **账户级 kill switch**：组合日亏损、单币亏损、连续 risk_stop、交易所异常时停止新开仓。

不建议完全移除止损。原始“多空双开、动量侧加仓、达到 basket TP 退出”的想法在强趋势里合理，但遇到快速反转或 regime 判定滞后时，库存会变成方向性暴露。只靠语义退出会更符合策略，但需要灾难止损兜底。

## 本轮测试口径

窗口：`2024-01-01` 到 `2024-03-31`。

主回测口径：

- `dual_add_trend`：`BTCUSDT,ETHUSDT,SOLUSDT`，`1min` execution，`2h` signal，趋势单开，basket TP；
- 成本压力：`fee_bps=8/12/20`，用它近似覆盖手续费、点差、滑点、网络通道差导致的额外成本；
- `100ms` 复核：仅 BTC，因为 `data/agg_data` 中缺少 ETH/SOL 的 2024Q1 aggTrades；
- `chop_grid`：`BTCUSDT,ETHUSDT,SOLUSDT`，`1min` execution，带 forced-exit slippage 和 `max_loss_per_grid`。

## dual_add 1min 参数扫描

### 默认小 TP

参数：

```text
step_atr_mult=0.50
tp_atr_mult=0.25
tp_pct=0.0005
max_loss_per_segment=0.01
```

结果：

| fee_bps | return_pct | trades | trade_win_rate | risk_stop_rate |
|---:|---:|---:|---:|---:|
| 8 | +39.29% | 736 | 81.25% | 7.22% |
| 12 | +24.29% | 678 | 77.58% | 12.37% |
| 20 | +17.53% | 600 | 74.50% | 13.40% |

### 放大 TP / step

参数：

```text
step_atr_mult=1.00
tp_atr_mult=0.75
tp_pct=0.0015
max_loss_per_segment=0.01
```

结果：

| fee_bps | return_pct | trades | trade_win_rate | risk_stop_rate |
|---:|---:|---:|---:|---:|
| 8 | +43.08% | 460 | 74.35% | 9.28% |
| 12 | +35.40% | 473 | 72.09% | 11.34% |
| 20 | +21.78% | 435 | 68.97% | 13.40% |

这组比默认小 TP 更少交易、更不依赖极小价差，在高成本下更有意义。

### 放大 TP / step + 放宽段级止损

参数：

```text
step_atr_mult=1.00
tp_atr_mult=0.75
tp_pct=0.0015
max_loss_per_segment=0.02
```

结果：

| fee_bps | return_pct | trades | trade_win_rate | risk_stop_rate |
|---:|---:|---:|---:|---:|
| 8 | +50.02% | 485 | 74.43% | 1.03% |
| 12 | +33.36% | 487 | 72.07% | 4.12% |
| 20 | +26.76% | 469 | 69.94% | 2.06% |

放宽止损在 1min 上改善了 Q1 表现，但它也把尾部风险让给 regime exit 和灾难止损，因此不能只看收益，后续需要更长周期和 100ms 复核。

## BTC 100ms aggTrades 复核

### 默认小 TP

`BTCUSDT`，`2024Q1`，趋势单开，`aggTrades -> 100ms OHLC`。

| fee_bps | return_pct | trades | trade_win_rate | risk_stop_rate |
|---:|---:|---:|---:|---:|
| 8 | +6.33% | 295 | 74.92% | 6.06% |
| 12 | +2.87% | 267 | 75.66% | 9.09% |
| 20 | -2.90% | 205 | 72.68% | 18.18% |

默认参数在 `fee_bps=20` 下转负，不能作为实盘候选。

### 放大 TP / step

参数：

```text
step_atr_mult=1.00
tp_atr_mult=0.75
tp_pct=0.0015
max_loss_per_segment=0.01
```

结果：

| fee_bps | return_pct | trades | trade_win_rate | risk_stop_rate |
|---:|---:|---:|---:|---:|
| 8 | +7.43% | 196 | 65.31% | 9.09% |
| 12 | +4.28% | 183 | 67.21% | 12.12% |
| 20 | -0.44% | 157 | 70.70% | 15.15% |

放大参数明显改善了 100ms 下的成本承受能力，但 `fee_bps=20` 仍接近失效。实盘前必须确认真实综合成本更接近 `8~12bps/side`，而不是 `20bps/side`。

## chop_grid 压力测试

口径：

```text
symbols=BTCUSDT,ETHUSDT,SOLUSDT
execution_timeframe=1min
fee_bps=12
maker_fee_bps=8
taker_fee_bps=12
forced_exit_slippage_bps=10
max_loss_per_grid=0.03
```

结果：

| return_pct | trades | trade_win_rate | forced_rate | max_drawdown |
|---:|---:|---:|---:|---:|
| +1.86% | 51 | 70.59% | 33.33% | -0.32% |

解释：

- `chop_grid` 的确不是“完全无止损”：本次回测使用了 `max_loss_per_grid=0.03` 和 forced exit slippage；
- 它对毫秒级 TP 不如 `dual_add` 敏感；
- 但 Q1 收益边际很薄，forced exit 占比高；
- 单边突破时仍可能因为 regime exit 滞后而吃亏。

## 当前可接受的上线前门槛

建议暂定如下 gate：

1. 只把 `1min` 作为主结果，`2h` 只能粗筛；
2. 候选参数必须在 `fee_bps=8/12/20` 三档成本压力下测试；
3. `fee_bps=12` 应显著为正，`fee_bps=20` 不应明显亏损；
4. 对候选参数做 `100ms aggTrades` 抽样复核；
5. `risk_stop_rate` 不应过高，否则说明止损在替代策略逻辑；
6. `forced_rate` 不应过高，否则说明大量收益来自正常段，尾部退出质量不足；
7. live engine replay 必须补齐，用同一套实盘逻辑回放历史 bars。

## 下一步建议

1. 将 `dual_add_trend` 的默认 profile 从小 TP 改为更大 TP/step 的候选，而不是直接用本轮结果上线。
2. 把 `max_loss_per_segment` 从固定默认值变成 profile 参数，至少比较 `0.01 / 0.02 / 0.03`。
3. 调整 live engine 的止损语义：主退出使用 regime exit，per-leg stop 只作为远端灾难保护。
4. 补 ETH/SOL 的 2024Q1 aggTrades 后，重跑三币 100ms 复核。
5. 增加 live engine replay runner，避免 research simulator 和实盘状态机分叉。

## 2026-05-08 改进后复测

已将 `dual_add_trend` 默认执行 profile 调整为：

```text
initial_legs=[TREND]
step_atr_mult=1.00
tp_atr_mult=0.75
tp_pct=0.0015
risk_stop_mode=regime_only
max_loss_per_segment=0.02
protection_stop_mode=catastrophic
catastrophic_stop_atr_mult=8.0
catastrophic_stop_tp_mult=8.0
```

含义：

- 默认不再初始多空双开，而是只开趋势方向；
- 主退出从紧 MTM stop 改为 trend/chop regime 语义退出；
- `max_loss_per_segment` 作为风险 profile / 部署 gate 保留；
- live engine 的 per-leg stop 放远，仅作为交易所侧灾难保护。

### 1min before / after

口径：`BTCUSDT,ETHUSDT,SOLUSDT`，`2024Q1`，`1min execution`，趋势单开。

| 版本 | fee_bps | return_pct | trades | segment_win_rate | risk_stop_rate | worst_segment |
|---|---:|---:|---:|---:|---:|---:|
| 放大 TP/step + stop2 | 8 | +50.02% | 485 | 85.57% | 1.03% | -2.01% |
| 改进版 regime_only | 8 | +53.78% | 489 | 86.60% | 0.00% | -1.33% |
| 放大 TP/step + stop2 | 12 | +33.36% | 487 | 77.32% | 4.12% | -2.15% |
| 改进版 regime_only | 12 | +41.19% | 505 | 78.35% | 0.00% | -1.41% |
| 放大 TP/step + stop2 | 20 | +26.76% | 469 | 68.04% | 2.06% | -2.17% |
| 改进版 regime_only | 20 | +30.77% | 478 | 69.07% | 0.00% | -1.57% |

结论：`1min` 主回测下，改成 regime 语义退出后，三档成本均提升，且 `risk_stop_rate` 降到 0，说明收益不再依赖紧 MTM 止损。

### BTC 100ms before / after

口径：`BTCUSDT`，`2024Q1`，`aggTrades -> 100ms OHLC`。

| 版本 | fee_bps | return_pct | trades | segment_win_rate | risk_stop_rate | worst_segment |
|---|---:|---:|---:|---:|---:|---:|
| 放大 TP/step + MTM stop | 8 | +7.43% | 196 | 72.73% | 9.09% | -1.00% |
| 改进版 regime_only | 8 | +11.12% | 207 | 78.79% | 0.00% | -1.14% |
| 放大 TP/step + MTM stop | 12 | +4.28% | 183 | 60.61% | 12.12% | -1.04% |
| 改进版 regime_only | 12 | +5.01% | 198 | 60.61% | 0.00% | -1.37% |
| 放大 TP/step + MTM stop | 20 | -0.44% | 157 | 54.55% | 15.15% | -1.04% |
| 改进版 regime_only | 20 | +5.14% | 174 | 60.61% | 0.00% | -1.35% |

结论：这是最关键的改善。改进前 BTC 100ms 在 `fee_bps=20` 已经转负；改进后 `fee_bps=20` 仍为正，说明语义退出降低了执行层止损对策略的破坏。

注意：这仍只是 BTC 100ms Q1。ETH/SOL 的 2024Q1 aggTrades 当前缺失，三币 100ms 同口径复核还不能完成。

## 2024 全年执行精度对比

目的：同一套改进版 `dual_add_trend` 参数，在 `2h / 1min / 100ms aggTrades` 三种执行精度下跑 BTC 2024 全年，优先选择不同精度结果差距不大的玩法。

口径：

```text
symbol=BTCUSDT
period=2024-01-01 ~ 2024-12-31
signal_timeframe=2h
profile=improved regime_only trend-only
fee_bps=8/12/20
```

100ms 因为数据量较大，按季度跑后汇总。输出路径：

```text
results/dual_add_trend/compare_precision_btc_2024_agg100ms_improved_fee_stress_20260508/precision_comparison.csv
```

### 年度汇总

| execution | fee_bps | segments | trades | return_pct | worst_segment | risk_stop_rate | forced_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2h | 8 | 129 | 452 | +40.60% | -1.75% | 0.00% | 45.58% |
| 1min | 8 | 129 | 687 | +42.94% | -1.07% | 0.00% | 25.04% |
| 100ms | 8 | 126 | 779 | +40.35% | -1.14% | 0.00% | 21.58% |
| 2h | 12 | 129 | 453 | +35.64% | -1.79% | 0.00% | 47.90% |
| 1min | 12 | 129 | 684 | +33.53% | -1.17% | 0.00% | 26.75% |
| 100ms | 12 | 126 | 743 | +28.55% | -1.37% | 0.00% | 24.05% |
| 2h | 20 | 129 | 441 | +22.06% | -2.12% | 0.00% | 51.70% |
| 1min | 20 | 129 | 620 | +22.73% | -1.32% | 0.00% | 31.45% |
| 100ms | 20 | 126 | 654 | +18.38% | -1.35% | 0.00% | 28.85% |

### 100ms 分季度结果

| quarter | fee_bps | return_pct | segments | trades | worst_segment | forced_rate |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 8 | +11.12% | 33 | 207 | -1.14% | 22.22% |
| Q2 | 8 | +6.11% | 23 | 145 | -0.83% | 20.00% |
| Q3 | 8 | +14.84% | 39 | 239 | -0.51% | 21.76% |
| Q4 | 8 | +8.28% | 31 | 188 | -0.38% | 22.34% |
| Q1 | 12 | +5.01% | 33 | 198 | -1.37% | 25.76% |
| Q2 | 12 | +5.55% | 23 | 136 | -0.77% | 20.59% |
| Q3 | 12 | +12.64% | 39 | 230 | -0.58% | 23.04% |
| Q4 | 12 | +5.35% | 31 | 179 | -0.71% | 26.82% |
| Q1 | 20 | +5.14% | 33 | 174 | -1.35% | 27.59% |
| Q2 | 20 | +2.51% | 23 | 116 | -0.89% | 28.45% |
| Q3 | 20 | +8.17% | 39 | 209 | -0.99% | 27.75% |
| Q4 | 20 | +2.57% | 31 | 155 | -0.85% | 31.61% |

### 执行精度差距判断

按年度看，改进版参数的执行精度差距可接受：

- `fee8`：`2h +40.60%`、`1min +42.94%`、`100ms +40.35%`，三者几乎一致；
- `fee12`：`2h +35.64%`、`1min +33.53%`、`100ms +28.55%`，100ms 有降温，但仍同量级；
- `fee20`：`2h +22.06%`、`1min +22.73%`、`100ms +18.38%`，高成本下也没有失效。

这比默认小 TP 参数健康很多。之前默认参数在 BTC 100ms Q1 的 `fee20` 已经转负；改进版在 BTC 100ms 全年 `fee20` 仍为 `+18.38%`。

当前优先玩法：

1. 使用改进版 `trend-only + basket TP + regime_only exit`；
2. 只把 `1min` 作为主回测口径；
3. 用 `100ms` 做候选参数复核；
4. 若 `2h / 1min / 100ms` 三者收益同量级，才允许进入 dry-run；
5. 若 100ms 明显低于 1min 或转负，淘汰该 profile。
