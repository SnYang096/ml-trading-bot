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

## Chop Grid：chop / box / chop OR box 对比

目的：验证网格是否应该在 `semantic_chop` 或 `box_prefilter` 任一条件成立时开仓，而不是只在 chop regime 中开仓。

口径：

```text
symbols=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
period=2024-01-01 ~ 2024-12-31
signal_timeframe=2h
execution_timeframe=1min
maker_fee_bps=12
taker_fee_bps=12
forced_exit_slippage_bps=10
funding_cost_bps_per_8h=1
max_loss_per_grid=0.03
```

输出路径：

```text
results/chop_grid/highcap_2024_1min_chop_or_box_compare_fee12_slip10/
```

### 汇总结果

| variant | 含义 | segments | trades | return_pct | win_rate | segment_win_rate | worst_segment | forced_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `semantic_chop` | 光 chop | 177 | 581 | +16.66% | 77.62% | 71.19% | -0.78% | 26.33% |
| `chop_not_box` | chop 且排除 box | 169 | 542 | +15.51% | 77.12% | 69.82% | -0.78% | 26.57% |
| `chop_or_box` | chop 或 box 任一成立 | 259 | 1294 | -69.00% | 74.81% | 56.37% | -3.98% | 22.80% |
| `box_only` | 只要 box 就开仓 | 88 | 699 | -78.04% | 72.25% | 34.09% | -3.98% | 20.89% |

### 单币表现

`semantic_chop` 单币：

| symbol | return_pct | trades | segment_win_rate |
|---|---:|---:|---:|
| SOLUSDT | +4.03% | 77 | 79.17% |
| ADAUSDT | +3.69% | 114 | 80.00% |
| ETHUSDT | +3.39% | 108 | 72.41% |
| XRPUSDT | +2.57% | 86 | 67.86% |
| BTCUSDT | +1.65% | 90 | 61.11% |
| BNBUSDT | +1.33% | 106 | 70.00% |

`box_only` 单币全部为负：

| symbol | return_pct | trades | segment_win_rate |
|---|---:|---:|---:|
| BTCUSDT | -6.73% | 179 | 52.94% |
| ADAUSDT | -10.25% | 115 | 35.71% |
| BNBUSDT | -12.97% | 140 | 35.29% |
| XRPUSDT | -14.07% | 79 | 36.36% |
| ETHUSDT | -16.03% | 80 | 15.38% |
| SOLUSDT | -18.00% | 106 | 25.00% |

### 判断

`chop OR box` 不适合作为当前 `chop_grid` 的开仓条件。它扩大了覆盖和交易数，但新增的 box 交易质量很差：

- `semantic_chop` 从 581 笔、`+16.66%`；
- 扩到 `chop_or_box` 后变成 1294 笔、`-69.00%`；
- `box_only` 全币种为负，说明 box 在当前定义下更像“压缩后等待突破”的结构，而不是天然适合网格的安全震荡。

当前建议：

1. `chop_grid` 主入口保持 `semantic_chop` 或 `chop_not_box`；
2. 不使用 `chop OR box`；
3. `box_prefilter` 更适合作为风险过滤、降杠杆、放宽 grid spacing 或减少层数的信号；
4. 若要单独做 box-grid，需要另一套参数和突破退出逻辑，不能复用当前 chop_grid 默认参数。

## 策略族差异：趋势肥尾 vs 多腿短周期 vs 清算逻辑

本节用于统一理解：`chop_grid` / `dual_add_trend` 与此前 BPC/ME/TPC/SRB 等趋势/肥尾策略为什么表现形态完全不同，以及未来“清算合约逻辑”更接近哪一类。

### 一句话结论

- **趋势 / 肥尾策略**：主要吃“方向 + 持有 + 右尾”，是 `beta + alpha + execution` 的组合。
- **`dual_add_trend`**：吃“已确认 regime 内的短期趋势脉冲 + basket 库存管理”，偏纯 alpha，但仍有方向暴露。
- **`chop_grid`**：吃“非趋势环境中的均值回归 / 波动收割”，几乎不吃 beta，主要是库存 alpha。
- **清算合约逻辑**：更像 **LV / FER / 短周期事件驱动策略**，不是 `chop_grid`；若做顺清算方向，接近 `dual_add_trend` / ME；若做清算后反转，接近 FER / mean reversion。

### 关键差异表

| 维度 | 趋势 / 肥尾策略 | `dual_add_trend` | `chop_grid` | 清算合约逻辑 |
|---|---|---|---|---|
| 核心收益来源 | 大趋势右尾、低频大单贡献 | trend regime 内短期脉冲，顺势加仓后 basket TP | chop regime 内价格来回穿网格 | 强平引发的被迫成交、流动性真空、overshoot / cascade |
| Alpha 语义 | 方向判断 + 让利润奔跑 | regime 过滤 + 顺动量库存管理 | regime 过滤 + 均值回归库存管理 | 杠杆脆弱性、清算燃料、订单簿/流动性错位 |
| Beta 暴露 | 明显，尤其多头趋势策略 | 中等，trend-only 有短期方向暴露 | 低，理想状态接近 market neutral | 取决于方向：顺清算方向有短 beta/alpha；反转型更接近 alpha |
| 持仓时间 | 长，数小时到数天甚至更长 | 短到中，通常在趋势 segment 内多次进出 | 短，网格 TP / regime exit | 很短，分钟到数小时，alpha 半衰期短 |
| 胜率形态 | 低胜率或中胜率，高盈亏比 | 中高胜率，中等盈亏比 | 高胜率，低单笔收益 | 不固定：顺 cascade 胜率可能低但尾部大；反转型胜率可高但止损敏感 |
| 收益分布 | 右偏肥尾，少数单贡献大 | 更平滑，但仍依赖趋势段质量 | 多小赢 + 少数突破亏损 | 极端事件驱动，厚尾但非平稳 |
| 主要风险 | 过早止盈、错过肥尾、趋势反转 | 执行成本、假趋势、反向后库存处理 | 单边突破、regime exit 滞后 | 清算假信号、流动性消失、滑点、交易所延迟 |
| 对执行精度敏感度 | 中等，入场可粗，出场/追踪重要 | 高，TP/step 与成本接近时尤其高 | 中高，网格层距越窄越敏感 | 极高，清算事件本身发生很快 |
| 适合指标 | MFE/MAE、尾部收益、持仓分布、趋势后续 | 1min/100ms 稳定性、fee stress、risk_stop/forced_rate | forced_rate、突破亏损、库存层数、chop coverage | liquidation volume、OI/FR、orderbook imbalance、post-liquidation path |
| 是否能只看 2h 回测 | 不能，但粗筛可用 | 不能，必须 1min/100ms 复核 | 不能，至少 1min | 更不能，必须 tick/秒级或事件流 |

### Alpha 到底来自哪里

#### 趋势 / 肥尾策略

趋势策略的收益不是单纯来自“预测下一根 K 线涨跌”，而是来自：

1. **regime / prefilter 找到右尾可能性更高的环境**；
2. **方向规则或模型给出正期望入场**；
3. **holding / trailing 允许少数大单贡献大部分收益**；
4. **组合层面用小亏损换大行情**。

所以它常常是：

```text
市场 beta（长期方向/资产趋势）
+ 策略 alpha（更好的入场和过滤）
+ execution alpha（不太早卖掉肥尾）
```

这类策略天然低胜率或中胜率，不应该用高胜率标准评价。它要看右尾、MFE 捕捉、回撤后的继续持有能力。

#### `dual_add_trend`

`dual_add_trend` 的 alpha 来自：

1. `trend_confidence` 确认非 chop 的方向 regime；
2. 只在趋势方向开第一腿，避免初始反向腿拖累；
3. 价格继续沿趋势走时加仓；
4. basket 达到净盈利目标后统一退出；
5. regime 失效时退出，而不是用很近的 per-leg stop 打掉。

它不像长趋势策略那样等待几天的大肥尾，而是吃趋势段里较短的脉冲。它的收益更像：

```text
短周期 regime alpha
+ 库存/加仓 execution alpha
- 成本/滑点/假趋势损耗
```

因此它比长趋势更高胜率、更短持仓，但也更怕成本和执行误差。

#### `chop_grid`

`chop_grid` 的 alpha 来自：

1. 正确识别“价格没有明确方向、容易来回穿越”的 chop regime；
2. 网格层距足够宽，能覆盖手续费、点差、滑点；
3. regime exit 足够及时，避免突破时库存单边累积；
4. forced exit 不频繁，否则小 TP 会被少数突破亏损吃掉。

它不是方向 alpha，而是：

```text
regime classification alpha
+ mean reversion / volatility harvesting
+ inventory management
- breakout tail risk
```

所以它高胜率、短持仓、小收益，但最怕“看起来像震荡、实际马上突破”的结构。前面的测试显示 `box_only` 和 `chop OR box` 很差，说明当前 `box_prefilter` 捕捉到的更像“压缩后等待突破”，不是安全网格环境。

### 清算合约逻辑更像哪一种

“清算合约逻辑”不应该归到 `chop_grid`。清算不是普通震荡，它是被迫成交、保证金约束和流动性缺口共同导致的事件。它更像一个独立 archetype：**Leverage Vulnerability / Liquidation Event**。

它可以分成两种玩法。

#### 1. 顺清算方向：cascade continuation

逻辑：

```text
高杠杆脆弱性
+ 价格击穿关键区
+ 清算触发
+ 流动性真空
=> 顺清算方向追随短期 cascade
```

它更像：

- `dual_add_trend`：顺动量加仓，快进快出；
- ME：吃动量加速；
- 但比它们更短周期、更依赖订单流/清算数据。

关键不是 chop，而是“谁被迫平仓、平仓是否会推动下一段价格”。

#### 2. 清算后反转：liquidation exhaustion reversal

逻辑：

```text
单边清算已经发生
+ OI 快速下降
+ 成交放量但价格不再延续
+ 盘口吸收 / wick 失败
=> 反向做 overshoot 回归
```

它更像：

- FER：失败衰竭反转；
- FBF/SRB 的某些反转语义；
- 短周期 mean reversion，但不是普通网格。

这类策略可以高胜率，但止损和执行要求很高，因为你是在接极端事件后的刀口。

### 和现有三类策略的关系

| 新策略方向 | 更接近谁 | 不像谁 | 原因 |
|---|---|---|---|
| 顺清算 cascade | `dual_add_trend` / ME | `chop_grid` | 它吃被迫成交导致的方向延续，不是均值回归 |
| 清算后反转 | FER / FBF / 事件反转 | 长趋势策略 | 它吃过度清算后的回补，不是长期 beta |
| 清算区间网格 | 需要新设计 | 当前 `chop_grid` | 清算前后突破风险极高，普通 grid 容易吃单边库存 |

### 如果要新增清算合约策略，建议定位

建议不要把它并入 `chop_grid` 或 `dual_add_trend`，而是新建一个独立策略族，例如：

```text
lv_liquidation_cascade
lv_liquidation_reversal
```

初始版本可以做两个 profile：

1. **Cascade profile**
   - 方向：顺清算方向；
   - 持仓：极短；
   - 退出：固定时间 / 动量衰减 / 清算量衰减；
   - 风险：滑点和追高，必须有硬止损。

2. **Exhaustion reversal profile**
   - 方向：逆最后一段清算 impulse；
   - 触发：清算放量后价格不能继续推进；
   - 退出：快速均值回归 TP；
   - 风险：cascade 二段延续，必须小仓位和快速止损。

### 对组合架构的意义

如果把当前策略放进组合层，可以这样理解：

```text
长期 / 中期趋势策略：负责 beta + 肥尾 alpha
dual_add_trend：负责趋势 regime 内的短周期 alpha
chop_grid：负责非趋势 regime 内的波动收割 alpha
liquidation strategy：负责杠杆脆弱事件 alpha
```

这四类不是互相替代，而是覆盖不同市场状态：

- 趋势策略：行情走出来后拿住；
- `dual_add_trend`：趋势段内短线收割；
- `chop_grid`：无趋势时收割来回波动；
- 清算策略：极端杠杆事件时吃强制流动。

关键上线原则：

1. 不要用同一套胜率标准评价所有策略；
2. 趋势策略看右尾和持仓质量；
3. 多腿短周期看执行精度和成本压力；
4. 清算策略看事件后路径、滑点、延迟和成交质量；
5. 清算策略必须用更细数据，不应只用 2h/1min OHLC 验证。
