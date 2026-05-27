# TPC Regime 有符号斜率 — label/IC 补扫

- **日期**: 2026-05-27
- **Scan**: `results/rd_loop/tpc_slope_signed/quick_scan/regime_slope_signed.md`
- **结论**: 支持补跑 **分符号** event_backtest；**不**因 recent IC 负而否定 slope

## condition-set（chop<=0.4, n=23194）

| 条件 | Δpp vs base | \|z\| | 备注 |
|---|---:|---:|---|
| H `\|pos\|>0.10` | +0.06 | 0.14 | 生产 regime |
| Fp_sym `\|pos\|>0.10` ∧ `\|slope\|>0.002` | **+1.01** | 2.03 | 已 backtest，R 弱于 H |
| **bear_pos** `<-0.10` | **+1.93** | 2.91 | 强于 bull_pos |
| **bull_pos** `>0.10` | **-2.03** | 2.86 | 与 BPC 同向 |
| slope_down `<-0.002` | +1.60 | 2.05 | |
| slope_up `>0.002` | +0.12 | 0.15 | 对称 slope 主要增益在 **负斜率** 侧？ |
| bear_trend pos∧slope 同向 | +1.58 | 2.03 | |
| bull_trend | +0.46 | 0.59 | |
| bull_pos∧bear_slope / bear_pos∧bull_slope | n=0 | — | 过严，需放宽或分桶 |

**含义**：recent 上 ema **负 IC** 与 **bull_pos label 变差** 一致；**bear 侧 / 负斜率** label 更好 → 值得测 **非对称 regime**（不是简单加 `|slope|`）。

## 下一步（未跑 backtest）

建 `tpc_regime_bear_trend_strategies` / `tpc_regime_slope_down_strategies` 等，填入 `tpc_regime_slope_signed_grid.yaml` 的 `runs`。
