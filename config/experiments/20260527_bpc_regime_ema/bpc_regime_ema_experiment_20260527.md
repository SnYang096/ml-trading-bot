# BPC Regime：box_legacy vs ema_only vs ema_slope

- **日期**: 2026-05-27
- **Grid**: `config/experiments/bpc_regime_ema_grid.yaml`
- **决策**: **生产保持 ema_only**；**不**默认加 slope（slope 仅 bull 段略优，recent 样本小，待季度复验）

## Event backtest

| 窗 | box_legacy | ema_only（生产） | ema_slope |
|---|---:|---:|---:|
| 2024 bull | 27 / +16.85 | 29 / +15.52 | 23 / **+17.24** |
| 2025–26 recent | 25 / -1.22 | 17 / +5.89 | 15 / **+6.89** |

**读法**：

- 去掉 box、改 EMA 后 recent 由 **-1.22 → +5.89**（相对 box_legacy 明显改善）。
- 加 slope：bull +1.7R vs ema_only；recent +1.0R、少 2 笔 — **边际改善，未做双段 Pareto 硬门槛**；暂不写入生产 regime（`locked: false` 实验树保留）。

## 复现

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/bpc_regime_ema_grid.yaml --quiet-signal-logs
```
