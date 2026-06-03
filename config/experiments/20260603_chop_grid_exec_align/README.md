# chop_grid — live-aligned execution replay (2026-06-03)

## 变更

- **Canonical exec:** `1min`（`grid_backtest.execution_timeframe` in `calibrate_roll.default.yaml`）
- **Segment window:** `[signal[s]+Δ, signal[e]+Δ)` — regime 在 2h 收盘确认后才模拟 fill（`subbar_replay.segment_execution_bounds`）
- **2h exec:** 仍可用作 legacy sensitivity，但走同一 subbar 窗口（不再在 entry signal bar 内 optimistic fill）
- **100ms:** `--execution-timeframe 100ms --agg-data-dir data/agg_data`（需 aggTrades zip）

## 跑四段 validate（1min canonical，prod 20bps）

```bash
python scripts/experiment_chop_grid_market_segment.py \
  --out-root results/chop_grid/experiments/exec_align_20260603/1min_prod \
  -- \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h \
  --no-maps
```

`execution-timeframe` 省略时默认 **1min**（来自 calibrate_roll）。

## OOS spacing sensitivity（optional）

```bash
python scripts/chop_grid_backtest.py \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --start 2025-10-01 --end 2026-03-31 \
  --grid-atr-mult 1.25 --grid-pct 0.012 \
  --no-maps \
  --out-dir results/chop_grid/experiments/exec_align_20260603/spacing_wide_oos
```
