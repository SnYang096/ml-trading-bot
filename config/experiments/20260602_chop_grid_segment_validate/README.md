# chop_grid — market_segment 四段稳定性验证

**日期：** 2026-06-02  
**策略：** `config/strategies/chop_grid` + `calibrate_roll.default.yaml`  
**窗口：** `config/market_segment.yaml` 四段

## 跑法

```bash
OUT=results/chop_grid/experiments/segment_validate_20260602

python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$OUT" \
  -- \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --no-maps
```

## 指标口径

- **`return_pct`**：等权 multileg 组合（per-symbol `pnl_per_capital` 均值 × 100）
- **`return_pct_pooled`**：旧口径（五币 trade 直接相加，仅作对照）

## 结论

见 [`DECISION.md`](DECISION.md)。**OOS eq-weight -0.75%** — pooled 口径会误读为 -3.7%，修正后接近 flat。
