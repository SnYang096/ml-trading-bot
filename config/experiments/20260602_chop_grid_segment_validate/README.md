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

- **`return_pct`**：timeline 组合 equity（按 exit 时间序，每笔 `pnl_per_capital / n_symbols` 累加 × 100）
- **`return_pct_eq_mean`**：per-symbol 终值等权均值（忽略路径）
- **`return_pct_pooled`**：五币 trade 直接相加（仅对照）
- **`max_drawdown_portfolio`**：timeline equity 路径最大回撤

## 结论

**历史（eq-weight 时代，2026-06-02）：** 见 [`DECISION.md`](DECISION.md)。  
**Timeline 重跑：** [`../20260603_multileg_segment_validate/chop_grid/`](../20260603_multileg_segment_validate/chop_grid/)
