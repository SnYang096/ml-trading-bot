# chop_grid — segment validate (timeline)

**Manifest：** [`segment_validate.yaml`](segment_validate.yaml)  
**父实验：** [`../README.md`](../README.md)

## 跑法

```bash
python scripts/experiment_chop_grid_market_segment.py \
  --out-root results/chop_grid/experiments/segment_validate_20260603_timeline \
  -- \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --initial-capital 10000 \
  --no-maps
```

## 结论

跑批后填写 [`DECISION.md`](DECISION.md)。20260602 结果：OOS eq-weight -0.75%，不建议 promote — 见旧 DECISION。
