# trend_scalp — segment validate (timeline)

**Manifest：** [`segment_validate.yaml`](segment_validate.yaml)  
**父实验：** [`../README.md`](../README.md)

## 跑法

```bash
python scripts/experiment_trend_scalp_market_segment.py \
  --out-root results/trend_scalp/experiments/segment_validate_20260603_timeline \
  --market-segment-path config/market_segment.yaml \
  -- \
  --config config/strategies/trend_scalp/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --initial-capital 10000 \
  --scale-max-loser-hold-to-signal \
  --take-profit-mode basket --no-reseed-on-flip \
  --risk-stop-mode regime_only --no-maps
```

## 结论

跑批后填写 [`DECISION.md`](DECISION.md)。20260602 eq-weight 结果见 [`../../20260602_trend_scalp_segment_validate/DECISION.md`](../../20260602_trend_scalp_segment_validate/DECISION.md)。
