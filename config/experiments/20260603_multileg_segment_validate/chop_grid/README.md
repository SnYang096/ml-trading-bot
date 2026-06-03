# chop_grid — segment validate (timeline)

**Manifest：** [`segment_validate.yaml`](segment_validate.yaml)  
**父实验：** [`../README.md`](../README.md)  
**Replenish 对照：** [`../../20260603_chop_grid_replenish_ablation/`](../../20260603_chop_grid_replenish_ablation/)（TP 后补挂 on/off）

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

默认 **replenish=unlimited**（`archetypes/execution.yaml` 的 `max_replenish_per_level_per_segment: null`）。关闭补挂加：`--max-replenish-per-level 0`。

## 结论

跑批后填写 [`DECISION.md`](DECISION.md)。20260602 结果：OOS eq-weight -0.75%，不建议 promote — 见旧 DECISION。
