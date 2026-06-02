# trend_scalp — market_segment 四段稳定性验证

**日期：** 2026-06-02  
**策略：** `config/strategies/trend_scalp`（prod archetype + `calibrate_roll.default.yaml` 回测 profile）  
**窗口：** `config/market_segment.yaml` 四段 canonical segments

## 假设

prod trend_scalp 底盘（TREND 单开 + basket TP + regime_only + flat_until_next_regime）在 bear / bull / recent range / recent OOS 上 **return_pct 均为正或至少不崩**，worst_segment 与 portfolio_cum_dd 可控。

## 物料

| 文件 | 说明 |
|------|------|
| [`segment_validate.yaml`](segment_validate.yaml) | 实验 manifest（symbols、segments、output_dir） |
| [`scripts/experiment_trend_scalp_market_segment.py`](../../../scripts/experiment_trend_scalp_market_segment.py) | 四段批量 runner |

## 跑法

```bash
EXP=config/experiments/20260602_trend_scalp_segment_validate
OUT=results/trend_scalp/experiments/segment_validate_20260602

python scripts/experiment_trend_scalp_market_segment.py \
  --out-root "$OUT" \
  --market-segment-path config/market_segment.yaml \
  -- \
  --config config/strategies/trend_scalp/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --scale-max-loser-hold-to-signal \
  --take-profit-mode basket --no-reseed-on-flip \
  --risk-stop-mode regime_only --no-maps
```

## 产物

- `results/trend_scalp/experiments/segment_validate_20260602/<segment_id>/` — 每段 `summary.csv`、`dual_add_trades.csv`
- `segment_summary.csv` — 四段汇总
- `segment_by_symbol.csv` — 分币 breakdown

## Backtrader 交叉验证

`pip install backtrader` 后：

```bash
python scripts/backtest_trend_scalp_backtrader.py \
  --start 2025-10-01 --end 2026-03-31 \
  --execution-timeframe 1min --scale-max-loser-hold-to-signal \
  --take-profit-mode basket --no-reseed-on-flip --risk-stop-mode regime_only \
  --compare-dir results/trend_scalp/experiments/segment_validate_20260602/recent_6m_oos \
  --out-dir results/trend_scalp/experiments/backtrader_crosscheck_recent_6m
```

与 diagnose 偏差 < 2%，见 [`DECISION.md`](DECISION.md) §7。

## 结论

**稳定** — 四段 eq-weight return 均正（20%–59%），五币 × 四段 20/20 格全正；worst_segment ~2%、portfolio_cum_dd ~5% 跨段一致。详见 [`DECISION.md`](DECISION.md)。
