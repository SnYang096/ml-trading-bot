# multileg segment validate — timeline portfolio metrics (2026-06-03)

**目的：** 在 `config/market_segment.yaml` 四段窗口上，用 **timeline 组合 equity** 作为 canonical `return_pct`，统一 trend_scalp 与 chop_grid 的 segment 稳定性验证。

**前置：** [`20260602_*` 实验](../20260602_trend_scalp_segment_validate/) 使用 eq-weight / pooled 口径；本目录为 **timeline 时代** 的正式 manifest，跑批产物写入新 `output_dir` 后缀 `_timeline`。

## 指标口径

| 字段 | 含义 |
|------|------|
| `return_pct` | timeline 组合终值（`exit_time` 序，每笔 `pnl_per_capital / n_symbols`） |
| `return_pct_eq_mean` | per-symbol 终值等权（忽略路径，对照） |
| `return_pct_pooled` | 五币 trade 直接相加（~5× 放大，对照） |
| `max_drawdown_portfolio` | timeline equity 路径最大回撤 |
| `daily_sharpe` | 日历日度 **组合** 收益 Sharpe（×√365） |
| `portfolio_cum_dd` | segment 级 pooled 累加 DD（legacy，与 timeline DD 并存） |

详见 [`METRICS.md`](METRICS.md)。

## 子实验

| 策略 | Manifest | 产物根目录 |
|------|----------|------------|
| trend_scalp | [`trend_scalp/segment_validate.yaml`](trend_scalp/segment_validate.yaml) | `results/trend_scalp/experiments/segment_validate_20260603_timeline` |
| chop_grid | [`chop_grid/segment_validate.yaml`](chop_grid/segment_validate.yaml) | `results/chop_grid/experiments/segment_validate_20260603_timeline` |

## 一键跑法

```bash
# trend_scalp
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

# chop_grid
python scripts/experiment_chop_grid_market_segment.py \
  --out-root results/chop_grid/experiments/segment_validate_20260603_timeline \
  -- \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --timeframe 2h --execution-timeframe 1min \
  --initial-capital 10000 \
  --no-maps
```

跑完后更新各策略子目录下的 `DECISION.md`（填入 `segment_summary.csv` 数字）。
