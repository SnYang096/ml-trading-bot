# Dual Add Trend Strategy Prototype

Research-only prototype for trading directional, non-chop crypto regimes with
bounded multi-leg inventory.

## 策略说明

`dual_add_trend` 的核心不是无限马丁，也不是普通网格。它先在趋势段开始时同时持有一份 `LONG` 和一份 `SHORT`，之后只允许沿当前趋势方向加仓。趋势方向每根 2H bar 动态更新；如果趋势掉头，策略会停止旧方向加仓，并强制处理旧方向库存，避免反向腿无限保留。

当前默认使用：

```text
初始: 1 LONG + 1 SHORT
加仓: 仅当前趋势方向
趋势掉头: close_offside_all
最大 gross exposure: 4 units
最大 net exposure: 2 units
段内最大亏损: 1%
```

## 适用阶段

这个策略应该放在 **非 chop、非 box 的趋势阶段**：

- `trend_confidence >= 0.80`
- `semantic_chop <= 0.25` 才允许入场
- 持仓期间要求 `trend_confidence >= 0.50`
- 持仓期间要求 `semantic_chop <= 0.40`
- 默认排除稳定 box：`exclude_box_prefilter: true`

为什么不放在 chop / box：

- `chop_grid` 更适合无明确方向的震荡段，它靠上下网格收割回摆。
- 稳定 box 更适合 CRF 这类边界反转策略。
- `dual_add_trend` 的收益来自趋势延续时沿趋势加仓；如果放在 chop 内，会频繁掉头、频繁强平旧方向腿，手续费和库存拖尾会吞掉利润。

## 止盈与费用

止盈必须是 fee-aware。当前默认：

```text
net_target_distance = max(0.25 * ATR14, 0.05% * entry_price)
tp_distance = fee_buffer + net_target_distance
fee_buffer = 2 * fee_bps * entry_price
```

研究脚本中 `fee_bps=4.0` 表示每侧 `4 bps`，因此一笔完整开平仓按 `8 bps` round trip 扣费。所有 `summary.csv` 和 `dual_add_segments.csv` 的 `pnl_per_capital` 都是扣费后结果。

全周期费用测算：

- 交易数：`15,791`
- round-trip fee：`8 bps`
- gross before fee：`9.0087`
- fee drag：`3.1582`
- net after fee：`5.8505`
- 费用吞掉约 `35.1%` 的 gross PnL

这说明费用非常重要；不能用固定 `$10` 毛利润止盈，必须确保扣除手续费后仍有足够净利润。

## 风控规则

必须同时满足：

- 禁止无限加仓：`max_adds_per_side: 3`
- 限制总仓位：`max_gross_exposure_units: 4`
- 限制净敞口：`max_net_exposure_units: 2`
- 段内硬止损：`max_loss_per_segment: 0.01`
- 趋势掉头时处理旧方向腿：默认 `close_offside_all`
- regime 失效时强制退出所有库存

掉头处理是关键。动态顺势加仓如果不处理旧方向腿，收益仍为正，但尾部会明显变差：

| Variant | Net PnL | Worst Segment | Risk Stop Rate |
| --- | ---: | ---: | ---: |
| keep old legs | `2.3323` | `-5.64%` | `5.73%` |
| close old add legs | `2.8347` | `-2.56%` | `3.80%` |
| close all offside legs | `3.1631` | `-2.13%` | `2.57%` |

因此默认选择 `close_offside_all`。

## Evidence Snapshot

Script:

```bash
python scripts/diagnose_dual_add_trend.py \
  --start 2022-01-01 --end 2026-03-31 \
  --regime trend \
  --add-mode trend \
  --flip-action close_offside_all \
  --tp-atr-mult 0.25 \
  --tp-pct 0.0005 \
  --max-loss-per-segment 0.01 \
  --max-gross-exposure 4 \
  --max-net-exposure 2 \
  --max-adds-per-side 3 \
  --exclude-box \
  --out-dir results/dual_add_trend/turbo-full-cycle
```

Full cycle: `2022-01-01..2026-03-31`, 2H, BTC/ETH/SOL/BNB/XRP/ADA.

Fee-aware net results:

- Segments: `3,118`
- Trades: `15,791`
- Trade win rate: `91.1%`
- Segment win rate: `76.7%`
- Net PnL after fees: `5.8505`
- Worst segment: `-2.13%`
- Risk stop rate: `2.76%`
- Max gross units: `4`
- Max net units: `2`

Equal-weight annualization:

```text
equal_weight_cumulative_return = total_net_pnl / 6 symbols = 97.51%
period_years = 4.244
CAGR = 17.40%
```

This is a capital-bucket estimate: each symbol receives one equal strategy
capital bucket, and PnL is averaged across symbols. It is not a leveraged account
equity curve and does not include funding, liquidation engine details, or extra
market-impact slippage.

## Current Status

Do not wire into the generic single-position `TradeIntent` / `event_backtest`
path yet. Like `chop_grid`, this strategy needs a dedicated multi-leg inventory
simulator because it depends on simultaneous long/short legs, add-on inventory,
trend-flip exits, gross exposure, net exposure, and segment-level forced exits.

The config in this folder mirrors `chop_grid` for research organization. A
production implementation would need first-class multi-leg execution accounting,
funding, exchange margin rules, liquidation buffers, and order fill simulation.
