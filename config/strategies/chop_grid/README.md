# Chop Grid Strategy Prototype

Research-only prototype for trading broad no-trend crypto regimes.

## 策略说明

`chop_grid` 是一个中性网格策略，不预测下一段是上涨还是下跌。它只判断当前是否进入广义无趋势/震荡区域；一旦进入，就在当前价格上下方同时挂多层限价单，用价格来回震荡收割网格间距。

开启网格的条件：

- `semantic_chop >= 0.40`
- 默认排除稳定 box：`exclude_box_prefilter: true`
- 含义：优先交易边界不稳定、但没有明确趋势的 chop 区；稳定 box 更适合 CRF 这类边界反转策略，不是当前网格的主场。

为什么默认排除稳定 box：

- 不是因为稳定 box 不能做网格。理论上，边界清楚的 box 也可以在箱体内部挂网格。
- 当前默认排除，是出于策略分工和统计结果：稳定 box 的边界更适合 CRF 做“靠下沿做多、靠上沿做空”的定向 range fade；`chop_grid` 主要解决的是边界不清楚但没有趋势的震荡。
- 统计上，`box_prefilter` 网格覆盖率较低，且历史诊断里 `box_prefilter` 网格只在 `4/5` 个 period 为正，2024 出现过较大亏损；`chop_not_box` 在同一批测试里 `5/5` 为正，更稳定。
- 稳定、窄、规则的 box 后面经常会有突破风险。网格在突破前容易逐层成交同一侧库存；一旦突破没有回归，就会变成一边倒持仓，最后靠 regime exit 强制平仓，盈亏比会恶化。
- 因此当前生产候选先采用 `chop_not_box`：把稳定 box 留给 CRF 或单独的 `box_grid` 变体，避免把两种不同市场结构混在一个参数里。

后续可以单独测试一个 `box_grid` 变体：只在稳定 box 内开网格，但要求更严格的最大持仓层数、更快退出、突破方向保护和更保守的强制止损。

退出网格的条件：

- `semantic_chop < 0.25` 时，认为趋势可能重新出现。
- 退出时撤掉所有未成交限价单。
- 已成交但未止盈的库存强制平仓；回测中这部分按 taker fee + forced-exit slippage 扣成本。

方向逻辑：

- 没有单一方向，不做趋势预测。
- 当前价格下方挂买入限价单，成交后形成 `LONG` 库存，目标是在上一格止盈。
- 当前价格上方挂卖出限价单，成交后形成 `SHORT` 库存，目标是在下一格止盈。
- 每一层库存独立成交、独立止盈；策略整体保持中性网格，而不是全仓做多或全仓做空。

网格间距：

```text
spacing = max(grid_atr_mult * ATR14, grid_min_pct * price)
默认 = max(0.50 * ATR14, 0.4% * price)
```

默认最多每边 `3` 层，等额 notional。

示例：

假设 BTC 当前价格是 `100000`，`ATR14 = 1200`。

```text
spacing = max(0.50 * 1200, 0.004 * 100000)
        = max(600, 400)
        = 600
```

则网格为：

```text
买入限价:
L1:  99400
L2:  98800
L3:  98200

卖出限价:
S1: 100600
S2: 101200
S3: 101800
```

如果价格先跌到 `99400`，L1 买单成交，形成一份多头库存；之后价格反弹到 `100000`，这份多头止盈退出。

如果价格先涨到 `100600`，S1 卖单成交，形成一份空头库存；之后价格回落到 `100000`，这份空头止盈退出。

如果 chop 失效，例如 `semantic_chop` 从 `0.45` 降到 `0.20`，策略不再等待回归，而是撤掉未成交订单，并强制平掉所有库存。

主要风险：

- 趋势突然出现，一侧库存持续被套。
- K 线 high/low 触价不等于实盘一定成交，存在排队、部分成交和 missed fill。
- 强制退出可能需要 taker 单，成本高于普通网格止盈。
- 合约还需要考虑 funding 成本。

当前费用口径：

- 网格止盈：maker entry + maker exit。
- 强制退出：maker entry + taker exit + forced-exit slippage。
- 按持仓时长扣 `funding_cost_bps_per_8h`。

## Thesis

Stable causal boxes are rare and hard to identify before the fact. Crypto more
often enters broad `semantic_chop`: BB compression plus low multi-horizon
direction confidence. In those periods, exact upper/lower boundaries are
unstable, so a neutral grid can harvest oscillation without requiring a clean
box.

## Proposed Rules

- Entry regime: `semantic_chop >= 0.40`
- Exit regime: close all grid inventory once `semantic_chop < 0.25`
- Preferred universe: high-liquidity symbols only
- Grid spacing: start with `0.50 ATR`, compare with `0.75 ATR`
- Max levels: `3` per side
- Positioning: neutral grid, fixed notional per level
- Fees/slippage: diagnostic assumes `4 bps` per side-equivalent round trip

## Evidence Snapshot

Script: `scripts/diagnose_chop_grid.py`

Base run (`grid_atr_mult=0.75`, `2022..2026Q1`, BTC/ETH/SOL/BNB/XRP/ADA):

- `semantic_chop` covers about `29%` of bars.
- `chop_not_box` covers about `27%` of bars.
- `box_prefilter` covers only about `8%` of bars.
- `semantic_chop` grid: positive in `5/5` periods.
- `chop_not_box` grid: positive in `5/5` periods.
- `box_prefilter` grid: positive in `4/5` periods, with a large 2024 loss.

Spacing sensitivity:

- `0.50 ATR` in `chop_not_box`: `5/5` positive periods, strongest total PnL.
- `0.75 ATR` in `chop_not_box`: `5/5` positive periods, lower turnover.
- `1.00 ATR` in `chop_not_box`: `5/5` positive periods, lower PnL.

## Current Status

Do not wire into the generic single-position pipeline yet. The current
`event_backtest` path is position-intent based, while this strategy needs
multi-level inventory management and forced regime exit. Keep it as a standalone
grid simulator until the execution model supports grid inventory explicitly.
