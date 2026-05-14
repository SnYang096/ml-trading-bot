# Strategy Map 指标语义与使用说明

本文定义 Strategy Map 看板里与策略执行相关的指标语义，便于实盘与回测做同口径对比。

## 视角与变量

- `scope`: `trend` 或 `hedge`。
- `job`: 由 `scope` 派生。
  - `trend` -> `quant-trend-fattail`
  - `hedge` -> `quant-hedge-multileg`
- `strategy` / `symbol` / `timeframe`: 在已筛选 `job` 的前提下级联查询。

这保证了趋势与多腿不会在同一组下拉里串数据。

## 事件指标（交易地图标记）

核心指标：

- `mlbot_strategy_event_total{scope,strategy,symbol,event,side}`（Counter）
- `mlbot_strategy_event_price{scope,strategy,symbol,event,side}`（Gauge）

标准事件名：

- `signal`: 策略信号通过（统计窗口聚合）。
- `entry`: 开仓成功（实盘下单成功后打点）。
- `exit`: 平仓成功（本地触发平仓或交易所回报同步关闭后打点）。
- `reject`: 被风控或规则拒绝（统计窗口聚合或执行路径拒绝）。

说明：

- `entry/exit` 用于和回测交易地图的开平点直接对齐。
- `signal/reject` 更偏过程监控，常用于解释为何有信号但没有成交。

## 看板推荐读法

### 策略视角（Strategy-first）

1. 选 `scope + strategy`，`symbol` 可先选 `All`。
2. 先看 `Position Snapshot`（`mlbot_position_qty` / `mlbot_position_notional_usdt`）确认当前真实持仓分布。
3. 再看 `Event Flow` 中 `entry/exit` 与 `signal/reject` 的节奏是否匹配。

### Symbol 视角（Symbol-first）

1. 固定 `symbol`，`strategy` 选 `All`。
2. 对比不同策略在同一 symbol 上的 `entry/exit` 密度与方向差异。
3. 配合 OHLC/特征层判断是否存在“同币不同策略行为冲突”。

## 与回测对比建议

- 对齐字段：`strategy`、`symbol`、`event(entry/exit)`、时间窗口。
- 对齐粒度：建议统一到 `5m` 或 `15m` 统计窗口，避免 scrape 抖动影响。
- 对齐结论：
  - 先比 `entry/exit` 次数与方向；
  - 再比持仓时间分布与回撤段；
  - 最后解释 `signal` 与 `reject` 差异（实盘执行约束导致）。
