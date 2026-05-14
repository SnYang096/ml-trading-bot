# Live 三进程节拍与存储说明

本文说明当前主网三进程架构下，`quant-feature-bus`、`quant-trend-fattail`、`quant-hedge-multileg` 各自多久读取 / 计算 / 落盘 / 同步账户与订单。结论基于当前代码和 `.github/workflows/deploy.yml` 中的 systemd 参数。

## 总览

| 进程 | 主要职责 | 当前线上节拍 |
| --- | --- | --- |
| `quant-feature-bus` | 唯一 Binance 行情 WebSocket owner；聚合 1min bar；计算特征；写磁盘 Feature Bus | tick 实时进入；已完成 1min bar 每分钟落盘；特征每 60 秒检查一次，达到 `--feature-compute-interval-minutes=15` 才批量计算并落盘 |
| `quant-trend-fattail` | 趋势 / fat-tail 账户消费者；只读 Feature Bus；执行 directional strategies | `MLBOT_FEATURE_BUS_POLL_SECONDS=5`，每 5 秒检查 bus 是否有新 bar / feature；账户 REST 指标默认每 30 秒刷新；User Data Stream 事件实时进入 |
| `quant-hedge-multileg` | hedge 多腿账户消费者；只读 Feature Bus；执行多腿策略 | `--poll-seconds 5`，每 5 秒读取 feature-store / 1min execution bars；`--reconcile-interval-seconds` 默认 60 秒做交易所对账 |

这里的“轮询”不等于“每次都重算或下单”。消费者通常是高频检查磁盘上是否有新 timestamp；只有出现新 feature / 新 bar / 到达对账周期时才进入较重逻辑。

## `quant-feature-bus`: 行情、特征计算与磁盘落地

部署命令见 `.github/workflows/deploy.yml` 中 `quant-feature-bus.service`：

- `--symbols ${MLBOT_LIVE_SYMBOLS}`
- `--feature-bus-root live/shared_feature_bus`
- `--live-storage-base live/highcap/data`
- `--feature-compute-interval-minutes` 未显式传入，使用 `scripts/run_market_feature_publisher.py` 默认值 `15`
- `--fast-bar-threshold-pct 0.03`
- `--fast-bar-bucket-seconds 10`

### 行情进入

`run_market_feature_publisher.py` 启动 `BinanceWebSocketClient`，每条 Binance aggTrade tick 进入：

1. `FastMoveBarEmitter.on_tick(tick)`：如果 10 秒桶内价格波动超过 `3%`，写一条 `fast_intraminute` execution bar 到 Feature Bus。
2. `manager.on_trade_tick(...)`：进入对应 symbol 的 `OrderFlowListener`。

因此，bus 进程的 tick 输入是实时的，不是 1 分钟轮询。

### 1min bar 落盘

`OrderFlowListener` 在 tick 跨过新的分钟边界时 finalize 上一根 1min bar：

- 写入 live storage：`live/highcap/data/...`
- 调用 `on_bar_callback`
- publisher 注入的 callback 会调用 `FeatureBusWriter.append_bar_1m(...)`

Feature Bus 路径：

- `live/shared_feature_bus/bars_1min/<SYMBOL>.parquet`
- `live/shared_feature_bus/latest/bars_1min/<SYMBOL>.json`

也就是说，正常情况下 **每个活跃 symbol 每分钟最多新增一根已完成 1min bar**。另外如果触发 fast bar，则可能在一分钟内额外写一条 `fast_intraminute` bar。

### 特征计算与落盘

publisher 的每个 symbol listener 都启动 `_periodic_tasks()`：

- 每 60 秒醒一次；
- 如果距离 `last_feature_compute_time` 超过 `feature_compute_interval_minutes * 60`，就执行 `_compute_and_save_15min_features()`；
- 当前默认 `feature_compute_interval_minutes=15`，所以特征计算目标节拍是 **15 分钟一次**；
- 计算时会读取长窗口历史 1min bars / ticks，再合并内存 buffer；
- 主 timeframe 来自策略配置，例如 BPC / ME 等 meta 中定义的 timeframe；publisher 会把每个计算出的 timeframe 写到 Feature Bus。

Feature Bus 路径：

- `live/shared_feature_bus/features/<timeframe>/<SYMBOL>.parquet`
- `live/shared_feature_bus/latest/features/<timeframe>/<SYMBOL>.json`
- 当 timeframe 是 `120T` 时，代码还会额外写一份别名 `2h`。

注意：`primary` 不是一个自然时间周期；它是消费者侧配置中选择的特征快照名 / 默认入口。真实特征时间框架仍取决于 publisher 写入的 `<timeframe>`。

## `quant-trend-fattail`: bus 消费、信号、账户与订单同步

部署命令见 `.github/workflows/deploy.yml` 中 `quant-trend-fattail.service`：

- `MLBOT_FEATURE_SOURCE=bus`
- `MLBOT_FEATURE_BUS_ROOT=live/shared_feature_bus`
- `MLBOT_FEATURE_BUS_POLL_SECONDS=5`
- `MLBOT_FEATURE_BUS_MAX_STALENESS_SECONDS=1800`
- `MLBOT_FEATURE_BUS_BARS_LOOKBACK=240`
- `MLBOT_LIVE_WARMUP_DAYS=0`

### 读取 bus 的频率

`scripts/run_live.py` 中 `_run_external_feature_bus_mode()` 使用：

- `poll_seconds = MLBOT_FEATURE_BUS_POLL_SECONDS`，当前线上是 **5 秒**；
- 每轮读取 `latest_snapshot_age_seconds(...)`；
- `provider.poll_bars()` 读取新 1min bars；
- `provider.poll()` 读取新 feature events；
- 有新 event 时调用 `listener._handle_features(...)`。

所以 trend 进程 **每 5 秒检查一次 bus**，但真正策略决策只在 bus 出现新 feature event 时发生。publisher 当前通常每 15 分钟计算一次 feature，因此 trend 的主要开仓信号也通常跟随这个特征节拍，而不是每 5 秒重新发信号。

### 交易信号多久发出

trend 信号的触发点是 `listener._handle_features(...)`：

- `LivePCM.decide(...)` 调用各策略；
- `StatsCollector.record_strategy_eval(...)` 记录漏斗；
- PCM 通过后生成 `TradeIntent`；
- 若系统模式允许交易，`TradeExecutor.execute(...)` 下单。

因此 trend 的“发信号”节拍由 **Feature Bus 新 feature event** 决定。当前 publisher 默认每 15 分钟计算特征，所以可理解为 **15 分钟级别信号**；但进程为了低延迟消费，仍每 5 秒 poll bus。

### 1min bar 对 trend 的作用

trend 也会从 bus 读取 1min bars，并通过 `_enforce_bus_execution_bars(...)` 给已有仓位做执行侧管理，例如止损 / 止盈 / position tracker 检查。这里更接近“仓位管理刷新”，不是每分钟都重新开仓。

### 账户与订单同步

trend 账户 / 订单有两条路径：

1. **User Data Stream**：
   - `MultiSymbolManager` 初始化 `BinanceUserStream`；
   - `run_live.py` 在 bus 模式启动后 `manager.user_stream.start()`；
   - listenKey keepalive 默认 **30 分钟**；
   - 成交 / 订单事件实时进入 `listener.on_execution_report(...)`；
   - 账户更新事件实时进入 `listener.on_account_update(...)`。

2. **REST / Prometheus 指标刷新**：
   - `_periodic_market_update()` 中默认 `MLBOT_MARKET_DATA_INTERVAL=30`；
   - 每 **30 秒**调用 `METRICS.update_account_data()` 拉 `fapi/v2/account`；
   - 同一周期也刷新 funding / mark price / OI 等公开市场指标。

另外，`OrderFlowListener._periodic_tasks()` 里每 **30 分钟**会做一次 Binance 时间同步与 slot / 持仓一致性检查，用于释放 stale slot。

### trend 数据库存储

trend 使用两类 SQLite：

1. 监控统计库：
   - 路径：`$MLBOT_LIVE_BASE/data/db/live_monitor.db`，默认 `live/highcap/data/db/live_monitor.db`
   - 表：`stats_15min`
   - 内容：每次 `_handle_features()` 结束后 flush，包含当前 symbol 的漏斗、持仓快照、system health。

2. 订单管理库：
   - 路径：`MLBOT_ORDER_MANAGEMENT_DB_PATH`，默认 `data/order_management.db`
   - 主要表：`orders`、`positions`、`position_operations`、`stop_loss_trailing`、`safety_state`、`slots_state`、`add_position_state`
   - 写入点：`OrderManager.place_order()` / cancel / position manager / constitution runtime 等。

`stats_15min` 是监控汇总；`orders` / `positions` 才是交易执行状态。

## `quant-hedge-multileg`: feature-store 消费、执行与对账

部署命令见 `.github/workflows/deploy.yml` 中 `quant-hedge-multileg.service`：

- `MLBOT_MULTI_LEG_SYMBOLS=BTCUSDT`
- `MLBOT_ACCOUNT_SCOPE=multi_leg`
- `--bar-source feature-store`
- `--feature-bus-root live/shared_feature_bus`
- `--feature-store-timeframe primary`
- `--feature-store-execution-timeframe 1min`
- `--poll-seconds 5`
- 未显式传 `--reconcile-interval-seconds`，使用默认 **60 秒**

### 读取 bus 的频率

`MultiLegLiveDaemon.run_forever()` 每轮：

1. 调用 `run_once()`
2. `METRICS.multi_leg_daemon_polls_total.inc(1)`
3. `await asyncio.sleep(self.poll_seconds)`

当前线上 `--poll-seconds 5`，所以多腿 **每 5 秒醒一次**。空轮不会下单，只是检查是否有新数据 / 是否到对账周期。

### 多腿读什么数据

`FeatureStoreBarProvider.latest_closed_bars()` 每轮对每个 symbol：

1. 读取 `latest_features(symbol, timeframe=self.timeframe)`；
   - 线上传的是 `primary`；
   - 如果 `primary` 没有对应数据，就不会产出事件。
2. 读取 `latest_bars_1m(symbol, after=last_seen)`；
   - 有新的 1min execution bars 时，每一根新 bar 都会和最新 signal feature 合成一个 `MultiLegBarEvent`；
   - 初次启动只回放 `initial_backfill_bars` 根，默认 1 根。
3. 如果没有新 1min bar，但 signal timestamp 变了，也会产出一个基于 signal 的事件。

因此多腿是：

- **5 秒检查一次磁盘**
- **1min bar 有新增时推进执行逻辑**
- **signal feature 有新增时推进信号逻辑**

这就是它看起来比 trend 更“高频”的原因：它显式把 `1min` execution bars 纳入多腿 engine 的运行节拍，且每 5 秒检查一次是否有新 1min bar。

### 多腿交易信号多久发出

多腿信号来自每个 engine 的 `on_bar(...)` / action 生成逻辑（由 `MultiLegLiveDaemon.run_once()` 调用）。实际节拍取决于：

- `--poll-seconds`：检查新数据的最大延迟；
- `--feature-store-execution-timeframe 1min`：有新 1min execution bar 时可触发一轮；
- `--feature-store-timeframe primary`：signal feature 更新时可触发一轮；
- 策略自身是否在该 bar / feature 下产生 action。

所以多腿不是每 5 秒发信号，而是 **每 5 秒检查；有新 1min bar 或新 signal feature 时才进入 engine；engine 决定是否输出 action**。

### 多腿账户与订单同步

多腿同步分三层：

1. **User Data Stream**：
   - `run_multi_leg_live.py` 在 mainnet/testnet 下启动 `BinanceUserStream`；
   - 成交 / 订单事件实时进入每个 orchestrator 的 `on_execution_report(...)`；
   - listenKey keepalive 默认 **30 分钟**。

2. **REST 账户指标**：
   - `run_multi_leg_live.py` 的 `_periodic_process_metrics()` 默认也使用 `MLBOT_MARKET_DATA_INTERVAL=30`；
   - 每 **30 秒**刷新进程 health 与 `METRICS.update_account_data()`；
   - 因为 `MLBOT_ACCOUNT_SCOPE=multi_leg`，这里使用 `MULTI_LEG_*` key。

3. **订单 / 持仓对账**：
   - `--reconcile-interval-seconds` 默认 **60 秒**；
   - 或者本轮产生 action 时立即对账；
   - 对账会调用 `adapter.sync_open_orders(symbol)` 和 `adapter.sync_positions(symbol)`；
   - 再由 `MultiLegReconciler` 比对 engine local state 与 exchange truth。

这部分 REST 调用比 trend 更明显，因为 multi-leg 的 daemon 每轮都判断是否需要 reconcile，且一旦有 action 就会立即拉交易所 open orders / positions。

### multi-leg 数据库存储

多腿使用独立库：

- 路径：`--multi-leg-db-path`，默认 `data/multi_leg_order_management.db`
- 当前部署未显式传入该参数，所以使用默认路径

主要表：

- `multi_leg_runs`：每次 daemon run 的 run_id、mode、strategies、symbols
- `multi_leg_orders`：本地 / 交易所订单 ID 映射、策略、symbol、leg、状态
- `multi_leg_positions`：engine 自己维护的 leg inventory，不等同于 classic `positions`
- `multi_leg_execution_reports`：User Data Stream 的成交 / 订单事件审计
- `multi_leg_reconciliation_snapshots`：定期对账快照与 drift 诊断

多腿也会共享 schema 文件里的 classic 表定义，但业务上隔离使用 `multi_leg_*` 表。

## “轮询频率”与“计算/下单频率”的区别

| 项目 | 轮询 / 检查 | 真正重计算 / 信号 | 账户 REST | 订单/持仓同步 |
| --- | --- | --- | --- | --- |
| bus publisher | WebSocket tick 实时；定期任务每 60 秒检查 | 默认每 15 分钟批量计算 feature；1min bar 每分钟落地 | 无交易账户 | 无交易订单 |
| trend | 每 5 秒 poll Feature Bus | 新 feature event 到达时调用 PCM / 策略；当前通常 15 分钟级别 | 每 30 秒 `fapi/v2/account` 指标刷新；User Stream 实时账户事件 | User Stream 实时订单事件；每 30 分钟 slot / 持仓一致性检查 |
| multi-leg | 每 5 秒 poll feature-store | 新 1min execution bar 或新 signal feature 到达时 engine 运行；是否 action 由策略决定 | 每 30 秒账户指标刷新；User Stream 实时账户事件 | User Stream 实时订单事件；默认每 60 秒 reconcile，或有 action 时立即 reconcile |

## 当前容易误解的点

1. trend 的 5 秒 poll 不是每 5 秒重算特征；特征由 bus publisher 计算并落盘。
2. trend 的 15 分钟更接近特征 / 信号节拍，不是进程休眠 15 分钟。
3. multi-leg 的 5 秒 poll 会产生频繁 daemon tick，但 `bars=0 actions=0` 为空轮；它的有效执行节拍主要由新 1min execution bar、新 signal feature、以及 60 秒对账决定。
4. multi-leg 比 trend 更容易产生 REST 压力，因为它有独立 reconcile 机制，且有 action 时会立即拉 open orders / positions。
5. 如果要降低 multi-leg 资源与 REST 压力，优先考虑把 `--poll-seconds` 从 5 提到 10-30，或把 `--reconcile-interval-seconds` 从 60 提高；但会增加新 1min bar / action 被处理的最大延迟。

## 建议

当前配置适合低延迟观察，但多腿空轮较多。若策略不需要 5 秒级响应，可考虑：

- `--poll-seconds 10`：最大多 5 秒延迟，空轮减半。
- `--poll-seconds 30`：最大 30 秒级响应，适合低频多腿策略。
- `--reconcile-interval-seconds 120` 或更高：降低 REST open orders / positions 压力；保留 User Data Stream 作为实时成交来源。

不要把 `quant-feature-bus` 的 WebSocket 或特征计算搬回消费者。消费者应该继续只读 Feature Bus，避免 trend 与 multi-leg 反复重复计算行情和特征。
