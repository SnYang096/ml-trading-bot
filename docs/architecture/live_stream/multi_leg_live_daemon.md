# 多腿（Multi-Leg）实盘 / 影子盘守护进程架构

## 名字是什么意思

- **multi leg（多腿）**：指同一标的、同一账户下，策略**不是**「单一净仓位一条腿」模型，而是可能同时存在：
  - 多条挂单（多笔 `place` / `cancel`）
  - 多空同时持仓（对冲、网格、双开等）
  - 多条「加仓腿」与多条「止盈/强平腿」
  这类库存与旧版 `TradeIntent` 单仓位语义不同，需要单独的库存与对账模型。

- **live daemon（实盘守护进程）**：指长期运行的进程/循环：按周期拉取**已收盘**的 K 线与特征，驱动引擎产出动作，经风控与执行适配器下单，并周期性对账。对应入口脚本：`scripts/run_multi_leg_live.py`，核心循环类：`MultiLegLiveDaemon`。

本文描述 **独立多腿策略** 在 **影子盘 / 测试网** 下的运行路径，当前覆盖：

- `chop_grid`
- `dual_add_trend`

这些策略**有意**与方向性单仓位 `TradeIntent` 实盘路径分离，因为它们同时持有 long/short 库存、存在挂单、需要 gross/net 敞口上限，以及**交易所真实状态**与**本地策略状态**的对账。

---

## 与方向性实盘流程对齐（双进程、双账户并行）

**方向性趋势 / fat-tail 实盘**：入口 `scripts/run_live.py`。数据来自 Feature Bus：

```text
quant-feature-bus（行情 WebSocket → 磁盘 Feature Bus）
  → quant-trend-fattail / MultiSymbolManager → OrderFlowListener
  → GenericLiveStrategy（BPC / ME / SRB / TPC 等 TradeIntent 策略）
  → LivePCM / OrderManager → BinanceAPI
```

**Hedge 多腿进程（并行）**：入口 `scripts/run_multi_leg_live.py`。与方向性路径**并存**，不替换 `run_live`：

- **驱动 bar**：来自 `quant-feature-bus` 写出的磁盘 Feature Bus；离线回放可用 parquet。
- **交易所推送**：在 `--mode testnet/mainnet` 且使用真实 `BinanceAPI` 时，通过 **合约 User Data Stream（WebSocket，`BinanceUserStream`）** 接收订单/成交，经 `MultiLegLiveOrchestrator.on_execution_report` 进入引擎；它只作用于 hedge 多腿账户，不是行情 WebSocket。

**推荐：chop_grid / dual_add 使用另一个币安子账户**

- 与 BPC 等共用同一 API Key 会导致持仓/挂单在同一账户内混用，风险与对账边界都不清晰。
- 多腿 testnet 优先使用专用环境变量（与 `OrderManager` / `run_live` 的 testnet 变量解耦）：
  - `MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY`
  - `MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET`
- 若未设置上述变量，脚本仍回落到 `BINANCE_FUTURES_TESTNET_API_KEY` / `SECRET`（兼容单账户、单进程）。

同一台机器上并行跑两个进程时：`quant-trend-fattail` 加载方向性账户配置；`quant-hedge-multileg` 在独立 systemd 单元中使用 `MULTI_LEG_*` 专用密钥。

---

## Directional Trend / Fat-tail 路径

现有实盘主路径为：

```text
quant-feature-bus
-> disk Feature Bus
-> quant-trend-fattail / MultiSymbolManager
-> IncrementalFeatureComputer
-> GenericLiveStrategy
-> LivePCM
-> OrderManager
-> BinanceAPI
```

适用于 **单仓位意图** 的策略：

- 同一时间以单一方向决策为主
- 组合层通过 `LivePCM` 做 slot 仲裁
- 订单与仓位生命周期走 `OrderManager` / `PositionManager`
- 策略状态主要在 `GenericLiveStrategy` 与 PCM 状态中

这是 BPC / TPC / ME / SRB 等 `TradeIntent` 风格策略的合适路径。

---

## 多腿 Live 路径（新路径）

新路径为：

```text
慢信号：
disk Feature Bus（生产）或 Parquet 回放（离线）
-> FeatureStoreBarProvider
-> ChopGridLiveEngine / DualAddTrendLiveEngine
-> MultiLegLiveOrchestrator（编排器）
-> MultiLegPortfolioRiskGovernor（账户级风控）
-> MultiLegExecutionAdapter（下单适配器）
-> BinanceAPI
-> MultiLegReconciler（对账）
-> 引擎状态更新 / 回调

快执行：
Binance User Data Stream (WS)
-> on_execution_report
-> 引擎确认 leg 成交
-> 自动生成 per-leg reduce-only 保护单（TP / SL）
-> MultiLegExecutionAdapter
```

`--bar-source feature-store` 用于生产 Feature Bus 消费；`--bar-source parquet` 仅用于回放/影子。hedge 多腿生产入口不再打开 market WebSocket。

### 单行情源 Feature Bus 模式

若希望一台机器只保留一个 market WebSocket，可运行独立发布进程：

```text
run_market_feature_publisher.py
-> live/shared_feature_bus/bars_1min/*.parquet
-> live/shared_feature_bus/features/{TIMEFRAME}/*.parquet
```

随后两个消费者都读同一份已闭合快照：

```text
quant-trend-fattail / run_live.py
  MLBOT_FEATURE_SOURCE=bus
  -> FeatureBusReader
  -> LivePCM / OrderManager

quant-hedge-multileg / run_multi_leg_live.py
  --bar-source feature-store
  -> FeatureBusReader
  -> MultiLegLiveOrchestrator
```

此模式下，`run_live.py` 不再启动 market `BinanceWebSocketClient`，但仍保留 User Stream、PCM、`OrderManager` 与持仓管理；hedge 多腿继续使用独立账户/独立进程边界。

执行时钟与信号时钟分离：

- `features/{TIMEFRAME}` 是慢信号（如 `240T`、`120T`、`60T`、`2h`）。
- `bars_1min` 是执行时钟。trend/fat-tail bus 模式用它驱动软件止盈/止损检查；hedge 多腿 `feature-store` 用它驱动 grid fill、target exit、dual_add target/add 等执行动作。
- publisher 在任意 tick 到来时检测当前 10s 微窗口；只要价格相对该窗口 open 波动超过 3%，立即额外写一条 `_bar_kind=fast_intraminute` 的补充执行 bar。标准 1m bar 不被覆盖，后续仍正常写出。

守护进程入口示例：

```bash
python scripts/run_multi_leg_live.py --mode shadow --once
```

持续影子模式：

```bash
python scripts/run_multi_leg_live.py \
  --mode shadow \
  --strategies chop_grid,dual_add_trend \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
```

测试网需设置环境变量后运行（**第二账户**请用 `MULTI_LEG_*`）：

```bash
export MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY=...
export MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET=...

python scripts/run_multi_leg_live.py \
  --mode testnet \
  --bar-source feature-store \
  --symbols BTCUSDT \
  --strategies chop_grid,dual_add_trend
```

若与 `run_live` 共用同一 testnet 密钥（不推荐），必须显式添加 `--allow-shared-account`，脚本会打印 warning。

---

## 组件说明

### 引擎（Engines）

`ChopGridLiveEngine` 与 `DualAddTrendLiveEngine` **自持策略状态**，并产出普通 `dict` 动作（`place` / `cancel` / `market_exit` 等）。

**必须实现的 hooks**（供编排器与对账使用）：

```text
local_order_snapshots()
local_position_snapshots()
on_execution_results(results)
on_reconciliation_report(report)
on_execution_report(report)
```

引擎**不直接**调用 Binance；交易所交互统一走适配器。

### 编排器（Orchestrator）

`MultiLegLiveOrchestrator` 将「单引擎」接到「共享账户安全 + 执行」：

```text
引擎动作
-> 同步交易所挂单 / 持仓
-> governor.check_actions(...)
-> adapter.execute_actions(...)
-> reconciler.reconcile(...)
-> 回调引擎
```

并提供 `on_execution_report(report)`，便于将 Binance **用户数据流**里归一化后的成交回报（含部分成交）转发给引擎。

### 组合级风控（Portfolio Governor）

`MultiLegPortfolioRiskGovernor` 面向多腿策略的**账户级**硬约束，例如：

- 组合总 gross 名义敞口上限
- 组合总 net 名义敞口上限
- 单标的 gross / net 上限
- 最大挂单数（resting orders）

`cancel` 与 `market_exit` **始终放行**（默认视为降风险动作）。

### 执行适配器（Execution Adapter）

`MultiLegExecutionAdapter` 是薄层期货执行适配（旧名 `GridExecutionAdapter` 暂保留为兼容 alias）：

- `place` → Binance 限价单
- `cancel` → 撤单
- `market_exit` → 市价减仓（reduce-only 语义由 `BinanceAPI` 与对冲模式处理）
- `place_protection` → per-leg `STOP_MARKET` / `TAKE_PROFIT_MARKET`
- `cancel_protection` → 撤销保护单

真实执行需要 **Binance 合约对冲模式（Hedge Mode）**，避免 long/short 在单向持仓模式下被交易所净额合并，破坏「按腿」记账。

保护单默认按 logical leg 的数量挂 `reduce_only=True`，显式传 `positionSide=LONG/SHORT`，不使用 `closePosition=True`，避免误关整个 symbol side。

### 对账器（Reconciler）

`MultiLegReconciler` 对比 **引擎自持状态** 与 **交易所真相**：

- 本地有挂单、交易所已不存在（可能已成交/已撤/拒单）
- 交易所有挂单、本地无记录（孤儿单）
- 本地库存数量与交易所持仓数量漂移

对孤儿挂单可生成 `cancel` 建议动作。  
**持仓不一致**默认以报告为主，不自动全平：是否强平需要按策略/账户事先写死的人审策略。

---

## 与旧 Live 的关系

两条路径应 **并存、互补**，而不是互相替代。

**用 Directional Trend / Fat-tail Live** 当：

- 单仓位 alpha 策略
- `TradeIntent`
- `LivePCM` slot 仲裁
- `OrderManager` / `PositionManager` 生命周期

**用多腿 Live** 当：

- 中性网格库存
- 双开 + 顺势加仓库存
- 多空同时存在
- 策略自持「按腿」库存
- 通过 client id 前缀区分策略（如 `cg_`、`dat_`）

在真资金前，应在两条路径之上再加一层**账户总控**。当前 hedge 多腿路径自带 governor；trend/fat-tail 路径有 `LivePCM` / constitution 约束。未来可引入 **account master governor** 同时观测两条路径的总敞口。

---

## 策略组合关系

`chop_grid` 与 `dual_add_trend` **互补**：

- `chop_grid`：偏广义无趋势 / `semantic_chop` 震荡区
- `dual_add_trend`：偏非 chop、非 box 的趋势段

若 **同一期货账户** 同时跑两者，必须收紧账户级上限：

- 总 gross
- 总 net
- 单标的 gross/net
- 最大挂单数
- 单日亏损熔断
- 对账漂移触发 kill switch

---

## 当前限制

守护进程可用于 **影子 / 测试网硬化**，默认**不**等同于主网生产就绪。

后续生产化工作包括但不限于：

- 已完成：`BinanceUserStream` 在 `run_multi_leg_live.py` 的 testnet + `BinanceAPI` 下接入 `MultiLegLiveOrchestrator.on_execution_report`
- 已完成：`--bar-source feature-store` 消费 quant-feature-bus 写出的磁盘 Feature Bus，作为 hedge 多腿慢信号输入
- 已完成：per-leg reduce-only TP / SL 保护单动作与 `multi_leg_*` 独立持久化表
- 主网多腿：与 `MULTI_LEG_*` 对称的专用主网 API 环境变量（若扩展 `--mode live`）
- 进程重启后基于 `multi_leg_orders` / engine state 完整恢复保护单映射
- 明确「持仓漂移」策略：仅告警 vs 自动减仓
- 跨 trend/fat-tail Live 与 hedge multi-leg Live 的 account master governor
- 对「被拒动作」「对账漂移」做指标与告警
- 下单前按交易所规则做数量/价格步进与精度处理
- funding、保证金与强平缓冲监控

---

## 验证命令

核心测试：

```bash
pytest tests/unit/test_chop_grid_live_engine_hooks.py \
  tests/unit/test_dual_add_trend_live_engine.py \
  tests/order_management/test_multi_leg_daemon.py \
  tests/order_management/test_multi_leg_orchestrator.py \
  tests/order_management/test_multi_leg_risk_governor.py \
  tests/order_management/test_multi_leg_reconciliation.py \
  tests/order_management/test_grid_execution_adapter.py -q
```
