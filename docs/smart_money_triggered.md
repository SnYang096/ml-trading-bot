## Smart Money Triggered (Alltick) — Realtime + Manual Signals

### 组件
- `smart_money_triggered/alltick_ws.py`：Alltick websocket 客户端（心跳 22000 + 订阅 22004，推送 22998）
- `smart_money_triggered/tick_store.py`：100ms tick 聚合并按 `data/smart_money_triggered/<symbol>/<date>.parquet` 持久化
- `smart_money_triggered/signals.py`：订单流指标与信号判定
- `smart_money_triggered/engine.py`：实时采集、手动/定时计算信号的入口
- `smart_money_triggered/config_loader.py`：读取 `config/smart_money_triggered/*.yaml` 中的 token 与符号列表

### 依赖
- `pip install websockets pandas numpy pyyaml`
- 环境变量：`ALL_TICK_KEY`（或直接在 `config/smart_money_triggered/key.yaml` 中写 token 字符串）

### 快速开始
```python
import asyncio
from smart_money_triggered import SmartMoneyEngine, load_settings

engine = SmartMoneyEngine(load_settings())

# 1) 启动实时订阅与100ms聚合，保存到 data/smart_money_triggered
task = asyncio.run(engine.start_realtime())  # 或在 event loop 内 await

# 2) 手动计算某天(默认当天)14:50之前的信号
res = engine.compute_signal_for_day("000001.SZ", trading_date="2024-12-11")
print(res.decision, res.debug)

# 3) 定时每天14:50触发一次（事件驱动调度）
# await engine.run_daily_signal_loop(time_str="14:50")
```

### 指标说明（基于 100ms 聚合窗）
- `takebuy`: 主动买量 / 总量
- `cvd_slope`: 近 30 分钟累计成交量差（买卖 delta 累积）的线性回归斜率
- `cluster_score`: 买/卖占比 ≥ 0.7 的窗口占比
- `vpin`: 按成交量分桶（默认 1000）计算的成交量不平衡概率
- `vwap`: 截至截点的 VWAP
- `decision` 规则（默认）：`takebuy>0.65 & cvd_slope>0 & cluster>0.7 & price>0.98*VWAP & vpin<0.75` → LONG；反向条件近似 → SHORT

### 目录与文件
- 配置：`config/smart_money_triggered/china_stocks.yaml`, `cryptos.yaml`, `key.yaml`
- 数据输出：`data/smart_money_triggered/<symbol>/<YYYY-MM-DD>.parquet`

### 提示
- Alltick 要求 10s 心跳；断线自动重连。
- 订阅请求为覆盖模式，`alltick_ws` 每次重连都会发送全量 symbol 列表。
- 若需 QuestDB/数据库写入，可在 `TickStorage.append` 中替换为自定义 sink。

