# 实时流计算（权威入口）

**最后更新**: 2026-04  
**相关文档**: [主文档索引](../../README.md)

本目录说明**当前主线**：`BinanceWebSocket` → `MultiSymbolManager` → `OrderFlowListener` → **`GenericLiveStrategy.decide`**（+ 可选 `OrderManager`），数据落 **`StorageManager` Parquet**（`--live-root` 下），与 **QuestDB 为中心** 的旧草案无关。

## 阅读顺序（编号短文）

1. [`01_一致性原则与契约.md`](./01_一致性原则与契约.md)
2. [`02_事件流与时间对齐.md`](./02_事件流与时间对齐.md)
3. [`03_特征计算_状态与缓存.md`](./03_特征计算_状态与缓存.md)
4. [`04_存储_回放与审计.md`](./04_存储_回放与审计.md)
5. [`05_补全_对账与异常处理.md`](./05_补全_对账与异常处理.md)
6. [`06_实盘稳定性运行手册.md`](./06_实盘稳定性运行手册.md)

**专题**（与 `run_live` 强相关）：

- [`数据补全架构.md`](./数据补全架构.md)
- [`实盘特征计算机制.md`](./实盘特征计算机制.md)

**事件回测（与实盘同逻辑链）**：[`docs/architecture/event_drive_backtest/`](../event_drive_backtest/)

## 实盘启动（主入口）

- **脚本**：[`scripts/run_live.py`](../../../scripts/run_live.py)  
- **命令与环境变量**：根目录 [`README_CN.md`](../../../README_CN.md)

说明摘要：

- 多策略由 **`LivePCM`** 与宪法配置协同；策略体为 **`GenericLiveStrategy`**（非 Nautilus `Strategy` 适配器主线）。
- 设置 **`MLBOT_ORDER_MANAGER_ENABLED=true`** 时注入 **`OrderManager`**（SL/TP、持仓时间、追踪止损等）；研究/回测默认可不启用。

## `reference/`

已清理与代码不符的 QuestDB 中心化长文；见 [`reference/README.md`](./reference/README.md)。

**历史 legacy**（旧流程/选型）：[`docs/archive/live_stream/legacy/`](../../archive/live_stream/legacy/)
