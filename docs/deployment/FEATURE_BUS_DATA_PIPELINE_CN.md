# 实盘三层数据管线（Archive / Memory / Bus）

本文澄清 `quant-feature-bus` 容器内部的数据流向，以及下游策略和 Trade Map 各自从哪一层读数据。配套节拍说明见 [`LIVE_CADENCE_AND_STORAGE_CN.md`](LIVE_CADENCE_AND_STORAGE_CN.md)。

## TL;DR

- **Archive、Memory、Bus 是三个独立的存储层**，由 `quant-feature-bus` 在 tick 进入时**并行**写入；不是一个层流向另一个层。
- **特征计算只读 archive + memory，不读 bus**。Bus 是 publisher 的输出，给下游策略和 Trade Map 用。
- **Warmup 不创造 archive 也不写 bus**：它只在容器启动时从 archive 读一段，恢复 memory 的滑窗状态。
- **Trade Map 长历史靠 archive + macro daily 拼接**，bus 不必长。

## 三层数据

| 层 | 路径 | 内容 | 容量 / 保留 | 写入方式 |
|----|------|------|------------|----------|
| **Archive** | `live/highcap/data/bars/<SYMBOL>/<YYYY-MM-DD>.parquet` 等 | 按天分片的 1m bars / ticks / features | 长期保留（数月-数年） | WS tick → `OrderFlowListener` 跨分钟时 append 当天 parquet |
| **Memory** | 进程内存 `MemoryWindow` | 最近 4h 的 1m bars 和 tick buffer | 4h 滚动（`memory_window_hours`） | WS tick → `OrderFlowListener.add` |
| **Bus** | `live/shared_feature_bus/bars_1min/<SYMBOL>.parquet` 与 `features/<TF>/<SYMBOL>.parquet` | 滚动 1m bars + 已算好的特征 | 由 `max_rows`/`warmup_days` 控制，**不是天数也不是事件队列** | WS bar 关闭 → `on_bar_callback` → `append_bar_1m`；compute 完成 → `append_features` |

另一个独立数据：**`live/highcap/data/macro/spot_klines/`** 与 **`spot_weekly_ema200/`**，由 Vision spot daily 转出来，专供 publisher 启动时计算 weekly EMA200 种子，以及 Trade Map 长窗口（1d/1w）。

## 数据流（Publisher 内部）

```mermaid
flowchart TB
    WS[Binance WebSocket] -->|aggTrade tick| Listener[OrderFlowListener]

    Listener -->|每分钟关一根 bar| AppendArchive[append 当天 parquet]
    AppendArchive --> Archive[(Archive<br/>live/highcap/data)]

    Listener -->|放入内存| MemoryWindow[memory_window<br/>4h 滑窗]

    Listener -->|on_bar_callback| AppendBus[append_bar_1m]
    AppendBus --> Bus[(Bus<br/>shared_feature_bus<br/>bars_1min)]

    Compute[每 15 分钟<br/>compute_features_batch]
    MemoryWindow -.读.-> Compute
    Archive -.读最近一段.-> Compute

    MacroSeed[(macro/spot_weekly_ema200<br/>Vision 长历史 seed)] -.读.-> Compute

    Compute -->|append_features| BusF[(Bus<br/>features/<TF>)]

    style Archive fill:#1e4a72
    style Bus fill:#5a3070
    style BusF fill:#5a3070
    style MemoryWindow fill:#2d5f3f
    style MacroSeed fill:#774422
```

关键点：

- **Archive 和 Bus 独立写入**：tick 进来时一并触发两边 append，bus 不是"压缩存档"，archive 不是"由 bus 滚出来"。
- **特征计算只用 archive + memory**：见 `OrderFlowListener._compute_15min_features` → `_merge_bars(bars_disk, bars_buffer)` → `compute_features_batch`。
- **Bus 是输出端**：features 通过 `FeatureBusDecisionSink` 写入 `bus/features/<TF>/`；1m bars 通过 `on_bar_callback` 写入 `bus/bars_1min/`。

## Warmup 真正做了什么

`MultiSymbolManager.warmup_all(days=180)` → `OrderFlowListener.warmup` → `_restore_state`：

```
特征计算已改为磁盘批量模式 (compute_features_batch)，
不再需要通过回放 bars/ticks 重建流式状态。
```

含义：

1. 从 `storage_manager` 读最近 N 天数据（**只读 archive，不写 bus**）。
2. 把最近一批 1m bars 放入 `memory_window`，让滑窗"非空"。
3. 恢复 `last_feature_compute_time` 等时间戳，避免启动后立即重复计算。

> **不会**把 N 天数据 dump 到 bus parquet。Bus 仍然是从启动时点 0 开始累积 1m bars。

## 下游消费者从哪里读

```mermaid
flowchart LR
    Bus[(Bus features + bars_1min)] --> R[FeatureBusReader]

    R -->|latest_features| Trend[trend/swing live]
    R -->|latest_features + latest_bars_1m| Chop[chop_grid live]

    UI[Trade Map UI] -->|fetch_ohlcv 短窗| Stitch[stitch_live_storage_and_bus]
    Bus -.bus tail.-> Stitch
    Archive[(Archive)] -.历史段.-> Stitch

    UI -->|fetch_ohlcv 1d/1w| MacroFlow[macro spot_klines<br/>+ bus tail]
```

| 消费者 | 读 archive | 读 memory | 读 bus | 读 macro daily |
|--------|------------|-----------|--------|----------------|
| Publisher 内部 `compute_features_batch` | ✅ 主历史 | ✅ 最近 4h | ❌ | ✅ weekly EMA200 seed |
| trend/swing live (`run_live.py`) | ❌ | ❌ | ✅ features + bars | ❌ |
| chop_grid live (`run_multi_leg_live.py`) | ❌ | ❌ | ✅ features + bars | ❌ |
| Trade Map UI 短窗 (15min, 2h) | ✅ stitch | ❌ | ✅ stitch | ❌ |
| Trade Map UI 长窗 (1d, 1w) | ❌ | ❌ | ✅ tail | ✅ 主历史 |

策略消费 bus 时通常只取**最新一行**或 `bars_lookback=240`（≈ 4 小时），与 bus 总长度无关。

## Bus 容量的真实含义

`scripts/run_market_feature_publisher.py` 直接用 `--max-rows` 创建 writer，不再受 `--warmup-days` 影响：

```python
writer = FeatureBusWriter(args.feature_bus_root, max_rows=int(args.max_rows))
```

线上 systemd 当前是 `--max-rows 10080 --warmup-days 7` → 上限 **10080 行（≈ 7 天 1m bars）**。

历史上曾经有 `effective_max_rows_for_warmup(max_rows, warmup_days)` 自动放大逻辑（`max(max_rows, warmup_days*24*60)`），用于"console 直接读 bus 看长历史"的旧形态。Trade Map 引入 stitching 之后这条耦合不再有意义，反而让 backfill 脚本误把 prod bus 砍短，已删除。

但 **bus 实际行数 = 容器持续运行的分钟数**：

- 容器刚启动 → bus = 0 行
- 运行 1 天 → bus ≈ 1440 行
- 运行 5 天 → bus ≈ 7200 行
- 运行 ≥ 180 天 → bus 触顶 259200 后开始 tail

注释里"warmup window"的来历是历史的：在 Trade Map 加 stitching 之前，console 只读 bus，所以 bus 必须能装下完整可视窗口。**stitching 实现后，这个上限可以缩到 1 周（10080 行）就够策略 lookback 用**，UI 长历史靠 archive / macro 拼接。

### 性能对比（粗估）

| `max_rows` | 时间跨度 | parquet 大小 | `pd.read_parquet` 全量读 |
|-----------|----------|--------------|----------------------------|
| 5,000 | ~3.5 天 | ~200 KB | ~10 ms |
| 10,080 | ~7 天 | ~400 KB | ~20 ms |
| 259,200 | ~180 天 | ~10 MB+ | ~500 ms+ |

下游每分钟 poll，每个 symbol 都全量读，所以缩短 bus 对内存和延迟都有收益。

## 一次性 Backfill 的 non-shrinking 语义

`scripts/sync_feature_bus_bars_from_archive.py` 调用 `merge_bars_1m(..., preserve_history=True)`：

- **不**应用 `tail(max_rows)`，所以脚本默认 `max_rows=5000` 不会把 prod 已有的 ~7000 行 bus 砍短。
- 在线 publisher 走默认 `preserve_history=False`，仍然按 `max_rows` 滚动。

`auto_gap_fill` 把补出来的 bar 同步进 bus 时也走 `preserve_history=True`，原因相同。

## 配置参考

| 项 | 当前值 | 历史值 | 备注 |
|----|--------|--------|------|
| `--warmup-days` | 7 | 180 | warmup 已不再被特征计算依赖。`compute_features_batch` 改为每次直接从 archive 读 150 天；`memory_window` 容量 4h，所以 warmup_days 只要≥1 都够 `_restore_state` 填满 |
| `--max-rows` | 10080 | 3000 | 7 天 1m bars。策略 lookback 通常 240 bars (≈4h)，UI 长历史靠 archive / macro 拼接 |
| `MLBOT_AUTO_GAP_FILL_MIN_GAP_MINUTES` | 60 | 60 | <60min 小洞 archive 自身就有，影响 Trade Map 显示但不影响特征 |

`max_rows` 从 259200 降到 10080 后，bus parquet 从 10 MB 量级降到约 400 KB，下游每分钟全量 `pd.read_parquet` 从 ~500ms 降到 ~20ms。180 天 warmup 启动时把 ~26 万行 1m bars 灌入 4h cap 的 memory_window，绝大多数被立即 evict，纯属浪费启动时间和内存峰值；7 天足够。

## 流式 incremental vs 磁盘批量 (compute_features_batch)

> "磁盘不能计算吧，还是要读到内存？" — 对，两条路径都要读到内存。区别是 **state 是否跨 tick 持久化** 以及由此决定的 **重启代价**。

```mermaid
flowchart LR
    subgraph stream [流式 incremental（已废弃）]
        T1[tick] --> S1[on_tick state.add]
        T2[tick] --> S2[on_tick state.add]
        T3[tick] --> S3[on_tick state.add]
        S1 -.persist.-> S2 -.persist.-> S3
        S3 --> Out1[输出最新一行]
        Out1 --> R1[重启需回放<br/>180 天 ticks/bars<br/>重建 EMA/quantile state]
    end

    subgraph batch [磁盘批量（当前）]
        Tick[每 15 分钟] --> Read[load_range 150 天 bars<br/>→ DataFrame]
        Read --> Compute[一次性 rolling/quantile/atr<br/>整段算完]
        Compute --> Out2[输出最新一行]
        Out2 --> Drop[DataFrame 释放]
        Drop --> R2[重启无需回放]
    end
```

| 维度 | 流式 incremental | 磁盘批量 (`compute_features_batch`) |
|------|------------------|----------------------------------------|
| 单次开销 | O(1)，每根 bar 维护几个累加器 | O(N)，N ≈ 150 天 × 1440 ≈ 22 万行 |
| 触发频率 | 每根 bar 都调 | 每 15 分钟 1 次 |
| 内存占用 | 长期保持（EMA / quantile buffer） | 计算窗口短，结束即释放 |
| 跨 tick 状态 | **持久化在 publisher 内存** | **无**，每次重新算 |
| 重启代价 | 必须回放历史 bars 重建 state | 直接读盘算一次 |
| 出错可观察性 | 状态飘移难定位 | 每次窗口独立可重放、diff 容易 |

代码上的体现：

```python
# src/live_data_stream/order_flow_listener.py
bar_lookback_days = 150
bar_start = (now - timedelta(days=bar_lookback_days)).strftime("%Y-%m-%d")
bars_disk = self.storage_manager.bar_1min.load_range(self.symbol, bar_start, bar_end)
bars_merged = self._merge_bars(bars_disk, bars_buffer)  # archive 150d ⊕ memory 4h
features = self.feature_computer.compute_features_batch(bars_1min=bars_merged, ...)
```

迁移到批量后，**state 不再活在 publisher 内存里，只活在 archive parquet 上**。这就是 `--warmup-days 180` 失去意义的根因——回放出来的 state 无人消费。

## warmup_days 的现状

`--warmup-days` 已经不直接驱动特征计算，但还服务两件事：

1. **`_restore_state` 给 `memory_window` 喂启动 bars**：4h cap，所以 warmup_days≥1 就够（实现里取 tail，避免无谓地把数十万行送进 add() 后立刻被淘汰）。
2. **`_startup_gap_repair` 的回扫窗口**：`max(--auto-gap-fill-startup-lookback-hours, warmup_days*24)`，决定启动时 archive 完整性扫描多久。

> 已删除的耦合：`max_rows = effective_max_rows_for_warmup(max_rows, warmup_days)`。bus 容量现在只由 `--max-rows` 决定。

未来如果完全改用 archive-only 启动校验，`--warmup-days` 也可以下沉成 `--startup-scan-hours` 之类的语义化参数。

## 排查路径

1. **Trade Map 看到 K 线断开** → 先看 archive 是否完整（`/app/live/highcap/data/bars/<SYM>/<date>.parquet`）；如果 archive 完整、bus 有洞，跑 `sync_feature_bus_bars_from_archive.py`。
2. **策略读不到最新 bar** → 检查 publisher `_make_bar_write_callback` 是否 throw、`bus/bars_1min/<SYM>.parquet` 的 mtime 是否在更新。
3. **特征计算失败** → 看 publisher 日志 `compute_features_batch`；这条路径不依赖 bus，问题多在 archive / memory 端。
4. **weekly EMA200 异常** → 检查 `live/highcap/data/macro/spot_weekly_ema200/` seed 是否存在。
