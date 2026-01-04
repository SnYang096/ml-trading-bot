# 上线最小闭环（MVP）工作流：从特征搜索到可部署模型

本文件是上线/交付导向的 **MVP 版本流程**。README 只保留“最小命令 + 入口链接”，详细解释统一放在这里。

## 关键概念：为什么不是用 Nautilus 做 OOS？

- **`holdout-eval`（VectorBT 回测）**：用于“统计意义上的 OOS 验收”（快、可批量、与 feature-search 口径一致）。
- **Nautilus（事件驱动）**：用于“回测=实盘一致性验证”（慢、强依赖实盘执行/事件流细节，目的不是优化 Sharpe）。

因此：
- **OOS 指标验收**：用 `mlbot diagnose holdout-eval`
- **实盘一致性验收**：用 Nautilus 相关文档/工具（当前 `mlbot backtest nautilus` 仍是 TODO）

## 时间段怎么切？

推荐使用“最后 6 个月 holdout”的模板：

- 总窗口：\([T0, T_{end}]\)
- 研发/迭代窗口：\([T0, T_{holdout\_start})\)
- 最终 Holdout：\([T_{holdout\_start}, T_{end}]\)（只验收，不再调参）

例：`T_end=2025-10-31`，最后 6 个月 holdout：
- `T_holdout_start=2025-05-01`
- 研发窗口：`2024-01-01 ~ 2025-04-30`

## MVP 闭环：每一步跑什么命令？

### 0) 质量闸门（推荐）

```bash
make test-key-features-all
mlbot diagnose feature-contract --no-docker
```

### 1) 特征搜索（研发期，禁止看见 holdout）

一键：Pool‑B（factor-eval）→ semantic+Pool‑B 搜索（pipeline）→ 写回 YAML + 报告

```bash
mlbot diagnose poolb-semantic-search \
  --strategies <strategy_dir_name> \
  --tag <TAG> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 --end-date 2025-04-30 \
  --search-algo pipeline \
  --expand-semantic-singletons \
  --regen-poolb --rerun-search \
  --no-docker
```

产物（示例）：
- Pool‑B：`results/pools/<strategy>/pool_b/<TAG>/features_pool_b.yaml`
- 搜索结果：`results/feature_group_search/<strategy>_pipeline_poolb_semantic_<TAG>/feature_group_search_result.json`
- 写回 YAML：`config/strategies/<strategy>/features_suggested_pipeline_poolb_semantic_<TAG>.yaml`

### 2) Rolling（研发期/上线后监控）

用途：walk-forward 观察跨月稳定性（仍然只跑到研发窗口末尾，避免污染 holdout）。

```bash
mlbot train rolling \
  --config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start 2024-01-01 --end 2025-04-30 \
  --initial-train-months 6 --min-train-months 3 \
  --no-docker
```

上线后持续监控（不是 warm-start 增量学习；只是继续往后滚）：

```bash
mlbot train rolling \
  --config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --update-only \
  --no-docker
```

### 3) 最终验收：6 个月 Holdout（只验收，不再调参）

```bash
mlbot diagnose holdout-eval \
  --config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --train-start-date 2024-01-01 \
  --holdout-start-date 2025-05-01 \
  --holdout-end-date 2025-10-31 \
  --output-root results/holdout_eval \
  --deterministic \
  --no-docker
```

验收标准（建议）：
- **硬门槛**：results.json 正常、非占位 sharpe、trades 达标、sharpe 有限数、ModelArtifact/preprocessor 存在
- **软门槛**：相对 baseline 提升、回撤可控、相关性为正、与 rolling 最近月中位水平一致

### 4) 上线产物：训练最终模型（尽量用满数据）

通过 holdout 后，训练“最终可部署模型”（全窗训练）：

```bash
mlbot train final \
  --config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --output-root models \
  --deterministic \
  --no-docker
```

默认输出：
- `models/<strategy_name>/`（ModelArtifact + preprocessor + results 等）

### 5) 事件驱动一致性验证（Nautilus）

定位：验证“回测=实盘”，不是拿来做特征/模型选择的主指标。

参考文档：
- `docs/live_stream/reference/Nautilus_Trader_集成指南.md`
- `docs/live_stream/06_实盘稳定性运行手册.md`
- `docs/live_stream/07_与NautilusTrader对齐清单.md`


