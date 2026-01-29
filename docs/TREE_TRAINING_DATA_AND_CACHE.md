# 树模型训练：数据与特征来源（FeatureStore vs Cache）

## 1. 树训练用的是什么？

树策略训练（`train_strategy_pipeline.py`）**同时**用到两套机制，顺序是：

| 层级 | 用途 | 路径/机制 | 说明 |
|------|------|-----------|------|
| **原始数据** | OHLCV K 线 | `--data-path`（默认 `data/parquet_data`） | 由 `DataHandler` → `MarketDataLoader` 加载；会先看 **timeframe 缓存**，没有再从 parquet 按月读取并 resample，然后写入缓存。 |
| **特征** | 特征矩阵 | **FeatureStore**（主） + **FeatureComputer 缓存**（辅） | 先按 **FeatureStore** 读（按月 parquet）；读不到或版本不对再 **现场计算**，计算时用 **cache/features** 的按月 .pkl 缓存，算完会写回 FeatureStore。 |

所以：**树模型训练 = 用 FeatureStore（读优先），算特征时用 FeatureComputer 的 cache；原始数据用 data-path + timeframe 缓存。**

---

## 2. 原始数据（K 线）

- **入口**：`DataHandler.load_ohlcv(symbol, timeframe, start_date, end_date)`。
- **实现**：`MarketDataLoader.load_data()`：
  - 先看 **timeframe 缓存**（如 `cache/timeframes` 下按 symbol+timeframe 的 parquet）是否存在；
  - 有则直接读缓存；
  - 没有则从 `data_path` 下该 symbol 的 **按月 parquet** 里选覆盖 `[start_date, end_date]` 的文件，resample 成目标 timeframe，再写入缓存。
- **结论**：K 线要么来自 **已有 timeframe 缓存**，要么来自 **data/parquet_data** 里该 symbol 的月度文件。若 parquet_data 里只有某几月（例如 BTCUSDT 只有 2024-03、04、08、2025-10），且没有旧缓存，那实际只会用到这几月，**不会有真正的 2023 年数据**。

---

## 3. 特征（FeatureStore + FeatureComputer cache）

- **入口**：`run_feature_pipeline()` → `StrategyFeatureLoader.load_features_from_requested()`。
- **参数**：`--feature-store-dir`（默认 `feature_store`）、`--feature-store-layer`（由 `resolve_layer_name(config_dir)` 按 features.yaml 等算 hash，如 `features_xxxx`）。
- **流程**：
  1. 若提供了 `feature_store_dir` + `feature_store_layer` + symbol + timeframe，则 **先读 FeatureStore**：`store.read_range(spec, index.min(), index.max())`（按月 parquet）。
  2. 若读到的表里 **包含所有 requested 特征列** 且版本一致，则直接返回，**不再计算**。
  3. 否则 **现场计算**：用 `FeatureComputer.compute_features()`，内部会：
     - 使用 **cache_dir**（默认 `cache/features`）下的 **按月 .pkl** 缓存（按 feature + 月份）；
     - 缺的月份再算，算完写回 cache 并写回 FeatureStore（auto materialize）。
- **结论**：树训练 **优先用 FeatureStore**；只有读不到或版本不对时才算，算的时候用 **cache/features** 的按月 cache。训练“很快”通常是因为：FeatureStore 里已有该 layer 的完整月份，或 cache/features 里已有大部分月份。

---

## 4. 为什么训练很快？

可能原因组合：

1. **FeatureStore 已热**：该策略的 layer（由 features.yaml 等 hash 得到）之前跑过，已写入 `feature_store/features_xxx/BTCUSDT/240T/` 下多个月份，本次直接读 parquet，几乎不计算。
2. **FeatureComputer 缓存已热**：`cache/features` 里已有大量「特征×月份」的 .pkl，本次只补算缺的月份。
3. **Timeframe 缓存已热**：K 线从 `cache/timeframes` 等直接读，没有重新 resample。
4. **数据量不大**：例如 2023-01 到 2025-05 的 4H 约 2377 根 bar，特征 33 个，算一遍或读一遍都不算重。

**若你改了 features.yaml**（例如加了 ME archetype 特征），layer hash 会变，FeatureStore 会变成“新 layer”，通常 **读不到**，会走现场计算；这时若 `cache/features` 里已有部分特征×月份，仍会很快。

---

## 5. 数据有没有问题？怎么自查？

- **“从 2023 年开始训练”**：真正用到的是 **DataHandler 能拿到的日期范围**。若 `data/parquet_data` 里 BTCUSDT 只有 2024-03、04、08、2025-10 等零散月份，且没有旧的时间段缓存，那实际 **不会** 有 2023 年数据；`--start-date 2023-01-01` 只是裁剪，不会凭空变出 2023 的 K 线。
- **建议自查**：
  1. **原始数据范围**：看 `data/parquet_data` 下 `BTCUSDT_*.parquet` 有哪些月份；再看是否有 `cache/timeframes`（或 DataHandler 实际用的缓存路径）里 BTCUSDT 240T 的缓存，以及该缓存的起止时间。
  2. **训练日志**：日志里会打印 `Cropped data to [2023-01-01, 2025-05-31], rows=2377` 一类信息；若上游只有 4 个月，实际行数会少很多（例如约几百根），可据此判断是否真的覆盖 2023。
  3. **FeatureStore**：看 `feature_store/` 下对应 layer（如 `features_*`）里 `BTCUSDT/240T/` 有哪些 `YYYY-MM.parquet`，确认是否覆盖你期望的训练区间。

---

## 6. 小结

| 问题 | 答案 |
|------|------|
| 树模型训练用 FeatureStore 还是 cache？ | **都用**：特征以 **FeatureStore** 为主（读优先），计算时用 **FeatureComputer** 的 **cache/features**（按月 .pkl）；原始 K 线用 **data-path** + **timeframe 缓存**。 |
| 为什么训练这么快？ | 多半是 FeatureStore 或 cache/features 或 timeframe 缓存已热，或数据量不大（两千多 bar）。 |
| 数据会不会有问题？ | 若 parquet_data 里没有 2023 年的月份且没有旧缓存，则“从 2023 年开始训练”实际不会包含 2023 年数据；建议按上面步骤查 parquet、timeframe 缓存和日志里的行数/日期范围。 |
