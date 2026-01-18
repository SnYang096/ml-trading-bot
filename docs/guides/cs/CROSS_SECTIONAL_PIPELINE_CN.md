# Cross-Sectional（CS）全流程：因子评估 → 筛选 → 回测（YAML 驱动）

本文档面向 `src/cross_sectional/`，目标是：
- 给定一组币种（token universe）与一个因子集合
- 在一个时间窗口内做 **截面因子有效性评估**（rank-IC + Long/Short quantile spread）
- 自动筛选出一组因子
- 输出可复现的结果目录与清单

> 关键设计：**同一套 panel（MultiIndex: timestamp,symbol）** 贯穿评估/筛选/训练；评估阶段内置一个简单的 Long/Short 回测（含 turnover + fee）。

---

## 1) 概念与产物

### Panel（CS 面板）
CS 面板是一个表（或 MultiIndex DataFrame），至少包含：
- `timestamp`
- `symbol`
- 一组 factor columns（因子/特征列）
- `future_return_<horizon>`（目标收益；如不存在可用 `close` 推导）

### 评估输出（factor-eval）
`mlbot cross-section factor-eval` 会输出：
- `summary.csv`：每个因子的
  - **rank-IC** 统计（`ic_mean/ic_ir/...`）
  - **Long/Short 回测** 统计（Sharpe、turnover、fee 后收益等）
- `long_short_timeseries__<factor>.csv`：每因子一份时序（gross/net/cum 等）

### 筛选输出（select）
`mlbot cross-section select` 会输出：
- `selected_factors.txt`
- `selection_summary.json`

---

## 2) 推荐路径（可复现）：先落盘 panel，再跑 pipeline

### 2.1 一键跑 CS pipeline（YAML）

```bash
mlbot cross-section workflow \
  --config config/cross_sectional/pipeline_alpha101_cs_rank_4h_feature_store.yaml \
  --no-docker
```

产物默认落到 `output_root`（示例：`results/cross_sectional/pipeline_alpha101_cs_rank_4h/`）：
- `factor_eval/summary.csv`
- `factor_eval/summary.json`
- `factor_eval/long_short_timeseries__*.csv`
- `selected_factors.txt`（若开启 select）
- `selection_summary.json`
- `pipeline_manifest.json`
- `index.html`（总报告页）

---

## 2.5（推荐）CS 缓存机制：monthly FeatureStore（第二次跑复用）

为了满足“第一次可以慢、第二次换时间窗口可复用”的需求，CS 提供了一个**月分区 FeatureStore**：

- 写入路径：`feature_store/<layer>/<symbol>/<timeframe>/YYYY-MM.parquet`
- 复用点：
  - `mlbot cross-section rank`（按日取某个 bar 做排序）
  - `mlbot cross-section factor-eval`（以 FeatureStore 为数据源评估因子）
  - `mlbot cross-section pipeline`（panel.source=feature_store 时会先落盘 `panel_from_feature_store.parquet`，然后可继续 select/train）

### 构建缓存（不含 ticks，OHLCV-only）

```bash
mlbot cross-section build-store \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml \
  --factor-set crypto_ts_compatible_core \
  --no-docker
```

> 默认会跳过已存在的月份文件；换时间窗口只会补齐缺失月份（可用 `--overwrite` 强制重算）。

### 2.5.1 Alpha101（CS 横截面 rank）缓存

Alpha101-CS 必须跨资产一起算（因为有 `rank(axis=1)`），所以 `build-store` 会按月：
1) 加载全部 symbols（含 warmup）
2) 计算一次 alpha101_cs 面板
3) 再按 symbol 拆分写入各自的月分区文件

```bash
mlbot cross-section build-store \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml \
  --factor-set crypto_alpha101_cs_rank \
  --features-store-layer cs_alpha101_cs_rank_4h_v1 \
  --no-docker
```

Alpha101-CS “为何不走 per-feature DAG”的架构说明见：
- `docs/architecture/CROSS_SECTIONAL_ALPHA101_FEATURESTORE_ARCH_CN.md`

---

## 3) 因子集合（YAML factor sets）

项目内置了一个推荐集合：
- `config/cross_sectional/cs_factor_sets_crypto.yaml`
  - `factor_sets.crypto_cs_core`: 主要是 `cs_crypto_*`（相对动量/相对波动/volume share/orderflow spread 等）

在 pipeline YAML 里引用：

```yaml
factor_eval:
  factor_set_yaml: config/cross_sectional/cs_factor_sets_crypto.yaml
  factor_set: crypto_cs_core
```

---

## 4) 直接跑 factor-eval（不走 pipeline）

### 4.1 从 panel 文件评估

```bash
mlbot cross-section factor-eval \
  --input results/feature_exports/cs_panel.parquet \
  --factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml \
  --factor-set crypto_cs_core \
  --horizon 12 --min-assets 8 --quantiles 5 --fee-bps 2 \
  --output-dir results/cross_sectional/factor_eval_demo \
  --no-docker
```

### 4.2（可选）从 FeatureStore 评估（不落盘 panel）

> 适合快速看结果；但不如 panel 文件可复现。

```bash
mlbot cross-section factor-eval \
  --features-store-root feature_store \
  --features-store-layer <features_layer> \
  --timeframe 240T \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --factors rsi,macd,atr \
  --horizon 12 \
  --output-dir results/cross_sectional/factor_eval_from_fs \
  --no-docker
```

---

## 5) 训练（可选）

当你有了 `selected_factors.txt`，可以把它喂给 CS 模型训练：

```bash
mlbot cross-section train \
  --input results/feature_exports/cs_panel.parquet \
  --feature-file results/cross_sectional/pipeline_demo_crypto_4h/selected_factors.txt \
  --horizon 12 \
  --model boosting \
  --output-dir results/cross_sectional/models/demo \
  --no-docker
```


