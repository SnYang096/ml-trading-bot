# CS Alpha101（横截面 rank）架构：为什么不走 TS 依赖 DAG，以及如何做可复用缓存

本文解释 `alpha101_cs_*`（原始 Alpha101 横截面版本）在本仓库的落地方式：
- 为什么它**不适合**像 TS 特征那样用 `feature_dependencies.yaml` 的 per-symbol DAG 计算
- 为什么它应该被建模为一个 **multi-asset 计算节点**
- 如何通过 **monthly FeatureStore** 达到“第一次慢、第二次换时间窗口复用”的目标

---

## 1) 关键差异：TS DAG vs CS Alpha101

### TS（依赖 DAG / per-symbol）
TS 的特征大多可以写成“单个 symbol 的 DataFrame → 输出列”，所以用 DAG 很自然：
- 节点粒度：`feature_key`（例如 `atr_ratio_f`）
- 依赖：A 依赖 B（输出列复用）
- 缓存：按 symbol / timeframe / month / feature_key 复用

### CS Alpha101（横截面 rank / multi-asset）
原始 Alpha101 的实现大量使用 **横截面 rank**（按同一 timestamp 在多资产间做 percentile rank）：
- 输入不是单资产 df，而是 “tickers in columns, dates in rows” 的宽表
- `rank(df)` 定义为 `df.rank(axis=1, pct=True)`（每一行跨资产排名）

因此：
- 如果你把它拆成 per-symbol DAG，会丢失“横截面 rank”的语义（需要同时看到所有资产）
- 即使强行拆，也会造成重复计算（每个 symbol 都要重算全截面），缓存收益很差

结论：Alpha101-CS 更合理的建模方式是：
> **按月/按窗口：加载多币种 → 宽表计算一次 → 再按 symbol 拆分落盘**

---

## 2) 本仓库的实现方式（multi-asset node）

### 2.1 计算器
- 计算入口：`src/cross_sectional/alpha101_cs_rank.py`
- 输入：`{symbol: OHLCV_df}`（不需要 ticks）
- 内部做：
  - 组宽表 `open/high/low/close/volume`（columns name = `ticker`）
  - 推导：`returns`、`adv20`、`vwap`（OHLC 近似）
  - 调用 `src/cross_sectional/factors/alpha_functions.py` 中的 alpha00x/alpha101
  - 输出：MultiIndex `(timestamp, symbol)`，列名 `alpha101_cs_XXX`

### 2.2 为什么没有把它放进 `feature_dependencies.yaml` 的 DAG？
原因不是“DAG 不重要”，而是**DAG 的节点粒度变了**：
- 对 Alpha101-CS 来说，最小可复用单元不是“单个特征”，而是“跨资产一次性计算（按月）”。

因此我们把它接入到 CS 的 FeatureStore 构建阶段，而不是 TS 的 per-feature DAG。

---

## 3) 缓存：monthly FeatureStore（换时间窗口复用）

### 3.1 写入结构
使用统一的月分区结构：

`feature_store/<layer>/<symbol>/<timeframe>/YYYY-MM.parquet`

### 3.2 构建入口
- CLI：`mlbot cross-section build-store`
- 实现：`src/cross_sectional/feature_store_builder.py`
  - 如果 `factor_set` 包含 `alpha101_cs_` 前缀列，会走 multi-asset 分支：
    - 按月加载所有 symbols（含 warmup）
    - 计算一次 alpha 面板
    - 月内切片
    - 按 symbol 拆分写入各自的 `YYYY-MM.parquet`

### 3.3 复用点
一旦 FeatureStore 落盘：
- `mlbot cross-section rank`：按日读取月分区 → 取当日 bar → 排序
- `mlbot cross-section factor-eval`：可直接用 FeatureStore source 做因子评估
- `mlbot cross-section pipeline`：`panel.source=feature_store` 时会先落盘 panel，然后可继续 select/train

---

## 4) 测试与质量闸门（按仓库标准）
针对 Alpha101-CS，我们补齐了：
- **No-lookahead**：只改变未来数据，历史输出不变
- **Streaming vs Batch parity**：分段（带 overlap warmup）与一次性全量输出一致
- **Multi-asset normalization**：rank 输出在每个 timestamp 上的范围/均值合理，且对“全局计价单位缩放”不敏感


