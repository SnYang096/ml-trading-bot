### 目标

把“研究 / 回测 / 实盘”的特征计算统一成一条管线，并通过两类缓存显著加速：

- **Plan B FeatureStore（离线特征仓库）**：把特征落盘为分区 Parquet，研究/回测/实盘可直接读取复用
- **Plan A Incremental State Cache（增量状态缓存）**：对 stateful tick/orderflow 特征，跨月/跨批次保存 `final_state` 并作为下一段 `initial_state` 续算

同时解决历史上的稳定性问题：

- **宽表问题**：特征函数不再把整张宽表复制/回传，只保留 `output_columns`
- **OOM（132GiB）**：避免 train/test/volatility 之间复用错误 index 的 memory cache 导致 index union 爆炸
- **LightGBM 维度不一致（181 vs 61）**：预测时强制对齐训练列

---

### 当前架构概览

#### 1) 特征定义（DAG）

统一由 `config/feature_dependencies.yaml` 管理：

- **dependencies**：特征依赖
- **required_columns**：输入依赖列
- **output_columns**：输出列（合并/缓存时只允许这些列）
- **compute_func / compute_params / column_mappings**：计算函数与参数

#### 2) 特征计算（核心执行器）

`src/features/loader/parallel_computer.py`：

- 默认 **feature-level 顺序执行**（稳定、低内存、与实盘一致）
- 按依赖层级执行（每层可选并行，但默认关闭）
- **只合并 `output_columns`**
- **memory cache 仅在 index 完全一致时复用**（防止 index union 爆炸）

#### 3) 特征加载入口

`src/features/loader/strategy_feature_loader.py`：

- `load_features_from_requested(df, requested_features, ...)`
- 支持（opt-in）从 FeatureStore 读取：如果 store 已包含所需输出列，直接返回（不再计算）

---

### Plan B：FeatureStore（离线特征仓库）

实现：`src/feature_store/feature_store.py`

#### 分区布局（方案 1：全局分层）

以 layer 作为命名空间（缓存“切换开关”），分区按月：

- `{root}/{layer}/{symbol}/{timeframe}/{YYYY-MM}.parquet`
- `{root}/{layer}/{symbol}/{timeframe}/{YYYY-MM}.meta.json`

推荐的 layer：

- `base_v1`：OHLCV cheap 特征（快、稳定）
- `heavy_v1`：orderflow/WPT/spectrum 等重特征（先合并层，后续可再拆）

#### 生成特征仓库（示例）

用现有脚本按月增量写入：

```bash
cd /home/yin/trading/ml_trading_bot
python scripts/run_feature_store_sr_reversal.py --symbol BTCUSDT --timeframe 240T --layer heavy_v1 --output-dir feature_store
```

#### 读取特征仓库（opt-in）

`StrategyFeatureLoader.load_features_from_requested()` 支持：

- `feature_store_dir="feature_store"`
- `feature_store_layer="heavy_v1"`
- `feature_store_symbol="BTCUSDT"`
- `feature_store_timeframe="240T"`

如果仓库里已包含请求的 `output_columns`，会直接 join 并返回。

---

### Plan A：Incremental State Cache（增量状态缓存）

适用：tick/orderflow 这类 **stateful** 特征，跨月/跨批次需要连续性：

- VPIN：跨月 bucket 未填满部分需要延续
- Trade Clustering：run/window 的状态需要延续

#### 已支持的实现点

- **VPIN**：`src/data_tools/tick_loader.py`
  - 按月缓存（标准缓存可只保存 `final_state`，状态缓存可保存 `(buckets, final_state)`）
  - 支持把上月 `final_state` 纳入下月 cache key（state-aware）
- **Trade Clustering**：`src/features/time_series/utils_order_flow_features.py`
  - `compute_trade_clustering_from_ticks(... initial_state)` 返回 `(df, final_state)`
  - 按月流式处理 ticks，并跨月传递 state

#### 测试

- `tests/integration/test_vpin_incremental_cache.py`：验证 VPIN 跨月 state-aware cache key 会生成

---

### 关键稳定性修复记录（为何之前会炸）

#### 1) 132GiB OOM（形状 (34611346, 510)）

根因：

- memory cache 过去按 `feature_name` 复用，但 train/test/volatility 的 index 不同
- Pandas 合并时会做 index union，可能把 bar-index 与 tick-index 合并成千万级行数
- `result_df[all_cols]` 触发 block copy，导致尝试分配超大数组

修复：

- memory cache 只在 index 完全一致时复用，否则视为 miss
- 当 df index signature 改变时主动清空 memory cache
- `StrategyFeatureLoader` 返回列选择使用更保守的 `loc[:, cols]` 并去重列名，避免巨型复制

#### 2) LightGBM 预测维度不一致（181 vs 61）

根因：

- 训练时选了 61 个波动率特征，但预测时传入了更宽的 X

修复：

- `LightGBMTrainer` 保存训练列清单
- `predict()` 自动补齐缺列、丢弃多列、按训练列顺序重排

---

### 迁移建议（下一步重构方向）

1) **baseline 去 wrapper / 纯函数化**
- 让 baseline 特征函数“窄输入、窄输出、不要原地扩张 df”
- 最终删除临时 wrapper 文件

2) **Feature DAG 合约强化**
- 更严格地 enforce：required/output/index/dtype

3) **Processor 链（训练/实盘一致）**
- fill/clip/normalize/dtype downcast 统一化，并持久化参数用于实盘

4) **统一 CLI（console script）**
- 后续用 console script 替代 scripts/* 的 sys.path 注入


