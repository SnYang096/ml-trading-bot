# ML Trading Bot（中文）

本仓库包含因子研究、模型训练、回测与数据管道的生产就绪组件。**本 README 保持尽量短**：只提供“命令 + 推荐流程 + 入口文档链接”。研究解释型内容已迁移到独立文档。

---

## 快速开始

1) 创建虚拟环境（conda、venv 等）并激活  
2) 以可编辑模式安装：

```bash
pip install -e .[dev]
```

3) （可选但推荐）安装 Git pre-commit 钩子：

```bash
make install-hooks
```

4) 查看命令：

```bash
mlbot --help
mlbot analyze --help
mlbot train --help
mlbot diagnose --help
mlbot optimize --help
mlbot data --help
```

---

## 数据管道（下载 → 转换 → 训练）

```bash
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

mlbot data convert

# 或一次性跑完
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

### Universe 驱动（多币种批量：下载 + 转 parquet）

`pipeline-universe` 会按 universe 配置解析 symbol 列表，并 **下载后立刻转成 Parquet**：

```bash
mlbot data pipeline-universe \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a \
  --start-year 2023 \
  --start-month 1 \
  --no-docker
```

### 扩展数据：市值（Market Cap）与资金费率（Funding Rate）

#### 1) 市值快照（CoinGecko）

```bash
export COINGECKO_API_KEY='...'

mlbot data update-market-cap \
  --config config/data/market_cap.yaml \
  --max-age-days 7 \
  --no-docker
```

结果默认落盘：
- `data/market_cap/<SYMBOL>.parquet`
- `data/market_cap/market_cap_manifest.json`

#### 2) Binance 资金费率（按月 ZIP → Parquet）

```bash
mlbot data download-funding-rate \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2024 \
  --start-month 1 \
  --progress-every 10 \
  --no-docker
```

结果默认落盘：
- ZIP：`data/funding_rate/zip/`
- Parquet：`data/funding_rate/parquet/`

---

## 推荐工作流（最小闭环，6 步）

> 推荐先读：`docs/时序模型/完整流程指南.md`

### 步骤 0：验证特征正确性（推荐）

```bash
make test-key-features-all
```

### 步骤 1：因子筛选（Filter，生成 Pool B）

> **注意**：`features_all.yaml` 只包含原始特征，不包含语义特征。语义特征通过 `config/feature_groups_<strategy>_semantic.yaml` 单独管理。

```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --output-dir results/pools/sr_reversal_long/pool_b \
  --export-yaml results/pools/sr_reversal_long/pool_b/features_pool_b.yaml \
  --remove-correlated \
  --filter-by-best-lag \
  --no-docker
```

**说明**：
- Pool B 包含经过 IC/IR 筛选的原始特征（DTW、EVT、GARCH、Hilbert 等）
- 语义特征不在这里评估，而是通过语义 groups 管理

### 步骤 2：特征组合搜索（Wrapper 主力，feature-group-search）

```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_long \
  -s BTCUSDT \
  -t 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --objective Sharpe_mean \
  --min-trades 10 \
  --max-steps 6 \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_long/pool_b/features_pool_b.yaml \
  --deterministic \
  --writeback-yaml config/strategies/sr_reversal_long/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_reversal_long_best_combo \
  --no-docker
```

**它是怎么做 group search + Pool B search 的？（核心机制）**
- **候选组来源（groups）**：默认会自动选择
  - `config/feature_groups_<strategy_dir>_semantic.yaml`（存在则优先）
  - 否则 `config/feature_groups.yaml`
  - 再否则才回退到内置默认组
- **Pool B 如何参与**：`feature-group-search` 会自动读取 `factor-eval` 的 Pool B 输出（默认约定在 `results/pools/<strategy_dir>/pool_b/`），把其中“候选特征”**当作单特征组（singleton groups）**加入候选池，与语义 groups 一起竞争。
- **职责分离**：
  - **Pool B**：发现未被语义化的有效原始特征（数据驱动，经过 IC/IR 筛选）
  - **语义 groups**：提供经过语义化的特征（人工筛选，经过语义化）
  - 两者互补，一起竞争，找到最佳组合
- **搜索算法**：**Greedy Forward Selection（贪心前向）**  
  先跑 **baseline（只用 base features）**，然后每一步把“当前已选组合 + 某个候选组”都跑一遍 multi-seed，选择能让 `--objective`（如 `Sharpe_mean`）提升最多的那个组加入；若**没有任何组能严格提升**，就停止（并记录 stop_reason）。
- **为什么不会组合爆炸**：它不是穷举所有组合，而是每一步只做一次“加一组”的比较；复杂度大致是 \(O(\text{steps} \times \text{groups} \times \text{seeds})\)。

> **详细工作流**：参考 `docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md`

**Semantic groups 单例展开（可选）**：
- 默认情况下，semantic groups 作为整体选择（如 `trade_cluster_scene: [trade_cluster_scene_semantic_scores_f]`）
- 但同一个 semantic feature node 可能包含多个语义（如 compression/ignition/absorption/exhaustion），这些语义可能对策略有相反的作用
- 使用 `--expand-semantic-singletons` 可以将 semantic groups 展开为单例（每个输出列单独作为一个候选组）
- 这样可以选择更精细的语义（例如只选择 `ignition` 而不选择 `exhaustion`）
- **注意**：展开后候选组数量增加，评估时间可能增加（但语义特征数量通常较少，影响有限）
- 详细说明：参考 `docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`

- **产物**：
  - HTML 报告（含 baseline/stop_reason/每步候选评分与被拒原因）
  - `features_suggested.yaml`（可写回，含 provenance 元数据）

### 步骤 3：特征消融（Wrapper 验证 / A-B 对比）

```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"
```

### 步骤 4：模型对比（诊断）

```bash
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

**如何一次对比多个策略？**
- **方式**：`--strategy-config` 支持逗号分隔多个策略目录（同一套 symbol/timeframe/split/seed 下跑多次训练+回测，然后汇总成一个对比报告）。

```bash
# 对比同一策略的不同配置变体（例如：baseline vs mainline vs 带波动率模型）
# 注意：可以省略 config/strategies/ 前缀，工具会自动补全
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_rr_reg_long,sr_reversal_rr_reg_long_mainline,sr_reversal_rr_reg_long_vol \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --test-size 0.3 \
  --seed 42 \
  --deterministic \
  --no-docker
```

**通用参数（最常用）**：
- **`--strategy-config`**：一个或多个策略目录（逗号分隔）
- **`--symbol`**：单标的或多标的（逗号分隔）
- **`--timeframe`**：K 线周期（如 `240T`）
- **`--start-date/--end-date`**：可选的研究窗口裁剪（内部用环境变量裁剪数据，保证可复现）
- **`--test-size`**：外层 train/test 时间切分比例
- **`--seed` / `--deterministic`**：可复现控制
- **输出**：默认生成 `model_comparison_report.html` + `model_comparison_results.json`

> SR Reversal 专用的“规则 vs ML vs 波动率”对比仍保留为：`mlbot diagnose sr-reversal-model-comparison`

**它能否完全替代 步骤 3/4？**
- **不能完全替代**。`feature-group-search` 解决的是“在固定 Task/回测口径下，哪组特征组合更好”的问题；但：
  - **步骤 3（消融）** 仍推荐用于**确认**：把最终 `features_suggested.yaml` 与关键对照（baseline / 某个语义块 / 某个 Pool B 块）做 A/B，验证结论可复现、差异来自特征而不是偶然波动。
  - **步骤 4（模型对比/诊断）** 仍必需用于**排错与归因**：比如 0 trades、标签稀疏、预测塌缩、回测参数不一致等；这些不是 feature-group-search 自己能“搜索出来”的。

**三者区别（快速对照）**
- **`feature-group-search`（Wrapper / 选组合）**：在一个策略目录内，用 groups + Pool B 候选做 greedy 组合搜索，目标是“找到更好的特征组合”，产物是 `features_suggested.yaml` + 报告。
- **`strategy-feature-compare`（Wrapper / 做 A/B）**：在同一策略目录下，对比多个 `features*.yaml`（例如 baseline vs suggested），回答“结论是否稳、增益来自哪里”，更偏 confirm。
- **`model-comparison`（诊断 / 找问题 & 横向比较）**：对比多个策略目录在同一口径下的训练+回测结果，并汇总诊断块（labels/preds/trades），用于排错与归因，不负责自动选特征组合。

### 步骤 5：滚动训练（生产评估 / Walk-Forward）

**用途**：模拟真实生产环境的时间序列训练，使用 **扩展窗口（expanding window）** 方式，每个月用之前所有历史数据训练，然后在该月测试。这是**生产部署前的最终验证阶段**。

**核心机制**：
- **扩展窗口**：第 N 个月的测试集，使用第 1 到 N-1 个月的所有数据作为训练集（而不是固定窗口）
- **月度切分**：按月份自动切分数据，每个测试月独立训练+回测
- **输出**：每个月的模型 + 月度结果汇总（`monthly_results.json` + HTML 报告）

**使用场景**：
1. **生产前最终验证**：在完成特征选择（Step 2）和模型对比（Step 4）后，用 rolling 训练验证策略在**跨时间窗口的稳定性**（避免过拟合到某个特定时期）
2. **概念漂移检测**：观察不同月份的 Sharpe/DD/trades 变化，识别是否存在概念漂移（例如市场 regime 切换导致模型失效）
3. **月度模型更新**：生产环境可以每月运行 `--update-only`，只训练新月份，复用已有模型

**参数说明**：
- **`--initial-train-months`**：第一个测试月需要至少 N 个月的历史数据作为训练集（例如 6 个月）
- **`--min-train-months`**：后续月份如果历史数据不足 N 个月，则跳过该月（例如 3 个月）
- **`--start/--end`**：可选，限制 rolling 的时间范围（不指定则使用全部可用数据）
- **`--update-only`**：增量更新模式，只训练新月份，跳过已训练的月份

```bash
# 完整 rolling 训练（首次运行）
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3

# 增量更新（只训练新月份，用于生产月度更新）
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

**输出位置**：
- 模型：`results/rolling/<strategy_name>/<YYYY-MM>/model.pkl`
- 汇总结果：`results/rolling/<strategy_name>/monthly_results.json`
- HTML 报告：`results/rolling/<strategy_name>/monthly_rolling_report.html`

---

### 步骤 6：上线前回测（Nautilus Trader 事件驱动）

**用途**：使用 **Nautilus Trader** 进行事件驱动回测，确保回测逻辑与实盘完全一致（避免向量化回测的 lookahead bias 和滑点假设偏差）。

**完整工作流程**：
1. **步骤 5（滚动训练 + vectorbt）**：用很长时间（例如 2024-01 ~ 2025-10）做滚动训练，每个月用 vectorbt 回测，得到每个月的模型和回测结果。**这是验证 OOS 性能和模型稳定性的主要阶段**。
   - 例如：2025-10 月测试 = 用 2024-01 ~ 2025-09 训练，在 2025-10 测试（vectorbt 回测）
2. **步骤 6（事件驱动回测）**：用步骤 5 得到的**最新模型**（例如 2025-10 月的模型）做 Nautilus 事件驱动回测。**关键问题**：如何选择回测时间窗口？
   - **问题**：如果用 2025-05 ~ 2025-10 做事件驱动回测，那么 2025-05 ~ 2025-09 在滚动训练中已经作为训练数据使用过，无法和 vectorbt 的测试结果对比（因为 vectorbt 中这些月份是训练数据，不是测试数据）
   - **解决方案**：见下方"事件驱动回测的时间窗口选择"
3. **步骤 7（前瞻性测试，可选）**：如果希望更保险，可以等待未来数据（例如 2025-11 ~ 2025-12），用最新模型做前瞻性测试，然后再上线。

**与滚动训练的关系**：
- **滚动训练使用 vectorbt 回测**：`rolling` 训练已经用 vectorbt 做了回测，这是**快速、批量评估**的理想选择，已经验证了跨时间窗口的 OOS 性能稳定性。
- **事件驱动回测的作用**：不是替代 vectorbt，而是**验证实盘逻辑一致性**。主要目的是：
  1. **验证执行逻辑**：确保实盘的订单执行、滑点、延迟等与回测假设一致
  2. **验证特征计算**：确保实盘的特征计算逻辑与训练时完全一致（特别是 tick 级特征、状态连续性）
  3. **验证模型加载**：确保实盘能正确加载模型和 preprocessor
  4. **验证边界情况**：在更长的时间窗口中发现潜在的边界问题（例如特征计算在特殊市场条件下的异常）
- **事件驱动回测的时间窗口选择（关键问题）**：
  
  **问题**：如果用 2025-05 ~ 2025-10 做事件驱动回测，那么 2025-05 ~ 2025-09 在滚动训练中已经作为训练数据使用过，无法和 vectorbt 的测试结果对比（因为 vectorbt 中这些月份是训练数据，不是测试数据）。
  
  **解决方案**：
  - **方案 B1（推荐，严格对比）**：只用**最后一个测试月**（例如 2025-10）做事件驱动回测，直接对比 vectorbt 和 Nautilus 的回测结果。这是严格 OOS，可以验证两种回测方法的一致性。
  - **方案 B2（充分验证逻辑一致性）**：用**最后 3-6 个月**（例如 2025-05 ~ 2025-10）做事件驱动回测，但明确区分：
    - **2025-10（最后一个测试月）**：严格 OOS，对比 vectorbt 和 Nautilus 的结果，验证一致性
    - **2025-05 ~ 2025-09（训练数据）**：虽然这些数据在训练时已经用过了，但事件驱动回测的目的不是验证 OOS 性能，而是验证**逻辑一致性**（特征计算、状态连续性、边界情况等）。只要结果不太离谱（例如没有异常错误、特征计算正常、状态传递正确），就可以认为逻辑一致性验证通过。
  - **方案 B3（折中）**：用**最后 2-3 个月**（例如 2025-08 ~ 2025-10）做事件驱动回测，既保证有足够时间验证逻辑一致性，又尽量减少使用训练数据的比例。
  
  **推荐**：**方案 B1**（只用最后一个测试月）是最严格的，可以确保对比的公平性。如果担心一个月太短，可以用**方案 B2**，但需要明确：2025-05 ~ 2025-09 的回测结果只用于验证逻辑一致性，不用于性能评估。

**核心机制**：
- **事件驱动架构**：按时间顺序处理每个 bar/tick，模拟真实交易环境
- **特征一致性**：使用与训练时相同的特征计算逻辑（`StrategyFeatureLoader`）
- **模型 + Preprocessor 一致性**：加载训练时保存的模型和 preprocessor，确保 transform 逻辑一致

**关键问题解答**：

1. **Transform 和归一化如何与训练保持一致？**
   - **Preprocessor 保存**：当前 `train_strategy_pipeline.py` 和 `rolling_train.py` 只保存了模型（`model.pkl`），**未保存 preprocessor**。需要补充保存逻辑：
     ```python
     # 在训练脚本中补充（TODO）
     preprocessor_path = output_dir / "preprocessor.pkl"
     joblib.dump(preprocessor, preprocessor_path)
     ```
   - **特征计算一致性**：使用相同的 `StrategyFeatureLoader` 和 `feature_dependencies.yaml`，确保特征计算逻辑一致
   - **归一化参数**：如果使用了 `UnifiedNormalizer` 或 `TalibFeatureEngineer` 的 scaler，需要单独保存（`scaler.pkl`）
   - **建议**：封装一个 `ModelArtifact` 类，统一保存 `model + preprocessor + used_features + feature_config`，见 `docs/时序模型/工作流："预处理 + 模型 + 后处理"一体化保存与部署.md`

2. **模型训练时间 vs 回测时间窗口（OOS 问题）**
   - **严格 OOS 原则**：如果模型在 2025 年 X 月训练，**不能**用 2025 年 X 月及之前的数据做上线前回测（这是 in-sample，会高估性能）
   - **滚动训练的特殊性**：滚动训练使用**扩展窗口（expanding window）**，每个测试月都是严格 OOS（因为训练时看不到测试月的数据）。例如：
     - 2025-06 月测试：用 2024-01 ~ 2025-05 训练（严格 OOS）
     - 2025-07 月测试：用 2024-01 ~ 2025-06 训练（严格 OOS）
     - 2025-10 月测试：用 2024-01 ~ 2025-09 训练（严格 OOS）
   - **推荐方案**：
     - **方案 A（滚动训练最后一个月，推荐）**：直接用 `rolling` 训练的最后一个月（例如 2025-10）作为“上线前回测”。**这是严格 OOS**，因为训练时看不到这个月的数据。滚动训练已经用 vectorbt 做了回测，可以直接使用结果。
     - **方案 B（事件驱动验证）**：在方案 A 的基础上，用 Nautilus Trader 做**事件驱动回测**，验证与 vectorbt 回测的一致性。**关键问题**：如何选择回测时间窗口？
       
       **方案 B1（推荐，严格对比）**：只用**最后一个测试月**（例如 2025-10）做事件驱动回测，直接对比 vectorbt 和 Nautilus 的回测结果。这是严格 OOS，可以验证两种回测方法的一致性。
       
       **方案 B2（充分验证逻辑一致性）**：用**最后 3-6 个月**（例如 2025-05 ~ 2025-10）做事件驱动回测，但明确区分：
       - **2025-10（最后一个测试月）**：严格 OOS，对比 vectorbt 和 Nautilus 的结果，验证一致性
       - **2025-05 ~ 2025-09（训练数据）**：虽然这些数据在训练时已经用过了，但事件驱动回测的目的不是验证 OOS 性能，而是验证**逻辑一致性**（特征计算、状态连续性、边界情况等）。只要结果不太离谱（例如没有异常错误、特征计算正常、状态传递正确），就可以认为逻辑一致性验证通过。
       
       **为什么方案 B2 需要 3-6 个月？**
       - **覆盖更多市场场景**：不同市场 regime（趋势/震荡/极端波动），一个月可能只覆盖一种场景
       - **验证状态连续性**：某些特征（如 TradeCluster、Footprint、VPIN）需要跨 bar 的状态维护，需要更长时间验证状态是否正确传递
       - **发现边界问题**：在更长的时间窗口中发现潜在的边界情况（例如数据缺失、异常值处理、特征计算异常）
       
       **推荐**：**方案 B1**（只用最后一个测试月）是最严格的，可以确保对比的公平性。如果担心一个月太短，可以用**方案 B2**，但需要明确：2025-05 ~ 2025-09 的回测结果只用于验证逻辑一致性，不用于性能评估。
     - **方案 C（前瞻性测试）**：滚动训练到 2025-10，然后用**未来数据**（2025-11 ~ 2025-12）做上线前回测。这需要等待未来数据，适合长期验证。
   - **最佳实践**：
     - **如果模型是滚动训练出来的**：推荐**方案 A**，因为滚动训练已经做了 walk-forward，最后一个测试月已经是严格 OOS，且已经用 vectorbt 做了回测。
     - **如果需要验证实盘逻辑一致性**：在方案 A 基础上加**方案 B**（Nautilus 事件驱动回测），但这不是为了验证 OOS 性能，而是为了确保实盘与回测逻辑一致。
     - **如果需要更长期验证**：使用**方案 C**（前瞻性测试），但需要等待未来数据。

3. **总体架构图**
   - 见 `docs/ARCHITECTURE.md` 中的 ASCII 架构图（策略层 → 模型层 → 特征层 → 数据层）
   - 详细架构说明：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`

**Nautilus Trader 回测命令**（待实现）：

```bash
# 上线前回测（事件驱动，验证实盘逻辑一致性，建议用 3-6 个月）
mlbot backtest nautilus \
  --config config/strategies/sr_reversal_long \
  --model-path results/rolling/sr_reversal_long/2025-10/model.pkl \
  --preprocessor-path results/rolling/sr_reversal_long/2025-10/preprocessor.pkl \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2025-05-01 \
  --end 2025-10-31 \
  --output-dir results/nautilus_backtest/sr_reversal_long_2025-05_to_2025-10
```

**注意**：这里用 2025-05 ~ 2025-10（6 个月）是为了充分验证逻辑一致性。虽然 2025-05 ~ 2025-09 在滚动训练中已经作为训练数据使用过，但事件驱动回测的目的不是验证 OOS 性能（这个已由滚动训练验证），而是验证实盘逻辑与回测逻辑的一致性。

**集成指南**：
- Nautilus Trader 集成：`docs/Nautilus_Trader_集成指南.md`
- 实盘特征加载：`src/time_series_model/live/nautilus_strategy_with_features.py`

**TODO（待实现）**：
1. ✅ 补充 `preprocessor` 保存逻辑到训练脚本
2. ✅ 实现 `mlbot backtest nautilus` 命令
3. ✅ 封装 `ModelArtifact` 类（model + preprocessor + metadata）
4. ✅ 在 Nautilus 回测中加载并验证 preprocessor 一致性

---

## 文档入口（建议先读）

### 核心工作流文档

- **推荐特征工作流**：`docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md`
  - Pool B（数学/数值特征的"海选池"）vs Semantic 特征（人类可理解的语义因子）
  - 完整的三阶段工作流：生成 Pool B → 准备语义 groups → feature-group-search
  - 强调：Semantic 特征需要人类维护，从 Pool B 深度加工而来

- **特征工作流修复总结**：`docs/strategies/FEATURES_ALL_SELF_CONTAINED_FIX.md`
  - `features_all.yaml` 自包含修复
  - `factor-eval` 和 `feature-group-search` 的正确使用方式

- **最佳特征配置汇总**：`docs/strategies/BEST_FEATURE_CONFIGURATIONS.md`
  - 各策略的最佳特征配置
  - Pool B 与语义特征的关系说明

- **Semantic groups 单例展开**：`docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`
  - 为什么需要展开 semantic groups 为单例
  - 性能分析和优化建议
  - 实现方案和使用方式

- **Factor-Eval 输出分析**：`docs/strategies/FACTOR_EVAL_OUTPUT_ANALYSIS.md`
  - factor-eval 的筛选标准和阈值
  - 经验输出特征数量（50-150 个）
  - 优化建议

- **树模型对相反特征的处理**：`docs/strategies/TREE_MODEL_OPPOSITE_FEATURES.md`
  - 树模型能否自动处理相反特征
  - 不展开 semantic groups 的效果分析
  - 推荐策略

- **项目 TODO List**：`docs/TODO_LIST.md`
  - 所有待完成任务的详细说明
  - 按优先级和类别组织
  - 包含任务作用、命令示例、预期结果等

### 架构文档

- **系统架构图**：`docs/ARCHITECTURE.md`（包含完整的架构层次图）

- **特征列表与归一化状态**：`docs/architecture/FEATURE_CATALOG.md`
  - 全部 208 个特征节点的归一化状态
  - 归一化方法说明（ATR 归一化、百分比归一化、相似度转换等）
  - 按类别分组的特征列表
  - 树模型 vs NN 模型的使用建议

- **特征归一化策略**：`docs/architecture/FEATURE_NORMALIZATION_POLICY.md`
  - Phase 1/2/3 归一化实现进度
  - 跨资产可比性验证
  - 因果性归一化方法（避免未来泄露）

- **工业化 Experiment Loop（Layer A/B/C、TaskSpec、Filter→Wrapper、稳定性口径）**：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- **NN 多头 Path Primitives + Router→Execution（NO/MEAN/TREND）**：`docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`
- **研究 Playbook（标签/执行一致性、timeframe、仓位管理）**：`docs/guides/RESEARCH_PLAYBOOK_CN.md`
- **Policy 执行假设（intrabar vs close）**：`docs/strategies/POLICY_EXECUTION_ASSUMPTIONS_CN.md`
- **NN 多头 → 3-action → RL/BC e2e（长文档）**：`docs/guides/NNMULTIHEAD_3ACTION_E2E_CN.md`

---

## 获取帮助

```bash
mlbot --help
```


