# 统一研发与实盘模拟管线设计

## 1. 背景与目标

当前 `pipeline run` 更偏“单次窗口最优”，对长期实盘行为模拟不足。  
本方案目标是建立一个统一框架：

- 慢变量（结构）低频更新：特征集合、prefilter/gate/entry_filter 结构、策略方向启停基线
- 快变量（参数）高频更新：月度阈值、execution、slot 分配
- 输出可拼接的长期月度 OOS 结果，用于上线前评估与回归

## 2. 术语与边界

- **慢变量**：变化慢、影响结构稳定性，默认每 3 个月或触发式更新
- **快变量**：变化快、影响当月执行表现，默认每月更新
- **carry_forward**：当月未满足 adopt 但可继承上期参数继续运行
- **quality ranking**：在同侧机会拥挤时，对 `(symbol, side)` 进行优先级排序

### 2.1 三种「用法」对照：经典 stage、滚动 stage、配置模式 A/B

容易混淆的有三层东西：**同一套 `mlbot pipeline run`，靠 `--stage` 与 YAML 里的 `rolling.mode` 区分**。下面分开说「相同点」「不同点」「各自干什么」。

#### 相同点

- **入口一致**：`mlbot pipeline run`（或 `python scripts/auto_research_pipeline.py`），共用 `config/prod_train_pipeline_*.yaml` 等配置；`--strategy` / `--all`、`--config`、`--end-date`、`--dry-run` 等通用。
- **步骤内核一致**：无论哪种用法，真正干活时仍会调用同一批子流程（数据、Feature Store、prefilter、gate、entry_filter、execution 优化、事件回测、PCM 等），差异在于**编排方式**（跑几次、时间窗怎么滑、是否做 SHAP/结构快照）。
- **产物根目录一致**：由 `output.history_dir` 决定；滚动类结果落在 `_rolling_sim/<run_id>/` 等子路径（见 §5）。

#### 不同点（三张「菜单」）

| 维度 | ① 经典分段（老管线） | ② 慢变量 / 快变量（滚动阶段） | ③ 模式 A / 模式 B（实施文档 01） |
|------|----------------------|-------------------------------|-----------------------------------|
| **是什么** | 一组 **`--stage`**：`full`、`prefilter`、`gate`、`entry_filter`、`execution_opt`、`event_backtest`、`pcm_joint`、`pcm_slot_grid` | 另一组 **`--stage`**：`slow_snapshot`、`fast_month`、`rolling_sim` | **不是**第三套命令；是 YAML 里 **`rolling.mode`**：`slow_realistic` 与 `turbo_fixed_features` |
| **时间怎么切** | 默认**一个**全局窗口（由 `dates` + `end-date`/自动检测决定）；`full` 在该窗口内跑通整条链 | **按日历推进**：季末/月末切片；`rolling_sim` 从 holdout 起点月到截止月**逐月**重复「校准窗 + 当月 OOS」 | 在 ② 的每一次循环里，决定**要不要**做季频结构更新、**要不要**做特征搜索 |
| **典型用途** | 单次训练与上线前实验、分层调试某一步、跑完一次 PCM slot 网格 | 模拟「慢结构 + 快阈值」的长期行为；月度 ledger、拼接 summary、交易地图 | 在同一滚动框架下对比：**偏真**（季更结构 + SHAP）vs **偏快**（固定特征、只调阈值） |

#### 各自作用（一句话）

1. **经典分段**：适合「这一截数据上把 pipeline 跑通/调参/对比」，不强制按月滚动。
2. **滚动阶段**：适合「像实盘一样按月推进」，看阈值漂移、side state、多月拼接表现。
3. **模式 A / B**：在 **②** 已经选好的前提下，再选** realism vs 速度**；命令仍是 `rolling_sim` / `fast_month`，换的是配置文件里 `rolling.mode`（如 `prod_train_pipeline_2h_strict_2024bull.yaml` vs `*_turbo_2024bull.yaml`）。

#### 与《A快速启动命令》的对应关系

- **①** 对应其中 **full / 分层 stage / execution_opt / event_backtest / pcm_*** 等块。
- **②** 对应其中 **slow_snapshot、fast_month、rolling_sim** 块。
- **③** 对应 **实施文档 01** 里两种 YAML（strict ≈ 模式 A，turbo ≈ 模式 B），仍使用 **②** 里的命令。

## 3. 配置契约

统一在 `config/prod_train_pipeline_2h.yaml`、`config/prod_train_pipeline_4h.yaml` 及派生文件（如 `*_strict_2024bull.yaml`、`*_turbo_2024bull.yaml`）中维护。

### 3.1 CLI、截止日与因果性

- 滚动 / 经典单次流程由 **`--stage`**（如 `rolling_sim`、`fast_month`、`full`、`slow_snapshot`）选择，**无**单独的 `pipeline_mode` 配置项。
- **数据截止日**：优先 **`--end-date`**；若未传且配置中存在 **`dates.end_date`**，则使用该值；否则 **`auto_research_pipeline.py` 自动检测** parquet 最新日。
- **慢变量因果性（运营向）**：按季跑 `slow_snapshot` 时，应对**每一季**显式传入 **`--end-date` 为该季末日**（as-of），避免把**未来 K 线**纳入结构训练。`dates.start_date` 仅决定全局训练样本起点与合法性（例如配合 `calibration_months` 避免空 train），**不**等价于「看未来」；未来是否进入模型由 **`end-date` / `dates.end_date`** 决定。

### 3.2 顶层 YAML 参数说明

以下按配置文件**出现顺序**说明语义；具体数值以仓库内 YAML 为准。

#### `dates`

| 字段 | 说明 |
|------|------|
| `start_date` | 全局训练数据起点；滚动月内校准窗需要此前有足够历史。 |
| `end_date` | 可选。锁定滚动/展示用数据终点；不设则依赖 CLI 或自动检测。 |
| `holdout_months` | 自 `end-date` 起向前算的 holdout 月数；影响单次 `full` 与 **`rolling_sim` 起始月份列表**（从 holdout 起点月到截止月逐月推进）。 |
| `validation_months` | holdout 内用于 Gate 等调参的子窗长度；与 `holdout_months` 共同决定 train/val/test 切分。 |

#### `training`

| 字段 | 说明 |
|------|------|
| `seeds` | 多 seed 训练列表。 |
| `seed_selection` | 多 seed 时选优准则（如 `best_sharpe`）。 |

#### `global_toggles`

| 字段 | 说明 |
|------|------|
| `locked_threshold_tuning_enabled` | 是否启用与 locked 规则相关的阈值调优路径（与策略内 `kpi_gates` 等配合）。 |

#### `strategy_scope`

| 字段 | 说明 |
|------|------|
| `direction` | `all` \| `long` \| `short`。`**--all` 时**只跑名称中含 `-long-` 或 `-short-` 的策略子集。 |

#### `rolling`（`rolling_sim` / `fast_month` 核心）

由 `load_pipeline_config()` 校验并写回规范化结构。

| 字段 | 说明 |
|------|------|
| `mode` | `slow_realistic`：季频可触发慢结构快照 + 月频阈值校准；`turbo_fixed_features`：固定特征根目录、默认不做特征搜索，月频阈值 + OOS；`legacy`：兼容旧行为。 |
| `windows.calibration_months` | 每月测试月之前，用最近 N 个月做阈值/全层校准窗长度。 |
| `windows.structure_lookback_months` | 慢结构快照回看月数（如 SHAP/结构训练窗）。 |
| `slow_realistic.cadence_months` | 从滚动**第 0 月起每隔 N 个月**触发一次 `_run_slow_structure_snapshot_for_month`（与 `month_idx % cadence == 0` 对齐）。 |
| `slow_realistic.triggered_retrain_enabled` | 为 `false` 时不再按上述节拍跑慢快照，始终用仓库 `config/strategies`（或上游指定根）。 |
| `turbo_fixed_features.fixed_strategies_root` | turbo 模式下使用的策略配置根路径（通常 `config/strategies`）。 |
| `turbo_fixed_features.disable_feature_search` | `true` 时关闭 SHAP/特征搜索，仅阈值与下游层。 |

#### `slow_loop`（运营/设计契约）

与 `rolling_long_horizon` 文档中的「慢变量」叙述对齐，**与同仓库其它 prod 配置保持同形**。  
**实现注记**：`rolling_sim` 的季频结构更新**直接读 `rolling.slow_realistic.*`**；`turbo` 模式不按季重跑结构。`slow_loop` 当前**不**被 `auto_research_pipeline.py` 解析，主要用于 strict/turbo 配置对照与后续工具扩展。

| 字段 | 说明 |
|------|------|
| `cadence_months` | 慢更新建议周期（月），宜与 `rolling.slow_realistic.cadence_months` 同值对照。 |
| `triggered_retrain.enabled` | 运营语义：是否允许调度/漂移触发慢变量重训。 |
| `freeze_outputs.enabled` | 是否对慢变量产出做冻结/归档。 |
| `freeze_outputs.keep_latest` | 冻结物最多保留份数。 |

#### `fast_loop`

`fast_loop` 由 `load_pipeline_config()` 规范化后供 `fast_month` / `rolling_sim` 使用。  
兼容优先级：`rolling.mode=legacy` 时保持旧行为（忽略 `fast_loop` 分支开关）。

| 字段 | 说明 |
|------|------|
| `step_months` | 快变量更新步长（月），与月度滚动一致时为 `1`。 |
| `threshold_calibration.enabled` | 是否跑阈值校准链路。 |
| `prefilter.optimize` | 是否在校准窗内重新搜索 prefilter 规则（`false` 时复用现有 `archetypes/prefilter.yaml`）。 |
| `symbol_threshold_calibration.enabled` | 预留字段：当前代码未消费，不影响运行结果。 |
| `execution_opt.enabled` | 是否跑 execution 优化（sym-r 网格等）。 |
| `pcm_eval.enabled` | 是否在流程中纳入 PCM 相关评估（与联合回测步骤配合）。 |

#### `symbol_policy`

| 字段 | 说明 |
|------|------|
| `mode` | `global_only` \| `symbol_only` \| `hybrid_carry_forward`：symbol 侧状态与 carry-forward 策略。 |
| `carry_forward_ttl_months` | carry-forward 最长延续月数。 |
| `enable_threshold` | 当月指标高于此阈值等条件时倾向 `active`（与实现内 quality/Sharpe 逻辑配合）。 |
| `min_symbol_trades_soft` | symbol 侧软最小交易数门槛。 |
| `carry_forward_hard_fail_rules` | 如 `min_sharpe_r`：跌破则 hard-fail，不再 carry。 |

#### `direction_stack`

| 字段 | 说明 |
|------|------|
| `mode` | `ema200` \| `vwap_long_anchor` \| `ensemble`：方向过滤/叠加逻辑（V1 常用 `ema200`）。 |
| `ema_debounce_bars` | EMA 方向翻转去抖 K 线数。 |
| `vwap_window_days` | VWAP 锚定窗口（天），非 `vwap_*` 模式时可忽略。 |

#### `slot_allocation`

| 字段 | 说明 |
|------|------|
| `mode` | 如 `quality_ranked`：按质量分分配 slot。 |
| `max_symbols_per_side` | 每侧最多同时参与排序/占位的 symbol 数。 |
| `reserve_slots_for_secondary_symbols` | 为次优 symbol 预留 slot 策略相关参数。 |
| `slot_opportunity_penalty` | 机会拥挤惩罚系数。 |
| `quality_score_weights` | `history_edge` / `now_strength` 权重（与文档 §6 一致）。 |
| `quality_lookback_months` | 质量分历史项回看月数。 |

#### `stitching`

| 字段 | 说明 |
|------|------|
| `enabled` | 是否在 `rolling_sim` 末尾拼接多月指标。 |
| `metrics` | 写入 `stitched_summary` 的指标名列表。 |
| `export_trade_map_html` | 是否导出拼接交易地图索引 HTML。 |

#### `event_backtest`

| 字段 | 说明 |
|------|------|
| `enabled` / `promote` | 是否跑事件回测及是否晋升产物到策略目录。 |
| `sym_r_default` | 默认 `initial:step:max` 止损网格字符串。 |
| `sym_r_by_family` | 按家族（`bpc`/`fer`/`me`）覆盖 sym-r。 |
| `exec_objective_default` / `exec_objective_params` | 默认优化目标（如 `sharpe`）及风险惩罚参数。 |
| `exec_objective_by_family` | 按家族覆盖目标（如 ME 用 `risk_aware`）。 |

#### `pcm_slot_grid`

在 PCM 联合回测之后可选的多 slot 方案网格；`plateau_delta`、`min_trades_soft`、`penalties`、`cases[]`（每 case 的 `slots` 与 `per_strategy_limits`）定义评分与推荐逻辑。详见 YAML 内注释。

#### `universe_group` / `data_path` / `download`

| 区块 | 说明 |
|------|------|
| `universe_group` | 从 `file` 中解析 `universe_set` + `group` 得到合约列表。 |
| `data_path` | Parquet 数据根目录。 |
| `download` | `enabled`、`data_dir`、`parquet_dir`：是否跑下载与落盘路径。 |

#### `strategies.<name>`（单策略）

每个策略下常见键：`config`、`timeframe`、`dates`（可覆盖 holdout/validation）、`features_gate`、`labels_gate`、`has_prefilter`、`has_direction`、`kpi_gates`（prefilter/gate/entry_filter/backtest/deploy）、`shap_override` 等。细节以各策略目录与 YAML 为准。

#### `shap_feature_selection` / `comparison` / `retrain_triggers` / `deploy_gate` / `data_flow`

主要用于**经典 full 流水线**与实验门控：SHAP 折数与 top_k、对比采纳阈值、再训触发、上线人工确认、各步骤输入文件名约定等；与 `rolling_sim` 并存于同一文件时，按 `stage` 决定实际执行子集。

#### `output`

| 字段 | 说明 |
|------|------|
| `history_dir` | 实验与 `_rolling_sim`、`_pcm_joint` 等产物的根目录（相对仓库根）。 |

**PCM / 宪法**：联合事件回测允许的策略集以 **`config/constitution/constitution.yaml`** 中 `resource_allocation.enabled_archetypes` 为准；与管线里 `strategy_scope` 过滤后的策略名**取交集**，二者需同时覆盖才能在 Step 9.5 进入 PCM。

### 3.3 各层 KPI 参数来源与数据窗（rolling / fast_month）

#### KPI 参数来源（按层）

| 层 | 参数来源 | 主要脚本 |
|---|---|---|
| Prefilter | `strategies.<name>.kpi_gates.prefilter.*` | `scripts/analyze_archetype_feature_stratification.py` |
| Gate | `strategies.<name>.kpi_gates.gate.*` | `scripts/train_strategy_pipeline.py` + `scripts/optimize_gate_unified.py` |
| Entry Filter | `strategies.<name>.kpi_gates.entry_filter.*` | `scripts/optimize_entry_filter_plateau.py` |
| Backtest/Deploy 门槛 | `strategies.<name>.kpi_gates.backtest/deploy.*` | `scripts/backtest_execution_layer.py`（及结果汇总） |

#### 数据窗对照（以 `fast_month` 为例）

| 环节 | 窗口 |
|---|---|
| 全量特征构建 | `start_date ~ calib_end` |
| 模型训练集（Train） | `start_date ~ calib_start`（不含 holdout） |
| 阈值校准（Prefilter/Gate/Entry） | `calib_start ~ calib_end` |
| Execution 网格优化 | `calib_start ~ calib_end`（可由 `fast_loop.execution_opt.enabled` 关闭） |
| 当月事件回测（OOS） | `test_start ~ test_end`（可由 `event_backtest.enabled` 关闭） |

其中 `calib_start/calib_end/test_start/test_end` 由 `rolling.windows.calibration_months` 与目标月份共同决定；`test` 为目标自然月，`calib` 为其之前 N 个月。

## 4. 命令与阶段

### 4.1 现有阶段（兼容保留）

- `full`
- `prefilter`
- `gate`
- `entry_filter`
- `execution_opt`
- `event_backtest`
- `pcm_joint`
- `pcm_slot_grid`

### 4.2 新增阶段（本方案）

- `slow_snapshot`：仅慢变量快照（到 entry_filter）
- `fast_month --month YYYY-MM`：单月快变量复盘
- `rolling_sim`：按 holdout 月份逐月执行 fast loop 并拼接

### 4.3 新增辅助命令

- `mlbot pipeline report-side-state --run-id <run_id>`
- `mlbot pipeline debug-quality --run-id <run_id> --month YYYY-MM`

## 5. 产物契约

滚动模拟根目录：`results/.../_rolling_sim/<run_id>/`

- `monthly_ledger.jsonl`：月度摘要流水
- `stitched_summary.json`：拼接汇总指标
- `trading_map_stitched.html`：月度交易地图索引
- `fast_month_<YYYY-MM>/`
  - `fast_month_summary.json`
  - `quality_ranking_<YYYY-MM>.json`
  - `symbol_side_state.json`

## 6. 质量分（V1）

### 6.1 评分

V1 用轻量可解释分数：

- `Qv1 = 0.55 * history_edge + 0.45 * now_strength`
- 当前实现默认使用事件回测指标近似 `history_edge + 风险惩罚`
- 预留 `cvd_accel_aligned / price_efficiency_aligned` 作为 `now_strength` 的增强输入

### 6.2 排序与并列规则（Tie-break）

主排序按 `Qv1` 降序；并列时按：

1. `near_stop_rate` 低优先
2. `max_drawdown_r` 低优先
3. `n_trades` 高优先
4. `strategy` 字典序（保证复现）

## 7. Symbol Side 状态机（V1）

每个策略/方向状态：

- `active`
- `carry_forward`
- `disabled`

更新规则（简化）：

- 若当月 `sharpe_r > enable_threshold` 且 `n_trades >= min_symbol_trades_soft` -> `active`
- 否则若为 long 且上月为 `active/carry_forward` 且未触发 hard-fail -> `carry_forward`
- 否则 `disabled`

## 8. 与实盘主循环对齐

`run_live.py` 对齐建议：

- 热更新：阈值、execution、side state
- 需重启：特征集合与 constitution 硬约束
- retrain 检查仍由现有周期任务触发，rolling 结果作为决策依据

## 9. 验证计划

1. **兼容性**：`classic` 流程输出不变
2. **功能性**：新阶段产物齐全、CLI 可调用
3. **一致性**：`monthly_ledger` 与 `stitched_summary` 聚合一致
4. **可复现**：固定 seed/date 结果稳定
5. **策略对比**：`global_only` vs `hybrid_carry_forward`，`quality_ranked` vs baseline

## 10. 风险与后续

- V1 风险：小样本导致质量分波动
  - 缓解：最小交易数门槛、hard-fail、可复现排序
- V1.1 方向：
  - 引入 `vwap_long_anchor` 方向模式实现
  - 引入更细粒度 symbol 级 CVD/效率特征与权重学习

## 11. 代码布局（模块化迁移）

为降低 `auto_research_pipeline.py` 单文件复杂度，当前已引入 `scripts/pipeline/` 子包作为迁移承载层：

- `scripts/pipeline/context.py`: `PROJECT_ROOT`、`DEFAULT_CONFIG`、`PipelineContext`。
- `scripts/pipeline/config.py`: 配置契约与日期/月份工具（`load_pipeline_config`、`resolve_strategy_dates` 等）。
- `scripts/pipeline/steps.py`: 通用步骤执行与输出解析（`run_step`、`find_output_dir`、`parse_backtest_stdout`）。
- `scripts/pipeline/strategy_pipeline.py`: 单策略主链路桥接（`run_strategy_pipeline`）。
- `scripts/pipeline/rolling.py`: `fast_month` / `slow_snapshot` 桥接入口。
- `scripts/pipeline/events.py`: execution/event/PCM 相关桥接入口。
- `scripts/pipeline/cli.py`: CLI 主入口桥接。
- `scripts/pipeline/graph.py`: 阶段注册表脚手架（后续可演进为统一调度表）。

兼容性原则：

- `mlbot pipeline run` 仍通过 `scripts/auto_research_pipeline.py` 路径调用，不改现有命令。
- 迁移采取“桥接优先、行为等价”策略：先抽离公共能力，再逐步把实现体迁出。
